"""
SQLite cache for session scores.

Keyed by (file_path, tier) — invalidated automatically when the file's
mtime changes. Cache failures are always silent so they never break the
main analysis pipeline.

Cache location: ~/.mindcheck/cache.db
"""

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from mindcheck.parser import Session, Message
from mindcheck.signals.structural import StructuralSignals
from mindcheck.signals.semantic import SemanticSignals
from mindcheck.signals.llm import LLMSignals

# Bump this when the stored schema changes — forces a full re-analysis.
CACHE_VERSION = 7  # bumped: session type detection + per-type scoring weights (v7)


# ── Public API ────────────────────────────────────────────────────────────────

def get_cached(file_path: Path, mtime: float, tier: int):
    """Return a cached SessionScore if the file is unchanged, else None."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT mtime, version, data FROM scores WHERE file_path=? AND tier=?",
            (str(file_path), tier),
        ).fetchone()
        conn.close()

        if row is None:
            return None

        cached_mtime, version, data_json = row

        if abs(cached_mtime - mtime) > 0.001:   # file has been modified
            return None
        if version != CACHE_VERSION:             # schema changed
            return None

        return _deserialize(data_json)
    except Exception:
        return None


def save_cached(file_path: Path, mtime: float, tier: int, score) -> None:
    """Persist a SessionScore to the cache. Silent on failure."""
    try:
        conn = _connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO scores
                (file_path, tier, mtime, version, data, cached_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(file_path), tier, mtime, CACHE_VERSION,
             _serialize(score), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def cache_stats() -> dict:
    """Return basic cache statistics for the `mindcheck cache` command."""
    try:
        conn = _connect()
        total = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        by_tier = conn.execute(
            "SELECT tier, COUNT(*) FROM scores GROUP BY tier"
        ).fetchall()
        size = get_cache_path().stat().st_size
        conn.close()
        return {"total": total, "by_tier": dict(by_tier), "size_bytes": size}
    except Exception:
        return {"total": 0, "by_tier": {}, "size_bytes": 0}


def clear_cache() -> int:
    """Delete all cached scores. Returns number of rows deleted."""
    try:
        conn = _connect()
        n = conn.execute("DELETE FROM scores").rowcount
        conn.commit()
        conn.close()
        return n
    except Exception:
        return 0


def get_cache_path() -> Path:
    cache_dir = Path.home() / ".mindcheck"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "cache.db"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_cache_path()))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            file_path  TEXT    NOT NULL,
            tier       INTEGER NOT NULL,
            mtime      REAL    NOT NULL,
            version    INTEGER NOT NULL,
            data       TEXT    NOT NULL,
            cached_at  REAL    NOT NULL,
            PRIMARY KEY (file_path, tier)
        )
    """)
    conn.commit()
    return conn


def _serialize(score) -> str:
    """Serialise a SessionScore to a JSON string."""
    s = score.session
    st = score.structural
    sem = score.semantic

    return json.dumps({
        # Session identity
        "id":               s.id,
        "tool":             s.tool,
        "file_path":        str(s.file_path),
        "created_at":       s.created_at.isoformat() if s.created_at else None,
        "archived":         s.archived,
        "turn_count":       s.turn_count,
        "total_user_chars": s.total_user_chars,
        "total_ai_chars":   s.total_ai_chars,
        # Structural signals
        "structural": {
            "question_ratio":      st.question_ratio,
            "message_ratio":       st.message_ratio,
            "turn_count":          st.turn_count,
            "urgency_count":       st.urgency_count,
            "delegation_count":    st.delegation_count,
            "gratitude_count":     st.gratitude_count,
            "prior_attempt_count": st.prior_attempt_count,
            "avg_user_length":     st.avg_user_length,
            "long_message_ratio":  st.long_message_ratio,
            "single_line_ratio":   st.single_line_ratio,
        },
        # Semantic signals (per_message excluded — too large)
        "semantic": {
            "hypothesis_level_avg": sem.hypothesis_level_avg,
            "ownership_score":         sem.ownership_score,
            "critical_engagement":  sem.critical_engagement,
            "self_reliance":        sem.self_reliance,
            "metacognition_score":  sem.metacognition_score,
            "delegation_penalty":   sem.delegation_penalty,
            "task_breakdown":       sem.task_breakdown,
        },
        "composite": score.composite,
        "session_type": score.session_type,
    })


def _deserialize(data_json: str):
    """Reconstruct a SessionScore from a JSON string."""
    # Import here to avoid circular imports
    from mindcheck.scorer import SessionScore

    d = json.loads(data_json)

    # Rebuild a minimal Session — actual message content not needed since all
    # signals are pre-computed; dummy messages satisfy the property accessors.
    turn_count     = d["turn_count"]
    total_user     = d["total_user_chars"]
    total_ai       = d["total_ai_chars"]
    avg_user_chars = total_user // max(turn_count, 1)
    avg_ai_chars   = total_ai  // max(turn_count, 1)

    messages = []
    for _ in range(turn_count):
        messages.append(Message("user",      "u" * avg_user_chars))
        messages.append(Message("assistant", "a" * avg_ai_chars))

    created_at = None
    if d.get("created_at"):
        try:
            created_at = datetime.fromisoformat(d["created_at"])
        except Exception:
            pass

    session = Session(
        id=d["id"],
        tool=d["tool"],
        file_path=Path(d["file_path"]),
        messages=messages,
        created_at=created_at,
        archived=d.get("archived", False),
    )

    st_d = d["structural"]
    structural = StructuralSignals(
        question_ratio      = st_d["question_ratio"],
        message_ratio       = st_d["message_ratio"],
        turn_count          = st_d["turn_count"],
        urgency_count       = st_d["urgency_count"],
        delegation_count    = st_d["delegation_count"],
        gratitude_count     = st_d["gratitude_count"],
        prior_attempt_count = st_d["prior_attempt_count"],
        avg_user_length     = st_d["avg_user_length"],
        long_message_ratio  = st_d["long_message_ratio"],
        single_line_ratio   = st_d["single_line_ratio"],
    )

    sem_d = d["semantic"]
    semantic = SemanticSignals(
        hypothesis_level_avg = sem_d["hypothesis_level_avg"],
        ownership_score         = sem_d["ownership_score"],
        critical_engagement  = sem_d["critical_engagement"],
        self_reliance        = sem_d["self_reliance"],
        metacognition_score  = sem_d["metacognition_score"],
        delegation_penalty   = sem_d["delegation_penalty"],
        task_breakdown       = sem_d.get("task_breakdown", {}),
    )

    result = SessionScore(session=session)
    result.structural    = structural
    result.semantic      = semantic
    result.composite     = d["composite"]
    result.session_type  = d.get("session_type", "coding")
    return result
