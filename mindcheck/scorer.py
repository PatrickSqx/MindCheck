"""
Scorer — aggregates signals into a composite cognitive engagement score.
"""

from dataclasses import dataclass, field
from pathlib import Path
from mindcheck.parser import Session
from mindcheck.signals.structural import StructuralSignals, extract_structural
from mindcheck.signals.semantic import SemanticSignals, extract_semantic
from mindcheck.signals.llm import LLMSignals, extract_llm


@dataclass
class SessionScore:
    session: Session
    structural: StructuralSignals = field(default_factory=StructuralSignals)
    semantic: SemanticSignals = field(default_factory=SemanticSignals)
    llm: LLMSignals = field(default_factory=LLMSignals)
    session_type: str = "coding"        # "coding" | "research" | "creative" | "casual"
    session_type_confidence: float = 0.0
    composite: float = 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the score breakdown."""
        s = self.session
        st = self.structural
        sem = self.semantic
        return {
            "session_id": s.id,
            "tool": s.tool,
            "archived": s.archived,
            "turn_count": s.turn_count,
            "session_type": self.session_type,
            "composite_score": round(self.composite, 1),
            "tier1_score": round(self.tier1_score, 1),
            "signals": {
                "hypothesis_level_avg": round(sem.hypothesis_level_avg, 2),
                "ownership": round(sem.ownership_score, 2),
                "critical_engagement": round(sem.critical_engagement, 2),
                "self_reliance": round(sem.self_reliance, 2),
                "metacognition": round(sem.metacognition_score, 2),
                "delegation_penalty": round(sem.delegation_penalty, 2),
            },
            "structural": {
                "question_ratio": round(st.question_ratio, 2),
                "message_ratio": round(st.message_ratio, 2),
                "turn_count": st.turn_count,
                "avg_user_length": round(st.avg_user_length, 1),
            },
            "task_breakdown": sem.task_breakdown,
        }

    @property
    def tier1_score(self) -> float:
        """Structural-only score (0–100), always available regardless of tier."""
        s = self.structural
        raw = (
            min(s.question_ratio * 100, 100) * 0.40 +
            min(s.turn_count / 10 * 100, 100) * 0.30 +
            min(s.message_ratio * 200, 100)   * 0.20 +
            min(s.prior_attempt_count / max(s.turn_count, 1) * 100, 100) * 0.10
        )
        delegation_ratio = s.delegation_count / max(s.turn_count, 1)
        penalty = min(delegation_ratio * 30, 20)
        return max(0.0, min(100.0, raw - penalty))

    def compute_composite(self, max_tier: int = 2):
        """
        Composite score (0–100) weighted across signal categories.

        Weights vary by session type (coding/research/creative/casual) —
        loaded from session_type.SCORING_WEIGHTS. This ensures research
        conversations aren't penalised for asking questions, and creative
        sessions aren't penalised for delegation.

        Tier 1 only — structural signals carry 100% of weight:
          - Question ratio:    40%  (curiosity vs commands)
          - Turn count:        30%  (dialogue depth)
          - Message ratio:     20%  (how much the user writes)
          - Prior attempts:    10%  (showed effort before asking)
        """
        from mindcheck.signals.session_type import SCORING_WEIGHTS

        s = self.structural
        sem = self.semantic
        w = SCORING_WEIGHTS.get(self.session_type, SCORING_WEIGHTS["coding"])

        # ── Tier 1: structural score (always computed) ──────────────────────
        structural_score = (
            min(s.question_ratio * 100, 100) * 0.40 +
            min(s.turn_count / 10 * 100, 100) * 0.30 +
            min(s.message_ratio * 200, 100)   * 0.20 +
            min(s.prior_attempt_count / max(s.turn_count, 1) * 100, 100) * 0.10
        )

        # ── Tier 1 only: rescale so structural carries full weight ──────────
        if max_tier == 1:
            delegation_ratio = s.delegation_count / max(s.turn_count, 1)
            penalty = min(delegation_ratio * 30, w["delegation_max"])
            self.composite = max(0.0, min(100.0, structural_score - penalty))
            return self.composite

        # ── Tier 2+: weighted composite using session-type weights ─────────
        hypothesis_score = sem.hypothesis_level_avg / 4 * 100
        ownership_score  = sem.ownership_score * 100
        critical_score   = sem.critical_engagement * 100
        self_reliance    = sem.self_reliance * 100
        metacognition    = sem.metacognition_score * 100

        # LLM refinement (Tier 3) is already baked into semantic.hypothesis_level_avg
        # by score_session() before this method is called — no separate adjustment needed.
        composite = (
            structural_score * w["structural"] +
            hypothesis_score * w["hypothesis"] +
            ownership_score  * w["ownership"] +
            critical_score   * w["critical_engagement"] +
            self_reliance    * w["self_reliance"] +
            metacognition    * w["metacognition"] +
            (sem.delegation_penalty * -w["delegation_max"])
        )

        self.composite = max(0.0, min(100.0, composite))
        return self.composite


def score_session(session: Session, max_tier: int = 2) -> SessionScore:
    """Score a single session up to the specified analysis tier.

    Results are cached in ~/.mindcheck/cache.db keyed by (file_path, tier).
    A cached result is returned immediately if the file hasn't changed.
    """
    from mindcheck.cache import get_cached, save_cached

    # For imported sessions (many conversations from one file), make the
    # cache key unique by appending the session ID to the path.
    is_import = session.tool in ("claude_chat", "chatgpt")
    if is_import:
        cache_path = Path(f"{session.file_path}#{session.id}")
    else:
        cache_path = session.file_path

    try:
        mtime = session.file_path.stat().st_mtime
    except OSError:
        mtime = 0.0   # file gone (e.g. temp extract) — skip cache lookup

    cached = get_cached(cache_path, mtime, max_tier)
    if cached is not None:
        cached.session.archived = session.archived
        return cached

    result = SessionScore(session=session)

    # Tier 1: always run
    result.structural = extract_structural(session)

    # Tier 2: embeddings (local, free)
    if max_tier >= 2:
        # Classify session type first — determines prototype overrides & scoring weights
        from mindcheck.signals.session_type import classify_session_type
        stype = classify_session_type(session)
        result.session_type = stype.type
        result.session_type_confidence = stype.confidence

        result.semantic = extract_semantic(session, session_type=stype.type)

    # Tier 3: LLM classification (optional, cheap)
    # Only runs if a provider is configured via `mindcheck config`
    if max_tier >= 3:
        from mindcheck.config import load_config, is_tier3_configured
        cfg = load_config()
        if is_tier3_configured(cfg):
            result.llm = extract_llm(session, result.semantic, cfg)
            # Apply corrected hypothesis avg if Tier 3 ran
            if result.llm.ran and result.llm.messages_reclassified > 0:
                result.semantic.hypothesis_level_avg = result.llm.hypothesis_level_avg

    result.compute_composite(max_tier=max_tier)
    save_cached(cache_path, mtime, max_tier, result)
    return result


def score_sessions(sessions: list[Session], max_tier: int = 2) -> list[SessionScore]:
    """Score a list of sessions, showing a progress bar."""
    from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TextColumn
    from rich.console import Console

    results = []
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[tool]}[/dim]"),
        console=console,
        transient=True,   # clears itself when done
    ) as progress:
        task = progress.add_task("Scoring sessions…", total=len(sessions), tool="")
        for session in sessions:
            progress.update(task, tool=f"{session.tool} · {session.id[:30]}")
            results.append(score_session(session, max_tier))
            progress.advance(task)

    return results
