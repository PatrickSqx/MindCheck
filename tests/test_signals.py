"""Tests for structural signal extraction (Tier 1 — no API, no embeddings)."""

import json
import tempfile
from pathlib import Path
from mindcheck.parser import parse_session
from mindcheck.signals.structural import extract_structural


def _session_from_messages(user_messages: list[str], ai_responses: list[str] = None):
    """Build a minimal session from message lists."""
    msgs = []
    ai = ai_responses or ["ok"] * len(user_messages)
    for u, a in zip(user_messages, ai):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for m in msgs:
        f.write(json.dumps(m) + "\n")
    f.close()
    session = parse_session(Path(f.name))
    Path(f.name).unlink()
    return session


def test_question_ratio_high():
    session = _session_from_messages([
        "why does this happen?",
        "what is the difference between X and Y?",
        "how does async work?",
    ])
    sig = extract_structural(session)
    assert sig.question_ratio == 1.0


def test_question_ratio_low():
    session = _session_from_messages([
        "fix this",
        "just do it",
        "make it work",
    ])
    sig = extract_structural(session)
    assert sig.question_ratio == 0.0


def test_delegation_detected():
    session = _session_from_messages([
        "just fix the bug",
        "just write the function",
        "just implement it",
    ])
    sig = extract_structural(session)
    assert sig.delegation_count == 3


def test_prior_attempt_detected():
    session = _session_from_messages([
        "I tried moving the middleware but it still fails",
        "I already checked the network tab",
    ])
    sig = extract_structural(session)
    assert sig.prior_attempt_count == 2


def test_message_ratio():
    session = _session_from_messages(
        user_messages=["short"],
        ai_responses=["a" * 1000],
    )
    sig = extract_structural(session)
    assert sig.message_ratio < 0.1  # user wrote much less than AI


def test_turn_count():
    session = _session_from_messages(["a", "b", "c", "d", "e"])
    sig = extract_structural(session)
    assert sig.turn_count == 5
