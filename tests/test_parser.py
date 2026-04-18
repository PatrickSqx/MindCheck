"""Tests for the session parser."""

import json
import tempfile
from pathlib import Path
from mindcheck.parser import parse_session, Session


def _write_jsonl(messages: list[dict]) -> Path:
    """Helper: write a JSONL session file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for msg in messages:
        f.write(json.dumps(msg) + "\n")
    f.close()
    return Path(f.name)


def test_parse_basic_jsonl():
    path = _write_jsonl([
        {"role": "user", "content": "fix this bug"},
        {"role": "assistant", "content": "Here is the fix..."},
        {"role": "user", "content": "why does that work?"},
    ])
    session = parse_session(path)
    assert session is not None
    assert session.turn_count == 2
    assert len(session.user_messages) == 2
    assert session.user_messages[0].content == "fix this bug"
    path.unlink()


def test_user_messages_only():
    path = _write_jsonl([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "a" * 5000},  # long AI response
        {"role": "user", "content": "thanks"},
    ])
    session = parse_session(path)
    assert session is not None
    # total_ai_chars should be much larger
    assert session.total_ai_chars > session.total_user_chars
    # but user_messages should have only 2 entries
    assert len(session.user_messages) == 2
    path.unlink()


def test_empty_file():
    path = _write_jsonl([])
    session = parse_session(path)
    assert session is None
    path.unlink()


def test_malformed_jsonl():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write("not json\n{also not json\n")
    f.close()
    session = parse_session(Path(f.name))
    assert session is None
    Path(f.name).unlink()
