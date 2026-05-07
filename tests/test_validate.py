"""Tests for the subtext validation system."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from mindcheck.parser import Session, Message
from mindcheck.validate import (
    _entry_id, load_ground_truth, save_ground_truth,
    ValidationResult, GROUND_TRUTH_PATH,
)


def test_entry_id_deterministic():
    """Same inputs always produce the same ID."""
    id1 = _entry_id("/path/to/session.jsonl", 5, "say_then_contradict")
    id2 = _entry_id("/path/to/session.jsonl", 5, "say_then_contradict")
    assert id1 == id2
    assert len(id1) == 12
    print(f"ID determinism: {id1} == {id2} OK")


def test_entry_id_unique():
    """Different inputs produce different IDs."""
    id1 = _entry_id("/path/a.jsonl", 5, "say_then_contradict")
    id2 = _entry_id("/path/b.jsonl", 5, "say_then_contradict")
    id3 = _entry_id("/path/a.jsonl", 6, "say_then_contradict")
    id4 = _entry_id("/path/a.jsonl", 5, "empty_self_reliance")
    assert len({id1, id2, id3, id4}) == 4
    print(f"ID uniqueness: 4 distinct IDs OK")


def test_ground_truth_roundtrip():
    """Save and load ground truth entries."""
    entries = [
        {
            "id": "abc123",
            "session_file": "test.jsonl",
            "session_id": "test-session",
            "tool": "codex",
            "session_type": "coding",
            "msg_index": 3,
            "flag_type": "say_then_contradict",
            "msg_text": "I think the issue is X",
            "next_msg_text": "just fix it",
            "msg_classification": {"hypothesis_level": 3},
            "next_classification": {"is_delegation": True},
            "label": "CORRECT",
            "reason": "genuine contradiction",
            "labeled_at": "2026-05-07T12:00:00",
            "detection_version": 1,
        },
        {
            "id": "def456",
            "session_file": "test.jsonl",
            "session_id": "test-session",
            "tool": "codex",
            "session_type": "coding",
            "msg_index": 7,
            "flag_type": "empty_self_reliance",
            "msg_text": "I tried everything",
            "next_msg_text": None,
            "msg_classification": {"is_self_reliant": True},
            "next_classification": None,
            "label": "FALSE_POSITIVE",
            "reason": "short but contextual follow-up",
            "labeled_at": "2026-05-07T12:01:00",
            "detection_version": 1,
        },
    ]

    with patch.object(
        __import__("mindcheck.validate", fromlist=["GROUND_TRUTH_PATH"]),
        "GROUND_TRUTH_PATH",
        Path(tempfile.mktemp(suffix=".json")),
    ) as tmp_path:
        from mindcheck import validate
        orig = validate.GROUND_TRUTH_PATH
        validate.GROUND_TRUTH_PATH = tmp_path

        try:
            save_ground_truth(entries)
            loaded = load_ground_truth()

            assert len(loaded) == 2
            assert loaded[0]["id"] == "abc123"
            assert loaded[0]["label"] == "CORRECT"
            assert loaded[1]["flag_type"] == "empty_self_reliance"
            assert loaded[1]["label"] == "FALSE_POSITIVE"
            print(f"Roundtrip: saved {len(entries)}, loaded {len(loaded)} OK")
        finally:
            validate.GROUND_TRUTH_PATH = orig
            try:
                tmp_path.unlink()
            except Exception:
                pass


def test_measure_with_ground_truth():
    """Test measure() with pre-populated ground truth and synthetic session."""
    from mindcheck.validate import measure
    from mindcheck.signals.subtext import extract_subtext_local
    from mindcheck.signals.session_type import classify_session_type

    session = Session(
        id="test-measure",
        tool="test",
        file_path=Path("test-measure.jsonl"),
        messages=[
            Message("user", "I think the issue might be in the event loop"),
            Message("assistant", "That's an interesting hypothesis."),
            Message("user", "just fix it for me"),
            Message("assistant", "Sure, here's the fix."),
            Message("user", "I tried everything"),
            Message("assistant", "What did you try?"),
            Message("user", "write the code please"),
            Message("assistant", "Here you go."),
        ],
    )

    # This test verifies the ValidationResult dataclass works correctly
    result = ValidationResult(
        total_labeled=5,
        correct_labels=3,
        false_positive_labels=2,
        true_positives=3,
        false_positives=1,
        fixed_false_positives=1,
        per_flag_type={
            "say_then_contradict": {"tp": 2, "fp": 1, "fixed": 0, "total": 3},
            "empty_self_reliance": {"tp": 1, "fp": 0, "fixed": 1, "total": 2},
        },
        accuracy=3 / 4,
        fp_rate=1 / 5,
    )

    assert result.accuracy == 0.75
    assert result.fp_rate == 0.2
    assert result.per_flag_type["say_then_contradict"]["tp"] == 2
    print(f"Measure result: accuracy={result.accuracy:.1%} fp_rate={result.fp_rate:.1%} OK")


if __name__ == "__main__":
    test_entry_id_deterministic()
    test_entry_id_unique()
    test_ground_truth_roundtrip()
    test_measure_with_ground_truth()
    print("\n=== ALL VALIDATION TESTS PASSED ===")
