"""Quick smoke test for subtext integration."""
from pathlib import Path
from mindcheck.parser import Session, Message
from mindcheck.signals.subtext import SubtextSignals, extract_subtext_local


def test_local_patterns():
    """Test that local subtext detection catches common patterns."""
    # Simulate a say-then-contradict + empty self-reliance + passive streak
    classifications = [
        {"is_metacognitive": True, "hypothesis_level": 3,
         "content": "I think the issue might be in the event loop"},
        {"is_delegation": True, "hypothesis_level": 0,
         "content": "just fix it for me"},
        {"is_self_reliant": True,
         "content": "I tried everything"},
        {"is_delegation": True,
         "content": "write the code please"},
        {"hypothesis_level": 0, "content": "ok"},
        {"hypothesis_level": 0, "content": "looks good"},
        {"hypothesis_level": 0, "content": "makes sense"},
        {"hypothesis_level": 1, "content": "sure"},
    ]

    session = Session(
        id="test", tool="test", file_path=Path("test.jsonl"),
        messages=[Message("user", "hi"), Message("assistant", "hello")] * 4,
    )

    sig = extract_subtext_local(session, classifications)

    print(f"authenticity_score:       {sig.authenticity_score:.2f}")
    print(f"say_then_contradict_candidates: {sig.say_then_contradict_candidates}")
    print(f"say_then_contradict (confirmed): {sig.say_then_contradict}")
    print(f"empty_self_reliance:      {sig.empty_self_reliance}")
    print(f"passive_acceptance_streak: {sig.passive_acceptance_streak}")
    print(f"hypothesis_without_followup: {sig.hypothesis_without_followup}")
    print(f"contradictions_found:     {sig.contradictions_found}")
    print(f"performative_count:       {sig.performative_count}")

    # Assertions
    # say_then_contradict is now candidates-only at Tier 2 (needs LLM to confirm)
    assert sig.say_then_contradict_candidates >= 1, f"Expected STC candidates, got {sig.say_then_contradict_candidates}"
    assert sig.say_then_contradict == 0, f"Confirmed STC should be 0 at Tier 2, got {sig.say_then_contradict}"
    assert sig.empty_self_reliance >= 1, f"Expected empty self-reliance, got {sig.empty_self_reliance}"
    assert sig.passive_acceptance_streak >= 3, f"Expected passive streak >= 3, got {sig.passive_acceptance_streak}"
    assert sig.authenticity_score < 1.0, f"Expected penalised authenticity, got {sig.authenticity_score}"
    assert sig.hypothesis_without_followup >= 1, f"Expected dropped hypothesis, got {sig.hypothesis_without_followup}"

    print("\nAll assertions passed!")


def test_genuine_engagement():
    """Test that genuine engagement gets high authenticity."""
    classifications = [
        {"is_metacognitive": True, "hypothesis_level": 2,
         "content": "I think the problem is a race condition in the callback"},
        {"is_critical": True, "hypothesis_level": 3,
         "content": "Wait, that doesn't match because the lock should prevent concurrent access"},
        {"hypothesis_level": 4, "is_user_driven": True,
         "content": "I tested it and the lock was being released too early, let me try a different approach"},
        {"is_metacognitive": True, "hypothesis_level": 3,
         "content": "Actually I think the issue is that we need a reentrant lock instead"},
    ]

    session = Session(
        id="test2", tool="test", file_path=Path("test.jsonl"),
        messages=[Message("user", "hi"), Message("assistant", "hello")] * 4,
    )

    sig = extract_subtext_local(session, classifications)

    print(f"\nGenuine engagement test:")
    print(f"authenticity_score:       {sig.authenticity_score:.2f}")
    print(f"contradictions_found:     {sig.contradictions_found}")
    print(f"performative_count:       {sig.performative_count}")

    assert sig.authenticity_score == 1.0, f"Expected full authenticity for genuine engagement, got {sig.authenticity_score}"
    assert sig.contradictions_found == 0
    assert sig.performative_count == 0

    print("All assertions passed!")


def test_scorer_integration():
    """Test that subtext integrates into the scorer pipeline."""
    from mindcheck.scorer import SessionScore

    session = Session(
        id="test3", tool="test", file_path=Path("test.jsonl"),
        messages=[Message("user", "hi"), Message("assistant", "hello")],
    )
    result = SessionScore(session=session)

    # SubtextSignals should exist with defaults
    assert hasattr(result, "subtext")
    assert result.subtext.authenticity_score == 1.0
    assert result.subtext.contradictions_found == 0

    print("\nScorer integration test:")
    print(f"subtext field exists: True")
    print(f"default authenticity: {result.subtext.authenticity_score}")
    print("All assertions passed!")


def test_cache_roundtrip():
    """Test that subtext survives cache serialization."""
    from mindcheck.cache import _serialize, _deserialize

    session = Session(
        id="test-cache", tool="claude_code", file_path=Path("test.jsonl"),
        messages=[Message("user", "hello world"), Message("assistant", "hi there")],
    )

    from mindcheck.scorer import SessionScore
    result = SessionScore(session=session)
    result.subtext.authenticity_score = 0.75
    result.subtext.say_then_contradict_candidates = 5
    result.subtext.say_then_contradict = 2
    result.subtext.empty_self_reliance = 1
    result.subtext.passive_acceptance_streak = 4
    result.composite = 55.0

    serialized = _serialize(result)
    restored = _deserialize(serialized)

    assert restored.subtext.authenticity_score == 0.75
    assert restored.subtext.say_then_contradict_candidates == 5
    assert restored.subtext.say_then_contradict == 2
    assert restored.subtext.empty_self_reliance == 1
    assert restored.subtext.passive_acceptance_streak == 4

    print("\nCache roundtrip test:")
    print(f"authenticity survived: {restored.subtext.authenticity_score}")
    print(f"say_then_contradict_candidates survived: {restored.subtext.say_then_contradict_candidates}")
    print(f"say_then_contradict survived: {restored.subtext.say_then_contradict}")
    print("All assertions passed!")


if __name__ == "__main__":
    test_local_patterns()
    test_genuine_engagement()
    test_scorer_integration()
    test_cache_roundtrip()
    print("\n=== ALL TESTS PASSED ===")
