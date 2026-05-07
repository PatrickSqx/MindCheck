"""Pull actual flagged message pairs from a real session for ground truth review."""
from mindcheck.parser import parse_session
from mindcheck.signals.semantic import extract_semantic
from mindcheck.signals.subtext import extract_subtext_local
from mindcheck.signals.session_type import classify_session_type
from pathlib import Path


def find_sessions():
    """Find sessions with subtext flags."""
    results = []

    # Check codex
    codex_dir = Path.home() / ".codex"
    for f in codex_dir.rglob("*.jsonl"):
        try:
            s = parse_session(f)
            if s and 15 <= s.turn_count <= 80:
                results.append(s)
        except Exception:
            pass

    # Check claude
    claude_dir = Path.home() / ".claude" / "projects"
    for f in claude_dir.rglob("*.jsonl"):
        if "subagent" in str(f):
            continue
        try:
            s = parse_session(f)
            if s and 15 <= s.turn_count <= 80:
                results.append(s)
        except Exception:
            pass

    return results


def analyze_session(session, max_flags=8):
    print(f"Session: {session.id[:60]}")
    print(f"Turns: {session.turn_count}")
    print(f"Tool: {session.tool}")
    print()

    stype = classify_session_type(session)
    sem = extract_semantic(session, session_type=stype.type)
    sub = extract_subtext_local(session, sem.per_message)

    if sub.say_then_contradict == 0 and sub.empty_self_reliance == 0:
        return False

    print(f"Say-then-contradict: {sub.say_then_contradict}")
    print(f"Empty self-reliance: {sub.empty_self_reliance}")
    print(f"Hypothesis w/o followup: {sub.hypothesis_without_followup}")
    print(f"Authenticity: {sub.authenticity_score:.2f}")
    print()

    user_msgs = session.user_messages
    classifications = sem.per_message
    shown = 0

    for flag_entry in sub.flags:
        if shown >= max_flags:
            break
        idx = flag_entry["index"]
        flags = flag_entry["flags"]
        if not flags:
            continue
        if idx >= len(user_msgs):
            continue

        # Skip very short messages (likely system output)
        content = user_msgs[idx].content
        if len(content) < 10:
            continue

        msg_text = content[:250].replace("\n", " ").strip()
        flag_str = ", ".join(flags)
        print(f"=== [{idx}] {flag_str} ===")
        print(f"  YOU: {msg_text}")

        c = classifications[idx] if idx < len(classifications) else {}
        details = []
        if c.get("hypothesis_level", 0) > 0:
            details.append(f"hyp={c['hypothesis_level']}")
        if c.get("is_metacognitive"):
            details.append("metacognitive")
        if c.get("is_delegation"):
            details.append("DELEGATION")
        if c.get("is_self_reliant"):
            details.append("self-reliant")
        print(f"       ({', '.join(details) or 'no flags'})")

        if "say_then_contradict" in flags and idx + 1 < len(user_msgs):
            next_text = user_msgs[idx + 1].content[:250].replace("\n", " ").strip()
            next_c = classifications[idx + 1] if idx + 1 < len(classifications) else {}
            print(f"  THEN: {next_text}")
            next_details = []
            if next_c.get("hypothesis_level", 0) > 0:
                next_details.append(f"hyp={next_c['hypothesis_level']}")
            if next_c.get("is_delegation"):
                next_details.append("DELEGATION")
            print(f"       ({', '.join(next_details) or 'no flags'})")

        print()
        shown += 1

    return True


if __name__ == "__main__":
    sessions = find_sessions()
    if not sessions:
        print("No suitable sessions found")
        exit()

    # Sort by turn count, try each until we find one with good flags
    found = 0
    for session in sorted(sessions, key=lambda s: s.turn_count):
        if found >= 2:
            break
        if analyze_session(session):
            found += 1
            print("=" * 70)
            print()
