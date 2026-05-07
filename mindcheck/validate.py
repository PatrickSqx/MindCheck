"""
Subtext validation — ground truth collection and accuracy measurement.

Workflow:
  1. `sample()` — scan real sessions, find flagged pairs, merge into ground truth
  2. `label_interactive()` — terminal labeling of unlabeled entries
  3. `measure()` — re-run detection on labeled sessions, compare against labels
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


GROUND_TRUTH_PATH = Path.home() / ".mindcheck" / "subtext_ground_truth.json"


@dataclass
class ValidationResult:
    total_labeled: int
    correct_labels: int
    false_positive_labels: int
    true_positives: int
    false_positives: int
    fixed_false_positives: int
    per_flag_type: dict
    accuracy: float
    fp_rate: float


def _entry_id(session_file: str, msg_index: int, flag_type: str) -> str:
    raw = f"{session_file}:{msg_index}:{flag_type}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_ground_truth() -> list[dict]:
    if GROUND_TRUTH_PATH.exists():
        try:
            return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_ground_truth(entries: list[dict]) -> None:
    GROUND_TRUTH_PATH.parent.mkdir(exist_ok=True)
    GROUND_TRUTH_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sample(
    window: str = "90d",
    max_per_session: int = 5,
    max_total: int = 50,
    skip_archived: bool = True,
    tools: Optional[list[str]] = None,
) -> int:
    from mindcheck.parser import auto_discover_sessions
    from mindcheck.signals.semantic import extract_semantic
    from mindcheck.signals.subtext import extract_subtext_local, SUBTEXT_DETECTION_VERSION
    from mindcheck.signals.session_type import classify_session_type

    existing = load_ground_truth()
    existing_ids = {e["id"] for e in existing}

    sessions = auto_discover_sessions(window=window, skip_archived=skip_archived)
    if tools:
        sessions = [s for s in sessions if s.tool in tools]

    new_count = 0

    for session in sessions:
        if new_count >= max_total:
            break
        if session.turn_count < 2:
            continue

        try:
            stype = classify_session_type(session)
            sem = extract_semantic(session, session_type=stype.type)
        except Exception:
            continue

        if not sem.per_message:
            continue

        sub = extract_subtext_local(session, sem.per_message)
        user_msgs = session.user_messages
        session_new = 0

        for flag_entry in sub.flags:
            if session_new >= max_per_session or new_count >= max_total:
                break

            idx = flag_entry["index"]
            flags = flag_entry["flags"]
            if not flags or idx >= len(user_msgs):
                continue

            for flag_type in flags:
                eid = _entry_id(str(session.file_path), idx, flag_type)
                if eid in existing_ids:
                    continue

                msg_text = user_msgs[idx].content[:300]
                next_msg_text = None
                next_cls = None

                if flag_type == "say_then_contradict" and idx + 1 < len(user_msgs):
                    next_msg_text = user_msgs[idx + 1].content[:300]
                    if idx + 1 < len(sem.per_message):
                        next_cls = sem.per_message[idx + 1]

                cls = sem.per_message[idx] if idx < len(sem.per_message) else {}

                entry = {
                    "id": eid,
                    "session_file": str(session.file_path),
                    "session_id": session.id[:60],
                    "tool": session.tool,
                    "session_type": stype.type,
                    "msg_index": idx,
                    "flag_type": flag_type,
                    "msg_text": msg_text,
                    "next_msg_text": next_msg_text,
                    "msg_classification": cls,
                    "next_classification": next_cls,
                    "label": "UNLABELED",
                    "reason": "",
                    "labeled_at": None,
                    "detection_version": SUBTEXT_DETECTION_VERSION,
                }

                existing.append(entry)
                existing_ids.add(eid)
                new_count += 1
                session_new += 1

    save_ground_truth(existing)
    return new_count


def label_interactive(batch_size: int = 10) -> int:
    entries = load_ground_truth()
    unlabeled = [e for e in entries if e["label"] == "UNLABELED"]

    if not unlabeled:
        print("No unlabeled entries. Run `mindcheck validate sample` first.")
        return 0

    print(f"Found {len(unlabeled)} unlabeled entries. Showing up to {batch_size}.")
    print()

    labeled_count = 0
    for entry in unlabeled[:batch_size]:
        _print_entry(entry)

        while True:
            try:
                choice = input("  Label [C]orrect / [F]alse positive / [S]kip / [Q]uit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice in ("c", "correct"):
                entry["label"] = "CORRECT"
                try:
                    reason = input("  Reason (optional, Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = ""
                entry["reason"] = reason
                entry["labeled_at"] = datetime.now().isoformat()
                labeled_count += 1
                break
            elif choice in ("f", "fp", "false_positive"):
                entry["label"] = "FALSE_POSITIVE"
                try:
                    reason = input("  Reason (optional, Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = ""
                entry["reason"] = reason
                entry["labeled_at"] = datetime.now().isoformat()
                labeled_count += 1
                break
            elif choice in ("s", "skip"):
                break
            elif choice in ("q", "quit"):
                save_ground_truth(entries)
                print(f"\nLabeled {labeled_count} entries. Saved.")
                return labeled_count
            else:
                print("  Invalid choice. Use C, F, S, or Q.")

        print()

    save_ground_truth(entries)
    print(f"\nLabeled {labeled_count} entries. Saved.")
    return labeled_count


def _print_entry(entry: dict) -> None:
    print("=" * 70)
    print(f"  Flag: {entry['flag_type']}")
    print(f"  Session: {entry['session_id']}  ({entry['tool']}, {entry['session_type']})")
    print(f"  Message [{entry['msg_index']}]:")

    try:
        msg = entry["msg_text"].replace("\n", " ").strip()[:250]
        print(f"    {msg}")
    except UnicodeEncodeError:
        print(f"    {repr(entry['msg_text'][:250])}")

    cls = entry.get("msg_classification", {})
    details = []
    hl = cls.get("hypothesis_level", 0)
    if hl > 0:
        details.append(f"hyp={hl}")
    for sig in ("is_metacognitive", "is_delegation", "is_self_reliant",
                "is_critical", "is_user_driven"):
        if cls.get(sig):
            details.append(sig.replace("is_", ""))
    print(f"    Classification: {', '.join(details) or 'none'}")

    if entry.get("next_msg_text"):
        print(f"  Then [{entry['msg_index'] + 1}]:")
        try:
            next_msg = entry["next_msg_text"].replace("\n", " ").strip()[:250]
            print(f"    {next_msg}")
        except UnicodeEncodeError:
            print(f"    {repr(entry['next_msg_text'][:250])}")

        next_cls = entry.get("next_classification") or {}
        nd = []
        nhl = next_cls.get("hypothesis_level", 0)
        if nhl > 0:
            nd.append(f"hyp={nhl}")
        if next_cls.get("is_delegation"):
            nd.append("DELEGATION")
        if next_cls.get("is_self_reliant"):
            nd.append("self_reliant")
        print(f"    Classification: {', '.join(nd) or 'none'}")

    print("-" * 70)


def measure() -> Optional[ValidationResult]:
    from mindcheck.parser import parse_session
    from mindcheck.signals.semantic import extract_semantic
    from mindcheck.signals.subtext import extract_subtext_local
    from mindcheck.signals.session_type import classify_session_type

    entries = load_ground_truth()
    labeled = [e for e in entries if e["label"] in ("CORRECT", "FALSE_POSITIVE")]

    if not labeled:
        return None

    by_session: dict[str, list[dict]] = {}
    for e in labeled:
        by_session.setdefault(e["session_file"], []).append(e)

    true_positives = 0
    false_positives = 0
    fixed_fps = 0
    per_flag: dict[str, dict] = {}

    for session_file, file_entries in by_session.items():
        session_path = Path(session_file)
        if not session_path.exists():
            continue

        try:
            session = parse_session(session_path)
        except Exception:
            continue
        if not session:
            continue

        try:
            stype = classify_session_type(session)
            sem = extract_semantic(session, session_type=stype.type)
        except Exception:
            continue

        if not sem.per_message:
            continue

        sub = extract_subtext_local(session, sem.per_message)

        current_flags = set()
        for flag_entry in sub.flags:
            for ft in flag_entry["flags"]:
                current_flags.add((flag_entry["index"], ft))

        for entry in file_entries:
            ft = entry["flag_type"]
            if ft not in per_flag:
                per_flag[ft] = {"tp": 0, "fp": 0, "fixed": 0, "total": 0}
            per_flag[ft]["total"] += 1

            still_flagged = (entry["msg_index"], ft) in current_flags

            if entry["label"] == "CORRECT":
                if still_flagged:
                    true_positives += 1
                    per_flag[ft]["tp"] += 1

            elif entry["label"] == "FALSE_POSITIVE":
                if still_flagged:
                    false_positives += 1
                    per_flag[ft]["fp"] += 1
                else:
                    fixed_fps += 1
                    per_flag[ft]["fixed"] += 1

    total = true_positives + false_positives + fixed_fps
    denom = true_positives + false_positives
    accuracy = true_positives / denom if denom > 0 else 1.0
    fp_rate = false_positives / total if total > 0 else 0.0

    return ValidationResult(
        total_labeled=len(labeled),
        correct_labels=sum(1 for e in labeled if e["label"] == "CORRECT"),
        false_positive_labels=sum(1 for e in labeled if e["label"] == "FALSE_POSITIVE"),
        true_positives=true_positives,
        false_positives=false_positives,
        fixed_false_positives=fixed_fps,
        per_flag_type=per_flag,
        accuracy=accuracy,
        fp_rate=fp_rate,
    )
