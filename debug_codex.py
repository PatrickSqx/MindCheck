"""
Codex directory diagnostic — shows file structure and session_meta
contents so we can identify how agent sessions differ from main sessions.
"""

import json
import os
from pathlib import Path

home = Path.home()
localappdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))

search_dirs = [
    home / ".codex",
    localappdata / "Codex",
]

for root in search_dirs:
    if not root.exists():
        continue

    print(f"\n{'='*60}")
    print(f"ROOT: {root}")
    print(f"{'='*60}")

    jsonl_files = list(root.rglob("*.jsonl"))
    print(f"Total .jsonl files: {len(jsonl_files)}\n")

    for f in sorted(jsonl_files, key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        size = f.stat().st_size
        rel = f.relative_to(root)
        print(f"  {size:>10,}B  {rel}")

        # Read first few lines to find session_meta
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if i > 10:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "session_meta":
                            payload = obj.get("payload", {})
                            # Print the full session_meta payload
                            print(f"    session_meta: {json.dumps(payload)[:300]}")
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"    Error reading: {e}")

print("\nDone.")
