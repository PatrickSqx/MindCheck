"""
Cursor chat data finder — broad search across all Cursor directories.
"""

import os
import json
import sqlite3
from pathlib import Path

home = Path.home()
appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
localappdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))

search_roots = [
    appdata / "Cursor",
    localappdata / "Cursor",
    home / ".cursor",
    home / "AppData" / "Roaming" / "Cursor",
]

print("=== Searching for all Cursor data files ===\n")

for root in search_roots:
    if not root.exists():
        print(f"[SKIP] {root}")
        continue
    print(f"[FOUND] {root}")
    for f in sorted(root.rglob("*"), key=lambda x: x.stat().st_size if x.is_file() else 0, reverse=True):
        if not f.is_file():
            continue
        size = f.stat().st_size
        if size < 500:
            continue  # skip tiny files
        ext = f.suffix.lower()
        # Show all non-trivial files
        marker = ""
        name_lower = f.name.lower()
        if ext in (".vscdb", ".db", ".sqlite", ".json", ".jsonl"):
            marker = " <-- DB/JSON"
        if any(w in name_lower for w in ("chat", "convers", "composer", "prompt", "message", "history", "session")):
            marker = " <-- *** LIKELY CHAT DATA ***"
        print(f"  {size:>10,}  {f.relative_to(root)}{marker}")

print("\n=== Done ===")
print("Look for '*** LIKELY CHAT DATA ***' lines above.")
print("If nothing flagged, Cursor may store chat server-side only.")
