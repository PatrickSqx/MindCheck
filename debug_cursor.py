"""
Cursor SQLite diagnostic v2 — finds where chat data actually lives.
"""

import os
import json
import sqlite3
from pathlib import Path

appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))

search_dirs = [
    appdata / "Cursor" / "User" / "workspaceStorage",
    appdata / "Cursor" / "User" / "globalStorage",
    appdata / "Cursor" / "User",
]

db_files = []
for d in search_dirs:
    if d.exists():
        for ext in ("*.vscdb", "*.db", "*.sqlite"):
            db_files += list(d.rglob(ext))

# Deduplicate, sort by size descending (biggest = most likely to have chat data)
seen = set()
unique = []
for f in db_files:
    if f not in seen:
        seen.add(f)
        unique.append(f)
unique.sort(key=lambda f: f.stat().st_size, reverse=True)

print(f"Found {len(unique)} database file(s), sorted by size:\n")
for f in unique:
    print(f"  {f.stat().st_size:>10,} bytes  {f}")

print()

for db_path in unique:
    size = db_path.stat().st_size
    print(f"{'='*70}")
    print(f"FILE: {db_path.name}  ({size:,} bytes)")
    print(f"PATH: {db_path.parent}")

    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        tables = [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        print(f"TABLES: {tables}")

        for table in tables:
            try:
                count = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                cols = [r[1] for r in c.execute(f'PRAGMA table_info("{table}")').fetchall()]
                print(f"\n  [{table}] {count} rows, columns: {cols}")

                if "key" in cols and "value" in cols:
                    rows = c.execute(f'SELECT key, value FROM "{table}"').fetchall()
                    for key, value in rows:
                        # Decode value
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8", errors="replace")
                            except Exception:
                                value = f"<binary {len(value)} bytes>"

                        val_str = str(value)

                        # Flag keys that look chat-related
                        key_lower = str(key).lower()
                        is_interesting = any(w in key_lower for w in (
                            "chat", "convers", "composer", "prompt", "message",
                            "history", "session", "ai", "llm", "bubble", "thread"
                        ))

                        if is_interesting:
                            print(f"\n    *** KEY: {key!r}")
                            print(f"        VAL[:300]: {val_str[:300]!r}")
                        else:
                            print(f"    KEY: {key!r}  (val len={len(val_str)})")

            except Exception as e:
                print(f"  Error reading {table}: {e}")

        conn.close()

    except Exception as e:
        print(f"  Could not open: {e}")

print(f"\n{'='*70}")
print("Done.")
