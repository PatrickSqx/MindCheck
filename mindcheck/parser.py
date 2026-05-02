"""
Session parser — reads JSONL/JSON/SQLite from Claude, Cursor, Codex, Gemini,
Claude Chat, and ChatGPT exports.
Extracts user messages only (AI responses are discarded for analysis).
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class Message:
    role: str           # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None


@dataclass
class Session:
    id: str
    tool: str           # "claude" | "cursor" | "codex" | "gemini" | "claude_chat" | "chatgpt"
    file_path: Path
    messages: list[Message] = field(default_factory=list)
    created_at: Optional[datetime] = None
    archived: bool = False

    @property
    def user_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]

    @property
    def turn_count(self) -> int:
        return len(self.user_messages)

    @property
    def total_user_chars(self) -> int:
        return sum(len(m.content) for m in self.user_messages)

    @property
    def total_ai_chars(self) -> int:
        return sum(len(m.content) for m in self.messages if m.role == "assistant")


def get_known_directories() -> dict[str, list[Path]]:
    """Return known session directories for each AI tool."""
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    localappdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))

    return {
        "Claude Code": [
            home / ".claude" / "projects",
        ],
        # Cursor stores agent transcripts locally as JSONL under ~/.cursor/projects/
        # (distinct from AppData/Roaming/Cursor which is VS Code UI state only)
        "Cursor": [
            home / ".cursor" / "projects",
        ],
        "Codex": [
            home / ".codex",
            localappdata / "Codex",
        ],
        "Gemini CLI": [
            home / ".gemini",
            home / ".config" / "gemini",
        ],
    }


def auto_discover_sessions(window: str = "30d", skip_archived: bool = False) -> list[Session]:
    """Discover sessions from all known directories within the time window."""
    days = _parse_window(window)
    cutoff = datetime.now() - timedelta(days=days)
    sessions = []

    for tool, paths in get_known_directories().items():
        for path in paths:
            if not path.exists():
                continue
            sessions.extend(_scan_directory(path, tool, cutoff, skip_archived))

    return sessions


def discover_sessions(path: Path, cutoff: Optional[datetime] = None,
                      skip_archived: bool = False) -> list[Session]:
    """Discover sessions from a specific directory."""
    tool = _detect_tool(path)
    return _scan_directory(path, tool, cutoff, skip_archived)


def parse_session(file_path: Path) -> Optional[Session]:
    """Parse a single session file."""
    tool = _detect_tool(file_path)
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".jsonl":
            return _parse_jsonl(file_path, tool)
        elif suffix == ".json":
            # Check if this is a Claude Chat or ChatGPT export (array of conversations)
            if tool in ("claude_chat", "chatgpt"):
                return None  # Multi-conversation files handled by parse_export()
            return _parse_json(file_path, tool)
        elif suffix == ".db" or suffix == ".sqlite":
            return _parse_sqlite(file_path, tool)
    except Exception:
        return None

    return None


def parse_export(file_path: Path, format: str = "auto") -> list[Session]:
    """Parse an exported conversations file (Claude Chat or ChatGPT).

    Supports both raw JSON files and ZIP archives containing JSON files.
    Returns a list of Session objects — one per conversation.
    """
    import zipfile
    import tempfile

    # Handle ZIP archives — extract JSON files and parse each
    if zipfile.is_zipfile(file_path) and file_path.suffix.lower() == ".zip":
        all_sessions = []
        with zipfile.ZipFile(file_path) as zf:
            json_files = sorted(n for n in zf.namelist() if n.endswith(".json"))
            for jf in json_files:
                tmp_dir = Path(tempfile.mkdtemp())
                tmp_file = tmp_dir / Path(jf).name
                try:
                    with zf.open(jf) as src, open(tmp_file, "wb") as dst:
                        dst.write(src.read())
                    extracted = parse_export(tmp_file, format=format)
                    # Point file_path back to the original ZIP so scorer
                    # can stat() it for caching (temp files are deleted below)
                    for s in extracted:
                        s.file_path = file_path
                    all_sessions.extend(extracted)
                finally:
                    tmp_file.unlink(missing_ok=True)
                    tmp_dir.rmdir()
        return all_sessions

    # Handle directories — scan for JSON files inside
    if file_path.is_dir():
        all_sessions = []
        for jf in sorted(file_path.rglob("*.json")):
            all_sessions.extend(parse_export(jf, format=format))
        # Also check for zips
        for zf in sorted(file_path.rglob("*.zip")):
            if "conversations" in zf.name.lower() or "chatgpt" in zf.name.lower():
                all_sessions.extend(parse_export(zf, format=format))
        return all_sessions

    if format == "auto":
        format = _detect_export_format(file_path)

    if format == "claude_chat":
        return _parse_claude_chat_export(file_path)
    elif format == "chatgpt":
        return _parse_chatgpt_export(file_path)
    else:
        return []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _scan_directory(path: Path, tool: str, cutoff: Optional[datetime],
                    skip_archived: bool = False) -> list[Session]:
    sessions = []
    for f in path.rglob("*"):
        if f.suffix.lower() not in (".jsonl", ".json", ".db", ".sqlite"):
            continue
        if cutoff and datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            continue
        # Skip Claude Code subagent sessions — they are AI-to-AI, not human conversations
        if f.stem.startswith("agent-"):
            continue
        # Skip Cursor subagent sessions — stored in agent-transcripts/{id}/subagents/
        if "subagents" in f.parts:
            continue

        is_archived = any(p.startswith("archived") for p in f.relative_to(path).parts)

        if skip_archived and is_archived:
            continue

        session = parse_session(f)
        # Require at least 3 user messages with meaningful content (≥12 chars each)
        # to avoid polluting reports with button-click / one-word sessions.
        if session and session.turn_count > 0:
            meaningful = sum(
                1 for m in session.user_messages if len(m.content.strip()) >= 12
            )
            if meaningful >= 3:
                session.archived = is_archived
                sessions.append(session)
    return sessions


def _detect_tool(path: Path) -> str:
    path_str = str(path).lower()
    if "claude" in path_str:
        return "claude"
    if "cursor" in path_str:
        return "cursor"
    if "codex" in path_str:
        return "codex"
    if "gemini" in path_str:
        return "gemini"
    return "unknown"


def _parse_jsonl(file_path: Path, tool: str) -> Optional[Session]:
    """Parse JSONL session files — handles two formats:

    Format A (Claude Code):
        {"role": "user", "content": "..."}

    Format B (Codex CLI):
        {"timestamp": "...", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "..."}]}}
    """
    messages = []
    created_at = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ── Codex session_meta — detect agent sessions early ──────────
            if obj.get("type") == "session_meta" and "payload" in obj:
                source = obj["payload"].get("source", "")
                # Sub-agent sessions have source as a dict: {"subagent": {...}}
                # Main human sessions have source as a string: "vscode"
                if isinstance(source, dict) and "subagent" in source:
                    return None  # AI-to-AI agent session — skip entirely
                continue

            # ── Format B: Codex CLI (nested payload) ──────────────────────
            if obj.get("type") == "response_item" and "payload" in obj:
                payload = obj["payload"]
                role = payload.get("role", "")
                # skip developer/system messages — not the human user
                if role not in ("user", "assistant"):
                    continue
                content = _extract_content(payload)
                ts = _extract_timestamp(obj)
                if not content:
                    continue
                if role == "user":
                    messages.append(Message("user", content, ts))
                    if created_at is None:
                        created_at = ts
                else:
                    messages.append(Message("assistant", content, ts))
                continue

            # skip Codex meta lines (session_meta, etc.)
            if "type" in obj and "payload" in obj:
                continue

            # ── Format A: Claude Code (flat role/content) ──────────────────
            role = obj.get("role") or obj.get("type", "")
            content = _extract_content(obj)
            ts = _extract_timestamp(obj)

            if role in ("user", "human"):
                messages.append(Message("user", content, ts))
                if created_at is None:
                    created_at = ts
            elif role in ("assistant", "ai"):
                messages.append(Message("assistant", content, ts))

    if not messages:
        return None

    return Session(
        id=file_path.stem,
        tool=tool,
        file_path=file_path,
        messages=messages,
        created_at=created_at,
    )


def _parse_json(file_path: Path, tool: str) -> Optional[Session]:
    """Parse Gemini CLI JSON format."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    messages = []

    # Handle array of messages
    items = data if isinstance(data, list) else data.get("messages", data.get("history", []))
    for item in items:
        role = item.get("role", "")
        content = _extract_content(item)
        ts = _extract_timestamp(item)

        if role in ("user", "human"):
            messages.append(Message("user", content, ts))
        elif role in ("assistant", "model", "ai"):
            messages.append(Message("assistant", content, ts))

    if not messages:
        return None

    return Session(
        id=file_path.stem,
        tool=tool,
        file_path=file_path,
        messages=messages,
    )


def _parse_sqlite(file_path: Path, tool: str) -> Optional[Session]:
    """Parse Cursor SQLite workspace storage."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(file_path))
        cursor = conn.cursor()

        # Cursor stores chat in key-value blob — try common table/key names
        tables = [r[0] for r in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        messages = []
        for table in tables:
            try:
                rows = cursor.execute(f'SELECT key, value FROM "{table}"').fetchall()
                for key, value in rows:
                    if "chat" in str(key).lower() or "conversation" in str(key).lower():
                        parsed = _try_parse_blob(value)
                        if parsed:
                            messages.extend(parsed)
            except Exception:
                continue

        conn.close()

        if not messages:
            return None

        return Session(
            id=file_path.stem,
            tool=tool,
            file_path=file_path,
            messages=messages,
        )
    except Exception:
        return None


def _try_parse_blob(value) -> list[Message]:
    """Try to extract messages from a SQLite blob value."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return []
    try:
        data = json.loads(value)
        messages = []
        items = data if isinstance(data, list) else data.get("messages", [])
        for item in items:
            role = item.get("role", "")
            content = _extract_content(item)
            if role in ("user", "human") and content:
                messages.append(Message("user", content))
            elif role in ("assistant", "ai") and content:
                messages.append(Message("assistant", content))
        return messages
    except Exception:
        return []


def _detect_export_format(file_path: Path) -> str:
    """Auto-detect whether a JSON file is a Claude Chat or ChatGPT export."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read just enough to identify the format
            chunk = f.read(2000)
            if '"chat_messages"' in chunk and '"sender"' in chunk:
                return "claude_chat"
            if '"mapping"' in chunk and '"current_node"' in chunk:
                return "chatgpt"
    except Exception:
        pass
    return "unknown"


def _parse_claude_chat_export(file_path: Path) -> list[Session]:
    """Parse Claude Chat export (conversations.json from claude.ai).

    Format: array of conversations, each with chat_messages where
    sender is "human" or "assistant".
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    sessions = []
    for conv in data:
        messages = []
        chat_msgs = conv.get("chat_messages", [])

        for msg in chat_msgs:
            sender = msg.get("sender", "")
            text = msg.get("text", "")

            # Also try content[0].text if text is empty
            if not text:
                content_list = msg.get("content", [])
                if isinstance(content_list, list):
                    for block in content_list:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            break

            if not text:
                continue

            ts = _extract_timestamp(msg)

            if sender == "human":
                messages.append(Message("user", text.strip(), ts))
            elif sender == "assistant":
                messages.append(Message("assistant", text.strip(), ts))

        if not messages:
            continue

        created_at = None
        raw_ts = conv.get("created_at")
        if raw_ts:
            try:
                created_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except Exception:
                pass

        conv_id = conv.get("uuid", conv.get("id", file_path.stem))
        conv_name = conv.get("name", "")
        session_id = conv_name[:50] if conv_name else conv_id[:50]

        sessions.append(Session(
            id=session_id,
            tool="claude_chat",
            file_path=file_path,
            messages=messages,
            created_at=created_at,
        ))

    return sessions


def _parse_chatgpt_export(file_path: Path) -> list[Session]:
    """Parse ChatGPT export (conversations-xxx.json from OpenAI data export).

    Format: array of conversations using a tree structure (mapping dict with
    parent/children nodes). We walk from root to current_node to reconstruct
    the actual conversation thread.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    sessions = []
    for conv in data:
        mapping = conv.get("mapping", {})
        if not mapping:
            continue

        # Find root node (the one with no parent or parent not in mapping)
        root_id = None
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent is None or parent not in mapping:
                root_id = node_id
                break

        if not root_id:
            continue

        # Walk the tree: follow first child from root to build the main thread
        messages = []
        current_id = root_id
        visited = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id, {})
            msg = node.get("message")

            if msg:
                author = msg.get("author", {})
                role = author.get("role", "")
                content_obj = msg.get("content", {})
                content_type = content_obj.get("content_type", "")
                parts = content_obj.get("parts", [])

                # Extract text from parts
                text_parts = []
                for part in parts:
                    if isinstance(part, str) and part.strip():
                        text_parts.append(part.strip())
                text = " ".join(text_parts)

                if text:
                    ts = None
                    create_time = msg.get("create_time")
                    if create_time and isinstance(create_time, (int, float)):
                        try:
                            ts = datetime.fromtimestamp(create_time)
                        except Exception:
                            pass

                    if role == "user":
                        messages.append(Message("user", text, ts))
                    elif role == "assistant":
                        messages.append(Message("assistant", text, ts))
                    # Skip system, tool, and other roles

            # Move to first child
            children = node.get("children", [])
            current_id = children[0] if children else None

        if not messages:
            continue

        created_at = None
        raw_ts = conv.get("create_time")
        if raw_ts and isinstance(raw_ts, (int, float)):
            try:
                created_at = datetime.fromtimestamp(raw_ts)
            except Exception:
                pass

        conv_id = conv.get("conversation_id", conv.get("id", ""))
        title = conv.get("title", conv.get("name", ""))
        session_id = title[:50] if title else conv_id[:50]

        sessions.append(Session(
            id=session_id,
            tool="chatgpt",
            file_path=file_path,
            messages=messages,
            created_at=created_at,
        ))

    return sessions


def _extract_content(obj: dict) -> str:
    """Extract text content from various message formats.

    Handles:
      - Flat:   {"role": "user", "content": "..."}           (Claude Code)
      - Nested: {"role": "user", "message": {"content": "..."}} (Cursor agent)
      - Blocks: {"content": [{"type": "text", "text": "..."}]}
    """
    # Cursor agent format: content lives inside a "message" wrapper
    msg = obj.get("message")
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = obj.get("content", obj.get("text", ""))

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Skip tool_result blocks — they are AI-internal, not human text
                if part.get("type") == "tool_result":
                    continue
                val = part.get("text", part.get("content", ""))
                # val might itself be a list (nested content blocks) — flatten one level
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    for subpart in val:
                        if isinstance(subpart, str):
                            parts.append(subpart)
                        elif isinstance(subpart, dict):
                            t = subpart.get("text", "")
                            if isinstance(t, str):
                                parts.append(t)
        return " ".join(p for p in parts if p).strip()
    return ""


def _extract_timestamp(obj: dict) -> Optional[datetime]:
    """Try to extract a timestamp from a message object."""
    for key in ("timestamp", "created_at", "time", "ts"):
        val = obj.get(key)
        if val:
            try:
                if isinstance(val, (int, float)):
                    return datetime.fromtimestamp(val / 1000 if val > 1e10 else val)
                return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            except Exception:
                continue
    return None


def _parse_window(window: str) -> int:
    """Parse '30d', '7d', '90d' into integer days."""
    window = window.strip().lower()
    if window.endswith("d"):
        return int(window[:-1])
    if window.endswith("w"):
        return int(window[:-1]) * 7
    if window.endswith("m"):
        return int(window[:-1]) * 30
    return 30
