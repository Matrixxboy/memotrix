"""Chat exports (JSON / JSONL / WhatsApp txt) as episodic memory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.utils.exceptions import EmptyDocumentError
from memotrix.utils.outputSturcture import build_document

_WHATSAPP = re.compile(r"^\[(.+?)\] (.+?): (.*)$")
_TEXT_KEYS = ("content", "text", "message", "body")
_AUTHOR_KEYS = ("role", "user", "author", "from", "sender", "name")
_TIME_KEYS = ("timestamp", "ts", "time", "date", "created_at")


def _pick(item: Dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _message_line(item: Dict[str, Any]) -> str:
    author = _pick(item, _AUTHOR_KEYS) or "unknown"
    body = _pick(item, _TEXT_KEYS)
    when = _pick(item, _TIME_KEYS)
    if when:
        return f"{when} {author}: {body}"
    return f"{author}: {body}"


def _messages_from_data(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("messages", "chats", "history"):
            nested = data.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return []


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
    lines = ["# Chat"]
    for item in messages:
        body = _pick(item, _TEXT_KEYS)
        if not body:
            continue
        lines.append(_message_line(item))
    return "\n\n".join(lines)


def whatsapp_to_text(raw: str) -> str:
    lines = ["# Chat"]
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _WHATSAPP.match(stripped)
        if match:
            lines.append(f"{match.group(1)} {match.group(2)}: {match.group(3)}")
        else:
            lines.append(stripped)
    return "\n\n".join(lines)


class ChatExtractor(BaseExtractor):
    supported_extensions = (".json", ".jsonl", ".txt")

    def extract(self, path: Path, data: Any = None, *, raw_text: str | None = None):
        path = Path(path)
        suffix = path.suffix.lower()
        if raw_text is not None or (suffix == ".txt" and data is None):
            text = whatsapp_to_text(raw_text if raw_text is not None else self.read_text(path))
        elif data is not None:
            text = messages_to_text(_messages_from_data(data))
        elif suffix == ".jsonl":
            messages: List[Dict[str, Any]] = []
            for line in self.read_text(path).splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    messages.append(item)
            text = messages_to_text(messages)
        else:
            if data is None:
                with path.open("r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
            text = messages_to_text(_messages_from_data(data))
        if text.strip() in {"", "# Chat"}:
            raise EmptyDocumentError(f"No chat messages in {path.name}")
        return build_document(
            path,
            text,
            extra_metadata={"file_type": "chat", "kind": "episodic"},
        )
