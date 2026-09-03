"""JSON / text sniffing so domain extractors win over generic dumpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

_WHATSAPP = re.compile(r"^\[.+?\] .+: ")

_TEXT_KEYS = ("content", "text", "message", "body")
_AUTHOR_KEYS = ("role", "user", "author", "from", "sender")
_TIME_KEYS = ("timestamp", "ts", "time", "date", "created_at")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def looks_like_fhir(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("resourceType"), str)


def looks_like_geojson(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return data.get("type") in ("FeatureCollection", "Feature")


def _is_chat_message(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    has_text = any(isinstance(item.get(key), str) and item.get(key).strip() for key in _TEXT_KEYS)
    strong_author = any(item.get(key) not in (None, "") for key in _AUTHOR_KEYS)
    named_with_time = item.get("name") not in (None, "") and any(
        item.get(key) not in (None, "") for key in _TIME_KEYS
    )
    return has_text and (strong_author or named_with_time)


def looks_like_chat(data: Any) -> bool:
    if isinstance(data, list) and data:
        sample = data[:8]
        return sum(1 for item in sample if _is_chat_message(item)) >= max(1, (len(sample) + 1) // 2)
    if isinstance(data, dict):
        for key in ("messages", "chats", "history"):
            nested = data.get(key)
            if looks_like_chat(nested):
                return True
    return False


def looks_like_jsonld_graph(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if "@graph" in data:
        return True
    return "@context" in data and "@id" in data


def looks_like_whatsapp(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()][:8]
    if len(lines) < 2:
        return False
    hits = sum(1 for line in lines if _WHATSAPP.match(line))
    return hits >= (len(lines) + 1) // 2


def sniff_json_kind(data: Any) -> Optional[str]:
    if looks_like_fhir(data):
        return "fhir"
    if looks_like_geojson(data):
        return "geojson"
    if looks_like_jsonld_graph(data):
        return "knowledge_graph"
    if looks_like_chat(data):
        return "chat"
    return None
