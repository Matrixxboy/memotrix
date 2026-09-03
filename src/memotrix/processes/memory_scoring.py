"""Memory-specific ranking helpers (recency, access, importance)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional


RECENCY_HALF_LIFE_DAYS = 30.0


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def memory_score_multiplier(
    payload: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """
    Soft boost for agent memory ranking.

    recent / frequently-used / high-importance memories outrank ties.
    Multiplier is intentionally mild so relevance still dominates.
    """
    now = now or datetime.now(timezone.utc)
    last = (
        _parse_ts(payload.get("last_accessed"))
        or _parse_ts(payload.get("created_at"))
        or _parse_ts(payload.get("ingested_at"))
    )
    if last is None:
        recency = 0.5
    else:
        age_days = max(0.0, (now - last).total_seconds() / 86400.0)
        recency = math.exp(-age_days / max(half_life_days, 1e-6))

    access = float(payload.get("access_count") or 0)
    importance = float(payload.get("importance") or 1.0)

    return (
        1.0
        + 0.15 * recency
        + 0.05 * math.log1p(access)
        + 0.08 * math.log1p(max(importance, 0.0))
    )


def stamp_new_memory_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure ingest-time memory fields exist on a chunk payload."""
    now = datetime.now(timezone.utc).isoformat()
    payload.setdefault("created_at", now)
    payload.setdefault("ingested_at", now)
    payload.setdefault("last_accessed", now)
    payload.setdefault("access_count", 0)
    payload.setdefault("importance", 1.0)
    payload.setdefault("memory_type", payload.get("memory_type") or "semantic")
    return payload


def bump_access(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record a successful retrieval surface for MemGPT-style importance."""
    now = datetime.now(timezone.utc).isoformat()
    payload["last_accessed"] = now
    payload["access_count"] = int(payload.get("access_count") or 0) + 1
    return payload
