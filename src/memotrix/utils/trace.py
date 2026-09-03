"""Terminal process tracing for Memotrix pipelines.

Enable/disable with MEMOTRIX_TRACE=1 (default) or 0.
Truncate long prompts with MEMOTRIX_TRACE_MAX=4000 (chars).

Keep this module free of other src.* imports so it can be imported early.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional


def _enabled() -> bool:
    raw = (os.getenv("MEMOTRIX_TRACE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _max_chars() -> int:
    try:
        return max(200, int(os.getenv("MEMOTRIX_TRACE_MAX") or "4000"))
    except ValueError:
        return 4000


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _clip(text: str, limit: Optional[int] = None) -> str:
    limit = limit if limit is not None else _max_chars()
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + f"\n… [{len(text)} chars total]"


def mlog(stage: str, message: str, *args: Any) -> None:
    """One-line process log: [Memotrix|STAGE] message"""
    if not _enabled():
        return
    if args:
        try:
            message = message % args
        except Exception:  # noqa: BLE001
            message = f"{message} {' '.join(str(a) for a in args)}"
    stage = (stage or "app").upper().replace(" ", "_")
    line = f"[{_ts()}] [Memotrix|{stage}] {message}"
    print(line, file=sys.stderr, flush=True)


def mlog_block(stage: str, title: str, body: str, *, max_chars: Optional[int] = None) -> None:
    """Multi-line dump (prompts, context, scripts)."""
    if not _enabled():
        return
    stage = (stage or "app").upper().replace(" ", "_")
    clipped = _clip(body or "", max_chars)
    bar = "-" * 56
    print(
        f"[{_ts()}] [Memotrix|{stage}] {title}\n{bar}\n{clipped}\n{bar}",
        file=sys.stderr,
        flush=True,
    )


def mlog_kv(stage: str, title: str, **fields: Any) -> None:
    """Key/value summary line."""
    if not _enabled():
        return
    parts = []
    for k, v in fields.items():
        if isinstance(v, str) and len(v) < 80:
            parts.append(f"{k}={v!r}")
        else:
            parts.append(f"{k}={v}")
    mlog(stage, f"{title} | " + " ".join(parts))
