"""Plain log files as timestamped windows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.utils.exceptions import EmptyDocumentError
from memotrix.utils.outputSturcture import build_document

LINES_PER_WINDOW = 40
_ISO = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)


def _timestamp(line: str) -> Optional[str]:
    match = _ISO.search(line.lstrip())
    return match.group("ts") if match else None


def _windows(lines: List[str]) -> List[str]:
    windows: List[str] = []
    for start in range(0, len(lines), LINES_PER_WINDOW):
        chunk = lines[start : start + LINES_PER_WINDOW]
        stamps = [ts for ts in (_timestamp(line) for line in chunk) if ts]
        heading = "# Log window"
        if stamps:
            heading += f" {stamps[0]} .. {stamps[-1]}"
        windows.append(heading + "\n\n" + "\n".join(chunk))
    return windows


class LogExtractor(BaseExtractor):
    supported_extensions = (".log",)

    def extract(self, path: Path):
        path = Path(path)
        raw_lines = self.read_text(path).splitlines()
        lines = [line for line in raw_lines if line.strip()]
        if not lines:
            raise EmptyDocumentError(f"Empty log file: {path.name}")
        text = "\n\n".join(_windows(lines))
        return build_document(
            path,
            text,
            extra_metadata={"file_type": "log", "kind": "log", "line_count": len(lines)},
        )
