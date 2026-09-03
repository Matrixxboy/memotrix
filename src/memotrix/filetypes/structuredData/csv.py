from pathlib import Path
from typing import Any, Dict, List

from memotrix.utils.outputSturcture import build_document
from .base import BaseExtractor

import csv

# Keep batches small so hybrid search can surface every section of large CSVs.
ROWS_PER_BATCH = 8


class CSVExtractor(BaseExtractor):
    supported_extensions = (".csv",)

    def extract(self, path: Path):
        from memotrix.utils.trace import mlog

        mlog("csv", f"extract start: {path.name}")
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                headers = []
            rows = [
                ["" if cell is None else str(cell).strip() for cell in row]
                for row in reader
                if any((cell or "").strip() for cell in row)
            ]

        headers = [h.strip() for h in headers]
        mlog(
            "csv",
            f"extract done: {path.name} rows={len(rows)} cols={len(headers)} "
            f"headers={headers[:12]}{'…' if len(headers) > 12 else ''}",
        )
        header_line = " | ".join(headers) if headers else ""
        paragraphs: List[str] = []
        if header_line:
            paragraphs.append(f"Columns: {header_line}")

        for row in rows:
            width = len(headers) if headers else len(row)
            padded = (row + [""] * width)[:width]
            if headers:
                pairs = [
                    f"{headers[i]}: {padded[i]}"
                    for i in range(width)
                    if headers[i] or padded[i]
                ]
                paragraphs.append(" | ".join(pairs) if pairs else " | ".join(padded))
            else:
                paragraphs.append(" | ".join(padded))

        text = "\n\n".join(paragraphs) if paragraphs else ""
        tables: List[Dict[str, Any]] = []
        if headers or rows:
            tables.append(
                {
                    "headers": headers,
                    "rows": rows,
                    "sheet": path.stem,
                    "text": text,
                }
            )

        return build_document(
            path=path,
            text=text,
            tables=tables,
            extra_metadata={
                "file_type": "csv",
                "library": "native",
                "column_count": len(headers),
                "row_count": len(rows),
                "headers": headers,
                "batch_count": max(1, (len(rows) + ROWS_PER_BATCH - 1) // ROWS_PER_BATCH)
                if rows
                else 0,
            },
        )
