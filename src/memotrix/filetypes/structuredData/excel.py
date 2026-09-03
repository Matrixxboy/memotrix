from pathlib import Path
from typing import Any, Dict, List

from .base import BaseExtractor
from memotrix.utils.outputSturcture import build_document


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class ExcelExtractor(BaseExtractor):
    supported_extensions = (".xlsx",)

    def extract(self, path: Path):
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("openpyxl is required for Excel extraction") from exc

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet_texts: List[str] = []
        tables: List[Dict[str, Any]] = []

        for sheet in workbook.worksheets:
            raw_rows: List[List[str]] = []
            for row in sheet.iter_rows(values_only=True):
                cells = [_cell(v) for v in row]
                if any(cells):
                    raw_rows.append(cells)

            if not raw_rows:
                continue

            headers = raw_rows[0]
            data_rows = raw_rows[1:] if len(raw_rows) > 1 else []

            # One paragraph per data row (header prefix) so chunking keeps rows intact.
            paragraphs: List[str] = []
            header_line = " | ".join(headers)
            paragraphs.append(f"Sheet: {sheet.title}\nColumns: {header_line}")
            for row in data_rows:
                # Pad/truncate to header width for stable column alignment
                width = len(headers)
                padded = (row + [""] * width)[:width]
                pairs = [
                    f"{headers[i]}: {padded[i]}"
                    for i in range(width)
                    if headers[i] or padded[i]
                ]
                paragraphs.append(" | ".join(pairs) if pairs else " | ".join(padded))

            sheet_text = "\n\n".join(paragraphs)
            sheet_texts.append(sheet_text)
            tables.append(
                {
                    "headers": headers,
                    "rows": data_rows,
                    "sheet": sheet.title,
                    "text": sheet_text,
                }
            )

        text = "\n\n".join(sheet_texts)
        return build_document(
            path,
            text,
            tables=tables,
            extra_metadata={"file_type": "excel", "library": "openpyxl"},
        )
