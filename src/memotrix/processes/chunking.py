import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from memotrix.utils.models import DocumentData
from memotrix.utils.ai_integration import build_image_search_text, parse_image_classification

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Keep a whole unit when it is near the chunk size limit.
PARAGRAPH_SLACK = 0.2
SENTENCE_SLACK = 0.15
WORD_SLACK = 0.1

SENTENCE_SPLIT = re.compile(r'(?<=[.!?…])(?:["\'\)\]]*)?\s+')


@dataclass
class Chunk:
    id: str
    text: str
    is_dense_indexable: bool
    is_sparse_indexable: bool
    payload: Dict[str, Any]


def generate_chunk_id(text: str, metadata: Dict[str, Any]) -> str:
    """Generate a deterministic id from content plus source/section/index."""
    parts = [
        text,
        str(metadata.get("source_path") or ""),
        str(metadata.get("filename") or ""),
        str(metadata.get("section_title") or ""),
        str(metadata.get("type") or ""),
        str(metadata.get("chunk_index") if metadata.get("chunk_index") is not None else ""),
        str(metadata.get("table_index") if metadata.get("table_index") is not None else ""),
        str(metadata.get("row_start") if metadata.get("row_start") is not None else ""),
        str(metadata.get("image_index") if metadata.get("image_index") is not None else ""),
    ]
    content = "\0".join(parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _joined_length(parts: List[str], separator: str) -> int:
    if not parts:
        return 0
    return sum(len(part) for part in parts) + len(separator) * (len(parts) - 1)


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def _split_sentences(text: str) -> List[str]:
    parts = SENTENCE_SPLIT.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_words(text: str) -> List[str]:
    return text.split()


def _apply_overlap(previous: List[str], separator: str, overlap: int) -> List[str]:
    if overlap <= 0 or not previous:
        return []

    kept: List[str] = []
    total = 0
    for unit in reversed(previous):
        extra = len(separator) if kept else 0
        if total + len(unit) + extra > overlap:
            break
        kept.insert(0, unit)
        total += len(unit) + extra
    return kept


def _chunk_units(
    units: List[str],
    *,
    separator: str,
    max_size: int,
    slack: float,
    overlap: int,
    subdivide: Optional[Callable[[str], List[str]]] = None,
) -> List[str]:
    """
    Pack units without splitting them.
    Near-limit units stay whole; oversized units fall back to subdivide().
    """
    if not units:
        return []

    limit = max_size + int(max_size * slack)
    chunks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        chunks.append(separator.join(current))
        current = _apply_overlap(current, separator, overlap)

    for unit in units:
        if len(unit) > limit:
            flush()
            if subdivide:
                chunks.extend(subdivide(unit))
            else:
                chunks.append(unit)
            continue

        if not current:
            current = [unit]
            continue

        if _joined_length(current + [unit], separator) <= limit:
            current.append(unit)
        else:
            flush()
            if current and _joined_length(current + [unit], separator) <= limit:
                current.append(unit)
            else:
                current = [unit]

    if current:
        chunks.append(separator.join(current))

    return chunks


def _chunk_sentence(
    sentence: str,
    *,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    return _chunk_units(
        _split_words(sentence),
        separator=" ",
        max_size=chunk_size,
        slack=WORD_SLACK,
        overlap=overlap,
        subdivide=None,
    )


def _chunk_paragraph(
    paragraph: str,
    *,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    return _chunk_units(
        _split_sentences(paragraph),
        separator=" ",
        max_size=chunk_size,
        slack=SENTENCE_SLACK,
        overlap=overlap,
        subdivide=lambda sentence: _chunk_sentence(
            sentence, chunk_size=chunk_size, overlap=overlap
        ),
    )


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text without breaking paragraphs, sentences, or words when possible.
    Whole paragraphs near the size limit are kept intact.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    return _chunk_units(
        paragraphs,
        separator="\n\n",
        max_size=chunk_size,
        slack=PARAGRAPH_SLACK,
        overlap=overlap,
        subdivide=lambda paragraph: _chunk_paragraph(
            paragraph, chunk_size=chunk_size, overlap=overlap
        ),
    )


def _extract_page_metadata(title: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if not title:
        return meta

    page_m = re.search(r"\bPage\s+(\d+)\b", title, re.I)
    if page_m:
        meta["page_number"] = int(page_m.group(1))

    slide_m = re.search(r"\bSlide\s+(\d+)\b", title, re.I)
    if slide_m:
        meta["slide_number"] = int(slide_m.group(1))

    sheet_m = re.search(r"\bSheet:\s*([^\n|]+)", title, re.I)
    if sheet_m:
        meta["sheet_name"] = sheet_m.group(1).strip()

    return meta


def chunk_document(
    doc: DocumentData,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Splits the document into semantic chunks suitable for dual-representation indexing.
    Text is chunked. Images have descriptions embedded. Tables have headers embedded.
    """
    chunks: List[Chunk] = []

    # Excel/CSV-style docs already get dense row chunks from ``tables`` —
    # skip section text to avoid duplicate / truncated row packs.
    skip_sections = bool(
        doc.tables
        and doc.metadata.get("file_type") in {"excel", "csv", "xlsx", "xls"}
    )

    doc_name = doc.metadata.get("filename") or doc.metadata.get("source_path") or ""
    doc_name = Path(doc_name).name if doc_name else ""

    if not skip_sections:
        for section in doc.sections:
            title = section.get("title", "")
            content = section.get("content", "")

            if not content.strip():
                continue

            page_meta = _extract_page_metadata(title)
            splits = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            for i, split in enumerate(splits):
                header_parts = []
                if doc_name:
                    header_parts.append(f"Document: {doc_name}")
                if "page_number" in page_meta:
                    header_parts.append(f"Page: {page_meta['page_number']}")
                elif "slide_number" in page_meta:
                    header_parts.append(f"Slide: {page_meta['slide_number']}")
                elif title:
                    header_parts.append(f"Section: {title}")

                header = f"[{' | '.join(header_parts)}]\n" if header_parts else ""
                full_text = f"{header}{split}" if header and not split.startswith("[") else split

                payload = {
                    **doc.metadata,
                    **page_meta,
                    "type": "text",
                    "section_title": title,
                    "chunk_index": i,
                    "chunk_text": full_text,
                    "raw_text": split,
                }
                chunk_id = generate_chunk_id(full_text, payload)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=full_text,
                        is_dense_indexable=True,
                        is_sparse_indexable=True,
                        payload=payload,
                    )
                )

    for i, table in enumerate(doc.tables):
        headers = table.get("headers", []) or []
        rows = table.get("rows") or []
        sheet = table.get("sheet") or ""

        # Prefer real row chunks so spreadsheet queries can retrieve every line.
        if headers and rows:
            header_line = " | ".join(str(h) for h in headers)
            # Pack a few rows per chunk (~keeps lists complete under moderate top_k).
            batch_size = 8
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                line_parts = []
                if doc_name:
                    line_parts.append(f"[Document: {doc_name} | Table]")
                if sheet:
                    line_parts.append(f"Sheet: {sheet}")
                line_parts.append(f"Table columns: {header_line}")
                for row in batch:
                    width = len(headers)
                    padded = (list(row) + [""] * width)[:width]
                    pairs = [
                        f"{headers[j]}: {padded[j]}"
                        for j in range(width)
                        if headers[j] or padded[j]
                    ]
                    line_parts.append(" | ".join(pairs))
                table_text = "\n".join(line_parts)
                payload = {
                    **doc.metadata,
                    "type": "table",
                    "table_index": i,
                    "row_start": start,
                    "row_end": start + len(batch) - 1,
                    "chunk_text": table_text,
                }
                chunk_id = generate_chunk_id(table_text, payload)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=table_text,
                        is_dense_indexable=True,
                        is_sparse_indexable=True,
                        payload=payload,
                    )
                )
            continue

        if not headers and "text" in table:
            table_text = table["text"]
        else:
            table_text = f"Table containing columns: {', '.join(headers)}."

        if doc_name:
            table_text = f"[Document: {doc_name} | Table]\n{table_text}"

        payload = {
            **doc.metadata,
            "type": "table",
            "table_index": i,
            "chunk_text": table_text,
        }
        chunk_id = generate_chunk_id(table_text, payload)
        chunks.append(
            Chunk(
                id=chunk_id,
                text=table_text,
                is_dense_indexable=True,
                is_sparse_indexable=True,
                payload=payload,
            )
        )

    for i, image in enumerate(doc.images):
        description = image.get("description", "")
        tags = list(image.get("tags", []) or [])
        path = image.get("path", "")
        visual_kind = str(image.get("visual_kind") or "other").lower()
        caption = str(image.get("caption") or "").strip()
        is_decorative = bool(image.get("is_decorative"))

        if not description:
            description = f"An image with tags: {', '.join(tags)}"
        elif not isinstance(description, str):
            description = str(description)

        # Recover classification from older captions that only have free text.
        if visual_kind == "other" or not caption:
            parsed = parse_image_classification(description)
            if visual_kind == "other":
                visual_kind = parsed["visual_kind"]
            if not caption:
                caption = parsed["caption"]
            is_decorative = is_decorative or parsed["is_decorative"]

        if not caption:
            caption = Path(path).name if path else f"image_{i}"

        search_text = build_image_search_text(
            description=description,
            visual_kind=visual_kind,
            caption=caption,
            tags=tags,
            is_decorative=is_decorative,
        )

        if doc_name:
            search_text = f"[Document: {doc_name} | Image]\n{search_text}"

        payload = {
            **doc.metadata,
            "type": "image",
            "image_index": i,
            "image_path": str(path),
            "tags": tags,
            "visual_kind": visual_kind,
            "caption": caption,
            "is_decorative": is_decorative,
            "chunk_text": search_text,
        }
        chunk_id = generate_chunk_id(search_text, payload)
        chunks.append(
            Chunk(
                id=chunk_id,
                text=search_text,
                is_dense_indexable=True,
                is_sparse_indexable=True,
                payload=payload,
            )
        )

    if getattr(doc, "summary", None):
        summary_text = doc.summary
        if doc_name:
            summary_text = f"[Document: {doc_name} | Summary]\n{summary_text}"
        payload = {
            **doc.metadata,
            "type": "summary",
            "chunk_text": summary_text,
        }
        chunk_id = generate_chunk_id(summary_text, payload)
        chunks.append(
            Chunk(
                id=chunk_id,
                text=summary_text,
                is_dense_indexable=True,
                is_sparse_indexable=True,
                payload=payload,
            )
        )

    if getattr(doc, "transcripts", None):
        current_chunk_text = ""
        current_start = -1.0
        current_end = -1.0
        chunk_index = 0
        
        for segment in doc.transcripts:
            text = segment["text"]
            start = segment["start_time"]
            end = segment["end_time"]
            
            if not current_chunk_text:
                current_chunk_text = text
                current_start = start
                current_end = end
            elif len(current_chunk_text) + len(text) < chunk_size:
                current_chunk_text += " " + text
                current_end = end
            else:
                full_t_text = f"[Document: {doc_name} | Audio Transcript]\n{current_chunk_text}" if doc_name else current_chunk_text
                payload = {
                    **doc.metadata,
                    "type": "transcript",
                    "chunk_index": chunk_index,
                    "start_time": current_start,
                    "end_time": current_end,
                    "chunk_text": full_t_text,
                }
                chunk_id = generate_chunk_id(full_t_text, payload)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=full_t_text,
                        is_dense_indexable=True,
                        is_sparse_indexable=True,
                        payload=payload,
                    )
                )
                chunk_index += 1
                current_chunk_text = text
                current_start = start
                current_end = end
                
        if current_chunk_text:
            full_t_text = f"[Document: {doc_name} | Audio Transcript]\n{current_chunk_text}" if doc_name else current_chunk_text
            payload = {
                **doc.metadata,
                "type": "transcript",
                "chunk_index": chunk_index,
                "start_time": current_start,
                "end_time": current_end,
                "chunk_text": full_t_text,
            }
            chunk_id = generate_chunk_id(full_t_text, payload)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=full_t_text,
                    is_dense_indexable=True,
                    is_sparse_indexable=True,
                    payload=payload,
                )
            )

    return chunks
