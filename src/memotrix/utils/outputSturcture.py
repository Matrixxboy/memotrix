import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .exceptions import EmptyDocumentError
from .models import DocumentData


def get_file_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> List[Dict[str, Any]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    sections: List[Dict[str, Any]] = []
    current_title = "Document"
    current_lines: List[str] = []

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            if current_lines:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                    "level": 1,
                })
            current_title = heading_match.group(2).strip()
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines or not sections:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
            "level": 1,
        })

    return sections


def build_document(
    path: Path,
    text: str,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
    tables: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Dict[str, Any]]] = None,
    language: Optional[str] = None,
    summary: Optional[str] = None,
    transcripts: Optional[List[Dict[str, Any]]] = None,
) -> DocumentData:
    images = images or []
    cleaned_text = normalize_whitespace(text)

    # Image-only / scanned docs: synthesize searchable text from captions.
    if not cleaned_text and images:
        from memotrix.filetypes.document.media import image_descriptions_as_text

        cleaned_text = normalize_whitespace(image_descriptions_as_text(images))

    if not cleaned_text and not images:
        raise EmptyDocumentError(f"The document at {path} is empty.")

    metadata = get_file_metadata(path)
    if extra_metadata:
        metadata.update(extra_metadata)

    sections = split_sections(cleaned_text) if cleaned_text else []
    validation = {
        "is_empty": False,
        "word_count": len(cleaned_text.split()) if cleaned_text else 0,
        "char_count": len(cleaned_text),
        "has_tables": bool(tables),
        "has_images": bool(images),
        "image_count": len(images),
    }

    return DocumentData(
        metadata=metadata,
        sections=sections,
        tables=tables or [],
        images=images,
        text=cleaned_text,
        raw_text=cleaned_text,
        language=language,
        summary=summary,
        transcripts=transcripts or [],
        validation=validation,
    )
