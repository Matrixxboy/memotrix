"""Filetype extraction package."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, List

from memotrix.filetypes.audio.audio import AudioExtractor
from memotrix.filetypes.communication.chat import ChatExtractor
from memotrix.filetypes.communication.email import EmailExtractor
from memotrix.filetypes.document import DocumentExtractor
from memotrix.filetypes.geospatial.geojson import GeoJSONExtractor
from memotrix.filetypes.images.images import ImageExtractor
from memotrix.filetypes.knowledge.graph import KnowledgeGraphExtractor
from memotrix.filetypes.logs.log import LogExtractor
from memotrix.filetypes.medical.fhir import FHIRExtractor
from memotrix.filetypes.programmingFiles.extractor import ProgrammingFileExtractor
from memotrix.filetypes.scorm.scorm import ScormExtractor
from memotrix.filetypes.sniff import (
    looks_like_whatsapp,
    sniff_json_kind,
)
from memotrix.filetypes.structuredData import StructuredDataExtractor
from memotrix.filetypes.structuredData.json import JSONExtractor
from memotrix.filetypes.video.video import VideoExtractor
from memotrix.utils.exceptions import UnsupportedDocumentTypeError

_KG_EXTENSIONS = KnowledgeGraphExtractor.supported_extensions
_EMAIL_EXTENSIONS = EmailExtractor.supported_extensions
_GEO_EXTENSIONS = GeoJSONExtractor.supported_extensions
_LOG_EXTENSIONS = LogExtractor.supported_extensions
_AUDIO_EXTENSIONS = AudioExtractor.supported_extensions


def supported_extensions() -> List[str]:
    """Extensions routed by ``extract_file`` (wired extractors only)."""
    return sorted(
        {
            *DocumentExtractor.EXTRACTORS.keys(),
            *StructuredDataExtractor.EXTRACTORS.keys(),
            *ImageExtractor.SUPPORTED_EXTENSIONS,
            *VideoExtractor.supported_extensions,
            *ScormExtractor.supported_extensions,
            *ProgrammingFileExtractor.supported_extensions,
            *_AUDIO_EXTENSIONS,
            *_KG_EXTENSIONS,
            *_EMAIL_EXTENSIONS,
            *_GEO_EXTENSIONS,
            *_LOG_EXTENSIONS,
            ".jsonl",
        }
    )


def _extract_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    kind = sniff_json_kind(data)
    if kind == "fhir":
        return FHIRExtractor().extract(path, data=data)
    if kind == "geojson":
        return GeoJSONExtractor().extract(path, data=data)
    if kind == "chat":
        return ChatExtractor().extract(path, data=data)
    if kind == "knowledge_graph":
        return KnowledgeGraphExtractor().extract(path, data=data)
    return JSONExtractor().extract(path, data=data)


def _extract_jsonl(path: Path) -> Any:
    messages: list[dict] = []
    events: list[str] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(item, dict):
            messages.append(item)
            events.append(json.dumps(item, ensure_ascii=False))
    extra = {"file_type": "jsonl", "kind": "text"}
    if skipped:
        extra["skipped_lines"] = skipped
    if messages:
        from memotrix.filetypes.sniff import looks_like_chat

        if looks_like_chat(messages):
            doc = ChatExtractor().extract(path, data=messages)
            if skipped:
                doc.metadata["skipped_lines"] = skipped
            return doc
    from memotrix.utils.outputSturcture import build_document

    text = "\n\n".join(events)
    return build_document(path, text or "\n".join(events), extra_metadata=extra)


def _extract_zip(path: Path) -> Any:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        raise UnsupportedDocumentTypeError(
            f"Unsupported file type: .zip ({path.name} is not a valid zip archive)."
        ) from exc
    if "imsmanifest.xml" in names:
        return ScormExtractor().extract(path)
    raise UnsupportedDocumentTypeError(
        "Unsupported file type: .zip. Only SCORM packages that contain "
        "imsmanifest.xml are supported."
    )


def extract_file(
    path: str | Path,
    *,
    describe_images: bool = True,
    generate_srt: bool = False,
) -> Any:
    """Unified extractor that routes by file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in _KG_EXTENSIONS:
        return KnowledgeGraphExtractor().extract(path)
    if suffix in _GEO_EXTENSIONS:
        return GeoJSONExtractor().extract(path)
    if suffix in _EMAIL_EXTENSIONS:
        return EmailExtractor().extract(path)
    if suffix in _LOG_EXTENSIONS:
        return LogExtractor().extract(path)
    if suffix == ".jsonl":
        return _extract_jsonl(path)
    if suffix == ".json":
        return _extract_json(path)
    if suffix == ".txt":
        from memotrix.filetypes.document.txt import TXTExtractor

        raw = TXTExtractor().read_text(path)
        if looks_like_whatsapp(raw):
            return ChatExtractor().extract(path, raw_text=raw)
        return DocumentExtractor.extract(path, describe_images=describe_images)
    if suffix == ".xls":
        raise UnsupportedDocumentTypeError(
            "Unsupported file type: .xls. Convert to .xlsx "
            "(openpyxl cannot read BIFF .xls files)."
        )
    if suffix in DocumentExtractor.EXTRACTORS:
        return DocumentExtractor.extract(path, describe_images=describe_images)
    if suffix in StructuredDataExtractor.EXTRACTORS:
        return StructuredDataExtractor.extract(path)
    if suffix in ImageExtractor.SUPPORTED_EXTENSIONS:
        return ImageExtractor.extract(path, describe_images=describe_images)
    if suffix in VideoExtractor.supported_extensions:
        return VideoExtractor(generate_srt=generate_srt).extract(path)
    if suffix == ".scorm":
        return ScormExtractor().extract(path)
    if suffix == ".zip":
        return _extract_zip(path)
    if suffix in ProgrammingFileExtractor.supported_extensions:
        return ProgrammingFileExtractor().extract(path)
    if suffix in _AUDIO_EXTENSIONS:
        return AudioExtractor(generate_srt=generate_srt).extract(path)

    supported_ext = ", ".join(supported_extensions())
    raise UnsupportedDocumentTypeError(
        f"Unsupported file type: {suffix}. Supported: {supported_ext}"
    )
