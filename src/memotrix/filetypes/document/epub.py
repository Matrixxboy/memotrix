"""EPUB ebooks as ordered chapter text."""

from __future__ import annotations

import posixpath
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.filetypes.document.html import extract_text_from_html
from memotrix.utils.exceptions import DocumentExtractionError, EmptyDocumentError
from memotrix.utils.outputSturcture import build_document


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _opf_path(zf: zipfile.ZipFile) -> str:
    try:
        container = zf.read("META-INF/container.xml")
    except KeyError as exc:
        raise DocumentExtractionError("EPUB is missing META-INF/container.xml") from exc
    root = ET.fromstring(container)
    for elem in root.iter():
        if _local(elem.tag) == "rootfile":
            href = elem.get("full-path")
            if href:
                return href
    raise DocumentExtractionError("EPUB container.xml has no rootfile")


def _spine_docs(zf: zipfile.ZipFile, opf_path: str) -> list[str]:
    opf = ET.fromstring(zf.read(opf_path))
    base = posixpath.dirname(opf_path)
    manifest: dict[str, str] = {}
    spine: list[str] = []
    for elem in opf.iter():
        name = _local(elem.tag)
        if name == "item" and elem.get("id") and elem.get("href"):
            manifest[elem.get("id") or ""] = elem.get("href") or ""
        elif name == "itemref" and elem.get("idref"):
            spine.append(elem.get("idref") or "")
    docs: list[str] = []
    for idref in spine:
        href = manifest.get(idref)
        if not href:
            continue
        full = posixpath.normpath(posixpath.join(base, href)) if base else href
        docs.append(full)
    return docs


class EPUBExtractor(BaseExtractor):
    supported_extensions = (".epub",)

    def extract(self, path: Path):
        path = Path(path)
        try:
            with zipfile.ZipFile(path) as zf:
                opf_path = _opf_path(zf)
                docs = _spine_docs(zf, opf_path)
                chapters: list[str] = ["# Book"]
                for index, name in enumerate(docs, start=1):
                    try:
                        raw = zf.read(name)
                    except KeyError:
                        continue
                    body = extract_text_from_html(raw)
                    if body.strip():
                        chapters.append(f"# Chapter {index}\n\n{body.strip()}")
                text = "\n\n".join(chapters)
                chapter_count = max(0, len(chapters) - 1)
        except zipfile.BadZipFile as exc:
            raise DocumentExtractionError(f"Invalid EPUB zip: {path.name}") from exc
        if len(text.strip()) <= len("# Book"):
            raise EmptyDocumentError(f"No chapter text in {path.name}")
        return build_document(
            path,
            text,
            extra_metadata={"file_type": "epub", "kind": "text", "chapter_count": chapter_count},
        )
