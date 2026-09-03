"""Knowledge graph extractors — triples as searchable English sentences."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple
from xml.etree import ElementTree as ET

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.utils.exceptions import ConfigurationError, DocumentExtractionError, EmptyDocumentError
from memotrix.utils.outputSturcture import build_document

TRIPLE_BATCH = 25
_NT_LINE = re.compile(r"^(\S+)\s+(\S+)\s+(.+?)\s*\.\s*$")

_RDF_FORMATS = {
    ".ttl": "turtle",
    ".nt": "nt",
    ".nq": "nquads",
    ".rdf": "xml",
    ".owl": "xml",
    ".trig": "trig",
    ".jsonld": "json-ld",
}


def _local_name(term: object) -> str:
    text = str(term).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    if text.startswith('"'):
        end = text.rfind('"')
        if end > 0:
            text = text[1:end]
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    if ":" in text and not text.startswith("http"):
        return text.rsplit(":", 1)[-1]
    return text


def _sentences(triples: Sequence[Tuple[object, object, object]]) -> List[str]:
    lines: List[str] = []
    for subject, predicate, obj in triples:
        lines.append(f"{_local_name(subject)} {_local_name(predicate)} {_local_name(obj)}.")
    return lines


def _batched_markdown(sentences: Sequence[str]) -> str:
    if not sentences:
        return ""
    parts: List[str] = []
    for start in range(0, len(sentences), TRIPLE_BATCH):
        batch = sentences[start : start + TRIPLE_BATCH]
        heading = f"# Graph facts {start // TRIPLE_BATCH + 1}"
        parts.append(heading + "\n\n" + "\n".join(batch))
    return "\n\n".join(parts)


def _parse_ntriples(text: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _NT_LINE.match(line)
        if not match:
            continue
        triples.append((match.group(1), match.group(2), match.group(3)))
    return triples


def _parse_graphml(path: Path) -> List[Tuple[str, str, str]]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    labels: dict[str, str] = {}
    for node in root.iter(f"{ns}node"):
        node_id = node.get("id") or ""
        label = node_id
        for data in node.findall(f"{ns}data"):
            key = (data.get("key") or "").lower()
            if key in {"label", "name", "d0", "d1"} and (data.text or "").strip():
                label = data.text.strip()
                break
        labels[node_id] = label

    triples: List[Tuple[str, str, str]] = []
    for edge in root.iter(f"{ns}edge"):
        source = labels.get(edge.get("source") or "", edge.get("source") or "source")
        target = labels.get(edge.get("target") or "", edge.get("target") or "target")
        rel = "related_to"
        for data in edge.findall(f"{ns}data"):
            key = (data.get("key") or "").lower()
            if key in {"label", "name", "relation", "d0", "d1"} and (data.text or "").strip():
                rel = data.text.strip()
                break
        if edge.get("label"):
            rel = edge.get("label") or rel
        triples.append((source, rel, target))
    return triples


def _rdflib_triples(path: Path, rdf_format: str) -> List[Tuple[object, object, object]]:
    try:
        from rdflib import Graph
    except ImportError as exc:
        raise ConfigurationError(
            "Knowledge-graph formats other than N-Triples and GraphML require rdflib. "
            "Install with: pip install memotrix[extractors]"
        ) from exc
    graph = Graph()
    try:
        graph.parse(path, format=rdf_format)
    except Exception as exc:  # noqa: BLE001
        raise DocumentExtractionError(f"Failed to parse knowledge graph {path.name}: {exc}") from exc
    return list(graph)


class KnowledgeGraphExtractor(BaseExtractor):
    supported_extensions = (
        ".ttl",
        ".nt",
        ".nq",
        ".rdf",
        ".owl",
        ".trig",
        ".jsonld",
        ".graphml",
    )

    def extract(self, path: Path, data: object | None = None):
        path = Path(path)
        suffix = path.suffix.lower()
        triples: List[Tuple[object, object, object]]

        if suffix == ".graphml":
            triples = list(_parse_graphml(path))
        elif suffix == ".nt":
            try:
                triples = list(_rdflib_triples(path, "nt"))
            except ConfigurationError:
                triples = list(_parse_ntriples(self.read_text(path)))
        elif suffix in _RDF_FORMATS:
            triples = list(_rdflib_triples(path, _RDF_FORMATS[suffix]))
        elif data is not None:
            triples = list(_rdflib_triples_from_jsonld(data))
        else:
            raise DocumentExtractionError(f"Unsupported knowledge graph type: {suffix}")

        sentences = _sentences(triples)
        text = _batched_markdown(sentences)
        if not text.strip():
            raise EmptyDocumentError(f"No triples found in {path}")
        return build_document(
            path,
            text,
            extra_metadata={
                "file_type": "knowledge_graph",
                "kind": "graph",
                "triple_count": len(sentences),
            },
        )


def _rdflib_triples_from_jsonld(data: object) -> List[Tuple[object, object, object]]:
    try:
        from rdflib import Graph
    except ImportError as exc:
        raise ConfigurationError(
            "JSON-LD knowledge graphs require rdflib. "
            "Install with: pip install memotrix[extractors]"
        ) from exc
    import json

    graph = Graph()
    graph.parse(data=json.dumps(data), format="json-ld")
    return list(graph)
