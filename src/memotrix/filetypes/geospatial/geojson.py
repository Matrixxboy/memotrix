"""GeoJSON features as searchable place descriptions (no spatial index)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.utils.exceptions import EmptyDocumentError
from memotrix.utils.outputSturcture import build_document


def _first_coordinate(coords: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(coords, (list, tuple)) or not coords:
        return None
    head = coords[0]
    if isinstance(head, (int, float)) and len(coords) >= 2:
        return float(coords[0]), float(coords[1])
    return _first_coordinate(head)


def _geometry_text(geometry: Any) -> str:
    if not isinstance(geometry, dict):
        return "unknown geometry"
    gtype = str(geometry.get("type") or "Geometry")
    coords = geometry.get("coordinates")
    if gtype == "Point":
        pair = _first_coordinate(coords)
        if pair:
            return f"Point at {pair[0]}, {pair[1]}"
    pair = _first_coordinate(coords)
    if pair:
        return f"{gtype} near {pair[0]}, {pair[1]}"
    return gtype


def _properties_text(properties: Any) -> str:
    if not isinstance(properties, dict) or not properties:
        return ""
    parts: List[str] = []
    for key, value in list(properties.items())[:12]:
        if value is None or value == "":
            continue
        if isinstance(value, (dict, list)):
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def _feature_name(feature: Dict[str, Any], index: int) -> str:
    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    for key in ("name", "title", "label", "id", "NAME"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fid = feature.get("id")
    if fid not in (None, ""):
        return str(fid)
    return f"feature {index + 1}"


def features_to_text(data: Dict[str, Any]) -> Tuple[str, int]:
    if data.get("type") == "Feature":
        features = [data]
    else:
        features = list(data.get("features") or [])
    lines: List[str] = ["# Places"]
    count = 0
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        name = _feature_name(feature, index)
        geom = _geometry_text(feature.get("geometry"))
        props = _properties_text(feature.get("properties"))
        sentence = f'Feature "{name}" is a {geom}.'
        if props:
            sentence += f" Properties: {props}."
        lines.append(sentence)
        count += 1
    return "\n\n".join(lines), count


class GeoJSONExtractor(BaseExtractor):
    supported_extensions = (".geojson",)

    def extract(self, path: Path, data: Any = None):
        path = Path(path)
        if data is None:
            with path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        if not isinstance(data, dict):
            raise EmptyDocumentError(f"{path.name} is not a GeoJSON object")
        text, count = features_to_text(data)
        if count == 0:
            raise EmptyDocumentError(f"No GeoJSON features in {path.name}")
        return build_document(
            path,
            text,
            extra_metadata={
                "file_type": "geojson",
                "kind": "geospatial",
                "feature_count": count,
            },
        )
