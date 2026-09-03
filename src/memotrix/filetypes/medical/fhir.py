"""FHIR JSON as short clinical narratives (not DICOM / imaging)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.utils.exceptions import EmptyDocumentError
from memotrix.utils.outputSturcture import build_document


def _text_from_codeable(concept: Any) -> str:
    if isinstance(concept, str):
        return concept
    if not isinstance(concept, dict):
        return ""
    if concept.get("text"):
        return str(concept["text"])
    coding = concept.get("coding") or []
    if coding and isinstance(coding[0], dict):
        return str(coding[0].get("display") or coding[0].get("code") or "")
    return ""


def _human_name(resource: Dict[str, Any]) -> str:
    names = resource.get("name") or []
    if isinstance(names, dict):
        names = [names]
    if not names:
        return ""
    first = names[0] if isinstance(names[0], dict) else {}
    if first.get("text"):
        return str(first["text"])
    given = " ".join(first.get("given") or [])
    family = first.get("family") or ""
    return " ".join(part for part in (given, family) if part).strip()


def _resource_id(resource: Dict[str, Any]) -> str:
    return str(resource.get("id") or resource.get("resourceType") or "resource")


def narrate_resource(resource: Dict[str, Any]) -> str:
    rtype = str(resource.get("resourceType") or "Resource")
    rid = _resource_id(resource)
    if rtype == "Patient":
        name = _human_name(resource) or rid
        gender = resource.get("gender") or ""
        birth = resource.get("birthDate") or ""
        extras = ", ".join(part for part in (gender, birth) if part)
        return f"Patient {name} (id={rid})" + (f", {extras}." if extras else ".")
    if rtype == "Observation":
        code = _text_from_codeable(resource.get("code")) or "observation"
        value = resource.get("valueString")
        if value is None and isinstance(resource.get("valueQuantity"), dict):
            qty = resource["valueQuantity"]
            value = f"{qty.get('value', '')} {qty.get('unit') or qty.get('code') or ''}".strip()
        when = resource.get("effectiveDateTime") or resource.get("issued") or ""
        return f"Observation {code}={value or '?'} for {rid}" + (f" at {when}." if when else ".")
    if rtype == "Condition":
        code = _text_from_codeable(resource.get("code")) or "condition"
        status = ""
        if isinstance(resource.get("clinicalStatus"), dict):
            status = _text_from_codeable(resource["clinicalStatus"])
        return f"Condition {code} (id={rid})" + (f", status={status}." if status else ".")
    if rtype == "MedicationRequest":
        med = resource.get("medicationCodeableConcept") or resource.get("medicationReference") or {}
        label = _text_from_codeable(med) if isinstance(med, dict) else str(med)
        if isinstance(med, dict) and med.get("display"):
            label = str(med["display"])
        status = resource.get("status") or ""
        return f"MedicationRequest {label or rid}" + (f", status={status}." if status else ".")
    if rtype == "Encounter":
        status = resource.get("status") or ""
        klass = resource.get("class") or {}
        klass_text = klass.get("display") or klass.get("code") if isinstance(klass, dict) else ""
        period = resource.get("period") or {}
        start = period.get("start") if isinstance(period, dict) else ""
        return (
            f"Encounter {rid}"
            + (f" class={klass_text}" if klass_text else "")
            + (f" status={status}" if status else "")
            + (f" start={start}" if start else "")
            + "."
        )
    if rtype == "DiagnosticReport":
        code = _text_from_codeable(resource.get("code")) or "report"
        status = resource.get("status") or ""
        conclusion = resource.get("conclusion") or ""
        return f"DiagnosticReport {code} (id={rid})" + (
            f", {status} {conclusion}.".strip() if status or conclusion else "."
        )
    bits: List[str] = [f"{rtype} id={rid}"]
    for key, value in resource.items():
        if key in {"resourceType", "id", "meta", "text"}:
            continue
        if isinstance(value, str) and value.strip() and len(bits) < 6:
            bits.append(f"{key}={value.strip()[:80]}")
    return ", ".join(bits) + "."


def collect_resources(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if data.get("resourceType") == "Bundle":
        resources: List[Dict[str, Any]] = []
        for entry in data.get("entry") or []:
            if isinstance(entry, dict) and isinstance(entry.get("resource"), dict):
                resources.append(entry["resource"])
        return resources
    return [data]


class FHIRExtractor(BaseExtractor):
    supported_extensions = (".json",)

    def extract(self, path: Path, data: Any = None):
        path = Path(path)
        if data is None:
            with path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        if not isinstance(data, dict):
            raise EmptyDocumentError(f"{path.name} is not a FHIR JSON object")
        resources = collect_resources(data)
        lines = ["# Clinical records"]
        for resource in resources:
            if isinstance(resource, dict):
                lines.append(narrate_resource(resource))
        text = "\n\n".join(lines)
        if len(lines) <= 1:
            raise EmptyDocumentError(f"No FHIR resources in {path.name}")
        return build_document(
            path,
            text,
            extra_metadata={
                "file_type": "fhir",
                "kind": "medical",
                "resource_count": len(resources),
            },
        )
