from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentData:
    metadata: Dict[str, Any]
    sections: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    raw_text: Optional[str] = None
    language: Optional[str] = None
    summary: Optional[str] = None
    transcripts: List[Dict[str, Any]] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)