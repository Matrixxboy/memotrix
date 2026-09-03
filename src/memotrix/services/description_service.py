from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.ai_integration import describe_image, get_ai_router


class DescriptionService:
    """Service for generating descriptions using the configured AI backend."""

    def __init__(self, ai_router=None) -> None:
        self.ai_router = ai_router or get_ai_router()

    def describe_image(
        self, image_path: str | Path, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Return the full searchable caption (includes classification block)."""
        result = describe_image(str(image_path), image_metadata=metadata)
        return str(result.get("description") or "")

    def describe_image_structured(
        self, image_path: str | Path, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return description plus visual_kind / caption / is_decorative."""
        return describe_image(str(image_path), image_metadata=metadata)

    def describe_text(self, text: str, context: Optional[str] = None) -> str:
        prompt = "You are generating a concise description for stored content."
        if context:
            prompt += f" Context: {context}"
        prompt += f"\nContent:\n{text}"
        return self.ai_router.generate_text(prompt)

    def describe_payload(self, payload: Dict[str, Any], context: Optional[str] = None) -> str:
        prompt = "You are generating a concise description for structured data."
        if context:
            prompt += f" Context: {context}"
        prompt += f"\nPayload:\n{payload}"
        return self.ai_router.generate_text(prompt)
