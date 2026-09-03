from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ...utils.ai_integration import describe_image
from memotrix.utils.outputSturcture import build_document


@dataclass
class ImageExtractionResult:
    path: Path
    metadata: Dict[str, Any]
    text: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    document: Any = None


class ImageExtractor:
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif", ".svg", ".heic"}

    @classmethod
    def extract(cls, path: str | Path, *, describe_images: bool = True):
        """
        Return DocumentData so standalone images flow through the same ingest path
        as PDF/DOCX (text sections + image chunks with OCR-aware captions).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The image file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"The path is not a file: {path}")

        suffix = path.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {suffix}")

        try:
            from PIL import Image as PILImage
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for image extraction") from exc

        if suffix == ".heic":
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError as exc:
                raise RuntimeError("pillow-heif is required to extract .heic files") from exc

        path_to_open = path
        tmp_png_path = None
        if suffix == ".svg":
            try:
                import cairosvg
                import tempfile
                import hashlib
                path_hash = hashlib.md5(str(path.absolute()).encode()).hexdigest()
                tmp_png_path = Path(tempfile.gettempdir()) / f"svg_temp_{path_hash}.png"
                cairosvg.svg2png(url=str(path), write_to=str(tmp_png_path))
                path_to_open = tmp_png_path
            except ImportError as exc:
                raise RuntimeError("cairosvg is required to extract .svg files") from exc

        try:
            with PILImage.open(path_to_open) as image:
                image.load()
                metadata = {
                    "filename": path.name,
                    "extension": suffix,
                    "format": image.format or suffix.upper().lstrip("."),
                    "size": image.size,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "path": str(path.resolve()),
                }

            if describe_images:
                result = describe_image(str(path_to_open), image_metadata=metadata)
            else:
                result = {
                    "description": f"Image file {path.name}",
                    "visual_kind": "other",
                    "caption": path.name,
                    "is_decorative": False,
                    "metadata": metadata,
                }
            description = result.get("description") or ""
            if isinstance(description, dict):
                description = str(description.get("description", description))
            description = str(description).strip()
            visual_kind = str(result.get("visual_kind") or "other")
            caption = str(result.get("caption") or path.name).strip()
            is_decorative = bool(result.get("is_decorative"))

            tags = [
                metadata["format"].lower(),
                f"{metadata['width']}x{metadata['height']}",
                visual_kind,
            ]
            if is_decorative:
                tags.append("decorative")
            images = [
                {
                    "path": str(path.resolve()),
                    "description": description,
                    "tags": tags,
                    "context": f"Standalone image {path.name}",
                    "metadata": result.get("metadata") or metadata,
                    "visual_kind": visual_kind,
                    "caption": caption,
                    "is_decorative": is_decorative,
                }
            ]

            # Caption becomes searchable section text (OCR text included when present).
            body = (
                f"## Image: {path.name}\n\n{description}"
                if description
                else f"## Image: {path.name}\n\nImage file {path.name}."
            )
            return build_document(
                path,
                body,
                images=images,
                extra_metadata={
                    "file_type": "image",
                    "library": "Pillow+Vision",
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "extracted_image_count": 1,
                },
            )
        finally:
            if tmp_png_path and tmp_png_path.exists():
                try:
                    tmp_png_path.unlink()
                except OSError:
                    pass
