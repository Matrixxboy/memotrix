"""
Extract embedded / page images from documents, store them on disk,
and produce searchable vision captions (OCR + visual description).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# Pages with less than this many chars of extractable text are treated as image-only.
IMAGE_ONLY_PAGE_CHAR_THRESHOLD = 40


def media_dir_for(source: Path) -> Path:
    """Stable per-document folder for extracted image assets."""
    from memotrix.utils.trace import mlog

    folder = source.parent / ".memotrix_media" / source.stem
    folder.mkdir(parents=True, exist_ok=True)
    mlog("media", f"media folder ready: {folder}")
    return folder


def save_image_bytes(
    data: bytes,
    dest_dir: Path,
    *,
    stem: str,
    ext: str = ".png",  # ignored — always saved as PNG
) -> Path:
    """
    Persist extracted image bytes as PNG only.

    JPEG/WebP/etc. payloads are decoded and re-encoded so every file under
    ``.memotrix_media`` ends with ``.png``.
    """
    from memotrix.utils.trace import mlog

    dest_dir.mkdir(parents=True, exist_ok=True)
    png_bytes = _to_png_bytes(data)
    digest = hashlib.sha256(png_bytes).hexdigest()[:10]
    path = dest_dir / f"{stem}_{digest}.png"
    created = not path.exists()
    if created:
        path.write_bytes(png_bytes)
    mlog(
        "media",
        f"{'saved' if created else 'exists'} {path.name} "
        f"({len(png_bytes):,} bytes png"
        + (f", from {len(data):,} raw)" if len(data) != len(png_bytes) else ")"),
    )
    return path


def _to_png_bytes(data: bytes) -> bytes:
    """Decode any common image format and return PNG bytes."""
    from io import BytesIO

    # Already PNG magic — keep as-is (avoids re-encode quality/size churn).
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to save extracted images as PNG") from exc

    with Image.open(BytesIO(data)) as img:
        # Preserve alpha when present; otherwise RGB.
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            converted = img.convert("RGBA")
        else:
            converted = img.convert("RGB")
        out = BytesIO()
        converted.save(out, format="PNG", optimize=True)
        return out.getvalue()


def _caption_sidecar_path(image_path: Path) -> Path:
    return Path(image_path).with_suffix(Path(image_path).suffix + ".caption.json")


def load_caption_sidecar(image_path: Path) -> Optional[Dict[str, Any]]:
    path = _caption_sidecar_path(image_path)
    if not path.is_file():
        return None
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and (data.get("description") or "").strip():
            return data
    except Exception:  # noqa: BLE001
        return None
    return None


def save_caption_sidecar(image_path: Path, payload: Dict[str, Any]) -> None:
    path = _caption_sidecar_path(image_path)
    try:
        import json

        path.write_text(
            json.dumps(
                {
                    "description": payload.get("description") or "",
                    "caption": payload.get("caption") or "",
                    "visual_kind": payload.get("visual_kind") or "other",
                    "is_decorative": bool(payload.get("is_decorative")),
                    "tags": list(payload.get("tags") or []),
                    "context": payload.get("context") or "",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


def describe_image_asset(
    image_path: Path,
    *,
    context: str = "",
    skip_ai: bool = False,
) -> Dict[str, Any]:
    """
    Run vision captioning and return a DocumentData.images entry.

    Description prioritizes readable text in the image (OCR-style),
    then visual context — so scanned pages and screenshots stay searchable.
    Also stores visual_kind / caption for precise image retrieval.
    Captions are persisted beside the image so memory rehydrate can reuse them.
    """
    from memotrix.utils.ai_integration import (
        describe_image,
        extract_image_metadata,
        guess_visual_kind_from_metadata,
        parse_image_classification,
    )
    from memotrix.utils.trace import mlog

    image_path = Path(image_path)
    mlog(
        "media",
        f"describe {image_path.name}"
        + (f" context={context!r}" if context else "")
        + (" [skip_ai]" if skip_ai else ""),
    )
    metadata = extract_image_metadata(str(image_path))
    tags = [
        str(metadata.get("format", "image")).lower(),
        f"{metadata.get('width', '?')}x{metadata.get('height', '?')}",
    ]
    if context:
        tags.append(context.replace(" ", "_")[:40])

    cached = load_caption_sidecar(image_path)
    if cached:
        # Prefer a real prior OCR caption over a skip_ai placeholder.
        description = str(cached.get("description") or "").strip()
        if description and "Vision AI not available" not in description:
            visual_kind = str(cached.get("visual_kind") or "other")
            caption = str(cached.get("caption") or image_path.name)
            is_decorative = bool(cached.get("is_decorative"))
            for extra in (visual_kind, "decorative" if is_decorative else ""):
                if extra and extra not in tags:
                    tags.append(extra)
            mlog("media", f"loaded caption sidecar for {image_path.name}")
            return {
                "path": str(image_path.resolve()),
                "description": description,
                "tags": tags,
                "context": context or str(cached.get("context") or ""),
                "metadata": metadata,
                "visual_kind": visual_kind,
                "caption": caption,
                "is_decorative": is_decorative,
            }

    if skip_ai:
        visual_kind = guess_visual_kind_from_metadata(
            metadata, filename=image_path.name
        )
        caption = f"{visual_kind} — {image_path.name}"
        description = (
            f"Image asset ({image_path.name})"
            + (f" — {context}" if context else "")
            + ". Vision AI not available; file stored for later description.\n"
            f"visual_kind: {visual_kind}\n"
            f"caption: {caption}\n"
            "is_decorative: no"
        )
        if visual_kind not in tags:
            tags.append(visual_kind)
        return {
            "path": str(image_path.resolve()),
            "description": description,
            "tags": tags,
            "context": context,
            "metadata": metadata,
            "visual_kind": visual_kind,
            "caption": caption,
            "is_decorative": False,
        }

    visual_kind = "other"
    caption = image_path.name
    is_decorative = False
    try:
        result = describe_image(
            str(image_path),
            image_metadata=metadata,
            context=context,
        )
        description = result.get("description") or ""
        if isinstance(description, dict):
            description = description.get("description", str(description))
        description = str(description).strip()
        visual_kind = str(result.get("visual_kind") or "other")
        caption = str(result.get("caption") or caption).strip()
        is_decorative = bool(result.get("is_decorative"))
        # Recover fields if the model put them only in free text.
        if visual_kind == "other" or not caption:
            parsed = parse_image_classification(description)
            if visual_kind == "other":
                visual_kind = parsed["visual_kind"]
            if not caption:
                caption = parsed["caption"] or caption
            is_decorative = is_decorative or parsed["is_decorative"]
    except Exception as exc:  # noqa: BLE001 — keep ingestion resilient
        visual_kind = guess_visual_kind_from_metadata(
            metadata, filename=image_path.name
        )
        caption = f"{visual_kind} — {image_path.name}"
        description = (
            f"Image asset ({image_path.name})"
            + (f" — {context}." if context else ".")
            + f" Description failed: {exc}\n"
            f"visual_kind: {visual_kind}\n"
            f"caption: {caption}\n"
            "is_decorative: no"
        )

    if context and context.lower() not in description.lower():
        description = f"[{context}] {description}"

    for extra in (visual_kind, "decorative" if is_decorative else ""):
        if extra and extra not in tags:
            tags.append(extra)

    payload = {
        "path": str(image_path.resolve()),
        "description": description,
        "tags": tags,
        "context": context,
        "metadata": metadata,
        "visual_kind": visual_kind,
        "caption": caption,
        "is_decorative": is_decorative,
    }
    if description and "Description failed:" not in description:
        save_caption_sidecar(image_path, payload)
    return payload



def image_descriptions_as_text(images: List[Dict[str, Any]]) -> str:
    """Flatten image captions into section text for image-only / sparse-text docs."""
    parts: List[str] = []
    for i, image in enumerate(images, start=1):
        desc = (image.get("description") or "").strip()
        ctx = image.get("context") or f"Image {i}"
        if desc:
            parts.append(f"## {ctx}\n\n{desc}")
    return "\n\n".join(parts)
