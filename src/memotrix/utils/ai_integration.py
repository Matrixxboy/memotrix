import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


from pathlib import Path
from typing import Dict, Any

import hashlib
import mimetypes
import time


import shutil
import tempfile
import uuid
from datetime import datetime



class AIModelProvider(ABC):
    """Base interface for AI providers used by the pipeline."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    def generate_vision(self, prompt: str, image_path: str, **kwargs: Any) -> str:
        """Override to support vision models."""
        combined_prompt = f"{prompt}\n\n[Image Path: {image_path}]"
        return self.generate_text(combined_prompt, **kwargs)


class LocalFallbackProvider(AIModelProvider):
    """Simple fallback provider used when no external AI model is configured."""

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        return (
            "AI model not configured. "
            f"Prompt received: {prompt[:200]}"
        )


class OpenAIProvider(AIModelProvider):
    """OpenAI chat + vision provider (chat.completions API)."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL

    def _client(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for OpenAIProvider") from exc
        return OpenAI(api_key=self.api_key)

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        system = kwargs.pop("system", None)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    def generate_vision(self, prompt: str, image_path: str, **kwargs: Any) -> str:
        import base64
        import mimetypes

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/jpeg"

        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()


class OpenRouterProvider(AIModelProvider):
    """OpenRouter-compatible provider (OpenAI SDK + OpenRouter base URL)."""

    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL") or self.DEFAULT_MODEL
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    def _client(self):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required for OpenRouterProvider "
                "(pip install openai)"
            ) from exc
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        system = kwargs.pop("system", None)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    def generate_vision(self, prompt: str, image_path: str, **kwargs: Any) -> str:
        import base64
        import mimetypes

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/jpeg"

        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()

class GoogleProvider(AIModelProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GOOGLE_MODEL")

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("google package is required for GoogleProvider") from exc

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            **kwargs,
        )
        return response.text

    def generate_vision(self, prompt: str, image_path: str, **kwargs: Any) -> str:
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        try:
            from google import genai
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("google and Pillow packages are required for GoogleProvider vision") from exc

        client = genai.Client(api_key=self.api_key)
        
        with Image.open(image_path) as img:
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, img],
                **kwargs,
            )
        return response.text

class AIModelRouter:
    """Routes prompts to the configured provider without changing the pipeline."""

    def __init__(self, provider: Optional[AIModelProvider] = None) -> None:
        self.provider = provider or self._build_default_provider()

    @staticmethod
    def _build_default_provider() -> AIModelProvider:
        # Prefer OpenAI for chat/vision; OpenRouter disabled for now.
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        if os.getenv("GOOGLE_API_KEY"):
            return GoogleProvider()
        return LocalFallbackProvider()

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        from memotrix.utils.trace import mlog, mlog_block

        system = kwargs.get("system")
        provider_name = type(self.provider).__name__
        mlog(
            "ai_text",
            f"→ {provider_name} | prompt={len(prompt or '')} chars"
            + (f" | system={len(system)} chars" if system else ""),
        )
        if system:
            mlog_block("ai_text", "SYSTEM PROMPT", str(system))
        mlog_block("ai_text", "USER / FINAL PROMPT", prompt or "")
        t0 = time.perf_counter()
        try:
            out = self.provider.generate_text(prompt, **kwargs)
        except Exception as exc:  # noqa: BLE001
            mlog("ai_text", f"✗ failed after {time.perf_counter() - t0:.2f}s: {exc}")
            raise
        mlog(
            "ai_text",
            f"← response {len(out or '')} chars in {time.perf_counter() - t0:.2f}s",
        )
        mlog_block("ai_text", "MODEL RESPONSE", out or "")
        return out

    def generate_vision(self, prompt: str, image_path: str, **kwargs: Any) -> str:
        from memotrix.utils.trace import mlog, mlog_block
        from pathlib import Path

        provider_name = type(self.provider).__name__
        name = Path(image_path).name if image_path else "?"
        mlog(
            "ai_vision",
            f"→ {provider_name} | image={name} | prompt={len(prompt or '')} chars",
        )
        mlog_block("ai_vision", f"VISION PROMPT ({name})", prompt or "")
        t0 = time.perf_counter()
        try:
            out = self.provider.generate_vision(prompt, image_path, **kwargs)
        except Exception as exc:  # noqa: BLE001
            mlog("ai_vision", f"✗ failed after {time.perf_counter() - t0:.2f}s: {exc}")
            raise
        mlog(
            "ai_vision",
            f"← response {len(out or '')} chars in {time.perf_counter() - t0:.2f}s",
        )
        mlog_block("ai_vision", f"VISION RESPONSE ({name})", out or "")
        return out


_router: Optional[AIModelRouter] = None


def get_ai_router() -> AIModelRouter:
    global _router
    if _router is None:
        _router = AIModelRouter()
    return _router


def configure_ai_provider(provider: AIModelProvider) -> None:
    global _router
    _router = AIModelRouter(provider=provider)




def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive intrinsic metadata from an image.
    """

    path = Path(image_path)

    metadata: Dict[str, Any] = {}

    # ---------------------------------------------------
    # File Metadata
    # ---------------------------------------------------

    stat = path.stat()

    metadata["filename"] = path.name
    metadata["stem"] = path.stem
    metadata["extension"] = path.suffix.lower()
    metadata["mime_type"] = mimetypes.guess_type(path)[0]
    metadata["absolute_path"] = str(path.resolve())

    metadata["size_bytes"] = stat.st_size

    metadata["created"] = stat.st_ctime
    metadata["modified"] = stat.st_mtime
    metadata["accessed"] = stat.st_atime

    # SHA256
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    metadata["sha256"] = sha.hexdigest()

    # ---------------------------------------------------
    # Image Properties
    # ---------------------------------------------------

    try:
        from PIL import Image, ExifTags, ImageStat
    except ImportError as exc:
        from memotrix.utils.exceptions import MissingDependencyError

        raise MissingDependencyError(
            "Pillow is required for image metadata. Install with: pip install memotrix[images]"
        ) from exc

    with Image.open(path) as img:

        metadata["format"] = img.format
        metadata["mode"] = img.mode

        metadata["width"] = img.width
        metadata["height"] = img.height

        metadata["aspect_ratio"] = round(
            img.width / img.height,
            4,
        )

        metadata["orientation"] = (
            "landscape"
            if img.width > img.height
            else "portrait"
            if img.height > img.width
            else "square"
        )

        metadata["megapixels"] = round(
            img.width * img.height / 1_000_000,
            2,
        )

        metadata["bands"] = img.getbands()

        metadata["has_alpha"] = "A" in img.getbands()

        metadata["is_animated"] = getattr(img, "is_animated", False)

        metadata["frame_count"] = getattr(img, "n_frames", 1)

        metadata["palette"] = img.palette is not None

        metadata["dpi"] = img.info.get("dpi")

        metadata["compression"] = img.info.get("compression")

        metadata["icc_profile"] = "icc_profile" in img.info

        metadata["transparency"] = img.info.get("transparency")

        metadata["background"] = img.info.get("background")

        metadata["loop"] = img.info.get("loop")

        metadata["duration"] = img.info.get("duration")

        # ---------------------------------------------
        # Color statistics
        # ---------------------------------------------

        stat = ImageStat.Stat(img.convert("RGB"))

        metadata["mean_rgb"] = stat.mean
        metadata["stddev_rgb"] = stat.stddev

        # ---------------------------------------------
        # EXIF
        # ---------------------------------------------

        exif = {}

        try:
            raw = img.getexif()

            for tag, value in raw.items():
                exif[ExifTags.TAGS.get(tag, tag)] = value

        except Exception:
            pass

        metadata["exif"] = exif

        # GPS
        metadata["gps"] = exif.get("GPSInfo")

        metadata["camera_make"] = exif.get("Make")
        metadata["camera_model"] = exif.get("Model")
        metadata["lens"] = exif.get("LensModel")
        metadata["software"] = exif.get("Software")

        metadata["datetime"] = exif.get("DateTime")
        metadata["datetime_original"] = exif.get("DateTimeOriginal")

        metadata["iso"] = exif.get("ISOSpeedRatings")

        metadata["f_number"] = exif.get("FNumber")

        metadata["focal_length"] = exif.get("FocalLength")

        metadata["exposure_time"] = exif.get("ExposureTime")

        metadata["flash"] = exif.get("Flash")

        metadata["white_balance"] = exif.get("WhiteBalance")

        metadata["metering_mode"] = exif.get("MeteringMode")

        metadata["exposure_mode"] = exif.get("ExposureMode")

        metadata["color_space"] = exif.get("ColorSpace")

    return metadata


def describe_image(
    image_path: str,
    image_metadata: Optional[Dict[str, Any]] = None,
    *,
    context: str = "",
) -> Dict[str, Any]:
    """
    Analyze an image with a Vision AI model.

    Captioning goals (in order):
    1. Transcribe ALL readable text in the image (OCR-style, preserve layout cues).
    2. Describe charts/diagrams/UI structure.
    3. Summarize the visual scene for retrieval.
    4. Classify visual_kind + short caption for precise image retrieval.
    """

    metadata = image_metadata or extract_image_metadata(image_path)

    source = Path(image_path)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    unique_name = (
        f"{source.stem}"
        f"_{source.suffix.replace('.', '').lower()}"
        f"_{timestamp}"
        f"_{uuid.uuid4().hex}"
        f"{source.suffix.lower()}"
    )

    temp_dir = Path(tempfile.gettempdir()) / "memotrix_images"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_image = temp_dir / unique_name

    shutil.copy2(source, temp_image)

    try:
        context_line = f"\nDocument context: {context}\n" if context else "\n"
        prompt = f"""You are an expert document OCR + image captioning model for a retrieval system.
{context_line}
Produce a searchable description of this image with these sections:

1) TEXT IN IMAGE (OCR):
- Transcribe ALL visible text exactly (headings, body, labels, captions, UI strings, stamps, brand names).
- Preserve reading order. Use line breaks for layout. If no text, write "None".

2) STRUCTURE:
- Note tables, charts, diagrams, forms, screenshots, stamps, photos, logos, or icons.
- For charts: axes, legends, key values. For diagrams: main nodes/relationships.
- For logos/icons: brand/org name if readable, shape/colors, and whether it is header/footer chrome.

3) VISUAL SUMMARY:
- 2–4 sentences on what the image shows overall, for semantic search.

4) CLASSIFICATION:
- visual_kind: exactly one of logo | icon | photo | diagram | chart | screenshot | page_render | stamp | other
- caption: one short line (max 12 words) naming the primary subject for search
- is_decorative: yes or no (yes = repeated header/footer chrome, spacer, or tiny icon; no = main content)

Keep the whole response under ~280 words. Prefer exact text over vague paraphrases.
Use this exact CLASSIFICATION format:
visual_kind: <kind>
caption: <short caption>
is_decorative: <yes|no>
"""
        description = get_ai_router().generate_vision(
            prompt=prompt,
            image_path=str(temp_image),
        )

    finally:
        if temp_image.exists():
            temp_image.unlink(missing_ok=True)

    parsed = parse_image_classification(description or "")
    # Heuristic fallback when the model omits classification.
    if parsed["visual_kind"] == "other":
        guessed = guess_visual_kind_from_metadata(metadata, filename=source.name)
        if guessed != "other":
            parsed["visual_kind"] = guessed
    if not parsed["caption"]:
        parsed["caption"] = (
            f"{parsed['visual_kind']} from {source.name}"
        )

    return {
        "description": description,
        "metadata": metadata,
        "visual_kind": parsed["visual_kind"],
        "caption": parsed["caption"],
        "is_decorative": parsed["is_decorative"],
    }


_VISUAL_KINDS = frozenset(
    {
        "logo",
        "icon",
        "photo",
        "diagram",
        "chart",
        "screenshot",
        "page_render",
        "stamp",
        "other",
    }
)


def parse_image_classification(text: str) -> Dict[str, Any]:
    """Extract visual_kind / caption / is_decorative from a vision caption."""
    raw = text or ""
    kind_m = re.search(
        r"visual_kind\s*:\s*(logo|icon|photo|diagram|chart|screenshot|page_render|stamp|other)\b",
        raw,
        re.I,
    )
    cap_m = re.search(r"caption\s*:\s*(.+)", raw, re.I)
    dec_m = re.search(r"is_decorative\s*:\s*(yes|no)\b", raw, re.I)

    visual_kind = (kind_m.group(1).lower() if kind_m else "other")
    if visual_kind not in _VISUAL_KINDS:
        visual_kind = "other"

    caption = ""
    if cap_m:
        caption = cap_m.group(1).strip().splitlines()[0].strip()
        # Drop trailing structured keys if the model jammed them on one line.
        caption = re.split(r"\bis_decorative\b", caption, maxsplit=1)[0].strip(" -:;|")
        if len(caption) > 120:
            caption = caption[:117].rstrip() + "…"

    is_decorative = False
    if dec_m:
        is_decorative = dec_m.group(1).lower() == "yes"
    elif visual_kind in {"logo", "icon"} and re.search(
        r"\b(header|footer|chrome|watermark|spacer)\b", raw, re.I
    ):
        is_decorative = True

    return {
        "visual_kind": visual_kind,
        "caption": caption,
        "is_decorative": is_decorative,
    }


def guess_visual_kind_from_metadata(
    metadata: Optional[Dict[str, Any]],
    *,
    filename: str = "",
) -> str:
    """Cheap size/name heuristic when vision omits classification."""
    name = (filename or "").lower()
    if re.search(r"\b(logo|brand)\b", name):
        return "logo"
    if "icon" in name or "spacer" in name:
        return "icon"
    if "render" in name:
        return "page_render"

    meta = metadata or {}
    try:
        w = int(meta.get("width") or 0)
        h = int(meta.get("height") or 0)
    except (TypeError, ValueError):
        return "other"
    if w <= 0 or h <= 0:
        return "other"
    area = w * h
    ratio = max(w, h) / max(min(w, h), 1)
    if area < 12_000 or (area < 40_000 and ratio > 2.5):
        return "logo" if ratio > 1.4 else "icon"
    return "other"


def build_image_search_text(
    *,
    description: str,
    visual_kind: str = "other",
    caption: str = "",
    tags: Optional[List[str]] = None,
    is_decorative: bool = False,
) -> str:
    """Compose dense/sparse index text that surfaces kind + caption for retrieval."""
    parts: List[str] = [
        f"visual_kind: {visual_kind or 'other'}",
        f"caption: {caption or 'image'}",
        f"is_decorative: {'yes' if is_decorative else 'no'}",
    ]
    tag_list = [str(t) for t in (tags or []) if t]
    if tag_list:
        parts.append("tags: " + ", ".join(tag_list))
    body = (description or "").strip()
    if body:
        parts.append(body)
    return "\n".join(parts)
