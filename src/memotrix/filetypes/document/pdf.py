from pathlib import Path
from typing import Any, Dict, List

from .base import BaseExtractor
from .media import (
    IMAGE_ONLY_PAGE_CHAR_THRESHOLD,
    describe_image_asset,
    media_dir_for,
    save_image_bytes,
)
from memotrix.utils.outputSturcture import build_document


class PDFExtractor(BaseExtractor):
    supported_extensions = (".pdf",)

    def extract(self, path: Path, *, describe_images: bool = True):
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover
            from memotrix.utils.exceptions import MissingDependencyError
            raise MissingDependencyError("PyMuPDF (fitz) is required for PDF extraction. Run `pip install memotrix[pdf]`.") from exc

        from memotrix.utils.trace import mlog

        path = Path(path)
        mlog("pdf", f"extract start: {path.name} describe_images={describe_images}")
        doc = fitz.open(str(path))
        out_dir = media_dir_for(path)
        page_blocks: List[str] = []
        images: List[Dict[str, Any]] = []
        seen_xrefs: set[int] = set()
        image_only_pages = 0
        skip_ai = not describe_images

        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            page_text = (page.get_text() or "").strip()
            if page_text:
                page_blocks.append(f"## Page {page_no}\n\n{page_text}")

            is_image_only = len(page_text) < IMAGE_ONLY_PAGE_CHAR_THRESHOLD

            if is_image_only:
                # Scanned page: one full-page render → OCR + visual caption
                image_only_pages += 1
                mlog("pdf", f"page {page_no}: image-only → full-page render")
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                render_path = out_dir / f"page{page_no}_render.png"
                pix.save(str(render_path))
                asset = describe_image_asset(
                    render_path,
                    context=f"PDF page {page_no} scanned/image page",
                    skip_ai=skip_ai,
                )
                images.append(asset)
                desc = asset.get("description") or ""
                if desc:
                    page_blocks.append(f"## Page {page_no} (image OCR)\n\n{desc}")
                continue

            # Mixed text+images: extract figures between the text
            for img_info in page.get_images(full=True):
                xref = int(img_info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted = doc.extract_image(xref)
                except Exception as e:
                    mlog("pdf", f"Warning: Failed to extract image xref {xref} on page {page_no}: {e}")
                    continue
                img_bytes = extracted.get("image")
                if not img_bytes:
                    continue
                width = int(extracted.get("width") or 0)
                height = int(extracted.get("height") or 0)
                if width and height and (width * height) < 8_000:
                    continue
                img_path = save_image_bytes(
                    img_bytes,
                    out_dir,
                    stem=f"page{page_no}_img{xref}",
                )
                images.append(
                    describe_image_asset(
                        img_path,
                        context=f"PDF page {page_no} embedded image",
                        skip_ai=skip_ai,
                    )
                )

        text = "\n\n".join(page_blocks)
        mlog(
            "pdf",
            f"extract done: {path.name} pages={doc.page_count} "
            f"images={len(images)} image_only_pages={image_only_pages} "
            f"text_chars={len(text)}",
        )
        return build_document(
            path,
            text,
            images=images,
            extra_metadata={
                "file_type": "pdf",
                "library": "PyMuPDF",
                "page_count": doc.page_count,
                "embedded_image_count": len(seen_xrefs),
                "image_only_pages": image_only_pages,
                "extracted_image_count": len(images),
            },
        )
