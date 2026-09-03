"""Smoke tests for document image extraction helpers (no vision API required)."""

from pathlib import Path

from memotrix.filetypes.document.media import (
    IMAGE_ONLY_PAGE_CHAR_THRESHOLD,
    image_descriptions_as_text,
    media_dir_for,
    save_image_bytes,
)
from memotrix.utils.outputSturcture import build_document


def test_save_image_bytes_is_stable(tmp_path: Path):
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    p1 = save_image_bytes(data, tmp_path, stem="fig")
    p2 = save_image_bytes(data, tmp_path, stem="fig")
    assert p1 == p2
    assert p1.exists()
    assert p1.suffix.lower() == ".png"


def test_save_image_bytes_converts_jpeg_to_png(tmp_path: Path):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()
    assert jpeg_bytes[:2] == b"\xff\xd8"

    path = save_image_bytes(jpeg_bytes, tmp_path, stem="photo", ext=".jpg")
    assert path.suffix.lower() == ".png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_document_from_images_only(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4")
    images = [
        {
            "path": str(tmp_path / "page1.png"),
            "description": "TEXT IN IMAGE: Invoice #42\nTotal due: $19",
            "tags": ["png"],
            "context": "PDF page 1 scanned/image page",
        }
    ]
    doc = build_document(path, "", images=images, extra_metadata={"file_type": "pdf"})
    assert "Invoice #42" in doc.text
    assert doc.validation["has_images"] is True
    assert len(doc.images) == 1


def test_image_descriptions_as_text():
    text = image_descriptions_as_text(
        [{"context": "Slide 2", "description": "Chart of Q3 revenue"}]
    )
    assert "Slide 2" in text
    assert "Q3 revenue" in text


def test_media_dir_and_threshold(tmp_path: Path):
    src = tmp_path / "report.pdf"
    src.write_text("x", encoding="utf-8")
    folder = media_dir_for(src)
    assert folder.name == "report"
    assert IMAGE_ONLY_PAGE_CHAR_THRESHOLD > 0
