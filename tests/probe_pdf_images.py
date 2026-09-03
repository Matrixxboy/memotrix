"""
Probe PDF image extraction (no vision API by default).

Creates a sample PDF with:
  - Page 1: text + embedded figure
  - Page 2: image-only (scanned-style) page

Then runs PDFExtractor(describe_images=False) and reports saved files.

Usage:
  python -m tests.probe_pdf_images
  python -m tests.probe_pdf_images --pdf path/to/file.pdf
  python -m tests.probe_pdf_images --describe   # also call OpenRouter vision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def make_sample_pdf(path: Path) -> Path:
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)

    # Figure for page 1 (embedded)
    fig = Image.new("RGB", (320, 180), color=(30, 90, 160))
    draw = ImageDraw.Draw(fig)
    draw.rectangle([20, 20, 300, 160], outline=(255, 220, 80), width=3)
    draw.text((40, 70), "Memotrix Sample Chart", fill=(255, 255, 255))
    draw.text((40, 100), "Q3 Revenue: $42k", fill=(255, 220, 80))
    fig_path = path.parent / "_probe_fig.png"
    fig.save(fig_path)

    # Full-page image for page 2 (image-only)
    scan = Image.new("RGB", (600, 800), color=(250, 248, 240))
    sdraw = ImageDraw.Draw(scan)
    sdraw.text((40, 40), "SCANNED INVOICE", fill=(20, 20, 20))
    sdraw.text((40, 100), "Invoice #: INV-7781", fill=(20, 20, 20))
    sdraw.text((40, 140), "Customer: Matrixboy Labs", fill=(20, 20, 20))
    sdraw.text((40, 180), "Total due: $199.00", fill=(20, 20, 20))
    sdraw.text((40, 240), "(This page has no PDF text layer)", fill=(80, 80, 80))
    scan_path = path.parent / "_probe_scan.png"
    scan.save(scan_path)

    doc = fitz.open()

    # Page 1 — text + embedded image
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((72, 72), "Quarterly Report", fontsize=18)
    page1.insert_text(
        (72, 110),
        "Below is an embedded chart image between paragraphs of text.",
        fontsize=11,
    )
    page1.insert_image(fitz.Rect(72, 150, 392, 330), filename=str(fig_path))
    page1.insert_text(
        (72, 360),
        "The chart above shows Q3 revenue. This paragraph follows the image.",
        fontsize=11,
    )

    # Page 2 — image only (almost no extractable text)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_image(page2.rect, filename=str(scan_path))

    doc.save(str(path))
    doc.close()
    return path


def report(pdf_path: Path, *, describe: bool) -> int:
    from memotrix.filetypes.document.media import media_dir_for
    from memotrix.filetypes.document.pdf import PDFExtractor

    print(f"PDF: {pdf_path}")
    print(f"describe_images={describe}")
    print("-" * 60)

    doc = PDFExtractor().extract(pdf_path, describe_images=describe)
    media = media_dir_for(pdf_path)

    print("Metadata:")
    for key in (
        "page_count",
        "embedded_image_count",
        "image_only_pages",
        "extracted_image_count",
    ):
        print(f"  {key}: {doc.metadata.get(key)}")

    print(f"\nMedia folder: {media}")
    files = sorted(media.glob("*")) if media.exists() else []
    if not files:
        print("  (empty — extraction did not save files)")
    else:
        for f in files:
            size = f.stat().st_size
            print(f"  {f.name}  ({size:,} bytes)")

    print(f"\nDocumentData.images ({len(doc.images)}):")
    for i, img in enumerate(doc.images, start=1):
        p = Path(img.get("path") or "")
        exists = p.exists()
        desc = (img.get("description") or "")[:160].replace("\n", " ")
        print(f"  [{i}] exists={exists}  {p.name}")
        print(f"      context: {img.get('context')}")
        print(f"      desc: {desc}...")

    print("\nText preview (first 500 chars):")
    print((doc.text or "")[:500])
    print("-" * 60)

    ok_files = media.exists() and any(media.glob("*"))
    ok_images = len(doc.images) >= 1
    if ok_files and ok_images:
        print("RESULT: PDF image extraction looks good.")
        return 0
    print("RESULT: extraction incomplete — check PyMuPDF / PDF contents.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe PDF image extraction")
    parser.add_argument("--pdf", type=Path, default=None, help="Existing PDF to probe")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Also run OpenRouter vision captions (needs API key)",
    )
    args = parser.parse_args()

    if args.pdf:
        pdf_path = args.pdf.resolve()
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}", file=sys.stderr)
            return 2
    else:
        out = ROOT / "tests" / "fixtures" / "pdf_probe" / "sample_mixed.pdf"
        print(f"Building sample PDF at {out}")
        make_sample_pdf(out)
        pdf_path = out

    return report(pdf_path, describe=args.describe)


if __name__ == "__main__":
    raise SystemExit(main())
