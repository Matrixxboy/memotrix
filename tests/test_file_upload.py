"""
Manual integration test: upload a folder of real files and run the full Memotrix pipeline.

Usage:
  python tests/test_file_upload.py                              # folder picker dialog
  python tests/test_file_upload.py --folder path/to/docs        # ingest entire folder
  python tests/test_file_upload.py --file ./filesForTests/      # from repo root or tests/
  python tests/test_file_upload.py --file tests/filesForTests/  # explicit path
  python tests/test_file_upload.py --file path/to/doc.pdf       # single file
  python tests/test_file_upload.py --folder path/to/docs --query "what is TUG?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memotrix.config import resolve_database_url
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.filetypes.document.document import DocumentExtractor
from memotrix.filetypes.structuredData.structuredData import StructuredDataExtractor
from memotrix.processes.chunking import chunk_document
from memotrix.processes.document_store import DocumentStore
from memotrix.processes.ingestion import IngestionPipeline
from memotrix.processes.retrieval import RetrievalPipeline
from memotrix.utils.exceptions import ConfigurationError, DocumentExtractionError, UnsupportedDocumentTypeError
from memotrix.utils.models import DocumentData
from memotrix.vectorDB.hybrid import HybridSearchEngine
from memotrix.vectorDB.postgres_store import PgDenseIndex, PgSparseIndex, PostgresChunkStore, create_postgres_indexes

DEFAULT_SAMPLE_DIR = Path(__file__).parent / "fixtures"
DEFAULT_TEST_DIR = Path(__file__).parent / "filesForTests"
SUPPORTED_EXTENSIONS = set(DocumentExtractor.EXTRACTORS) | set(StructuredDataExtractor.EXTRACTORS)


def resolve_cli_path(cli_path: str) -> Path:
    """Resolve a user path from cwd, project root, or the tests/ folder."""
    raw = Path(cli_path).expanduser()
    candidates = [
        raw.resolve(),
        (PROJECT_ROOT / raw).resolve(),
        (Path(__file__).parent / raw).resolve(),
    ]

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Path not found: {cli_path}\n"
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def pick_folder() -> Path | None:
    """Open a native folder picker for bulk upload."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Select a folder to ingest")
    root.destroy()
    return Path(selected) if selected else None


def extract_file(path: Path) -> DocumentData:
    suffix = path.suffix.lower()
    if suffix in DocumentExtractor.EXTRACTORS:
        return DocumentExtractor.extract(path)
    if suffix in StructuredDataExtractor.EXTRACTORS:
        return StructuredDataExtractor.extract(path)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise UnsupportedDocumentTypeError(f"Unsupported file type '{suffix}'. Supported: {supported}")


def collect_files(folder: Path, *, recursive: bool = True) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()

    files = sorted(
        p.resolve()
        for p in candidates
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def resolve_folder_path(cli_folder: str | None) -> Path:
    if cli_folder:
        path = resolve_cli_path(cli_folder)
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a folder: {path}")
        return path

    picked = pick_folder()
    if picked is not None:
        return picked.resolve()

    if DEFAULT_SAMPLE_DIR.is_dir() and collect_files(DEFAULT_SAMPLE_DIR, recursive=False):
        print(f"No folder selected. Using bundled sample folder: {DEFAULT_SAMPLE_DIR}")
        return DEFAULT_SAMPLE_DIR.resolve()

    raise FileNotFoundError(
        "No folder selected and no sample fixtures found. "
        "Pass --folder or run with a display so the folder picker can open."
    )


def resolve_input_paths(cli_path: str | None, *, recursive: bool = True) -> list[Path]:
    """Accept a file or folder path and return the list of files to ingest."""
    if cli_path:
        path = resolve_cli_path(cli_path)
        if path.is_file():
            return [path]
        if path.is_dir():
            file_paths = collect_files(path, recursive=recursive)
            if not file_paths:
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise FileNotFoundError(
                    f"No supported files found in {path}. Supported extensions: {supported}"
                )
            return file_paths
        raise ValueError(f"Path is neither a file nor a folder: {path}")

    folder = resolve_folder_path(None)
    file_paths = collect_files(folder, recursive=recursive)
    if not file_paths:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported files found in {folder}. Supported extensions: {supported}"
        )
    return file_paths


def default_query(docs: list[DocumentData]) -> str:
    for doc in docs:
        for section in doc.sections:
            content = section.get("content", "").strip()
            if content:
                words = content.split()
                return " ".join(words[:6])
    if docs:
        return docs[0].metadata.get("filename", "document")
    return "document"


def ingest_files(
    file_paths: list[Path],
    *,
    database_url: str,
    embeddings,
    reset_db: bool = False,
) -> tuple[PgDenseIndex, PgSparseIndex, PostgresChunkStore, DocumentStore, list[DocumentData], int, list[str]]:
    dense_index, sparse_index, pg_store = create_postgres_indexes(
        connection=database_url,
        dim=embeddings.dimension,
    )
    if reset_db:
        pg_store.clear()
        print("Cleared PostgreSQL chunk table.")

    document_store = DocumentStore()
    ingestion = IngestionPipeline(
        dense_index, sparse_index, embeddings=embeddings, document_store=document_store
    )

    docs: list[DocumentData] = []
    total_chunks = 0
    failures: list[str] = []

    print(f"\nIngesting {len(file_paths)} file(s)...")
    for i, file_path in enumerate(file_paths, start=1):
        print(f"\n  [{i}/{len(file_paths)}] {file_path.name}")
        try:
            doc = extract_file(file_path)
            chunks = chunk_document(doc)
            if not chunks:
                print("      skipped — no chunks produced")
                continue

            ingestion.ingest(doc, source_path=file_path)
            docs.append(doc)
            total_chunks += len(chunks)
            print(
                f"      ok — {doc.validation.get('word_count', 0)} words, "
                f"{len(doc.sections)} section(s), {len(chunks)} chunk(s)"
            )
        except (DocumentExtractionError, FileNotFoundError, ValueError, RuntimeError) as exc:
            failures.append(f"{file_path.name}: {exc}")
            print(f"      failed — {exc}")

    return (dense_index, sparse_index, pg_store, document_store, docs, total_chunks, failures)


def safe_preview(text: str, limit: int = 160) -> str:
    preview = text[:limit] + ("..." if len(text) > limit else "")
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return preview.encode(encoding, errors="replace").decode(encoding)


def run_pipeline(
    file_paths: list[Path],
    query: str | None = None,
    *,
    database_url: str,
    embeddings,
    reset_db: bool = False,
) -> None:
    print("\n=== Memotrix bulk folder upload test ===")
    print(f"Files found: {len(file_paths)}")
    for path in file_paths:
        print(f"  - {path}")

    dense_index, sparse_index, pg_store, document_store, docs, total_chunks, failures = ingest_files(
        file_paths,
        database_url=database_url,
        embeddings=embeddings,
        reset_db=reset_db,
    )

    try:
        print(f"\nIndexed: {len(docs)} file(s), {total_chunks} total chunk(s) [PostgreSQL + pgvector]")
        if failures:
            print(f"Failed: {len(failures)} file(s)")
            for msg in failures:
                print(f"  - {msg}")

        if not docs:
            print("\nNothing indexed — no searchable content from this folder.")
            return

        search_query = query or default_query(docs)
        print(f"\nQuery: {search_query!r}")

        hybrid = HybridSearchEngine(dense_index, sparse_index)
        retrieval = RetrievalPipeline(
            hybrid, embeddings=embeddings, document_store=document_store
        )
        results = retrieval.retrieve(search_query, top_k=5)

        print(f"\nTop {len(results)} result(s):")
        for i, hit in enumerate(results, start=1):
            score = hit.get("score", 0.0)
            source = hit.get("filename", "?")
            is_top = i == 1
            content = hit.get("content") if is_top else hit.get("chunk_text", "")
            label = "full file content" if is_top and hit.get("content") else "chunk preview"
            preview = safe_preview(content, limit=300 if is_top else 160)
            print(f"\n  [{i}] score={score:.4f} file={source} type={hit.get('type', '?')} ({label})")
            print(f"      {preview}")
            if is_top and hit.get("matched_chunk"):
                print(f"      matched chunk: {safe_preview(hit['matched_chunk'], limit=120)}")

        print("\nDone.")
    finally:
        pg_store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload real files or a folder and test Memotrix ingestion + retrieval."
    )
    parser.add_argument(
        "--folder", "-d",
        help="Path to a folder (opens picker if neither --file nor --folder is given)",
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to a file or folder (folders are ingested in bulk)",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Only ingest files in the top folder")
    parser.add_argument("--query", "-q", help="Search query after ingestion")
    parser.add_argument(
        "--database-url",
        help="PostgreSQL URL (or set DATABASE_URL). No default credentials.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model id (or set EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Clear the PostgreSQL chunk table before ingesting",
    )
    args = parser.parse_args()

    if args.file and args.folder:
        raise ValueError("Use either --file or --folder, not both.")

    cli_path = args.file or args.folder
    file_paths = resolve_input_paths(cli_path, recursive=not args.no_recursive)
    try:
        database_url = args.database_url or resolve_database_url()
        model = args.embedding_model or os.environ.get("EMBEDDING_MODEL")
        if not model:
            raise ConfigurationError("Pass --embedding-model or set EMBEDDING_MODEL")
        embeddings = HuggingFaceEmbeddings(model=model)
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    run_pipeline(
        file_paths,
        query=args.query,
        database_url=database_url,
        embeddings=embeddings,
        reset_db=args.reset_db,
    )


if __name__ == "__main__":
    main()
