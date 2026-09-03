"""
Memotrix context CLI — ingest files, retrieve agent memory, synthesize context via OpenRouter.

Usage (from repo root):
  python -m tests.context_cli ingest --path tests/fixtures/agent_memory
  python -m tests.context_cli query "What UI theme does the user prefer?"
  python -m tests.context_cli ask "What UI theme does the user prefer?"
  python -m tests.context_cli chat

Requires OPENROUTER_API_KEY in .env for `ask` / `chat` (retrieval-only `query` works without it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from memotrix.config import resolve_database_url
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.filetypes.document.document import DocumentExtractor
from memotrix.filetypes.images.images import ImageExtractor
from memotrix.filetypes.structuredData.structuredData import StructuredDataExtractor
from memotrix.processes.document_store import DocumentStore, estimate_tokens
from memotrix.processes.ingestion import IngestionPipeline
from memotrix.processes.retrieval import RetrievalPipeline
from memotrix.utils.ai_integration import OpenRouterProvider, configure_ai_provider, get_ai_router
from memotrix.utils.exceptions import ConfigurationError, UnsupportedDocumentTypeError
from memotrix.utils.models import DocumentData
from memotrix.vectorDB.hnsw_index import HNSWDenseIndex
from memotrix.vectorDB.sparse_index import BM25SparseIndex
from memotrix.vectorDB.hybrid import HybridSearchEngine

SUPPORTED_EXTENSIONS = {
    *DocumentExtractor.EXTRACTORS.keys(),
    *StructuredDataExtractor.EXTRACTORS.keys(),
    *ImageExtractor.SUPPORTED_EXTENSIONS,
}

CONTEXT_SYSTEM_PROMPT = """You are Memotrix context assembler for an AI agent.
Given a user query and retrieved memory chunks, produce a tight, high-signal context block.

Rules:
- Prefer facts present in the retrieved memory; do not invent sources.
- Merge near-duplicates; drop noise and filler.
- Keep the answer short enough for an agent loop (roughly under 400 words unless asked otherwise).
- Structure as:
  1) Direct answer (2–4 sentences)
  2) Key facts (bullets)
  3) Source files (filenames only)
If memory is insufficient, say what is missing instead of guessing.
"""


def extract_file(path: Path) -> DocumentData:
    suffix = path.suffix.lower()
    if suffix in DocumentExtractor.EXTRACTORS:
        return DocumentExtractor.extract(path)
    if suffix in StructuredDataExtractor.EXTRACTORS:
        return StructuredDataExtractor.extract(path)
    if suffix in ImageExtractor.SUPPORTED_EXTENSIONS:
        return ImageExtractor.extract(path)
    raise UnsupportedDocumentTypeError(f"Unsupported file type: {suffix}")


def collect_files(path: Path, *, recursive: bool = True) -> List[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


class MemotrixContextApp:
    """Wires ingestion + hybrid retrieval + OpenRouter into one CLI surface."""

    def __init__(
        self,
        *,
        backend: str = "memory",
        embedding_model: str,
        openrouter_model: Optional[str] = None,
        top_k: int = 3,
        max_tokens: int = 800,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if not embedding_model:
            raise ConfigurationError(
                "Pass --embedding-model or set EMBEDDING_MODEL"
            )
        self.embeddings = HuggingFaceEmbeddings(model=embedding_model)
        self.backend = backend
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.memory_type = memory_type
        self.session_id = session_id
        self.document_store = DocumentStore()
        self._store = None

        if backend == "postgres":
            from memotrix.vectorDB.postgres_store import create_postgres_indexes

            dense, sparse, store = create_postgres_indexes(
                connection=resolve_database_url(),
                dim=self.embeddings.dimension,
            )
            self._store = store
            self.dense = dense
            self.sparse = sparse
        else:
            self.dense = HNSWDenseIndex(
                dim=self.embeddings.dimension, max_elements=50_000
            )
            self.sparse = BM25SparseIndex()

        self._rebuild_pipelines()

        if os.getenv("OPENROUTER_API_KEY"):
            configure_ai_provider(
                OpenRouterProvider(
                    model=openrouter_model or os.getenv("OPENROUTER_MODEL")
                )
            )

    def _rebuild_pipelines(self) -> None:
        self.ingest_pipeline = IngestionPipeline(
            self.dense,
            self.sparse,
            embeddings=self.embeddings,
            document_store=self.document_store,
        )
        self.hybrid = HybridSearchEngine(self.dense, self.sparse)
        self.retrieval = RetrievalPipeline(
            self.hybrid,
            embeddings=self.embeddings,
            document_store=self.document_store,
            expand_mode="neighbors",
        )

    def close(self) -> None:
        if self._store is not None:
            self._store.close()

    def ingest_path(
        self,
        path: Path,
        *,
        recursive: bool = True,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        if reset:
            if self.backend == "postgres" and self._store is not None:
                self._store.clear()
            else:
                self.dense = HNSWDenseIndex(
                    dim=self.embeddings.dimension, max_elements=50_000
                )
                self.sparse = BM25SparseIndex()
            self.document_store = DocumentStore()
            self._rebuild_pipelines()

        files = collect_files(path, recursive=recursive)
        prepared: List[Tuple[DocumentData, str]] = []
        errors: List[str] = []

        for file_path in files:
            try:
                doc = extract_file(file_path)
                resolved = str(file_path.resolve())
                doc.metadata["source_path"] = resolved
                if memory_type:
                    doc.metadata["memory_type"] = memory_type
                if session_id:
                    doc.metadata["session_id"] = session_id
                prepared.append((doc, resolved))
            except Exception as exc:  # noqa: BLE001 — keep going on bad files
                errors.append(f"{file_path.name}: {exc}")

        stats = (
            self.ingest_pipeline.ingest_many(prepared)
            if prepared
            else {"chunks": 0, "merged": 0, "inserted": 0}
        )
        return {
            "files_found": len(files),
            "files_ingested": len(prepared),
            "errors": errors,
            **stats,
            "backend": self.backend,
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        return self.retrieval.search(
            query,
            top_k=self.top_k,
            max_tokens_returned=self.max_tokens,
            session_id=self.session_id,
            memory_type=self.memory_type,
        )

    def format_raw_context(self, query: str, results: List[Dict[str, Any]]) -> str:
        blocks: List[str] = [f"Query: {query}", ""]
        total_tokens = 0
        for i, hit in enumerate(results, start=1):
            source = Path(hit.get("source_path") or "").name or hit.get("filename", "?")
            text = hit.get("chunk_text") or hit.get("matched_chunk") or ""
            score = float(hit.get("score") or 0.0)
            tokens = int(hit.get("tokens_estimate") or estimate_tokens(text))
            total_tokens += tokens
            blocks.append(
                f"[{i}] score={score:.4f} tokens~={tokens} "
                f"file={source} section={hit.get('section_title', '')}"
            )
            blocks.append(text.strip())
            blocks.append("")
        blocks.append(f"Total tokens~={total_tokens}")
        return "\n".join(blocks).strip()

    def synthesize_context(self, query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant memory found for this query."

        memory_blob = self.format_raw_context(query, results)
        prompt = (
            f"User query:\n{query}\n\n"
            f"Retrieved memory:\n{memory_blob}\n\n"
            "Assemble the best context for an agent to answer this query."
        )
        return get_ai_router().generate_text(prompt, system=CONTEXT_SYSTEM_PROMPT)

    def ask(self, query: str) -> Dict[str, Any]:
        results = self.search(query)
        return {
            "query": query,
            "hits": len(results),
            "raw_context": self.format_raw_context(query, results),
            "better_context": self.synthesize_context(query, results),
            "sources": [
                Path(r.get("source_path") or "").name or r.get("filename", "")
                for r in results
            ],
        }

    def ensure_fixture_corpus(self) -> Optional[Dict[str, Any]]:
        if len(self.document_store) > 0 or self.backend != "memory":
            return None
        fixture = ROOT / "tests" / "fixtures" / "agent_memory"
        if not fixture.exists():
            return None
        print(f"(auto-ingesting {fixture})", file=sys.stderr)
        summary = self.ingest_path(fixture)
        _print_ingest_summary(summary)
        return summary


def _print_ingest_summary(summary: Dict[str, Any]) -> None:
    print(
        f"Ingested {summary['files_ingested']}/{summary['files_found']} files "
        f"-> chunks={summary.get('chunks', 0)} inserted={summary.get('inserted', 0)} "
        f"merged={summary.get('merged', 0)} backend={summary.get('backend')}"
    )
    for err in summary.get("errors") or []:
        print(f"  ! {err}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memotrix-context",
        description="Memotrix CLI: ingest → retrieve → OpenRouter context synthesis",
    )
    parser.add_argument(
        "--backend",
        choices=["memory", "postgres"],
        default="memory",
        help="Index backend (default: in-memory)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL"),
        help="Embedding model id (or set EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--openrouter-model",
        default=None,
        help="OpenRouter model (default: OPENROUTER_MODEL or openai/gpt-4o-mini)",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--memory-type", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a file or folder")
    ingest.add_argument("--path", required=True, type=Path)
    ingest.add_argument("--no-recursive", action="store_true")
    ingest.add_argument("--reset", action="store_true")
    ingest.add_argument("--session-id", default=None)
    ingest.add_argument("--memory-type", default=None)

    query = sub.add_parser("query", help="Retrieve raw neighbor-window context (no LLM)")
    query.add_argument("text", nargs="+", help="Query string")

    ask = sub.add_parser("ask", help="Retrieve + OpenRouter → better agent context")
    ask.add_argument("text", nargs="+", help="Query string")
    ask.add_argument("--json", action="store_true")

    chat = sub.add_parser("chat", help="Interactive ask loop")
    chat.add_argument("--ingest", type=Path, default=None)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.embedding_model:
        parser.error("Pass --embedding-model or set EMBEDDING_MODEL")

    app = MemotrixContextApp(
        backend=args.backend,
        embedding_model=args.embedding_model,
        openrouter_model=args.openrouter_model,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        memory_type=args.memory_type,
        session_id=args.session_id,
    )

    try:
        if args.command == "ingest":
            summary = app.ingest_path(
                args.path,
                recursive=not args.no_recursive,
                memory_type=args.memory_type or app.memory_type,
                session_id=args.session_id or app.session_id,
                reset=args.reset,
            )
            _print_ingest_summary(summary)
            return 0 if summary["files_ingested"] else 1

        if args.command == "query":
            app.ensure_fixture_corpus()
            q = " ".join(args.text)
            print(app.format_raw_context(q, app.search(q)))
            return 0

        if args.command == "ask":
            if not os.getenv("OPENROUTER_API_KEY"):
                print(
                    "OPENROUTER_API_KEY is not set in .env — "
                    "add it for OpenRouter synthesis, or use `query` for raw context.",
                    file=sys.stderr,
                )
                return 2
            app.ensure_fixture_corpus()
            payload = app.ask(" ".join(args.text))
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print("=== Better context (OpenRouter) ===\n")
                print(payload["better_context"])
                print("\n=== Sources ===")
                print(", ".join(payload["sources"]) or "(none)")
            return 0

        if args.command == "chat":
            if not os.getenv("OPENROUTER_API_KEY"):
                print("OPENROUTER_API_KEY is not set in .env", file=sys.stderr)
                return 2
            if args.ingest:
                _print_ingest_summary(app.ingest_path(args.ingest))
            else:
                app.ensure_fixture_corpus()

            print("Memotrix chat — empty line or 'quit' to exit.\n")
            while True:
                try:
                    line = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line or line.lower() in {"quit", "exit", "q"}:
                    break
                payload = app.ask(line)
                print("\n--- context ---")
                print(payload["better_context"])
                print(f"\nsources: {', '.join(payload['sources']) or '(none)'}\n")
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
