"""
Phase 0 eval harness for Memotrix agent-memory retrieval.

Measures Recall@5, MRR, and per-query latency against tests/eval_set.jsonl
using the fixture corpus in tests/fixtures/agent_memory/.

Usage (from repo root):
    python -m tests.eval_harness
    python -m tests.eval_harness --top-k 5 --max-tokens 800
    python -m tests.eval_harness --expand-mode none   # ablate expansion
    python -m tests.eval_harness --legacy-full-file    # old behavior baseline

Outputs a JSON summary to stdout and optionally --out path.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.processes.document_store import DocumentStore
from memotrix.processes.ingestion import IngestionPipeline
from memotrix.processes.retrieval import RetrievalPipeline
from memotrix.utils.outputSturcture import build_document
from memotrix.vectorDB.hnsw_index import HNSWDenseIndex
from memotrix.vectorDB.hybrid import HybridSearchEngine
from memotrix.vectorDB.sparse_index import BM25SparseIndex

DEFAULT_EVAL_SET = ROOT / "tests" / "eval_set.jsonl"
DEFAULT_CORPUS = ROOT / "tests" / "fixtures" / "agent_memory"


@dataclass
class QueryResult:
    id: str
    query: str
    hit: bool
    reciprocal_rank: float
    latency_ms: float
    top_sources: List[str]
    matched_contains: Optional[str]


@dataclass
class EvalSummary:
    n_queries: int
    recall_at_k: float
    mrr: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    top_k: int
    expand_mode: str
    embedding_model: str
    misses: List[str]


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_corpus_docs(corpus_dir: Path) -> List[Any]:
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc = build_document(
            path,
            text,
            extra_metadata={
                "source_path": str(path.resolve()),
                "memory_type": _guess_memory_type(path.name),
            },
        )
        docs.append(doc)
    if not docs:
        raise FileNotFoundError(f"No markdown fixtures in {corpus_dir}")
    return docs


def _guess_memory_type(filename: str) -> str:
    name = filename.lower()
    if name.startswith("session_"):
        return "episodic"
    if name.startswith("procedures_"):
        return "procedural"
    if name.startswith("semantic_"):
        return "semantic"
    return "semantic"


def is_relevant(result: Dict[str, Any], relevant: Sequence[Dict[str, str]]) -> bool:
    """A hit counts if any returned chunk matches a ground-truth source+contains."""
    source = (
        Path(result.get("source_path") or "").name
        or result.get("filename")
        or ""
    )
    haystacks = [
        result.get("matched_chunk") or "",
        result.get("chunk_text") or "",
        result.get("content") or "",
    ]
    blob = "\n".join(haystacks).lower()

    for gold in relevant:
        gold_source = gold.get("source", "")
        needle = gold.get("contains", "")
        if gold_source and source != gold_source:
            continue
        if needle and needle.lower() in blob:
            return True
    return False


def first_relevant_rank(
    results: Sequence[Dict[str, Any]], relevant: Sequence[Dict[str, str]]
) -> Optional[int]:
    for rank, result in enumerate(results, start=1):
        if is_relevant(result, relevant):
            return rank
    return None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def build_pipeline(
    docs: List[Any],
    *,
    embedding_model: str,
    expand_mode: str,
    legacy_full_file: bool,
) -> RetrievalPipeline:
    embeddings = HuggingFaceEmbeddings(model=embedding_model)
    dense = HNSWDenseIndex(dim=embeddings.dimension, max_elements=max(1000, len(docs) * 50))
    sparse = BM25SparseIndex()
    store = DocumentStore()
    ingest = IngestionPipeline(
        dense,
        sparse,
        embeddings=embeddings,
        document_store=store,
        enable_dedup=False,  # keep fixture chunks distinct for recall measurement
    )
    ingest.ingest_many(docs)

    hybrid = HybridSearchEngine(dense, sparse)
    return RetrievalPipeline(
        hybrid,
        embeddings=embeddings,
        document_store=store,
        expand_mode="full_file" if legacy_full_file else expand_mode,
        expand_top_file_content=legacy_full_file,
        track_access=False,
        # Measure retrieval quality without agent latency shortcuts.
        enable_conditional_rerank=False,
        enable_memory_boost=False,
    )


def run_eval(
    pipeline: RetrievalPipeline,
    eval_rows: List[Dict[str, Any]],
    *,
    top_k: int,
    max_tokens_returned: Optional[int],
) -> tuple[EvalSummary, List[QueryResult]]:
    per_query: List[QueryResult] = []
    hits = 0
    rr_sum = 0.0
    latencies: List[float] = []
    misses: List[str] = []

    for row in eval_rows:
        qid = row["id"]
        query = row["query"]
        relevant = row.get("relevant", [])

        t0 = time.perf_counter()
        results = pipeline.search(
            query, top_k=top_k, max_tokens_returned=max_tokens_returned
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        rank = first_relevant_rank(results, relevant)
        hit = rank is not None
        rr = 1.0 / rank if rank else 0.0
        if hit:
            hits += 1
        else:
            misses.append(qid)
        rr_sum += rr

        matched = None
        if rank:
            matched = relevant[0].get("contains") if relevant else None

        per_query.append(
            QueryResult(
                id=qid,
                query=query,
                hit=hit,
                reciprocal_rank=rr,
                latency_ms=latency_ms,
                top_sources=[
                    Path(r.get("source_path") or "").name or r.get("filename", "")
                    for r in results
                ],
                matched_contains=matched if hit else None,
            )
        )

    n = len(eval_rows) or 1
    summary = EvalSummary(
        n_queries=len(eval_rows),
        recall_at_k=hits / n,
        mrr=rr_sum / n,
        latency_ms_mean=statistics.fmean(latencies) if latencies else 0.0,
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        top_k=top_k,
        expand_mode=pipeline.expand_mode,
        embedding_model=embedding_model_name(pipeline),
        misses=misses,
    )
    return summary, per_query


def embedding_model_name(pipeline: RetrievalPipeline) -> str:
    name = getattr(pipeline.encoder, "model_name_or_path", None)
    if isinstance(name, str) and name:
        return name
    return getattr(pipeline, "embedding_model_name", None) or os.environ.get("EMBEDDING_MODEL") or ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Memotrix retrieval eval harness")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--no-token-cap", action="store_true")
    parser.add_argument(
        "--expand-mode",
        choices=["neighbors", "none", "full_file"],
        default="neighbors",
    )
    parser.add_argument(
        "--legacy-full-file",
        action="store_true",
        help="Force pre-Phase-3 full-file expansion for before/after comparison",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL"),
        help="SentenceTransformer model id (or set EMBEDDING_MODEL)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-query", action="store_true")
    args = parser.parse_args(argv)
    if not args.embedding_model:
        parser.error("Pass --embedding-model or set EMBEDDING_MODEL")

    eval_rows = load_eval_set(args.eval_set)
    docs = load_corpus_docs(args.corpus)

    print(
        f"Ingesting {len(docs)} docs / {len(eval_rows)} queries "
        f"(model={args.embedding_model}, expand={args.expand_mode})…",
        file=sys.stderr,
    )
    pipeline = build_pipeline(
        docs,
        embedding_model=args.embedding_model,
        expand_mode=args.expand_mode,
        legacy_full_file=args.legacy_full_file,
    )

    max_tokens = None if args.no_token_cap else args.max_tokens
    summary, per_query = run_eval(
        pipeline,
        eval_rows,
        top_k=args.top_k,
        max_tokens_returned=max_tokens,
    )

    payload: Dict[str, Any] = {
        "summary": asdict(summary),
        "config": {
            "eval_set": str(args.eval_set),
            "corpus": str(args.corpus),
            "max_tokens_returned": max_tokens,
            "legacy_full_file": args.legacy_full_file,
        },
    }
    if args.per_query:
        payload["per_query"] = [asdict(q) for q in per_query]

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)

    s = summary
    print(
        f"\nRecall@{s.top_k}={s.recall_at_k:.3f}  MRR={s.mrr:.3f}  "
        f"latency_ms mean={s.latency_ms_mean:.1f} p50={s.latency_ms_p50:.1f} "
        f"p95={s.latency_ms_p95:.1f}",
        file=sys.stderr,
    )
    if s.misses:
        print(f"Misses ({len(s.misses)}): {', '.join(s.misses)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
