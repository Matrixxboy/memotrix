"""Shared embedding defaults and helpers for agent-memory retrieval."""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# No silent model/dim defaults. Callers must pass an explicit model name.
# HuggingFaceEmbeddings infers dimension from the loaded model.

# BGE asymmetric retrieval: prefix queries, leave passages bare.
BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

# Phase 4 — near-duplicate merge threshold (cosine similarity)
DEDUP_COSINE_THRESHOLD = 0.95

# Phase 2 — skip cross-encoder only when RRF already has a clear winner
RERANK_SKIP_MARGIN = 0.015
RERANK_AGREE_MARGIN = 0.008

_encoder_cache: Dict[str, Any] = {}
_reranker_cache: Dict[str, Any] = {}
_model_lock = threading.Lock()
_warmup_started = False


def is_bge_model(model_name: str) -> bool:
    name = model_name.lower()
    return "bge-" in name or name.startswith("baai/bge")


def format_query_for_embedding(query: str, model_name: str) -> str:
    text = query.strip()
    if is_bge_model(model_name) and text:
        return f"{BGE_QUERY_INSTRUCTION}{text}"
    return text


def cache_key_for_query(query: str, model_name: str) -> str:
    normalized = " ".join(query.strip().lower().split())
    raw = f"{model_name}::{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hub_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cached_locally(model_name: str) -> bool:
    """True when HuggingFace hub already has this model snapshot."""
    safe = "models--" + model_name.replace("/", "--")
    root = _hub_cache_dir() / safe
    if not root.is_dir():
        return False
    snapshots = root / "snapshots"
    if snapshots.is_dir() and any(snapshots.iterdir()):
        return True
    return any(root.rglob("config.json"))


def _mlog(msg: str) -> None:
    try:
        from memotrix.utils.trace import mlog

        mlog("boot", msg)
    except Exception:  # noqa: BLE001
        pass


def get_sentence_transformer(model_name: str):
    """Load SentenceTransformer once per process (shared by ingest + retrieval)."""
    cached = _encoder_cache.get(model_name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _encoder_cache.get(model_name)
        if cached is not None:
            return cached

        local_only = _model_cached_locally(model_name)
        if local_only:
            # Avoid Hub network checks that can hang for minutes on Windows.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            _mlog(
                f"loading embedding model from local cache: {model_name} "
                "(importing torch/sentence_transformers can take ~30s)"
            )
        else:
            _mlog(
                f"downloading embedding model: {model_name} "
                "(first download can take several minutes)"
            )

        _mlog("importing sentence_transformers...")
        from sentence_transformers import SentenceTransformer

        _mlog(f"constructing SentenceTransformer({model_name})...")
        try:
            model = SentenceTransformer(model_name, local_files_only=local_only)
        except Exception:
            # Cache detection can be wrong; fall back to online load once.
            if local_only:
                _mlog("local cache load failed; retrying with Hub access...")
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                model = SentenceTransformer(model_name, local_files_only=False)
            else:
                raise
        _encoder_cache[model_name] = model
        _mlog(f"embedding model ready: {model_name}")
        return model


def get_cross_encoder(model_name: str):
    """Load CrossEncoder once per process (lazy; not required for server boot)."""
    cached = _reranker_cache.get(model_name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _reranker_cache.get(model_name)
        if cached is not None:
            return cached

        local_only = _model_cached_locally(model_name)
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            _mlog(f"loading reranker from local cache: {model_name}")
        else:
            _mlog(f"loading reranker model: {model_name}")

        from sentence_transformers import CrossEncoder

        try:
            model = CrossEncoder(model_name, local_files_only=local_only)
        except TypeError:
            # Older CrossEncoder may not accept local_files_only.
            model = CrossEncoder(model_name)
        except Exception:
            if local_only:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                model = CrossEncoder(model_name)
            else:
                raise
        _reranker_cache[model_name] = model
        _mlog(f"reranker model ready: {model_name}")
        return model


def warm_embedding_model_async(model_name: str) -> None:
    """Preload the embedder in a daemon thread so the first chat is not blocked."""
    global _warmup_started
    if _warmup_started or model_name in _encoder_cache:
        return
    _warmup_started = True

    def _run() -> None:
        try:
            _mlog(f"background warmup start: {model_name}")
            get_sentence_transformer(model_name)
            _mlog(f"background warmup done: {model_name}")
        except Exception as exc:  # noqa: BLE001
            _mlog(f"background warmup FAILED: {exc}")

    threading.Thread(target=_run, name="embed-warmup", daemon=True).start()


class QueryEmbeddingCache:
    """Session-scoped LRU cache for repeated / near-identical agent queries."""

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[str, List[float]] = OrderedDict()

    def get(self, key: str) -> Optional[List[float]]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, vector: Sequence[float]) -> None:
        self._store[key] = list(vector)
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
