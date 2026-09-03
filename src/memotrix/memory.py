"""LangChain-style Memory facade: compose embeddings + store, then add/search."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from memotrix.config import MemoryConfig, resolve_database_url
from memotrix.embeddings import Embeddings, HuggingFaceEmbeddings, coerce_embeddings
from memotrix.processes.document_store import DocumentStore, estimate_tokens
from memotrix.processes.ingestion import IngestionPipeline
from memotrix.processes.retrieval import RetrievalPipeline
from memotrix.utils.exceptions import ConfigurationError
from memotrix.utils.models import DocumentData
from memotrix.vectorDB.hybrid import HybridSearchEngine
from memotrix.vectorstores import InMemoryStore, PostgresStore, VectorStoreBundle


class Memory:
    """
    Composable hybrid memory.

    >>> from memotrix import Memory
    >>> from memotrix.embeddings import HuggingFaceEmbeddings
    >>> from memotrix.vectorstores import PostgresStore
    >>> embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
    >>> memory = Memory(
    ...     embeddings=embeddings,
    ...     store=PostgresStore(connection=os.environ["DATABASE_URL"], embeddings=embeddings),
    ... )
    >>> memory.add("report.pdf")
    >>> memory.search("what is the revenue?")
    """

    def __init__(
        self,
        embeddings: Union[Embeddings, str],
        *,
        store: Any = None,
        connection: Optional[str] = None,
        backend: Optional[str] = None,
        config: Optional[MemoryConfig] = None,
        document_store: Optional[DocumentStore] = None,
        extract_file: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.embeddings = coerce_embeddings(embeddings)
        self.document_store = document_store or DocumentStore()
        self.connection: Optional[str] = None
        self._extract_file = extract_file

        bundle = self._resolve_store(
            store=store,
            connection=connection,
            backend=backend,
        )
        self._bundle = bundle
        self.dense = bundle.dense
        self.sparse = bundle.sparse
        self.native_store = bundle.native

        ingest_cfg = self.config.ingest
        retrieval_cfg = self.config.retrieval
        chunking_cfg = self.config.chunking
        reranker = (
            retrieval_cfg.reranker_model
            or self.config.reranker_model
        )
        self.ingest_pipeline = IngestionPipeline(
            self.dense,
            self.sparse,
            embeddings=self.embeddings,
            document_store=self.document_store,
            dedup_threshold=ingest_cfg.dedup_threshold,
            enable_dedup=ingest_cfg.enable_dedup,
            chunk_size=chunking_cfg.chunk_size,
            chunk_overlap=chunking_cfg.chunk_overlap,
        )
        self.hybrid = HybridSearchEngine(self.dense, self.sparse)
        self.retrieval = RetrievalPipeline(
            self.hybrid,
            embeddings=self.embeddings,
            reranker_model_name=reranker,
            document_store=self.document_store,
            expand_mode=retrieval_cfg.expand_mode,
            neighbor_window=retrieval_cfg.neighbor_window,
            enable_memory_boost=retrieval_cfg.enable_memory_boost,
            enable_conditional_rerank=retrieval_cfg.enable_rerank and bool(reranker),
            fusion_k=retrieval_cfg.fusion_k,
            query_cache_size=retrieval_cfg.query_cache_size,
        )
        self._rehydrate_document_store()

    def _resolve_store(
        self,
        *,
        store: Any,
        connection: Optional[str],
        backend: Optional[str],
    ) -> VectorStoreBundle:
        if store is not None and not isinstance(store, str):
            if isinstance(store, VectorStoreBundle):
                return store
            bundle = getattr(store, "bundle", None)
            if isinstance(bundle, VectorStoreBundle):
                self.connection = getattr(store, "connection", None) or connection
                return bundle
            dense = getattr(store, "dense", None)
            sparse = getattr(store, "sparse", None)
            if dense is not None and sparse is not None:
                return VectorStoreBundle(
                    dense=dense,
                    sparse=sparse,
                    native=getattr(store, "native", None),
                )
            raise ConfigurationError(
                "store must be InMemoryStore, PostgresStore, or a bundle with dense+sparse indexes"
            )

        if isinstance(store, str):
            backend_name = store
        elif backend is not None:
            backend_name = backend
        elif connection:
            backend_name = "postgres"
        else:
            backend_name = self.config.backend

        if connection and backend_name == "memory":
            raise ConfigurationError(
                "connection= is incompatible with backend='memory'; "
                "pass backend='postgres' or omit backend to auto-select postgres"
            )

        if backend_name == "postgres":
            dsn = resolve_database_url(connection=connection or self.config.connection)
            self.connection = dsn
            pg = PostgresStore(
                connection=dsn,
                embeddings=self.embeddings,
                table_name=self.config.table_name,
            )
            return pg.bundle
        if backend_name in (None, "memory"):
            mem = InMemoryStore(
                self.embeddings,
                max_elements=self.config.max_elements,
            )
            return mem.bundle
        raise ConfigurationError(
            f"Unknown memory backend {backend_name!r}; use 'memory' or 'postgres'"
        )

    def _iter_payloads(self):
        iterator = getattr(self.dense, "iter_payloads", None)
        if not callable(iterator):
            return
        yield from iterator()

    def _rehydrate_document_store(self) -> None:
        payloads = [payload for _chunk_id, payload in self._iter_payloads()]
        if payloads:
            self.document_store.rehydrate(payloads)

    @classmethod
    def from_config(cls, config: MemoryConfig, *, embeddings: Optional[Embeddings] = None) -> "Memory":
        embed = embeddings
        if embed is None:
            if not config.embedding_model:
                raise ConfigurationError(
                    "MemoryConfig.embedding_model is required unless you pass embeddings="
                )
            embed = HuggingFaceEmbeddings(model=config.embedding_model)
        return cls(
            embeddings=embed,
            connection=config.connection,
            backend=config.backend,
            config=config,
        )

    @classmethod
    def from_env(cls) -> "Memory":
        """Build from environment variables. Fails if model / DSN are missing."""
        return cls.from_config(MemoryConfig.from_env())

    def extract(
        self,
        path: str | Path,
        *,
        describe_images: Optional[bool] = None,
        generate_srt: bool = False,
    ):
        extractor = self._extract_file
        if extractor is None:
            from memotrix.filetypes import extract_file as extractor

        describe = (
            self.config.ingest.describe_images
            if describe_images is None
            else describe_images
        )
        return extractor(path, describe_images=describe, generate_srt=generate_srt)

    def add(
        self,
        path: str | Path,
        *,
        describe_images: Optional[bool] = None,
        generate_srt: bool = False,
    ) -> Dict[str, Any]:
        """Extract a file and ingest its chunks."""
        source = Path(path)
        doc = self.extract(source, describe_images=describe_images, generate_srt=generate_srt)
        resolved = str(source.resolve())
        doc.metadata["source_path"] = resolved
        doc.metadata.setdefault("memory_type", "semantic")
        stats = self.ingest_pipeline.ingest_many([(doc, resolved)])
        return {
            "filename": source.name,
            "path": resolved,
            "chunks": int(stats.get("chunks") or 0),
            "inserted": int(stats.get("inserted") or 0),
            "merged": int(stats.get("merged") or 0),
            "extracted_images": int((doc.metadata or {}).get("extracted_image_count") or 0)
            or len(getattr(doc, "images", None) or []),
        }

    def add_text(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        memory_type: str = "semantic",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a free-text memory (facts, chat turns, preferences)."""
        if text is None or not str(text).strip():
            raise ConfigurationError("add_text requires non-empty text")

        source = source_id or f"text:{uuid.uuid4()}"
        meta = dict(metadata or {})
        meta["source_path"] = source
        meta.setdefault("filename", Path(str(source)).name)
        meta["memory_type"] = memory_type
        if session_id is not None:
            meta["session_id"] = session_id

        doc = DocumentData(
            metadata=meta,
            sections=[{"title": meta.get("title") or "", "content": str(text)}],
            text=str(text),
            validation={},
        )
        stats = self.ingest_pipeline.ingest_many([(doc, source)])
        return {
            "filename": meta["filename"],
            "path": source,
            "chunks": int(stats.get("chunks") or 0),
            "inserted": int(stats.get("inserted") or 0),
            "merged": int(stats.get("merged") or 0),
            "extracted_images": 0,
        }

    def add_documents(
        self,
        items: Sequence,
    ) -> dict:
        """Ingest already-extracted DocumentData objects."""
        return self.ingest_pipeline.ingest_many(items)

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        max_tokens_returned: Optional[int] = None,
        filters: Any = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        expand_mode: Optional[str] = None,
    ) -> List[dict]:
        """
        Hybrid search over ingested memories.

        Extra filters (``session_id``, ``memory_type``, or a ``filters`` dict)
        are exact-match constraints on chunk payloads.
        """
        retrieval_cfg = self.config.retrieval
        return self.retrieval.search(
            query,
            top_k=top_k if top_k is not None else retrieval_cfg.top_k,
            max_tokens_returned=(
                max_tokens_returned
                if max_tokens_returned is not None
                else retrieval_cfg.max_tokens
            ),
            filters=filters,
            session_id=session_id,
            memory_type=memory_type,
            expand_mode=expand_mode,
        )

    def delete(self, source_path: str) -> int:
        """Remove every chunk for ``source_path`` (or filename) from both indexes."""
        if not source_path:
            return 0
        dense_deleted = self.dense.delete_by_source_path(source_path)
        sparse_deleted = self.sparse.delete_by_source_path(source_path)
        self.document_store.delete_source(source_path)
        return max(int(dense_deleted or 0), int(sparse_deleted or 0))

    def list_sources(self) -> List[Dict[str, Any]]:
        """Unique sources currently in the store, with chunk counts."""
        counts: Dict[str, Dict[str, Any]] = {}
        for _chunk_id, payload in self._iter_payloads():
            key = payload.get("source_path") or payload.get("filename") or ""
            if not key:
                continue
            entry = counts.setdefault(
                key,
                {
                    "source_path": payload.get("source_path") or key,
                    "filename": payload.get("filename") or Path(str(key)).name,
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        return list(counts.values())

    def as_retriever(self) -> RetrievalPipeline:
        return self.retrieval

    def as_stack(self) -> Dict[str, Any]:
        """Compatibility dict used by the Memotrix server/agent."""
        return {
            "dense": self.dense,
            "sparse": self.sparse,
            "store": self.native_store,
            "document_store": self.document_store,
            "hybrid": self.hybrid,
            "ingest_pipeline": self.ingest_pipeline,
            "retrieval": self.retrieval,
            "memory": self,
        }

    def close(self) -> None:
        self._bundle.close()


__all__ = ["Memory", "estimate_tokens"]
