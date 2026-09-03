from pathlib import Path
from typing import List, Sequence

from memotrix.processes.chunking import CHUNK_OVERLAP, CHUNK_SIZE, Chunk, chunk_document
from memotrix.processes.document_store import DocumentStore, document_key
from memotrix.processes.memory_scoring import stamp_new_memory_payload
from memotrix.utils.exceptions import ConfigurationError
from memotrix.utils.models import DocumentData
from memotrix.vectorDB.base import DenseIndex, SparseIndex
from memotrix.vectorDB.embeddings import DEDUP_COSINE_THRESHOLD


class IngestionPipeline:
    """
    Handles chunking a document, embedding the chunks,
    and storing them in the dual-representation Vector DB.
    """

    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        embeddings=None,
        document_store: DocumentStore | None = None,
        *,
        embedding_model_name: str | None = None,
        dedup_threshold: float = DEDUP_COSINE_THRESHOLD,
        enable_dedup: bool = True,
        encoder=None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        if embeddings is None and embedding_model_name:
            from memotrix.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model=embedding_model_name)
        if embeddings is None:
            raise ConfigurationError(
                "IngestionPipeline requires embeddings=... "
                "(HuggingFaceEmbeddings(model=...) or FakeEmbeddings(dim=...))"
            )
        self.embeddings = embeddings
        self.embedding_model_name = getattr(embeddings, "model_name", embedding_model_name or "")
        self._encoder = encoder
        self.document_store = (
            document_store if document_store is not None else DocumentStore()
        )
        self.dedup_threshold = dedup_threshold
        self.enable_dedup = enable_dedup
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self.embeddings
        return self._encoder

    def ingest(self, doc: DocumentData, source_path: str | Path | None = None) -> None:
        """Process a single document and add its chunks to the indexes."""
        self.ingest_many([(doc, source_path)])

    def ingest_many(
        self,
        items: Sequence[tuple[DocumentData, str | Path | None]]
        | Sequence[DocumentData],
    ) -> dict:
        """
        Phase 6 — batch-embed across many files in one encoder forward pass.
        Also applies Phase 5 orphan cleanup and Phase 4 near-dupe merge.
        """
        prepared: List[tuple[DocumentData, str | None]] = []
        for item in items:
            if isinstance(item, DocumentData):
                prepared.append((item, None))
            else:
                doc, source_path = item
                prepared.append((doc, str(source_path) if source_path else None))

        from memotrix.utils.trace import mlog

        mlog("ingest", f"ingest_many start: {len(prepared)} document(s)")

        all_chunks: List[Chunk] = []
        sources_cleaned: set[str] = set()

        for doc, source_path in prepared:
            if source_path is not None:
                path = Path(source_path)
                try:
                    resolved = str(path.resolve()) if path.exists() else str(source_path)
                except OSError:
                    resolved = str(source_path)
                doc.metadata["source_path"] = resolved

            key = document_key(doc.metadata)
            # Drop prior chunks and adjacency before registering the new text.
            if key and key not in sources_cleaned:
                self._delete_source(key)
                sources_cleaned.add(key)

            self.document_store.register(key, doc.text)
            mlog(
                "ingest",
                f"chunking {Path(source_path).name if source_path else key or '?'} "
                f"text_chars={len(doc.text or '')} "
                f"images={len(getattr(doc, 'images', None) or [])} "
                f"tables={len(getattr(doc, 'tables', None) or [])}",
            )

            chunks = chunk_document(
                doc, chunk_size=self.chunk_size, overlap=self.chunk_overlap
            )
            mlog("ingest", f"  → {len(chunks)} chunk(s)")
            for chunk in chunks:
                stamp_new_memory_payload(chunk.payload)
                if "memory_type" in doc.metadata:
                    chunk.payload["memory_type"] = doc.metadata["memory_type"]
                if "session_id" in doc.metadata:
                    chunk.payload["session_id"] = doc.metadata["session_id"]
                self.document_store.register_chunk(chunk.payload)
                all_chunks.append(chunk)

        if not all_chunks:
            mlog("ingest", "ingest_many done: no chunks produced")
            return {"chunks": 0, "merged": 0, "inserted": 0}

        texts = [c.text for c in all_chunks]
        # Phase 6 — one batched encode for the whole ingest set.
        vectors = self.embeddings.embed_documents(texts)

        dense_ids: List[str] = []
        dense_vectors: List[List[float]] = []
        dense_payloads: List[dict] = []
        sparse_ids: List[str] = []
        sparse_texts: List[str] = []
        sparse_payloads: List[dict] = []
        merged = 0

        for chunk, vector in zip(all_chunks, vectors):
            vec = list(vector)
            payload = dict(chunk.payload)
            chunk_id = chunk.id

            if self.enable_dedup and chunk.is_dense_indexable:
                merged_id = self._try_merge_duplicate(vec, payload, chunk_id)
                if merged_id is not None:
                    merged += 1
                    continue

            if chunk.is_dense_indexable:
                dense_ids.append(chunk_id)
                dense_vectors.append(vec)
                dense_payloads.append(payload)
            if chunk.is_sparse_indexable:
                sparse_ids.append(chunk_id)
                sparse_texts.append(chunk.text)
                sparse_payloads.append(payload)

        if dense_ids:
            self.dense_index.add(dense_ids, dense_vectors, dense_payloads)
        if sparse_ids:
            self.sparse_index.add(sparse_ids, sparse_texts, sparse_payloads)

        stats = {
            "chunks": len(all_chunks),
            "merged": merged,
            "inserted": len(dense_ids),
        }
        mlog(
            "ingest",
            f"ingest_many done: chunks={stats['chunks']} "
            f"inserted={stats['inserted']} merged={stats['merged']}",
        )
        return stats

    def _delete_source(self, source_path: str) -> None:
        self.dense_index.delete_by_source_path(source_path)
        self.sparse_index.delete_by_source_path(source_path)
        self.document_store.delete_source(source_path)

    def _try_merge_duplicate(
        self, vector: List[float], payload: dict, chunk_id: str
    ) -> str | None:
        """
        Phase 4 — if an existing chunk is near-identical (cosine > threshold),
        merge into it (boost importance) instead of inserting a duplicate.
        Only merge within the same source so sessions/files stay isolated.
        """
        hits = self.dense_index.search(vector, k=1)
        if not hits:
            return None

        existing_id, score, existing_payload = hits[0]
        if score < self.dedup_threshold:
            return None
        if existing_id == chunk_id:
            return None

        new_source = payload.get("source_path") or payload.get("filename")
        old_source = existing_payload.get("source_path") or existing_payload.get("filename")
        if new_source != old_source:
            return None

        new_text = payload.get("chunk_text", "")
        old_text = existing_payload.get("chunk_text", "")
        merged_payload = dict(existing_payload)
        merged_payload["importance"] = float(merged_payload.get("importance") or 1.0) + 1.0
        merged_payload["access_count"] = int(merged_payload.get("access_count") or 0)
        if len(new_text) > len(old_text):
            merged_payload["chunk_text"] = new_text
        for key in ("session_id", "memory_type"):
            if payload.get(key):
                merged_payload[key] = payload[key]

        self.dense_index.update_payload(existing_id, merged_payload)
        self.sparse_index.update_payload(existing_id, merged_payload)
        if len(new_text) > len(old_text):
            self.dense_index.add([existing_id], [vector], [merged_payload])
            self.sparse_index.add([existing_id], [new_text], [merged_payload])

        return existing_id
