"""Embedding interfaces — dimension is always inferred from the model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from memotrix.utils.exceptions import ConfigurationError
from memotrix.vectorDB.embeddings import (
    format_query_for_embedding,
    get_sentence_transformer,
)


class Embeddings(ABC):
    """LangChain-style embedding interface used by Memory and vector stores."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width reported by the loaded model — never a hardcoded constant."""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return type(self).__name__


class HuggingFaceEmbeddings(Embeddings):
    """
    Sentence-Transformers embeddings.

    Dimension is read from the model (``get_sentence_embedding_dimension``),
    not from a config constant.
    """

    def __init__(
        self,
        model: str,
        *,
        query_instruction: Optional[str] = None,
    ) -> None:
        if not model or not str(model).strip():
            raise ConfigurationError("HuggingFaceEmbeddings requires an explicit model name")
        self._model_name = str(model).strip()
        self._query_instruction = query_instruction
        self._client = None
        self._dimension: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def client(self):
        if self._client is None:
            self._client = get_sentence_transformer(self._model_name)
        return self._client

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            dim_fn = getattr(self.client, "get_sentence_embedding_dimension", None)
            if callable(dim_fn):
                self._dimension = int(dim_fn())
            else:
                probe = self.embed_query("dimension-probe")
                self._dimension = len(probe)
            if self._dimension <= 0:
                raise ConfigurationError(
                    f"Embedding model {self._model_name!r} reported invalid dimension {self._dimension}"
                )
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.client.encode(texts, convert_to_numpy=True, batch_size=64)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        if self._query_instruction:
            payload = f"{self._query_instruction}{text}"
        else:
            payload = format_query_for_embedding(text, self._model_name)
        return self.client.encode(payload, convert_to_numpy=True).tolist()

    @classmethod
    def from_env(cls) -> "HuggingFaceEmbeddings":
        from memotrix.config import require_env

        return cls(model=require_env(
            "EMBEDDING_MODEL",
            hint="Example: EMBEDDING_MODEL=BAAI/bge-small-en-v1.5",
        ))


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for unit tests — dimension is always explicit."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ConfigurationError("FakeEmbeddings dim must be a positive integer")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"fake/{self._dim}"

    def _vector(self, text: str) -> List[float]:
        seed = sum(ord(ch) for ch in text) or 1
        return [((seed * (i + 1)) % 97) / 97.0 for i in range(self._dim)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class OpenAIEmbeddings(Embeddings):
    """
    OpenAI embeddings. Dimension is inferred from the first API response.

    Requires the ``openai`` extra: ``pip install memotrix[openai]``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
    ) -> None:
        if not model or not str(model).strip():
            raise ConfigurationError("OpenAIEmbeddings requires an explicit model name")
        self._model_name = str(model).strip()
        self._api_key = (api_key or "").strip() or None
        self._client = None
        self._dimension: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ConfigurationError(
                    "OpenAIEmbeddings requires the openai package. "
                    "Install with: pip install memotrix[openai]"
                ) from exc
            import os

            key = self._api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ConfigurationError(
                    "OpenAIEmbeddings requires api_key=... or OPENAI_API_KEY"
                )
            self._client = OpenAI(api_key=key)
        return self._client

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed_query("dimension-probe")
            self._dimension = len(probe)
            if self._dimension <= 0:
                raise ConfigurationError(
                    f"Embedding model {self._model_name!r} reported invalid dimension {self._dimension}"
                )
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self._model_name, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_documents([text])
        return vectors[0]


def coerce_embeddings(embeddings: Embeddings | str) -> Embeddings:
    if isinstance(embeddings, Embeddings):
        return embeddings
    if isinstance(embeddings, str) and embeddings.strip():
        return HuggingFaceEmbeddings(model=embeddings)
    raise ConfigurationError(
        "embeddings must be an Embeddings instance or a non-empty model name string"
    )
