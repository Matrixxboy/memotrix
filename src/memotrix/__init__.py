"""Memotrix — composable hybrid memory / RAG for agents.

Install::

    pip install memotrix[postgres,embeddings,extractors]

Quick start::

    from memotrix import Memory
    from memotrix.embeddings import HuggingFaceEmbeddings
    from memotrix.vectorstores import PostgresStore

    embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
    memory = Memory(
        embeddings=embeddings,
        store=PostgresStore(
            connection="postgresql://user:pass@localhost:5432/memotrix",
            embeddings=embeddings,
        ),
    )
    memory.add("report.pdf")
    memory.add_text("User prefers dark mode.", memory_type="semantic")
    print(memory.search("what is the revenue?"))
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "MemoryConfig",
    "Embeddings",
    "HuggingFaceEmbeddings",
    "FakeEmbeddings",
    "OpenAIEmbeddings",
    "PostgresStore",
    "InMemoryStore",
    "ConfigurationError",
    "build_retrieval_stack",
]


def __getattr__(name: str) -> Any:
    if name == "Memory":
        from memotrix.memory import Memory

        return Memory
    if name == "MemoryConfig":
        from memotrix.config import MemoryConfig

        return MemoryConfig
    if name in {"Embeddings", "HuggingFaceEmbeddings", "FakeEmbeddings", "OpenAIEmbeddings"}:
        from memotrix import embeddings as _embeddings

        return getattr(_embeddings, name)
    if name in {"PostgresStore", "InMemoryStore"}:
        from memotrix import vectorstores as _stores

        return getattr(_stores, name)
    if name == "ConfigurationError":
        from memotrix.utils.exceptions import ConfigurationError

        return ConfigurationError
    if name == "build_retrieval_stack":
        from memotrix.api import build_retrieval_stack

        return build_retrieval_stack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
