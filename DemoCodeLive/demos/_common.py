"""Shared Memory factory for DemoCodeLive.

Uses FakeEmbeddings so demos run without downloading models.
"""

from __future__ import annotations

from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings


def make_memory(*, dim: int = 16) -> Memory:
    return Memory(embeddings=FakeEmbeddings(dim=dim), backend="memory")
