"""11 — PostgresStore when DATABASE_URL is set."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings
from memotrix.vectorstores import PostgresStore

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main() -> None:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("SKIP: set DATABASE_URL to run Postgres demo")
        return
    embeddings = FakeEmbeddings(dim=8)
    memory = Memory(
        embeddings=embeddings,
        store=PostgresStore(
            connection=dsn,
            embeddings=embeddings,
            table_name="memotrix_demolive",
        ),
    )
    print(memory.add(SAMPLES / "notes.txt"))
    print(memory.search("revenue", top_k=1))
    memory.close()


if __name__ == "__main__":
    main()
