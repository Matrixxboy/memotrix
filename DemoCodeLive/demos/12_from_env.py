"""12 — Memory.from_env() when EMBEDDING_MODEL is set."""

from __future__ import annotations

import os

from memotrix import Memory


def main() -> None:
    if not os.getenv("EMBEDDING_MODEL"):
        print("SKIP: set EMBEDDING_MODEL (and DATABASE_URL if MEMOTRIX_BACKEND=postgres)")
        return
    memory = Memory.from_env()
    memory.add_text("from_env demo fact", source_id="from-env-demo")
    print(memory.search("demo fact", top_k=1))
    memory.delete("from-env-demo")
    memory.close()


if __name__ == "__main__":
    main()
