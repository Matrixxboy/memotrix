"""File ingest into Postgres. Requires DATABASE_URL and an embedding model.

    pip install -e ./memory[postgres,local,embeddings,extractors]
    set DATABASE_URL=postgresql://user:pass@localhost:5432/memotrix
    set EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
"""

import os
from pathlib import Path

from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings, HuggingFaceEmbeddings

example_file = Path(__file__).with_name("sample.txt")
if not example_file.exists():
    example_file.write_text(
        "Q3 revenue was 42 million dollars. The north region led growth.\n",
        encoding="utf-8",
    )

if os.getenv("EMBEDDING_MODEL"):
    embeddings = HuggingFaceEmbeddings(model=os.environ["EMBEDDING_MODEL"])
else:
    embeddings = FakeEmbeddings(dim=8)

dsn = os.getenv("DATABASE_URL")
if dsn:
    memory = Memory(embeddings=embeddings, backend="postgres", connection=dsn)
else:
    memory = Memory(embeddings=embeddings, backend="memory")

print(memory.add(example_file))
print(memory.search("what is the revenue?", top_k=3))
print(memory.list_sources())
memory.close()
