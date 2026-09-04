# How to Migrate to PostgreSQL (pgvector)

Memotrix uses an in-memory `hnswlib` dense index and a `rank-bm25` sparse index by default. This is perfect for prototyping, but all memory is wiped when your Python script exits.

For persistent, scalable, and ACID-compliant storage, Memotrix natively supports PostgreSQL with the `pgvector` extension.

## 1. Prerequisites

You need the `postgres` optional dependency installed:

```bash
pip install "memotrix[postgres]"
```

You also need a PostgreSQL database with the `pgvector` extension installed. If you have Docker, you can run the provided `docker-compose.yml`:

```bash
cd memory/
docker compose up -d
```

## 2. Initialize Memory with Postgres

There are two ways to tell Memotrix to use PostgreSQL: using configuration variables or directly passing a connection string.

### Method A: Environment Variables (Recommended)

Set the `DATABASE_URL` environment variable:
```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/memotrix"
```

Initialize `Memory` and let it auto-detect the backend:
```python
import os
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")

# Memotrix will automatically use PostgresStore because connection= is provided
memory = Memory(
    embeddings=embeddings, 
    connection=os.environ["DATABASE_URL"]
)
```

### Method B: Explicit Initialization

You can explicitly pass a `PostgresStore` instance. This is useful if you need to override the default table name (`memotrix_chunks`).

```python
import os
from memotrix import Memory
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.vectorstores import PostgresStore

embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")

store = PostgresStore(
    connection=os.environ["DATABASE_URL"],
    embeddings=embeddings,
    table_name="my_custom_chunks_table"
)

memory = Memory(embeddings=embeddings, store=store)
```

## 3. What Happens Under the Hood?

When you initialize a `PostgresStore`:
1. Memotrix connects to Postgres and ensures the `pgvector` extension is enabled.
2. It creates the chunk table (if it doesn't exist) with a `vector` column sized exactly to your `embeddings.dimension`.
3. It creates an HNSW index on the vector column.
4. It creates a `tsvector` index for sparse keyword search (BM25 approximation).

> [!WARNING]
> If you change your embedding model to one with a different dimension (e.g., switching from `bge-small` [384] to `openai` [1536]), you MUST use a new `table_name` or drop the old table. Postgres cannot alter the dimension of an existing `vector` column without wiping the data.
