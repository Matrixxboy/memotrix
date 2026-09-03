import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memotrix.config import resolve_database_url
from memotrix.embeddings import HuggingFaceEmbeddings
from memotrix.processes.document_store import DocumentStore
from memotrix.processes.retrieval import RetrievalPipeline
from memotrix.utils.exceptions import ConfigurationError
from memotrix.vectorDB.hybrid import HybridSearchEngine
from memotrix.vectorDB.postgres_store import create_postgres_indexes


def check_queries(user_query=None):
    if user_query:
        queries = [user_query]
    else:
        query_file = Path(__file__).parent / "queries_to_check.txt"
        if not query_file.exists():
            print(f"Query file not found: {query_file}")
            return

        queries = [line.strip() for line in query_file.read_text().splitlines() if line.strip()]
        if not queries:
            print("No queries found in the file.")
            return

    print("Connecting to PostgreSQL...")
    try:
        embeddings = HuggingFaceEmbeddings.from_env()
        dense_index, sparse_index, pg_store = create_postgres_indexes(
            connection=resolve_database_url(),
            dim=embeddings.dimension,
        )
        document_store = DocumentStore()
        hybrid_engine = HybridSearchEngine(dense_index, sparse_index)
        retrieval = RetrievalPipeline(
            hybrid_engine=hybrid_engine,
            embeddings=embeddings,
            document_store=document_store,
        )

        for query in queries:
            print("\n======================================")
            print(f"Querying: '{query}'")
            results = retrieval.retrieve(query, top_k=3)

            print(f"Found {len(results)} results:")
            for i, res in enumerate(results):
                print(f"  Rank {i+1}: Score={res.get('score', 0):.4f} File={res.get('filename')}")
                print(f"  Text: {res.get('chunk_text', '')[:200]}...\n")
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return
    finally:
        try:
            pg_store.close()
        except Exception:
            pass


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    check_queries(q)
