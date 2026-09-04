# How to Tune Hybrid Search

Memotrix performs **Hybrid Search** by default. This means every query is executed against two different indexes:
1. **Dense Index (Vectors):** Finds chunks with similar semantic meaning (e.g., "puppy" matches "dog").
2. **Sparse Index (Keywords):** Finds chunks containing exact keyword matches (e.g., "RX-78-2" matches exactly "RX-78-2").

Memotrix fuses these two sets of results using **Reciprocal Rank Fusion (RRF)**.

## Tuning Reciprocal Rank Fusion

The `fusion_k` parameter controls the RRF formula:
`Score = 1 / (fusion_k + rank)`

By default, Memotrix uses `fusion_k = 60`, which is the industry standard recommendation. However, you can tune this via the `RetrievalConfig`.

```python
from memotrix import Memory, MemoryConfig
from memotrix.config import RetrievalConfig

config = MemoryConfig(
    backend="memory",
    embedding_model="BAAI/bge-small-en-v1.5",
    retrieval=RetrievalConfig(
        top_k=5, 
        fusion_k=100  # Give less weight to the absolute top rankers
    )
)

memory = Memory.from_config(config)
```

## Tuning Memory Boosts

Memotrix applies a multiplier to the final RRF score based on how the memory is accessed. This is called the `memory_multiplier`.

It boosts chunks that:
1. Were created recently.
2. Have been accessed frequently (`access_count`).
3. Have been flagged as important during duplicate merging (`importance`).

If you want pure relevance scores without temporal or access bias, you can disable this:

```python
config = MemoryConfig(
    retrieval=RetrievalConfig(
        enable_memory_boost=False
    )
)
```

## Adding a Cross-Encoder Reranker

RRF is a mathematical fusion technique. For the absolute highest precision, you should pass the fused results through a **Cross-Encoder Reranker**. 
The reranker reads both the query and the chunk text and outputs a highly accurate relevance score.

Memotrix supports this out of the box. Just provide a `reranker_model`:

```python
config = MemoryConfig(
    backend="memory",
    embedding_model="BAAI/bge-small-en-v1.5",
    retrieval=RetrievalConfig(
        enable_rerank=True,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
)
memory = Memory.from_config(config)
```

> [!TIP]
> Rerankers are extremely slow compared to vector search. Memotrix automatically skips the reranker if the top result from both the Dense and Sparse indexes agree, saving significant latency. This conditional logic is enabled by default.
