# Chunking and Neighbor Expansion

When you ingest a massive document into Memotrix, it must be broken down into smaller pieces so that the embedding models and LLMs can process it.

## The Problem with Naive Chunking
If you split a document into 500-character blocks, you inevitably cut sentences in half. 
For example:
- **Chunk A:** "The user's password is "
- **Chunk B:** "hunter2."

If the user asks "What is my password?", the vector search will likely match **Chunk A**, but the LLM won't be able to answer the question because the actual value is in **Chunk B**.

## Overlapping
Memotrix solves this at the ingestion layer using **Overlap**.
By default, chunks are 500 characters long with a 50-character overlap. This means Chunk B contains the last 50 characters of Chunk A, bridging the context gap.

## Neighbor Expansion (The Document Store)
Even with overlapping, a 500-character chunk might lack the necessary context to fully answer a complex question.

Memotrix uses a unique **DocumentStore** architecture to solve this. 
When a document is ingested, Memotrix records the *adjacency* of every chunk. It knows that Chunk 2 came immediately after Chunk 1 in the original file.

During retrieval, Memotrix finds the best matching chunks in the vector index, and then **expands** those chunks by grabbing their neighbors from the DocumentStore.

### Expand Modes

You can control this behavior via the `expand_mode` parameter in `RetrievalConfig` (or passed directly to `search()`):

1. **`neighbors` (Default):**
   Memotrix expands the matched chunk to include its immediate neighbors (± `neighbor_window`, default is 1). It dynamically expands until it hits the `max_tokens_returned` limit. This gives the LLM the exact matched fact, plus the surrounding paragraphs for context.

2. **`full_file`:**
   Memotrix takes the matched chunks and returns the *entire original text* of the files those chunks came from. This is useful for "Summarize this document" queries, but it is highly likely to exceed the LLM's context window for large files.

3. **`none`:**
   Memotrix returns exactly what was stored in the vector index. No expansion occurs. Use this if your chunks are already massive, or if you are retrieving highly structured data (like single-sentence JSON logs) where neighbors are irrelevant.
