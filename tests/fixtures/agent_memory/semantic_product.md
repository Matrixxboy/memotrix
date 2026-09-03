# Semantic Facts — Product Identity

## What Memotrix Is

Memotrix is a multimodal AI memory / RAG pipeline. It ingests real files, chunks them intelligently, and retrieves the best content using hybrid search plus reranking.

## What It Is Not

Memotrix is not a chat UI. It is not an in-memory toy index for production. It is designed as a persistent memory layer that other agents (like context-gate) call into.

## Success Metrics

Accuracy must be measured with an eval set, not vibes. Latency per query matters because agents call memory constantly inside their loops.
