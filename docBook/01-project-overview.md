---
title: "Project Overview"
category: "Overview"
status: "Generated"
last_updated: "2026-09-04"
---

# Project Overview

## Overview
**Memotrix** is a composable hybrid memory and Retrieval-Augmented Generation (RAG) framework designed for building intelligent AI agents. It operates as a pip-installable library akin to LangChain, providing developers with powerful tools to integrate both dense (vector) and sparse (keyword) search capabilities.

## Purpose
The primary purpose of Memotrix is to provide AI agents with a robust, scalable, and easy-to-use memory system. It abstracts away the complexity of managing vector databases, hybrid search algorithms, and document extraction, allowing developers to focus on building agent logic.

## Technical Details
Memotrix supports both local, in-memory operations using HNSW and BM25, as well as scalable backend storage using PostgreSQL with pgvector. It features a wide array of document extractors capable of parsing PDFs, Office documents, HTML, JSON, and even audio files.

### Why It Matters
AI agents require context to perform effectively. By providing a structured, hybrid memory system, Memotrix ensures that agents can retrieve relevant facts, past conversational turns, and procedural knowledge rapidly and accurately.
