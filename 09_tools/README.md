# 09 — Tools

> **Status:** In progress. The eval & observability layer is written; vector database and framework comparisons are still planned.

## Contents

| File | What It Covers |
|------|-----------------|
| [01-eval-observability-comparison.md](01-eval-observability-comparison.md) | Ragas vs. TruLens vs. DeepEval vs. LlamaIndex eval vs. LangChain eval — feature matrix, verified code snippets, decision criteria |

## What this section will cover

Opinionated, comparison-driven tool guides for each pipeline layer:

- **Vector databases** *(planned)* — benchmark-backed comparison (Pinecone, Weaviate, Qdrant, Milvus, pgvector) with selection criteria beyond marketing pages
- **Frameworks** *(planned)* — when LangChain/LlamaIndex/Haystack/LangGraph help and when they get in the way
- **Eval & observability** *(written — see Contents above)* — Ragas, TruLens, DeepEval, LlamaIndex eval, LangChain eval compared on the same scenario. LangSmith/Arize Phoenix/Langfuse tracing tools are covered in [`01_concepts/observability_and_evaluation_ops.md`](../01_concepts/observability_and_evaluation_ops.md) rather than duplicated here.

## Intended format

One comparison file per layer: feature matrix → decision criteria → migration notes → interview-relevant talking points.

## In the meantime

- Quick tool inventory by layer: [`cheatsheets/CHEATSHEET.md`](../cheatsheets/CHEATSHEET.md)
- Vector DB internals (HNSW, IVF, PQ): [`01_concepts/vector_databases.md`](../01_concepts/vector_databases.md)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md).
