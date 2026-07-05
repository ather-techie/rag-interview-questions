# Pattern: Fan-Out / Fan-In (Parallel Retrieval + Merge)

> Issue multiple independent retrieval queries in parallel, then merge and deduplicate the results before generation.

---

## Problem

A single retrieval query from a complex question often misses important aspects. Consider:

> "Compare the trade-offs between HNSW and IVF indexing for a 500M-document corpus."

A single embedding of this query is a compromise between "HNSW", "IVF", "trade-offs", and "500M documents." Dense retrieval returns the vector closest to that centroid — but the best passages for *HNSW-specific* trade-offs and the best passages for *IVF-specific* trade-offs may be in very different parts of the index, neither of which is closest to the composite query vector.

Fan-out decomposes the query into focused sub-queries, retrieves for each in parallel, then merges.

---

## Architecture

```
Original Query: "Compare HNSW vs IVF for 500M documents"
      │
      ▼
┌─────────────────────────────────────────┐
│  Query Decomposer (1 LLM call, Haiku)   │
└──────────────────┬──────────────────────┘
                   │ generates N sub-queries
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
 "HNSW trade-  "IVF trade-  "vector index
  offs large    offs large   500M docs
  scale"        scale"       scaling"
      │            │            │
      ▼            ▼            ▼
 Retrieve k=5  Retrieve k=5  Retrieve k=5
      │            │            │
      └────────────┼────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  Merge + Dedup   │  ← RRF or cross-encoder rerank
         │  (fan-in step)   │
         └────────┬─────────┘
                  │ top-k unique passages
                  ▼
              Generate
```

---

## Implementation

```python
import asyncio
import anthropic
import json

client = anthropic.Anthropic()

DECOMPOSE_PROMPT = """Break the following question into 2-4 focused sub-queries that each target a specific aspect.
Each sub-query should be independently retrievable.
Output JSON array: ["sub-query 1", "sub-query 2", ...]"""

def decompose_query(query: str) -> list[str]:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=DECOMPOSE_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    sub_queries = json.loads(resp.content[0].text)
    # Always include the original query
    return [query] + sub_queries


async def retrieve_async(query: str, vector_db, k: int = 5) -> list[dict]:
    """Async wrapper around synchronous vector DB search."""
    results = await asyncio.to_thread(vector_db.search, query, k)
    for r in results:
        r["sub_query"] = query  # track which sub-query found this
    return results


async def fan_out_retrieve(
    query: str,
    vector_db,
    k_per_query: int = 5,
    k_final: int = 7,
) -> list[dict]:
    sub_queries = decompose_query(query)
    
    # Fan-out: retrieve for all sub-queries in parallel
    all_results = await asyncio.gather(*[
        retrieve_async(q, vector_db, k_per_query)
        for q in sub_queries
    ])
    
    # Fan-in: deduplicate and merge with RRF
    flat = [r for results in all_results for r in results]
    merged = rrf_merge_with_dedup(flat, k=k_final)
    
    return merged


def rrf_merge_with_dedup(results: list[dict], k: int = 7, rrf_k: int = 60) -> list[dict]:
    """Merge results from multiple sub-queries using RRF; deduplicate by doc_id."""
    # Group results by doc_id and accumulate RRF score from all sub-queries
    scores = {}
    
    # Sort within each sub-query group by score (higher = better rank)
    sub_query_results = {}
    for r in results:
        sq = r["sub_query"]
        sub_query_results.setdefault(sq, []).append(r)
    
    for sq, sq_results in sub_query_results.items():
        sq_results.sort(key=lambda x: x["score"], reverse=True)
        for rank, r in enumerate(sq_results):
            doc_id = r["doc_id"]
            scores.setdefault(doc_id, {"doc": r, "rrf": 0.0})
            scores[doc_id]["rrf"] += 1.0 / (rrf_k + rank + 1)
    
    merged = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [item["doc"] for item in merged[:k]]
```

---

## When to Fan-Out

Fan-out is justified when:

| Query Type | Single Retrieve? | Fan-Out? |
|-----------|-----------------|---------|
| Single concept ("what is HNSW?") | ✓ sufficient | Over-engineering |
| Multi-aspect comparison ("HNSW vs IVF") | Sometimes | ✓ better recall |
| Synthesis ("best practices for large-scale RAG") | Often misses aspects | ✓ better coverage |
| Entity lookup ("what is doc ID 1234?") | ✓ sufficient | Wasteful |

Use the router pattern to decide whether to fan-out: the router classifies "compare / synthesize / multi-aspect" queries and the fan-out kicks in only for those.

---

## RAG-Fusion as a Special Case

RAG-Fusion (architecture #18) is fan-out with LLM-generated query variants (not sub-queries):

```
Original: "How does HNSW scale?"
                │
                ▼ (RAG-Fusion: rephrase, not decompose)
  "HNSW scalability properties"
  "hierarchical navigable small world performance"
  "HNSW memory and latency at scale"
```

Fan-out decomposes into *aspects*; RAG-Fusion rephrases the *same question* in multiple ways to improve recall of paraphrased passages. Use RAG-Fusion for single-concept queries where phrasing diversity matters; use Fan-out for genuinely multi-aspect queries.

---

## Key Takeaways

1. **Fan-out improves recall** for multi-aspect queries where a single query vector is a centroid between aspects.
2. **Deduplicate before generation** — without dedup, the LLM sees the same passage 3× and over-weights it.
3. **RRF is the right merge strategy** — it handles different score scales from different sub-queries.
4. **Always include the original query** as one of the sub-queries — the decomposer may miss the holistic intent.
5. **Cap sub-query count at 4** — beyond 4, retrieval latency grows without proportional recall improvement.
