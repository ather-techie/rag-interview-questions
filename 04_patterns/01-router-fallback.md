# Pattern: Router + Fallback

> Route each query to the cheapest retriever that can handle it; fall back to progressively more powerful (and expensive) retrievers only when cheaper ones fail.

---

## Problem

A single retrieval strategy cannot be optimal for all query types. Dense semantic search handles paraphrase and concept queries well but fails on exact-match lookups (product codes, names, citations). BM25 handles exact match but misses semantic intent. Agentic loops handle complex multi-hop queries but add 5–10× latency and cost for simple queries that a single retrieval call could answer.

Without routing, every query pays the cost of the most expensive path even when a cheaper one would suffice.

---

## Solution: Router + Fallback Chain

```
Query
  │
  ▼
┌─────────────────┐
│  Query Router   │  ← classifies query type (fast, cheap)
└────────┬────────┘
         │
   ┌─────┴──────────────────────────────────────┐
   │                                            │
   ▼                                            ▼
TIER 1: Semantic Cache            TIER 2: Dense Retrieval
(hit → return immediately)        (most queries land here)
   │                                            │
   │ cache miss                                 │ low confidence
   ▼                                            ▼
TIER 3: Hybrid (dense + BM25)       TIER 4: Agentic Loop
(exact match + semantic)            (complex multi-hop only)
```

---

## Router Implementation

```python
import anthropic
import json

client = anthropic.Anthropic()

ROUTER_PROMPT = """Classify the user's query into exactly one category:
- "cache_lookup": greeting, trivial, or recently asked
- "exact_match": looking for a specific identifier, name, code, or citation
- "semantic": conceptual question answerable from a single passage
- "hybrid": requires both keyword precision and semantic understanding
- "agentic": requires multi-step reasoning, comparison across many documents, or synthesis

Output JSON: {"type": "...", "confidence": 0.0-1.0}"""

def route_query(query: str) -> dict:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        system=ROUTER_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    return json.loads(resp.content[0].text)


def retrieve_with_fallback(query: str, semantic_cache, vector_db, bm25_index, agent_fn) -> list[dict]:
    route = route_query(query)
    query_type = route["type"]
    
    # Tier 1: semantic cache
    cached = semantic_cache.get(query)
    if cached:
        return cached
    
    # Tier 2: pure dense (most queries)
    if query_type in ("semantic",):
        results = vector_db.search(query, k=5)
        if results and results[0]["score"] > 0.75:
            return results
        # Score too low → fall through to hybrid
        query_type = "hybrid"
    
    # Tier 3: hybrid dense + BM25
    if query_type in ("exact_match", "hybrid"):
        dense_results = vector_db.search(query, k=10)
        bm25_results  = bm25_index.search(query, k=10)
        results = rrf_merge(dense_results, bm25_results, k=5)
        if results and results[0]["score"] > 0.5:
            return results
        query_type = "agentic"
    
    # Tier 4: agentic loop (expensive — last resort)
    if query_type == "agentic":
        return agent_fn(query)
    
    return []
```

---

## Fallback Trigger Conditions

| Trigger | Condition | Action |
|---------|-----------|--------|
| Cache miss | Cosine similarity < threshold | Drop to dense retrieval |
| Low retrieval confidence | Top-1 score < 0.75 | Drop to hybrid |
| Empty results | len(results) == 0 | Drop to next tier |
| Query complexity signal | Router returns "agentic" | Skip to agentic tier directly |
| Retrieval timeout | Response > 500ms | Return cached degraded result |

---

## Cost and Latency by Tier

| Tier | Latency | Cost per query | When Used |
|------|---------|----------------|-----------|
| Semantic cache | <10ms | ~$0.000001 | ~20% of queries |
| Dense only | 30–100ms | ~$0.001 | ~60% of queries |
| Hybrid | 100–300ms | ~$0.003 | ~15% of queries |
| Agentic loop | 2–10s | ~$0.05–$0.20 | ~5% of queries |

---

## Anti-patterns

- **Routing to agentic by default**: even a 10% agentic rate with $0.10/query = $10K/month at 1M queries/month.
- **No fallback from cache**: a stale cache answer for a fresh query erodes trust fast.
- **Over-routing to hybrid**: hybrid is only better than dense for keyword-sensitive queries (~15–20% of typical traffic).
- **Routing on query length alone**: long queries are not necessarily complex; short queries are not necessarily simple.

---

## Interview Q&A

**Q: How would you design a query router for a production RAG system?**

A practical router has three components: (1) a rule-based pre-filter for deterministic cases (explicit entity codes → exact match tier; greetings → cache); (2) a lightweight classifier (Haiku or a fine-tuned small model) for semantic classification of query intent; (3) confidence-based promotion — even if the router says "semantic," if the retrieved results score poorly, fall through to the next tier automatically. The router itself should add <50ms latency and cost <$0.0001 per call to remain net-positive. Track "tier distribution" as a KPI — if the agentic tier grows above 10% of traffic, investigate whether the router is miscategorizing or the query mix has genuinely shifted.
