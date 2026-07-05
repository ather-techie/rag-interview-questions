# Pattern: Naive → Advanced → Modular → Agentic Migration Path

> How to upgrade a RAG system incrementally without a full rewrite, with specific forcing functions that justify each upgrade.

---

## The Core Principle

Don't build the most complex system upfront. Build the simplest system that meets your current requirements, measure where it fails, and upgrade only the component that is actually failing. Each upgrade has a specific *forcing function* — a metric that justifies the added complexity and cost.

```
Naive RAG ──► Advanced RAG ──► Modular RAG ──► Agentic RAG
    │               │               │               │
    │           Add when:       Add when:       Add when:
    │           Retrieval       One strategy    Query scope
    │           recall < 0.7    underperforms   is unknown
    │           for some query  on some query   at query time
    │           types           types
    │
  Start here
```

---

## Stage 1: Naive RAG

**What it is:** Chunk → embed → store → retrieve top-k → generate. No query transformation, no reranking.

**Build it when:** Starting a new system, unknown query distribution, <1M documents, latency SLA < 500ms.

```python
def naive_rag(query: str, vector_db, llm) -> str:
    chunks = vector_db.search(query, k=5)
    context = "\n\n".join(c["text"] for c in chunks)
    return llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
```

**Measure:** Recall@5 on a 50-query probe set. Deploy evaluation in CI.

**Forcing function to advance:** Recall@5 < 0.70 on your probe set after tuning chunk size and embedding model.

---

## Stage 2: Advanced RAG

**What it adds over Naive:** Query rewriting, hybrid search (dense + BM25), reranker.

```python
def advanced_rag(query: str, vector_db, bm25_index, reranker, llm) -> str:
    # Query transformation
    rewritten = rewrite_query(query)  # HyDE or simple rewrite
    
    # Hybrid retrieval
    dense_results = vector_db.search(rewritten, k=20)
    bm25_results  = bm25_index.search(rewritten, k=20)
    merged        = rrf_merge(dense_results, bm25_results, k=20)
    
    # Reranking
    reranked = reranker.rerank(rewritten, merged, top_k=5)
    
    context  = "\n\n".join(c["text"] for c in reranked)
    return llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
```

**Cost added:** ~$0.001/query (reranker call + extra retrieval).

**Forcing functions:**
- Recall@5 is low for *exact match queries* (product codes, names) → add BM25 hybrid
- Recall@5 is good but Faithfulness is low → add reranker (retrieves correct but less precisely relevant chunks)
- Query phrasing diverges from document phrasing → add HyDE or query rewriting

---

## Stage 3: Modular RAG

**What it adds over Advanced:** Multiple pluggable retrieval strategies, routed by query type.

```python
STRATEGY_MAP = {
    "semantic":   lambda q: vector_db.search(q, k=5),
    "exact_match": lambda q: bm25_index.search(q, k=5),
    "hybrid":     lambda q: rrf_merge(vector_db.search(q, k=10), bm25_index.search(q, k=10), k=5),
    "graph":      lambda q: graph_db.traverse(q, max_hops=2),
    "sql":        lambda q: text_to_sql(q, db_schema),
}

def modular_rag(query: str, llm) -> str:
    strategy_name = router.classify(query)  # "semantic" / "hybrid" / "graph" / "sql"
    chunks        = STRATEGY_MAP[strategy_name](query)
    context       = "\n\n".join(c["text"] for c in chunks)
    return llm.generate(f"Context:\n{context}\n\nQuestion: {query}")
```

**Cost added:** Router call + strategy-specific retrieval costs.

**Forcing functions:**
- Some query *types* (structured, graph traversal) always fail with semantic search → add specialized retrievers
- Different content *sources* need different retrieval (a DB + document corpus) → route by query and source
- You have >3 distinct retrieval strategies worth combining → the routing logic is complex enough to justify a Modular pattern

---

## Stage 4: Agentic RAG

**What it adds over Modular:** The LLM decides the retrieval strategy, can retry, and can chain multiple hops.

```python
def agentic_rag(query: str, max_iterations: int = 5) -> str:
    return run_react_agent(
        query=query,
        tools=[semantic_retrieve, hybrid_retrieve, graph_retrieve, web_search],
        max_iterations=max_iterations,
    )
```

**Cost added:** 3–10× more LLM calls; $0.05–$0.20/query.

**Forcing functions:**
- Queries require unknown number of retrieval hops (the retrieval scope is dynamically determined)
- Retrieval must branch conditionally based on what was found (e.g., "if the document mentions X, also retrieve Y")
- The system needs to recover from retrieval failures (re-query with different terms if first attempt fails)

---

## Decision Matrix

| Signal | Stage to Add |
|--------|-------------|
| Recall@5 < 0.70 overall | Naive → Advanced |
| Exact match queries fail (product codes, names) | Add BM25 hybrid |
| Faithfulness low despite good recall | Add reranker |
| Some query categories always fail | Advanced → Modular |
| Different data sources need different retrieval | Advanced → Modular |
| Multi-hop questions fail | Modular → Agentic |
| Query scope unknown at design time | Modular → Agentic |

---

## What NOT to Do

- **Don't skip stages**: going from Naive to Agentic without measuring whether Advanced RAG solves your problems first adds cost and complexity for unknown benefit.
- **Don't add a reranker before measuring recall**: if recall is already 0.95, a reranker won't help — you're paying for nothing.
- **Don't add agentic loops for simple queries**: 60% of production RAG traffic is simple factual lookups that Naive RAG answers correctly.
- **Don't use the same chunk size after adding reranking**: reranking tolerates larger initial pools (k=20–50) — increase k when you add a reranker.

---

## Interview Q&A

**Q: How would you upgrade a production Naive RAG system to handle complex multi-hop queries without rewriting everything?**

Incrementally. First add a query classifier that detects multi-hop queries (signals: "compare", "what are all the ways", "across all documents"). Route those queries to an agentic sub-path while keeping Naive RAG for simple queries. The agentic sub-path can start as just two serial retrievals with a query reformulation step between them — not a full ReAct agent. Measure if that solves the failing cases before adding more complexity. The key discipline: add one component at a time, measure its impact on your probe set, and only keep it if it moves the metric.
