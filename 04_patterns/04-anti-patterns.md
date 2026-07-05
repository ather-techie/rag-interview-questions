# Pattern: RAG Anti-Patterns

> The five most common design mistakes in RAG systems, why they hurt, and what to do instead.

---

## Anti-Pattern 1: Over-Chunking

**What it looks like:** Chunks set to 100–200 tokens because "smaller = more precise retrieval."

**Why it hurts:**
- Chunks too small lose sentence context: "The approach has three downsides." → which approach?
- Individual sentences have high variance in embedding quality
- A 100-token chunk retrieved from a 1000-token section misses the section's thesis
- Recall@k drops because relevant information spans multiple chunks but k is fixed

**The fix:** Start at 512 tokens with 10% overlap. Measure Recall@5. Only reduce if you find evidence that large chunks are diluting relevance with unrelated material.

```python
# Anti-pattern
chunks = split_by_tokens(document, chunk_size=128, overlap=0)

# Better
chunks = split_by_tokens(document, chunk_size=512, overlap=64)

# Even better: use semantic boundaries
chunks = split_by_paragraph_or_section(document, max_tokens=512)
```

---

## Anti-Pattern 2: Premature Agentification

**What it looks like:** Wrapping every RAG call in a ReAct loop because "agents are more powerful."

**Why it hurts:**
- 5–10× latency (every query makes 3–6 LLM calls instead of 1)
- 10–50× cost (multiple completions)
- Harder to debug (the error is 3 hops back from the wrong answer)
- Agents can loop indefinitely without guardrails

**The fix:** Measure Recall@5 and answer accuracy on a probe set. If Naive RAG already achieves 0.85+ recall and 0.80+ answer accuracy, don't add an agent. Agents are for cases where retrieval scope is *unknown* at query time — if you know the retrieval pattern, encode it in a static or modular system.

```
Decision: Do you need an agent?

Is the retrieval scope unknown at query time? ──► No ──► Don't use an agent.
Is the query multi-hop and the hops are conditional on retrieved content? ──► No ──► Don't use an agent.
Does the answer require comparing results across many separate retrievals? ──► No ──► Use fan-out instead.
                                                                              ──► Yes ──► Use an agent.
```

---

## Anti-Pattern 3: Reranking Everything

**What it looks like:** Adding a cross-encoder reranker to every retrieval call, regardless of query type.

**Why it hurts:**
- Cross-encoders add 200–800ms latency per call (sequential, not parallelizable)
- For queries where the top-1 dense result is already correct, reranking changes nothing
- Rerankers themselves can fail: they're trained on MS-MARCO, which doesn't cover all domains

**The fix:** Add reranking only when dense retrieval is *measurably* the bottleneck. Signs you need a reranker:

- Faithfulness metric is low despite high Recall@5 (correct document retrieved but at position 3–5, not 1)
- Query-document vocabulary divergence is high (documents use different terminology than queries)
- You retrieve k=20 to improve recall but want to compress back to k=5 for context quality

```python
# Anti-pattern: always rerank
def retrieve(query):
    results = vector_db.search(query, k=5)
    return reranker.rerank(query, results, top_k=5)  # wastes 400ms when k is already small

# Better: rerank only when initial pool is large
def retrieve(query, use_reranker=False):
    k_initial = 20 if use_reranker else 5
    results   = vector_db.search(query, k=k_initial)
    if use_reranker:
        return reranker.rerank(query, results, top_k=5)
    return results
```

---

## Anti-Pattern 4: One Index for All Tenants

**What it looks like:** Storing all tenants' documents in the same vector namespace with `metadata.tenant_id` filtering.

**Why it hurts:**
- **Performance**: metadata pre-filtering on a large index is slow at high QPS; post-filtering with `tenant_id` can return too few results after filtering
- **Security**: a misconfigured query without the `tenant_id` filter exposes all tenants' data
- **Recall degradation**: pre-filtering reduces the candidate pool for ANN search — a tenant with 1K documents in a 10M-document index gets poor ANN recall
- **Index growth**: one large index is harder to version, migrate, or roll back than per-tenant indexes

**The fix:** Per-tenant namespace isolation for different-sensitivity tenants; per-tier namespace isolation for same-sensitivity with different SLAs.

```python
# Anti-pattern: shared index with filter
def retrieve_naive(query, tenant_id):
    return vector_db.search(
        query,
        k=5,
        filter={"tenant_id": {"$eq": tenant_id}},  # dangerous if omitted
    )

# Better: per-tenant namespace
def retrieve_safe(query, tenant_id):
    return vector_db.search(
        query,
        k=5,
        namespace=f"tenant:{tenant_id}",  # server-side isolation, can't be omitted
    )
```

---

## Anti-Pattern 5: No Probe Set = Flying Blind

**What it looks like:** Deploying RAG changes (new chunking, new embedding model, new reranker) without a golden eval set.

**Why it hurts:**
- You can't measure whether a change improved or regressed quality
- Metrics like "we added a reranker" become marketing without measurement
- Regressions ship to production silently

**The fix:** Maintain a labeled probe set of 50–200 queries with known relevant passages. Run it in CI before every deployment. Block if Recall@5 drops more than 2%.

```python
# Minimum viable eval harness
def run_eval(retrieval_fn, probe_set: list[dict], k: int = 5) -> dict:
    recall_hits = 0
    for sample in probe_set:
        results    = retrieval_fn(sample["query"], k=k)
        result_ids = {r["doc_id"] for r in results}
        if sample["relevant_doc_id"] in result_ids:
            recall_hits += 1
    
    recall_at_k = recall_hits / len(probe_set)
    return {"recall_at_k": recall_at_k, "k": k, "n_queries": len(probe_set)}

# In CI:
# metrics = run_eval(new_retrieval_fn, PROBE_SET)
# assert metrics["recall_at_k"] >= BASELINE_RECALL - 0.02, f"Recall regression: {metrics}"
```

---

## Summary Table

| Anti-Pattern | Root Cause | Fix |
|--------------|-----------|-----|
| Over-chunking | "Smaller = better" without measurement | Start at 512 tokens; measure first |
| Premature agentification | "Agents are powerful" without need | Measure if you need multi-hop first |
| Reranking everything | "Rerankers improve quality" always | Only add when dense is the bottleneck |
| One index for all tenants | Simplicity over security | Per-tenant namespace isolation |
| No probe set | "We'll know if it's bad" | 50-query golden set in CI before day 1 |
