# 08 — Cascading Retrieval Failure

> Query expansion, HyDE, or multi-hop retrieval designed to recover from a weak initial retrieval instead amplifies the error — each stage compounds the mistake rather than correcting it.

---

## Q1. What is cascading retrieval failure and how does it differ from simple retrieval failure? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Simple retrieval failure:** The first retrieval step returns no relevant documents, and the system either abstains or hallucinates. The error is visible and isolated.

**Cascading retrieval failure:** A recovery mechanism — query expansion, HyDE, iterative retrieval, or multi-hop reasoning — is triggered by a weak first retrieval, but instead of improving results, it confidently propagates the wrong direction across multiple retrieval rounds.

```
Simple failure:
  Query: "What is quantum error correction?"
  Retrieval: [nothing relevant] → "I don't know" ✓ (correct abstention)

Cascading failure:
  Query: "What is quantum error correction?"
  Round 1 retrieval: weak → returns docs about "quantum computing basics"
  HyDE/expansion: "Quantum error correction is about fixing errors in quantum circuits [assumed]"
  Round 2 retrieval: based on that wrong hypothesis → returns more "quantum computing basics"
  Round 3: same wrong neighborhood → LLM generates confident wrong answer
```

**Why cascading failure is worse than simple failure:**

| Property | Simple Failure | Cascading Failure |
|----------|---------------|-------------------|
| Detectable? | Often yes (no results / abstention) | Often no (confident answer returned) |
| LLM behavior | Hedges or refuses | Generates with false confidence |
| Retrieval rounds wasted | 1 | 3–5+ |
| Latency impact | None (fast fail) | High (multiple slow rounds) |
| Downstream harm | Low | High (wrong answer delivered) |

**Common triggers:**

- HyDE (Hypothetical Document Embedding) generates a plausible but wrong hypothesis that pulls retrieval into a wrong semantic neighborhood
- Multi-query expansion creates multiple variants of a wrong interpretation of the query
- Multi-hop RAG correctly follows a chain of relationships that starts from a wrong first hop
- FLARE retrieves mid-generation based on a wrong sentence it already generated

</details>

---

## Q2. How does HyDE cause cascading retrieval failure and how do you prevent it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**HyDE** (Hypothetical Document Embedding) asks the LLM to generate a hypothetical answer, then uses the embedding of that answer as the retrieval query instead of the original query.

**Normal HyDE behavior:**
```
Query: "What is Transformer architecture?"
HyDE: "A Transformer uses self-attention with Q, K, V matrices to compute attention scores..."
Embedding of hypothesis → retrieves actual papers on Transformer architecture ✓
```

**HyDE cascading failure:**
```
Query: "How does LoRA fine-tuning reduce memory usage?"
HyDE (LLM's wrong assumption): "LoRA reduces memory by compressing the full weight
  matrix using knowledge distillation and pruning techniques..."
              ↑ wrong — LoRA uses low-rank decomposition, not compression/pruning

HyDE embedding → points toward "knowledge distillation" and "pruning" literature
Retrieved docs: papers on model compression (correct topic: wrong subtopic)
LLM generates: confident but wrong explanation mixing LoRA with compression techniques
```

**Detection:**

```python
def check_hyde_coherence(original_query: str, hypothesis: str, retrieved_docs: list[str]) -> float:
    """
    Check whether the retrieved docs are semantically closer to the original query
    or to the HyDE hypothesis. A large gap may indicate the hypothesis misled retrieval.
    """
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    q_emb  = model.encode(original_query)
    h_emb  = model.encode(hypothesis)
    d_embs = model.encode(retrieved_docs)
    
    q_similarity = float(util.cos_sim(q_emb,  d_embs.mean(axis=0)))
    h_similarity = float(util.cos_sim(h_emb,  d_embs.mean(axis=0)))
    
    # High h_similarity and low q_similarity = hypothesis dominated retrieval
    return q_similarity / (h_similarity + 1e-8)   # ratio < 0.7: suspicious

ratio = check_hyde_coherence(query, hypothesis, docs)
if ratio < 0.7:
    # Fall back to direct query embedding retrieval
    docs = direct_retrieval(query)
```

**Preventions:**

1. **Fallback comparison:** Run both direct retrieval (from query) and HyDE retrieval; take the one with higher self-consistency score
2. **Hypothesis confidence gate:** Only use HyDE when the LLM-generated hypothesis has high confidence (low perplexity); reject low-confidence hypotheses
3. **Cross-check hypothesis against query:** Verify the hypothesis embedding is directionally consistent with the original query embedding (cosine > 0.75)
4. **Ensemble instead of replace:** Combine HyDE embeddings with query embeddings rather than replacing the query entirely

```python
# Ensemble approach — blend query and hypothesis embeddings
import numpy as np

q_emb = embed(original_query)
h_emb = embed(hyde_hypothesis)
blended = 0.5 * q_emb + 0.5 * h_emb   # equal blend; tune alpha
blended /= np.linalg.norm(blended)      # re-normalize
results = vector_index.search(blended, k=10)
```

</details>

---

## Q3. How does cascading failure manifest in multi-hop RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Multi-hop RAG chains multiple retrieval rounds, where each round's query is formed from the previous round's results. A wrong first hop propagates through the entire chain.

```
User query: "Who is the CEO of the company that developed GPT-4?"

Round 1 — "What company developed GPT-4?"
  Correct answer: OpenAI
  Wrong retrieval (index is stale): returns docs about "Anthropic GPT research"
                                     ↑ error injected here

Round 2 — "Who is the CEO of Anthropic?"  ← built on wrong Round 1 answer
  Correct retrieval for the wrong question: returns "Dario Amodei"

Round 3 — Generate answer:
  "The CEO of the company that developed GPT-4 is Dario Amodei."
  ← completely wrong, stated with high confidence
```

**Why multi-hop amplifies errors:**
- Each hop narrows the retrieval to a specific sub-question
- Later hops have no visibility into whether earlier hops were correct
- The final LLM generation sees only the last hop's context — the original error is invisible

**Mitigation — re-anchor each hop against the original query:**

```python
def multi_hop_with_reanchoring(original_query: str, max_hops: int = 3) -> str:
    context = ""
    
    for hop in range(max_hops):
        # Formulate sub-query
        sub_query = formulate_sub_query(original_query, context)
        
        # Retrieve
        docs = retriever.retrieve(sub_query, k=5)
        
        # Reanchoring check: are the new docs still relevant to the ORIGINAL query?
        relevance_to_original = [
            rerank_score(original_query, doc) for doc in docs
        ]
        
        # Filter out docs with low relevance to original query
        docs = [d for d, s in zip(docs, relevance_to_original) if s > 0.4]
        
        if not docs:
            break   # This hop drifted too far — stop chain
        
        context += "\n".join(doc.page_content for doc in docs)
        
        # Check if we have enough to answer the original query
        if can_answer(original_query, context):
            break
    
    return generate(original_query, context)
```

**Confidence calibration across hops:**

Add an explicit self-consistency check after each hop:

```python
CONSISTENCY_PROMPT = """Original question: {original}
Current retrieved evidence: {evidence}

Does this evidence help answer the original question? 
If not, what went wrong in the reasoning chain?
Answer YES or EXPLAIN_PROBLEM."""
```

</details>

---

## Q4. How do you detect and recover from cascading retrieval failure in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Detection signals:**

```python
DETECTION_SIGNALS = {
    "low_answer_confidence": lambda response: response.logprobs_mean < 0.6,
    "high_retrieval_drift": lambda q, docs: semantic_drift(q, docs) > 0.4,
    "answer_contradicts_query": lambda q, a: nli_score(q, a) == "CONTRADICTION",
    "empty_hops": lambda hops: any(len(hop_results) == 0 for hop_results in hops),
    "repetitive_retrieval": lambda hops: len(set(
        doc.id for hop in hops for doc in hop)) < len(hops),  # same docs repeated
}

def semantic_drift(original_query: str, retrieved_docs: list[str]) -> float:
    """Measures how far retrieved docs drifted from the original query."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode(original_query)
    d_embs = model.encode(retrieved_docs)
    similarities = util.cos_sim(q_emb, d_embs)[0]
    return 1.0 - float(similarities.mean())   # high = large drift
```

**Recovery strategies:**

| Strategy | When to Apply | Mechanism |
|----------|--------------|-----------|
| **Reset and retry** | Drift detected after first hop | Discard multi-hop chain; retry with direct dense retrieval |
| **Fallback to BM25** | Semantic retrieval keeps drifting | BM25 keyword match anchors to original query terms |
| **Widen retrieval** | All hops return the same docs | Increase k, decrease similarity threshold |
| **Re-formulate query** | LLM-based: ask to rephrase differently | Reformulate original query with constraints ("avoid the subtopic X") |
| **Abstain** | All recovery attempts fail | Return "I don't have sufficient information" |

**Circuit breaker pattern:**

```python
class CascadeCircuitBreaker:
    def __init__(self, max_drift: float = 0.45, max_empty_hops: int = 1):
        self.max_drift = max_drift
        self.max_empty_hops = max_empty_hops
        self.empty_hops = 0

    def check(self, original_query: str, current_docs: list[str]) -> str:
        """Returns 'continue', 'fallback', or 'abstain'."""
        if not current_docs:
            self.empty_hops += 1
            if self.empty_hops >= self.max_empty_hops:
                return "abstain"
            return "fallback"
        
        drift = semantic_drift(original_query, [d.page_content for d in current_docs])
        if drift > self.max_drift:
            return "fallback"
        
        return "continue"
```

</details>

---

## Real-World Applications

- **Agentic research pipelines** (LangGraph, AutoGPT): Multi-step retrieval loops are prime targets for cascading failures; circuit breakers and re-anchoring are essential
- **FLARE** (Forward-Looking Active Retrieval): Documented that mid-generation retrieval based on low-confidence sentences can trigger cascading errors if the triggering sentence itself is wrong
- **RAG for code generation**: Wrong interpretation of the user's intent in the first retrieval round causes all subsequent steps to fetch irrelevant API documentation
