# Reranking: The Second-Stage Precision Filter

> The second-stage precision filter that separates good retrieval from great retrieval.

---

## What is Reranking?

Reranking is a second-stage step that takes the initial set of candidates returned by a fast, approximate retriever and re-scores them with a slower but more accurate model — usually a cross-encoder that looks at the query and each candidate together, rather than comparing pre-computed vectors independently. The goal is to fix the ordering: the first-stage retriever is optimized for speed and recall, and reranking trades a little latency for a lot more precision by pushing the truly relevant results to the top.

---

## Why Reranking Exists

The core compromise in bi-encoder dense retrieval: it's fast but approximate. You retrieve 50 candidates to find 5 truly relevant documents. Reranking fixes this.

```
Bi-Encoder Retrieval (Fast)
    │
    ├─ Embed query: Q ──────────────────────────────────┐
    │                                                    │
    ├─ Parallel encode all docs: D1, D2, ..., D10K ───┐ │
    │                                                 │ │
    └─ ANN search: similarity(Q, D_i) ──────────────┘ │
       │                                               │
       └─ Top-50 results (some irrelevant)           │
          │                                           │
          ├──► Cross-Encoder Reranker (Slower)       │
          │    │                                      │
          │    ├─ Encode [Q, D1], [Q, D2], ... [Q, D50] ─┐ (sequential!)
          │    │ (sees both Q and D together)            │
          │    │                                          │
          │    └─ Score each pair: P(relevant | Q, D)   │
          │       │                                      │
          │       └─ Rerank: Top-5 results            │
          │          (high precision)                 │
          │                                           │
          └───────────────────────────────────────────┘
          
Key trade-off: +50–150ms latency for 5–15% precision improvement
```

**Worked example:** Query: *"What's the maximum claim payout for water damage under a standard homeowner's policy?"* Initial bi-encoder retrieval pulls the top 20 chunks by embedding similarity, including: Chunk A (general homeowner's policy overview, mentions "water damage" once in passing), Chunk B (the exact clause on water-damage payout limits), Chunk C (flood-insurance exclusions — semantically similar, but flood ≠ water damage), and Chunk D (fire-damage payout limits — structurally similar language, wrong peril). A cross-encoder reranker scores each chunk *against the actual query* and reorders them: B → 0.94, A → 0.61, D → 0.40, C → 0.22 — correctly demoting C despite its high embedding similarity. Only the top 2–3 reranked chunks reach the LLM, so the model answers from Chunk B instead of getting diluted or misled by Chunk C.

---

## Cross-Encoder Architecture

Why cross-encoders are different from bi-encoders at the model level.

### Bi-Encoder
```
Query: "How to train a model?"
    ├─ Embed independently ──► [0.5, 0.2, 0.1, ...]
    │
Doc: "Training deep networks requires..."
    ├─ Embed independently ──► [0.4, 0.3, 0.2, ...]
    │
└─ Compare vectors (dot product or cosine)
   Score: 0.92
```

**Limitation:** Model never sees both query and document together. It's a distance metric, not a judgement of relevance.

### Cross-Encoder
```
Input: [Q] "How to train a model?" [SEP] D: "Training deep networks requires..."
    │
    ├─ Single BERT-like model
    │  └─ Full attention across Q and D together
    │
    └─ Output: Single relevance score (0–1)
       "Probability this document answers the query"
```

**Advantage:** Model sees full context. Can use linguistic patterns that only appear in Q+D pairs.

### Training Signal

Cross-encoders are trained on ranking loss:
```
For each query Q:
  - Positive document: P (relevant) → target score 1.0
  - Negative documents: N1, N2, ... (irrelevant) → target score 0.0

Loss = MarginRankingLoss(score(P) > score(N_i) + margin for all i)
```

---

## Popular Cross-Encoder Models

| Model | Latency (per pair) | NDCG@10 on MSMARCO | License | Size | When to Use |
|-------|-------------------|-------------------|---------|------|-----------|
| `ms-marco-MiniLM-L-6-v2` | 2ms | 33.6 | Apache 2.0 | 22 MB | Default choice; fast |
| `ms-marco-MiniLM-L-12-v2` | 5ms | 34.6 | Apache 2.0 | 34 MB | Slightly better; still fast |
| `ms-marco-ELECTRA-base` | 10ms | 35.7 | Apache 2.0 | 110 MB | Higher quality; slower |
| `Cohere Rerank 3` | 50ms | 39.2 (proprietary) | Proprietary | API | Highest quality; cloud-dependent |
| `BGE-reranker-large` | 15ms | 37.3 | MIT | 500 MB | High quality; open-source |
| `Jina Reranker v2` | 20ms | 38.1 | Apache 2.0 | 400 MB | Multilingual; high quality |

---

## Reranker Flavors

### Point-Wise Reranking

**Mechanism:** Score each candidate independently.

```python
def pointwise_rerank(query, candidates, cross_encoder):
    scores = []
    for doc in candidates:
        score = cross_encoder.predict([[query, doc]])[0][0]
        scores.append((doc, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

**Pros:** Simple, parallelizable
**Cons:** Ignores ranking context (that doc3 was ranked below doc2)

---

### List-Wise Reranking

**Mechanism:** LLM sees all candidates at once and outputs a ranked list.

```python
def listwise_rerank(query, candidates, llm):
    prompt = f"""Given the question: {query}

Rank these documents by relevance:
{chr(10).join([f"{i+1}. {doc}" for i, doc in enumerate(candidates)])}

Output the ranking as 1, 3, 2, ... (document numbers in order)"""
    
    ranking = llm.generate(prompt)
    return parse_ranking(ranking)
```

**Pros:** Contextual (sees all docs together); often more accurate
**Cons:** Expensive (one long LLM call); slower

**Benchmark:** ListWise often outperforms point-wise by 2–5% on web search tasks.

---

### RankGPT (Sun et al., 2023)

**Mechanism:** Use GPT-4 as a list-wise reranker with a sliding window (for large result sets).

```python
def rankgpt(query, candidates, window_size=20, step=10):
    """Rank large result sets with GPT-4 using sliding window."""
    ranking = list(range(len(candidates)))
    
    # Sliding window: rerank in chunks, update order
    for i in range(0, len(ranking), step):
        window = ranking[i:i+window_size]
        window_docs = [candidates[j] for j in window]
        
        new_order = listwise_rerank_window(query, window_docs, gpt4)
        ranking[i:i+len(new_order)] = new_order
    
    return ranking
```

**Why it works:** GPT-4 sees multiple candidates and can make nuanced judgements.

**Cost:** ~$1 per 100 queries (GPT-4 is expensive).

---

## Integration Patterns

### Standard Pattern: Dense → Rerank

**Most common.** Retrieve top-k with dense, rerank to top-j.

```python
def retrieve_and_rerank(query: str, k: int = 50, j: int = 5):
    # Stage 1: Dense retrieval (fast, approximate)
    dense_results = dense_retrieval(query, k=k)
    
    # Stage 2: Cross-encoder reranking (slow, precise)
    reranked = cross_encoder_rerank(query, dense_results, top_j=j)
    
    return reranked
```

**Latency breakdown (typical):**
- Dense embedding: 5ms
- Vector DB search: 20ms
- Cross-encoder (k=50 → j=5): 100ms (10 pairs × 10ms each if batched)
- **Total: ~125ms**

**Code: Full Pipeline**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

dense_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('ms-marco-MiniLM-L-6-v2')
client = QdrantClient(':memory:')

def rag_retrieve(query: str) -> list:
    # Embed query
    query_emb = dense_model.encode(query)
    
    # Dense retrieval
    dense_results = client.search(
        collection_name='documents',
        query_vector=query_emb,
        limit=50
    )
    
    # Extract documents
    documents = [result.payload['text'] for result in dense_results]
    
    # Cross-encoder reranking
    scores = cross_encoder.predict([[query, doc] for doc in documents])
    
    # Sort and return top-5
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:5]]
```

---

## Latency Budget Analysis

How much latency reranking adds, and when it's worth it.

| Configuration | Dense Latency | Reranker Latency | Total | NDCG@5 | Worth It? |
|---|---|---|---|---|---|
| No reranking (top-10) | 25ms | — | 25ms | 0.68 | Baseline |
| Rerank k=10→5 | 25ms | 50ms | 75ms | 0.74 | ✓ Yes (+6% quality) |
| Rerank k=20→5 | 25ms | 80ms | 105ms | 0.76 | ✓ Maybe (+8% quality) |
| Rerank k=50→5 | 25ms | 200ms | 225ms | 0.78 | ✗ Risky (if budget <300ms) |
| RankGPT k=50→5 | 25ms | 1000ms | 1025ms | 0.82 | ✗ Too slow for interactive |

**Rule of Thumb:**
- Latency budget >300ms: Rerank k=50
- Latency budget 200–300ms: Rerank k=20
- Latency budget <200ms: No reranking

---

## Reranker Failure Modes

### Position Bias

**Problem:** Cross-encoders can be sensitive to input order. Document order in the [Q, D] pair affects score.

**Example:**
```
Rerank with doc first: "Doc: ... [SEP] Q: ..." → Score: 0.92
Rerank with query first: "Q: ... [SEP] Doc: ..." → Score: 0.87
```

**Fix:** Always use the same order. Shuffle input order at test time.

```python
def rerank_debiased(query, documents, cross_encoder):
    """Rerank with random order to mitigate position bias."""
    scores = []
    for doc in documents:
        # Shuffle input order
        if random.random() > 0.5:
            score = cross_encoder.predict([[query, doc]])[0][0]
        else:
            score = cross_encoder.predict([[doc, query]])[0][0]
        scores.append((doc, score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

### Length Bias

**Problem:** Longer documents can score higher simply because they contain more terms.

**Example:**
- Short doc (50 words): "RAG is retrieval-augmented generation." → Score: 0.7
- Long doc (500 words): "RAG is... [400 more words]" → Score: 0.85 (despite not being more relevant)

**Fix:** Normalize by document length.

```python
def rerank_length_normalized(query, documents, cross_encoder):
    scores = []
    for doc in documents:
        raw_score = cross_encoder.predict([[query, doc]])[0][0]
        
        # Length normalization: penalize very long docs
        doc_length = len(doc.split())
        normalized = raw_score / (1 + 0.5 * np.log(doc_length / 100))
        
        scores.append((doc, normalized))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

---

### Domain Mismatch

**Problem:** Cross-encoder trained on web search performs poorly on legal/medical/code documents.

**Example:**
- General cross-encoder on legal query: NDCG@5 = 0.45
- Legal-fine-tuned cross-encoder on legal query: NDCG@5 = 0.72

**Fix:** Use domain-specific cross-encoder OR fine-tune on your domain.

---

## When to Skip Reranking

**Criteria:**
- Latency budget <200ms total
- Corpus is small (<10K documents; recall@50 is already high)
- Queries are highly specific (low ambiguity; dense retrieval is precise)
- Cost is critical (cross-encoder inference is expensive)

**Cost/Benefit Test:**

```python
def should_rerank(query: str, dense_results: list, cross_encoder) -> bool:
    """Decide whether reranking is worth the cost."""
    
    # Measure dense retrieval quality
    dense_ndcg = compute_ndcg(dense_results, labeled_relevant)
    
    # Rerank and measure improvement
    reranked = cross_encoder_rerank(query, dense_results)
    reranked_ndcg = compute_ndcg(reranked, labeled_relevant)
    
    improvement = reranked_ndcg - dense_ndcg
    
    # If improvement <0.02 (2%), skip reranking
    return improvement > 0.02
```

---

## Key Takeaways

1. **Reranking is almost always worth the latency cost.** 5–10% precision improvement for 50–150ms is a good trade.
2. **Start with `ms-marco-MiniLM-L-6-v2`.** Fast, accurate, open-source.
3. **k=20→5 is the sweet spot.** Rerank top-20 to top-5. Balances cost and quality.
4. **Beware of position and length bias.** Shuffle input order; normalize by length.
5. **Domain-specific cross-encoders are worth fine-tuning.** If NDCG<0.65 on your domain, fine-tune or use a domain-specific model.
