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
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2ms | 33.6 | Apache 2.0 | 22 MB | Default choice; fast |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 5ms | 34.6 | Apache 2.0 | 34 MB | Slightly better; still fast |
| `cross-encoder/ms-marco-ELECTRA-base` | 10ms | 35.7 | Apache 2.0 | 110 MB | Higher quality; slower |
| `Cohere Rerank 4` | 50ms | Proprietary eval suite (not public MSMARCO) | Proprietary | API | Highest quality; cloud-dependent; ships alongside a lighter "Fast" variant |
| `BGE-reranker-large` | 15ms | 37.3 | MIT | 500 MB | High quality; open-source |
| `Jina Reranker v2` | 20ms | 38.1 | Apache 2.0 | 400 MB | Multilingual; high quality |

> **Cohere Rerank version history:** Rerank 3.5 (Dec 2024) added stronger reasoning over complex/constrained queries and much better multilingual and cross-lingual search (+26% vs. Rerank 3 on cross-lingual benchmarks, SOTA across 10+ business languages). Rerank 4 (Dec 2025), offered in "Pro" and "Fast" variants, is the current flagship as of mid-2026 and further improves accuracy and speed for enterprise retrieval. Check Cohere's docs for the latest model ID before integrating, since this lineup updates roughly annually.

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

**Cost:** LLM-based reranking is meaningfully more expensive per query than a cross-encoder — often 1–2 orders of magnitude higher, since it re-processes the full text of every candidate as input tokens on each call (and sliding-window approaches multiply that by the number of windows), instead of a single cheap forward pass through a small purpose-built model. LLM API pricing changes frequently and varies a lot by model tier, so check current provider pricing (OpenAI, Anthropic, Google, etc.) before budgeting rather than relying on a fixed per-query figure.

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
- Cross-encoder (k=50 → j=5): 200ms (50 pairs × ~4ms each if batched)
- **Total: ~225ms**

**Code: Full Pipeline**

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

dense_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
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

**Fix:** For most cross-encoders, query and document are *not* interchangeable — the model is trained with a fixed `[query, document]` order, so swapping the arguments doesn't cancel out bias; it feeds the model an input it never saw during training and degrades quality. The correct mitigation is to always call the model with the order specified in its model card (never mix orders per-call). Order-sensitivity is then a property of that specific model, not something to patch at inference time. A small number of models are explicitly trained to be order-symmetric — only average scores across both orderings if the model card confirms that support.

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

## Score Calibration and Thresholding

Reranker scores are a ranking signal, not a probability — treating them as one is a common production mistake.

**Problem:** Raw cross-encoder outputs (whether logits or a sigmoid-squashed score) are trained purely with a ranking loss (see Training Signal above) — the objective only pushes `score(positive) > score(negative) + margin` for candidates *within the same query*. Nothing in that objective calibrates the absolute value of the score against a global notion of "relevant" vs. "not relevant," and nothing ties one query's score scale to another's. A score of 0.83 for one query and 0.83 for a different query aren't the same thing — they can reflect very different amounts of true relevance, because the model only ever learned to compare candidates against each other inside a single query's context. The reliable information is the *relative order* the scores impose on one query's candidate set, not the raw magnitude.

**Why it matters in production:** It's tempting to add a rule like "if `reranker_score < 0.5`, respond with 'insufficient evidence' instead of answering" as a cheap abstention guardrail. That works fine for top-k selection (ranking within a set), but misfires as an absolute cutoff: score distributions shift across query types and domains. A well-answered factual query in a narrow domain might top out around 0.6, while an ambiguous or multi-hop query in a broad domain might have its best candidate sitting at 0.9. A single global threshold tuned on one slice of traffic ends up over-triggering abstention on some query types and under-triggering it on others — inconsistent behavior that's hard to debug because "the score" looks like a probability but isn't one.

**Fix:** Calibrate the score against labeled data, or normalize within each query's candidate set — don't threshold the raw score directly.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np

# --- Option 1: fit a calibration function on a labeled validation set ---
# raw_scores: cross-encoder outputs; labels: 1 if human-judged relevant, else 0

# Platt scaling: 1D logistic regression mapping raw score -> probability
platt = LogisticRegression()
platt.fit(raw_scores.reshape(-1, 1), labels)
calibrated_prob = platt.predict_proba(new_score.reshape(-1, 1))[:, 1]

# Isotonic regression: monotonic (non-parametric) mapping; needs more data
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_scores, labels)
calibrated_prob = iso.predict(new_score)

# Threshold on the CALIBRATED probability, never on the raw score
if calibrated_prob < 0.5:
    return "insufficient evidence"

# --- Option 2: no labeled data? normalize within the query's candidate set ---
def normalize_scores(scores: list) -> list:
    """Min-max normalize reranker scores within one query's candidate set."""
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)  # degenerate case: all candidates scored ~equally
    return [(s - lo) / (hi - lo) for s in scores]

def softmax_normalize(scores: list) -> list:
    """Softmax over one query's candidate set — separation, not absolute score."""
    exp_scores = np.exp(np.array(scores) - max(scores))
    return list(exp_scores / exp_scores.sum())
```

Both approaches turn the reranker's raw ranking signal into something that behaves consistently as an abstention threshold across query types and domains, instead of a number that only means something relative to its own query's candidate set.

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
2. **Start with `cross-encoder/ms-marco-MiniLM-L-6-v2`.** Fast, accurate, open-source.
3. **k=20→5 is the sweet spot.** Rerank top-20 to top-5. Balances cost and quality.
4. **Beware of position and length bias.** Shuffle input order; normalize by length.
5. **Domain-specific cross-encoders are worth fine-tuning.** If NDCG<0.65 on your domain, fine-tune or use a domain-specific model.

---

## Related

- [Fine-Tuning](./fine_tuning.md) — how to fine-tune a cross-encoder reranker for your domain instead of using an off-the-shelf model.
- [Retrieval Strategies](./retrieval_strategies.md) — reranking is a second-stage step downstream of the first-stage retriever covered here.
- [Evaluation Metrics](./evaluation_metrics.md) — how to measure reranker quality (NDCG, precision) before and after reranking.
