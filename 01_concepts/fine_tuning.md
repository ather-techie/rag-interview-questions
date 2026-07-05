# Fine-Tuning for RAG: Adapting Embeddings and Rerankers to Your Domain

> When off-the-shelf retrieval models plateau, fine-tuning embeddings and rerankers closes the domain gap — but it is the *last* lever to pull, not the first.

---

## What is Fine-Tuning (for RAG)?

Fine-tuning is the process of taking an off-the-shelf embedding or reranker model and further training its weights on data from your specific domain, so it learns the vocabulary and relevance judgments that generic models miss. In a RAG context, it's typically the last lever to pull — after chunking, retrieval strategy, and prompting have already been tuned — because it requires labeled data and ongoing maintenance that cheaper fixes don't.

---

## When to Fine-Tune vs. Cheaper Fixes

Fine-tuning is expensive: it requires labeled data, training infrastructure, a domain eval set, and (for embeddings) a full corpus re-index. Most retrieval problems have cheaper fixes. Exhaust those first.

```
Retrieval quality problem detected
    │
    ├─ 1. Prompt / query rewriting fixes ──► hours of effort
    │      (query expansion, HyDE, instruction prefixes)
    │
    ├─ 2. Chunking fixes ──────────────────► hours–days
    │      (size, overlap, structure-aware splitting)
    │
    ├─ 3. Hybrid search (BM25 + dense) ────► days
    │      (catches exact-match domain terms embeddings miss)
    │
    ├─ 4. Add / swap a reranker ───────────► days
    │      (cross-encoder catches bi-encoder mistakes)
    │
    └─ 5. Fine-tune ───────────────────────► weeks
           (only when 1–4 plateau below your quality bar)
```

### The Decision Signal

Fine-tune when **domain vocabulary mismatch is measurable**: build a domain eval set (query → relevant-chunk pairs) and track Recall@10. If Recall@10 plateaus well below acceptable (e.g., stuck at 0.60 when you need 0.85+) despite chunking tuning, hybrid search, and a reranker, the embedding model simply doesn't understand your domain's semantics — no amount of pipeline tuning fixes that.

### Decision Table: Symptom → Cheaper Fix → When Fine-Tuning Is Justified

| Symptom | Cheaper Fix First | Fine-Tuning Justified When |
|---------|-------------------|----------------------------|
| Exact domain terms (SKUs, drug names, statute IDs) not retrieved | Hybrid search (BM25 + dense) | Terms are retrieved but *semantically misranked* even with hybrid |
| Relevant doc retrieved at rank 15–50, not top-5 | Add a cross-encoder reranker | Reranker also misranks because it doesn't know domain semantics |
| Paraphrased domain queries fail ("MI" vs. "heart attack" vs. "myocardial infarction") | Query expansion / synonym injection | Synonym lists are unbounded; expansion can't keep up |
| Relevant content exists but chunks split it badly | Re-chunk (structure-aware, overlap) | Recall@10 still low after chunking sweep |
| Recall fine, but answers cite wrong passages | Reranker, smaller top-k | Reranker NDCG also <0.65 on domain eval set |
| Recall@10 plateaus <0.70 on a 200+ query domain eval set after all of the above | — (you've exhausted cheap fixes) | **Yes — fine-tune embeddings and/or reranker** |

**Rule of thumb:** If you can't measure the problem on a held-out domain eval set, you're not ready to fine-tune — you'd have no way to know whether it worked.

---

## Training-Data Collection

Fine-tuning quality is bounded by training-data quality. You need (query, positive, negatives) triplets. Four sourcing strategies, roughly in order of signal quality:

### 1. Mining Positives from Click / Usage Logs

If your RAG system (or any internal search) is live, user behavior is free labels:

```python
def mine_positives_from_logs(search_logs):
    """Clicked/copied/cited results are weak positive labels."""
    pairs = []
    for entry in search_logs:
        query = entry["query"]
        for doc in entry["results"]:
            # Strong signal: user clicked AND dwelled / copied / thumbs-up
            if doc["clicked"] and doc["dwell_seconds"] > 30:
                pairs.append((query, doc["chunk_id"]))
    return pairs
```

**Caveats:** click data is noisy (position bias — users click rank 1 regardless) and biased toward what the *current* system already retrieves. Debias by discounting top-ranked clicks or only using clicks at rank ≥3.

### 2. Hard-Negative Mining

Contrastive training needs negatives, and *hard* negatives (lexically similar but not relevant) teach the model far more than random ones.

| Strategy | Mechanism | Difficulty of Negatives |
|----------|-----------|------------------------|
| **Random / in-batch negatives** | Other examples' positives in the same training batch double as negatives | Easy (often trivially irrelevant) |
| **BM25 hard negatives** | Run BM25, take top-k results that are *not* labeled relevant | Hard (share vocabulary with query) |
| **ANCE-style (model-mined)** | Periodically re-encode corpus with the *training* model; take its top-ranked non-relevant docs as negatives | Hardest (exactly the model's current mistakes) |

```python
def mine_bm25_hard_negatives(query, positive_id, bm25_index, k=20):
    """Top BM25 hits that aren't the positive = hard negatives."""
    candidates = bm25_index.search(query, k=k)
    return [doc for doc in candidates if doc.id != positive_id][:5]
```

**Warning:** hard-negative mining can produce **false negatives** — docs that are actually relevant but unlabeled. These poison training. Filter with a cross-encoder: if the cross-encoder scores a "negative" >0.9, drop it.

### 3. Synthetic Query Generation (GPL / Promptagator Pattern)

No usage logs? Generate queries from your corpus with an LLM:

```
Corpus chunks
    │
    ├─ For each chunk: LLM generates 1–3 plausible queries
    │   ("Write a question a customer would ask that this
    │     passage answers.")
    │
    ├─ Filter: round-trip check — does retrieval with the
    │   generated query rank the source chunk highly?
    │   Discard pairs that fail. (Promptagator filtering)
    │
    ├─ Mine hard negatives per query (BM25 / dense)
    │
    └─ Optionally pseudo-label with a cross-encoder
        (GPL: cross-encoder score = soft training target)
```

```python
def generate_synthetic_pairs(chunks, llm):
    pairs = []
    for chunk in chunks:
        prompt = (
            "Write a realistic user question that the following "
            f"passage answers. Passage:\n{chunk.text}\n\nQuestion:"
        )
        query = llm.invoke(prompt).strip()
        pairs.append((query, chunk.id))
    return pairs
```

**Key risk:** the model overfits to the *LLM's query style*, not real users' style (see Evaluation section).

### 4. How Much Data Do You Need?

Rough orders of magnitude:

| Adaptation Method | Pairs Needed | Why |
|-------------------|-------------|-----|
| LoRA-style / light adaptation of an embedding model | ~1,000–5,000 | Few trainable params; less data to fit |
| Full contrastive fine-tune of a bi-encoder | ~10,000–50,000+ | All params updated; needs broad coverage |
| Cross-encoder reranker fine-tune | ~5,000–20,000 | Pairwise task is data-efficient vs. bi-encoder |
| From-scratch domain embedding model | Millions | Don't do this — adapt a pre-trained model |

Below ~1,000 pairs, prefer prompt-level fixes or a better off-the-shelf model — fine-tuning on tiny data risks catastrophic forgetting of general semantics.

---

## Training Loop Overview

### The Contrastive Objective: InfoNCE / MultipleNegativesRankingLoss

The workhorse loss for bi-encoder fine-tuning. For a batch of (query, positive) pairs, every other positive in the batch serves as a negative:

```
Batch of N pairs: (q1,p1), (q2,p2), ..., (qN,pN)

For query q1:
  positive:  p1                        ← pull together
  negatives: p2, p3, ..., pN (in-batch) ← push apart

Loss (InfoNCE):
  L = -log( exp(sim(q1,p1)/τ) / Σ_j exp(sim(q1,pj)/τ) )

  = softmax cross-entropy over the batch,
    where the "correct class" is the true positive
```

Larger batch = more in-batch negatives = stronger signal. This is why embedding fine-tuning benefits disproportionately from large batches (and why GradCache / gradient checkpointing tricks exist).

### Bi-Encoder Fine-Tuning (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# (query, positive) pairs — in-batch negatives are implicit.
# Optionally add explicit hard negatives as a third text.
train_examples = [
    InputExample(texts=["error code E-4012 dishwasher",
                        "Error E-4012 indicates a drain pump fault..."]),
    InputExample(texts=["coverage limit water damage condo policy",
                        "Condominium policies cap water damage claims at...",
                        "Water damage to vehicles is covered under..."]),  # hard negative
    # ... thousands more
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(model)  # InfoNCE-style

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,            # 1–3 epochs; contrastive FT overfits fast
    warmup_steps=100,
)
model.save("./bge-base-finetuned-domain")
```

### Cross-Encoder Reranker Fine-Tuning

The reranker is often the *better first fine-tuning target*: it needs less data, and — critically — **no re-indexing** when it changes.

```python
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

# Explicit binary labels: 1 = relevant, 0 = not relevant
train_examples = [
    InputExample(texts=["error code E-4012 dishwasher",
                        "Error E-4012 indicates a drain pump fault..."], label=1),
    InputExample(texts=["error code E-4012 dishwasher",
                        "Error E-2201 relates to the heating element..."], label=0),
    # ... include mined hard negatives as label=0
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

model.fit(
    train_dataloader=train_dataloader,
    epochs=2,
    warmup_steps=100,
)
model.save("./reranker-finetuned-domain")
```

### LoRA / PEFT for Larger Models

For 7B-scale embedding models (E5-mistral, GTE-Qwen), full fine-tuning is memory-prohibitive. LoRA injects small trainable low-rank matrices and freezes the base:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(base_model, lora_config)
# Trainable params: ~0.5% of total → fits on a single GPU,
# needs fewer pairs (~1K–5K), reduces catastrophic forgetting.
```

---

## Evaluation

Fine-tuning without a domain benchmark is flying blind. Build the benchmark *before* training.

### Build a Domain Benchmark

- **Held-out** query → relevant-chunk pairs (never seen in training): 100–500 queries minimum.
- Source from real user queries if at all possible; hand-label relevance.
- Keep a slice of **general-domain** queries too (e.g., a BEIR subset) to detect regression on out-of-domain semantics.

### Measure Before vs. After

| Metric | Measures | Target Movement |
|--------|----------|-----------------|
| Recall@k (k=5,10,50) | "Is the answer in the candidate set at all?" | The primary fine-tuning success metric |
| MRR | Rank of the *first* relevant result | Good for single-answer queries |
| NDCG@k | Graded relevance across the full ranking | Best when relevance isn't binary |

```python
# The report that justifies (or kills) the fine-tune:
#
#                      Recall@10   MRR    NDCG@10
# Base model (domain)     0.61    0.42     0.55
# Fine-tuned (domain)     0.84    0.63     0.74    ← +23pp recall
# Base model (general)    0.78    0.59     0.71
# Fine-tuned (general)    0.75    0.56     0.68    ← -3pp: acceptable drift
```

### Beware Overfitting to Synthetic Query Style

If training queries came from an LLM, the model may learn "LLM-generated question style" rather than domain semantics. Symptoms: large gains on a synthetic eval set, flat performance on real user queries. **Always evaluate on human-written queries**, even if you trained on synthetic ones. Mixing paraphrase variants and multiple generation prompts during data creation reduces this risk.

### The Re-Index Requirement (Operational Cost Callout)

> **Every time the embedding model changes, every vector in the corpus must be regenerated.** Old vectors and new vectors live in incompatible spaces — you cannot mix them in one index.

Operational implications:

- **Cost:** re-embedding 10M chunks is a real compute bill and hours-to-days of pipeline time.
- **Cutover:** you need a blue/green index strategy (build new index alongside, swap atomically) or accept downtime.
- **Iteration speed:** this is the strongest argument for fine-tuning the **reranker first** — a reranker swap requires *zero* re-indexing, so you can iterate daily instead of weekly.

---

## Cost / Effort Comparison: The Full Ladder of Fixes

| Intervention | Effort | Cost | Risk | When It Pays Off |
|--------------|--------|------|------|------------------|
| **Prompt / query-rewrite fixes** | Hours | ~$0 | Minimal | Always try first; fixes phrasing-level mismatch |
| **Hybrid search (add BM25)** | Days | Low (infra only) | Low | Exact-match domain terms (IDs, codes, names) |
| **Reranker swap (off-the-shelf)** | Days | Low–medium (inference latency/cost) | Low | Recall is fine, precision at top-k is not |
| **Reranker fine-tune** | 1–2 weeks | Medium (labeling + 1 GPU) | Medium (overfit) | Domain misranking persists; needs ~5K+ pairs; **no re-index needed** |
| **Embedding fine-tune** | 2–4 weeks | Medium–high (labeling + GPU + **full re-index**) | Medium-high (forgetting, re-index ops) | Recall@10 plateaus on domain eval despite everything above |
| **LLM fine-tune (Self-RAG style)** | 1–3 months | High ($10K–50K+ training + annotation) | High (stale weights, serving complexity) | Generation-side failures (hallucination, retrieval gating) at high query volume in a stable domain |

**Reading the table top-down is the interview answer:** each row only becomes justified when the rows above it have measurably failed on a held-out domain eval set.

---

## Interview Gotchas

1. **"We had bad retrieval, so we fine-tuned the embeddings" is a red flag answer.** The interviewer wants to hear the ladder: prompt → chunking → hybrid → reranker → fine-tune, with a *measured* plateau (e.g., Recall@10 stuck at 0.6 on a domain eval set) justifying the jump.

2. **Know why hard negatives matter.** Random negatives are too easy — the model learns nothing. BM25/ANCE hard negatives are lexically close but irrelevant, which is exactly the boundary the model must learn. Also mention the false-negative poisoning risk.

3. **Fine-tune the reranker before the embeddings.** Less data needed, and no re-indexing — you can ship and iterate without touching the vector store. Many candidates miss this ordering.

4. **The re-index trap.** If asked "what happens after you deploy a fine-tuned embedding model?", the answer is: regenerate every vector, blue/green index cutover, and a plan for documents ingested mid-migration. Mixing vectors from two models in one index silently breaks similarity scores.

5. **Synthetic data ≠ free lunch.** GPL/Promptagator-style generation works, but evaluate on *real* user queries — gains on synthetic eval sets often don't transfer if the LLM's query style differs from users'.

6. **Quantify the data requirements.** ~1K pairs for LoRA-style adaptation, ~10K+ for full contrastive fine-tuning, ~150K examples for Self-RAG-style LLM training. Citing orders of magnitude signals hands-on experience.

7. **Catastrophic forgetting is the hidden failure mode.** A domain-fine-tuned embedding model can regress on general queries. Always report before/after on a general benchmark slice (e.g., BEIR subset), not just the domain set.
