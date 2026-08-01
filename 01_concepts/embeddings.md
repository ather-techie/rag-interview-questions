# Embeddings: The Mathematical Foundation of Semantic Search

> The mathematical foundation of semantic search — what embedding models learn and why it matters for retrieval.

---

## What is an Embedding?

An embedding is a dense numerical vector that represents a piece of text (or image, audio, etc.) such that texts with similar meaning end up close together in vector space. Instead of matching on exact keywords, an embedding model lets you compare meaning: a query and a document can be judged "similar" by measuring the distance between their vectors. This is what makes semantic search — and therefore retrieval in RAG — possible at all.

---

## What an Embedding Model Learns

An embedding model maps text → dense vectors such that semantically similar texts are geometrically close. This property enables retrieval by similarity.

### The Training Objective: Contrastive Loss

Embedding models learn via contrastive loss. For each query, the model is shown:
- A positive document (relevant)
- N negative documents (irrelevant)

The loss penalizes high similarity to negatives and rewards high similarity to positives.

```
Training Example:
  Query: "How do you make pasta?"
  Positive: "Boil water, add salt, then pasta..."   ← similarity should be high
  Negative: "The history of ancient Rome..."        ← similarity should be low
  Negative: "How to train a neural network..."      ← similarity should be low
  
  Embedding space (2D projection):
  
        pasta_doc
           •
          / \
         /   \
        /     \
    query •     • rome_doc
        \     /
         \   /
          \ /
           •
       ai_doc
  
  Loss penalizes: proximity to rome_doc, ai_doc
  Loss rewards: proximity to pasta_doc
```

### The Three Encoder Architectures

**Bi-Encoder (Most Common in RAG)**
- Encodes query and documents separately
- At retrieval time: embed query once, compare against pre-computed document vectors
- Fast: O(1) retrieval after O(corpus_size) pre-computation
- Trade-off: Cannot see both query and document simultaneously; less accurate but enables large-scale search
- Used by: OpenAI embeddings, BGE, E5, Cohere Embed, Voyage AI, Gemini Embedding

**Cross-Encoder**
- Encodes query and document together
- Output: single relevance score
- Cannot be used for first-stage retrieval (no pre-computed document vectors)
- Accurate: Sees both inputs simultaneously
- Role in RAG: Post-retrieval reranking (covered in `01_concepts/reranking.md`)

**Adapter (Hybrid)**
- Bi-encoder for retrieval, cross-encoder for reranking
- The standard production pattern (covered in `02_interview_bank/02-advanced-rag.md`)

---

## The Major Embedding Model Families

| Model | Dimensions | Context Window | License | MTEB Score | Best Use Case |
|-------|-----------|-----------------|---------|-----------|---------------|
| `text-embedding-3-small` | 1,536 (supports truncation to 512 via Matryoshka) | 8,192 | Proprietary | 62.3 | General-purpose (Recommended starting point) |
| `text-embedding-3-large` | 3,072 | 8,192 | Proprietary | 64.6 | High-precision retrieval; supports Matryoshka |
| `BGE-large-en-v1.5` | 1,024 | 512 | Apache 2.0 | 64.2 | Open-source alternative to OpenAI; good for English |
| `E5-mistral-7b-instruct` | 4,096 | 32,768 | MIT | 61.5 | Long-context support; multilingual |
| `Cohere Embed v4` | 256 / 512 / 1,024 / 1,536 (Matryoshka) | 128,000 | Proprietary | 65.2 | Multimodal — unified embeddings for text, images, and interleaved text+image in one vector; supersedes v3 (retained `input_type` parameter for query vs. document embeddings) |
| `nomic-embed-text` | 768 | 8,192 | Apache 2.0 | 62.4 | Open-source; competitive with OpenAI; uses Matryoshka |
| `voyage-3-large` | 256 / 512 / 1,024 / 2,048 (Matryoshka) | 32,000 | Proprietary | 65.1 | Highest-accuracy general-purpose retrieval; native `output_dtype` for int8/uint8/binary/ubinary at embed time |
| `gemini-embedding-001` | 3,072, truncatable to 1,536 / 768 / 256 (Matryoshka) | 2,048 per input | Proprietary | 68.3 (MTEB multilingual) | Google's unified successor to `text-embedding-004`; top-ranked on the MTEB multilingual leaderboard |
| `jina-embeddings-v3` | 1,024, down to 32 (Matryoshka) | 8,192 | CC BY-NC 4.0 (commercial license available) | 65.5 | Multilingual (89 languages) with task-specific LoRA adapters (retrieval, classification, clustering, etc.) |
| `Qwen3-Embedding-8B` | Up to 4,096, flexible 32–4,096 | 32,768 | Apache 2.0 | 70.6 (MTEB multilingual, #1 as of Jun 2025) | Best-in-class open-source multilingual embedding; also ships as 0.6B / 4B variants for lighter deployments |

### MTEB Benchmark Explained

MTEB (Massive Text Embedding Benchmark) originally evaluated embeddings on 56 mostly-English datasets across 8 task types (retrieval, clustering, classification, etc.). It has since been superseded by **MMTEB** (Massive Multilingual Text Embedding Benchmark, a 2025 community-driven expansion): 500+ quality-controlled tasks spanning 250+ languages, plus harder task types like instruction-following, long-document retrieval, and code retrieval. A score of 60+ is still a reasonable production-grade bar on the English subset, but scores aren't directly comparable across MTEB versions — always check which benchmark revision a leaderboard number came from.

**What MTEB tests well:** General-purpose retrieval on diverse text
**What MTEB misses:** Domain-specific performance (medical, legal, code)

### The Matryoshka Trick: Embedding Dimension Truncation

OpenAI's `text-embedding-3` models support Matryoshka embeddings. The key insight: you can truncate the embedding to fewer dimensions with minimal quality loss.

```python
from openai import OpenAI

client = OpenAI()

# Full 1536-dim embedding (native size) via the dimensions parameter
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What is RAG?",
    dimensions=1536
)
full_embedding = response.data[0].embedding  # length: 1536

# Truncate to 256 dims — OpenAI handles this server-side
response_truncated = client.embeddings.create(
    model="text-embedding-3-small",
    input="What is RAG?",
    dimensions=256
)
truncated = response_truncated.data[0].embedding  # length: 256

# Saves 50% storage + 50% retrieval latency with ~1% recall loss
```

**When to use:** If your latency or storage budget is tight. Trade-off: ~1% recall loss per 50% dimension reduction.

### Binary / Int8 Quantization: Reducing Per-Dimension Precision

Matryoshka truncation shrinks the *number* of dimensions; quantization shrinks the *precision* of each dimension — the two techniques compose (e.g., truncate to 512 dims, then quantize to int8).

| Technique | Storage per vector | Compression vs. float32 | Typical recall impact |
|-----------|--------------------|--------------------------|------------------------|
| float32 (baseline) | 4 bytes/dim | 1x | — |
| int8 (scalar quantization) | 1 byte/dim | ~4x | Small (~1-2%), especially with a float32 rescoring pass over top candidates |
| binary (1 bit/dim, sign only) | 1 bit/dim | ~32x | Larger (often 5-10%+), usually mitigated by over-fetching a bigger candidate set and reranking with the original float32 vectors |

**When to use:** Binary quantization + rescoring is the standard pattern for very large indexes (100M+ vectors) where storage and Hamming-distance search speed dominate cost. Int8 is a lower-risk default when you want most of the storage win with minimal accuracy loss. Voyage AI, Cohere, and OpenAI now expose native `output_dtype` options (`int8`, `uint8`, `binary`, `ubinary`) at embedding time, so quantization no longer requires a separate post-processing step.

---

## Similarity Metrics

All three metrics measure how close two vectors are. The choice matters for retrieval quality.

### Cosine Similarity

**Formula (plaintext):** similarity = (A · B) / (||A|| × ||B||)
- Normalizes by vector magnitude
- Range: [-1, 1], where 1 = identical direction, -1 = opposite
- Invariant to vector magnitude (doesn't care if vector is scaled)

**When to use:** Almost always. Default for embedding-based retrieval.

**When it fails:** Rarely. Vectors from the same embedding model are designed for cosine similarity.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

query_vec = np.array([1, 2, 3])
doc_vec = np.array([1, 2, 3.1])

sim = cosine_similarity([query_vec], [doc_vec])[0][0]
print(sim)  # Output: ~0.9998 (very close)
```

### Dot Product

**Formula:** similarity = A · B (no normalization)
- Sensitive to vector magnitude
- Range: [-∞, +∞]
- Faster to compute than cosine (no normalization step)

**When to use:** Only if the embedding model was explicitly trained with dot product (e.g., OpenAI `text-embedding-3-large` with "Matryoshka" training can use dot product).

**When it fails:** Most models are trained assuming normalized vectors (cosine). Using dot product on cosine-trained embeddings gives wrong results.

### Euclidean Distance

**Formula:** distance = √(Σ(A_i - B_i)²)
- Measures absolute distance in vector space
- Range: [0, ∞), where 0 = identical
- Sensitive to vector magnitude and dimensionality

**When to use:** Rarely in RAG. Some clustering algorithms use it.

**When it fails:** On high-dimensional sparse vectors; the curse of dimensionality makes Euclidean distance unreliable.

### Comparison Table

| Metric | Speed | Invariant to Scale? | Use in RAG? | Why / Why Not |
|--------|-------|-------------------|-----------|---------------|
| Cosine | Fast (normalized once, then dot product) | Yes | **Yes, default** | Designed for embedding vectors; scale-invariant |
| Dot Product | Fastest (just multiply) | No | Only if trained for it | Risky; requires model documentation |
| Euclidean | Moderate | No | No | Curse of dimensionality; unreliable in high dims |

### Code: Computing All Three

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

query = np.array([0.5, 0.3, 0.8, 0.2])
doc1 = np.array([0.5, 0.3, 0.8, 0.2])  # Identical
doc2 = np.array([0.1, 0.9, 0.2, 0.4])  # Different

# Cosine
cosine_sim_1 = cosine_similarity([query], [doc1])[0][0]  # 1.0 (identical)
cosine_sim_2 = cosine_similarity([query], [doc2])[0][0]  # ~0.15 (different)

# Dot Product
dot_prod_1 = np.dot(query, doc1)  # 1.02
dot_prod_2 = np.dot(query, doc2)  # 0.51

# Euclidean
euclidean_1 = np.linalg.norm(query - doc1)  # ~0.0 (identical)
euclidean_2 = np.linalg.norm(query - doc2)  # ~0.87 (different)

# Key insight: cosine ranks identically (doc1 > doc2) despite different magnitudes
# Euclidean depends on vector magnitude; cosine does not
```

**A more concrete, human-readable example:** embed the query "cancellation policy" into a vector, then run cosine similarity against every stored chunk vector. A similarity search over a policy-docs index might return the top-k chunks ranked by score — say, 0.89, 0.85, 0.81 — where the 0.89 chunk is the exact cancellation clause and the two runners-up are adjacent sections (e.g., "Claims Process," "Renewal Terms") that share vocabulary but aren't the answer. The numbers themselves aren't meaningful in isolation — what matters is the *relative ranking* they produce, which is why cosine similarity (invariant to scale) rather than raw dot product is the standard choice for ranking retrieval candidates.

---

## Embedding Quality Problems

Five failure modes that manifest in production RAG systems, with diagnostic tests for each.

### 1. Domain Mismatch

**The problem:** Embedding model trained on general text performs poorly on domain-specific terminology.

**Example:** Medical embeddings
- Query: "What is myocardial infarction?"
- General embedding model confuses this with general words about "muscle" and "damage"
- Medical-trained embeddings understand cardiology-specific semantics

**Detection:** Run retrieval on 20 domain-specific queries with labeled relevant documents. Compare NDCG@5 for general model vs. domain model. Gap >0.1 signals domain mismatch.

**Fix:** Fine-tune embeddings on domain data (see "Fine-Tuning Embeddings" section below).

### 2. Long-Document Degradation

**The problem:** Most embedding models truncate input at 512–8192 tokens. Longer documents lose information.

**Example:** A 50-page PDF
- Model truncates at 512 tokens → only first ~2 pages are embedded
- Later relevant content is lost
- Query for content on page 40 won't retrieve the document

**Detection:** Plot retrieval recall vs. document length on your corpus. Recall should be constant; if it drops for long documents, you have degradation.

**Fix:** Use a longer-context embedding model (E5-mistral-7b: 32K tokens) or chunk documents aggressively (covered in `01_concepts/chunking_strategies.md`).

### 3. Language / Script Mismatch

**The problem:** Monolingual embeddings fail on non-English text.

**Example:** English embeddings on Chinese text
- English models see Chinese characters as atomic; no semantic relationship
- Retrieval fails even for semantically similar Chinese documents

**Detection:** Test on queries/documents in your target language(s). If NDCG drops >50% vs. English, you need multilingual embeddings.

**Fix:** Use multilingual embeddings (mBERT, XLM-RoBERTa, or multilingual versions of BGE/E5) trained on 50+ languages.

### 4. Antonym Collapse

**The problem:** Embeddings of opposite words can be very similar.

**Example:** "profit" and "loss"
- Both are financial terms; appear in similar contexts
- Embedding model assigns similar vectors
- Queries for "profit margins" might incorrectly retrieve "loss analysis"

**Detection:** Embed antonym pairs (profit/loss, hot/cold, increase/decrease). Compute cosine similarity. If >0.5, you have antonym collapse.

**Fix:** Use a reranker as post-processing (covered in `01_concepts/reranking.md`) to catch these reversals.

### 5. Semantic Drift Over Time

**The problem:** Corpus terminology evolves; embeddings don't.

**Example:** "COVID" and "pandemic"
- Pre-2020, these words were unrelated
- Post-2020, they're synonymous
- Old embeddings (pre-2020) don't reflect this change
- Retrieval quality degrades over time

**Detection:** Re-run NDCG on a fixed probe set monthly. If NDCG drops >5% without corpus changes, semantic drift is likely.

**Fix:** Re-index corpus periodically (quarterly or semi-annually) with the latest embedding model.

---

## Fine-Tuning Embeddings

When off-the-shelf embeddings don't work, fine-tune them on your domain.

> See [Fine-Tuning for RAG](./fine_tuning.md) for when and how to fine-tune embedding models and rerankers.

### Data Requirements

- **Minimum:** 1,000 (query, relevant_doc) pairs
- **Recommended:** 5,000+ pairs
- **Very effective:** 10,000+ pairs

### Mining Training Data

Use your existing systems to generate pairs:

```python
# From click logs
def mine_from_clicks(click_logs):
    pairs = []
    for user_id, query, clicked_docs in click_logs:
        if len(clicked_docs) > 0:
            # Positive: a document the user clicked
            positive_doc = clicked_docs[0]
            # Negatives: documents that appeared but weren't clicked
            negative_docs = [doc for doc in all_retrieved_docs if doc not in clicked_docs]
            pairs.append((query, positive_doc, negative_docs))
    return pairs

# From feedback
def mine_from_feedback(feedback_logs):
    pairs = []
    for query, doc, rating in feedback_logs:
        if rating >= 4:  # Thumbs up
            pairs.append((query, doc, True))
        elif rating <= 2:  # Thumbs down
            pairs.append((query, doc, False))
    return pairs
```

### Fine-Tuning with sentence-transformers

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Prepare training data
train_examples = [
    InputExample(texts=["What is RAG?", "RAG stands for Retrieval-Augmented Generation..."], label=0.9),
    InputExample(texts=["What is RAG?", "The history of Ancient Rome"], label=0.1),
    # ... more examples
]

# Set up training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
)

model.save("./fine-tuned-embeddings")
```

### Evaluation Loop

```python
from sentence_transformers.util import pytorch_cos_sim

def evaluate_embeddings(model, test_queries, test_docs):
    """Compute NDCG@5 for evaluation set."""
    ndcg_scores = []
    
    for query, relevant_docs in test_queries:
        query_emb = model.encode(query, convert_to_tensor=True)
        doc_embs = model.encode(test_docs, convert_to_tensor=True)
        
        # Compute similarities
        similarities = pytorch_cos_sim(query_emb, doc_embs)[0]
        
        # Rank documents
        ranked = np.argsort(-similarities.cpu().numpy())
        
        # Compute NDCG@5
        ndcg = compute_ndcg(ranked[:5], relevant_docs)
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)

# Before fine-tuning
baseline_ndcg = evaluate_embeddings(model, test_queries, test_docs)  # e.g., 0.72

# After fine-tuning
finetuned_ndcg = evaluate_embeddings(model, test_queries, test_docs)  # e.g., 0.85

print(f"Improvement: {finetuned_ndcg - baseline_ndcg:.2%}")  # +13%
```

---

## Embeddings in the RAG Stack

How embedding model choice affects different RAG architectures.

| RAG Type | Embedding Requirement | Why | Recommendation |
|----------|----------------------|-----|-----------------|
| Naive RAG | Crucial; does 80% of work | Poor embeddings → poor retrieval | Use text-embedding-3-small minimum |
| Advanced RAG | Still crucial; reranker compensates for some embedding errors | Reranker catches embedding mistakes | Fine-tune if NDCG@5 <0.75 |
| Modular RAG | Depends on modules chosen | Sparsity module is embedding-agnostic | Start with text-embedding-3-small; specialize if needed |
| Adaptive RAG | Critical for routing decisions | Router classifier depends on embedding quality | Use robust, general embeddings (not domain-specific) |
| Agentic RAG | Crucial; agent relies on initial retrieval | Agent can't fix fundamental retrieval failures | Invest in good embeddings; agent won't save bad retrieval |
| Self-RAG | Critical; fine-tuning amplifies embedding quality | Feedback signal trains on top of embeddings | Use production-quality embeddings before fine-tuning |

---

## Key Takeaways

1. **Cosine similarity is the default.** Use it unless the model documentation explicitly says otherwise.
2. **MTEB >60 is production-grade.** Don't use models below this threshold without domain-specific justification.
3. **Domain mismatch is the #1 cause of RAG failures.** Test your embeddings on domain-specific queries early.
4. **Fine-tuning requires 1K+ labeled pairs.** Start with zero-shot embeddings; only fine-tune if your NDCG@5 <0.75.
5. **Embedding quality cascades.** Poor embeddings can't be fixed by better rerankers or LLMs. Fix retrieval first.

---

## Interview Q&A

**Q: How do you handle queries in low-resource languages where your embedding model has poor coverage?** `[Intermediate]`

Several options depending on budget: (1) **Multilingual embedding model** — swap to `multilingual-e5-large`, `LaBSE`, or `multilingual-bge` which are trained on 100+ languages and maintain cross-lingual alignment (an English query can match a French document). Quality is lower than monolingual models for high-resource languages but acceptable for many use cases. (2) **Translate-then-embed** — translate the query to English using a translation API before embedding; only works if your corpus is in English. Simple and high quality but adds latency and API cost. (3) **Fine-tune a multilingual model** on domain-specific cross-lingual pairs if off-the-shelf multilingual quality is insufficient. Test on a held-out set in each target language to confirm adequate coverage before deploying.

---

**Q: What is the query-document asymmetry problem, and how do models like HyDE, INSTRUCTOR, and E5 address it?** `[Advanced]`

**The problem:** Queries are short (3–15 words) and often keywords ("Python async error"), while documents are long and descriptive ("Python provides several mechanisms for asynchronous programming..."). Embedding both in the same space causes a distributional mismatch — the query vector rarely lands close to the document vector even when semantically relevant.

**How each approach addresses it:**
- **HyDE** (Hypothetical Document Embedding): generates a hypothetical full-text answer to the query, embeds *that* as the search vector. This puts the query in document-like distributional space. Risk: if the generated hypothesis is wrong, retrieval goes in the wrong direction.
- **INSTRUCTOR**: trained with task-specific instructions prepended to inputs ("Represent this query for searching technical documents: ..."). The instruction shifts the embedding to match the target distribution.
- **E5** (EmbEddings from bidirEctional Encoder rEpresentations): uses the prefix "query: " or "passage: " before text at both training and inference time, teaching the model that queries and passages should be handled differently even in the same embedding space.
- **Asymmetric fine-tuning**: fine-tune with query–passage pairs using different prompts/prefixes for each side.

---

## Related

- [Multimodal Embeddings](./multimodal_embeddings.md) — embedding cross-modal (image/audio) content into the same vector space
- [Fine-Tuning for RAG](./fine_tuning.md) — fine-tuning embedding models on domain-specific pairs
- [Reranking](./reranking.md) — cross-encoder reranking to catch bi-encoder mistakes like antonym collapse
- [Chunking Strategies](./chunking_strategies.md) — mitigating long-document truncation and context loss before embedding
