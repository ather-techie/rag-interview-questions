# 30 — ColRAG / ColBERT-Based RAG

> Multi-vector late interaction: every token in the query scores against every token in each document, giving bi-encoder speed with cross-encoder-like precision.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query                                     Document Corpus
  │                                            │
  ▼                                            ▼
Per-Token Query Encoder              Per-Token Document Encoder
  │  [q₁, q₂, ..., qₘ]                         │  [d₁, d₂, ..., dₙ] per doc (offline)
  │                                            ▼
  │                                   Token-level ANN Index (PLAID)
  │                                            │
  │                                            ▼
  │                                   Fast Candidate Pre-filter (top 100–1000)
  │                                            │
  └───────────────► MaxSim Late-Interaction Scorer ◄──────────────┘
                     score = Σᵢ max_j( qᵢ · dⱼ )
                              │
                              ▼
                    (optional) Cross-Encoder Reranker
                              │
                              ▼
                    Top-k Passages → LLM Generation
```

### Key Components

| Component | Responsibility |
|---|---|
| Per-token Query Encoder | Encodes the query into one embedding per token instead of a single pooled vector |
| Per-token Document Encoder | Encodes each document into one embedding per token, computed offline and stored in the index |
| Token-level ANN Index | Stores compressed per-token vectors and narrows the corpus to a fast candidate set |
| MaxSim Late-Interaction Scorer | For each query token, finds its best-matching document token and sums the scores |
| Optional Reranker | Refines the top candidates further when extra precision is needed |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Model | ColBERTv2 (`colbert-ir/colbertv2.0`) |
| Index | PLAID index with residual (2-bit) compression |
| Wrapper library | RAGatouille |
| Integration | `RAGatouilleLangChainRetriever`, LlamaIndex ColBERT integrations |

---

## Q1. What is ColBERT and how does it differ from standard bi-encoders? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**ColBERT** (Contextualized Late Interaction over BERT) is a retrieval model that produces **one embedding per token** rather than one embedding per document. This enables "late interaction" — the query and document representations are compared at the token level, not the sentence level.

**Standard bi-encoder (dense retrieval):**
```
Query: "How does RAG work?"
    │
    ▼
[BERT encoder] → single vector q ∈ ℝ^768
                                    │
                                    ▼
                             cosine_sim(q, d) per doc
```

**ColBERT:**
```
Query: "How does RAG work?"
    │
    ▼
[BERT encoder] → one vector per token: [q₁, q₂, q₃, q₄] ∈ ℝ^(4×128)

Document: "RAG retrieves documents..."
    │
    ▼
[BERT encoder] → one vector per token: [d₁, d₂, ..., d₂₀] ∈ ℝ^(20×128)

MaxSim scoring:
  score = Σᵢ max_j( qᵢ · dⱼ )
          ↑ for each query token, find best matching doc token → sum
```

**MaxSim** ensures every query token can find its best match in the document independently, capturing token-level semantic alignment that a single document vector cannot.

**Key comparison:**

| Property | Bi-encoder | ColBERT |
|----------|-----------|---------|
| Embeddings per doc | 1 | N (one per token) |
| Index size | Small | Large (N× larger) |
| Query-time compute | Dot product | MaxSim (batched matmul) |
| Quality | Good | Better on hard queries |
| Storage cost | Low | High |

</details>

---

## Q2. How does ColBERT achieve bi-encoder-like speed with cross-encoder-like quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

ColBERT achieves this through **precomputed document token embeddings** and **efficient MaxSim computation**.

**Step 1 — Offline indexing (done once):**
```
For each document d:
    encode d → [d₁, d₂, ..., dₙ]    (n token vectors)
    store all token vectors in a compressed index
```

**Step 2 — Query time:**
```
encode query q → [q₁, q₂, ..., qₘ]  (m token vectors, where m << n)

For each candidate document (from a fast ANN pre-filter):
    MaxSim(q, d) = Σᵢ max_j (qᵢ · dⱼ)
    
This is a batched matrix multiply — fast on GPU.
```

**Why it's fast:** Document token vectors are precomputed. Query tokens are small (typically < 32 tokens). The MaxSim for a query against 1000 candidates is a batched matrix operation executable in ~10ms on GPU.

**Why it's accurate:** Each query token can match the most relevant document token independently. For a query like "transformer attention mechanism", the token "attention" finds an exact match in documents that discuss self-attention even if the document never uses the phrase "transformer attention mechanism" as a unit.

**Two-stage pipeline used in practice:**

```
Stage 1: ANN retrieval (fast, approximate)
         → Retrieve 100–1000 candidate document IDs
           using compressed doc-level vector (mean of token embeddings)

Stage 2: ColBERT MaxSim re-scoring (precise)
         → Load token embeddings for candidates
         → Compute MaxSim → rerank → top-k final results
```

This amortizes the MaxSim cost over a small candidate set, achieving latency similar to a standard reranker but with better quality.

</details>

---

## Q3. How do you build a ColBERT index and integrate it into a RAG pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Using RAGatouille (the recommended production library):**

```python
from ragatouille import RAGPretrainedModel

# Load a pre-trained ColBERTv2 model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index your corpus
RAG.index(
    collection=["Document 1 text...", "Document 2 text...", ...],
    index_name="my_knowledge_base",
    max_document_length=256,     # tokens per passage
    split_documents=True         # auto-split long docs into passages
)
```

**Querying:**
```python
results = RAG.search(
    query="What is the difference between RAG and fine-tuning?",
    k=5    # top-5 passages
)

# results is a list of dicts:
# [{"content": "...", "score": 24.7, "rank": 1, "document_id": "..."}, ...]
```

**Full RAG pipeline integration:**
```python
from anthropic import Anthropic

client = Anthropic()

def colbert_rag(query: str) -> str:
    # 1. ColBERT retrieval
    hits = RAG.search(query=query, k=5)
    context = "\n\n".join(hit["content"] for hit in hits)
    
    # 2. LLM generation
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

**LangChain integration:**
```python
from ragatouille.integrations.langchain import RAGatouilleLangChainRetriever

retriever = RAGatouilleLangChainRetriever(model=RAG, k=5)
# Use as a drop-in replacement for any LangChain retriever
```

**Index storage:** ColBERT indexes are stored on disk (as `.pt` files) and loaded into memory at query time. For a 1M-passage corpus with 128-dim vectors: ~1M × 100 tokens/passage × 128 dims × 2 bytes ≈ **25 GB** — significantly larger than a single-vector index (~1M × 768 × 4 bytes ≈ 3 GB). Plan for this storage delta.

</details>

---

## Q4. What are the trade-offs of ColBERT vs. a standard dense retriever + cross-encoder reranker? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Dimension | ColBERT | Dense + Cross-Encoder |
|-----------|---------|----------------------|
| **Retrieval quality** | Very high | High (comparable or slightly lower) |
| **Latency (total)** | 20–80ms | 25–200ms |
| **Index size** | Large (10–30× dense) | Small (dense) |
| **Serving complexity** | Single model, custom index | Two models, standard vector DB |
| **Re-indexing cost** | High (re-embed all tokens) | Medium (re-embed one vector per doc) |
| **Domain adaptation** | Fine-tune ColBERT end-to-end | Fine-tune retriever or reranker independently |
| **Passage length limit** | ~256–512 tokens | Up to model context window |

**When ColBERT wins:**
- Hard retrieval problems where query-document lexical overlap is low
- Queries with multiple independent concepts ("Python async error handling in FastAPI")
- Budget constraint that rules out separate reranker API calls
- Need for sub-100ms end-to-end latency including ranking

**When standard dense + reranker wins:**
- Existing vector DB infrastructure (Pinecone, Weaviate, Qdrant) — ColBERT needs a specialized index
- Need to rerank across modalities (text + metadata filtering)
- Corpus exceeds the storage budget for token-level embeddings
- Need for explainability at the passage level (ColBERT token scores are harder to surface to users)

**Compression trick to reduce ColBERT storage:**
ColBERTv2 uses residual compression — token vectors are compressed to ~2 bits per dimension using Product Quantization, reducing index size by 10× with < 3% quality degradation.

```python
RAG.index(
    collection=passages,
    index_name="compressed_index",
    nbits=2    # 2-bit quantization (ColBERTv2 default)
)
```

</details>

---

## Q5. How do you fine-tune a ColBERT model for a specific domain? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

ColBERT fine-tuning uses the same contrastive learning setup as bi-encoder fine-tuning, but the loss is computed with MaxSim scoring instead of dot product.

**Training data format (triplets):**
```
(query, positive_passage, negative_passage)
```

**Fine-tuning with ColBERT's training loop:**
```python
from colbert.training.trainer import Trainer
from colbert.infra import Run, RunConfig, ColBERTConfig

with Run().context(RunConfig(experiment="domain-ft")):
    config = ColBERTConfig(
        bsize=16,
        lr=1e-5,
        warmup=2000,
        dim=128,
        doc_maxlen=256,
        mask_punctuation=True,
    )
    
    trainer = Trainer(
        triples="path/to/training_triples.tsv",   # (qid, pos_pid, neg_pid)
        queries="path/to/queries.tsv",
        collection="path/to/passages.tsv",
        config=config,
    )
    
    trainer.train(checkpoint="colbert-ir/colbertv2.0")
```

**Generating training data for a new domain:**
1. Collect query → relevant document pairs from user click logs or expert annotation
2. Mine hard negatives: for each query, retrieve top-50 with BM25, exclude known positives — the top-10 false positives are the hardest negatives
3. Optionally: use GPL (Generative Pseudo-Labeling) to generate synthetic query–positive pairs from your corpus

**When to fine-tune vs. use ColBERTv2 off-the-shelf:**
Fine-tuning is warranted when: (a) your domain has specialized vocabulary (medical, legal, code), (b) standard benchmarks show Recall@10 below 0.70 with the base model, or (c) queries have structural patterns the base model hasn't seen (code search, SQL queries).

</details>

---

## Real-World Applications

- **Vespa.ai** uses ColBERT-style multi-vector scoring in production at scale
- **Stanford BEIR benchmark**: ColBERTv2 achieves state-of-the-art on zero-shot retrieval across 18 diverse datasets
- **Code search**: GitHub Copilot and similar systems use multi-vector representations for token-level code matching
- **Legal RAG**: Multi-vector retrieval improves recall on clause-level legal document search where key terms can appear anywhere in a passage
