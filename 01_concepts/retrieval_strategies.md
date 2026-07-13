# Retrieval Strategies: Beyond Top-k Cosine Search

> Beyond top-k cosine search — the full repertoire of retrieval techniques and when each one wins.

---

## What is Retrieval (in RAG)?

Retrieval is the step in a RAG pipeline that takes a user's query and returns the most relevant chunks or documents from an index to ground the LLM's answer. The simplest form is top-k cosine similarity search over embeddings, but "retrieval" in practice covers a much wider toolbox — hybrid search, query rewriting, multi-vector methods, and more — chosen based on what actually improves relevance for a given use case.

This only works because retrieval is *semantic*, not keyword-based: a query like "car won't start in cold weather" can surface a document about "battery performance in low temperatures" — different words, same underlying meaning — because their embeddings land close together in vector space. A keyword-only search would miss that document entirely.

---

## Retrieval as a Ranking Problem

Retrieval is not search. It's **ranking**: given a corpus and a query, surface the most relevant context within a latency budget.

The fundamental constraint: you cannot optimize all three simultaneously.

```
         Precision (relevance)
              ▲
              │
        Dense │ Sparse-only ✗
        only  │   (too slow)
         ✗   │
              │  Dense + Rerank ✓✓✓
              │      ●
              │     ●●
              │    ● ●
         Sparse├─●───────────► Latency
         only  ● ●  
         ✓    ●   ●
              │     ●
         Dense├──────●
         only ●       ●
              │  Hybrid ✓✓
              │
         Recall
```

**Dense retrieval:** Fast, semantic, but misses exact matches
**Sparse (BM25):** Slow, keyword-based, but catches exact matches
**Hybrid:** Balanced, but requires merging two rankings
**Dense + Rerank:** Best precision, but needs two passes

---

## Dense Retrieval (Bi-Encoder)

The baseline in almost every RAG system.

**How it works:**
1. Offline: Embed all documents with a bi-encoder (e.g., `text-embedding-3-small`)
2. Index: Store vectors in a vector DB (HNSW, IVF, etc.)
3. Online: Embed query, run ANN search, return top-k

```python
from sentence_transformers import SentenceTransformer
import faiss

# Offline: embed corpus
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = ["RAG is...", "Embeddings are...", ...]
embeddings = model.encode(documents)

# Index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Online: query
query = "What is RAG?"
query_embedding = model.encode(query)
distances, indices = index.search(query_embedding.reshape(1, -1), k=5)
results = [documents[i] for i in indices[0]]
```

**Strengths:**
- Captures semantic intent and paraphrases
- Fast (single ANN call)
- Generalizes across domains (zero-shot)

**Weaknesses:**
- Exact matches miss ("product code XYZ123" won't match a query with "XYZ123")
- Out-of-vocabulary terms (acronyms, proper nouns)
- Antonym collapse (opposite words embed similarly)

---

## Sparse Retrieval: BM25

The statistical baseline. Still beats dense on certain queries.

**BM25 Formula (plaintext):**

For each document D and query Q:
```
score(D, Q) = sum over query terms T of:
  IDF(T) × (f(T, D) × (k1 + 1)) / (f(T, D) + k1 × (1 - b + b × (len(D) / avglen)))
  
where:
  IDF(T) = log((N - n(T) + 0.5) / (n(T) + 0.5))
  f(T, D) = frequency of term T in document D
  N = total documents
  n(T) = documents containing T
  k1 = 1.5 (term saturation parameter)
  b = 0.75 (length normalization)
```

**Intuition:** Reward term frequency + penalize document length + down-weight common terms

**Code:**

```python
from rank_bm25 import BM25Okapi

# Tokenize corpus
corpus = ["RAG is retrieval augmented generation", "Embeddings are dense vectors"]
tokenized_corpus = [doc.split() for doc in corpus]

# Build BM25
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "What is RAG"
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)

# Top-k
top_k_indices = np.argsort(-scores)[:5]
```

**Strengths:**
- Exact match (keyword exact = high score)
- Fast (no embeddings needed)
- Transparent (interpretable scores)

**Weaknesses:**
- No semantics (paraphrases don't match)
- Sensitive to tokenization
- Poor on short queries ("embeddings" vs "embedding")

---

## Hybrid Retrieval: Dense + Sparse

Combine both to get the best of both worlds.

**Reciprocal Rank Fusion (RRF):**

```
Query: "BERT transformer attention"
    │
    ├─ Dense search → [doc1: rank 1, doc2: rank 2, doc3: rank 5]
    ├─ Sparse (BM25) search → [doc3: rank 1, doc1: rank 4, doc4: rank 2]
    │
    ├─ Compute RRF scores
    │  doc1: 1/(60+1) + 1/(60+4) = 0.0164 + 0.0154 = 0.0318
    │  doc3: 1/(60+5) + 1/(60+1) = 0.0152 + 0.0164 = 0.0316
    │  doc2: 1/(60+2) + 0 = 0.0161
    │  doc4: 0 + 1/(60+2) = 0.0161
    │
    └─ Final ranking: doc1, doc3, doc2, doc4
```

**Why RRF works:** It's rank-based, not score-based. Robust to different score distributions.

**Code:**

```python
import numpy as np

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """Merge dense and sparse results using RRF."""
    scores = {}
    
    # Dense contributions
    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Sparse contributions
    for rank, doc_id in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]
```

**Benchmark Results (Formal et al., 2021):**
- Dense alone: Recall@10 = 0.68
- Sparse alone: Recall@10 = 0.62
- Hybrid (RRF): Recall@10 = 0.76 (+11% over best single)

---

## Query-Side Transformations

Modify the query to improve retrieval.

### Query Rewriting

**Idea:** Reformulate an ill-formed or context-dependent query into a clean, self-contained one *before* it's embedded — distinct from the expansion/decomposition techniques below, which operate on an already well-formed query.

- User asks: *"y is my app slow after last update"* → rewritten to *"What are the common causes of application performance degradation after a software update?"* — shorthand expanded into something retrievable.
- User asks a follow-up with a dangling pronoun: *"How do I fix it?"* (after an earlier question about OOM errors) → rewritten using conversation history to *"How do I fix an out-of-memory (OOM) error in a Kubernetes pod?"*

Without rewriting, both queries embed to something too thin or too ambiguous to retrieve well — the pronoun case in particular loses all its meaning once conversation context is dropped.

### HyDE (Hypothetical Document Embeddings)

**Idea:** Generate a hypothetical answer; embed that; use it for retrieval.

**Why it works:** Hypothetical answers are closer to relevant documents in embedding space than the original query.

```
Query: "What is RAG?"
   │
   ├─ Generate hypothetical answer (LLM)
   │  "RAG stands for Retrieval-Augmented Generation. It's a technique where..."
   │
   ├─ Embed hypothetical answer
   │
   └─ Search vector DB with hypothesis embedding
      └─ Retrieve docs similar to the hypothesis
         └─ More relevant than searching with original query!
```

**Code:**

```python
def hyde(query: str, llm, embedding_model, vector_db):
    # Generate hypothetical answer
    prompt = f"Provide a detailed answer to: {query}"
    hypothesis = llm.generate(prompt)
    
    # Embed hypothesis, retrieve
    hyp_embedding = embedding_model.encode(hypothesis)
    results = vector_db.search(hyp_embedding, k=5)
    
    return results
```

### Multi-Query Expansion

**Idea:** Generate multiple phrasings of the same query; retrieve for all; union results.

```
Query: "How to fine-tune embeddings?"
   │
   ├─ Paraphrase 1: "Adapt embeddings to domain"
   ├─ Paraphrase 2: "Domain-specific embedding training"
   ├─ Paraphrase 3: "Embedding model tuning"
   │
   ├─ Retrieve for each
   │  Results1: [doc1, doc2, doc3]
   │  Results2: [doc2, doc4, doc5]
   │  Results3: [doc1, doc3, doc6]
   │
   └─ Union + rank by frequency
      Final: [doc1 (2x), doc2 (2x), doc3 (2x), doc4, doc5, doc6]
```

### Step-Back Prompting (Zheng et al., 2023)

**Idea:** Generalize the query to a broader concept; retrieve at that level.

```
Query: "What is the gradient descent update rule?"
   │
   ├─ Step back: "How do optimization algorithms work?"
   │
   └─ Retrieve documents about optimization
      └─ Get foundational context before diving into gradients
```

---

## Context-Side Strategies

Optimize what context is returned.

### MMR (Maximal Marginal Relevance)

**Idea:** Retrieve for relevance AND diversity. Avoid redundant results.

**Formula:**
```
score(doc_i) = λ × relevance(doc_i, query) - (1 - λ) × max(similarity(doc_i, selected_docs))
```

**Intuition:** Penalize similarity to already-selected docs. Force diversity.

```python
def mmr_rerank(query_embedding, candidate_embeddings, lambda_param=0.5, k=5):
    scores = []
    selected = []
    
    for candidate in candidate_embeddings:
        relevance = cosine_similarity(query_embedding, candidate)
        
        # Diversity penalty: similarity to already selected
        redundancy = 0
        for selected_doc in selected:
            redundancy = max(redundancy, cosine_similarity(candidate, selected_doc))
        
        mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
        scores.append((candidate, mmr_score))
    
    # Sort and select top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores[:k]]
```

**When to use:** Search results where redundancy hurts (news, QA — you want diverse perspectives)

### Contextual Compression

**Idea:** Retrieve large chunk, use LLM to extract only relevant sentence.

```
Retrieved Chunk (200 tokens):
  "RAG systems combine retrieval and generation. This enables LLMs to
   reference external knowledge. The retrieval step uses embeddings...
   [100 more tokens about unrelated topics...]"
   
   │
   └─ Compress: "Extract the most relevant sentence for: What is RAG?"
   
   Result (10 tokens):
   "RAG systems combine retrieval and generation to enable LLMs to
    reference external knowledge."
```

**Code:**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMCompressor

compression = LLMCompressor.from_llm_and_prompt(llm, prompt_template)
retriever = ContextualCompressionRetriever(
    base_compressor=compression,
    base_retriever=dense_retriever
)

compressed_docs = retriever.get_relevant_documents(query)
```

---

## Multi-Hop and Iterative Retrieval

Handle questions that require multiple retrieval steps.

**The Problem:** Single retrieval often can't answer multi-hop questions.

```
Question: "Which author won a Turing Award and wrote about neural networks?"

Step 1: Retrieve "Turing Award winners"
  → [Alan Turing, Geoffrey Hinton, ...]

Step 2: For each winner, retrieve "their papers about neural networks"
  → Geoffrey Hinton: Many papers on backprop, RNNs, etc.

Result: Geoffrey Hinton is the answer
```

### FLARE (Jiang et al., 2023)

**Mechanism:** Generate tentatively. When uncertain, pause and re-retrieve.

```
Query: "Who is the author of BERT and what are their other works?"

Generate (tentative):
  "BERT was authored by [pause: uncertain]..."
  
Detect uncertainty:
  "I don't know who wrote BERT"
  
Re-retrieve:
  Query: "Who wrote BERT paper"
  Result: "Devlin et al., 2018"
  
Continue generating:
  "BERT was authored by Jacob Devlin et al. in 2018. 
   Their other works include..."
```

---

## Strategy Selection Guide

| Query Type | Recommended Strategy | Why |
|---|---|---|
| **Exact match** (product codes, IDs) | Sparse (BM25) or Hybrid | Dense misses exact keywords |
| **Semantic** (definition, explanation) | Dense | Captures paraphrases |
| **Multi-hop** (requires reasoning) | Multi-query expansion or FLARE | Single retrieval insufficient |
| **Ambiguous** (multiple valid answers) | MMR | Avoid redundancy |
| **Domain-specific** (medical, legal) | Dense + Rerank | Domain embeddings essential |
| **Short query** (<3 words) | Hybrid | Dense struggles with short queries |
| **Long query** (full sentence+) | Dense + HyDE | HyDE helps with long context |

---

## Decision Tree: Which Strategy to Start With

```
Question 1: Does your corpus contain exact-match queries?
  ├─ Yes → Use Hybrid (Dense + BM25 with RRF)
  └─ No → Use Dense
  
Question 2: Do queries require multiple hops?
  ├─ Yes → Add multi-query expansion or FLARE
  └─ No → Stick with above
  
Question 3: Do you need diverse results?
  ├─ Yes → Add MMR post-processing
  └─ No → Stick with above
  
Question 4: Is latency critical (<200ms)?
  ├─ Yes → Remove reranking, compression
  └─ No → Can add reranking
```

---

## Key Takeaways

1. **Start with dense retrieval.** It's fast and works for most cases.
2. **Add hybrid (RRF) if exact matches matter.** RRF adds <50ms latency.
3. **Use HyDE for short or vague queries.** LLM-generated hypotheses often outrank original query.
4. **MMR is worth it for diverse results.** Don't use unless you specifically need it.
5. **Multi-hop requires multi-query or FLARE.** Single retrieval can't bridge reasoning gaps.

---

## ColBERT: Multi-Vector Late Interaction Retrieval

*Introduced by Omar Khattab et al. (Stanford, 2020); significantly improved in ColBERTv2 (2022). Practical deployment via RAGatouille library.*

### What It Is

Standard bi-encoders compress an entire document into a **single vector**. ColBERT takes a different approach: encode the query and document into **one vector per token**, then score them with a **MaxSim** operation at query time.

```
Standard bi-encoder:
  Query "memory leak Python" → [0.2, 0.8, ...] (1 vector, 1536 dims)
  Document                  → [0.3, 0.7, ...] (1 vector, 1536 dims)
  Score = dot_product(query_vec, doc_vec)

ColBERT:
  Query "memory leak Python" → [[q1], [q2], [q3]] (3 token vectors)
  Document                  → [[d1], [d2], ..., [dN]] (N token vectors)
  Score = Σ_i max_j(cosine(qi, dj))  ← MaxSim: each query token finds its best matching doc token
```

### Why It Outperforms Bi-Encoders on Specialized Domains

Single-vector bi-encoders must compress all semantic information into one fixed-size vector. For short queries or domain-specific terms, this compression loses nuance.

ColBERT's per-token representation allows precise matching: if the query contains the technical term "CUDA memory leak", each token's vector finds the best matching token in the document — even if the document uses "GPU memory exhaustion" (synonymous but different tokens).

**Benchmark improvements (BEIR benchmark):**
- General-domain: ColBERT ≈ bi-encoder (similar)
- Domain-specific (medical, legal, scientific): ColBERT outperforms by 8–15% Recall@10

### Architecture

| Stage | ColBERT | Bi-encoder |
|---|---|---|
| **Indexing** | Encode every token in every document; store all token vectors | Encode each document as one vector |
| **Storage** | O(N × avg_tokens × dim) — ~100–200× more storage than bi-encoder | O(N × dim) |
| **Query encoding** | Encode query tokens (fast, done at query time) | Encode query as one vector |
| **Scoring** | MaxSim over all (query_token, doc_token) pairs for top-k candidates | Dot product of single vectors |
| **Latency** | Higher (MaxSim is expensive for large candidate sets) | Lower |

### Practical Deployment: RAGatouille

[RAGatouille](https://github.com/answerdotai/ragatouille) wraps ColBERT in a simple API:

```python
from ragatouille import RAGPretrainedModel

# Load a pre-trained ColBERT model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents
RAG.index(
    collection=["Document 1 text...", "Document 2 text...", ...],
    index_name="my_index",
    max_document_length=256,  # Tokens per document chunk
    split_documents=True
)

# Retrieve
results = RAG.search(query="Python memory leak in Django", k=5)
# Returns: list of {content, score, rank, document_id}
```

### When to Use ColBERT

| Scenario | Use ColBERT | Use Bi-encoder |
|---|---|---|
| Domain-specific vocabulary (medical, legal, code) | ✓ | |
| High-precision retrieval is critical | ✓ | |
| Storage cost is a constraint (large corpus) | | ✓ |
| Latency < 100ms is required | | ✓ (ColBERT can be slow) |
| General-domain open QA | | ✓ (bi-encoder sufficient) |
| Already using a reranker | | ✓ (reranker closes the gap) |

### Trade-off Summary

- **Storage:** A 1M-document corpus with avg 100 tokens/doc requires ~150B vectors. At 128 dims × 2 bytes = 256 bytes/vector: ~38 TB. Compression (scalar quantization) brings this to ~5–10 TB — significant but feasible for high-value use cases.
- **Latency:** ColBERT with PLAID (efficient indexing) achieves <100ms for 1M docs. Vanilla ColBERT is slower.
- **Quality ceiling:** On benchmarks where a good reranker closes the gap (general domain), the storage and latency cost of ColBERT may not be worth it. On specialized domains without large reranker training data, ColBERT's advantage persists.

---

## Additional Interview Questions

**Q: How do you handle conflicting information across retrieved passages?**

First, detect the conflict: check whether multiple passages make contradictory factual claims about the same subject (NLI-based contradiction detection, or simply prompting the LLM). Handling options: (1) **Merge with attribution** — present all conflicting claims, each with its source, and let the LLM or the user decide; (2) **Majority vote** — if 3 passages agree and 1 disagrees, prefer the majority position and flag the minority; (3) **Confidence weighting** — prefer passages from more recent documents or higher-authority sources (journal papers > blog posts); (4) **Abstain** — for high-stakes domains (medical, legal), if passages conflict, surface the conflict explicitly rather than picking a side. Never silently resolve a conflict by choosing one passage arbitrarily.

---

**Q: How does retrieval change for multi-document synthesis queries vs. single-fact lookup?**

Single-fact lookup: precision matters most — retrieve the one most relevant chunk, use a high similarity threshold, and prefer reranking to push the exact answer to the top. Multi-document synthesis (e.g., "Compare the approaches taken by these three papers"): recall and diversity matter more — use MMR to avoid redundant chunks from the same document, retrieve a larger k (10–20 instead of 3–5), lower the similarity threshold to capture different perspectives, and pass multiple documents explicitly labeled to the LLM. The generation prompt also changes: "Synthesize across the following documents" rather than "Answer using the following passage."

---

**Q: How would you design retrieval for a corpus of billions of documents?**

A billion-document corpus requires a two-tier retrieval architecture: (1) **Coarse retrieval** — use a combination of BM25 (inverted index, scales to any size) and IVF+PQ quantized dense embeddings to retrieve a candidate set of ~1000–10000 documents from the full corpus. Partition the index (by topic, date, or document type) and route queries to the relevant shard. (2) **Fine-grained re-ranking** — apply a cross-encoder or ColBERT reranker over the candidate set. At this scale, self-hosted FAISS with IVF+PQ (128-dim quantized to 8 bits) is essential — managed services like Pinecone are prohibitively expensive at 1B+ vectors. Use hierarchical navigable small world (HNSW) within each partition for speed, and distribute shards across multiple machines.
