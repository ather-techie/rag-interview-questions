# 01 — Naive / Basic RAG

> The simplest RAG form: chunk → embed → store → retrieve → generate.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
[Offline Indexing]
Docs → Chunker (fixed-size / sliding window) → Embedding Model → Vector DB
                                                                     │
[Online Query]                                                       │
Query → Embedding Model → ANN Search (top-k) ────────────────────────┘
                                │
                                ▼
                        Top-k chunks → Generator LLM → Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| Document Loader / Chunker | Splits raw documents into fixed-size (or overlapping) chunks for embedding |
| Embedding Model | Converts text chunks and queries into dense vectors in the same semantic space |
| Vector Store | Persists chunk embeddings and metadata; serves ANN queries |
| Retriever (top-k ANN) | Finds the k most similar chunks to the query embedding via cosine/dot-product search |
| Generator LLM | Concatenates retrieved chunks into a prompt and produces the final answer |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Orchestration | LangChain, LlamaIndex |
| Vector Store | Chroma, Pinecone, Weaviate, pgvector, FAISS |
| Embedding Models | OpenAI `text-embedding-3-small/large`, BGE, E5 |
| Generator LLM | GPT-4o-mini, Claude, Llama 3 |

---

## Q1. What is Naive RAG and how does it work at a high level? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naive RAG is the foundational form of retrieval-augmented generation, following a fixed three-step pipeline:

1. **Indexing** — Documents are split into fixed-size chunks, each chunk is embedded into a vector, and stored in a vector database.
2. **Retrieval** — At query time, the query is embedded and the top-k most similar chunks are fetched via cosine (or dot-product) similarity.
3. **Generation** — Retrieved chunks are concatenated into a prompt and passed to the LLM to produce the final answer.

It is simple to implement but suffers from poor precision, no query understanding, and context fragmentation.

```
[Offline Indexing]
Docs → Chunker → Embedder ──► Vector DB
                                  ▲
[Online Query]                    │
Query → Embedder → ANN Search ────┘ → Top-k chunks → LLM → Answer
```

</details>

---

## Q2. What are the key limitations of Naive RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Limitation | Description |
|---|---|
| Chunking artifacts | Fixed-size splits can cut context mid-sentence, losing coherence |
| No query understanding | Raw query may not align with how content is stored |
| Low precision | Top-k cosine search can return irrelevant chunks |
| No feedback loop | Retriever and generator are not jointly optimized |
| Context stuffing | All chunks passed verbatim — no reranking or filtering |

</details>

---

## Q3. What embedding strategies are commonly used in Naive RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Dense embeddings** — Models like `text-embedding-3-small`, `BGE-large`, or `E5-mistral` map text to dense float vectors.
- **Sparse embeddings** — BM25 or TF-IDF produce sparse keyword-weighted vectors, better for exact-match queries.
- **Chunk overlap** — A sliding window (e.g., 50-token overlap) reduces boundary cutoff artifacts.
- **Sentence-level chunking** — Using sentence or paragraph boundaries instead of fixed token counts preserves semantic units.

The choice of embedding model heavily influences retrieval quality; domain-specific fine-tuned embeddings often outperform general-purpose ones.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)
results = vectorstore.similarity_search(query, k=5)
```

</details>

---

## Q4. How do you evaluate the quality of retrieval in a Naive RAG system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Common retrieval evaluation metrics:

- **Context Precision** — What fraction of retrieved chunks are actually relevant?
- **Context Recall** — What fraction of all relevant information was retrieved?
- **MRR (Mean Reciprocal Rank)** — Measures how high the first relevant chunk ranks.
- **NDCG** — Weighs relevant results appearing higher in the list more heavily.
- **RAGAS** — An open-source framework that evaluates faithfulness, answer relevance, context precision, and context recall end-to-end.

A common pitfall is optimizing only for recall (retrieve everything) at the cost of precision (retrieve junk).

</details>

---

## Q5. In what scenarios would you still choose Naive RAG over more complex approaches? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naive RAG remains valid when:

1. **Prototyping / POCs** — Speed of setup outweighs retrieval quality; you can iterate later.
2. **Small, clean corpora** — A compact, well-structured knowledge base achieves high precision without reranking.
3. **Low-latency requirements** — No reranking or multi-step retrieval means faster end-to-end response.
4. **Cost constraints** — Fewer API calls and no additional reranker model reduces operational cost.
5. **Narrow-domain queries** — When all documents are highly relevant and query distribution is predictable, Naive RAG performs surprisingly well.

Always profile your specific use case before over-engineering — Naive RAG often gets you 80% of the way.

</details>

---

## Q6. How do chunking strategies affect retrieval quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Chunking strategy directly impacts both retrieval precision and recall:

- **Fixed-size chunks (e.g., 512 tokens)** — Efficient and predictable, but may split semantic units, losing context boundaries.
- **Sliding window overlap** — Reduces boundary artifacts by repeating context; increases index size but improves continuity.
- **Semantic/sentence-based chunking** — Preserves natural language boundaries, reducing fragmentation but adding computational cost.
- **Hierarchical chunking** — Chunk documents into paragraphs, then sub-chunks; allows retrieval at multiple granularities.

```
[Naive: Fixed 512-token chunks]
"The CEO announced..."   |  "...a $100M acquisition"  |  "The deal closes next"
                                 ^ boundary cut loses context

[With sliding window 50-token overlap]
"The CEO announced..."        |  "announced a $100M acquisition deal"     |  "acquisition deal closes next"
        └─ preserved ─────────────────────────────────────────────────────────────┘
```

A practical benchmark: measure retrieval recall vs. index size. Overlap 10-15% of chunk size often balances both.

```python
# Compare strategies
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter
)

# Fixed chunk, no overlap
fixed = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)

# Sliding window
sliding = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

# Semantic (split by sentence, then combine up to 512 tokens)
semantic = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=512,
    chunk_overlap=50
)
```

</details>

---

## Q7. How does approximate nearest neighbor (ANN) search work in vector DBs? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Exact cosine similarity search is O(n·d) — infeasible at scale. Vector DBs use approximate indexing to trade small accuracy loss for massive speed gains.

| Technique | Data Structure | Query Time | Index Size | Best For |
|-----------|----------------|-----------|-----------|----------|
| **HNSW** | Hierarchical graph | O(log n) | +30% | General-purpose; balanced speed/memory |
| **IVF** | Inverted file clusters | O(k log n) | Compact | Very large (>10M) embeddings |
| **PQ** | Product quantization | O(1) lookup | Tiny | Mobile/edge; <1% recall loss tolerable |
| **LSH** | Hash buckets | O(1) lookup | Compact | Approximate fingerprinting |

**HNSW (Hierarchical Navigable Small World)** is the dominant production choice:

```
Level 2:     A ─── B
             │     │
Level 1:  C─ A ─── B ─── D
          │  │     │     │
Level 0:  C─ A ─── B ─── D ─── E ─── F
```

Query routing: start at top, traverse down to nearest neighbor at each level, then local search at Layer 0.

**Tuning parameters:**
- `M` (max neighbors per node): Higher M = more accurate but slower indexing.
- `ef` (search width): Higher ef = more accurate but slower queries.
- `efConstruction`: Controls indexing time.

Most vector DBs (Qdrant, Weaviate, Chroma) expose these knobs. A typical production setting: `M=16, ef=200, efConstruction=500` balances recall (~99%) and latency (<100ms).

</details>

---

## Q8. How do you handle document updates and deletions in a Naive RAG system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Updating or deleting documents in a vector store requires careful handling:

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **Full re-index** | Rebuild entire index from scratch | Simple; no stale data | Downtime; expensive for large corpora |
| **Soft delete** | Mark documents deleted; filter at query time | Zero downtime | Bloats index; requires filtering logic |
| **Versioned IDs** | Assign version tags (doc_v1, doc_v2); delete old versions | Rollback-safe; atomic | Complex ID management |
| **Lazy deletion** | Mark for deletion; compact during maintenance window | Balances simplicity & cost | Stale results until compact |
| **Hybrid: Update w/ new embed** | Delete old ID, insert new embedding with same semantic ID | Minimal downtime | Requires atomic operations |

**Best practice for production:**

1. Use soft delete with a `deleted_at` timestamp in metadata.
2. Filter deletions in query-time post-processing (cheap).
3. Periodically compact the index (off-peak) to reclaim space.
4. For critical documents, maintain a document version table (DB) separate from the vector index.

```python
# Soft-delete example with Chroma
vectorstore.delete(ids=["doc_123"])  # Soft marks as deleted

# Query-time filtering
results = vectorstore.similarity_search(query, k=10)
filtered = [r for r in results if not r.metadata.get("deleted")]
```

</details>

---

## Q9. What is semantic caching and how does it reduce Naive RAG latency? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Semantic caching stores embedding vectors of previous queries and reuses them if new queries are semantically similar, avoiding re-embedding and re-retrieval.

```
[Naive: No cache]
Query → Embed → Vector DB search → Chunks → LLM → Answer
 └─ 50ms    └─ 200ms              └─ 100ms

[With semantic cache]
Query → Embed → Cache lookup (similarity > 0.95?) ──[HIT]──> Cached chunks → LLM
                    └─ 5ms (fast)
                    └─[MISS]──> Vector DB search → Cache update
```

**Implementation:**

1. Embed the incoming query.
2. Search the cache (a smaller, in-memory vector DB or hash table).
3. If similarity > threshold (e.g., 0.95), reuse cached chunks.
4. Otherwise, perform normal retrieval and cache the result.

**Trade-offs:**

- Cache hit rate depends on query distribution (repetitive queries → high hit rate).
- Risk: cached results become stale if indexed documents change.
- Memory overhead for cache storage.

```python
from redis import Redis
import numpy as np

class SemanticCache:
    def __init__(self, threshold=0.95):
        self.cache = Redis(host='localhost')
        self.threshold = threshold
    
    def get_cached_result(self, query_embedding):
        """Check cache for semantically similar query."""
        cached_queries = self.cache.hgetall("query_embeddings")
        for cached_id, cached_vec in cached_queries.items():
            similarity = np.dot(query_embedding, np.frombuffer(cached_vec, dtype=np.float32))
            if similarity > self.threshold:
                return self.cache.get(f"result:{cached_id}")
        return None
    
    def cache_result(self, query_embedding, chunks, query_id):
        """Store query and retrieved chunks."""
        self.cache.hset("query_embeddings", query_id, query_embedding.tobytes())
        self.cache.set(f"result:{query_id}", json.dumps(chunks), ex=86400)  # 24h TTL
```

**Production systems (e.g., Anthropic's prompt caching)** use token-based caching at the LLM layer instead, which is more general but less semantic.

</details>

---

## Q10. Build an end-to-end Naive RAG system in under 50 lines of Python `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Here is a complete, runnable Naive RAG pipeline using LangChain, Chroma, and OpenAI:

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load and chunk documents
loader = TextLoader("document.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Create embeddings and store in vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunks, 
    embeddings,
    persist_directory="./chroma_db"
)

# 3. Build RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 4. Query
response = qa_chain({"query": "What is the main topic?"})
print(response["result"])
for doc in response["source_documents"]:
    print(f"Source: {doc.metadata}")
```

**Key points:**
- `RecursiveCharacterTextSplitter` handles heterogeneous documents (code, prose, tables).
- `OpenAIEmbeddings` uses the latest text-embedding-3-small (cheaper, better than ada-002).
- `RetrievalQA` wraps the retriever + LLM; `chain_type="stuff"` concatenates all chunks into one prompt.
- Persist the vector store to disk for reuse without re-embedding.

To extend this to production: add reranking (Q2), query rewriting (Q2), or a custom retriever with filtering (Q3).

</details>

---

## Q11. How do you estimate and optimize embedding and vector database storage costs as your Naive RAG corpus scales to tens of millions of documents? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Storage cost breakdown:**

1. **Embedding storage** — Each document chunk is embedded into a vector (e.g., 1536 dimensions for text-embedding-3-small).
   - Vector size: 1536 float32 = 6144 bytes per embedding.
   - 10M documents → 10M × 6144 bytes ≈ 61 GB of raw embedding data.
   - Compression (quantization) can reduce this by 4-10x.

2. **Vector database overhead** — Additional metadata (doc ID, source, chunk ID, timestamps).
   - ~500 bytes per record in metadata.
   - 10M documents → 5 GB metadata.
   - Total with embeddings: ~70 GB.

3. **Indexing overhead** — HNSW (Hierarchical Navigable Small World) or IVF (Inverted File) indexes add 10-30% extra storage.
   - Total: ~85 GB for 10M documents.

**Cost estimation:**

| Vector DB | Cost per GB/month | 10M docs (85 GB) | 100M docs (850 GB) |
|-----------|---------|---------|---------|
| **Pinecone (standard)** | $0.25 | $21.25 | $212.50 |
| **Weaviate (self-hosted on AWS)** | $0.10 (EC2 + EBS) | $8.50 | $85 |
| **Qdrant (self-hosted)** | $0.08 (compute + storage) | $6.80 | $68 |
| **FAISS (local)** | $0 (library cost) | ~$5 (server compute) | ~$30 |

**Embedding generation cost:**

Embedding 10M documents upfront:
- OpenAI text-embedding-3-small: $0.02 per 1M tokens.
- Assume 200 tokens per chunk (average).
- 10M × 200 = 2B tokens → $40.
- For 100M documents: $400.

**Optimization strategies:**

1. **Quantization** — Reduce vectors from float32 to int8 or binary, cutting storage by 4-10x.
   ```python
   from langchain.embeddings.base import Embeddings
   
   # Original: 1536 dims × 4 bytes = 6144 bytes
   # Quantized: 1536 dims × 1 byte = 1536 bytes (4x savings)
   
   vector_quantized = vector.astype(np.int8)
   ```
   Trade-off: slight loss in retrieval quality (~1-3% F1 drop).

2. **Hierarchical storage** — Store full embeddings for recent/hot documents, quantized embeddings for older data.
   - Hot tier (last 30 days): full precision.
   - Warm tier (30 days - 1 year): int8 quantization.
   - Cold tier (>1 year): archived to S3, re-embed on demand.

3. **Selective embedding** — Don't embed every document; use keywords/heuristics to index only high-value documents.
   - E.g., only embed documents with >5 views/month.
   - Saves 30-50% of embedding costs.

4. **Batch embedding and caching** — Embed documents in batches (1000 at a time) and cache results to avoid re-embedding.

5. **Index compression** — Use HNSW with reduced connectivity (M=8 instead of 16) to save index overhead.

**Example cost reduction (100M documents):**

```
Baseline:
- Embeddings: 100M × 6144 bytes = 600 GB → $60/month (at $0.1/GB)
- Metadata: 100M × 500 bytes = 50 GB → $5/month
- Index overhead: 50 GB → $5/month
- Total: $70/month

With optimizations:
- Quantization (4x): 150 GB → $15/month
- Selective embedding (40% reduction): 60M × 6144 bytes = 370 GB → $37/month (60M only)
- Total: $52/month (26% savings)
```

**Operational cost:**

Beyond storage, account for:
- Query latency (faster indexes cost more storage).
- Replication/backup (2-3x storage for HA).
- Data refresh (re-embedding cost if corpus changes).

Monitor storage utilization monthly and trigger optimization if >80% of capacity.

</details>

---

## Q12. How can an adversary poison a Naive RAG system by injecting malicious documents, and what defences prevent this? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Poisoning attack scenarios:**

1. **False information injection** — Attacker uploads a document claiming "Company X's product is dangerous" with high-quality writing to game retrieval ranking.
   - Naive RAG retrieves it as a top result.
   - LLM generates an answer that accepts the claim uncritically.
   - Downstream: misinformation spreads.

2. **Hallucination amplification** — Attacker injects a document that contradicts other sources but is more recent/authoritative-looking (e.g., fake press release).
   - Multiple conflicting sources confuse the LLM.
   - LLM may hallucinate a "synthesis" that favors the attacker's narrative.

3. **Adversarial embeddings** — Attacker crafts text that embeds near many innocent queries but contains malicious content.
   - Example: A fake recipe that embeds near "how to make brownies" but contains harmful instructions.
   - Naive systems retrieve it for innocent queries.

4. **Trojan embeddings** — Attacker fine-tunes a small "trigger" document that causes the embedding model to misbehave (only works if attacker controls embeddings).

**Defences:**

**1. Content moderation and scanning:**

Screen all ingested documents for:
- Prohibited content (hate speech, violence, explicit).
- Known malicious URLs or file hashes.
- Factual claims against a trusted knowledge base.

```python
from better_profanity import profanity
import hashlib

def screen_document(text, doc_id):
    # Profanity check
    if profanity.contains_profanity(text):
        return False, "Contains profanity"
    
    # Known bad hashes
    doc_hash = hashlib.sha256(text.encode()).hexdigest()
    if doc_hash in KNOWN_MALICIOUS_HASHES:
        return False, "Known malicious content"
    
    # Fact-check critical claims (e.g., via external API)
    if contains_medical_claims(text):
        verified = fact_check_api.verify(text)
        if not verified:
            return False, "Unverified medical claims"
    
    return True, "OK"
```

**2. Source attribution and trust scoring:**

Assign a trust score to each document based on source:
- Official company sources: 1.0
- Peer-reviewed publications: 0.95
- User-generated content: 0.5
- Unknown sources: 0.3

During retrieval, weight results by trust score:

```python
def retrieve_with_trust_weighting(query, k=5):
    results = vectorstore.search(query, k=k*2)  # Fetch 2x to allow filtering
    
    # Score by trust
    scored_results = [
        (doc, embedding_similarity * trust_score[doc.source])
        for doc, similarity in results
    ]
    
    # Return top-k by combined score
    return sorted(scored_results, key=lambda x: x[1], reverse=True)[:k]
```

**3. Retrieval diversity and contradiction detection:**

If multiple retrieved documents contradict each other, flag it:

```python
def detect_contradictions(retrieved_docs):
    embeddings = embed_documents([doc.text for doc in retrieved_docs])
    
    # Compute pairwise cosine similarity between documents
    for i, j in itertools.combinations(range(len(docs)), 2):
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        
        if similarity < 0.3:  # Low similarity suggests contradiction
            # Flag this to user or lower confidence
            return True, (retrieved_docs[i], retrieved_docs[j])
    
    return False, None
```

**4. LLM-level verification:**

Add a verification step where the LLM is asked to:
- Cite sources for each claim.
- Identify conflicting information.
- Assign confidence levels (high/medium/low).

```python
verification_prompt = f"""
Based on the retrieved documents, answer the query.
For each claim, cite the source and confidence (high/medium/low).
If documents contradict, explicitly note the conflict.

Query: {query}
Retrieved documents: {retrieved_docs}

Answer:
"""
```

**5. Continuous monitoring and feedback:**

Log all retrieved documents and user feedback:
- Did users find the answer helpful?
- Did users report misinformation?
- Flag documents with high negative feedback.

```python
def log_query_feedback(query_id, retrieved_docs, user_feedback):
    for doc in retrieved_docs:
        rating = user_feedback.get("helpful_docs", [])
        if doc.id not in rating:
            # Low user satisfaction with this doc
            doc.trustworthiness -= 0.05
            
            if doc.trustworthiness < 0.2:
                # Quarantine the document
                mark_for_review(doc.id)
```

**6. Rate limiting on ingestion:**

Prevent attackers from flooding the system with documents:
- Limit uploads per user/IP: 100 docs/day.
- Manual review for bulk uploads.
- Require identity verification for document sources.

**7. Differential privacy on embeddings:**

Add noise to embeddings to prevent adversarial optimization:

```python
def add_embedding_noise(embedding, epsilon=1.0):
    # Laplace noise scaled by privacy budget epsilon
    noise = np.random.laplace(0, 1 / epsilon, len(embedding))
    return embedding + noise
```

Trade-off: slight accuracy loss (~2-5% F1), but much harder to craft adversarial documents.

**Defence-in-depth approach:**

Combine multiple layers:
1. Content moderation (prevents obvious attacks).
2. Source trust scoring (deprioritizes untrusted sources).
3. Contradiction detection (alerts on inconsistencies).
4. User feedback loops (continuous improvement).
5. Regular audits (manual review of top-retrieved docs).

An attacker would need to defeat multiple layers, making poisoning expensive and risky.

**Monitoring dashboard:**

Track:
- % of ingested documents flagged by moderation.
- Average trust score of retrieved documents.
- User satisfaction per retrieved document.
- Documents with high contradiction flags.

</details>

---

## Real-World Applications

| Application | Domain | Why Naive RAG Fits |
|---|---|---|
| Internal HR / Policy chatbot | Enterprise | Static policy docs are updated infrequently; a fixed chunk-embed-retrieve pipeline is easy to maintain and audit |
| E-commerce product FAQ bot | Retail | Product specs and FAQs are structured, short, and don't require multi-hop reasoning — single-shot retrieval is sufficient |
| University knowledge base Q&A | Education | Course catalogues and FAQs have low query complexity; Naive RAG is fast to prototype and deploy on limited budgets |
| Customer support first-line triage | SaaS / Support | Answers common "how do I?" questions from documentation before escalating to agents |
| Small-team documentation search | Startups / DevTools | When the corpus is < 10k chunks, Naive RAG delivers adequate precision without orchestration overhead |
