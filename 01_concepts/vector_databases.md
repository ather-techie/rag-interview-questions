# Vector Databases: Storage and Search for Embeddings

> The storage and search layer every RAG system depends on — how vector databases work, how they differ, and how to choose one.

---

## What is a Vector Database?

A vector database is a storage system purpose-built to hold embeddings and quickly find the ones most similar to a given query vector, typically using approximate nearest-neighbor (ANN) indexing rather than brute-force comparison. It's the component that sits between "I have millions of embedded chunks" and "give me the top-k most relevant ones in milliseconds." Beyond raw similarity search, most vector databases also handle metadata filtering, hybrid (keyword + vector) search, and horizontal scaling as your index grows.

---

## What a Vector Database Does

A vector database has three responsibilities:

1. **Persist embeddings:** Store vectors at scale (millions to billions)
2. **Index for ANN search:** Enable approximate nearest neighbor (ANN) search in milliseconds
3. **Filter by metadata:** Support metadata predicates (namespace, document_id, tags, etc.)

### Three Deployment Models

**Vector Library (In-Process)**
- Examples: FAISS, Annoy, Sklearn
- Deployment: Code library; vectors in memory or on disk
- Pros: Simple, no network latency, free
- Cons: Single-machine scale only; no concurrent writes; no multi-tenancy

**Vector Database (Self-Hosted)**
- Examples: Qdrant, Milvus, Weaviate
- Deployment: Separate service you run
- Pros: Multi-machine scale, concurrent access, managed backups
- Cons: You manage infrastructure; requires DevOps

**Managed Vector Service (Cloud)**
- Examples: Pinecone, Weaviate Cloud, Milvus Cloud
- Deployment: Cloud service; vendor manages infrastructure
- Pros: Auto-scaling, backups, enterprise support
- Cons: Cost, vendor lock-in, latency (network)

### Three Architecture Layers

```
Client (Embedding Service)
    │
    ├──► Query Vector (Embedding) [or metadata filter]
    │
    ├──► Vector DB Service
    │    │
    │    ├─ Indexing Layer
    │    │  └─ Algorithm: HNSW / IVF / PQ
    │    │
    │    ├─ Filtering Layer
    │    │  └─ Metadata predicates (where document_id = X)
    │    │
    │    └─ Storage Layer
    │       └─ Disk: vectors, index, metadata
    │
    └──► Result: Top-k vectors + metadata
```

---

## Indexing Algorithms Deep Dive

### HNSW (Hierarchical Navigable Small World)

**Concept:** A multi-layer graph structure. Each layer is sparser than the previous, enabling fast navigation.

```
Layer 2 (sparse):     1 ──────── 5
                      │          │
Layer 1 (medium):    1 ─ 2 ─ 3 ─ 4 ─ 5
                     │ X   X X X │
Layer 0 (dense):    1─2─3─4─5─6─7─8─9  (all vectors)
```

**Query Flow:**
1. Enter at top layer
2. Greedily navigate toward query vector (nearest neighbor to query)
3. Drop to next layer, start from neighbor in previous layer
4. Repeat until bottom layer
5. Return top-k from bottom layer

**Tuning Parameters:**
- `M`: Degree of each node (default 12). Larger M → more connections → slower builds, faster search
- `ef_construction`: Size of dynamic candidate list during construction (default 200). Larger ef → better search but slower construction
- `ef`: Search parameter (default M × 2). Larger ef → more accurate but slower

**Complexity:**
- Build: O(N log N) where N = vectors
- Query: O(log N) expected; worst-case O(N)
- Memory: O(N × (M + overhead))

**Strengths:** Fast search, low memory, works well in practice
**Weaknesses:** Build time is slow (can't do incremental updates efficiently)

---

### IVF (Inverted File Index)

**Concept:** Pre-cluster vectors with k-means. At query time, search only nearby clusters.

```
All Vectors (1M total)
    │
    ├─ Cluster 1 (100K vectors)
    │  └─ Indexed with HNSW
    │
    ├─ Cluster 2 (100K vectors)
    │  └─ Indexed with HNSW
    │
    └─ ...
    
Query:
  1. Find nearest cluster(s) to query (coarse quantization)
  2. Search within top-k clusters (fine search)
  3. Return top vectors
```

**Tuning Parameters:**
- `num_clusters` (nlist): Number of k-means clusters. Larger → better granularity but slower clustering
- `nprobe`: How many clusters to search at query time. Larger nprobe → higher recall, slower

**Complexity:**
- Build: O(N × k-means iterations)
- Query: O(nprobe × cluster_size)
- Memory: O(N + cluster_centers)

**Strengths:** Fast clustering, scalable, low memory
**Weaknesses:** Cluster boundaries can hurt recall; requires re-clustering on inserts

---

### Product Quantization (PQ)

**Concept:** Decompose vectors into subspaces. Each subspace is quantized to a smaller representation (codebook).

```
Original Vector: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  (1024 dims, 4 bytes each = 4 KB)
                          │
                    Split into subspaces
                          │
Subspace 1: [0.1, 0.2] → Codebook index 3 (1 byte)
Subspace 2: [0.3, 0.4] → Codebook index 7 (1 byte)
...

Compressed: [3, 7, 5, 2, 1, 8, 4, 6]  (8 bytes = 99.8% compression!)
```

**Quantization Loss:** ~1–2% recall loss with 100x compression (typical)

**Strengths:** Enormous compression (RAM feasible for billions of vectors); fast distance computation
**Weaknesses:** Approximate; requires separate codebook per dataset; hard to tune

---

## Comparison Table: HNSW vs. IVF vs. PQ

| Algorithm | Build Speed | Query Speed | Memory | Recall @99% | Best For | Worst For |
|-----------|-------------|-----------|--------|------------|----------|-----------|
| HNSW | Slow (hours for 100M) | Fast (<10ms) | High (16 bytes/vector) | 99%+ | <100M vectors, high recall needed | Massive scale, memory-constrained |
| IVF | Fast (minutes) | Moderate (50–200ms) | Medium (4 bytes/vector) | 95–98% | 100M–1B vectors, balanced | Dynamic inserts; exact recall required |
| PQ | Fast (minutes) | Very fast (<5ms) | Very low (1 byte/vector) | 90–95% | 1B+ vectors, cost-critical | Exact retrieval, high recall required |

---

## System Comparison Table: All 8 Popular Vector Databases

| System | Deployment | Algorithm | Hybrid Search | Metadata Filtering | Scaling | License | Best For |
|--------|-----------|-----------|--------------|-------------------|---------|---------|----------|
| FAISS | Library | IVF/HNSW/PQ | No | No | Single machine | MIT | Research, prototypes |
| Chroma | Self-hosted | HNSW | No | Yes | Single machine | Apache 2.0 | Local development |
| Qdrant | Self-hosted | HNSW | Yes (BM25) | Yes | Horizontal (sharding) | AGPL/Commercial | Production, open-source |
| Weaviate | Self-hosted | HNSW | Yes (BM25) | Yes | Horizontal (replication) | BSL/Commercial | Production, enterprises |
| Pinecone | Managed Cloud | Proprietary (HNSW-based) | No | Yes | Auto-scaling | Proprietary | Fast onboarding, managed |
| Milvus | Self-hosted | IVF/HNSW/PQ | No | Yes | Horizontal | AGPL | Large-scale, cost-conscious |
| pgvector | PostgreSQL ext. | IVF/HNSW | Yes (full-text) | Yes (SQL) | Horizontal (Postgres cluster) | PostgreSQL License | Existing Postgres users |
| Redis | In-memory | HNSW | No | Yes (Lua) | Horizontal (cluster) | Redis License | Low-latency, cache-like |

---

## Metadata Filtering and Its Performance Cost

Filtering is non-trivial. Your choice of filtering strategy significantly affects recall.

### Strategy 1: Pre-Filter

**Mechanism:** Filter metadata first, then search within filtered set.

**Example:** "Find similar docs WHERE user_id = 123"
```
All Vectors (1M)
    │
    ├─ Pre-filter on metadata
    │  └─ Vectors where user_id = 123 (10K)
    │
    └─ ANN search within 10K vectors
       └─ Return top-5
```

**Pros:** No index wasted on irrelevant vectors
**Cons:** If filtered set is small, recall suffers (fewer vectors to search)

---

### Strategy 2: Post-Filter

**Mechanism:** Retrieve top-k candidates, then filter.

**Example:**
```
All Vectors (1M)
    │
    ├─ ANN search (no filter)
    │  └─ Top-50 candidates
    │
    └─ Post-filter on metadata
       └─ Keep only user_id = 123 (might be 0–3 matches!)
           └─ Return top-5 (or fewer)
```

**Pros:** Retrieval sees full index (high recall)
**Cons:** Might not retrieve enough; wasted computation on filtered-out vectors

---

### Strategy 3: ACORN-Style (Interleaved)

**Mechanism:** Interleave filtering during graph traversal.

**How:** During HNSW traversal, skip nodes that don't match metadata filter.

**Pros:** Balances recall and efficiency
**Cons:** Complex to implement; requires index-aware filtering

---

## Hybrid Search Architecture

Combine dense (semantic) + sparse (keyword) retrieval.

```
Query: "bert transformer attention mechanism"
    │
    ├─ Dense Retrieval
    │  ├─ Embed query with model
    │  └─ Search vector DB → [doc1: 0.95, doc2: 0.87, doc3: 0.81]
    │
    ├─ Sparse Retrieval (BM25)
    │  └─ BM25 exact match → [doc3: 42, doc1: 38, doc2: 22]
    │
    ├─ Merge with RRF
    │  └─ RRF score = 1/(k+dense_rank) + 1/(k+sparse_rank)
    │     Results: doc1 (0.67), doc3 (0.62), doc2 (0.48)
    │
    └─ Final ranking: [doc1, doc3, doc2]
```

**RRF Formula (plaintext):**
```
score(document) = sum of (1 / (k + rank_in_result_set))
  where k = 60 (standard default)
```

**Example Calculation:**
```
Document appears:
  - 1st in dense results: 1/(60+1) = 0.0164
  - 3rd in sparse results: 1/(60+3) = 0.0154
  - Total RRF score: 0.0318
```

**Code: Hybrid Retrieval in Weaviate**

```python
from weaviate import Client

client = Client("http://localhost:8080")

# Hybrid search: dense + BM25 automatically merged
results = client.query.get("Document", ["title", "content"]) \
    .with_hybrid(
        query="bert transformer mechanism",
        alpha=0.5  # 50% dense + 50% sparse
    ) \
    .with_limit(5) \
    .do()

print(results)
```

---

## Production Concerns

### 1. Index Persistence and Warm-Up Latency

**Problem:** After restart, index must be loaded into memory. This can take minutes for large indexes.

**Solution:** Pre-warm index by querying high-traffic vectors before serving traffic.

```python
def warm_up_index(vector_db, num_vectors: int = 1000):
    """Pre-load index into memory."""
    for i in range(num_vectors):
        # Query random vectors (doesn't matter if they exist)
        vector_db.search(random_vector(), k=1)
```

### 2. Replication for Read Throughput

**Problem:** Single vector DB node maxes out at ~1K QPS.

**Solution:** Replicate index across multiple nodes. Load-balance queries.

```
Client Load Balancer
    │
    ├─ VectorDB Node 1 (read replica)
    ├─ VectorDB Node 2 (read replica)
    └─ VectorDB Node 3 (read replica)
```

### 3. Write Throughput Constraints

**Problem:** HNSW is slow to build incrementally (graph construction is sequential). IVF requires re-clustering.

**Solution:** Use write-optimized storage (like append-only log) + async batch indexing.

```
New Documents
    │
    ├─ Write to append-only log (fast)
    │
    └─ Async Background Process
       ├─ Batch embed (100 at a time)
       ├─ Batch insert into index
       └─ Re-index if needed (scheduled, not per-insert)
```

### 4. Memory Pressure

**Thresholds:**

| Vector Count | HNSW Memory | IVF Memory | PQ Memory |
|---|---|---|---|
| 100K | 2 GB | 400 MB | 50 MB |
| 1M | 20 GB | 4 GB | 500 MB |
| 100M | 2 TB | 400 GB | 50 GB |

**Recommendation:** Use PQ compression for >100M vectors.

---

## Selecting a Vector Database: Decision Tree

```
Question 1: Managed or Self-Hosted?
  │
  ├─ MANAGED (prefer hands-off)
  │  └─ Question 2: Existing Postgres?
  │     ├─ No → Use Pinecone (simplest)
  │     └─ Yes → Use pgvector (native integration)
  │
  └─ SELF-HOSTED (control + cost)
     └─ Question 2: Corpus size?
        ├─ <100M vectors → Use Qdrant (best balance)
        └─ >100M vectors + cost-critical → Use Milvus (PQ compression)
```

---

## Configuration Priority

When first deploying any vector DB, set these in order:

1. **Algorithm:** HNSW for <100M vectors; IVF+PQ for >100M
2. **M (HNSW) or nlist (IVF):** Start with defaults; tune only if query is slow
3. **Replication:** Set up read replicas if QPS >500
4. **TTL:** Set appropriate expiration for stale vectors
5. **Backup:** Automated daily snapshots

---

## Key Takeaways

1. **HNSW is the default for most systems** (<100M vectors). It's fast and simple.
2. **IVF + PQ for massive scale** (>1B vectors). Compression is mandatory.
3. **RRF is the gold standard** for merging dense + sparse results.
4. **Pre-filter when possible**, but measure recall impact.
5. **Start with a managed service** (Pinecone) if you're unsure. Migrate to self-hosted later.

---

## Additional Interview Questions

**Q: How would you design the vector DB layer for a RAG system handling 1 billion documents?**

At 1B vectors, three changes are mandatory: (1) **Quantization** — use IVF+PQ (Product Quantization) to compress each vector from 768 × 4 bytes = 3KB to ~96 bytes (32× compression with 8-bit PQ). Without compression, 1B vectors require ~3 TB of RAM — impossible on a single machine. (2) **Sharding** — partition the index across multiple nodes. Shard by document type, time range, or a hash of the document ID. At query time, fan out to all shards and merge results with RRF. (3) **Tiered storage** — keep the hot index (recent documents, frequently accessed) in memory; cold segments on NVMe SSD with demand loading. At 1B scale, managed services (Pinecone, Weaviate Cloud) become prohibitively expensive; self-hosted Milvus with GPU indexing or FAISS on a fleet of CPU/GPU machines is the standard choice. Plan for 48–72 hours to build the full index from scratch; use incremental indexing for ongoing updates.

---

**Q: What is the cold-start problem in vector DB index warm-up and how do you mitigate it?**

After a service restart or new node addition, the HNSW graph must be loaded from disk into RAM before it can serve fast ANN queries. Until warm-up completes, queries fall back to slow linear scan or fail entirely — this is the cold-start problem. Mitigations: (1) **Pre-warm on startup** — load and run a set of probe queries after index load to populate OS page cache and HNSW graph traversal paths before the service starts accepting traffic. (2) **Read replicas** — never restart the primary index node; scale by adding replicas while keeping at least one warm replica live. (3) **Memory-lock the index** (`mlock` on Linux) to prevent the OS from paging out the index under memory pressure. (4) **Index snapshotting** — checkpoint the loaded (not just serialized) index state so restart resumes from an in-memory snapshot rather than deserializing from scratch. Qdrant and Milvus both support memory-mapped files (`mmap`) that let the OS page index segments in lazily, reducing the hard blocking period at startup.
