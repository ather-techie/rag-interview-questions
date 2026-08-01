# Vector Databases: Storage and Search for Embeddings

> The storage and search layer every RAG system depends on — how vector databases work, how they differ, and how to choose one.

---

## What is a Vector Database?

A vector database is a storage system purpose-built to hold embeddings and quickly find the ones most similar to a given query vector, typically using approximate nearest-neighbor (ANN) indexing rather than brute-force comparison. It's the component that sits between "I have millions of embedded chunks" and "give me the top-k most relevant ones in milliseconds." Beyond raw similarity search, most vector databases also handle metadata filtering, hybrid (keyword + vector) search, and horizontal scaling as your index grows.

**Why this beats keyword search:** a user searching "cancel my coverage" against a keyword index would look for the literal words "cancel," "my," "coverage" and might miss a document titled "Policy Termination Procedures" — there's no exact word match. A vector database embeds the query and every document, then finds that "Policy Termination Procedures" is conceptually close to "cancel my coverage" in embedding space even though the two share almost no keywords. The same holds for "car won't start in cold weather" surfacing a document about "battery performance in low temperatures" — different words, same underlying meaning, close together as vectors.

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

### A Fourth Model: Serverless Vector Databases

A newer variant of the managed cloud model has emerged: **serverless** vector databases, where compute and storage scale independently and automatically, and billing is per-operation (reads/writes) and per-GB stored rather than per-node-hour. This differs from a "provisioned" or "dedicated" managed deployment (e.g., traditional Pinecone pods, a fixed Qdrant Cloud cluster), where you pay for a running node/cluster whether or not it's handling traffic.

**Serverless Vector Database**
- Examples: Pinecone Serverless, Zilliz Cloud Serverless (managed Milvus), MongoDB Atlas Vector Search (serverless instances), Turbopuffer (object-storage-native)
- Deployment: No clusters to size or provision; scales to zero; billed per read/write unit + storage
- Pros: No idle cost for bursty/intermittent workloads; no capacity planning; scales automatically to large corpora
- Cons: Cold-start latency on the first query after an idle period; can cost more than a provisioned cluster at high, steady QPS

**Serverless vs. Provisioned/Dedicated: Tradeoffs**

| Dimension | Serverless (Pinecone Serverless, Zilliz Cloud Serverless) | Provisioned/Dedicated (Pinecone pods, Qdrant Cloud clusters) |
|---|---|---|
| Billing | Pay-per-read/write-unit + per-GB storage; near-zero idle cost | Fixed hourly/monthly cost per node, regardless of utilization |
| Cold-start latency | ~200ms–2s on first query after an idle period (architectural; cannot be disabled) | None once running — consistently low (e.g., ~30ms p99 on Pinecone pods) |
| Cost at high, sustained QPS | Can exceed provisioned cost (e.g., write-heavy agentic workloads) | Cheaper once utilization is high and steady |
| Cost at bursty/intermittent traffic | Cheaper — nothing charged during idle periods | Wasteful — you pay for idle capacity |
| Scaling | Automatic, per-request | Manual resize or autoscaling within configured limits |
| Ops burden | Lowest | Low-to-moderate (size and monitor the cluster) |

Note the spectrum isn't binary: Weaviate Cloud's serverless tier prices by dimensions stored rather than by node-hour, while Qdrant Cloud remains cluster-based (hourly-billed, no per-query charge) — i.e., provisioned rather than serverless.

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
Original Vector: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  (8 dims, 4 bytes each = 32 bytes)
                          │
                    Split into subspaces
                          │
Subspace 1: [0.1, 0.2] → Codebook index 3 (1 byte)
Subspace 2: [0.3, 0.4] → Codebook index 7 (1 byte)
...

Compressed: [3, 7, 5, 2]  (4 bytes = 87.5% compression!)
```

**Quantization Loss:** ~1–2% recall loss with 100x compression (typical)

**Strengths:** Enormous compression (RAM feasible for billions of vectors); fast distance computation
**Weaknesses:** Approximate; requires separate codebook per dataset; hard to tune

---

### DiskANN (Disk-Based ANN)

**Concept:** A graph-based ANN index — built on the *Vamana* graph algorithm — originally developed by Microsoft Research (NeurIPS 2019) to scale ANN search to billions of vectors without requiring the whole index to fit in RAM. Unlike HNSW, which needs the full graph and full-precision vectors resident in memory, DiskANN keeps full-precision vectors on SSD and holds only a PQ-compressed copy of every vector in RAM for graph navigation — trading a small amount of query latency for dramatically lower memory cost at massive scale.

```
Query Vector
    │
    ├─ Navigate Vamana graph using PQ-compressed vectors (RAM only, no disk reads)
    │  └─ Identify candidate set (cheap, approximate)
    │
    └─ Fetch full-precision vectors for candidates from SSD
       └─ Re-rank with exact distances → Return top-k
```

**Why it exists:** Pure in-memory graphs (HNSW) become RAM-prohibitive well before 1B vectors — practical ceilings are roughly 100–200M vectors on a single node at comparable latency/recall. DiskANN's published benchmarks reach ~95% recall at ~5ms query latency over **1 billion** points on a single workstation with just 64GB RAM plus an SSD.

**Tradeoff:** Slightly higher per-query latency than a fully in-memory HNSW graph (SSD reads add microseconds-to-low-milliseconds per lookup), in exchange for roughly 10–100x lower memory cost at billion-vector scale.

**Who uses it (verified):**
- **Milvus / Zilliz Cloud** — DiskANN is a selectable index type alongside HNSW/IVF/PQ for on-disk, billion-scale collections.
- **Azure Cosmos DB** (for NoSQL, and for MongoDB vCore) — DiskANN is the built-in, generally-available vector index, recommended above ~1M documents.
- **Azure Database for PostgreSQL – Flexible Server** — Microsoft's `pg_diskann` extension adds a DiskANN (Vamana + PQ) index type to Postgres.
- **pgvectorscale** (Timescale) — an open-source Postgres extension adding a DiskANN-derived index (Statistical Binary Quantization instead of PQ) alongside pgvector's HNSW/IVFFlat.

---

## Comparison Table: HNSW vs. IVF vs. PQ vs. DiskANN

| Algorithm | Build Speed | Query Speed | Memory | Recall @99% | Best For | Worst For |
|-----------|-------------|-----------|--------|------------|----------|-----------|
| HNSW | Slow (hours for 100M) | Fast (<10ms) | High (~6 KB raw @1536-dim fp32 + 20–50% graph overhead ≈ 8–9 KB/vector) | 99%+ | <100M vectors, high recall needed | Massive scale, memory-constrained |
| IVF | Fast (minutes) | Moderate (50–200ms) | Medium (4 bytes/vector) | 95–98% | 100M–1B vectors, balanced | Dynamic inserts; exact recall required |
| PQ | Fast (minutes) | Very fast (<5ms) | Very low (1 byte/vector) | 90–95% | 1B+ vectors, cost-critical | Exact retrieval, high recall required |
| DiskANN | Slow (graph build, disk-resident) | Moderate (~5ms, SSD-bound) | Very low in RAM (PQ-compressed vectors only; full vectors on SSD) | ~95% (at 1B scale) | 1B+ vectors, RAM-constrained/cost-critical | Sub-millisecond latency needs; no SSD available |

---

## System Comparison Table: All 8 Popular Vector Databases

| System | Deployment | Algorithm | Hybrid Search | Metadata Filtering | Scaling | License | Best For |
|--------|-----------|-----------|--------------|-------------------|---------|---------|----------|
| FAISS | Library | IVF/HNSW/PQ | No | No | Single machine | MIT | Research, prototypes |
| Chroma | Self-hosted / Managed Cloud | HNSW | No | Yes | Single machine (self-hosted); serverless/distributed via Chroma Cloud | Apache 2.0 | Local development; Chroma Cloud for managed scale |
| Qdrant | Self-hosted | HNSW | Yes (BM25) | Yes | Horizontal (sharding) | AGPL/Commercial | Production, open-source |
| Weaviate | Self-hosted | HNSW | Yes (BM25) | Yes | Horizontal (replication) | BSD-3-Clause | Production, enterprises |
| Pinecone | Managed Cloud | Proprietary (HNSW-based) | Yes (sparse-dense index) | Yes | Auto-scaling | Proprietary | Fast onboarding, managed |
| Milvus | Self-hosted | IVF/HNSW/PQ | Yes (BM25 sparse + dense) | Yes | Horizontal | AGPL | Large-scale, cost-conscious |
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
    │     Results: doc1 (0.0325), doc3 (0.0323), doc2 (0.0320)
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
| 100K | ~850 MB | 400 MB | 50 MB |
| 1M | ~8.5 GB | 4 GB | 500 MB |
| 100M | ~850 GB | 400 GB | 50 GB |

*(HNSW figures assume 1536-dim fp32 embeddings: ~6 KB raw vector + ~40% graph overhead ≈ 8.5 KB/vector.)*

**Recommendation:** Use PQ compression for >100M vectors.

---

## Selecting a Vector Database: Decision Tree

```
Question 1: Managed or Self-Hosted?
  │
  ├─ MANAGED (prefer hands-off)
  │  └─ Use Pinecone (simplest)
  │
  └─ SELF-HOSTED (control + cost)
     └─ Question 2: Existing Postgres or corpus size?
        ├─ Existing Postgres → Use pgvector (native integration)
        ├─ <100M vectors → Use Qdrant (best balance)
        └─ >100M vectors + cost-critical → Use Milvus (PQ compression)
```

### Second Decision Point: Serverless or Provisioned?

If you land on "MANAGED" above, there's a follow-up choice between a serverless and a provisioned/dedicated tier:

```
Question: What's the traffic pattern?
  │
  ├─ Bursty / intermittent, cost-sensitive, cold starts acceptable
  │  └─ Use Serverless (Pinecone Serverless, Zilliz Cloud Serverless)
  │
  └─ High, steady QPS; latency-critical (e.g., real-time chat)
     └─ Use Provisioned/Dedicated (Pinecone pods, Qdrant Cloud clusters)
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
6. **DiskANN trades RAM for SSD** to reach billion-vector scale (~95% recall, ~5ms) on a single node — used by Milvus/Zilliz, Azure Cosmos DB, and pgvectorscale.
7. **Serverless suits bursty traffic; provisioned suits steady, high QPS.** Serverless (Pinecone Serverless, Zilliz Cloud Serverless) has no idle cost but adds cold-start latency; dedicated clusters cost more when idle but stay consistently fast.

---

## Interview Q&A

**Q: How would you design the vector DB layer for a RAG system handling 1 billion documents?** `[Advanced]`

At 1B vectors, three changes are mandatory: (1) **Quantization** — use IVF+PQ (Product Quantization) to compress each vector from 768 × 4 bytes = 3KB to ~96 bytes (32× compression with 8-bit PQ). Without compression, 1B vectors require ~3 TB of RAM — impossible on a single machine. (2) **Sharding** — partition the index across multiple nodes. Shard by document type, time range, or a hash of the document ID. At query time, fan out to all shards and merge results with RRF. (3) **Tiered storage** — keep the hot index (recent documents, frequently accessed) in memory; cold segments on NVMe SSD with demand loading. At 1B scale, managed services (Pinecone, Weaviate Cloud) become prohibitively expensive; self-hosted Milvus with GPU indexing or FAISS on a fleet of CPU/GPU machines is the standard choice. Plan for 48–72 hours to build the full index from scratch; use incremental indexing for ongoing updates.

---

**Q: What is the cold-start problem in vector DB index warm-up and how do you mitigate it?** `[Advanced]`

After a service restart or new node addition, the HNSW graph must be loaded from disk into RAM before it can serve fast ANN queries. Until warm-up completes, queries fall back to slow linear scan or fail entirely — this is the cold-start problem. Mitigations: (1) **Pre-warm on startup** — load and run a set of probe queries after index load to populate OS page cache and HNSW graph traversal paths before the service starts accepting traffic. (2) **Read replicas** — never restart the primary index node; scale by adding replicas while keeping at least one warm replica live. (3) **Memory-lock the index** (`mlock` on Linux) to prevent the OS from paging out the index under memory pressure. (4) **Index snapshotting** — checkpoint the loaded (not just serialized) index state so restart resumes from an in-memory snapshot rather than deserializing from scratch. Qdrant and Milvus both support memory-mapped files (`mmap`) that let the OS page index segments in lazily, reducing the hard blocking period at startup.

---

**Q: When would you choose DiskANN over HNSW for an index?** `[Advanced]`

Choose DiskANN when the corpus is large enough that an in-memory HNSW graph becomes RAM-prohibitive — roughly above 100–200M vectors, and especially in the 1B+ range. HNSW requires the full graph and full-precision vectors resident in RAM; at 1536-dim fp32 that's ~8–9 KB/vector, so 1B vectors would need on the order of 8–9 TB of RAM. DiskANN (Microsoft Research's Vamana-graph algorithm) instead keeps only PQ-compressed vectors in RAM for graph navigation and stores full-precision vectors on SSD, fetching them only to re-rank a small candidate set. Published results show ~95% recall at ~5ms latency over 1 billion points using just 64GB RAM and an SSD on a single node. The cost is a small amount of added per-query latency from SSD reads versus a pure in-memory graph — acceptable for most RAG workloads, but not for sub-millisecond latency requirements. In production, this shows up as a selectable index type in Milvus/Zilliz Cloud, the built-in vector index in Azure Cosmos DB, Microsoft's `pg_diskann` extension for Azure PostgreSQL, and Timescale's pgvectorscale (DiskANN-derived, using Statistical Binary Quantization).

---

**Q: What are the tradeoffs between a serverless and a provisioned (dedicated) vector database deployment?** `[Intermediate]`

Serverless vector databases (Pinecone Serverless, Zilliz Cloud Serverless, MongoDB Atlas Vector Search's serverless instances, Turbopuffer) scale compute and storage independently and automatically, billing per read/write operation and per-GB stored rather than per-node-hour — so idle time costs little to nothing, which is ideal for bursty or intermittent traffic. The tradeoff is cold-start latency: after a period of inactivity, the first query to a namespace or index can take anywhere from ~200ms to a couple of seconds while the system loads it into a multi-tenant worker, and this is architectural — it can't be tuned away on a serverless tier. Provisioned/dedicated deployments (Pinecone pods, Qdrant Cloud clusters) charge a fixed cost per node-hour regardless of utilization, but stay warm and deliver consistent low latency (e.g., ~30ms p99 on Pinecone pods) with no cold-start penalty. At high, steady QPS, provisioned deployments are usually cheaper and more predictable; at low or spiky QPS, serverless avoids paying for idle capacity. Some vendors sit in between — Weaviate Cloud's serverless tier bills by dimensions stored rather than by node-hour, while Qdrant Cloud remains purely cluster-based (provisioned) with no serverless option.

---

## Related

- [Retrieval Strategies](./retrieval_strategies.md) — hybrid dense + sparse retrieval built on top of the vector DB layer
- [Caching Strategies](./caching_strategies.md) — semantic and result caching to reduce vector DB query load and warm-up impact
- [Cost Optimization](./cost_optimization.md) — storage/compute cost tradeoffs for quantization, tiering, and managed vs. self-hosted deployments
- [Multi-Tenancy & Access Control](./multi_tenancy_access_control.md) — metadata filtering patterns for isolating tenant data in a shared index
