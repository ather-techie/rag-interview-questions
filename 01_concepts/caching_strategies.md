# Caching Strategies in RAG Systems

> Reduce latency and cost by serving repeated work from cache — but design the cache boundaries carefully to avoid stale or leaked responses.

---

## What is Caching (in RAG)?

Caching in RAG means storing the result of an expensive step — query embedding, ANN search, reranking, or LLM generation — so a repeated or similar request can be served from storage instead of recomputed. Because a single RAG request chains several costly operations, caching at the right layer (exact-match or semantic) cuts both latency and API/compute spend, at the cost of needing an invalidation and isolation strategy to avoid serving stale or cross-tenant results.

---

## Why Caching Matters in RAG

A RAG request chain has multiple expensive operations: embedding the query, ANN search, optional reranking, and LLM generation. Each step can be cached at a different granularity with different freshness requirements.

```
User Query
    │
    ▼
┌───────────────────────────────────────────┐
│ 1. Semantic Query Cache                   │  ← Cache entire response for near-duplicate queries
│    "What is Python?" ≈ "Explain Python"   │
└──────────────────────┬────────────────────┘
                       │ Cache miss
                       ▼
┌───────────────────────────────────────────┐
│ 2. Embedding Cache                        │  ← Cache query vector (cheap to store)
│    SHA-256(query) → embedding vector      │
└──────────────────────┬────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│ 3. Retrieval Result Cache                 │  ← Cache top-k results per query signature
│    (query_embedding_hash) → [doc_ids]     │
└──────────────────────┬────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│ 4. LLM Prompt Prefix Cache (KV Cache)     │  ← Cache system prompt + documents in GPU memory
│    Provider-level: Anthropic, OpenAI      │
└──────────────────────┬────────────────────┘
                       │
                       ▼
                  Final Response
```

---

## Cache Type 1: Semantic Query Cache

Caches the **full LLM response** keyed on the semantic meaning of the query — not the exact string. Two queries that mean the same thing share a cached response.

### How It Works

```
Query: "What is RAG?"
    │
    ├─ Embed query → vector v
    │
    ├─ Search cache index for vectors with cosine_sim(v, cached_v) > threshold
    │
    ├─ HIT:  return cached response immediately (0ms LLM call)
    └─ MISS: run full RAG pipeline → store (v, response) in cache
```

### Implementation with GPTCache / Custom Redis

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
import pickle
import hashlib
import time

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.redis = redis.Redis(host="localhost", port=6379)
        self.threshold = similarity_threshold
        self.cache_index = []   # in-memory ANN index: (embedding_bytes, cache_key, expires_at)

    def _embed(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)

    def get(self, query: str) -> str | None:
        q_emb = self._embed(query)
        now = time.time()
        self.cache_index = [e for e in self.cache_index if e[2] > now]   # prune expired entries
        for cached_emb_bytes, cache_key, _expires_at in self.cache_index:
            cached_emb = np.frombuffer(cached_emb_bytes, dtype=np.float32)
            score = float(np.dot(q_emb, cached_emb))   # normalized → cosine
            if score >= self.threshold:
                value = self.redis.get(cache_key)
                if value:
                    return pickle.loads(value)
        return None

    def set(self, query: str, response: str, ttl_seconds: int = 3600):
        q_emb = self._embed(query).astype(np.float32)
        cache_key = "sc:" + hashlib.sha256(query.encode()).hexdigest()
        self.redis.setex(cache_key, ttl_seconds, pickle.dumps(response))
        self.cache_index.append((q_emb.tobytes(), cache_key, time.time() + ttl_seconds))
```

### Similarity Threshold Tuning

| Threshold | Behavior | Risk |
|-----------|----------|------|
| 0.98+ | Only exact paraphrases hit | Low hit rate; near-zero latency savings |
| 0.92–0.97 | Near-duplicate queries hit | **Recommended** — good precision |
| 0.85–0.91 | Semantically similar queries hit | Risk of serving wrong answer for different intent |
| < 0.85 | Too broad — unrelated queries may hit | High false-positive rate |

**Test your threshold** with adversarial pairs:
- Should HIT: "What is RAG?" / "Explain retrieval augmented generation"
- Should MISS: "What is RAG?" / "What are the limitations of RAG?"

### When to Invalidate the Semantic Cache

| Trigger | Action |
|---------|--------|
| Underlying documents updated | Flush cache for affected topic clusters, or use TTL |
| Model upgrade (embeddings or LLM) | Full cache flush — cached embeddings are incompatible |
| Domain shift detected | Flush and rebuild |
| User-level personalization | Do not cache responses that depend on user context |

---

## Cache Type 2: Embedding Cache

The cheapest cache: store the embedding vector for a query string keyed on its exact text. Avoids an embedding API call on repeated identical queries.

```python
import hashlib, json

EMBEDDING_CACHE: dict[str, list[float]] = {}

def get_embedding(text: str, model_fn) -> list[float]:
    key = hashlib.sha256(text.encode()).hexdigest()
    if key not in EMBEDDING_CACHE:
        EMBEDDING_CACHE[key] = model_fn(text)
    return EMBEDDING_CACHE[key]
```

**Production note:** Persist this cache to Redis or disk — re-embedding at service restart wastes API budget. Invalidate on model version change (e.g., when upgrading from `text-embedding-3-small` to `text-embedding-3-large`).

---

## Cache Type 3: Retrieval Result Cache

Cache the **top-k document IDs** returned for a given query. Useful when the same query (or near-duplicate) is issued repeatedly and the index doesn't change frequently.

```python
import time

def cached_retrieve(query: str, retriever, cache: dict, ttl: int = 600) -> list[str]:
    cache_key = "ret:" + hashlib.sha256(query.encode()).hexdigest()
    if cache_key in cache:
        ts, result = cache[cache_key]
        if time.time() - ts < ttl:
            return result
    result = retriever.retrieve(query)
    cache[cache_key] = (time.time(), result)
    return result
```

**When this cache helps most:** High-traffic Q&A systems where the same questions recur (support chatbots, documentation search). **When it hurts:** If the index is updated frequently, stale retrieval results cause the LLM to answer from outdated chunks.

---

## Cache Type 4: LLM Prompt Prefix Caching (KV Cache)

The most impactful cache for cost reduction in RAG: cache the computed **key-value attention tensors** for the static parts of the prompt (system prompt, retrieved documents) so the LLM doesn't recompute them on every request.

### How Anthropic Prompt Caching Works

```
Prompt structure:
┌────────────────────────────────────────────────────┐
│ System prompt (500 tokens)          ← CACHED       │
│ Retrieved Document 1 (800 tokens)   ← CACHED       │
│ Retrieved Document 2 (800 tokens)   ← CACHED       │
│ Retrieved Document 3 (800 tokens)   ← CACHED       │
│                                                    │
│ User question (50 tokens)           ← NOT cached   │
└────────────────────────────────────────────────────┘

Without caching:  pay full price for all 2950 input tokens every call
With caching:     pay full price for the 50 uncached tokens, plus a cache-read
                  fee (~10% of base price) for the 2900 cached tokens
                  → roughly 85-90% lower effective input-token cost per call
                    (exact savings depend on how many calls reuse the cache
                    before the TTL expires — see below)
```

### Implementation

```python
import anthropic

client = anthropic.Anthropic()

def rag_generate_with_cache(question: str, retrieved_docs: list[str]) -> str:
    docs_text = "\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs)
    )
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": "You are a helpful assistant. Answer using only the provided documents.",
                "cache_control": {"type": "ephemeral"}   # cache this block
            },
            {
                "type": "text",
                "text": docs_text,
                "cache_control": {"type": "ephemeral"}   # cache retrieved docs too
            }
        ],
        messages=[
            {"role": "user", "content": question}   # variable — not cached
        ]
    )
    return response.content[0].text
```

### Cache TTL and Hit Conditions

| Provider | TTL | Cache Hit Condition |
|----------|-----|---------------------|
| Anthropic | 5 minutes by default (refreshed on each hit); 1 hour available as an opt-in extended TTL | Same model + same prefix bytes up to `cache_control` marker |
| OpenAI | 5–10 minutes | Automatic on prompts > 1024 tokens; same prefix |

**Anthropic's two cache TTL tiers:** `cache_control: {"type": "ephemeral"}` defaults to a 5-minute TTL, refreshed on every cache hit. You can instead request a 1-hour TTL with `cache_control: {"type": "ephemeral", "ttl": "1h"}`. The tradeoff is write cost: a cache write costs ~1.25× the base input price at the 5-minute TTL, but ~2× at the 1-hour TTL. The longer TTL is worth the extra write cost when a shared, expensive-to-recompute prefix (e.g., a large system prompt or a fixed tool/schema definition) needs to survive gaps between requests longer than 5 minutes — for example, a support agent session where the user takes a few minutes to read a response before asking a follow-up. For a RAG pipeline under continuous traffic (requests arriving well under 5 minutes apart), the default 5-minute TTL is usually sufficient and cheaper to write.

```python
system=[
    {
        "type": "text",
        "text": "You are a helpful assistant. Answer using only the provided documents.",
        "cache_control": {"type": "ephemeral", "ttl": "1h"}   # 1-hour TTL for a long-lived session
    }
]
```

**Implication for RAG:** If you retrieve different documents for each query, the "retrieved docs" block changes every call and won't hit the cache. Use prompt caching for the **system prompt and static reference documents** (fixed knowledge base sections), not for dynamically retrieved chunks.

### Cache-Augmented Generation (CAG)

The extreme form: pre-load the **entire knowledge base** into the LLM's KV cache once, eliminating retrieval entirely. Works only when:
- The knowledge base fits within the context window (≤ 200K tokens)
- The content is relatively static (cache TTL isn't constantly expiring)
- Query distribution is broad (retrieval's precision advantage is small)

See [17 — Cache-Augmented Generation](../02_interview_bank/17-cache-augmented-generation.md) for the full architecture.

---

## Multi-Tenant Cache Safety

The most dangerous caching mistake in multi-tenant RAG: serving one tenant's cached response to another tenant whose query is semantically similar.

```
Tenant A asks: "What is our Q3 revenue?"  → cached response: "$4.2M"
Tenant B asks: "What is our Q3 revenue?"  → semantic similarity 0.99 → WRONG: returns $4.2M
```

**Mitigation: Namespace cache keys by tenant**

```python
def tenant_safe_cache_key(tenant_id: str, query: str) -> str:
    return f"sc:{tenant_id}:{hashlib.sha256(query.encode()).hexdigest()}"
```

Also see [09 — Semantic Cache Leakage](../03_failure_modes/09-semantic_cache_leakage.md).

---

## Cache Invalidation Strategies

| Strategy | When to Use | Mechanism |
|----------|-------------|-----------|
| **TTL-based expiry** | Corpus updates on a known schedule | Set TTL = update frequency × safety margin |
| **Event-driven invalidation** | Real-time document updates via CDC/Kafka | Publish cache-bust events; invalidate affected keys |
| **Version tag invalidation** | Model upgrades, index rebuilds | Tag all cache keys with model/index version; flush on bump |
| **Selective topic flush** | Partial corpus updates | Tag cache entries with topic IDs; flush affected topics |
| **LRU eviction** | Cache size is bounded | Standard LRU; no explicit invalidation logic needed |

---

## Cache Stampede (Thundering Herd) and Cache Warming

### The Problem

A **cache stampede** (a.k.a. **thundering herd**) happens when a popular cache entry expires — or the cache is simply cold after a restart/deploy — and a burst of concurrent requests for that *same key* all miss at once. Each request independently believes it's the one that needs to recompute the value, so all of them re-run the same expensive operation simultaneously: re-embedding the same query N times, re-running the same LLM generation N times, or re-issuing the same ANN search N times. Best case, this wastes compute/API spend; worst case, the simultaneous spike of duplicate work overwhelms the retriever or LLM backend right when it's already under load (e.g., right after a deploy when *every* key is cold).

```
TTL expires on a hot key
    │
    ├─ Request A misses → starts expensive recompute
    ├─ Request B misses → starts expensive recompute  (duplicate work)
    ├─ Request C misses → starts expensive recompute  (duplicate work)
    └─ Request D misses → starts expensive recompute  (duplicate work)
         All 4 hit the LLM/embedder at once for a single logical value
```

### Mitigations

| Mitigation | Mechanism | Tradeoff |
|------------|-----------|----------|
| **Request coalescing / single-flight** | The first miss becomes the "leader" and computes the value; concurrent requests for the same key become "followers" that wait on the leader's in-flight result instead of recomputing | Followers pay the leader's full latency; needs an in-process lock (single instance) or a distributed lock (multi-instance) |
| **Locking around cache population** | A mutex (or distributed lock, e.g. Redis `SET key val NX PX <ttl>`) guards the recompute-and-write step so only the lock holder repopulates the cache | Must handle lock timeout/leader crash so followers aren't stuck waiting forever |
| **Probabilistic early expiration (a.k.a. XFetch)** | Recompute slightly *before* the actual TTL expiry, with probability increasing as the entry approaches expiry, so refreshes are spread out instead of all landing on the same instant | Some extra recompute work ahead of strict necessity; smooths the spike rather than fully eliminating duplicate work |

**Request coalescing sketch** (single-flight pattern — only the leader calls the expensive `compute_fn`, followers block on its result):

```python
import threading

_inflight: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()
_results: dict[str, str] = {}

def get_with_coalescing(key: str, compute_fn) -> str:
    with _inflight_lock:
        if key in _results:
            return _results[key]
        if key in _inflight:
            event = _inflight[key]              # someone else is already computing this key
            is_leader = False
        else:
            event = threading.Event()
            _inflight[key] = event               # claim leadership for this key
            is_leader = True

    if is_leader:
        try:
            value = compute_fn()                 # only the leader hits the LLM / embedder / retriever
            _results[key] = value
            return value
        finally:
            event.set()                           # wake up any followers
            with _inflight_lock:
                del _inflight[key]
    else:
        event.wait()                              # follower blocks instead of recomputing
        return _results[key]
```

In a multi-instance deployment, `threading.Lock`/`Event` only coalesces requests *within one process* — use a distributed primitive (Redis `SETNX`/lock, or a library like `redsync`) to coalesce across instances.

### Cache Warming

Rather than relying purely on **reactive** population (a cache entry is only ever filled the moment a real user request misses), **cache warming** proactively populates the cache with expected-popular entries *before* traffic hits it:

- **Post-deploy/restart:** re-run the top N most common queries (or re-seed the embedding cache for frequently-asked questions) immediately after a new instance comes up, so the first real users don't pay the cold-cache penalty — and don't collectively cause a stampede against a fresh backend.
- **Static prefixes:** pre-issue a request containing the system prompt / fixed tool-schema block before traffic ramps up, so the LLM prompt-prefix cache (see Cache Type 4 above) is already warm rather than every concurrent request paying full price for the first cache write.
- **Predictable spikes:** if a traffic spike is foreseeable (a scheduled release, a marketing push, a known daily peak), warm the cache for the queries expected to dominate that spike ahead of time rather than discovering the hot keys reactively under load.

Cache warming and stampede mitigations are complementary: warming reduces how often the cache is cold in the first place, while coalescing/locking/early-expiration limit the damage on the misses that still happen.

---

## Cost / Latency Impact Summary

| Cache Layer | Latency Saved | Cost Saved | Staleness Risk |
|-------------|---------------|------------|----------------|
| Semantic query cache (full response) | 500–2000ms | ~100% of LLM cost for hit | High if TTL too long |
| Embedding cache (exact match) | 20–100ms | Embedding API cost | None (deterministic) |
| Retrieval result cache | 10–50ms | ANN search cost | Medium (depends on index update freq) |
| LLM prompt prefix caching | 100–500ms | 70–95% of input token cost for static prefix | None (server-managed TTL) |

---

## Interview Q&A

**Q: What is the difference between exact-match caching and semantic caching in RAG?** `[Basic]`

Exact-match caching keys the cache on the verbatim query string — only byte-identical queries hit the cache. Semantic caching embeds the query and checks whether any cached query has cosine similarity above a threshold; semantically equivalent but differently phrased queries share the same cached response. Semantic caching has a much higher hit rate on natural-language queries but introduces the risk of false positives (queries with similar embeddings but different intent).

---

**Q: How does Anthropic/OpenAI prompt caching differ from application-level caching?** `[Intermediate]`

Prompt caching is **server-side**: the provider caches the computed KV attention tensors for the static prefix of your prompt on their infrastructure. You pay a smaller "cache read" fee instead of the full input token price. Application-level semantic caching is **client-side**: you skip the LLM call entirely and return a previously generated response. Prompt caching still runs inference (just cheaper); application-level caching returns a stored string.

---

**Q: Why is semantic caching dangerous in multi-tenant RAG?** `[Advanced]`

If two tenants submit semantically similar queries (e.g., both ask "What is our Q3 revenue?"), a naively implemented semantic cache keyed only on the query vector will return one tenant's response to the other. The response may contain confidential financial data. Mitigation: always include the tenant ID in the cache key, so each tenant has an isolated cache namespace.

---

**Q: When should you avoid caching in a RAG system?** `[Intermediate]`

Avoid caching: (1) when responses are personalized or user-context-dependent, (2) when the knowledge base updates more frequently than the cache TTL, (3) when the query distribution is long-tailed (each query is unique, so hit rate is near zero), and (4) when strict freshness SLAs require always reflecting the latest documents.

---

**Q: What is a cache stampede and how do you prevent one?** `[Advanced]`

A cache stampede (thundering herd) occurs when a popular cache entry expires — or the cache is cold after a restart — and many concurrent requests for that same key all miss at the same time, each independently re-running the expensive underlying operation (LLM generation, re-embedding, retrieval), wasting compute and potentially overwhelming the backend. Standard preventions: (1) request coalescing/single-flight, where only the first miss ("leader") computes the value while concurrent requests for the same key wait on that result instead of recomputing; (2) a lock (in-process mutex or distributed lock, e.g. Redis `SETNX`) around the recompute-and-write step so only one caller repopulates the entry; and (3) probabilistic early expiration, which recomputes slightly before the actual TTL with increasing probability as expiry nears, spreading refreshes out instead of letting them all cliff-edge at once. Cache warming — proactively populating expected-popular entries before traffic arrives, especially after a deploy/restart or ahead of a known spike — reduces how often a stampede-prone cold miss happens in the first place.
