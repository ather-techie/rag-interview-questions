# 09 — Semantic Cache Leakage

> A cached response generated for one user or tenant is served to a different user whose query is semantically similar but contextually distinct — silently exposing confidential or incorrect information across trust boundaries.

---

## Q1. What is semantic cache leakage and why is it unique to RAG systems? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Exact-match caching** (keyed on verbatim query strings) has no cross-user leakage risk: two different users must type identical characters to hit the same cache entry.

**Semantic caching** (keyed on query embedding similarity) creates a cross-user leakage risk because semantically similar queries — regardless of who asks them — can return the same cached response.

```
Exact-match cache — SAFE:
  Tenant A asks: "What is our Q3 revenue?"          → key: sha256("What is our Q3 revenue?")
  Tenant B asks: "What is our Q3 revenue?"          → same key → B sees A's answer ✗
  Tenant B asks: "Show me our Q3 revenue figures"   → different key → no hit ✓

Semantic cache without tenant isolation — UNSAFE:
  Tenant A asks: "What is our Q3 revenue?"          → embed → cache with score 0.98
  Tenant B asks: "Show me our Q3 revenue figures"   → embed → cosine(A, B) = 0.97 → HIT
  → Tenant B sees Tenant A's "$4.2M revenue" answer ✗ (data leakage)
```

**Why RAG amplifies this risk:**
- RAG answers are grounded in specific retrieved documents from the user's or tenant's corpus
- A cached answer therefore contains information from those specific documents
- If that cache entry is served to a different user, it leaks the underlying document content

**Affected scenarios:**
- Multi-tenant SaaS (each tenant has a private document corpus)
- User-personalized RAG (answers depend on user's access rights)
- Role-based access (a manager's question returns data an employee shouldn't see)
- Any system where different users should get different answers to similar questions

</details>

---

## Q2. How do you reproduce and confirm a semantic cache leakage vulnerability? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Minimal reproduction:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Tenant A's query and cached answer
tenant_a_query    = "What is our Q3 revenue?"
tenant_a_answer   = "Tenant A's Q3 revenue was $4.2M."

# Tenant B's query (similar topic, different tenant)
tenant_b_query    = "Show me the Q3 revenue figures for our company"

# Check semantic similarity
emb_a = model.encode(tenant_a_query, normalize_embeddings=True)
emb_b = model.encode(tenant_b_query, normalize_embeddings=True)
similarity = float(np.dot(emb_a, emb_b))

print(f"Similarity: {similarity:.3f}")   # → 0.946

# With a threshold of 0.93, this would be a cache HIT
# Tenant B would receive Tenant A's answer: "$4.2M"
```

**Security test harness:**

```python
def test_cache_isolation(cache, tenant_a_id, tenant_b_id):
    """Verify that tenant B cannot retrieve tenant A's cached response."""
    query_a = "What is our quarterly financial performance?"
    answer_a = "Q3 revenue: $4.2M, margin 32%"
    
    # Tenant A caches a response
    cache.set(tenant_a_id, query_a, answer_a)
    
    # Tenant B asks a semantically similar question
    query_b = "How did we perform financially last quarter?"
    result = cache.get(tenant_b_id, query_b)
    
    assert result is None, (
        f"SECURITY VIOLATION: Tenant B received Tenant A's cached answer: {result}"
    )
```

**Risk surface beyond the obvious:**
- **Across sessions for the same user**: If a user asks about a confidential document in session 1, and the cache TTL outlasts the user's permissions (document deleted, access revoked), session 2 can still get the cached answer
- **Snippet leakage in cache metadata**: Some implementations cache the retrieved passage IDs alongside the response — exposing document IDs across tenants
- **Similar queries, different time context**: "What is the current CEO?" cached last month returns a stale name to a new user even without multi-tenancy

</details>

---

## Q3. How do you implement a tenant-safe semantic cache? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The core fix: **always include the security context (tenant ID, user ID, permission set) in the cache key**, so semantically similar queries from different contexts never collide.

**Simple namespace-based isolation:**

```python
import hashlib, numpy as np, redis, pickle
from sentence_transformers import SentenceTransformer

class TenantSafeSemanticCache:
    def __init__(self, threshold: float = 0.94):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.redis = redis.Redis()
        self.threshold = threshold

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def _namespace_key(self, tenant_id: str, query_hash: str) -> str:
        # Include tenant_id in the Redis key — isolation at key level
        return f"sc:{tenant_id}:{query_hash}"

    def get(self, tenant_id: str, query: str) -> str | None:
        q_emb = self._embed(query)
        
        # Only search within this tenant's cache namespace
        pattern = f"sc:{tenant_id}:*"
        for key in self.redis.scan_iter(pattern, count=100):
            stored = self.redis.get(key)
            if not stored:
                continue
            cached = pickle.loads(stored)
            similarity = float(np.dot(q_emb, np.frombuffer(cached["emb"], np.float32)))
            if similarity >= self.threshold:
                return cached["response"]
        return None

    def set(self, tenant_id: str, query: str, response: str, ttl: int = 3600):
        q_emb = self._embed(query).astype(np.float32)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        key = self._namespace_key(tenant_id, query_hash)
        
        self.redis.setex(
            key, ttl,
            pickle.dumps({"emb": q_emb.tobytes(), "response": response})
        )
```

**Permission-hash based isolation (for user-level ACL):**

When ACLs are more complex than tenant IDs (e.g., user has access to documents A, B, C but not D), include a hash of the access set in the key:

```python
def permission_cache_key(user_id: str, user_doc_access: set[str], query: str) -> str:
    # Hash the sorted set of accessible document IDs
    access_hash = hashlib.sha256(
        ",".join(sorted(user_doc_access)).encode()
    ).hexdigest()[:12]
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
    return f"sc:{user_id}:{access_hash}:{query_hash}"
```

**Warning:** If a user's permissions change (access to a document is revoked), the cached response may still contain information from that document. Handle this with event-driven cache invalidation:

```python
def on_permission_revoked(user_id: str, doc_id: str, cache):
    """Called when a user loses access to a document."""
    # Simple approach: flush all caches for this user
    pattern = f"sc:{user_id}:*"
    keys = list(cache.redis.scan_iter(pattern))
    if keys:
        cache.redis.delete(*keys)
    
    # Better approach: tag cache entries with doc_ids and flush only affected entries
    # (requires storing doc_id metadata in each cache entry)
```

</details>

---

## Q4. What other information besides the full response can leak through a semantic cache? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Even if the response text is correctly isolated, a naively implemented cache can leak information through side channels.

**Side channel 1: Cache hit/miss timing**

```
Tenant B asks: "Do you have information about Project Nightingale?"
  → Cache HIT in 2ms   → B infers: another user has asked about Project Nightingale
  → Cache MISS in 40ms → B infers: nobody has asked about this topic before

Mitigation: Add jitter to cache responses; never return timing metadata.
```

**Side channel 2: Retrieved document IDs in cache metadata**

If your cache stores which document chunks were retrieved (to avoid re-retrieval on cache hit):

```python
# UNSAFE cache entry:
{
  "response": "...",
  "retrieved_doc_ids": ["doc_42", "doc_99", "doc_7"]  # ← leaks doc existence
}
```

If tenant B sees these document IDs, they learn that documents 42, 99, and 7 exist and contain relevant information — even if they don't have access to read those documents.

**Fix:** Never store or expose retrieved document IDs in cache entries that cross tenant/user boundaries.

**Side channel 3: Suggested follow-up queries**

Some RAG systems generate suggested follow-up questions. If these are cached and leaked, they reveal what information tenant A was researching:

```
Tenant A response cache:
  Follow-ups: ["What was the Q4 guidance?", "How does this compare to our competitor X?"]
  → Reveals A was researching competitor X
```

**Side channel 4: Streaming token count**

If the cache stores the token count of the cached response and that's visible in response metadata, the length of the answer may reveal whether a document is long/detailed (implying rich information).

**Defense-in-depth checklist:**

```python
CACHE_ENTRY_SAFE_FIELDS = {"response", "ttl", "created_at"}
CACHE_ENTRY_UNSAFE_FIELDS = {
    "retrieved_doc_ids",      # reveals document existence
    "chunk_texts",            # direct content leak
    "tenant_id_of_creator",   # reveals cross-tenant origin
    "user_id_of_creator",     # reveals user activity
    "retrieval_scores",       # reveals document relevance
}
```

</details>

---

## Q5. How do you audit a deployed semantic cache for leakage vulnerabilities? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Automated audit probe:**

```python
class CacheLeakageAuditor:
    def __init__(self, cache, embedding_model):
        self.cache = cache
        self.model = embedding_model
    
    def probe_cross_tenant_leakage(
        self,
        tenant_a: str,
        tenant_b: str,
        test_queries: list[str]
    ) -> list[dict]:
        findings = []
        
        for query in test_queries:
            # Store a canary response under tenant A
            canary = f"CANARY_RESPONSE_FOR_{tenant_a.upper()}_DO_NOT_LEAK"
            self.cache.set(tenant_a, query, canary)
            
            # Generate semantically similar queries
            variants = self._generate_variants(query)
            
            for variant in variants:
                result = self.cache.get(tenant_b, variant)
                if result == canary:
                    findings.append({
                        "severity": "CRITICAL",
                        "original_query": query,
                        "leaking_variant": variant,
                        "tenant_a": tenant_a,
                        "tenant_b": tenant_b,
                        "similarity": self._similarity(query, variant),
                    })
        
        return findings
    
    def _generate_variants(self, query: str) -> list[str]:
        """Generate semantically similar paraphrases."""
        # Use LLM to generate paraphrases
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content":
                f"Generate 5 different ways to phrase this question:\n{query}\nList only the questions."}]
        )
        return response.content[0].text.strip().split("\n")[:5]
    
    def _similarity(self, a: str, b: str) -> float:
        embs = self.model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
```

**Monitoring in production:**

```python
def log_cache_access(tenant_id: str, query: str, cache_hit: bool, response_source_tenant: str | None):
    """Anomaly: response_source_tenant != tenant_id on a cache hit."""
    if cache_hit and response_source_tenant and response_source_tenant != tenant_id:
        alert({
            "type": "CACHE_TENANT_MISMATCH",
            "querying_tenant": tenant_id,
            "response_tenant": response_source_tenant,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "severity": "CRITICAL",
        })
```

</details>

---

## Real-World Applications

- **Multi-tenant SaaS platforms**: Any product that caches RAG responses across users sharing infrastructure is vulnerable without explicit tenant namespacing
- **Healthcare RAG**: HIPAA implications — patient data from one patient's cached response must never surface to another patient's request
- **Financial compliance**: SEC/SOX requirements mean a trader's RAG responses about specific securities must not leak to other traders via cache
- **Document management systems** (SharePoint, Confluence): Permission changes (document moved to restricted folder) must trigger cache invalidation to prevent continued access via stale cached responses
