# Multi-Tenancy and Access Control: Who Can Retrieve What

> Retrieval is a read path into every document you've indexed — multi-tenant isolation and document-level ACLs decide whether that read path becomes a data breach.

---

## What is Multi-Tenancy (in RAG)?

Multi-tenancy in RAG means the system serves multiple users, customers, or organizations from a shared index while making sure each one only ever retrieves documents they're authorized to see. Because retrieval is a read path into everything you've indexed, without proper tenant isolation and document-level access control, a query from one user could surface another user's private data — turning a retrieval bug into a data breach.

---

## Why Access Control Is Harder in RAG Than in Search

A traditional database query touches rows the caller is explicitly authorized to read. A RAG query does something more dangerous: it takes *any* natural-language input and returns the semantically closest content from the *entire index* — then paraphrases it through an LLM. If the index contains documents the user shouldn't see, similarity search will happily surface them.

Two distinct problems get conflated in interviews. Keep them separate:

1. **Tenant isolation:** Customer A must never retrieve Customer B's documents. Coarse-grained, boundary is the tenant.
2. **Document-level access control:** Within one tenant, user Alice can see the HR folder and user Bob cannot. Fine-grained, boundary is the (user, document) pair, and it changes constantly.

```
                    ┌────────────────────────────────┐
                    │        The Two Boundaries      │
                    ├────────────────────────────────┤
  Tenant boundary   │ Tenant A  │  Tenant B          │  ← isolation model
  (coarse, stable)  │           │                    │     (Section below)
                    ├───────────┼────────────────────┤
  Document boundary │ Alice: HR docs, Eng docs       │  ← ACL propagation
  (fine, volatile)  │ Bob:   Eng docs only           │     (Section below)
                    └────────────────────────────────┘
```

Note this is a different threat than [prompt injection](prompt_injection_risks.md): injection manipulates the LLM via malicious *content*; access-control failure leaks *legitimate* content to the wrong principal. A system can be perfectly injection-hardened and still leak every salary spreadsheet to every employee.

---

## Tenant Isolation Models

Four models, in increasing order of isolation strength. (See [vector_databases.md](vector_databases.md) for which systems support namespaces/partitions natively.)

| Model | Isolation Strength | Cost | Operational Overhead | Noisy-Neighbor Risk | When to Use |
|---|---|---|---|---|---|
| **Shared index + metadata filter** (`tenant_id` field on every chunk) | Weakest — one missing filter clause leaks everything | Lowest (one index, shared resources) | Lowest (one thing to operate) | High (one tenant's traffic/corpus affects all) | Many small tenants (1000s+), low-sensitivity data, free tiers |
| **Namespace / partition per tenant** (Pinecone namespaces, Qdrant shard keys, Postgres schemas) | Medium — isolation enforced by the DB's routing layer, not your query code | Low–medium (shared cluster, per-tenant partitions) | Medium (partition lifecycle: create/delete on tenant onboard/offboard) | Medium (shared compute, separate data) | The default for most B2B SaaS |
| **Index / collection per tenant** | Strong — separate index structures, separate tuning possible | Medium–high (per-index memory floor; HNSW graphs don't share) | High (N indexes to monitor, upgrade, back up) | Low | Hundreds of tenants max; tenants with very different corpus sizes or recall needs |
| **Cluster / database per tenant** | Strongest — separate compute, storage, network, encryption keys, even region | Highest (per-tenant infra floor) | Highest (fleet management) | None | Regulated tenants (HIPAA, FedRAMP, data-residency), contractual single-tenancy |

### The Three Main Layouts

```
1. SHARED INDEX + FILTER          2. NAMESPACE PER TENANT          3. INDEX PER TENANT
┌─────────────────────────┐      ┌─────────────────────────┐      ┌──────────┐ ┌──────────┐
│       One Index         │      │       One Cluster       │      │ Index A  │ │ Index B  │
│  ┌───┐ ┌───┐ ┌───┐      │      │ ┌─────────┐ ┌─────────┐ │      │ ┌──────┐ │ │ ┌──────┐ │
│  │A:1│ │B:7│ │A:2│ ...  │      │ │ ns: A   │ │ ns: B   │ │      │ │HNSW A│ │ │ │HNSW B│ │
│  └───┘ └───┘ └───┘      │      │ │ ┌─┐┌─┐  │ │ ┌─┐┌─┐  │ │      │ └──────┘ │ │ └──────┘ │
│  vectors interleaved,   │      │ │ └─┘└─┘  │ │ └─┘└─┘  │ │      │ own      │ │ own      │
│  one HNSW graph         │      │ └─────────┘ └─────────┘ │      │ tuning,  │ │ tuning,  │
│                         │      │ separate graphs,        │      │ backups, │ │ backups, │
│  WHERE tenant_id = 'A'  │      │ shared compute          │      │ keys     │ │ keys     │
│  (enforced in YOUR code)│      │ (enforced by the DB)    │      │          │ │          │
└─────────────────────────┘      └─────────────────────────┘      └──────────┘ └──────────┘
  Leak = one bug away              Leak = DB routing bug             Leak = infra-level bug
```

**The interview-relevant distinction:** in model 1, isolation is a *convention your application must remember on every query path*. In models 2–4, isolation is *structural* — a query physically cannot traverse another tenant's graph. Structural isolation is what you want to claim in a security review.

**Hybrid pattern (common in practice):** namespace-per-tenant for the long tail of small tenants, dedicated index or cluster for the few large/regulated ones. Route at the API gateway based on a tenant tier flag.

---

## Document-Level ACL Propagation

Within a tenant, permissions live in the *source system* (SharePoint, Confluence, Google Drive, Jira). The index must reflect them.

### The Sync Pipeline

```
SharePoint / Confluence / Drive
    │
    ├──► Connector (crawl or change-feed/webhook)
    │     └──► For each document:
    │           ├─ content        → chunker → embedder
    │           └─ ACL            → resolve to principals
    │                                ├─ allowed_users:  ["alice@co.com"]
    │                                └─ allowed_groups: ["grp-hr", "grp-finance-leads"]
    │
    └──► Vector DB upsert: chunk + embedding + metadata
          {
            "tenant_id": "acme",
            "doc_id": "sp-4421",
            "allowed_users": ["alice@co.com"],
            "allowed_groups": ["grp-hr"]
          }
```

Every chunk inherits its parent document's ACL. At query time, retrieval filters to chunks where the requesting user (or one of their groups) appears in the allow lists.

### The Staleness Problem

ACLs are copied into the index, so they go stale the moment the source changes:

- HR revokes Bob's access to the compensation folder at 9:00.
- Your connector syncs hourly. Until ~10:00, the index still lists Bob.
- Bob queries "engineering salary bands" at 9:30 and gets the document.

This is the access-control flavor of the [stale index problem](../03_failure_modes/04-stale_index_problem.md) — and it's worse than stale *content*, because a stale permission is a live security hole, not just a wrong answer. Mitigations:

| Mitigation | Mechanism | Trade-off |
|---|---|---|
| Change-feed ACL sync | Subscribe to permission-change events (e.g., SharePoint change API, Drive activity feed); patch metadata only — no re-embed needed | Connector complexity; event feeds can drop/lag |
| Short sync interval for ACLs | Sync permissions every few minutes even if content syncs hourly | Source-system API rate limits |
| **Late-binding check** | After retrieval, verify the user can *still* open each source doc (live call to source API) before sending chunks to the LLM | +50–200ms per query; source API becomes a runtime dependency — but it's the only mitigation that closes the window to ~0 |
| Fail-closed deletes | On any doc-deleted or access-removed event, tombstone chunks immediately, reconcile later | Possible over-removal until reconciliation |

Senior answer: metadata filters as the cheap first gate, late-binding verification on the final top-k as the authoritative gate.

### Group Expansion: Where to Expand?

ACLs are usually granted to *groups*; queries come from *users*. Someone has to expand `alice → [grp-hr, grp-eng, grp-all-staff]` and groups can nest.

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Expand at query time** (recommended default) | Resolve the user's transitive group memberships from the IdP (cached ~5–15 min); filter chunks on `allowed_groups ∩ user_groups ≠ ∅` | Index stores compact group IDs; group membership changes propagate at cache-TTL speed without touching the index | Per-query IdP lookup (mitigate with cache); filter is an OR over possibly hundreds of groups |
| **Expand at index time** (flatten groups to users in chunk metadata) | Store `allowed_users: [every resolved member]` per chunk | Query filter is a single equality check — fast and simple | A single group-membership change forces metadata rewrites across *every* chunk that group touches; allow lists with 10K users bloat metadata; staleness window now applies to membership too |

Rule of thumb: **expand the user at query time, never flatten groups into the index.** Group membership changes far more often than document ACLs.

### Beyond Flat ACLs: Zanzibar / OpenFGA for Nested Permission Hierarchies

Everything above models permissions as flat allow-lists per chunk (`allowed_users`, `allowed_groups`). That's sufficient when the source system already hands you a flattened list per document. It breaks down when access is *inherited through a hierarchy* — folder → team → org, or document → workspace → tenant — because "can Alice see this doc" now requires walking a graph of relationships, not checking membership in two flat lists.

**Google Zanzibar** is the relationship-based access control (ReBAC) model behind Google's internal authorization system (paper: *"Zanzibar: Google's Consistent, Global Authorization System,"* USENIX ATC 2019), used to answer permission checks for Drive, YouTube, Calendar, and others at global scale (Google has reported 10M+ QPS, 99.999% availability, trillions of stored relationships, billions of users). Instead of per-object ACL rows, permissions are stored as relationship tuples of the form `<object>#<relation>@<user>` — e.g., `doc:planA#viewer@user:alice`, or `doc:planA#viewer@group:eng#member` (members of a group inherit the relation). A check like "can alice view doc:planA" walks this tuple graph, following per-relation rewrite rules (union/intersection/exclusion) until it resolves to a concrete user. This is strictly more expressive than flat ACLs or RBAC roles: it directly represents "viewer because member of the group that was granted viewer" or "editor because owner of the parent folder," without denormalizing every inherited grant onto every leaf object.

**OpenFGA** is the open-source implementation of the Zanzibar model — a CNCF project (incubating as of November 2025; originated at Auth0/Okta, donated to CNCF in 2022). You define an authorization model — types (`document`, `folder`, `team`, `user`) and, per type, the relations it supports and who can hold them:

```
type document
  relations
    define parent: [folder]
    define viewer: [user, team#member] or viewer from parent

type folder
  relations
    define viewer: [user, team#member]
```

Relationship tuples then instantiate the model (`document:q3-plan#parent@folder:finance`, `folder:finance#viewer@team:finance-leads#member`), and a `Check` call answers "can user:bob view document:q3-plan" by traversing that graph — including the `viewer from parent` rewrite, so folder-level access flows down to every document inside it without a tuple per document.

**Why this matters for RAG:** the flat `allowed_users`/`allowed_groups` filter earlier in this section is the right default when a connector already flattens permissions per document — most SaaS connectors do. It stops being sufficient once the system must reason about *inherited* access across a hierarchy the connector doesn't flatten (nested folders, team membership implying access to every team's resources, org-to-suborg delegation): "flatten at index time" reproduces the group-expansion staleness problem one hierarchy level deeper, and "expand at query time" now means a graph traversal (an OpenFGA-style `Check` call) rather than a set-intersection. In practice, the chunk metadata stays the same (`doc_id` + `tenant_id`), but the ACL predicate at query/late-binding time becomes "call the authorization service's Check API" instead of "intersect two sets." Worth naming this vocabulary in an interview when asked "what if permissions aren't already flat" — it signals awareness that ReBAC is a distinct, more expressive tier above RBAC/flat-ACL, not that you'd reimplement a relationship graph inside the vector DB's metadata filter.

---

## Filter-at-Query vs. Filter-Post-Retrieval

This builds on the pre/post-filter strategies in [vector_databases.md](vector_databases.md#metadata-filtering-and-its-performance-cost) — but with ACLs, the choice has *security* consequences, not just recall consequences.

### Pre-Filtering (filter inside the ANN search)

The metadata predicate is evaluated *during* index traversal; non-matching vectors are never candidates.

- **Correctness:** strong — unauthorized chunks cannot appear in results.
- **The recall catch:** with HNSW, a highly selective filter (user can see 0.1% of the corpus) means most graph edges lead to non-matching nodes. Greedy traversal can get stranded in regions with no eligible vectors, degrading recall or latency. Engines handle this differently: some fall back to brute-force scan below a match-ratio threshold (correct but slower); ACORN-style traversal skips ineligible nodes while continuing through them.

### Post-Filtering (retrieve first, filter after)

Fetch top-k ignoring ACLs, then drop unauthorized chunks.

- **Recall:** search sees the full graph.
- **The under-fill problem:** if the user can access 1% of the corpus, top-10 may contain *zero* authorized chunks → empty answer.
- **The side channels:** even though the user never sees filtered content, post-filtering leaks signal — variable result *counts* and *latency* correlated with how much restricted content matched the query, and naive implementations that rerank or log before filtering. ("I asked about 'Project Falcon' and got 2 results instead of 10... so restricted Falcon docs exist.")

### Over-Fetching (the pragmatic middle)

Retrieve k′ = k × (expansion factor, e.g., 5–10×), post-filter, return top-k. Tunable, but for highly restricted users no finite k′ guarantees k results — and k′ scales your reranking cost.

| Strategy | Authorization Correctness | Recall | Latency | Side Channels |
|---|---|---|---|---|
| Pre-filter (in-ANN) | Enforced at search layer | Can degrade on selective filters | Higher on selective filters (fallback scans) | Minimal |
| Post-filter | Enforced *only if every downstream consumer filters* | Full-graph recall | Fast search, may need retries | Count + timing leak |
| Over-fetch + filter | Same caveat as post-filter | Good for moderate selectivity | k′ × reranking cost | Reduced but present |

**Default answer for interviews: pre-filter on `tenant_id` (always) + ACL predicate, and let the engine's filtered-ANN implementation handle selectivity. Post-filtering as a *security boundary* is fragile because every new pipeline stage must re-remember to filter.**

### How Major Vector DBs Handle Filtered ANN

| System | Mechanism | Notes |
|---|---|---|
| Pinecone | Metadata filtering applied during search within a namespace | Combine namespace (tenant) + metadata filter (ACL); single-stage filtered search, no manual over-fetch needed |
| Qdrant | Payload filtering with payload indexes; filterable HNSW | Builds extra graph links so filtered traversal stays connected; `group_id`-style payload + shard keys for tenancy |
| Weaviate | `where` filter with inverted index on properties; multi-tenancy feature gives one shard per tenant | Tenant shards are structural isolation; `where` handles document ACLs |
| pgvector | Plain SQL `WHERE` before/alongside the vector operator | Planner may or may not use the HNSW index with the filter — iterative scan modes exist; bonus: Postgres row-level security can enforce tenancy *below* application code |
| Milvus | Boolean expression filtering + partitions/partition keys | Partition key per tenant ≈ namespace model |

### Code: Query-Time ACL Filter

```python
def retrieve_with_acl(query: str, user: User, k: int = 5) -> list[Chunk]:
    """Tenant isolation + document ACL enforced inside the ANN search."""

    # 1. Group expansion at query time (cached, TTL ~10 min)
    groups = idp.get_transitive_groups(user.id)   # ["grp-eng", "grp-all-staff"]

    # 2. Pre-filter: tenant scope AND (user allowed OR any group allowed)
    acl_filter = {
        "must": [
            {"key": "tenant_id", "match": {"value": user.tenant_id}},
        ],
        "should": [  # at least one must hold
            {"key": "allowed_users",  "match": {"any": [user.email]}},
            {"key": "allowed_groups", "match": {"any": groups}},
        ],
    }

    results = vector_db.search(
        vector=embed(query),
        filter=acl_filter,        # evaluated DURING graph traversal
        limit=k,
    )

    # 3. Late-binding check on the final candidates (closes the staleness window)
    authorized = [r for r in results
                  if source_system.can_read(user, r.payload["doc_id"])]

    audit_log.record(user=user, query=query, filter=acl_filter,
                     returned=[r.payload["chunk_id"] for r in authorized])
    return authorized
```

The critical property: the filter is built from the *authenticated session*, never from request parameters. Letting the client pass its own filter string is the metadata-injection vector described in [prompt_injection_risks.md](prompt_injection_risks.md) (`namespace = "admin" OR 1=1`).

---

## Leakage Surfaces Beyond Retrieval

Filtering the search is necessary, not sufficient. Five surfaces interviews love to probe:

**1. The embeddings themselves.**
Embedding inversion attacks reconstruct close approximations of the original text from its vector (research has recovered the majority of input tokens from common embedding models). Consequence: a vector store containing embeddings of confidential documents *is* a confidential data store. It inherits the source's data classification — same encryption-at-rest, network isolation, and access policies. "It's just floats" is the wrong answer.

**2. Shared semantic caches.**
A semantic cache keyed only on query similarity will serve Tenant B a cached answer generated from Tenant A's documents. Cache keys must include `tenant_id` — and for document-level ACLs, either the user's permission set (hash of sorted group list) or per-user caching. A cache added later "for cost savings" that sits *in front of* the ACL filter is a classic regression.

**3. LLM provider logging and retention.**
Retrieved chunks leave your boundary when sent to a hosted LLM. If the provider logs prompts, restricted document content now lives in their logs. Enterprise agreements with zero-retention, regional endpoints, or self-hosted models are part of the access-control story, not a separate procurement detail.

**4. Citations and metadata.**
Even if chunk *content* is filtered, returning "3 additional results withheld" or citing the title "Q3 Layoff Plan — CONFIDENTIAL" of a doc the user can't open leaks existence and topic. Filter citations, counts, and "related documents" UI with the same predicate as content.

**5. Cross-tenant contamination via fine-tuning.**
Fine-tuning a model (embedder, reranker, or generator) on pooled multi-tenant data can memorize and regurgitate one tenant's text in another tenant's session. Either train per-tenant, train only on tenant-consented/synthetic data, or don't fine-tune on customer corpora at all.

```
Query ──► Retrieval ──► Rerank ──► Cache ──► LLM ──► Answer + Citations
            │              │          │        │          │
        [filter here]   leak if    leak if   leak via   leak via titles,
         is step 1,     reranker   key omits provider   counts, "see also"
         not the        sees un-   tenant/   logs
         whole job      filtered   ACL set
                        docs
```

---

## Audit and Compliance

Access control you can't *demonstrate* doesn't exist, as far as an auditor is concerned.

### What to Log Per Query

| Field | Why |
|---|---|
| User ID + tenant ID + session | Attribute every retrieval to a principal |
| Timestamp, query text (or hash, if queries themselves are sensitive) | Reconstruct the event |
| **Filter actually applied** (tenant + resolved groups + ACL predicate) | Prove the gate was in place for *this* query — the single most valuable field |
| Chunk IDs + doc IDs returned (pre- and post-late-binding check) | Determine blast radius when a permission bug is found: "who saw doc X between T1 and T2?" |
| Late-binding check outcomes (chunks dropped) | Measures your staleness window in production |
| Model + prompt version | Tie the generated answer to its inputs |

Logs must be append-only/tamper-evident, and note the recursion: the audit log now contains queries and doc IDs, so *it* needs access control and a retention policy too (typical: 1–7 years depending on regime — SOC 2, HIPAA, internal policy).

### Demonstrating to Auditors

- **Design evidence:** architecture doc showing structural tenant isolation + the single enforcement point for ACL filters (one choke-point function, not filters sprinkled across call sites).
- **Operating evidence:** sampled query logs showing filters applied; ACL sync lag dashboards (P95 time from source revocation → index update).
- **Negative testing:** automated cross-tenant probes in CI — synthetic Tenant A user issues queries engineered to be nearest-neighbors of Tenant B content and must get zero results. This is the access-control equivalent of a recall probe set.

---

## Key Takeaways

1. **Isolation should be structural, not conventional.** Namespace/partition per tenant is the default; shared-index-plus-filter means one forgotten `WHERE` clause is a breach.
2. **ACLs in the index are a cache of the source's permissions** — treat staleness as a security bug, and close the window with query-time late-binding checks on the final top-k.
3. **Expand groups at query time, not index time.** Membership churns faster than document ACLs; flattening users into chunk metadata creates massive rewrite storms.
4. **Pre-filter inside the ANN search** and know the recall caveat for selective filters; post-filtering as a security boundary is fragile and leaks via counts and timing.
5. **The vector store inherits the source data's classification** — embeddings are invertible, caches must be tenant- and permission-keyed, and citations leak too.
6. **Log the filter you applied, not just the query** — blast-radius analysis and auditor evidence both depend on it.

---

## Interview Q&A

**Q: How would you design a RAG system over SharePoint that must respect existing document permissions?** `[Advanced]`

Walk through it as seven steps, in order: (1) **Clarify** — one org or multi-tenant? How fresh must permissions be (minutes vs. seconds)? Scale of users/groups? (Maps onto the requirements step in [system_design_principles.md](../00_overview/system_design_principles.md).) (2) **Ingestion** — a connector consumes the SharePoint change feed and produces chunks + embeddings + per-chunk ACL metadata (site/library/item permissions resolved to users + groups). (3) **Identity** — the user authenticates via the IdP (Entra ID); transitive group membership is expanded at query time and cached with a short TTL. (4) **Retrieval** — pre-filter inside the ANN search: `tenant + (user ∈ allowed_users OR groups ∩ allowed_groups)`. (5) **Staleness defense** — a late-binding permission check against SharePoint on the final top-k before generation, failing closed on connector gaps. (6) **Beyond retrieval** — ACL-aware citations, tenant+permission-set cache keys, a zero-retention LLM endpoint. (7) **Audit** — a per-query log of user, filter, and chunk IDs, cross-tenant probe tests, and a sync-lag SLO. Naming the **two-gate model** (cheap metadata pre-filter + authoritative late-binding check) and the **staleness window** explicitly is what separates a senior answer from "just add a metadata filter."

---

**Q: What's the risk of adding a pipeline stage — a reranker, a cache, a "related documents" widget — after the ACL filter has already run?** `[Advanced]`

Any stage added *after* the ACL gate that has its own access to document content can reintroduce a leak, even though retrieval itself was correctly filtered: a **reranker** that re-queries the index "for more candidates" without reapplying the filter; a **semantic cache** that returns an answer generated under a *different* user's (broader) permissions; a **"related documents" / link-expansion** step that follows hyperlinks from authorized chunks into unauthorized docs; or a **summarization memory** that condensed an earlier, more privileged session into reusable context. The principle to state: authorization must be enforced at the last point where document content enters the prompt, and every component between retrieval and generation must be ACL-aware or content-blind.

---

**Q: Why not just filter the LLM's output for restricted content instead of filtering what goes into the prompt?** `[Basic]`

Because by the time restricted content reaches the LLM it has already crossed the trust boundary — it may be logged or retained by the model provider regardless of what the final answer says. Output filtering also relies on the LLM (or a second LLM-as-guard) to correctly recognize and strip every leak, and that step is bypassable the same way any LLM behavior is bypassable. Filter inputs — what gets retrieved — not outputs.

---

**Q: A highly selective ACL filter is tanking retrieval recall — what's happening, and how do you fix it?** `[Advanced]`

When a user can see only a small slice of the corpus (say 0.1%), HNSW's greedy graph traversal can get stranded in regions where almost no node satisfies the filter, since most edges lead to ineligible vectors. Fixes: use a filterable-graph engine that builds extra links to keep filtered traversal connected (e.g., ACORN-style traversal), fall back to brute-force scan below a match-ratio threshold, or move to per-tenant/per-permission-class partitions so the filter *is* the partition boundary rather than a predicate layered on top of a shared graph.

---

**Q: How does tenant offboarding differ between a structurally isolated deployment and a shared-index-plus-filter deployment?** `[Intermediate]`

Structural models (namespace, index, or cluster per tenant) make offboarding easy — drop the namespace or index, which also doubles as a clean answer to a GDPR/right-to-erasure question. Shared-index models require a delete-by-filter pass across every chunk belonging to the tenant, plus verification that the vectors are actually gone from both the live index *and* any backups or snapshots — an operation that's easy to get subtly wrong (partial deletes, tombstones lingering in the HNSW graph, or backups quietly retaining "deleted" vectors).

---

**Q: How do you handle real-time permission changes — for example, a user loses access to a document mid-session?** `[Advanced]`

This is the ACL staleness problem at its most acute. Three layers needed: (1) **Short TTL on permission caches** — if ACLs are cached in the retrieval layer, set TTL to the acceptable staleness window (5–60 seconds for sensitive data). Every cache read should touch the permission store if the cached entry is older than TTL. (2) **Late-binding check on the final top-k** — after retrieval, re-verify the user's current permissions for each document in the final result set against the authoritative permission store (SharePoint API, database), not the index metadata. This is slower but eliminates stale-cache false positives. (3) **Session invalidation** — when a permission change event fires (via webhook, CDC, or audit log), invalidate any active sessions for that user that may have a cached answer containing the now-revoked document's content. If using a semantic cache, flush all cache entries whose source document IDs include the revoked document. The combination of late-binding + session invalidation means a user who loses access mid-session will be blocked at the next query, not at the next session.

---

**Q: What are the security risks of using a semantic cache in a multi-tenant RAG system?** `[Advanced]`

A semantic cache keyed only on query meaning — without a tenant or user scope — is a cross-tenant data leakage vector. If Tenant A asks "What is our Q3 revenue?" and the answer ($4.2M) is cached, Tenant B asking "Show me our Q3 revenue figures" may hit the cache and receive Tenant A's answer, since the query embeddings are semantically very close. Beyond the obvious: (1) **Timing side-channel** — a cache hit is 2ms vs. cache miss 200ms; Tenant B can infer whether a topic has been previously queried by another user just from response latency. (2) **Document-ID leakage in cache metadata** — implementations that cache retrieved chunk IDs alongside the answer expose which documents are relevant to a topic. (3) **Permission-change lag** — a cached answer may include content from a document that has since been classified as restricted; without permission-aware cache invalidation, the answer persists beyond the revocation. Mitigation: always namespace cache keys by tenant/user ID and include a hash of the user's current permission set in the key. See [09 — Semantic Cache Leakage](../03_failure_modes/09-semantic_cache_leakage.md).
