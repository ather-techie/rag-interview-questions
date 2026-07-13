# 15 — LightRAG

> Combines entity-relationship graph indexing with vector retrieval for dual-level (local + global) queries, at much lower build cost than Microsoft GraphRAG.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Docs
    │
    ▼
Entity-Relationship Extractor (LLM, per chunk)
    │
    ▼
Entity Resolution / Deduplication
    │
    ▼
Dual-level Graph Construction
    ├─ Local:  entity-specific keys (nodes + 1-hop edges)
    └─ Global: community/theme summaries (optional Leiden clustering)
    │
    ▼
Incremental Graph Updater (insert / update / tombstone documents)
    │
    ▼
Dual-level Hybrid Retriever
    ├─ Local search  (entity-anchored traversal)
    ├─ Global search (community/theme summaries)
    └─ Naive search  (flat vector fallback)
    │
    ▼
Generator ──► Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| Entity-Relation Extractor | LLM extracts entities and relationships per chunk |
| Dual-level Graph Index | Stores local entity-centric keys and global community/theme keys |
| Incremental Updater | Adds/updates/tombstones nodes and edges without a full rebuild |
| Local Retriever | Entity-anchored 1-hop traversal for relational queries |
| Global Retriever | Community-summary retrieval for synthesis queries |
| Generator | Synthesizes the answer from merged local+global (or naive) context |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Library | LightRAG (`pip install lightrag-hku`) |
| Graph storage | NetworkX (default, small corpora), Neo4j (production scale) |
| Community detection | Leiden algorithm (optional) |
| Embedding model | BGE, OpenAI text-embedding-3-small |
| Extraction LLM | GPT-4o-mini with structured-output entity/relationship extraction |

---

## Q1. What is LightRAG and what problem does it solve over standard vector RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**LightRAG** (2024) is a graph-enhanced RAG framework that indexes both *text chunks* and *entity-relationship pairs* extracted from those chunks, enabling two retrieval modes that flat vector RAG cannot handle:

**What standard vector RAG misses:**

1. **Relational queries** — "How are Entity A and Entity B connected?" A flat vector search returns chunks mentioning both, but does not reconstruct the relationship path between them.

2. **Global synthesis queries** — "What are the recurring themes across all documents in this corpus?" Flat vector retrieval only finds chunks near the query; it can't synthesize across the entire corpus.

**How LightRAG addresses this:**

```
Document corpus
     │
     ▼
Entity + Relationship Extractor (LLM)
     │
     ├─ Entities: (Apple Inc.), (Tim Cook), (iPhone 15)
     └─ Relationships: (Tim Cook) --[CEO of]--> (Apple Inc.)
                       (Apple Inc.) --[manufactures]--> (iPhone 15)
     │
     ▼
Dual-level Index:
  Local:  Entity-centric graph (for relational queries)
  Global: Community summaries (for synthesis queries)
  Vector: Standard chunk embeddings (for semantic queries)
```

LightRAG's key advantage over Microsoft GraphRAG: **lower build cost** — community detection (Leiden algorithm) is optional, not mandatory.

</details>

---

## Q2. How does LightRAG's dual-level retrieval work? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

LightRAG supports four retrieval modes, selectable per query:

**1. Local mode (entity-centric)**
- Extract entities from the query.
- Find those entities in the knowledge graph.
- Retrieve the entity's immediate neighbors (1-hop) and associated chunks.
- Best for: "What does Apple Inc. manufacture?" — entity-anchored factual queries.

**2. Global mode (community-centric)**
- Retrieve pre-computed community summaries from the graph.
- Merge summaries that are semantically relevant to the query.
- Best for: "What are the major themes in this corpus?" — no specific entity anchor.

**3. Hybrid mode (local + global)**
- Run both local and global retrieval; merge results.
- Best for: mixed queries ("Tell me about Apple Inc.'s role in the tech industry").

**4. Naive mode (flat vector only)**
- Standard ANN search on chunk embeddings.
- Fallback for queries that don't benefit from graph structure.

```python
from lightrag import LightRAG, QueryParam

rag = LightRAG(working_dir="./lightrag_store", llm_model_func=llm_func)

# Local: entity-anchored
result = rag.query("What products does Apple manufacture?",
                   param=QueryParam(mode="local"))

# Global: synthesis
result = rag.query("What are the major technology trends discussed?",
                   param=QueryParam(mode="global"))

# Hybrid: default for unknown query types
result = rag.query("How does Apple influence the smartphone market?",
                   param=QueryParam(mode="hybrid"))
```

</details>

---

## Q3. How does LightRAG build its knowledge graph? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

LightRAG's graph is built by an LLM extraction pipeline:

**Step 1 — Chunk the documents**

Standard chunking (e.g., 1,200 tokens per chunk with overlap).

**Step 2 — Extract entities and relationships per chunk**

For each chunk, prompt the LLM to extract:
- Entities: name, type, description
- Relationships: source entity, target entity, relationship type, description, strength (1–10)

```python
EXTRACTION_PROMPT = """Extract entities and relationships from the following text.
For each entity: name, type (PERSON/ORG/PRODUCT/CONCEPT/etc.), brief description.
For each relationship: (entity1) -[relationship_type]-> (entity2), description, strength 1-10.
Text: {chunk}
"""
```

**Step 3 — Entity resolution (deduplication)**

Entities with similar names are merged (e.g., "Apple", "Apple Inc.", "Apple Computer" → "Apple Inc."). LightRAG uses embedding similarity + LLM-based merge decisions.

**Step 4 — Graph storage**

Entities → nodes; relationships → edges. Stored in:
- NetworkX (in-memory, for small corpora)
- Neo4j or other graph databases (for production)

**Step 5 — Community summaries (optional)**

LightRAG optionally runs community detection (Leiden) on the graph and generates per-community summaries with an LLM — similar to Microsoft GraphRAG, but optional rather than required.

**Cost:** O(N) LLM calls for extraction (one per chunk) plus O(K) for entity resolution where K = number of entity collision pairs.

</details>

---

## Q4. How does LightRAG compare to Microsoft GraphRAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| | LightRAG | Microsoft GraphRAG |
|---|---|---|
| **Build pipeline** | Entity/relationship extraction → graph → optional community detection | Entity/relationship extraction → graph → mandatory Leiden community detection → community summaries |
| **Build cost** | O(N) LLM calls | O(N log N)–O(N²) LLM calls (community summarization is expensive) |
| **Global query support** | Community summaries (optional) | Community summaries (mandatory, primary design goal) |
| **Local query support** | Strong — entity-centric retrieval | Moderate — local search also supported |
| **Implementation** | Open-source Python library (pip install lightrag-hku) | Open-source but heavier infrastructure |
| **Graph storage** | NetworkX (default), Neo4j optional | Parquet files or CosmosDB |
| **Incremental updates** | Supported (insert new docs, extract new entities, merge) | Full rebuild preferred |
| **Best for** | Medium-scale corpora, mixed query types, lower ops overhead | Large-scale corpora where global synthesis is the primary use case |

**Rule of thumb:**
- Use **LightRAG** when you want graph-enhanced retrieval without the full complexity and build cost of GraphRAG.
- Use **Microsoft GraphRAG** when global community-level synthesis queries dominate and you have the build budget.

</details>

---

## Q5. How do you handle entity resolution in LightRAG for a large, noisy corpus? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Entity resolution (deduplication) is one of the hardest problems in knowledge graph construction. LightRAG faces it when the same entity is mentioned differently across documents.

**Common patterns requiring resolution:**
- Abbreviations: "US Fed" vs. "Federal Reserve" vs. "Fed"
- Aliases: "Elon Musk" vs. "the Tesla CEO" vs. "EM"
- Misspellings: "Microsoft" vs. "Microsft"
- Contextual references: "the company" (referring to Apple in context)

**LightRAG's default approach:**
1. Embed all extracted entity names.
2. Cluster entities with cosine similarity > threshold (e.g., 0.92).
3. Within each cluster, prompt an LLM: "Are these entity mentions the same entity? [list of mentions]" → merge if yes.

**Production improvements:**

1. **Type-constrained resolution** — Only compare entities of the same type (don't try to merge "Apple Inc." [ORG] with "Apple" [PRODUCT]).

2. **Anchor-based resolution** — If an entity has a high-confidence canonical name (from a knowledge base like Wikidata), use it as the anchor and resolve all variants to it.

3. **Incremental resolution** — When new documents arrive, only resolve new entities against the existing entity set (not full pairwise comparison).

4. **Confidence thresholds** — Don't auto-merge; instead, create candidate merge pairs with confidence scores. Auto-merge only above 0.95; flag 0.80–0.95 for human review.

**Cost of poor resolution:**
- Under-resolution: fragmented graph with duplicate nodes reduces recall for relational queries.
- Over-resolution: merging distinct entities creates false connections (e.g., merging "Apple Inc." with "Apple Records").

</details>

---

## Q6. What are LightRAG's failure modes compared to pure vector RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

LightRAG introduces new failure modes on top of standard vector RAG failures:

**1. Relationship hallucination during extraction**

The LLM extracting entities/relationships may invent relationships not present in the source text.

- Example: "Tim Cook announced a partnership with Google" (source says no such thing) → false edge in graph → false answers to relational queries.
- **Detection:** Faithfulness check: for each extracted relationship, verify it is supported by the source chunk via NLI.
- **Mitigation:** Low-temperature extraction prompts; structured output schemas; human review of high-centrality edges.

**2. Entity graph fragmentation**

Under-resolved entities create disconnected subgraphs. A query for "Apple Inc.'s revenue" may miss all chunks that referred to "Apple" without "Inc."

**3. Stale graph on document update**

If a document is updated, existing graph nodes and edges from that document are not automatically updated. Stale relationships can contradict new information.
- **Mitigation:** Tombstone nodes/edges from updated documents and re-extract.

**4. Graph query complexity for simple queries**

Running entity extraction + graph traversal for a query that only needs a single-chunk fact lookup wastes latency. Naive mode (flat vector) should be used for simple queries.

**5. Community summary quality degradation**

If community detection groups unrelated entities (a common failure of unsupervised clustering), community summaries are incoherent and global mode answers are poor.

</details>

---

## Q7. How do you decide when to use LightRAG vs. standard vector RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Use this decision framework:

```
Does your corpus contain rich entity relationships?
(e.g., people, organizations, products, events, policies, technical concepts)
  │
  ├─ No (e.g., FAQ docs, legal boilerplate, how-to guides)
  │    → Standard vector RAG (LightRAG adds cost with no benefit)
  │
  └─ Yes
       │
       Do your users ask relational queries?
       ("How are X and Y connected?", "Who works at which company?")
         │
         ├─ No → Standard vector RAG
         │
         └─ Yes
              │
              Do your users ask global synthesis queries?
              ("What are the main themes?", "Summarize all risk factors")
                │
                ├─ No → LightRAG local mode
                │
                └─ Yes → LightRAG hybrid/global mode
```

**Corpus size guidelines:**
- < 1,000 chunks: standard RAG is fine; graph overhead not worth it.
- 1,000–100,000 chunks: LightRAG adds clear value for relational queries.
- > 100,000 chunks: Consider Microsoft GraphRAG for global queries; LightRAG for local.

</details>

---

## Q8. How does LightRAG handle incremental document updates? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

LightRAG supports incremental updates, which is a key advantage over full-rebuild-on-update approaches:

**Adding new documents:**
```python
rag.insert("New document text here...")
# OR for file
rag.insert_file("path/to/new_doc.txt")
```

Internally:
1. Chunk the new document.
2. Extract entities and relationships (LLM calls for new chunks only).
3. Resolve new entities against the existing entity set (targeted, not full pairwise).
4. Add new nodes/edges to the graph.
5. Re-embed and index new chunks.
6. Optionally re-run community detection on affected subgraphs.

**Updating existing documents:**
1. Identify chunks sourced from the updated document (by `doc_id` metadata).
2. Tombstone all nodes/edges sourced from those chunks (`status: deprecated`).
3. Run the insert pipeline for the updated document.
4. After verification, delete tombstoned nodes/edges.

**Deleting documents:**
1. Find all nodes/edges with `source_doc_id = deleted_doc_id`.
2. Check if any of those nodes appear in other documents' relationships (don't delete shared entities).
3. Remove unique nodes/edges; decrement reference counts on shared nodes.

**Limitation:** If community detection was run, community summaries must be regenerated for affected communities after updates. This is the most expensive part of incremental updates in LightRAG with community mode enabled.

</details>

---

## Q9. How do you evaluate LightRAG's graph quality? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Graph quality evaluation covers both structural metrics and retrieval impact:

**Structural metrics:**

| Metric | How to measure | Target |
|--------|---------------|--------|
| Entity coverage | % of known named entities in corpus present in graph | > 90% |
| Relationship precision | % of extracted relationships that are faithful to source | > 95% |
| Duplicate entity rate | % of entity pairs that are the same entity but not merged | < 5% |
| Orphan node rate | % of nodes with no edges (often extraction artifacts) | < 10% |
| Average node degree | Mean edges per node | Domain-dependent; check against expected |

**Retrieval quality evaluation:**

Create three query sets:
1. **Relational queries** (test local mode): "What is the relationship between X and Y?" — expected: specific edge + supporting chunks.
2. **Synthesis queries** (test global mode): "What are the main themes in the corpus?" — expected: coherent synthesis, not just keyword matching.
3. **Factual queries** (test naive mode): Single-chunk fact lookups — expected: performance parity with standard RAG.

Measure: Precision@5, Recall@5, answer correctness (LLM-as-judge).

**Red flags:**
- Global mode answers that are generic ("The corpus discusses many interesting topics") — community summaries are too coarse.
- Local mode answers that confabulate relationships — extraction hallucination.
- Relational queries answered worse than vector-only RAG — graph not adding value.

</details>

---

## Q10. Design a production LightRAG system for a financial research corpus. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
CORPUS: Earnings reports, analyst notes, SEC filings, news articles
TARGET QUERIES:
  Local:  "What companies did Berkshire Hathaway acquire in 2023?"
  Global: "What are the main risks discussed across Q3 2024 filings?"
  Mixed:  "How did Apple's supply chain issues affect its revenue?"

INGESTION PIPELINE
──────────────────
Raw documents (PDF, HTML, text)
  → PDF parser / HTML stripper → plain text
  → Chunker (1,200 tokens, structure-aware — preserve table rows)
  → LightRAG.insert() per document:
       Entity/Rel extraction (gpt-4o-mini, structured output JSON schema)
       Entity resolution (embedding similarity threshold 0.92)
       Graph upsert (NetworkX → Neo4j for scale)
       Chunk embedding (text-embedding-3-small)
  → Metadata stored: {doc_id, source_date, company_ticker, filing_type}

QUERY PIPELINE
──────────────
User query
  → Query classifier:
       Relational keywords ("relationship", "connection", "acquired", "subsidiary") → local
       Synthesis keywords ("themes", "trends", "overall", "across all") → global
       Mixed or unclear → hybrid
  → LightRAG.query(query, mode=classified_mode)
  → Post-retrieve: metadata filter by date range if query contains temporal reference
  → Generation with citations (chunk IDs + entity sources)

GRAPH SCHEMA (financial domain)
────────────────────────────────
Entity types: COMPANY, PERSON, PRODUCT, MARKET, REGULATION, EVENT
Relationship types: ACQUIRED, COMPETES_WITH, SUPPLIES_TO, REGULATED_BY,
                    EXECUTIVE_OF, REPORTED_REVENUE_OF, RISKS_ASSOCIATED_WITH

MONITORING
──────────
- Entity extraction faithfulness: sample 50 new relationships/week → human spot-check
- Graph growth rate: new nodes/edges per day (sudden spike = potential hallucination)
- Query mode classification accuracy: evaluate on 100-query holdout set monthly
- P95 latency per mode: local < 500ms, global < 1500ms, hybrid < 2000ms
```

</details>

---

## Q11. How does LightRAG handle queries that span both local and global contexts? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

LightRAG's **hybrid mode** merges local and global retrieval results before passing context to the LLM.

**Hybrid mode internals:**

```
User query: "How does Apple's relationship with TSMC affect the global chip market?"

1. Entity extraction from query → [Apple, TSMC, chip market]

2. Local retrieval:
   - Find Apple and TSMC in graph
   - Traverse: Apple -[supplies_from]-> TSMC, TSMC -[produces]-> chips
   - Retrieve chunks associated with these relationships
   - Top-5 local chunks

3. Global retrieval:
   - Find community summaries that include Apple, TSMC, semiconductor industry
   - Top-3 community summaries

4. Merging strategy:
   - Deduplicate (chunks already referenced in community summaries)
   - Interleave: local chunks first (specific), then global summaries (context)
   - Token budget: local chunks get 60%, global summaries get 40%

5. LLM generation with merged context
```

**Why hybrid outperforms each mode alone:**
- Local mode alone: gives specific facts but may miss the market-wide context.
- Global mode alone: gives market trends but may miss the specific Apple-TSMC relationship.
- Hybrid: specific relationship + broader context → richer answer.

**Tuning the merge ratio:**

The 60/40 local/global split is a default. For queries that are more strategic ("market trends"), increase global weight. For queries that are more factual ("specific contract terms"), increase local weight. This can be made query-adaptive with a small classifier trained on query type.

</details>

---

## Q12. What are the security risks introduced by LightRAG's knowledge graph layer? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

LightRAG's graph introduces security risks beyond standard RAG:

**Risk 1 — Relationship inference leakage**

The graph stores explicit relationships that may be inferred from documents even when those relationships are not meant to be disclosed:

- Example: HR documents mention that Alice is Bob's manager. The graph stores this as an explicit edge. A query "Who manages Alice?" returns this relationship even if the querier shouldn't have access to org chart data.
- **Mitigation:** Apply ACL labels to graph edges, not just chunks. An edge should only be returned if the querier has access to the source chunk that generated it.

**Risk 2 — Entity-based data aggregation across tenant boundaries**

In a multi-tenant graph, an entity (e.g., a company name) may appear in multiple tenants' documents. If entity deduplication merges cross-tenant entities, a query against Tenant A's entity may surface relationships from Tenant B's documents.
- **Mitigation:** Scope entity resolution within tenant boundaries. Never merge entities across tenants. Tag all edges with `tenant_id` and filter at query time.

**Risk 3 — Graph poisoning via adversarial documents**

An attacker who can insert documents into the corpus can insert fabricated relationships:
```
"According to internal sources, [LegitCompany] is planning to acquire [TargetCompany]."
```
This creates a fake acquisition edge in the graph. Global synthesis queries will include this false relationship in synthesis answers.
- **Mitigation:** Relationship faithfulness gate at extraction time. Flag relationships asserting financial events (acquisitions, mergers) for human review.

**Risk 4 — Entity traversal as a covert data path**

In local mode, graph traversal can surface chunks that are semantically distant from the query but connected via entity relationships. This may surface chunks outside a user's intended access scope.
- **Mitigation:** Apply chunk-level ACL checks on all nodes returned via graph traversal, not just on directly retrieved chunks.

</details>

---

## Real-World Applications

| Application | Domain | Why LightRAG Fits |
|---|---|---|
| On-device mobile assistant (e.g., Apple Intelligence, on-device LLM apps) | Consumer / Mobile | Dual local/global retrieval modes work within tight memory and compute budgets without a separate cloud vector store |
| IoT edge knowledge system (factory floor, remote sites) | Industrial / Edge | Edge devices with intermittent connectivity need self-contained graph + vector retrieval; LightRAG's lightweight design avoids cloud dependency |
| Personal knowledge management app (e.g., Obsidian AI, Logseq AI) | Productivity | Users' personal note graphs benefit from entity-aware retrieval across local markdown files, with no server-side infrastructure required |
| Privacy-first enterprise assistant | Healthcare / Legal | Sensitive corpora that cannot leave the premises are served by LightRAG running entirely on-premises on commodity hardware |
| Small-business chatbot with limited budget | SMB / Startup | Simple, cost-efficient dual-mode retrieval delivers good enough quality for SMB corpora without the overhead of a full advanced RAG stack |
