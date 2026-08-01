# Knowledge Graph Construction for RAG

> Building the graph that graph-based RAG architectures depend on — entity extraction, relation modeling, and production maintenance.

---

## What is a Knowledge Graph in RAG Context?

A **knowledge graph (KG)** is a structured representation of entities (nodes) and the relationships between them (edges). In RAG systems, it serves as an alternative or complement to a vector index: instead of retrieving by semantic similarity, the system can traverse entity relationships, follow typed edges, and resolve multi-hop questions that flat vector search cannot answer reliably.

```
Vector Index (flat)                  Knowledge Graph (structured)
──────────────────                   ─────────────────────────────
Doc1: "Apple acquired Intel's modem division..." → Apple ──[acquired]──► Intel modem division
Doc2: "Tim Cook is Apple CEO"    →   Tim Cook ──[is_CEO_of]──► Apple
Doc3: "Intel makes CPUs"         →   Intel ──[manufactures]──► CPU

Query: "Who leads the company that acquired Intel's modem division?"
  Vector: may miss the chain          KG: traverse Apple→Tim Cook directly
```

**When a KG adds value over a flat vector index:**
- Multi-hop questions requiring entity chaining
- Queries about specific relationships ("Who reports to whom?", "Which products have this vulnerability?")
- Deduplication across documents mentioning the same entity with different phrasing
- Structured exploration (all relationships around an entity)

---

## The KG Build Pipeline

```
Raw Documents
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Entity Extraction                                     │
│    Identify: people, organizations, locations,          │
│    products, concepts, events, dates                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Relation Extraction                                   │
│    Find: (subject, predicate, object) triples           │
│    e.g. (Apple, acquired, Intel)                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Entity Resolution / Coreference                      │
│    "Apple", "Apple Inc.", "AAPL" → same node            │
│    "he" / "the CEO" → Tim Cook                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Graph Storage                                        │
│    Nodes with properties, edges with types/weights      │
│    Backends: Neo4j, NetworkX, ArangoDB, TigerGraph      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Maintenance                                          │
│    Add/update/delete nodes and edges as source          │
│    documents change; manage version history             │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Entity Extraction

### Traditional NER

Statistical and rule-based NER using libraries like **spaCy** or **Stanza** — fast, deterministic, but limited to a fixed label set (PERSON, ORG, GPE, DATE, etc.).

```python
import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Tim Cook, CEO of Apple Inc., announced the acquisition of Intel's modem division.")

for ent in doc.ents:
    print(f"{ent.text!r:30} → {ent.label_}")
# 'Tim Cook'                     → PERSON
# 'Apple Inc.'                   → ORG
# "Intel's modem division"       → ORG (partial)
```

**Limitations:** Misses domain-specific entities (medical concepts, legal terms, product IDs) and cannot extract relationship triplets.

### LLM-Based Extraction

LLMs can extract arbitrary entity types and relations in a single pass — no fine-tuning required, adapts to domain vocabulary.

```python
from anthropic import Anthropic

client = Anthropic()

EXTRACTION_PROMPT = """Extract all entities and relationships from the text below.

Output a JSON object with:
- "entities": list of {{"id": str, "type": str, "name": str, "attributes": dict}}
- "relations": list of {{"subject_id": str, "predicate": str, "object_id": str}}

Use consistent IDs for the same entity across mentions.

Text:
{text}"""

def extract_kg_triples(text: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}]
    )
    import json
    return json.loads(response.content[0].text)
```

**Output example:**
```json
{
  "entities": [
    {"id": "e1", "type": "PERSON",  "name": "Tim Cook",   "attributes": {"role": "CEO"}},
    {"id": "e2", "type": "COMPANY", "name": "Apple Inc.", "attributes": {"ticker": "AAPL"}},
    {"id": "e3", "type": "COMPANY", "name": "Intel",      "attributes": {"ticker": "INTC"}}
  ],
  "relations": [
    {"subject_id": "e1", "predicate": "is_CEO_of",  "object_id": "e2"},
    {"subject_id": "e2", "predicate": "acquired",   "object_id": "e3"}
  ]
}
```

### LlamaIndex and LangChain Extractors

```python
# LlamaIndex — PropertyGraphIndex builds the graph automatically
# (KnowledgeGraphIndex is deprecated; PropertyGraphIndex is the current API)
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

documents = SimpleDirectoryReader("./docs").load_data()
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[SimpleLLMPathExtractor(max_paths_per_chunk=10)],
    embed_kg_nodes=True,   # dual index: vector + graph
)

# LangChain — LLMGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-5")
transformer = LLMGraphTransformer(llm=llm)
graph_docs = transformer.convert_to_graph_documents(docs)
```

---

## Step 2: Relation Extraction

### Closed-Set (Fixed Predicate Schema)

Define a fixed ontology of predicates. The model classifies which predicate (if any) holds between two detected entities.

```
Schema: {works_for, is_subsidiary_of, acquired, located_in, founded_by, competes_with}

Input: ("Apple", "Tim Cook") → works_for (with direction: Tim Cook works_for Apple)
```

**Pro:** Consistent, queryable schema.  
**Con:** Misses novel relationship types outside the schema.

### Open-IE (Schema-Free)

Extract arbitrary natural-language predicates without a fixed schema. More expressive but harder to query uniformly.

```
"Tim Cook leads Apple's operations" → (Tim Cook, leads operations of, Apple)
```

OpenIE tools: **Stanford OpenIE**, **REBEL** (relation extraction with BART).

### Hybrid: Schema + LLM

Define a typed schema for important structured predicates; let the LLM extract free-text relations for everything else.

```python
SCHEMA = {
    "structured": ["acquired", "is_CEO_of", "is_subsidiary_of", "headquartered_in"],
    "freetext": True   # capture remaining relations as raw predicates
}
```

---

## Step 3: Entity Resolution

**The core problem:** The same real-world entity appears under different surface forms across documents.

```
"Apple Inc."  /  "Apple"  /  "AAPL"  /  "the Cupertino giant"
→ all should map to a single canonical node: entity_id = "apple_inc"
```

### Resolution Approaches

| Method | How | When to Use |
|--------|-----|-------------|
| **Exact string match** | Normalize (lowercase, strip punctuation) and match | Controlled vocabularies, IDs |
| **Alias lookup table** | Pre-built mapping: "AAPL" → "apple_inc" | Tickers, codes, known abbreviations |
| **Embedding similarity** | Embed entity surface forms; cosine similarity above threshold → merge | General entities without a lookup table |
| **LLM coreference** | Prompt LLM: "Are these the same entity?" | Ambiguous cases needing context |
| **Wikidata/DBpedia linking** | Link extracted entities to canonical Wikidata QIDs | Well-known entities in general-domain corpora |

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def resolve_entities(candidates: list[str], threshold: float = 0.92) -> dict[str, str]:
    embeddings = model.encode(candidates, convert_to_tensor=True)
    canonical = {}
    cluster_id = 0
    assigned = {}

    for i, name in enumerate(candidates):
        if name in assigned:
            canonical[name] = assigned[name]
            continue
        # Compare with all previous unassigned
        for j in range(i):
            score = util.cos_sim(embeddings[i], embeddings[j]).item()
            if score >= threshold:
                canonical[name] = canonical[candidates[j]]
                assigned[name] = canonical[name]
                break
        else:
            label = f"entity_{cluster_id}"
            canonical[name] = label
            assigned[name] = label
            cluster_id += 1

    return canonical
```

### Coreference Resolution (Within Documents)

Resolve pronouns and noun phrases to their antecedent entities within a document before extraction.

```
"Apple reported earnings. The company beat expectations. Its CEO commented..."
 ↑ entity       "the company" → Apple   "Its" → Apple   "CEO" → Tim Cook
```

Tools: **fastcoref** (fast, maintained, spaCy 3.x-compatible; supersedes the now-unmaintained neuralcoref), or LLM-based coreference in the extraction prompt.

---

## Step 4: Graph Storage

### Property Graph Model

Nodes and edges both carry typed properties — the most common model for RAG knowledge graphs.

```python
# Neo4j example via the Python driver
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def add_triple(tx, subject, predicate, obj, source_doc_id):
    tx.run(
        """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $obj})
        CALL apoc.merge.relationship(s, $predicate, {}, {source: $source}, o, {})
        YIELD rel
        RETURN rel
        """,
        subject=subject, obj=obj, predicate=predicate, source=source_doc_id
    )

with driver.session() as session:
    session.execute_write(add_triple, "Tim Cook", "is_CEO_of", "Apple Inc.", "doc_001")
```

### Lightweight In-Memory (NetworkX)

```python
import networkx as nx

G = nx.MultiDiGraph()

# Add nodes
G.add_node("Apple Inc.", type="COMPANY", ticker="AAPL")
G.add_node("Tim Cook",   type="PERSON",  role="CEO")

# Add typed edge
G.add_edge("Tim Cook", "Apple Inc.", relation="is_CEO_of", confidence=0.98)

# Multi-hop query
def find_entity_via_hops(G, start, relation_chain):
    current = {start}
    for rel in relation_chain:
        next_nodes = set()
        for node in current:
            for _, neighbor, data in G.out_edges(node, data=True):
                if data.get("relation") == rel:
                    next_nodes.add(neighbor)
        current = next_nodes
    return current

# "Who is CEO of companies Apple acquired?"
# apple_acquisitions = find_entity_via_hops(G, "Apple Inc.", ["acquired"])
# ceos = find_entity_via_hops(G, apple_acquisitions, ["has_CEO"])
```

### Dual Index (Graph + Vector)

Most production graph RAG systems maintain both a graph index (for structural traversal) and a vector index (for semantic similarity search into the graph).

```
Query: "Tell me about Apple's supply chain risks"
         │
         ├──► Vector search → find relevant entity cluster (Apple, suppliers)
         │
         └──► Graph traversal → expand: Apple -[uses]-> Suppliers -[located_in]-> Countries
```

---

## Step 5: Graph Maintenance

### Incremental Updates

When source documents change, the graph must be updated without a full rebuild.

```
Document updated → re-extract triples from updated passages
                 → compare new triples vs. stored triples for that document
                 → delete removed triples, add new triples
                 → re-run entity resolution on new entities
```

```python
def update_document_triples(doc_id: str, new_text: str, graph):
    # Delete all triples from this source document
    graph.delete_triples_by_source(doc_id)
    
    # Re-extract
    new_triples = extract_kg_triples(new_text)
    
    # Re-insert with resolution
    for triple in new_triples["relations"]:
        resolved_subject = resolve_entity(triple["subject_id"])
        resolved_object  = resolve_entity(triple["object_id"])
        graph.add_triple(resolved_subject, triple["predicate"], resolved_object, doc_id)
```

### Versioning and Provenance

Track which source document created each edge. Essential for:
- Deletion on document removal
- Conflict resolution (two docs disagree on a fact)
- Freshness scoring (prefer edges from recently updated documents)

```python
# Edge with provenance
{
  "subject": "Apple Inc.",
  "predicate": "acquired",
  "object": "Intel modem division",
  "source_doc": "doc_2024_q3_earnings",
  "extracted_at": "2024-09-15T10:00:00Z",
  "confidence": 0.91
}
```

---

## Comparison: LLM-Based vs. Traditional KG Construction

| Dimension | Traditional (NER + OpenIE) | LLM-Based |
|-----------|---------------------------|-----------|
| Setup effort | Medium (model selection, tuning) | Low (prompt engineering) |
| Speed | Fast (1K–10K docs/min) | Slow (1–10 docs/min at scale) |
| Domain adaptation | Requires fine-tuning | Prompt-level control |
| Relation types | Fixed schema or noisy open | Flexible, context-aware |
| Entity resolution | Separate step required | Can be in-prompt |
| Cost | Low (CPU/GPU local) | High (LLM API costs) |
| Consistency | High (deterministic) | Variable (LLM non-determinism) |

**Rule of thumb:** For large corpora (>100K documents), use traditional extraction for speed and LLM post-processing for quality refinement on ambiguous cases.

---

## Microsoft GraphRAG: Community-Based Retrieval for Global Questions

Standard entity-relation KG-RAG (Steps 1–4 above) answers questions anchored to specific entities well — "Who is Intel's CEO?", "What did Apple acquire?" — because the answer lives on a short traversal path from a named entry point. It breaks down on **holistic or global questions that require synthesizing information spread across the entire corpus**: "What are the main themes in this dataset?", "Summarize the key risks discussed across all these reports." No single entity or short relation chain answers a question like that, and a flat vector index does no better — top-k chunk retrieval simply cannot see "the whole picture" from a handful of similar chunks.

**Microsoft GraphRAG** (Edge et al., *"From Local to Global: A Graph RAG Approach to Query-Focused Summarization,"* Microsoft Research, April 2024, arXiv:2404.16130; open-sourced as the `graphrag` Python package) targets exactly this gap by adding a community-detection and pre-summarization layer on top of the same entity/relation graph described in Steps 1–3.

### Community Detection and Pre-Generated Summaries

After entity and relation extraction, GraphRAG runs the **Leiden algorithm** — a modularity-based community detection method that improves on the older Louvain algorithm by guaranteeing well-connected communities — over the entity graph. Leiden is applied recursively, producing a **hierarchy of communities**: broad, coarse-grained clusters at the root level (e.g., "supply chain risk") that recursively partition into progressively narrower sub-communities down to fine-grained leaf clusters (e.g., "Intel modem division divestiture").

For every community at every level of the hierarchy, an LLM generates a **community summary (report)** describing the entities, relationships, and salient claims within that cluster — entirely at *index* time, before any query arrives. This pre-generation is the expensive part of GraphRAG's indexing: on top of the LLM calls already needed for extraction, every community at every hierarchy level needs its own summarization pass.

```
Entity Graph (from Steps 1-3)
     │
     ▼  Leiden clustering (hierarchical, recursive)
┌─────────────────────────────────────────────────────────┐
│ Level 0 (root):     [   Community A   ][   Community B   ]│
│ Level 1 (mid):      [ C1 ][ C2 ]      [ C3 ][ C4 ][ C5 ]  │
│ Level 2 (leaf):     [c1a][c1b] ...                        │
└─────────────────────────────────────────────────────────┘
     │
     ▼  LLM summarizes every community, at every level — once, at index time
Pre-generated community summaries (stored; not regenerated per query)
```

### Local Search vs. Global Search

GraphRAG exposes the graph through two distinct query modes:

| | **Local Search** | **Global Search** |
|---|---|---|
| Best for | Specific, entity-anchored questions | Broad, corpus-wide / thematic questions |
| Mechanism | Vector search finds entry-point entities → fan out to their graph neighborhood (relationships, covariates, linked text chunks) | Map-reduce over *pre-generated* community summaries |
| Similar to | Standard KG-RAG traversal (Step 4 above) | Nothing in flat vector RAG — this is GraphRAG's distinctive contribution |
| Query-time cost | Low–moderate | Higher (many community summaries read per query — still cheaper than re-reading raw documents) |
| Example question | "What products does Intel's modem division make?" | "What are the main themes across this document set?" |

Global search's map-reduce runs in two stages: in the **map** step, the LLM independently reads each relevant community summary (in parallel batches) and produces a partial answer plus a self-rated importance score; in the **reduce** step, those partial answers are ranked and synthesized into one final response. Because the summaries were already generated at index time, global search never needs to stuff raw source documents into a long-context call at query time — it reasons over compact, pre-digested summaries instead.

A later addition, **DRIFT search** (Dynamic Reasoning and Inference with Flexible Traversal), hybridizes the two modes: it starts with a broad, community-level pass to establish context, generates follow-up questions from that pass, then runs local search to ground the answer in specific entities — re-ranking all results together to produce the final response.

### GraphRAG vs. Standard Entity-Relation KG-RAG

| Dimension | Standard KG-RAG (Steps 1–4 above) | Microsoft GraphRAG |
|-----------|-----------------------------------|---------------------|
| Query shape it targets | Specific facts, relationships, multi-hop chains | Both entity-specific (local) *and* corpus-wide thematic (global) |
| Extra build step | None beyond extraction + resolution | + Hierarchical Leiden clustering + LLM summary per community, per level |
| Indexing cost | Moderate (extraction + resolution) | High (extraction + resolution + summarization LLM calls across every community and level) |
| Query-time cost | Low (graph traversal) | Local: low. Global: higher (map-reduce over many summaries) |
| Answers "what are the themes here?" | Poorly — no aggregation mechanism | Well — this is the design target |

**Rule of thumb:** Reach for plain entity-relation traversal when questions are anchored to named entities or specific relationships and the corpus doesn't need thematic roll-ups — it's cheaper to build and query. Reach for GraphRAG's community layer when you expect genuinely global, "summarize / what-are-the-themes" questions over a large corpus, and the extra indexing cost (LLM summarization at every level of the community hierarchy) is worth paying up front to make those queries cheap and accurate at query time.

---

## Common Mistakes

1. **Skipping entity resolution** — creates duplicate nodes (Apple, Apple Inc., AAPL) that fragment the graph and break traversal.
2. **No source provenance on edges** — impossible to update or delete edges when documents change.
3. **Over-extracting relations** — every sentence produces triples; most are noise. Add a confidence threshold and deduplicate.
4. **Flat confidence: treating all edges equally** — weight edges by extraction confidence and recency; low-confidence edges can be held back from traversal.
5. **Graph-only retrieval** — a pure graph retriever fails on semantic queries with no entity anchor; always pair with a vector index.
6. **Rebuilding the entire graph on every update** — for large corpora this is prohibitively slow; use incremental per-document refresh.

---

## Interview Q&A

**Q: What is the difference between a knowledge graph and a vector index in a RAG system?** `[Basic]`

A vector index retrieves documents by embedding similarity — it's excellent for semantic queries but cannot follow entity relationships across documents. A knowledge graph stores entities and typed relationships as a graph, enabling multi-hop traversal ("Who leads the company that acquired Intel?") and structured lookups. In practice, production systems maintain both: vector search for semantic entry points into the graph, and graph traversal for relational reasoning once relevant entities are found.

---

**Q: How do you handle entity resolution when the same organization appears under different names?** `[Intermediate]`

Use a tiered approach: (1) normalize surface forms (lowercase, strip punctuation, expand abbreviations), (2) apply a curated alias table for known synonyms (tickers, legal name variants), (3) use embedding similarity — embed all candidate entity names and cluster those above a cosine threshold (~0.92), (4) for ambiguous cases, prompt an LLM with context to decide. Assign each cluster a canonical node ID and store all surface forms as aliases on the node.

---

**Q: Why is source provenance important on graph edges?** `[Intermediate]`

Without provenance, you cannot incrementally update the graph when source documents change. If you know each edge was extracted from a specific document, you can delete only that document's edges when the document is updated or removed, then re-extract from the new version. Without it, you'd need to rebuild the entire graph from scratch on every document change.

---

**Q: What kind of question does Microsoft GraphRAG solve that standard entity-relation KG-RAG and vector search both struggle with?** `[Intermediate]`

Global, corpus-wide questions like "what are the main themes in this dataset?" Vector search only returns top-k similar chunks, which never aggregates across the whole corpus. Standard entity-relation KG-RAG can traverse from a specific entity but has no mechanism for summarizing across many unrelated entities and clusters at once. GraphRAG addresses this by clustering the entity graph into hierarchical communities with the Leiden algorithm and pre-generating an LLM summary for each community at index time; a "global search" query then map-reduces over those summaries instead of raw documents.

---

**Q: Walk through the difference between GraphRAG's local search and global search modes.** `[Advanced]`

Local search behaves like standard KG-RAG: it uses vector similarity to find entry-point entities relevant to the query, then expands into their graph neighborhood — connected entities, relationships, covariates, and linked text chunks — to assemble context for a single LLM call. Global search instead runs a map-reduce over the pre-generated community summaries: in the map step, the LLM independently generates a partial answer and importance rating from each relevant community summary; in the reduce step, those partial answers are ranked and merged into one final response. Local search is cheap and precise for entity-anchored questions; global search costs more per query but is the only mode that can answer broad, thematic questions spanning the whole corpus, since the expensive summarization work was already paid for once at index time rather than repeated per query.

---

**Q: When would you choose a knowledge graph over pure vector retrieval?** `[Advanced]`

Choose a KG when: (1) queries require multi-hop reasoning across entities, (2) you need to retrieve by relationship type ("find all companies Apple acquired"), (3) entity deduplication is important for answer quality (multiple docs refer to the same entity differently), or (4) the domain has a well-defined ontology (medical codes, legal concepts, product hierarchies). Stick with vector retrieval when queries are primarily semantic/free-form, the corpus lacks clear entity structure, or build time is constrained.

---

## Related

- [Graph RAG](../02_interview_bank/05-graph-rag.md) — architecture patterns for combining knowledge graphs with vector retrieval
- [LightRAG](../02_interview_bank/15-lightrag.md) — dual-level graph indexing for efficient graph-based retrieval
- [HippoRAG](../02_interview_bank/20-hipporag.md) — neurobiologically-inspired graph memory for long-term retrieval
- [Agentic Orchestration](./agentic_orchestration.md) — coordinating multi-hop graph traversal within agent workflows
