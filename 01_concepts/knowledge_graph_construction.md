# Knowledge Graph Construction for RAG

> Building the graph that graph-based RAG architectures depend on — entity extraction, relation modeling, and production maintenance.

---

## What is a Knowledge Graph in RAG Context?

A **knowledge graph (KG)** is a structured representation of entities (nodes) and the relationships between them (edges). In RAG systems, it serves as an alternative or complement to a vector index: instead of retrieving by semantic similarity, the system can traverse entity relationships, follow typed edges, and resolve multi-hop questions that flat vector search cannot answer reliably.

```
Vector Index (flat)                  Knowledge Graph (structured)
──────────────────                   ─────────────────────────────
Doc1: "Apple acquired Intel..."  →   Apple ──[acquired]──► Intel
Doc2: "Tim Cook is Apple CEO"    →   Tim Cook ──[is_CEO_of]──► Apple
Doc3: "Intel makes CPUs"         →   Intel ──[manufactures]──► CPU

Query: "Who leads the company that acquired Intel?"
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
# LlamaIndex — KnowledgeGraphIndex builds the graph automatically
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs").load_data()
index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10,
    include_embeddings=True,   # dual index: vector + graph
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

Tools: **neuralcoref** (spaCy), **fastcoref**, or LLM-based coreference in the extraction prompt.

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
        MERGE (s)-[r:RELATION {type: $predicate, source: $source}]->(o)
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

## Common Mistakes

1. **Skipping entity resolution** — creates duplicate nodes (Apple, Apple Inc., AAPL) that fragment the graph and break traversal.
2. **No source provenance on edges** — impossible to update or delete edges when documents change.
3. **Over-extracting relations** — every sentence produces triples; most are noise. Add a confidence threshold and deduplicate.
4. **Flat confidence: treating all edges equally** — weight edges by extraction confidence and recency; low-confidence edges can be held back from traversal.
5. **Graph-only retrieval** — a pure graph retriever fails on semantic queries with no entity anchor; always pair with a vector index.
6. **Rebuilding the entire graph on every update** — for large corpora this is prohibitively slow; use incremental per-document refresh.

---

## Interview Q&A

**Q: What is the difference between a knowledge graph and a vector index in a RAG system?**

A vector index retrieves documents by embedding similarity — it's excellent for semantic queries but cannot follow entity relationships across documents. A knowledge graph stores entities and typed relationships as a graph, enabling multi-hop traversal ("Who leads the company that acquired Intel?") and structured lookups. In practice, production systems maintain both: vector search for semantic entry points into the graph, and graph traversal for relational reasoning once relevant entities are found.

---

**Q: How do you handle entity resolution when the same organization appears under different names?**

Use a tiered approach: (1) normalize surface forms (lowercase, strip punctuation, expand abbreviations), (2) apply a curated alias table for known synonyms (tickers, legal name variants), (3) use embedding similarity — embed all candidate entity names and cluster those above a cosine threshold (~0.92), (4) for ambiguous cases, prompt an LLM with context to decide. Assign each cluster a canonical node ID and store all surface forms as aliases on the node.

---

**Q: Why is source provenance important on graph edges?**

Without provenance, you cannot incrementally update the graph when source documents change. If you know each edge was extracted from a specific document, you can delete only that document's edges when the document is updated or removed, then re-extract from the new version. Without it, you'd need to rebuild the entire graph from scratch on every document change.

---

**Q: When would you choose a knowledge graph over pure vector retrieval?**

Choose a KG when: (1) queries require multi-hop reasoning across entities, (2) you need to retrieve by relationship type ("find all companies Apple acquired"), (3) entity deduplication is important for answer quality (multiple docs refer to the same entity differently), or (4) the domain has a well-defined ontology (medical codes, legal concepts, product hierarchies). Stick with vector retrieval when queries are primarily semantic/free-form, the corpus lacks clear entity structure, or build time is constrained.
