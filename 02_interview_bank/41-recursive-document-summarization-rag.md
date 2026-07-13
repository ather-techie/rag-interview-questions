# Recursive Document Summarization RAG

> A multi-level summarization hierarchy built offline from the original corpus — documents → section summaries → document summaries → corpus summaries — where query routing at inference time selects the right abstraction level rather than always retrieving raw chunks.

---

## Definition

**Recursive Document Summarization RAG** builds a multi-level tree of summaries over the corpus at index time. Unlike RAPTOR (#13) which uses hierarchical *clustering* (grouping semantically similar chunks bottom-up), this architecture summarizes *within* document boundaries top-down, preserving the original document structure:

```
Level 3: Corpus summary (1 per collection — "what does this knowledge base contain?")
Level 2: Document summary (1 per document — "what is this document about?")
Level 1: Section summaries (1 per section — "what does this section cover?")
Level 0: Original chunks (raw paragraphs/sentences — the actual source text)
```

At query time, a **level router** decides which tree level to retrieve from:
- Broad/orientation queries → Level 2–3 (summaries)
- Specific factual queries → Level 0–1 (raw chunks)
- Mixed queries → multi-level retrieval (retrieve from multiple levels, merge)

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Corpus
  │
  ▼
Chunk-level Summaries (Level 0: raw chunks)
  │  summarize groups of chunks
  ▼
Section-level Summaries (Level 1)
  │  summarize groups of sections
  ▼
Document-level Summaries (Level 2)
  │  summarize groups of documents
  ▼
Corpus-level Summary (Level 3)

  (4-level tree built along the document's natural
   structure — NOT clustering, unlike RAPTOR)

────────────────────── query time ──────────────────────

Query
  │
  ▼
Level Router
  (picks which tier to search based on query scope:
   overview → L2–3, section → L1, chunk → L0, multi → all)
  │
  ▼
Retriever (fetches from the chosen level)
  │
  ▼
Generator
```

### Key Components

| Component | Responsibility |
|---|---|
| **Recursive Summarizer (LLM)** | Generates faithful summaries bottom-up: chunks → sections → documents → corpus |
| **4-level Summary Tree Store** | Persists all levels of nodes (chunk/section/document/corpus) with parent/child links and embeddings |
| **Level Router** | Classifies each query's required abstraction level and selects which tier(s) to search |
| **Retriever** | Runs similarity search against the nodes at the routed level(s) |
| **Generator** | Produces the final answer from the retrieved nodes (optionally after drill-down) |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| **Summarization LLM** | Any LLM; GPT-4o-mini or Claude Haiku for cost-efficient recursive summarization |
| **Vector store** | Vector DB with level metadata (similar infra to RAPTOR, but hierarchy follows document structure rather than semantic clustering) |

---

## How It Differs from RAPTOR

| | RAPTOR (#13) | Recursive Summary RAG (#41) |
|---|---|---|
| **Structure** | Bottom-up clustering of similar chunks across documents | Top-down summarization within document boundaries |
| **Hierarchy axis** | Semantic similarity | Document structure (section → document → corpus) |
| **Cross-document nodes** | Yes — cluster nodes mix chunks from multiple documents | No — each node belongs to a single source document |
| **Best for** | Finding thematic connections across many documents | Navigating within-document structure at the right level |
| **Summary content** | Cluster topic summary | Faithful section/document summary |
| **Retrieval for detail** | Drill into cluster children | Drill into section chunks |

---

## Building the Summary Tree

```python
import anthropic
from dataclasses import dataclass, field
from typing import Optional

client = anthropic.Anthropic()


@dataclass
class SummaryNode:
    node_id: str
    level: int                    # 0 = chunk, 1 = section, 2 = document, 3 = corpus
    content: str                  # summary text (or original text for level 0)
    parent_id: Optional[str]      # ID of the parent summary node
    children_ids: list[str]       # IDs of children (sections or chunks)
    doc_id: str
    section_title: Optional[str] = None
    embedding: Optional[list[float]] = None


def summarize(texts: list[str], context: str = "") -> str:
    """Generate a faithful summary of the provided texts."""
    joined = "\n\n---\n\n".join(texts)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",   # cheap; summaries are mechanical
        max_tokens=512,
        system=(
            "Write a dense, faithful summary of the provided text. "
            "Preserve key facts, figures, and named entities. "
            "Do not add information not present in the text."
            + (f" Context: {context}" if context else "")
        ),
        messages=[{"role": "user", "content": joined}],
    )
    return resp.content[0].text


def build_summary_tree(
    documents: list[dict],   # [{id, title, sections: [{title, chunks: [str]}]}]
    embed_fn,                # function(text) -> list[float]
) -> list[SummaryNode]:
    """
    Build the full multi-level summary hierarchy.
    Returns a flat list of SummaryNode objects (store in a vector DB or dict).
    """
    all_nodes: list[SummaryNode] = []

    corpus_doc_summaries = []

    for doc in documents:
        doc_section_summaries = []

        # Level 0 + Level 1: chunk nodes and section summary nodes
        for section in doc["sections"]:
            chunk_nodes = []
            for i, chunk_text in enumerate(section["chunks"]):
                node = SummaryNode(
                    node_id=f"{doc['id']}::{section['title']}::chunk_{i}",
                    level=0,
                    content=chunk_text,
                    parent_id=f"{doc['id']}::{section['title']}::summary",
                    children_ids=[],
                    doc_id=doc["id"],
                    section_title=section["title"],
                )
                node.embedding = embed_fn(chunk_text)
                chunk_nodes.append(node)
                all_nodes.append(node)

            # Level 1: section summary
            section_summary_text = summarize(
                section["chunks"],
                context=f"Section '{section['title']}' from document '{doc['title']}'"
            )
            section_node = SummaryNode(
                node_id=f"{doc['id']}::{section['title']}::summary",
                level=1,
                content=section_summary_text,
                parent_id=f"{doc['id']}::summary",
                children_ids=[n.node_id for n in chunk_nodes],
                doc_id=doc["id"],
                section_title=section["title"],
            )
            section_node.embedding = embed_fn(section_summary_text)
            all_nodes.append(section_node)
            doc_section_summaries.append(section_summary_text)

        # Level 2: document summary
        doc_summary_text = summarize(
            doc_section_summaries,
            context=f"Document: '{doc['title']}'"
        )
        doc_node = SummaryNode(
            node_id=f"{doc['id']}::summary",
            level=2,
            content=doc_summary_text,
            parent_id="corpus::summary",
            children_ids=[
                f"{doc['id']}::{s['title']}::summary"
                for s in doc["sections"]
            ],
            doc_id=doc["id"],
        )
        doc_node.embedding = embed_fn(doc_summary_text)
        all_nodes.append(doc_node)
        corpus_doc_summaries.append(doc_summary_text)

    # Level 3: corpus-level summary
    corpus_summary_text = summarize(
        corpus_doc_summaries,
        context="Full document corpus"
    )
    corpus_node = SummaryNode(
        node_id="corpus::summary",
        level=3,
        content=corpus_summary_text,
        parent_id=None,
        children_ids=[f"{doc['id']}::summary" for doc in documents],
        doc_id="corpus",
    )
    corpus_node.embedding = embed_fn(corpus_summary_text)
    all_nodes.append(corpus_node)

    return all_nodes
```

---

## Query-Time Level Routing

```python
import json

ROUTER_PROMPT = """Classify this query's required abstraction level:

- "overview": The query asks what something is broadly, a summary of a topic, or what a document covers.
  Examples: "What is this report about?", "Summarize the key themes", "What topics are covered?"

- "section": The query targets a specific section or concept without needing the exact passage.
  Examples: "What does the methodology section say?", "What risks are mentioned in chapter 3?"

- "chunk": The query needs a specific fact, figure, or verbatim detail.
  Examples: "What was Q3 revenue?", "What is the exact definition of X?", "On what page is Y mentioned?"

- "multi": The query spans multiple levels (broad context + specific facts).
  Examples: "Explain X and give an example from the document", "What is the approach and what are the key results?"

Output JSON: {"level": "overview"|"section"|"chunk"|"multi", "reasoning": "one sentence"}
"""

LEVEL_MAP = {
    "overview": [2, 3],   # retrieve from document + corpus summaries
    "section":  [1],      # retrieve from section summaries
    "chunk":    [0],      # retrieve from raw chunks
    "multi":    [0, 1, 2] # retrieve from all levels, merge
}


def route_query(query: str) -> dict:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system=ROUTER_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    return json.loads(resp.content[0].text)


def retrieve_from_tree(
    query: str,
    nodes_by_level: dict[int, list[SummaryNode]],
    embed_fn,
    k: int = 5,
) -> list[SummaryNode]:
    """Route query to appropriate tree level(s) and retrieve."""
    import numpy as np

    route = route_query(query)
    target_levels = LEVEL_MAP[route["level"]]

    query_emb = np.array(embed_fn(query))
    results = []

    for level in target_levels:
        candidates = nodes_by_level.get(level, [])
        if not candidates:
            continue

        # Cosine similarity retrieval
        sims = [
            (node, np.dot(query_emb, np.array(node.embedding)) /
             (np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-9))
            for node in candidates
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        results.extend([node for node, _ in sims[:k]])

    return results


def drill_down(
    summary_node: SummaryNode,
    nodes_by_id: dict[str, SummaryNode],
    query: str,
    embed_fn,
    k: int = 3,
) -> list[SummaryNode]:
    """
    Given a section summary that was retrieved, fetch its most relevant child chunks.
    Enables coarse-to-fine retrieval: summary → specific passage.
    """
    import numpy as np

    child_nodes = [nodes_by_id[cid] for cid in summary_node.children_ids
                   if cid in nodes_by_id]
    if not child_nodes:
        return [summary_node]

    query_emb = np.array(embed_fn(query))
    sims = [
        (node, np.dot(query_emb, np.array(node.embedding)) /
         (np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-9))
        for node in child_nodes
    ]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in sims[:k]]
```

---

## Coarse-to-Fine Retrieval Pattern

A powerful combination: retrieve from summaries (high precision, low noise), then drill down to chunks within the matched summaries (high recall on the relevant passage):

```
Query: "What was the revenue growth rate in Q3?"

Step 1 (Level 2 retrieval):
  Route = "chunk" → retrieve from Level 0 directly

Step 2 (Level 1 summary retrieval):
  Route = "section" → find "Financial Results" section summary (score: 0.89)

Step 3 (Drill-down within matched section):
  Re-rank chunks within "Financial Results" section
  → "Q3 2023 revenue grew 18% YoY to $4.2B, driven by..."

Step 4: Generate answer from drilled-down chunks
```

```python
def coarse_to_fine_retrieve(
    query: str,
    nodes_by_level: dict[int, list[SummaryNode]],
    nodes_by_id: dict[str, SummaryNode],
    embed_fn,
    top_sections: int = 3,
    chunks_per_section: int = 3,
) -> list[SummaryNode]:
    """Always go through section summaries → drill to chunks."""
    # Step 1: retrieve top section summaries
    top_section_nodes = retrieve_from_tree(
        query, {1: nodes_by_level[1]}, embed_fn, k=top_sections
    )

    # Step 2: drill into each matched section
    chunk_results = []
    for section_node in top_section_nodes:
        chunks = drill_down(section_node, nodes_by_id, query, embed_fn, k=chunks_per_section)
        chunk_results.extend(chunks)

    return chunk_results
```

---

## Cost and Latency Profile

| Stage | Operation | Cost per Document | Notes |
|---|---|---|---|
| **Level 0** | Store raw chunks | $0 | Already done |
| **Level 1** | Summarize N sections | ~$0.0002 × N | Haiku; cheap |
| **Level 2** | Summarize document | ~$0.0002 | Haiku; 1 call |
| **Level 3** | Summarize corpus | ~$0.002 | Sonnet for quality |
| **Index build** | Embed all summary nodes | ~$0.0001 × total_nodes | text-embedding-3-small |
| **Query routing** | Classify query level | ~$0.00005 | Haiku |
| **Retrieval** | ANN on multi-level index | <10ms per level | FAISS |

For a 100-document corpus with 10 sections/document and 5 chunks/section:
- Total nodes: 100 × 5 chunks + 100 × 10 sections + 100 docs + 1 corpus = 1,601 nodes
- Index build cost: ~$0.02 in LLM calls + negligible embedding cost
- Rebuild: only changed documents need re-summarization (not full corpus)

---

## Summary Tree vs. RAPTOR at a Glance

```
Recursive Summary Tree                    RAPTOR
─────────────────────                    ──────
doc A                                    cluster_1
 ├── section A.1                          ├── chunk_A1 (doc A)
 │    ├── chunk_0                         ├── chunk_B3 (doc B)
 │    └── chunk_1                         └── chunk_C2 (doc C)
 └── section A.2                         cluster_2
      └── chunk_2                          ├── chunk_A5 (doc A)
                                           └── chunk_D1 (doc D)

Navigation: drill into a document         Navigation: drill into a topic
Best for: "What does doc A say about X?" Best for: "What do all docs say about topic Y?"
```

---

## Key Takeaways

1. **Level routing is the key innovation** — the system selects the right abstraction level per query rather than always retrieving raw chunks.
2. **RAPTOR clusters cross-document; Recursive Summary preserves document structure** — choose based on whether users navigate by topic (RAPTOR) or by document (this).
3. **Coarse-to-fine retrieval combines precision and depth** — retrieve section summaries for precision, drill to chunks for exactness.
4. **Build cost is modest** — Haiku summaries + cached embeddings; update only changed documents.
5. **Null-retrieval defense** — if the query hits Level 3 (corpus summary), a fallback to Level 2 retrieval prevents the system from returning a single mega-summary with no specifics.

---

## Interview Q&A

**Q: How is Recursive Document Summarization RAG different from RAPTOR?**

RAPTOR builds clusters bottom-up across document boundaries — similar chunks from different documents are grouped into a cluster node, and the cluster is summarized. This is powerful for thematic queries ("what do all these papers say about attention?") but loses document identity — a RAPTOR node might mix content from five different papers. Recursive Summary RAG summarizes top-down within document boundaries: each node belongs to a single source document. This preserves provenance — "which document says X?" is answerable. Use RAPTOR when users ask cross-document thematic questions; use Recursive Summary RAG when users navigate documents individually (e.g., a legal contract or annual report) and need to understand the document at multiple granularities.

---

**Q: How do you decide which tree level to retrieve from at query time?**

A small classifier (Haiku-class LLM, ~$0.00005/call) categorizes the query into: "overview" (broad, summary-seeking), "section" (specific section-level), "chunk" (exact fact/figure), or "multi" (needs both summary context and specific detail). The level mapping is deterministic: overview → Level 2–3, section → Level 1, chunk → Level 0, multi → all levels. An alternative is to skip routing and always use coarse-to-fine: retrieve from Level 1 (sections), then drill into matched sections' Level 0 chunks. This is slightly more expensive per query but eliminates the routing error mode (wrong level → poor recall).

---

**Q: What's the incremental update strategy when a document changes?**

Only the modified document's subtree needs to be rebuilt: (1) Re-chunk the changed sections, (2) re-embed the new chunks (Level 0), (3) re-summarize affected sections (Level 1), (4) re-summarize the document (Level 2), (5) optionally update the corpus summary (Level 3) if the document's contribution changed significantly. Level 3 re-summarization is the most expensive (Sonnet-class model) and can be deferred to a scheduled batch job rather than triggered on every document change. Store `last_modified` timestamps per node to efficiently detect which nodes are stale — nodes whose document has not changed since their `last_modified` can be skipped entirely.
