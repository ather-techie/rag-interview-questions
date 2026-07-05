# RAG Taxonomy: Classification and Architecture Mapping

> All 41 RAG architectures mapped by mechanism, data source, retrieval control, and production fit.

---

## Classification Axes

Every RAG system can be classified along four independent dimensions. Understanding these axes helps you systematically answer "How does X RAG work?" and "When do you choose Y over Z?"

**1. Retrieval Control**
- **Static**: Same retrieval strategy for all queries (naive, advanced RAG)
- **Dynamic**: Strategy adapts based on query characteristics (adaptive RAG)
- **Agent-Driven**: An LLM orchestrates retrieval decisions (agentic RAG)

**2. Data Modality**
- **Text-Only**: Documents are plain text (naive through modular RAG)
- **Structured**: Tables, schemas, SQL-queryable (structured RAG)
- **Graph**: Knowledge graph as retrieval source (graph RAG)
- **Multimodal**: Images, tables, text combined (multimodal RAG)

**3. Feedback Loop**
- **None**: No loop; retrieve once, generate once (naive, advanced, modular RAG)
- **Self-Critique**: LLM validates its own output, re-retrieves if needed (corrective, self-RAG)
- **External Signal**: User feedback or evaluation signal drives re-retrieval (self-RAG with fine-tuning)

**4. Retrieval Scope**
- **Single-Hop**: One retrieval call per query (naive, advanced RAG)
- **Multi-Hop**: Multiple retrieval calls chained together (agentic, adaptive RAG)
- **Iterative**: Retrieval continues until a stopping criterion (FLARE, self-RAG)

Each axis maps to a different set of failure modes and trade-offs. Mastering these axes makes every RAG interview question answerable.

---

## The Full Taxonomy: All 41 Architectures

| Architecture | Retrieval Control | Data Modality | Feedback Loop | Scope | Fine-tune Required? | Latency Class | Production Maturity |
|---|---|---|---|---|---|---|---|
| Naive RAG | Static | Text-only | None | Single-hop | No | Fast (<200ms) | Stable (since 2020) |
| Advanced RAG | Static | Text-only | None | Single-hop | No | Fast (<200ms) | Stable (mature) |
| Modular RAG | Static | Text-only | None | Single-hop | No | Fast (<200ms) | Stable (mature) |
| Adaptive RAG | Dynamic | Text-only | None | Single-hop or Multi-hop | No | Fast (<200ms) | Emerging (2023+) |
| Agentic RAG | Agent-driven | Text-only | Self-critique | Multi-hop | No | Slow (>1s) | Emerging (2023+) |
| Corrective RAG | Dynamic | Text-only | Self-critique | Single-hop | No | Medium (200ms–1s) | Emerging (2023+) |
| Self-RAG | Agent-driven | Text-only | External signal | Iterative | **Yes** | Medium (200ms–1s) | Cutting-edge (2023+) |
| Speculative RAG | Dynamic | Text-only | None | Single-hop | No | Fast (<200ms) | Research-frontier |
| Graph RAG | Agent-driven | Graph | Self-critique | Multi-hop | No | Slow (>1s) | Emerging (2024+) |
| Structured RAG | Dynamic | Structured | None | Single-hop | No | Medium (200ms–1s) | Stable (growing) |
| Multimodal RAG | Static | Multimodal | None | Single-hop | No | Medium (200ms–1s) | Emerging (2023+) |
| Long-Context RAG | Static | Text-only | None | Single-hop | No | High (>1s) | Stable (2024+) |
| RAPTOR | Static | Text-only | None | Multi-hop (tree) | No | Fast (query-time) | Emerging (2024+) |
| Contextual RAG | Static | Text-only | None | Single-hop | No | Fast (<200ms) | Emerging (2024+) |
| LightRAG | Agent-driven | Graph | None | Multi-hop | No | Medium (200ms–1s) | Emerging (2024+) |
| RAFT | Static | Text-only | None | Single-hop | **Yes** | Fast (<200ms) | Emerging (2024+) |
| CAG | Static | Text-only | None | Single-hop | No | Fast (<200ms)* | Emerging (2024+) |
| RAG-Fusion | Static | Text-only | None | Single-hop (N-query) | No | Medium (200ms–1s) | Emerging (2024+) |
| Iterative / Multi-hop RAG | Dynamic | Text-only | Self-critique (optional) | Iterative / Multi-hop | No | Slow (>1s) | Emerging (2023+) |
| HippoRAG | Agent-driven | Graph | None | Multi-hop (single pass) | No | Fast (query-time)* | Research-frontier (2024+) |
| Memory / Conversational RAG | Dynamic | Text-only | None | Single- or Multi-hop | No | Medium (200ms–1s) | Stable (growing) |
| HyDE | Static | Text-only | None | Single-hop | No | Medium (200ms–1s) | Stable (2022+) |
| FLARE | Agent-driven | Text-only | Self-critique | Iterative | No | Slow (>1s) | Emerging (2023+) |
| KAG | Dynamic | Graph + Text | Self-critique (logical) | Multi-hop | No | Medium (200ms–1s) | Emerging (2024+) |
| GraphReader | Agent-driven | Graph (notes) | Self-critique | Iterative / Multi-hop | No | Slow (>1s) | Emerging (2024+) |
| GNN-RAG | Static (learned) | Graph | None | Multi-hop | **Yes** (GNN) | Fast (query-time) | Emerging (2024+) |
| REALM †| Static (learned) | Text-only | None | Single-hop | **Yes** | Fast (<200ms) | Foundational (2020) |
| RETRO †| Static (frozen) | Text-only | None | Per-chunk | **Yes** | Medium (200ms–1s) | Foundational (2021) |
| Atlas †| Static (learned) | Text-only | None | Single-hop | **Yes** | Medium (200ms–1s) | Foundational (2022) |
| Fusion-in-Decoder †| Static | Text-only | None | Single-hop (many passages) | **Yes** (reader) | Medium (200ms–1s) | Foundational (2020) |
| ColRAG / ColBERT | Static | Text-only | None | Single-hop | No | Fast (<200ms) | Stable (2020+) |
| Agentic Web RAG | Agent-driven | Text-only (live web) | Self-critique (optional) | Multi-hop | No | Slow (>1s) | Stable (2023+) |
| Few-Shot Example RAG | Static | Text-only | None | Single-hop | No | Fast (<200ms) | Stable (growing) |
| Verifiable / Citation RAG | Static | Text-only | Self-critique (attribution) | Single-hop | No | Medium (200ms–1s) | Emerging (2023+) |
| Privacy-Preserving RAG | Static | Text-only | None | Single-hop | No | Medium (200ms–1s) | Emerging (2024+) |
| Streaming / Real-Time RAG | Dynamic | Text-only | None | Single-hop | No | Fast (<200ms) | Stable (growing) |
| Table-Aware RAG | Static | Structured (tables in docs) | None | Single-hop | No | Medium (200ms–1s) | Emerging (2023+) |
| Tree of Thought RAG | Agent-driven | Text-only | Self-critique | Multi-hop (tree) | No | Slow (>1s) | Research-frontier (2024+) |
| DPR (Dense Passage Retrieval) | Static (learned) | Text-only | None | Single-hop | **Yes** | Fast (<200ms) | Foundational (2020) |
| WebGPT / Tool-Augmented LM | Agent-driven | Text-only (live web) | Self-critique | Multi-hop | **Yes** (RLHF) | Slow (>1s) | Foundational (2021) |
| SURGE (Schema-Grounded RAG) | Static | Text-only | Self-critique (NLI) | Single-hop | No | Medium (200ms–1s) | Emerging (2024+) |
| Recursive Document Summarization RAG | Static | Text-only | None | Multi-hop (tree) | No | Fast (<200ms) | Emerging (2024+) |

\* Query-time latency is low; HippoRAG and CAG pay a large up-front (index-build / KV-cache) cost instead.
‡ DPR is also a training-time architecture but is listed separately as the foundational bi-encoder retrieval model rather than a full RAG pipeline.
† **Training-time / parametric** architectures: retrieval is integrated into pre-training or the model architecture (and the retriever/reader is trained), rather than bolted on at inference. They form a distinct branch from the inference-time architectures above.

---

## Taxonomy Tree Diagram

Here's how the core architectures relate (the tree shows the original 12 base types; the
17 newer architectures in the table above are specializations or compositions of these — e.g.
HippoRAG, LightRAG, KAG, GraphReader and GNN-RAG extend the Graph branch; FLARE and Iterative
RAG extend the iterative-scope branch; HyDE is a query-transformation layer over any static
retriever; and REALM/RETRO/Atlas/FiD form a separate **training-time / parametric** branch where
retrieval is baked into pre-training or architecture rather than added at inference). Every path
from root to leaf is a valid RAG system.

```
                          RAG
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
    Text-Only         Specialized        Context-Aware
        │           (Data Modality)      (Scope)
        │                 │                  │
   ┌────┼────┐        ┌───┼───┐         ┌───┴────┐
   │    │    │        │   │   │         │        │
Static Dyn  Agent   Graph Struct Multi  Long-  Spec
        │    │        │   │   │      Context  ulative
   ┌────┘    │    ┌───┘   │   │         │        │
   │         │    │       │   │         │        │
Naive    Adaptive Agentic GraphRAG Struct Multi   │
RAG      RAG      RAG     RAG     RAG    RAG   Long-Context
   │         │             │       │          │
Advanced  Modular      (multi-hop)         (single-hop,
RAG       RAG                            context
   │         │                           window)
Corrective  (self-critique)
RAG         Self-RAG
   │          │
(validation) (fine-tune)
```

---

## How New RAG Architectures Get Created

The field produces new architectures by combining these three generative patterns:

**Pattern 1: Add a New Retrieval Source**
- Example: Graph RAG adds knowledge graph to the retrieval set alongside text
- Example: Structured RAG routes queries to SQL tables instead of (or alongside) semantic search
- Result: Same orchestration logic as Advanced RAG, but the retrieve step branches based on query type
- Pseudocode:
  ```python
  def retrieve(query: str, use_graph: bool = True):
      if use_graph and is_entity_query(query):
          return retrieve_from_graph(query)
      else:
          return retrieve_from_text_index(query)
  ```

**Pattern 2: Add a Feedback Loop**
- Example: Corrective RAG adds a validation step: "Is retrieved context sufficient?"
- Example: Self-RAG adds a confidence score and iterates if uncertain
- Result: Single retrieval call becomes a conditional chain
- Pseudocode:
  ```python
  def retrieve_with_feedback(query: str):
      context = retrieve(query)
      if not is_sufficient(context):  # feedback
          context = retrieve_with_different_strategy(query)
      return context
  ```

**Pattern 3: Add Orchestration / Control Logic**
- Example: Agentic RAG adds an LLM agent that decides retrieve → generate → decide to re-retrieve
- Example: Adaptive RAG adds a classifier that picks which retrieval strategy to use
- Result: Retrieval decisions are no longer pre-determined; they're computed per query
- Pseudocode:
  ```python
  def orchestrate(query: str):
      strategy = pick_strategy(query)  # "dense", "hybrid", "re-rank", etc.
      context = retrieve(query, strategy)
      if agent_wants_more_context():
          context += retrieve(query, different_strategy)
      return context
  ```

Most production systems combine two or more patterns. For example, a system that uses Adaptive RAG (pattern 3) to pick a strategy, then adds Self-RAG's feedback loop (pattern 2) is a valid hybrid.

---

## Hybrid Systems in Practice

Most production RAG systems are not pure. They combine two or more of the 29 architectures.

| Common Combination | Why It's Done | Trade-off |
|---|---|---|
| Adaptive + Agentic | Route simple queries through fast adaptive RAG; use agent for complex multi-hop queries | Added complexity; must maintain two retrieval paths |
| Graph + Advanced | Use semantic search on text, entity linking on graphs; merge results with RRF | Embedding model for text, entity indexing for graph; slower but higher recall |
| Modular + Corrective | Plug in a validation step after any modular retrieval choice | Added latency (one extra validation call per query) |
| Self-RAG + Structured | Use Self-RAG for text, route structured queries to SQL directly | Requires query classifier; different evaluation metrics per path |
| Long-Context + Adaptive | For queries that fit in context window, use long-context; for others, use adaptive retrieval | Mixed latency; must document context window limits |

---

## Taxonomy Anti-patterns

Five common mistakes when classifying or choosing a RAG type:

**1. Over-Engineering (Choosing Agentic when Naive Would Do)**
- The mistake: "Our system is complex, so we need an agent"
- The reality: Naive RAG succeeds on 80% of retrieval benchmarks when done well (good chunks, good embeddings)
- The signal you've over-engineered: Latency increased, debuggability decreased, cost exploded, accuracy barely improved
- The fix: Measure NDCG@5 on a probe set before adding an agent. Agentic RAG is a forcing function, not a default

**2. Under-Specifying (Calling Everything "Advanced RAG")**
- The mistake: "We use advanced RAG with reranking and query expansion"
- The reality: Query expansion + reranking is just Advanced RAG. Calling it "advanced" is marketing, not architecture
- The signal: You can't distinguish your system from others in the 02_interview_bank taxonomy
- The fix: Classify along the four axes. Use the taxonomy table, not marketing words

**3. Modality Mismatch (Using Text Embeddings on Structured Data)**
- The mistake: "We embed SQL tables into vectors and search them"
- The reality: Text embeddings trained on prose poorly understand tabular semantics. Use Structured RAG instead
- The signal: Low retrieval recall on numeric comparisons and exact matches
- The fix: For structured data, use query-to-SQL + post-generation filtering (Structured RAG pattern)

**4. Scope Creep Without Feedback (Multi-Hop Without Validation)**
- The mistake: "We retrieve multiple times" without checking if the final answer is correct
- The reality: Multi-hop retrieval without validation leads to error accumulation ("hallucinations compounding")
- The signal: Longer answers contain more errors; deeper reasoning is worse than shallow
- The fix: Add feedback. Use Corrective RAG or Self-RAG to validate and re-retrieve if needed

**5. Ignoring Latency Budget**
- The mistake: "Agentic RAG is better, so we'll use it everywhere"
- The reality: Agentic RAG is ~5–10x slower than Naive RAG (multiple sequential calls). If latency budget is <500ms, Adaptive RAG is the better choice
- The signal: P95 latency is 5+ seconds; users see timeouts
- The fix: Measure latency distribution. Choose the simplest architecture that meets your SLA

---

## Using the Taxonomy

1. **To understand a new architecture**: Find it in the table. Read the retrieval control, feedback loop, and scope. That tells you the mechanism.
2. **To choose which one to implement**: Start with the left side of the table (Static, Single-Hop, No Feedback). Move right only if you hit a forcing function (the roadmap covers these).
3. **To explain the difference in an interview**: Use the classification axes. "Self-RAG differs from Advanced RAG in three ways: agent-driven orchestration, iterative scope, and external feedback loop."
4. **To spot opportunities for optimization**: Check the "Latency Class" column. If you're at Medium or Slow, can you move left (simpler orchestration)?
