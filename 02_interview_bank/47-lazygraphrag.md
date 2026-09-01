# 47 — LazyGraphRAG

> Defers all expensive LLM summarization from index time to query time — building only a cheap NLP-extracted concept graph upfront, then summarizing just the subgraph a specific query needs.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
INDEX TIME (cheap — no LLM calls)
        │
        ▼
Docs ──► NLP Noun-Phrase Extractor (no LLM — standard
         NLP entity/noun-phrase tagging)
        │
        ▼
Noun-Phrase Co-occurrence Graph
(edges = phrases appearing together in the
 same text window; no LLM summarization here)
        │
        ▼
Graph stored — indexing cost ≈ vector RAG,
~0.1% of full GraphRAG's indexing cost

════════════════════════════════════════════
QUERY TIME (expensive work deferred to here)
        │
        ▼
Query
        │
        ▼
Relevance Test Budget (iterative LLM relevance
testing — "is this subgraph region relevant
to THIS query?" — spend scales with the
query's difficulty, not a fixed index-time cost)
        │
        ▼
Best-First / Iterative Subgraph Expansion
(only expand the parts of the graph that pass
 the relevance test; skip everything else)
        │
        ▼
Query-Time-Only Summarization (LLM summarizes
ONLY the relevant subgraph found above — never
the whole graph, never done in advance)
        │
        ▼
Generator ──► Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| NLP Noun-Phrase Extractor | Cheap, non-LLM extraction of noun phrases / entities from text at index time |
| Co-occurrence Graph Builder | Builds a graph where edges represent phrases co-occurring in the same text window — no LLM relationship extraction, no LLM summarization |
| Relevance Test Budget | The single tunable parameter controlling cost/quality: how many iterative LLM relevance checks are spent narrowing down to the relevant subgraph per query |
| Iterative Subgraph Expansion | Best-first search over the graph, expanding only regions that pass the relevance test, instead of pre-computing global community summaries |
| Query-Time Summarizer | The ONLY place an LLM produces a summary — and only for the specific subgraph relevant to the current query |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Reference implementation | [microsoft/graphrag](https://github.com/microsoft/graphrag) (LazyGraphRAG ships as a mode/approach within the GraphRAG open-source library) |
| NLP extraction (index time) | spaCy / standard noun-phrase chunkers — no LLM calls required at index time |
| Query-time LLM | Any chat LLM (GPT-4o, Claude, Gemini) used only for relevance testing and final subgraph summarization |
| Source | [Microsoft Research Blog, Nov 2024 — "LazyGraphRAG: Setting a new standard for quality and cost"](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) (Darren Edge, Ha Trinh, Jonathan Larson) — this is a blog post, not an arXiv paper |

---

## Q1. What is LazyGraphRAG, and how does it differ from GraphRAG (file 05)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**LazyGraphRAG** is a Microsoft Research approach, published as a blog post ("LazyGraphRAG: Setting a new standard for quality and cost," Edge, Trinh & Larson, Microsoft Research Blog, November 2024 — not an arXiv paper) that pushes essentially all expensive LLM work from index time to query time.

**GraphRAG (file 05)** does the opposite: at *index* time it runs LLM-based entity and relationship extraction over every document, then runs Leiden community detection and generates LLM summaries for every community — all before a single query is ever asked. This upfront cost is what lets GraphRAG answer broad "global" questions well (it already has summaries of the whole corpus's themes ready to go), but it means you pay full LLM summarization costs for the entire corpus even if 90% of it is never queried.

**LazyGraphRAG's index-time step is almost free by comparison:** it only extracts noun phrases with cheap NLP tooling (no LLM) and builds a co-occurrence graph (phrases that appear near each other in the text are linked). No LLM ever touches the corpus at index time.

```
                    Index-time LLM cost         Query-time LLM cost
GraphRAG (05):      HIGH (entity/relation        LOW (reads pre-built
                    extraction + Leiden           community summaries)
                    community summarization
                    for the WHOLE corpus)

LazyGraphRAG:       ~ZERO (NLP noun-phrase        Scales with query difficulty
                    co-occurrence graph only,     (iterative relevance testing +
                    no LLM calls)                 summarization of ONLY the
                                                   relevant subgraph)
```

**The headline number:** LazyGraphRAG's indexing cost is reported as roughly equal to plain vector RAG, and about **0.1% of full GraphRAG's indexing cost** — while matching or beating GraphRAG's answer quality on global/community-level questions, because the expensive summarization is targeted exactly at what a given query needs instead of summarizing everything speculatively.

</details>

---

## Q2. How does the "relevance test budget" work as LazyGraphRAG's core cost/quality control? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Since LazyGraphRAG has no pre-built community summaries to fall back on, it must figure out — at query time — which part of the noun-phrase co-occurrence graph is actually relevant to the current question. It does this through **iterative relevance testing**: repeatedly asking an LLM (or a cheaper classifier) whether a given graph region/subgraph is relevant to the query, and only expanding further into regions that pass.

```python
def lazygraphrag_query(query: str, graph, llm, relevance_budget: int = 50) -> str:
    """
    relevance_budget: the single tunable parameter controlling cost vs. quality —
    how many relevance tests (LLM calls) this query is allowed to spend.
    """
    frontier = graph.get_seed_candidates(query)   # cheap initial candidates via co-occurrence match
    relevant_subgraph = set()
    tests_spent = 0

    while frontier and tests_spent < relevance_budget:
        candidate = frontier.pop_best_first()      # best-first: try most promising candidate next
        is_relevant = llm.judge_relevance(query, candidate)   # 1 relevance test = 1 budget unit
        tests_spent += 1

        if is_relevant:
            relevant_subgraph.add(candidate)
            frontier.extend(graph.neighbors(candidate))  # expand further ONLY from relevant nodes

    # Summarization happens ONCE, over ONLY the relevant subgraph found above
    summary = llm.summarize(relevant_subgraph)
    return llm.generate_answer(query, summary)
```

**Why "lazy" is the right word:** a classic GraphRAG-style system has already summarized every community whether or not it's ever queried — that work is "eager." LazyGraphRAG defers (is lazy about) doing any summarization until it knows exactly which subgraph a specific query needs, then does the minimum summarization work required.

**The budget as a dial:** increasing `relevance_budget` spends more LLM calls exploring the graph more thoroughly per query — trading query-time latency/cost for better recall on hard, broad "global" questions. Decreasing it makes queries cheaper and faster but risks missing relevant but harder-to-reach regions of the graph. This single parameter is what the blog post describes as controlling the cost-quality tradeoff "in a consistent manner" across the whole system.

</details>

---

## Q3. How does LazyGraphRAG's approach to "global" questions differ from LightRAG's dual-level retrieval (file 15)? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both systems care about answering broad, corpus-level ("global") questions well, but they solve it very differently:

**LightRAG (file 15)** builds an LLM-extracted entity-relationship graph at index time (still cheaper than full GraphRAG, but still an LLM pass over every chunk), then explicitly maintains **two levels** of the index simultaneously — local (entity-anchored, 1-hop) and global (community/theme summaries, optionally via Leiden). At query time it picks which level(s) to search based on the query type, but both levels already exist before any query arrives.

**LazyGraphRAG** has no LLM-extracted entity-relationship graph and no pre-built global-level summaries at all. Its index is a cheap noun-phrase co-occurrence graph. "Global" answering happens entirely at query time: the relevance-test-and-expand loop (Q2) naturally reaches a broader swath of the graph for a broad question than for a narrow one, and the query-time summarizer only ever summarizes what that specific query's exploration surfaced.

```
                 Index-time graph construction        "Global" question handling
LightRAG (15):   LLM extracts entities + relations,   Dedicated GLOBAL retrieval path
                 dual-level index built upfront        queries pre-built community/theme
                 (local + optional global/community)   summaries directly

LazyGraphRAG:    NLP noun-phrase co-occurrence only,   No dedicated global path — broader
                 no LLM extraction, no pre-built        questions simply cause the relevance-
                 global summaries                       test loop to expand further/wider
                                                         across the graph before summarizing
```

**The practical implication:** LightRAG pays a moderate, fixed LLM cost at index time in exchange for having both local and global answers ready instantly. LazyGraphRAG pays almost nothing at index time, but a global question costs noticeably more at query time than a narrow one (more relevance tests, wider subgraph, bigger summarization pass) — the cost is variable and query-dependent rather than fixed and upfront.

</details>

---

## Q4. What does the LazyGraphRAG index actually contain, given that it skips LLM entity/relationship extraction? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Where GraphRAG's graph nodes are LLM-identified entities (`"Apple Inc."`, `"Tim Cook"`) connected by LLM-typed relationship edges (`CEO_OF`), LazyGraphRAG's index is built entirely from cheap, deterministic NLP:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def build_cooccurrence_graph(documents: list[str], window_size: int = 100):
    """
    No LLM calls anywhere in this function — pure NLP noun-phrase extraction
    plus co-occurrence counting, exactly what makes LazyGraphRAG's indexing
    cost comparable to plain vector RAG.
    """
    graph = {}   # phrase -> {co-occurring phrase: count}

    for doc_text in documents:
        doc = nlp(doc_text)
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Any two noun phrases within `window_size` tokens of each other get an edge
        for i, phrase_a in enumerate(noun_phrases):
            for phrase_b in noun_phrases[i+1:i+window_size]:
                if phrase_a == phrase_b:
                    continue
                graph.setdefault(phrase_a, {}).setdefault(phrase_b, 0)
                graph[phrase_a][phrase_b] += 1
                graph.setdefault(phrase_b, {}).setdefault(phrase_a, 0)
                graph[phrase_b][phrase_a] += 1

    return graph  # cheap co-occurrence graph — ready for query-time relevance testing
```

**What this graph does NOT contain, unlike GraphRAG/LightRAG:**
- No typed relationships (no `ACQUIRED`, `CEO_OF` edge labels — just "these phrases co-occur")
- No entity resolution/deduplication pass (an LLM never confirmed that "Apple" and "Apple Inc." are the same entity)
- No community summaries of any kind

**Why this is still useful:** co-occurrence is a strong, cheap proxy for semantic relatedness — phrases that repeatedly appear near each other in text usually *are* related, even without an LLM confirming the relationship type. The relevance-testing step at query time (Q2) is where an LLM finally gets involved to judge whether that structural signal is actually relevant to a specific question — deferred exactly as far as possible.

</details>

---

## Q5. What are LazyGraphRAG's failure modes, and when would you still choose full GraphRAG or LightRAG instead? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Deferring cost to query time is not free — it just moves the bill and changes who pays it and when.

| Scenario | Best fit | Why |
|---|---|---|
| High query volume, corpus queried repeatedly with similar questions | **GraphRAG (05) or LightRAG (15)** | Pre-built summaries are reused across many queries — amortizing the upfront index cost. LazyGraphRAG re-runs relevance testing and summarization on every query (unless results are cached), so heavy repeat-query traffic can end up *more* expensive over time. |
| Low query volume, corpus rarely fully explored, cost-sensitive indexing | **LazyGraphRAG** | Index cost stays near-zero (~0.1% of GraphRAG) regardless of corpus size; you only pay for the parts of the graph actual queries touch. |
| Strict low-latency requirement per query (e.g. sub-second) | **GraphRAG/LightRAG** | Their summaries are pre-computed, so query-time work is comparatively small. LazyGraphRAG's iterative relevance-test loop adds real per-query latency, especially for broad/global questions with a large relevance budget. |
| Need typed relationships for downstream logic (e.g. "give me all `ACQUIRED` edges") | **GraphRAG/LightRAG/KAG (file 24)** | LazyGraphRAG's co-occurrence graph has no relationship types at all — it can't answer structured graph queries, only support relevance-guided text summarization. |
| Corpus changes frequently (streaming ingestion) | **LazyGraphRAG** | Cheap NLP-only indexing makes incremental updates far less costly than re-running LLM extraction/Leiden clustering/community re-summarization on every corpus change. |
| Very deep multi-hop reasoning requiring precise entity linking | **GraphRAG/LightRAG** | Entity resolution and typed edges give more reliable multi-hop traversal than a co-occurrence signal, which can produce false positives (two unrelated phrases that just happen to appear near each other often). |

**The core tradeoff, restated:** LazyGraphRAG's relevance-test budget makes cost-quality a *tunable, query-time* dial instead of a fixed *index-time* sunk cost — excellent for exploratory, low-repeat-query, cost-sensitive, or rapidly-changing corpora, but it gives up the "pay once, answer cheaply forever" property that makes GraphRAG's and LightRAG's upfront summarization worthwhile for high-volume, stable, frequently-repeated-question workloads.

**Mitigating the repeat-query weakness:** cache relevant-subgraph summaries keyed by query (or query cluster) so that once LazyGraphRAG has paid the relevance-testing cost for a question, semantically similar future queries can reuse the cached subgraph summary instead of re-running the full iterative search.

</details>

---

## Real-World Applications

- **Microsoft's GraphRAG open-source library**: LazyGraphRAG ships as a lower-cost mode within [microsoft/graphrag](https://github.com/microsoft/graphrag), positioned as the cost-sensitive alternative to full GraphRAG indexing
- **Exploratory research over large, rarely-fully-queried document collections**: near-zero indexing cost makes it practical to index entire archives "just in case," since cost is only incurred by the parts actually queried
- **Cost-sensitive enterprise deployments with unpredictable corpus growth**: incremental, cheap NLP-only indexing avoids the LLM-extraction bill scaling with every new document
- **Global/community-style questions on a budget** ("What are the main themes across this entire corpus?"): reported to match or beat full GraphRAG's global search quality at a fraction of the cost, at comparable query-time budget to vector RAG
- **Rapid prototyping of graph-based RAG** before committing to full GraphRAG's heavier indexing pipeline for a production system
