# 45 — LongRAG + Self-Route

> Retrieves much larger units — whole documents or grouped passages instead of ~100-token chunks — and lets a long-context LLM reader do the fine-grained extraction, while Self-Route decides per-query whether to retrieve at all or just stuff the whole corpus.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
                          Corpus
                            │
                            ▼
             Long-Unit Grouper (groups related
             passages / whole docs into ~4K-token
             retrieval units — 30x larger than a
             standard DPR-style 100-word chunk)
                            │
                            ▼
                Long Retriever (dense retrieval
                over far fewer, far larger units —
                e.g. 22M → 600K units on Wikipedia)
                            │
                            ▼
                 Top-k Long Units (4K–8K tokens
                 each, minimal fragmentation)
                            │
                            ▼
              Long-Context LLM Reader (extracts
              the answer from large, coherent
              units instead of stitching chunks)
                            │
                            ▼
                        Answer

──────────────── Self-Route (decision layer) ────────────────

        Query + top-k retrieved passages
                            │
                            ▼
        LLM self-assessment: "Can this be
        answered from the retrieved passages?"
                    │               │
              Yes ──┘               └── No / Not confident
                    │                           │
                    ▼                           ▼
         Answer via RAG (cheap,          Fall back to full
         short context)                  Long-Context stuffing
                                          (expensive, all passages)
```

### Key Components

| Component | Responsibility |
|---|---|
| Long-Unit Grouper | Merges small passages into large (~4K-token) retrieval units, or retrieves whole documents |
| Long Retriever | Dense retrieval over a much smaller pool of large units, reducing the "needle in 22M haystacks" burden |
| Long-Context LLM Reader | Reads large, semantically coherent units instead of fragmented chunks — no answer split across chunk boundaries |
| Self-Route Predictor | LLM self-assessment step: judges whether top-k retrieved passages are sufficient to answer the query |
| Long-Context Fallback | Full-corpus (or full top-N-document) stuffing invoked only when Self-Route flags the query as unanswerable from RAG alone |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Long-context LLMs | Claude (200K+ tokens), Gemini 1.5/2.x (1M–2M tokens), GPT-4.1/4o (128K+ tokens) |
| Retrieval unit construction | Section/document-level grouping instead of fixed-size chunkers (e.g. LangChain `RecursiveCharacterTextSplitter` at 4K+ token settings) |
| Dense retriever | Standard bi-encoder (e.g. `text-embedding-3-large`, BGE) over large units |
| Reference implementation | [TIGER-AI-Lab/LongRAG](https://tiger-ai-lab.github.io/LongRAG/) (open-source) |
| Routing signal | Self-reflection prompt (no separate classifier needed) — open-source alternative: a small fine-tuned router model |

---

## Q1. What is LongRAG, and how does it differ from Long-Context RAG (file 10)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**LongRAG** ("LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs," Jiang, Ma & Chen, 2024, [arXiv:2406.15319](https://arxiv.org/abs/2406.15319)) keeps the *retrieve-then-read* pipeline but changes the **unit of retrieval**. Instead of the traditional ~100-word DPR-style passage, it groups related text into much larger units — roughly 4K tokens each, about 30x larger — often whole documents or document clusters. On Wikipedia this shrinks the retrieval pool from ~22M small chunks to ~600K large units, so the retriever has far fewer "needles" to search through, and each retrieved unit gives the reader enough surrounding context that answers rarely get split across a chunk boundary.

**File 10 (Long-Context RAG)** is a different, more extreme point on the same spectrum: it mostly *removes* the retrieval-unit problem by stuffing entire documents (or the whole corpus, if it fits) into a 100K–1M token context window, often with only a coarse BM25/vector pre-filter to narrow which documents to include at all. It doesn't redesign what a "retrieval unit" is — it just asks "why chunk at all if the context window is big enough?"

```
                     Chunk size          What changes
Naive/Advanced RAG:  ~100-300 tokens     nothing — small units, cheap retrieval
LongRAG:             ~4K tokens/unit     retrieval UNIT size — still a retriever,
                                         fewer/bigger units, less fragmentation
Long-Context RAG:    whole doc(s)        retrieval STRATEGY — coarse filter only,
(file 10)                                then stuff everything into the window
```

**The key distinction:** LongRAG is still fundamentally a retrieval architecture — it just rebalances the "heavy retriever, light reader" imbalance of classic RAG by making units bigger. Long-Context RAG (file 10) is closer to abandoning fine-grained retrieval altogether and leaning on the model's context window and prompt caching instead.

</details>

---

## Q2. How does LongRAG rebalance the "heavy retriever, light reader" problem, and what results does it report? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The paper's core diagnosis: in classic RAG (e.g. DPR + a short-context reader), the retriever does *all* the hard work — searching millions of tiny 100-word passages to find the one that contains the answer — while the reader's job is trivial (extract the answer from a passage it already knows contains it). This is an imbalanced design: a "heavy" retriever paired with a "light" reader.

LongRAG rebalances this by making units 30x larger (grouping into ~4K-token units, sometimes whole Wikipedia articles), which:
- Shrinks the number of units the retriever must distinguish between (22M → 600K on Wikipedia) — an easier retrieval problem
- Pushes more of the actual reasoning burden onto the reader, which now needs a long-context LLM (since units are 4K+ tokens) capable of finding and synthesizing the answer within a longer span

```python
# Simplified: group short passages into long retrieval units
def build_long_units(passages: list[str], target_tokens: int = 4000) -> list[str]:
    units, current, current_len = [], [], 0
    for p in passages:
        p_len = count_tokens(p)
        if current_len + p_len > target_tokens and current:
            units.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(p)
        current_len += p_len
    if current:
        units.append("\n\n".join(current))
    return units  # ~30x fewer, ~30x larger than standard 100-word chunks

def long_rag_answer(query: str, long_retriever, reader_llm) -> str:
    top_units = long_retriever.search(query, k=4)          # search over large units
    context = "\n\n---\n\n".join(top_units)
    return reader_llm.generate(query=query, context=context)  # needs long context window
```

**Reported results (no additional training required):** LongRAG achieves 62.7% EM on Natural Questions and 64.3% EM on full-wiki HotpotQA, competitive with or exceeding fine-tuned short-chunk RAG pipelines, purely from the unit-size change plus an off-the-shelf long-context reader.

</details>

---

## Q3. How does Self-Route decide between RAG and full long-context stuffing at query time? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Self-Route** comes from a separate paper, "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach" (Li, Li, Zhang, Mei & Bendersky, Google Research, 2024, [arXiv:2407.16833](https://arxiv.org/abs/2407.16833)). Its finding: when a long-context LLM has enough budget, long-context stuffing (LC) tends to *outperform* RAG on average — but it is far more expensive per query. Self-Route is a routing mechanism that gets most of LC's quality at close to RAG's cost.

**The mechanism is two steps, using the same LLM for both:**

```python
SELF_ROUTE_CHECK_PROMPT = """You are given a question and some retrieved passages.
Decide if the passages contain enough information to answer the question.
Reply with exactly one word: "ANSWERABLE" or "UNANSWERABLE".

Question: {query}
Passages:
{top_k_passages}
"""

def self_route(query: str, all_passages: list[str], retriever, llm, k: int = 5) -> str:
    top_k = retriever.search(query, k=k)

    verdict = llm.generate(SELF_ROUTE_CHECK_PROMPT.format(
        query=query, top_k_passages="\n\n".join(top_k)
    )).strip()

    if verdict == "ANSWERABLE":
        # Cheap path: standard RAG with just the top-k passages
        return llm.generate(query=query, context="\n\n".join(top_k))
    else:
        # Expensive fallback: stuff the FULL passage pool (long-context mode)
        return llm.generate(query=query, context="\n\n".join(all_passages))
```

**Why this works:** most queries are perfectly answerable from a handful of top-k passages — the LLM's self-assessment is a cheap, reliable proxy for "did retrieval fail this query?" Only the harder tail of queries (multi-hop, needle scattered across many documents, retrieval literally missed the right passage) fall through to the expensive full long-context pass.

</details>

---

## Q4. How do LongRAG and Self-Route compose in a single pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The two ideas operate at different layers and are complementary, not competing:

```
┌─────────────────────────────────────────────────────────────┐
│ LongRAG = WHAT gets retrieved                                │
│   → redesigns the retrieval unit (4K-token units instead     │
│     of 100-word chunks) so each unit is coherent enough      │
│     that the reader rarely needs more context                │
├─────────────────────────────────────────────────────────────┤
│ Self-Route = WHETHER to retrieve at all, or go full long-ctx │
│   → a per-query decision layer that sits in FRONT of         │
│     generation, choosing cheap-RAG vs. expensive-LC          │
└─────────────────────────────────────────────────────────────┘
```

**Combined pipeline:**

```python
def longrag_with_self_route(query: str, long_units: list[str], long_retriever, llm) -> str:
    # 1. LongRAG: retrieve large, coherent units (not fragmented chunks)
    top_units = long_retriever.search(query, k=3)   # each unit ~4K tokens

    # 2. Self-Route: ask the LLM if these units are sufficient
    verdict = llm.generate(SELF_ROUTE_CHECK_PROMPT.format(
        query=query, top_k_passages="\n\n".join(top_units)
    )).strip()

    if verdict == "ANSWERABLE":
        return llm.generate(query=query, context="\n\n".join(top_units))  # cheap
    else:
        # Fall back to ALL long units, not just top-3 — still large units,
        # just no longer top-k-limited
        return llm.generate(query=query, context="\n\n".join(long_units))  # expensive
```

**Why compose them:** LongRAG's large units already reduce the odds that Self-Route triggers the expensive fallback (less fragmentation means fewer "the answer was split across two chunks" failures). Self-Route then adds a cost control on top, so you don't pay full long-context prices for the large majority of queries that a handful of well-sized units can already answer.

</details>

---

## Q5. What are the failure modes and cost tradeoffs of LongRAG-style large retrieval units? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Larger retrieval units trade one set of failure modes for another:

| Failure mode | Small chunks (Naive RAG) | Large units (LongRAG) |
|---|---|---|
| Answer split across chunk boundary | Common | Rare — unit is large enough to contain full context |
| Irrelevant content diluting the reader's attention | Rare — units are tightly scoped | Common — a 4K-token unit may be 90% irrelevant to the query |
| Retriever precision | Must be very precise (small target) | Easier — fewer, more distinguishable units |
| Cost per retrieved item | Low (small tokens in context) | High (4K tokens × k units = large prompt, expensive per call) |
| Reranking cost | Cheap (rerank many small candidates) | Expensive (reranking large units costs more per candidate) |
| Embedding quality | Pooled vector represents a narrow topic well | Pooled vector for a 4K-token unit can blur multiple sub-topics ("semantic dilution") |

**The core tension:** LongRAG reduces fragmentation-driven hallucination and retrieval misses, but it does so by asking the reader LLM to do more filtering work per call, and it makes every retrieved item more expensive to embed, index, rerank, and feed to the generator. This is the same "heavy reader" tradeoff the paper explicitly accepts in exchange for a "lighter retriever."

**Combining with other techniques to mitigate the downside:**

- **Self-Route** caps the blast radius: most queries never need the full long-context fallback, so the expensive path is rare rather than the default.
- **Hierarchical retrieval** (retrieve at the large-unit level, then a second pass extracts the specific sub-span within the winning unit) recovers some of small-chunk RAG's precision without giving up LongRAG's reduced fragmentation.
- **Prompt caching** amortizes the cost of repeatedly feeding the same large unit across multiple queries in a session (same mechanism used in Long-Context RAG, file 10).

**When LongRAG is the wrong choice:** if the corpus consists of many short, independent facts (e.g. a FAQ database or a table of key-value records), forcing 4K-token grouping only adds irrelevant padding — the imbalance LongRAG fixes doesn't exist in the first place, and Naive RAG's small chunks remain the better fit.

</details>

---

## Real-World Applications

- **Open-domain QA over Wikipedia-scale corpora**: LongRAG's own benchmark — grouping Wikipedia into document-level units instead of DPR's 100-word passages, evaluated on NQ and full-wiki HotpotQA
- **Enterprise search over long-form contracts/policies**: retrieving whole clauses or sections instead of arbitrary fixed-size chunks avoids splitting a single obligation across two chunks
- **Google's hybrid RAG/long-context search assistants**: Self-Route-style routing to avoid paying long-context token costs on the majority of simple, single-hop queries
- **Cost-sensitive production RAG**: Self-Route as a cheap "escalation" gate before falling back to an expensive long-context or agentic retrieval pass
- **Research/legal assistants with mixed query difficulty**: routing simple lookup questions through cheap RAG while escalating synthesis-heavy questions to full-document long-context reasoning
