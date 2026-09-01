# 51 — RQ-RAG (Refinement-aware Query for RAG)

> Fine-tunes the LLM itself to explicitly rewrite, decompose, or disambiguate a query — as a learned skill invoked via special tokens — rather than relying on a frozen model prompted with a fixed rewriting template.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
User Query
    │
    ▼
Fine-tuned Refinement Policy (LLM emits a special action token)
    │
    ├── <rewrite>       → produces a single reformulated query
    ├── <decompose>     → produces N sub-queries
    ├── <disambiguate>  → produces a clarified, specific query
    └── <no-op>         → uses the query as-is
    │
    ▼
Retriever (runs one retrieval call PER refined query)
    │
    ▼
Tree of Candidate Paths (each path = one refinement choice + its retrieved evidence + a draft answer)
    │
    ▼
PPL-based Path Selector (scores each path by output token perplexity / confidence)
    │
    ▼
Final Answer (from the highest-scoring path in the decoding tree)
```

### Key Components

| Component | Responsibility |
|---|---|
| Refinement-tuned LLM | Single model fine-tuned to output one of REWRITE / DECOMPOSE / DISAMBIGUATE / no-op via special control tokens, instead of a hand-written rewriting prompt |
| Query Refiner | Executes the chosen operation, producing one rewritten query, several decomposed sub-queries, or one disambiguated query |
| Retriever | Standard dense/sparse retriever, invoked once per refined query produced |
| Tree Decoder | Expands multiple candidate refinement paths in parallel (since refinement is stochastic/multi-choice at each step) |
| Path Selector | Picks the best final answer across candidate paths using a perplexity-based confidence score |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Base model | Llama2-7B (as used in the original paper), any open-weight instruction-tuned LLM |
| Fine-tuning data construction | Self-instruct-style generation of (query, refinement-type, refined query) triples, distilled from a stronger LLM |
| Fine-tuning framework | Hugging Face `transformers` + `trl` / `peft` (LoRA) for supervised fine-tuning on refinement traces |
| Retriever | Any dense retriever (FAISS, Elasticsearch BM25, ColBERT) — RQ-RAG is retriever-agnostic |
| Reference implementation | [chanchimin/RQ-RAG on GitHub](https://github.com/chanchimin/RQ-RAG) |

---

## Q1. What is RQ-RAG and how does it differ from the query-rewriting techniques (HyDE, Multi-Query, Step-Back) already used in Advanced RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**RQ-RAG** (Refinement-aware Query for RAG, from *"RQ-RAG: Learning to Refine Queries for Retrieval Augmented Generation"*, Chan et al., 2024, [arXiv:2404.00610](https://arxiv.org/abs/2404.00610)) **fine-tunes** an LLM so that query refinement becomes a *learned, explicit skill* the model performs on its own — rather than a behavior induced by a frozen model plus a clever prompt.

**02-advanced-rag.md already covers three prompted refinement tricks:**

| Technique | How it works | Model state |
|---|---|---|
| HyDE | Prompt the LLM to hallucinate a fake answer, embed that instead of the query | Frozen, zero-shot prompted |
| Multi-Query expansion | Prompt the LLM to generate N paraphrases of the query | Frozen, zero-shot prompted |
| Step-Back prompting | Prompt the LLM to generate a more abstract/general version of the query | Frozen, zero-shot prompted |

**RQ-RAG's difference:**

```
Prompted rewriting (HyDE / Multi-Query / Step-Back):
  Frozen LLM + hand-written instruction
       │
       ▼
  "Please rephrase this query for better retrieval..."
       │
       ▼
  One fixed strategy applied uniformly to every query

RQ-RAG:
  Fine-tuned LLM has LEARNED, from training data, when to:
       │
       ├─ REWRITE       (query is fine but phrased badly for retrieval)
       ├─ DECOMPOSE      (query bundles multiple sub-questions)
       ├─ DISAMBIGUATE   (query is ambiguous / underspecified)
       └─ pass through unchanged (query is already retrieval-ready)
       │
       ▼
  The model itself DECIDES which operation fits THIS query,
  and can chain several refinement steps adaptively.
```

**Key distinction:** prompted rewriting techniques apply *one* strategy to *every* query via prompt engineering on a frozen model. RQ-RAG fine-tunes the model to **select among multiple refinement operations per-query** and to **chain them**, closer to how a human researcher would first clarify an ambiguous question, then split it into parts, then reword each part for a search engine.

</details>

---

## Q2. What are the three refinement operations, and how are they represented at inference time? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG introduces three explicit refinement types, each triggered by a special control token the fine-tuned model learns to emit:

| Operation | Special token | Purpose | Example |
|---|---|---|---|
| **Rewrite** | `<rewrite>` | Rephrase a poorly-worded query into retrieval-friendly language | "that movie with the guy who lost his memory" → "Christopher Nolan Memento plot" |
| **Decompose** | `<decompose>` | Split a compound/multi-hop query into independent sub-queries | "Compare the GDP of France and its population growth rate" → ["What is France's GDP?", "What is France's population growth rate?"] |
| **Disambiguate** | `<disambiguate>` | Resolve an underspecified query into a concrete one | "What is the capital?" → "What is the capital of Australia?" (using conversation context) |

**Inference-time control flow (simplified):**

```python
def rq_rag_step(query: str, context: str = "") -> list[str]:
    """
    The fine-tuned model is prompted to first emit a refinement-type token,
    then the refined query/queries conditioned on that choice.
    """
    prompt = f"""{context}
Query: {query}
Choose a refinement action: <rewrite>, <decompose>, <disambiguate>, or <answer_directly>."""

    action = refinement_llm.generate(prompt, max_tokens=16)   # e.g. "<decompose>"

    if action == "<decompose>":
        sub_queries = refinement_llm.generate(
            f"{prompt}\n<decompose>\nSub-queries:", max_tokens=128
        )
        return parse_list(sub_queries)          # ["query A", "query B", ...]

    elif action == "<rewrite>":
        rewritten = refinement_llm.generate(f"{prompt}\n<rewrite>\nRewritten query:")
        return [rewritten]

    elif action == "<disambiguate>":
        clarified = refinement_llm.generate(f"{prompt}\n<disambiguate>\nClarified query:")
        return [clarified]

    return [query]   # <answer_directly> — no refinement needed
```

**Why special tokens instead of a free-form instruction?** Because the model was **fine-tuned** on traces containing these tokens, emitting `<decompose>` reliably switches the model into a *decomposition-conditioned generation mode* it has practiced — this is far more reliable at inference time than hoping a frozen model correctly interprets an ad hoc natural-language instruction like "decompose this if needed."

</details>

---

## Q3. How is the RQ-RAG training data constructed, and how is the model fine-tuned? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG needs **supervised traces** that pair a query with the correct refinement type and the correct refined output — this data doesn't exist naturally, so the paper synthesizes it using a stronger teacher LLM.

**Step 1 — Synthesize refinement traces:**

```python
TEACHER_PROMPT = """Given the query below, decide whether it needs to be:
(a) rewritten for better search-engine retrieval,
(b) decomposed into independent sub-questions, or
(c) disambiguated using the given context.
If none apply, say so.

Query: {query}
Context: {context}

Output format:
Action: <rewrite|decompose|disambiguate|none>
Refined: <the refined query or list of sub-queries>"""

# Run over a large pool of QA queries (e.g. from ambiguous-QA, multi-hop QA,
# and search-log-style datasets) using a strong teacher model (e.g. GPT-4-class)
# to generate (query, action, refined_query) triples.
```

**Step 2 — Filter with retrieval-and-answer feedback:**

A generated refinement is only kept if using it to retrieve documents actually **improves** downstream answer correctness versus using the raw query — this turns the dataset into a curated set of refinements that are empirically useful, not just plausible-looking.

**Step 3 — Supervised fine-tuning:**

```python
# Each training example is a full trace: query → action token → refined query
# → retrieved docs → answer, concatenated as one sequence, with loss computed
# only on the tokens the model must generate (action token + refined query + answer).

training_example = """Query: Compare the GDP of France and its population growth rate.
<decompose>
Sub-queries: ["What is France's GDP?", "What is France's population growth rate?"]
[... retrieved passages ...]
Answer: France's GDP is approximately $3.0T; its population growth rate is approximately 0.2% annually."""

# Standard causal-LM fine-tuning (full fine-tune or LoRA) on a base Llama2-7B checkpoint,
# using cross-entropy loss over the full trace.
```

**Step 4 — Multi-path tree decoding at inference:**

Because the same query could plausibly warrant different refinement types, RQ-RAG doesn't commit to a single greedy path — it expands a **decoding tree** with several candidate refinement branches (e.g. try both `<rewrite>` and `<decompose>`), retrieves for each, drafts an answer per branch, and then picks the best branch using a perplexity-based confidence score (see Q4).

</details>

---

## Q4. How does the tree-decoding / path-selection mechanism decide which refined query to actually use? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Because refinement is a **stochastic decision** the model makes per query, RQ-RAG doesn't rely on a single greedy refinement — it explores multiple candidate paths and scores them.

```
                         Original Query
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                   ▼
       <rewrite>          <decompose>        <disambiguate>
            │                  │                   │
     retrieve(rewritten)  retrieve(sub-q1)    retrieve(clarified)
            │             retrieve(sub-q2)          │
            ▼                  ▼                   ▼
       draft answer A     draft answer B       draft answer C
            │                  │                   │
            └──────────────────┼───────────────────┘
                                ▼
                    Score each path by output
                    perplexity / confidence
                                │
                                ▼
                    Select lowest-perplexity
                    (highest-confidence) path
                                │
                                ▼
                          Final Answer
```

```python
def rq_rag_tree_decode(query: str, context: str, k_branches: int = 3) -> str:
    candidate_paths = []

    for action in ["<rewrite>", "<decompose>", "<disambiguate>"]:
        refined_queries = refine(query, context, action)          # from Q2
        docs = [retriever.search(q) for q in refined_queries]
        answer, logprobs = generate_answer_with_logprobs(query, docs)

        # Confidence proxy: average per-token log-probability of the answer
        confidence = sum(logprobs) / len(logprobs)
        candidate_paths.append({"action": action, "answer": answer, "confidence": confidence})

    best = max(candidate_paths, key=lambda p: p["confidence"])
    return best["answer"]
```

**Why perplexity/confidence as the selector, and not another LLM judge?**
- It's essentially free — the generation model already computes token log-probabilities during decoding, no extra model call needed
- Empirically, answers grounded in *correctly refined* retrieval tend to have lower perplexity because the retrieved evidence is more directly relevant, making the answer more "predictable" given the context
- This keeps the extra latency of tree decoding bounded — the added cost is running the retriever + generator per branch, not an extra verification pass

**Multi-hop benefit:** for a query needing 2 hops, the `<decompose>` branch retrieves for each sub-query independently, giving the generator focused evidence for each hop rather than one retrieval call diluted across a compound question — this is the core reason RQ-RAG's reported gains are larger on multi-hop QA datasets than single-hop ones.

</details>

---

## Q5. What are the costs and failure modes of RQ-RAG compared to prompted rewriting, and when would you NOT use it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RQ-RAG trades **inference-time cost and engineering overhead** for **higher-quality, per-query-adaptive refinement**. That trade isn't always worth it.

**Costs:**

| Cost dimension | Prompted rewriting (HyDE/Multi-Query/Step-Back) | RQ-RAG |
|---|---|---|
| Training required | None — works with any frozen instruction-tuned LLM | Yes — requires building a refinement-trace dataset and fine-tuning |
| Inference latency | 1–2 extra LLM calls | Multiple branches × (refine + retrieve + generate) per query — tree decoding multiplies retrieval and generation calls by the number of branches explored |
| Maintainability | Swap prompt anytime, no retraining | Refinement behavior baked into weights — updating strategy requires re-fine-tuning |
| Portability across base models | Works with any LLM via prompting | Fine-tune is tied to a specific base model checkpoint; upgrading the base model means re-running the fine-tuning pipeline |

**Failure modes:**

1. **Distribution shift in refinement decisions** — if production queries look very different from the synthetic refinement-trace training data, the model may pick the wrong action (e.g. decomposing a query that was actually fine as-is), adding latency without benefit.
2. **Compounding retrieval calls** — `<decompose>` on a query with many implicit sub-questions can trigger a retrieval call per sub-query per branch; without a cap, this can multiply retrieval cost badly on adversarial or overly compound queries.
3. **Confidence-score miscalibration** — perplexity-based path selection can favor a fluent-but-wrong answer over a correct-but-awkwardly-phrased one, since perplexity measures predictability, not factual correctness.

**When to prefer prompted rewriting instead:**
- Rapid prototyping or low query volume, where the fine-tuning investment doesn't pay off
- Frequently swapping base models (e.g. evaluating multiple vendor LLMs) — prompted techniques port instantly, fine-tuned refinement doesn't
- Cost-sensitive deployments where the multi-branch tree-decoding overhead isn't justified by the accuracy gain

**When RQ-RAG is worth it:**
- High query volume with a stable base model, where the one-time fine-tuning cost amortizes
- Query workloads with a genuine mix of ambiguous, compound, and well-formed queries — a single hand-written prompt strategy can't adapt per-query the way a fine-tuned action-selection policy can
- Can be combined with **Adaptive RAG** (file 11) — the adaptive router decides *whether* to retrieve at all, while RQ-RAG decides *how* to refine the query once retrieval is triggered; the two are complementary routing layers, not competitors

</details>

---

## Real-World Applications

- **Multi-hop QA assistants**: Decomposing compound questions (e.g. "How does X compare to Y over time?") into independently retrievable sub-queries, similar in spirit to file 19's Iterative Multi-Hop RAG but driven by a fine-tuned action rather than an iterative loop
- **Conversational search**: The `<disambiguate>` operation resolves pronoun/ellipsis-heavy follow-up queries ("what about the second one?") using prior turn context, an alternative to the memory-augmentation approach in file 21 (Memory/Conversational RAG)
- **Enterprise search over jargon-heavy corpora**: `<rewrite>` learns domain-specific query reformulation (casual phrasing → internal terminology) without needing a hand-maintained synonym dictionary
- **Research assistants**: Chaining decomposition then rewriting lets the system handle "explain the tradeoffs between A and B, citing recent work" as a structured multi-step retrieval plan rather than one noisy retrieval call
