# 48 — Astute RAG (Conflict-Aware Knowledge Consolidation)

> Elicits the LLM's own parametric knowledge as an explicit "internal source," then iteratively consolidates it with retrieved passages — grouping consistent information and resolving conflicts source-by-source instead of blindly trusting whatever was retrieved.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
  │
  ├──► External Retriever ──► Retrieved Passages (tagged: source = "external")
  │
  └──► Internal Knowledge Generator
         (prompt the LLM to recall what it already knows about the query,
          WITHOUT showing it any retrieved documents)
              │
              ▼
         Generated Passages (tagged: source = "internal")
              │
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │   Iterative Source-Aware Knowledge Consolidation         │
  │   round 1: group passages that agree with each other     │
  │            across BOTH sources → consistent info clusters│
  │   round 2: isolate passages that conflict → flag as       │
  │            unreliable / contested, weigh by source trust  │
  │   round 3 (repeat until stable): merge, drop noise         │
  └─────────────────────────────────────────────────────────┘
              │
              ▼
  Source-Attributed Answer Generator
  (answer states which claims came from which reconciled source,
   and abstains or hedges on unresolved conflicts)
```

### Key Components

| Component | Responsibility |
|---|---|
| External Retriever | Standard dense/hybrid retriever fetching candidate passages from the corpus |
| Internal Knowledge Generator | Prompts the LLM to produce its own answer/passages from parametric memory alone, tagged as an internal "source" |
| Source-Aware Consolidator | Iteratively groups agreeing passages (regardless of source) and isolates conflicting ones, adjusting trust per source |
| Conflict Resolver | Applies reliability heuristics (source agreement count, internal-vs-external corroboration) to decide which claim wins |
| Source-Attributed Generator | Produces the final answer, explicitly citing which reconciled source(s) support each claim, or abstaining if conflict is unresolved |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Reference implementation | Astute RAG (Wang et al., 2024) — Google Cloud AI Research |
| Retriever | Any dense retriever (Contriever, DPR) or hybrid BM25 + dense stack |
| Internal-knowledge elicitation | Zero-context prompting of the same generator LLM (Gemini, GPT-4, Claude) |
| Consolidation logic | Custom iterative merge/conflict-detection prompts (no separate NLI model required in the original paper, though one can be substituted) |
| Evaluation benchmarks | Conflicting-evidence / counterfactual QA sets built on Natural Questions, TriviaQA, PopQA |

---

## Q1. What problem does Astute RAG solve, and how does it differ from standard RAG's handling of retrieved passages? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Astute RAG** (Wang, Wan, Sun, Chen, Arık — "Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models," arXiv:2410.07176, Google Cloud AI Research, Oct 2024) targets a failure mode standard RAG mostly ignores: **retrieval is never perfect**, and when it's wrong, standard RAG still forces the LLM to answer as if the retrieved text were ground truth.

**Standard RAG's naive assumption:**

```
Query: "Who is the CEO of Company X?"
Retrieved passage (outdated): "Company X's CEO is Jane Smith (as of 2019)."
Standard RAG → "The CEO of Company X is Jane Smith."   ← wrong, blindly trusted stale/irrelevant text
```

Standard RAG has no mechanism to notice that the retrieved passage might be outdated, irrelevant, or contradicted by what the model already knows — it just conditions generation on whatever came back from the retriever.

**Astute RAG's approach:**

1. Ask the LLM what it already knows about the query, *before* showing it any retrieved documents → this becomes an "internal" source, tagged the same way an external passage would be.
2. Consolidate the internal source and the external passages together, iteratively grouping agreeing statements and isolating conflicting ones.
3. Generate a final answer that is attributed to whichever reconciled source(s) actually support it — and can hedge or abstain if the conflict can't be resolved.

**Why this matters for robustness:** in worst-case scenarios where retrieval returns irrelevant or actively misleading passages, Astute RAG is reported to be the only tested method that still matches or beats the no-RAG (parametric-only) baseline — i.e., bad retrieval never makes it strictly worse than not retrieving at all. This directly targets the failure mode of "hallucination despite (bad) context," where the model confidently repeats something wrong because it was in the retrieved text rather than because it's true.

</details>

---

## Q2. How does Astute RAG elicit and represent the LLM's "internal" knowledge as a source? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The key trick is **adaptive internal knowledge generation**: prompt the same generator LLM to answer from memory alone, with no retrieved documents in context, and format the output as if it were just another retrieved passage.

```python
from anthropic import Anthropic

client = Anthropic()

def generate_internal_knowledge(query: str) -> str:
    """Elicit the LLM's own parametric knowledge, tagged as an internal 'source'."""
    prompt = f"""Answer the following question using only your own internal
knowledge. Do not say you don't know — provide your best recollection,
even if you are not fully certain. Write it as a short passage of facts,
the same way a retrieved document would be written.

Question: {query}

Internal knowledge passage:"""

    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text


def build_tagged_sources(query: str, external_passages: list[str]) -> list[dict]:
    internal_passage = generate_internal_knowledge(query)
    sources = [{"text": internal_passage, "source": "internal", "trust": "model_prior"}]
    sources += [
        {"text": p, "source": "external", "trust": "retrieval"}
        for p in external_passages
    ]
    return sources
```

**Why tag the source at all?** Because the consolidation step treats "internal" and "external" as two potentially unreliable sources of the *same kind* — neither is automatically trusted. This is different from a rerank-then-trust pipeline (like Corrective RAG's evaluator, which only judges *external* passages) — here the model's own prior is explicitly put in the same arena as retrieved text, so it can catch cases where retrieval is wrong AND cases where the model's own memory is wrong, by cross-checking one against the other.

</details>

---

## Q3. How does the iterative source-aware consolidation step actually merge and resolve conflicts? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Consolidation runs as a small loop over the combined (internal + external) passage set, repeatedly grouping and re-grouping until the groups stabilize.

```python
def consolidate_sources(query: str, sources: list[dict], max_rounds: int = 3) -> dict:
    """
    sources: [{"text": ..., "source": "internal"|"external", ...}, ...]
    Returns consolidated groups: agreed facts, and unresolved conflicts.
    """
    passages_block = "\n".join(
        f"[{i}] (source={s['source']}) {s['text']}" for i, s in enumerate(sources)
    )

    prompt = f"""You are reconciling multiple sources of information to answer a question.
Some sources may be outdated, irrelevant, or contradictory — including your own
internal-knowledge source.

Question: {query}

Sources:
{passages_block}

Step 1 — Group passages that AGREE with each other (regardless of source),
even if only partially.
Step 2 — Identify passages that CONFLICT with the agreed group, or with each
other, and explain the nature of the conflict.
Step 3 — For each conflict, decide which side is more likely correct based on:
  (a) how many independent sources support each side,
  (b) whether external and internal sources corroborate each other,
  (c) specificity/recency cues in the text itself.

Output as JSON: {{"agreed_facts": [...], "conflicts": [{{"claim_a":..., "claim_b":...,
"resolution": "claim_a" | "claim_b" | "unresolved", "reason": "..."}}]}}"""

    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text  # parse as JSON

def astute_rag(query: str, external_passages: list[str]) -> dict:
    sources = build_tagged_sources(query, external_passages)
    consolidated = consolidate_sources(query, sources)
    return consolidated
```

**Why iterative, not one-shot?** A single grouping pass can be fooled by a majority of near-duplicate but wrong passages (e.g., three retrieved chunks all copied from the same outdated web page). Repeating consolidation lets the model re-evaluate group membership after conflicts are surfaced — a passage initially placed in the "agreed" group can be demoted once a contradicting, better-corroborated group emerges in a later round.

**Resolution heuristics used in practice:**

| Signal | Interpretation |
|---|---|
| Internal + external sources agree | High confidence — corroborated across independent origins |
| Multiple external passages agree, internal disagrees | Trust external majority (retrieval likely reflects current facts model wasn't trained on) |
| Internal knowledge agrees with itself but no external passage supports it | Lower confidence — retrieval may be irrelevant, but don't discard outright |
| Passages conflict with no majority either way | Mark as unresolved conflict — surface to the final answer as a hedge |

</details>

---

## Q4. How does Astute RAG produce a source-attributed final answer, and what happens on unresolved conflicts? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The final generation step consumes the consolidated groups (not the raw passages) and is explicitly instructed to attribute claims and to hedge/abstain where consolidation left a conflict unresolved.

```python
def generate_final_answer(query: str, consolidated: dict) -> str:
    prompt = f"""Answer the question using the reconciled information below.
For each claim, note whether it is well-supported (agreed across sources)
or contested. If a key fact is contested and cannot be resolved, say so
explicitly rather than guessing.

Question: {query}

Agreed facts:
{consolidated['agreed_facts']}

Unresolved conflicts:
{[c for c in consolidated['conflicts'] if c['resolution'] == 'unresolved']}

Answer:"""

    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text
```

**Example behavior on a genuine conflict:**

```
Query: "What is the current CEO of Company X?"

Consolidated:
  agreed_facts: ["Company X is a publicly traded software firm."]
  conflicts:
    - claim_a: "CEO is Jane Smith" (internal knowledge, 2019 training cutoff)
    - claim_b: "CEO is John Doe" (2 external passages, dated 2024)
    - resolution: claim_b  (external majority + recency cues)

Final answer: "As of the most recent available information, the CEO of
Company X is John Doe. (Note: this reflects 2024 sources; earlier
information suggested Jane Smith, but that appears outdated.)"
```

**If resolution is truly ambiguous** (e.g., two external sources disagree with no recency or majority signal), the answer hedges explicitly ("Sources disagree on X; I cannot confirm which is correct") rather than picking one side arbitrarily — this is the abstention behavior that keeps Astute RAG from ever performing worse than the no-retrieval baseline.

</details>

---

## Q5. When does injecting the LLM's own (possibly wrong) parametric knowledge as a "source" backfire, and how would you guard against it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Astute RAG's core bet is that cross-checking internal vs. external knowledge catches more errors than it introduces. But there are failure modes where the internal source actively pollutes consolidation:

**Failure mode 1 — Confidently wrong parametric knowledge outvotes correct-but-sparse retrieval:**

```
Query: about a fact the model memorized incorrectly during pretraining
       (e.g., a common misconception repeated across the web)
External: 1 correct, low-salience passage
Internal: confidently wrong, matches the "popular myth"

Risk: consolidation may treat internal knowledge as corroborating a
similarly-wrong retrieved passage (if one exists), reinforcing the error
rather than catching it.
```

**Failure mode 2 — Internal knowledge on entities/events the model has never seen:**

If the query concerns something entirely outside the model's training data (e.g., an internal company doc, a very recent event), the "internal knowledge" step doesn't yield a genuine null result — the model is prompted to "provide your best recollection, even if not fully certain," which can produce a fluent, plausible-sounding fabrication that then gets treated as a real source in consolidation. This is the same underlying risk described generically as hallucination despite context — except here the hallucination originates from the internal-knowledge elicitation step rather than from misreading a retrieved passage.

**Mitigations:**

| Guard | Effect |
|---|---|
| Confidence-gate the internal source | Only include internal knowledge as a source if the model expresses calibrated confidence above a threshold |
| Down-weight internal source for volatile/time-sensitive queries | Detect query intent (e.g., "current," "latest," "as of") and structurally favor external sources for those |
| Cap internal source influence in consolidation | Never let the internal source alone break a tie among agreeing external passages |
| Combine with a separate factuality/hallucination check | Run the internal-knowledge passage through the same entailment-style verification used in Verifiable RAG before it's allowed to "vote" |

**Contrast with Corrective RAG:** Corrective RAG only ever scores/filters *external* retrieved passages (good/ambiguous/bad) and falls back to web search — it never introduces the model's own unretrieved prior as a competing source. Astute RAG's design is strictly more aggressive: it assumes retrieval AND the model's parametric memory can both be wrong, and resolves between them, which is more powerful in the imperfect-retrieval regime but structurally more exposed to the internal-knowledge-hallucination failure mode above if the confidence-gating isn't done carefully.

</details>

---

## Real-World Applications

- **Enterprise Q&A over frequently-changing knowledge bases**: guards against the LLM over-trusting stale cached/retrieved documentation when its own training data (or a more recent passage) actually has the current answer
- **Fact-checking / misinformation-resistant assistants**: explicitly designed to remain robust when a portion of retrieved evidence is misleading or adversarially poisoned
- **Search-augmented chat assistants** (Google/Gemini-style grounded search): reconciling model prior knowledge with live search snippets before answering
- **Regulatory/compliance QA**: flagging and surfacing genuine source conflicts (e.g., conflicting policy versions) instead of silently picking one
- **Multi-source enterprise search**: reconciling conflicting answers across multiple internal document repositories (wikis, tickets, policy PDFs) that may disagree with each other
