# 43 — Deep Research / Agentic Research RAG

> An orchestrator plans a research project, dispatches multiple search-read-synthesize loops (often across parallel sub-agents), and assembles the results into a long-form, multi-source cited report — trading single-turn latency for report-scale depth.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Research Brief (user's question/topic)
    │
    ▼
Lead/Orchestrator Agent ──► decomposes into sub-questions, builds a research plan
    │
    ▼
Sub-Agent Dispatch (parallel workers, one per sub-question)
    │
    ├──► Sub-Agent 1: search → read pages → extract findings ──┐
    ├──► Sub-Agent 2: search → read pages → extract findings ──┤
    └──► Sub-Agent N: search → read pages → extract findings ──┘
    │
    ▼
Findings Aggregator (merges sub-agent outputs, dedupes sources, tracks citations)
    │
    ▼
Cost/Latency Budget Monitor ──► loop back to dispatch more sub-agents if budget remains
    │                            and coverage gaps are detected
    ▼
Report Synthesizer (LLM writes long-form report, section by section, with inline citations)
    │
    ▼
Citation Aggregation & Formatting (dozens of sources → numbered bibliography)
    │
    ▼
Final Multi-Page Cited Report
```

### Key Components

| Component | Responsibility |
|---|---|
| Lead/Orchestrator Agent | Turns the research brief into a plan, decomposes it into parallelizable sub-questions |
| Research Sub-Agents | Each investigates one sub-question independently: search → fetch → read → extract findings |
| Findings Aggregator | Merges sub-agent outputs, deduplicates overlapping sources, tracks provenance per finding |
| Budget Monitor | Tracks token/dollar/time spend against a report budget; decides whether to spawn more sub-agents or wrap up |
| Report Synthesizer | Produces the final long-form report, weaving findings into sections with inline citations |
| Citation Aggregator | Collects citations from every sub-agent into a single deduplicated, numbered bibliography |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Commercial products | OpenAI Deep Research (ChatGPT), Gemini Deep Research (Google), Perplexity Labs |
| Open-source frameworks | LangChain `open_deep_research`, Hugging Face smolagents `open_deep_research` |
| Multi-agent orchestration | LangGraph (supervisor/worker graphs), CrewAI, AutoGen |
| Search/fetch backend | Tavily, Bing Search API, native web search tool use (Anthropic/OpenAI) |
| Evaluation | GAIA benchmark (general assistant tasks), DeepSearchQA (Google) |

---

## Q1. What is Deep Research and how does it differ from Agentic Web RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Deep Research** is an agentic pattern that produces a long-form, multi-source, cited *report* on a topic by running many search→read→synthesize cycles — often through parallel sub-agents — over a budget of minutes (sometimes tens of minutes) and dollars, rather than a single request-response turn. OpenAI's Deep Research (launched Feb 2025) and Gemini Deep Research are the flagship commercial examples; open-source implementations include LangChain's `open_deep_research` and Hugging Face smolagents' `open_deep_research`.

**Agentic Web RAG** (file 31, Perplexity-style) is the single-turn sibling: one query planner, one or a few web searches, one synthesized answer with inline citations — designed to return in seconds.

```
Agentic Web RAG (file 31):
  "What's the current inflation rate in the US?"
  → 1-3 searches → fetch top pages → single-paragraph cited answer
  Latency: seconds. Sources: ~3-10. Output: a paragraph.

Deep Research (this file):
  "Analyze the competitive landscape of GLP-1 weight-loss drugs and
   project market share shifts over the next 3 years."
  → decompose into ~5-10 sub-questions (market size, competitors, pipeline
    drugs, regulatory landscape, pricing trends...)
  → dispatch parallel sub-agents, each running its OWN multi-step search loop
  → aggregate findings, resolve conflicting sources, cite everything
  → synthesize a multi-page report with a bibliography
  Latency: minutes (5-30+). Sources: dozens. Output: a structured report.
```

| Dimension | Agentic Web RAG (file 31) | Deep Research |
|---|---|---|
| Output shape | Single answer/paragraph | Multi-section long-form report |
| Time budget | Seconds | Minutes to tens of minutes |
| Cost per query | Cents | Can run into dollars per report |
| Search depth | 1-3 searches, single agent | Dozens of searches across many sub-agents |
| Orchestration | Single query planner | Lead agent + parallel research sub-agents |
| Best for | Quick factual/current-events lookups | Market research, literature reviews, due diligence |

**Key insight:** Deep Research is not "Agentic Web RAG but with more searches" bolted onto the same loop — it introduces a *planning and decomposition layer* plus *sub-agent parallelism* that Agentic Web RAG's single-turn design doesn't need.

</details>

---

## Q2. How does the sub-agent / worker orchestration pattern work? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A **lead (orchestrator) agent** decomposes the research brief into independent sub-questions, then spawns one **worker sub-agent per sub-question**, running them in parallel to fit within a wall-clock time budget. Each worker runs its own inner search→read→extract loop, similar in shape to file 04's Agentic RAG loop, but scoped to a narrow sub-question.

```python
from concurrent.futures import ThreadPoolExecutor

def lead_agent_plan(research_brief: str, llm) -> list[str]:
    """Decompose the brief into independent, parallelizable sub-questions."""
    prompt = f"""Break this research task into 4-8 independent sub-questions
that can be researched in parallel. Each should be answerable without
needing the answer to another sub-question.

Research brief: {research_brief}

Return a JSON list of sub-questions."""
    return llm.generate_json(prompt)

def research_subagent(sub_question: str, search_tool, llm, max_steps: int = 6) -> dict:
    """One worker's own search -> read -> extract loop, scoped to one sub-question."""
    findings, sources = [], []
    query = sub_question
    for step in range(max_steps):
        results = search_tool.search(query, k=5)
        for r in results:
            page_text = fetch_and_clean(r["url"])
            extracted = llm.generate(
                f"Extract facts relevant to '{sub_question}' from:\n{page_text}"
            )
            findings.append(extracted)
            sources.append(r["url"])
        # Sub-agent decides whether it has enough or needs a refined query
        next_query = llm.generate(
            f"Given findings so far: {findings}\nDo you need another search? "
            f"If so, what query? If not, respond DONE."
        )
        if next_query.strip() == "DONE":
            break
        query = next_query
    return {"sub_question": sub_question, "findings": findings, "sources": sources}

def deep_research(research_brief: str, search_tool, llm) -> dict:
    sub_questions = lead_agent_plan(research_brief, llm)
    with ThreadPoolExecutor(max_workers=len(sub_questions)) as pool:
        results = list(pool.map(
            lambda q: research_subagent(q, search_tool, llm), sub_questions
        ))
    return {"sub_question_results": results}
```

**Why parallelize across sub-agents rather than one long sequential loop?**

- Wall-clock latency: 8 sub-questions run concurrently in ~1 sub-agent's time budget instead of 8x that time sequentially
- Context isolation: each sub-agent's context window stays focused on one sub-question, avoiding the "lost in the middle" degradation of stuffing everything into one giant agentic loop (relevant to file 10, Long-Context RAG, tradeoffs)
- Failure isolation: one sub-agent hitting a dead end (paywalled sources, no results) doesn't block the others

</details>

---

## Q3. How is the final report synthesized and how are citations aggregated across dozens of sources? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

After sub-agents return, the orchestrator must merge findings that may be redundant, contradictory, or differently sourced, then write a coherent report where every claim still traces back to a specific source.

```python
def aggregate_citations(sub_agent_results: list[dict]) -> dict:
    """Dedupe sources across all sub-agents, assign a single global citation ID."""
    url_to_id = {}
    next_id = 1
    for result in sub_agent_results:
        for url in result["sources"]:
            if url not in url_to_id:
                url_to_id[url] = next_id
                next_id += 1
    return url_to_id

def synthesize_report(research_brief: str, sub_agent_results: list[dict], llm) -> str:
    citation_map = aggregate_citations(sub_agent_results)

    findings_block = ""
    for result in sub_agent_results:
        findings_block += f"\n## {result['sub_question']}\n"
        for finding, src in zip(result["findings"], result["sources"]):
            cid = citation_map[src]
            findings_block += f"- {finding} [{cid}]\n"

    bibliography = "\n".join(
        f"[{cid}] {url}" for url, cid in sorted(citation_map.items(), key=lambda x: x[1])
    )

    prompt = f"""Write a structured, multi-section report answering this research
brief, using ONLY the findings below. Preserve every citation marker [N] next
to the claim it supports. Resolve any contradicting findings by noting the
disagreement explicitly rather than silently picking one.

Research brief: {research_brief}

Findings (grouped by sub-question):
{findings_block}
"""
    report_body = llm.generate(prompt, max_tokens=4000)
    return f"{report_body}\n\n## Sources\n{bibliography}"
```

**Handling contradictions across sub-agents:** unlike single-turn RAG where one retrieval set is used once, Deep Research routinely surfaces conflicting numbers from different sources (e.g. two market-size estimates that differ by 2x). The synthesizer is explicitly prompted to surface disagreement ("Source A estimates X; Source B estimates Y") rather than silently averaging or picking one — this is a distinct failure mode from standard RAG's single-hop citation problem (file 33, Verifiable/Citation RAG) because the *inputs themselves* disagree, not just the model's grounding of them.

</details>

---

## Q4. How do you budget cost and latency across dozens of searches per report? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A Deep Research run can spawn dozens of searches, page fetches, and LLM calls — each sub-agent step costs tokens (LLM calls) and possibly a paid search API call. Left unbounded, a single report can cost several dollars and take much longer than a user will tolerate. Production systems budget explicitly:

```python
class ResearchBudget:
    def __init__(self, max_dollars: float = 2.00, max_seconds: int = 900,
                 max_subagents: int = 8, max_searches_per_subagent: int = 6):
        self.max_dollars = max_dollars
        self.max_seconds = max_seconds
        self.max_subagents = max_subagents
        self.max_searches_per_subagent = max_searches_per_subagent
        self.spent_dollars = 0.0
        self.start_time = time.time()

    def can_continue(self) -> bool:
        elapsed = time.time() - self.start_time
        return self.spent_dollars < self.max_dollars and elapsed < self.max_seconds

    def charge(self, llm_call_cost: float):
        self.spent_dollars += llm_call_cost
```

**Budget levers reported by real products:**

| Lever | Example |
|---|---|
| Query tiers | OpenAI ChatGPT Pro: 250 deep-research queries/month, half "lightweight" (cheaper, faster, shallower); Plus/Team: 25/month |
| Time cap | OpenAI Deep Research browses for roughly 5-30 minutes per report |
| Sub-agent cap | Limit the lead agent's plan to N sub-questions regardless of how many it would ideally want |
| Early termination | If the budget monitor detects the marginal new finding rate has dropped (most new searches return already-seen facts), stop dispatching new sub-agents even if budget remains |
| Model tiering | Use a cheaper/faster model for sub-agent search-and-extract steps, reserve the most capable model for final synthesis |

**The core trade-off:** more sub-agents and more searches generally improve coverage and reduce the risk of missing a key source, but cost and latency scale roughly linearly with sub-agent count — production systems must cap this well before "complete" coverage, accepting some recall loss for bounded cost.

</details>

---

## Q5. What failure modes are unique to Deep Research, and how would you combine it with other bank architectures to mitigate them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Failure mode 1 — citation dilution across dozens of sources.** With single-turn RAG (file 33), verifying 3-10 citations against their source passages is tractable. At report scale with 30-80 sources, exhaustive per-claim NLI verification becomes expensive and slow — teams often verify only a sample or rely on the synthesizer's self-reported citation placement, which reintroduces the "hallucinated citation" risk that file 33 was built to solve. Mitigation: run file 33's attribution-verification step, but only on a report's *load-bearing claims* (numbers, direct quotes, contested statements) rather than every sentence, to keep verification cost bounded.

**Failure mode 2 — redundant/wasted sub-agent work.** Sub-agents dispatched in parallel don't see each other's findings mid-flight, so two sub-agents can independently research overlapping territory, burning budget without adding coverage. Mitigation: a lighter-weight coordination pass (sub-agents periodically report a one-line status back to the lead agent, which can redirect an idle sub-agent to an uncovered angle) — this pushes Deep Research toward the same dynamic re-planning pattern used in Adaptive RAG (file 11), but applied at the sub-agent-plan level instead of the single-query level.

**Failure mode 3 — stale or low-quality source over-reliance.** Because Deep Research optimizes for coverage within a budget, sub-agents under time pressure may accept the first few search results rather than critically filtering for authoritative sources, especially for fast-moving topics. Mitigation: add a lightweight source-quality filter (domain reputation, publication date recency) as a gate before a sub-agent's findings are handed to the aggregator — analogous to the freshness handling in Streaming/Real-Time RAG (file 35).

**Combining with other architectures:**

- **+ Verifiable Citation RAG (file 33):** sampled attribution verification on load-bearing claims only, to keep report-scale citation checking affordable
- **+ Adaptive RAG (file 11):** dynamic re-planning of the sub-question set mid-run based on early sub-agent findings, instead of a static upfront plan
- **+ Search-R1-style RL search (file 42):** replace each sub-agent's prompted search loop with an RL-trained search policy, reducing wasted/redundant searches per sub-agent since the retrieval decisions are learned rather than heuristically prompted

</details>

---

## Real-World Applications

- **OpenAI Deep Research** (ChatGPT, launched Feb 2025): autonomously browses the web for roughly 5-30 minutes to produce analyst-level cited reports for finance, science, policy, and engineering research
- **Gemini Deep Research** (Google): produces comprehensive multi-source reports, now integrated with Workspace content (Gmail, Chat, Drive) and offered via an Interactions API for developers
- **LangChain `open_deep_research`**: open-source, model-agnostic deep research agent supporting configurable search tools and MCP servers
- **Hugging Face smolagents `open_deep_research`**: open replication effort benchmarked on GAIA (general AI assistant tasks), reporting ~55% pass@1 versus ~67% for OpenAI's original
- **Enterprise due-diligence and market research**: competitive landscape analysis, literature reviews, and regulatory research where a single cited answer is insufficient and a structured multi-source report is the deliverable
