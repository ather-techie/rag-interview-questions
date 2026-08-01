# Agentic Orchestration: Designing RAG Agent Loops

> How to structure the control loop that drives retrieval decisions, when to stop iterating, and how to avoid runaway agents in production.

---

## What is Agentic Orchestration?

Agentic orchestration is the control logic that lets a language model decide *when* to retrieve, *what* to retrieve, *how many times* to repeat the retrieval loop, and *when to stop*. Whereas static RAG retrieves once before generation, an agentic loop treats retrieval as one tool among many — the model calls it, inspects the result, and decides whether to call it again (possibly with a different query), call a different tool, or produce the final answer.

The three canonical patterns are:

| Pattern | Control Flow | Best For |
|---------|-------------|---------|
| **ReAct** | Thought → Action → Observation → Thought → ... | Single-session QA; transparent reasoning |
| **Plan-and-Execute** | Plan all steps first → Execute independently | Long multi-step tasks; parallel sub-tasks |
| **Reflexion** | Generate → Evaluate → Revise → Generate | Tasks with a verifiable correctness criterion |

---

## Why Traditional (Pipeline) RAG Isn't Enough

### Worked Example: A Single-Pass Walkthrough

Query: *"What's covered under my auto policy's collision coverage?"*

1. The query is embedded → a single vector search runs against the policy-docs index.
2. The top-k chunks (say, top 5) come back based on similarity alone.
3. Those chunks are stuffed into the LLM's context along with the query.
4. The LLM generates one answer in one pass — there's no step where it checks whether those 5 chunks actually settle the question, and no way to retrieve again if they don't.

That single fixed pass is exactly what breaks down on the compound flood/water-damage query below — the pipeline has no mechanism to notice when one pass isn't enough.

A fixed retrieve-then-generate pipeline struggles with:

- **Single retrieval pass** — one fixed retrieve-then-generate step, with no ability to try again if the first pull misses.
- **No planning** — can't break a multi-part goal into ordered sub-steps before acting.
- **No tool selection** — always queries the same source, even when a query needs SQL, an API, or the web instead of the vector store.
- **No verification** — nothing checks whether the retrieved chunks actually support the answer before it's generated.
- **No iterative refinement** — if retrieval comes back thin or off-target, there's no mechanism to notice and re-query.
- **Difficult multi-hop reasoning** — questions that require chaining facts across multiple documents or sources don't fit a single retrieval pass.

Agentic orchestration solves these by wrapping retrieval in a control loop that can decide, verify, and adapt instead of executing one fixed pass.

## Query Rewriting

Before a query is decomposed or routed, it often needs to be *rewritten* into something retrieval can actually work with — this is a distinct, earlier step from Query Decomposition (which splits an already-clear question into independent sub-questions; see the flood/water-damage worked example below).

- User asks: *"y is my app slow after last update"* → rewritten to *"What are the common causes of application performance degradation after a software update?"* — expanding shorthand into a retrievable, well-formed question.
- User asks a follow-up with a pronoun: *"How do I fix it?"* (after an earlier question about OOM errors) → rewritten using conversation context to *"How do I fix an out-of-memory (OOM) error in a Kubernetes pod?"* — without resolving the pronoun, a bare "how do I fix it" embeds to something nearly meaningless.

Query Rewriting fixes an ill-formed single question; Query Decomposition splits a well-formed compound question into parts. A production system typically applies rewriting first, then decomposition if the rewritten query still spans multiple sub-questions.

## LLM vs. RAG vs. Agentic RAG

| Capability | LLM | RAG | Agentic RAG |
|---|---|---|---|
| Uses external knowledge | No | Yes | Yes |
| Planning | No | No | Yes |
| Tool calling | No | No | Yes |
| Multi-step reasoning | Limited | No | Yes |
| Reflection | No | No | Yes |
| SQL/API support | No | No (vector-only) | Yes |
| Dynamic retrieval | No | Single pass | Yes, iterative |

## RAG Types at a Glance

| Type | Description | Where it's exampled here |
|---|---|---|
| **Single-Agent RAG** | One centralized agent finds, directs, and compiles information on its own — efficient for a narrow domain and a single data source | [`04-agentic-rag.md`](../02_interview_bank/04-agentic-rag.md) Q1 |
| **Multi-Agent RAG** | A Manager/Orchestrator Agent delegates sub-tasks to specialist agents (retriever, verifier, etc.) and synthesizes their results | Worked Example below; [`04-agentic-rag.md`](../02_interview_bank/04-agentic-rag.md) Q1/Q8 |
| **Routing Agent** | Classifies query intent and picks the right tool or data source before any retrieval happens | Automatic Router worked example below |
| **Query Planning Agent** | Decomposes a complex, multi-part question into independent sub-queries, often run in parallel | Worked Example below (step 2) |
| **ReAct Agent** | Interleaves reasoning and retrieval in a serial loop — think, act, observe, repeat | ReAct section below |
| **Reflection & Correction Agent** | Checks retrieved evidence or a draft answer for relevance/hallucination and triggers a re-retrieval if it fails | Worked Example below (step 4); Interview Q&A below |
| **A-RAG (Adaptive/Hierarchical RAG)** | Reviews a brief summary first and only fetches the full, token-heavy chunk if the summary proves insufficient | [`11-adaptive-rag.md`](../02_interview_bank/11-adaptive-rag.md), [`04-agentic-rag.md`](../02_interview_bank/04-agentic-rag.md) Terminology Note |

## Agentic RAG Architecture: The Full Pipeline

Beyond the ReAct/Plan-and-Execute loop mechanics below, production agentic RAG systems are usually described as an outer control loop wrapped around the classic retrieve-then-generate pipeline:

```
User Query → Router → Query Planner → Retriever Agent(s) → Reranker →
             Reflection/Verification Agent → Manager/Orchestrator →
             Generation LLM → Response
```

Memory and the Guardrails/Safety layer are available to every stage rather than sitting at one point in the chain, and Observability/Tracing wraps the entire pipeline end-to-end so each decision is logged. **Not every deployment needs the full stack** — a narrow, single-source use case may only need Router + Retriever + Generator, while regulated, multi-source use cases (e.g., insurance policy Q&A spanning auto + homeowner's + claims data) benefit from the full stack.

| Component | Role | See also |
|---|---|---|
| Router / Routing Agent | Classifies query intent and picks the right tool or data source | [`04_patterns/01-router-fallback.md`](../04_patterns/01-router-fallback.md) |
| Query Planner (Decomposition Agent) | Breaks a complex, multi-part question into independent sub-queries | [`04_patterns/02-fan-out-fan-in.md`](../04_patterns/02-fan-out-fan-in.md) |
| Retriever Agent(s) | Executes similarity/hybrid search against the vector store (and other data sources) per sub-query | [`retrieval_strategies.md`](./retrieval_strategies.md) |
| Vector Store / Hybrid Index | Stores and indexes embeddings for fast similarity search | [`vector_databases.md`](./vector_databases.md) |
| Reranker | Re-scores retrieved chunks for relevance before they reach the LLM | [`reranking.md`](./reranking.md) |
| Reflection / Verification Agent | Checks retrieved evidence and draft answers for relevance, completeness, or hallucination | Q&A below |
| Manager / Orchestrator Agent | Coordinates sub-agents, resolves conflicting sub-answers, and synthesizes the final response | Q&A below |
| Memory (short-term + long-term) | Short-term holds session state so follow-ups don't re-retrieve everything; long-term persists across sessions | [`conversational_memory_architecture.md`](./conversational_memory_architecture.md) |
| Tool / Action Layer | Lets agents call external APIs, SQL databases, or web search beyond the vector store | — |
| Guardrails / Safety Layer | Validates inputs/outputs and enforces policy (e.g., PII redaction, scope limits) across every stage | [`prompt_injection_risks.md`](./prompt_injection_risks.md) |
| Generation / Synthesis LLM | Produces the final natural-language answer grounded in verified retrieved context | — |
| Observability / Tracing | Logs every agent decision, tool call, and retrieved chunk end-to-end | [`observability_and_evaluation_ops.md`](./observability_and_evaluation_ops.md) |

### Automatic Router: Worked Example

The Router / Routing Agent classifies intent *before* any retrieval happens, so a query never hits the wrong data source:

- *"What's our current cloud spend for Q2?"* → detected as structured/numeric → routes to a **SQL database**, not the vector store.
- *"What does our data retention policy say about backups?"* → detected as a policy/document lookup → routes to the **vector database** (semantic search over policy docs).
- *"What's the latest news on the Fed's interest rate decision?"* → detected as needing current/real-time info not in the knowledge base → routes to a **web search tool**.
- *"Summarize the attached contract."* → a document is provided directly → the router **skips retrieval entirely**, routing straight to the LLM with the doc as context.

A single always-vector-store pipeline would try to answer the cloud-spend question by embedding it and searching policy docs — no amount of reranking fixes a query routed to the wrong source entirely.

### Worked Example: A Compound, Multi-Source Query

Query: *"My basement flooded during a storm and my car parked in the driveway got water damage — am I covered, and what's my payout cap?"*

1. **Routing Agent** recognizes this is a compound question spanning two policy types (homeowner's + auto) and routes accordingly, rather than treating it as one vector search.
2. **Query Planning Agent** decomposes it into subqueries:
   - "Is storm/flood damage to a car covered under comprehensive auto insurance?"
   - "Does homeowner's insurance cover basement flooding from a storm?"
   - "What are the payout caps for each?"
3. **Retriever Agent(s)** run each subquery against the relevant policy documents (auto policy doc, homeowner's policy doc) — possibly in parallel.
4. **Reflection/Verification Agent** checks the retrieved clauses actually answer the subqueries — e.g., it catches that "flood" is often *excluded* from standard homeowner's policies and needs separate flood insurance, a detail a single-shot RAG might miss or conflate.
5. **Manager Agent** synthesizes: *"Your car's water damage is likely covered under comprehensive auto (cap: $X), but basement flooding from a storm is typically NOT covered under standard homeowner's insurance — you'd need separate flood coverage. Let me know if you have a flood policy to check."*

A single-shot pipeline RAG would embed the whole question once, retrieve against a single index, and risk conflating "water damage" across both policy types — likely missing the flood exclusion entirely, since it never issues a targeted subquery for it and never verifies the retrieved clause actually addresses flood vs. general water damage.

## Frameworks & Tooling

| Framework | Best For |
|---|---|
| LangChain | General orchestration |
| LangGraph | Graph-based state machine for cyclic, multi-agent workflows; built on LangChain |
| LlamaIndex | RAG pipelines |
| Microsoft Agent Framework (MAF) | Enterprise multi-agent systems (Microsoft/Azure ecosystem) |
| CrewAI | Role-based agent workflows |

**Microsoft's agent stack has consolidated.** AutoGen (research-driven multi-agent conversation patterns) and Semantic Kernel (production-grade enterprise SDK) were Microsoft's two separate, competing agent frameworks. In October 2025 Microsoft merged them into the **Microsoft Agent Framework (MAF)** — built by the same teams, combining AutoGen's multi-agent orchestration patterns with Semantic Kernel's enterprise features (state management, type safety, telemetry, broad model support). MAF reached a production-ready 1.0 release in 2026 and is the framework to reach for in a Microsoft/Azure shop today; AutoGen and Semantic Kernel are its predecessors, not independent current choices, and existing projects on either are expected to migrate to MAF.

None of these frameworks are required to build an agentic RAG loop — the ReAct/Plan-and-Execute implementations below use the raw Anthropic API directly — but they provide reusable scaffolding (tool registries, agent-to-agent messaging, state persistence) once a system grows past a handful of hand-rolled agents.

---

## Model Context Protocol (MCP)

MCP is an open protocol, originally introduced by Anthropic in November 2024, for standardizing how LLM applications connect to external tools, data sources, and prompt templates. Before MCP, every framework (LangChain, LlamaIndex, a hand-rolled ReAct loop) wired up its own bespoke integration for each external system — a SQL database, a ticketing API, a vector store each needed custom glue code, rewritten per framework. MCP replaces that N×M integration problem with a single client-server contract.

**Architecture:**

- A **Host** is the application the user interacts with (Claude Desktop, an IDE, a custom agent).
- A **Client** lives inside the host and manages a 1:1 connection to one MCP **Server**.
- A **Server** is an external program that exposes three primitives over a standard JSON-RPC interface: **Tools** (model-invokable functions, e.g. `retrieve`, `run_sql`), **Resources** (contextual data the client can read, analogous to a GET endpoint), and **Prompts** (reusable, user-selected prompt templates). Servers can run locally (subprocess, stdio) or remotely (Streamable HTTP).

```
┌────────────────────────┐        ┌───────────────────────────┐
│  Host (agent / IDE)    │        │  MCP Server                │
│   ┌──────────────┐     │  JSON- │   Tools: retrieve, run_sql │
│   │ MCP Client    │◄───┼─ RPC ─►│   Resources: doc_index/*   │
│   └──────────────┘     │        │   Prompts: templates       │
└────────────────────────┘        └───────────────────────────┘
```

**Why it matters for agentic RAG specifically:** the Retriever/Tool layer in the pipeline table above (Router, Retriever Agent(s), Tool/Action Layer) is exactly the surface MCP standardizes. Instead of writing a bespoke `retrieve()` function per vector store and a bespoke `run_sql()` per database, you stand up one MCP server per data source once, and *any* MCP-compatible agent — whether it's a hand-rolled ReAct loop, a LangGraph graph, or a Microsoft Agent Framework agent — can call it without new integration code. Swapping or adding a data source becomes a config change (point the client at a new server) rather than a code change in every framework you use.

**Relationship to the frameworks above:** MCP is not a competitor to LangGraph, MAF, LlamaIndex, or CrewAI — it's a wire protocol those frameworks consume. LangGraph and LangChain expose adapters that wrap MCP servers' tools as ordinary LangChain tools; Microsoft Agent Framework ships native MCP client support as part of its tool-calling story; and the raw-Anthropic-API ReAct loop earlier in this file could equally dispatch its `retrieve`/`web_search` tools to MCP servers instead of in-process functions, with no change to the agent loop itself. Adoption has moved fast since the November 2024 introduction: OpenAI adopted MCP across ChatGPT, its Agents SDK, and the Responses API in March 2025; Google DeepMind confirmed support in Gemini shortly after; and by late 2025 the protocol had moved to Linux Foundation governance with tens of thousands of production MCP servers — making it a reasonable default assumption for how a new agentic RAG system should expose its tools, rather than a niche or Anthropic-only convention.

---

## ReAct: The Baseline Agentic Pattern

ReAct (Reason + Act) interleaves natural language reasoning steps with tool calls. Each cycle is: **Thought** (plan next action) → **Action** (call a tool) → **Observation** (read result) → **Thought** (decide what to do next).

```python
import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "retrieve",
        "description": "Retrieve relevant passages from the knowledge base for a given query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k":     {"type": "integer", "description": "Number of passages to return", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for real-time or recent information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

def run_react_agent(user_query: str, max_iterations: int = 6) -> str:
    messages = [{"role": "user", "content": user_query}]
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        
        # Agent chose to answer directly
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        
        # Agent called a tool
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            
            messages.append({"role": "user", "content": tool_results})
        
    return "Max iterations reached without a final answer."


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "retrieve":
        return retrieve_from_index(inputs["query"], inputs.get("k", 5))
    if name == "web_search":
        return web_search(inputs["query"])
    return f"Unknown tool: {name}"
```

---

## Plan-and-Execute

Plan-and-Execute splits planning from execution. A "planner" LLM call produces a list of sub-tasks; each sub-task is then executed (possibly in parallel) by executor agents. Results are fed to a "synthesizer" LLM call for the final answer.

```
Query: "Compare RAG approaches for legal vs. medical domains"
    │
    ▼
PLANNER (1 LLM call)
  Plan:
    1. retrieve("RAG legal domain challenges")
    2. retrieve("RAG medical domain challenges")
    3. retrieve("domain-specific embedding models comparison")
    4. synthesize results
    │
    ▼ (parallel)
EXECUTORS
  Task 1: retrieve → [legal docs] ──┐
  Task 2: retrieve → [medical docs]─┤
  Task 3: retrieve → [model docs]  ─┤
                                    ▼
                              SYNTHESIZER (1 LLM call)
                                → Final answer
```

```python
import asyncio

async def plan_and_execute(query: str) -> str:
    # Step 1: Generate plan
    plan_response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        system="Generate a JSON list of retrieval sub-tasks needed to answer the query. "
               "Format: [{\"task\": \"...\", \"tool\": \"retrieve\", \"query\": \"...\"}]",
        messages=[{"role": "user", "content": query}],
    )
    tasks = parse_plan(plan_response.content[0].text)
    
    # Step 2: Execute all tasks in parallel
    results = await asyncio.gather(*[
        asyncio.to_thread(dispatch_tool, t["tool"], {"query": t["query"]})
        for t in tasks
    ])
    
    # Step 3: Synthesize
    context = "\n\n".join(f"[Task {i+1}: {t['task']}]\n{r}"
                          for i, (t, r) in enumerate(zip(tasks, results)))
    
    final = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Query: {query}\n\nResearch findings:\n{context}\n\nAnswer the query.",
        }],
    )
    return final.content[0].text
```

---

## Reflexion: Learning from Failed Attempts

Reflexion (Shinn et al., 2023) adds a self-critique loop on top of an attempt-evaluate cycle. When an attempt fails or scores poorly against some verifier, the agent doesn't just try again — it first generates a **verbal reflection**: a natural-language diagnosis, in the model's own words, of *why* the attempt failed. That reflection is appended to an **episodic memory buffer**, and the next attempt is run with the accumulated reflections injected into context, so the agent carries forward what it learned instead of repeating the same mistake blind.

```
Attempt 1 ──► Evaluate ──► FAIL ──► Reflect:
                                     "I searched for 'refund policy' but the
                                      docs use the term 'return window' —
                                      my query never matched anything."
                                          │
                                          ▼ (append to episodic memory buffer)
Attempt 2 (reflection prepended to context) ──► Evaluate ──► PASS ──► Done
```

```python
def reflexion_loop(query: str, max_attempts: int = 3) -> str:
    episodic_memory: list[str] = []

    for attempt in range(max_attempts):
        reflections = "\n".join(f"- {r}" for r in episodic_memory)
        prompt = (
            f"Question: {query}\n"
            + (f"\nLessons from prior failed attempts:\n{reflections}\n" if episodic_memory else "")
        )
        answer, context_used = run_retrieval_and_generate(prompt)

        verdict = evaluate(query, answer, context_used)  # e.g. groundedness/confidence check
        if verdict["passed"]:
            return answer

        # Verbal self-reflection on *why* this attempt failed, not just that it failed
        reflection = generate_reflection(query, answer, context_used, verdict["reason"])
        episodic_memory.append(reflection)

    return "Unable to produce a verified answer after max attempts."
```

**How this differs from retry-with-backoff:** plain retry-with-backoff re-issues the same (or a randomly jittered) request on the assumption that the failure was transient — a rate limit, a flaky network call. It carries no information forward between attempts. Reflexion assumes the failure was *substantive* — the query was wrong, the reasoning was wrong, the tool choice was wrong — and forces the agent to articulate the specific cause before trying again, then carries that diagnosis forward as context. Retry-with-backoff fixes "the same thing might work if I wait"; Reflexion fixes "the same thing will fail again unless I change something specific."

**When it's useful in agentic RAG:** whenever an attempt has a checkable outcome and failure isn't purely transient — e.g. a retrieval pass comes back with low-confidence or off-target chunks, or a generated answer fails the groundedness/sufficiency check described below. Rather than reissuing the identical query (which will return the identical thin results) or giving up after one miss, the agent reflects — "the query used generic phrasing but the index is indexed on domain-specific terminology; rephrase using policy terms" — and retries with that lesson in context. This is a natural extension of the Reflection/Verification Agent already in the architecture table above: verification catches *that* an answer is unsupported, Reflexion is what lets the *next* attempt actually do better instead of repeating the same retrieval.

**Pros/cons:**

| | |
|---|---|
| **Pros** | Turns a hard failure into targeted, self-diagnosed improvement; doesn't require any fine-tuning — reflections live in-context or in a persisted memory store; compounds across attempts within a session |
| **Cons** | Requires a reliable verifier/evaluator to trigger reflection in the first place — garbage-in verdicts produce garbage reflections; adds at least one extra LLM call per failed attempt; can loop indefinitely on tasks with no achievable correct answer unless capped by the same stopping criteria used elsewhere in this file |

---

## Stopping Criteria

Runaway loops are the primary production risk in agentic RAG. Use at least two stopping conditions:

### 1. Hard Iteration Cap

Always set a `max_iterations` limit. For most production RAG use cases, 3–5 iterations is sufficient; 10 is a safe hard cap.

### 2. Semantic Drift Detection

Stop if successive retrieval queries are too similar to previous ones — the agent is looping without making progress.

```python
from sklearn.metrics.pairwise import cosine_similarity

def is_looping(query_history: list[str], current_query: str, threshold: float = 0.95) -> bool:
    if len(query_history) < 2:
        return False
    embeddings = embed(query_history + [current_query])
    current_emb = embeddings[-1].reshape(1, -1)
    past_embs   = embeddings[:-1]
    sims = cosine_similarity(current_emb, past_embs)[0]
    return float(sims.max()) > threshold
```

### 3. Confidence / Completeness Signal

Ask the model to self-assess whether it has enough information before re-retrieving.

```python
SUFFICIENCY_PROMPT = """
Given the user's original question and the retrieved context so far, 
answer with a JSON object:
{"sufficient": true/false, "reason": "...", "missing": "what's still needed"}
"""

def has_sufficient_context(question: str, context: str) -> bool:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap check
        max_tokens=128,
        system=SUFFICIENCY_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\nContext: {context[:2000]}"}],
    )
    result = json.loads(resp.content[0].text)
    return result["sufficient"]
```

### 4. Token Budget Guard

Track total input tokens consumed and stop before hitting context window or cost limits.

```python
class BudgetedAgent:
    def __init__(self, max_input_tokens: int = 50_000):
        self.max_input_tokens = max_input_tokens
        self.tokens_used = 0
    
    def should_stop(self) -> bool:
        return self.tokens_used >= self.max_input_tokens
    
    def record(self, response):
        self.tokens_used += response.usage.input_tokens
```

---

## Stopping Criteria Comparison

| Criterion | Catches | Misses | Overhead |
|-----------|---------|--------|----------|
| Hard iteration cap | Infinite loops | Slow convergence without looping | None |
| Semantic drift | Query repetition | Conceptually similar but distinct queries | Embedding call per iteration |
| Confidence check | "Adequate context" loops | Complex tasks that always look incomplete | LLM call per iteration |
| Token budget | Cost explosions | Time-based SLAs | Token counting |

Use all four together in production.

---

## ReAct vs. Plan-and-Execute vs. Reflexion

| Dimension | ReAct | Plan-and-Execute | Reflexion |
|-----------|-------|------------------|-----------|
| **Latency** | Serial (each hop waits) | Parallel (tasks run concurrently) | High (iterate until correct) |
| **Debuggability** | High (thought trace) | Medium (plan visible but exec parallel) | Medium |
| **Best for** | Unknown scope; exploratory | Known decomposable tasks | Answer-verifiable tasks |
| **Failure mode** | Runaway loops | Bad plan → bad answers (no replanning) | Infinite refinement |
| **Token cost** | Moderate | Low (parallel = less wall-clock, same tokens) | High |

---

## Parallelism in Multi-Tool Calls

Modern LLM APIs (including Anthropic) return multiple `tool_use` blocks in a single response when the model wants to call tools in parallel. Always handle this correctly — execute all tool calls from one response before sending the next user turn.

```python
# WRONG: only executing the first tool call
for block in response.content:
    if block.type == "tool_use":
        result = dispatch_tool(block.name, block.input)
        break  # ← drops parallel calls

# CORRECT: execute all tool calls, return all results in one user turn
tool_results = []
for block in response.content:
    if block.type == "tool_use":
        result = dispatch_tool(block.name, block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })
messages.append({"role": "user", "content": tool_results})
```

---

## Agent State Management

For multi-turn or long-running agents, maintain explicit state outside the message list:

```python
from dataclasses import dataclass, field

@dataclass
class AgentState:
    query: str
    messages: list = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)
    query_history: list[str] = field(default_factory=list)
    iteration: int = 0
    tokens_used: int = 0
    
    def is_terminal(self, max_iter: int = 6, max_tokens: int = 50_000) -> bool:
        return self.iteration >= max_iter or self.tokens_used >= max_tokens
```

Externalizing state enables: checkpointing (resume after crash), observability (trace each iteration), and cost attribution per session.

---

## Production Guardrails Summary

```
┌──────────────────────────────────────────────┐
│          Agent Loop Entry                     │
│                                              │
│  ┌─────────────┐                             │
│  │ Iteration 1 │──► Tool call(s)            │
│  └──────┬──────┘        │                   │
│         │           Observations             │
│         ▼                │                   │
│  ┌─────────────┐         │                   │
│  │ Iteration 2 │◄────────┘                   │
│  └──────┬──────┘                             │
│         │                                    │
│    STOP if ANY:                              │
│    ├─ iteration >= max_iter                  │
│    ├─ semantic drift detected                │
│    ├─ confidence check passes                │
│    └─ token budget exceeded                  │
└──────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **ReAct is the default** for open-ended retrieval; Plan-and-Execute for decomposable tasks.
2. **Always combine multiple stopping criteria** — no single criterion catches all failure modes.
3. **Handle parallel tool calls** correctly; missing them silently breaks multi-tool responses.
4. **Use a cheap model** (Haiku) for sufficiency/routing checks; reserve the large model for generation.
5. **Externalize agent state** for observability, checkpointing, and cost tracking.

---

## Interview Q&A

**Q: What is the difference between ReAct and Plan-and-Execute in agentic RAG?** `[Intermediate]`

ReAct interleaves reasoning and retrieval in a serial loop — each observation informs the next thought. This makes it highly adaptive (it can change its approach mid-loop) but each hop adds latency because you can't parallelize. Plan-and-Execute separates planning from execution: one LLM call produces a complete task list, then all tasks run in parallel. This is faster for tasks with known structure (e.g., "retrieve from N sub-topics") but brittle when the plan is wrong — there's no replanning mid-execution. The heuristic: use ReAct when the query scope is unknown and you need adaptive exploration; use Plan-and-Execute when the task decomposes predictably and latency matters.

---

**Q: How do you prevent agentic RAG from running in infinite loops?** `[Advanced]`

Use four stopping conditions in combination: (1) a hard iteration cap (e.g., max 6 loops) as the last resort; (2) a semantic drift check that detects when the agent is issuing near-duplicate retrieval queries — cosine similarity > 0.95 between current query and any prior query signals looping; (3) a sufficiency check where a cheap model (Haiku) reads the accumulated context and outputs `{"sufficient": true/false}` — if sufficient, exit the loop; (4) a token budget guard that stops before the context window or cost limit is reached. In practice, condition 3 fires most often; conditions 1 and 4 are safety nets for edge cases.

---

**Q: How do you make an agentic RAG system observable in production?** `[Advanced]`

Instrument three things: (1) per-iteration traces — log each (query, tool_name, result_summary, tokens_used, iteration_number) to your observability system (Langfuse, Phoenix, or a custom trace table); (2) terminal reason — record *why* the loop stopped (end_turn, max_iter, drift, budget) so you can distinguish "answered correctly" from "gave up"; (3) user-visible citations — surface which retrieved passages contributed to the final answer. Without traces, debugging a multi-hop agent is nearly impossible because the error may be two or three hops back from the wrong final answer.

---

**Q: How does the full agentic RAG architecture pipeline differ from just running a ReAct loop?** `[Intermediate]`

ReAct describes the *inner* loop of one agent deciding whether to think, act, or answer. The full pipeline (`Router → Query Planner → Retriever Agent(s) → Reranker → Reflection/Verification Agent → Manager/Orchestrator → Generation LLM`) describes the *outer* system that loop runs inside — it adds stages ReAct alone doesn't cover: a router that picks the right data source before any retrieval happens, a planner that decomposes a compound question into independent subqueries up front, a reranker that re-scores what came back, and a manager that reconciles multiple sub-agents' answers into one response. In the insurance flood/water-damage worked example above, the ReAct-style "retrieve → observe → retrieve again" behavior only shows up *inside* the Retriever Agent step — the routing, decomposition, and cross-policy reconciliation happen in stages ReAct itself doesn't model.

---

**Q: Which framework should I pick for a given agentic RAG use case?** `[Intermediate]`

Match the framework to what's actually hard about the use case, not to popularity: LlamaIndex if the core problem is RAG pipeline quality (chunking, indexing, retrieval); LangChain for general-purpose orchestration when you're gluing together many tool types; LangGraph or CrewAI when the design is genuinely multi-agent with distinct roles that need to converse or loop (e.g., a researcher agent and a fact-checker agent) and you want explicit, inspectable control over the cyclic execution graph; Microsoft Agent Framework when the deployment target is already a Microsoft/Azure shop and enterprise integration (identity, governance) matters more than framework novelty — MAF is the current unified successor to AutoGen and Semantic Kernel, which merged into it in October 2025, so new Microsoft-ecosystem projects should target MAF rather than either legacy framework. For a single-agent, single-source system, skip the framework entirely — a hand-rolled loop against the raw LLM API (as shown above) has less overhead and is easier to debug.

---

**Q: Walk through why a single-shot RAG system would get the insurance flood/water-damage question wrong.** `[Advanced]`

Single-shot RAG embeds the entire compound question once — "basement flooded... car... water damage... am I covered" — and runs one similarity search against one index. Three failure modes follow: (1) if auto and homeowner's policies live in the same index, the top-k results are likely to be dominated by whichever policy type has denser/more numerous chunks near that embedding, so the other policy's relevant clause may not make the cut; (2) even if both surface, there's no step that verifies the retrieved "water damage" clause actually addresses *flood* specifically — homeowner's policies typically exclude flood while covering other water damage, a distinction a single pass has no mechanism to check; (3) there's no re-query if the first pull is thin, so an ambiguous or partial match gets passed straight to generation. The agentic version avoids all three by decomposing into policy-specific subqueries, retrieving each independently, and running a verification step that explicitly checks for the flood-exclusion nuance before the Manager Agent synthesizes an answer.

---

**Q: Give a concrete example of a Reflection/Correction agent catching a hallucination before it reaches the user.** `[Intermediate]`

Say the system has just generated an answer citing a support doc for "our refund window is 30 days." A Reflection/Correction agent re-reads the retrieved chunk it just cited alongside its own draft answer and notices the chunk never actually mentions a refund window at all — the number was invented, not retrieved. It flags this as a potential hallucination rather than letting the draft ship, and triggers a re-retrieval (a more targeted query like "refund window policy") before finalizing. This is the same mechanism as the flood-exclusion check in the worked example above, generalized: the agent doesn't just check that *some* chunk was retrieved, it checks that the specific claim in the answer is actually traceable to that chunk's text.
