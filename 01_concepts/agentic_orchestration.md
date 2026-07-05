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

**Q: What is the difference between ReAct and Plan-and-Execute in agentic RAG?**

ReAct interleaves reasoning and retrieval in a serial loop — each observation informs the next thought. This makes it highly adaptive (it can change its approach mid-loop) but each hop adds latency because you can't parallelize. Plan-and-Execute separates planning from execution: one LLM call produces a complete task list, then all tasks run in parallel. This is faster for tasks with known structure (e.g., "retrieve from N sub-topics") but brittle when the plan is wrong — there's no replanning mid-execution. The heuristic: use ReAct when the query scope is unknown and you need adaptive exploration; use Plan-and-Execute when the task decomposes predictably and latency matters.

---

**Q: How do you prevent agentic RAG from running in infinite loops?**

Use four stopping conditions in combination: (1) a hard iteration cap (e.g., max 6 loops) as the last resort; (2) a semantic drift check that detects when the agent is issuing near-duplicate retrieval queries — cosine similarity > 0.95 between current query and any prior query signals looping; (3) a sufficiency check where a cheap model (Haiku) reads the accumulated context and outputs `{"sufficient": true/false}` — if sufficient, exit the loop; (4) a token budget guard that stops before the context window or cost limit is reached. In practice, condition 3 fires most often; conditions 1 and 4 are safety nets for edge cases.

---

**Q: How do you make an agentic RAG system observable in production?**

Instrument three things: (1) per-iteration traces — log each (query, tool_name, result_summary, tokens_used, iteration_number) to your observability system (Langfuse, Phoenix, or a custom trace table); (2) terminal reason — record *why* the loop stopped (end_turn, max_iter, drift, budget) so you can distinguish "answered correctly" from "gave up"; (3) user-visible citations — surface which retrieved passages contributed to the final answer. Without traces, debugging a multi-hop agent is nearly impossible because the error may be two or three hops back from the wrong final answer.
