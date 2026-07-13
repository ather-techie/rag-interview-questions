# Tree of Thought RAG (ToT-RAG)

> Coupling Tree-of-Thought multi-branch reasoning with conditional retrieval at each reasoning node — for queries that require exploring competing hypotheses before committing to an answer.

---

## What is Tree of Thought RAG?

Tree of Thought (ToT) is a reasoning strategy where the LLM generates multiple candidate "thoughts" (partial solutions or hypotheses) at each step, evaluates them, and pursues the most promising branches — like a search tree over reasoning paths. ToT-RAG combines this with conditional retrieval: at each node in the reasoning tree, the agent may retrieve evidence from the knowledge base to evaluate or extend that branch.

The result is a system that can explore competing explanations, retrieve targeted evidence for each, and backtrack when a branch proves unfruitful — capabilities that flat ReAct loops lack.

```
Standard RAG:
  Query ──► Retrieve ──► Generate ──► Answer

ReAct RAG:
  Query ──► Think ──► Retrieve ──► Think ──► Retrieve ──► ... ──► Answer

ToT-RAG:
  Query ──► Generate N thoughts ──► Evaluate each
               │                         │
               ├─ Thought A ──► Retrieve evidence A ──► Score → prune?
               ├─ Thought B ──► Retrieve evidence B ──► Score → prune?
               └─ Thought C ──► Retrieve evidence C ──► Score → extend
                                        │
                                 Generate children of C
                                        │
                                ├─ Thought C1 ──► Retrieve ──► Score
                                └─ Thought C2 ──► Retrieve ──► Score → ANSWER
```

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
  │
  ▼
Thought Generator ──► candidate branches (thought 1, 2, 3, ...)
  │
  ▼
Branch Evaluator/Scorer ──► score each branch
  │
  ├── low score ──► prune
  │
  ▼ (promising branches)
Conditional Retriever ──► fetch evidence targeted at that branch
  │
  ▼
Search Controller (BFS / DFS / beam search)
  │   expands most promising branches, repeats
  │   Thought Generator → Evaluator → Retriever loop
  ▼
Generator ──► synthesizes final answer from best path + evidence
```

### Key Components

| Component | Responsibility |
|---|---|
| Thought Generator | Proposes multiple candidate reasoning branches at each node |
| Branch Evaluator/Scorer | Scores each branch's promise (0–1) and flags final-answer candidates |
| Conditional Retriever | Fetches evidence targeted at a specific branch/hypothesis rather than a global query |
| Search Controller | Expands the tree via BFS, DFS, or beam search, applying a prune threshold |
| Generator | Synthesizes the final answer from the winning path and its accumulated evidence |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Orchestration | Custom ToT controller (prompt-based, no dedicated library required), LangGraph for branch expansion/pruning |
| Retrieval | Any standard retriever / vector DB (Qdrant, Weaviate, Pinecone) |
| Models | Cheap model (Haiku) for thought generation/evaluation, stronger model (Sonnet) for final synthesis |

---

## When ToT-RAG Helps

ToT-RAG is worth the cost when:

1. **Multiple valid approaches exist**: "Should we use a vector DB or a full-text index for this use case?" — requires exploring both paths.
2. **The answer depends on evidence that rules out competing hypotheses**: "What caused the service outage?" — competing hypotheses need targeted evidence retrieval.
3. **Simple linear retrieval keeps finding irrelevant context**: flat RAG keeps retrieving the same top-k; ToT can branch to retrieve from different angles.
4. **Multi-step reasoning with intermediate verification**: math word problems, diagnostic reasoning, legal analysis.

Do *not* use ToT-RAG for simple factual queries — the overhead is significant (multiple LLM calls per branch per depth level).

---

## Core Data Structures

```python
from dataclasses import dataclass, field
from enum import Enum

class NodeState(Enum):
    OPEN      = "open"
    EVALUATED = "evaluated"
    PRUNED    = "pruned"
    FINAL     = "final"

@dataclass
class ThoughtNode:
    thought:    str                       # The partial reasoning / hypothesis
    depth:      int                       # Depth in the tree (0 = root)
    score:      float = 0.0              # Evaluation score (0–1)
    state:      NodeState = NodeState.OPEN
    evidence:   list[str] = field(default_factory=list)  # Retrieved passages
    children:   list["ThoughtNode"] = field(default_factory=list)
    parent:     "ThoughtNode" = None
```

---

## Full ToT-RAG Implementation

```python
import anthropic
import json

client = anthropic.Anthropic()

THOUGHT_GENERATOR_PROMPT = """You are a careful reasoner. Given a question and partial reasoning so far,
generate {n_thoughts} distinct candidate next-thoughts (hypotheses, approaches, or reasoning steps).
Each thought should explore a different angle or sub-question.
Output a JSON array of strings: ["thought1", "thought2", ...]"""

EVALUATOR_PROMPT = """Given a question, a partial reasoning path, and retrieved evidence,
score the reasoning path on a scale of 0.0–1.0:
  - 1.0: Strong evidence supports this path; likely leads to a correct answer
  - 0.5: Mixed evidence; worth exploring but uncertain
  - 0.0: Evidence contradicts or is irrelevant to this path; prune

Also output whether this thought is a final answer (is_final: true/false).

Output JSON: {"score": 0.0-1.0, "is_final": true/false, "reasoning": "..."}"""


def generate_thoughts(question: str, path: list[str], n: int = 3) -> list[str]:
    context = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(path))
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap; many calls per tree
        max_tokens=512,
        system=THOUGHT_GENERATOR_PROMPT.format(n_thoughts=n),
        messages=[{"role": "user", "content": f"Question: {question}\n\nReasoning so far:\n{context or 'None yet'}"}],
    )
    return json.loads(resp.content[0].text)


def evaluate_thought(question: str, path: list[str], evidence: list[str]) -> dict:
    path_text     = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(path))
    evidence_text = "\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence))
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=EVALUATOR_PROMPT,
        messages=[{"role": "user", "content":
            f"Question: {question}\n\nReasoning path:\n{path_text}\n\nEvidence:\n{evidence_text}"}],
    )
    return json.loads(resp.content[0].text)


def retrieve_for_thought(thought: str, retrieve_fn, k: int = 3) -> list[str]:
    """Retrieve evidence targeted at a specific reasoning branch."""
    results = retrieve_fn(thought, k=k)
    return [r["text"] for r in results]


def tot_rag(
    question: str,
    retrieve_fn,
    max_depth: int = 3,
    branching_factor: int = 3,
    beam_width: int = 2,     # keep top-k branches at each level
    prune_threshold: float = 0.3,
) -> str:
    """
    BFS Tree of Thought with retrieval at each node.
    Returns the best final answer found.
    """
    # Initialize with root thoughts
    root_thoughts = generate_thoughts(question, path=[], n=branching_factor)
    
    # Each beam item: (path, score, evidence)
    beam = [([], 0.5, [])]  # start with empty path
    
    for depth in range(max_depth):
        candidates = []
        
        for path, _, accumulated_evidence in beam:
            # Generate new thoughts from this path
            new_thoughts = generate_thoughts(question, path, n=branching_factor)
            
            for thought in new_thoughts:
                new_path     = path + [thought]
                # Retrieve evidence specifically for this thought branch
                evidence     = retrieve_for_thought(thought, retrieve_fn, k=3)
                all_evidence = accumulated_evidence + evidence
                
                evaluation   = evaluate_thought(question, new_path, all_evidence)
                score        = evaluation["score"]
                is_final     = evaluation["is_final"]
                
                if score < prune_threshold:
                    continue  # prune this branch
                
                if is_final:
                    # Generate final answer from this path
                    return synthesize_answer(question, new_path, all_evidence)
                
                candidates.append((new_path, score, all_evidence))
        
        if not candidates:
            break
        
        # Keep top beam_width branches by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]
    
    # Return best path found even without a definitive final answer
    if beam:
        best_path, _, best_evidence = beam[0]
        return synthesize_answer(question, best_path, best_evidence)
    
    return "Unable to find a confident answer."


def synthesize_answer(question: str, path: list[str], evidence: list[str]) -> str:
    path_text     = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(path))
    evidence_text = "\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence))
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=768,
        messages=[{"role": "user", "content":
            f"Question: {question}\n\nReasoning path:\n{path_text}\n\nEvidence:\n{evidence_text}\n\n"
            f"Based on the reasoning and evidence above, provide a clear, concise final answer."}],
    )
    return resp.content[0].text
```

---

## Search Strategies in ToT

### BFS (Breadth-First)

Explores all nodes at depth d before moving to d+1. Best when you want to compare all hypotheses at the same level.

```python
# BFS: process all candidates at each depth before going deeper
# (the implementation above uses BFS with a beam)
```

### DFS (Depth-First)

Pursue the most promising branch to full depth before backtracking. Better for deep reasoning chains where early branching cost is high.

```python
def tot_dfs(question: str, retrieve_fn, path: list, depth: int, max_depth: int) -> str:
    if depth >= max_depth:
        return synthesize_answer(question, path, [])
    
    thoughts = generate_thoughts(question, path, n=3)
    evaluated = []
    
    for thought in thoughts:
        evidence   = retrieve_for_thought(thought, retrieve_fn)
        evaluation = evaluate_thought(question, path + [thought], evidence)
        evaluated.append((thought, evaluation["score"], evidence, evaluation["is_final"]))
    
    # Sort by score, try best first
    evaluated.sort(key=lambda x: x[1], reverse=True)
    
    for thought, score, evidence, is_final in evaluated:
        if score < 0.3:
            break  # remaining are worse, prune
        if is_final:
            return synthesize_answer(question, path + [thought], evidence)
        result = tot_dfs(question, retrieve_fn, path + [thought], depth + 1, max_depth)
        if result:
            return result
    
    return None
```

---

## Cost Management

ToT-RAG is expensive. A tree of depth 3, branching factor 3, and beam width 2 makes:
- Thought generation: 2 × 3 generations per level × 3 levels = 18 LLM calls
- Evaluation: ~18 evaluation calls
- Retrieval: ~18 retrieval calls

Use these controls:

```python
class ToTCostController:
    def __init__(self, max_llm_calls: int = 20, max_retrieval_calls: int = 15):
        self.llm_calls       = 0
        self.retrieval_calls = 0
        self.max_llm         = max_llm_calls
        self.max_retrieval   = max_retrieval_calls
    
    def can_generate(self) -> bool:
        return self.llm_calls < self.max_llm
    
    def can_retrieve(self) -> bool:
        return self.retrieval_calls < self.max_retrieval
    
    def record_llm_call(self):       self.llm_calls += 1
    def record_retrieval(self):      self.retrieval_calls += 1
```

Practical budget: use Haiku for generation and evaluation (cheap), Sonnet only for the final synthesis call.

---

## ToT-RAG vs. ReAct vs. Standard RAG

| Dimension | Standard RAG | ReAct RAG | ToT-RAG |
|-----------|-------------|-----------|---------|
| **Reasoning shape** | None | Linear chain | Tree (branching) |
| **Backtracking** | No | No | Yes (pruning) |
| **Best for** | Simple factual Q | Sequential multi-hop | Competing hypotheses |
| **LLM calls** | 1 | 3–6 | 15–50+ |
| **Latency** | <1s | 3–10s | 15–60s |
| **Debuggability** | Low | High (trace) | High (tree) |

---

## Key Takeaways

1. **ToT-RAG is a premium tool** — reserve for queries with genuinely competing hypotheses; it costs 10–50× more than standard RAG.
2. **Use BFS + beam width** for exploration-heavy tasks; DFS for deep reasoning chains.
3. **Haiku for generation/evaluation, Sonnet for synthesis** — the cost difference between models is the difference between a viable and prohibitive system.
4. **Score + prune aggressively** — a pruning threshold of 0.3–0.4 eliminates most low-value branches early.
5. **The tree structure is an audit log** — each branch shows what the system considered and why it was kept or pruned.

---

## Interview Q&A

**Q: What is Tree of Thought RAG and when does it outperform ReAct RAG?**

Tree of Thought RAG generates multiple competing reasoning paths (thoughts) at each step, retrieves targeted evidence for each branch, and prunes low-scoring branches before exploring further — like a best-first search over the reasoning space. ReAct uses a linear chain: each thought directly informs the next without exploring alternatives. ToT-RAG outperforms ReAct when the correct answer depends on ruling out plausible-but-wrong hypotheses (diagnostic reasoning, root cause analysis, multi-interpretation queries). It underperforms when the query has a clear path to the answer — in those cases, ReAct's serial chain is simpler, cheaper, and equally accurate.

---

**Q: How do you keep ToT-RAG costs from exploding in production?**

Four controls: (1) cap branching factor to 2–3 — the marginal value of a fourth branch rarely justifies the cost; (2) set a hard call budget (max LLM calls = 20, max retrieval calls = 15) and return the best answer found so far when the budget is exhausted; (3) use a small cheap model (Haiku) for thought generation and evaluation, reserving Sonnet only for the final synthesis call; (4) set an aggressive pruning threshold (0.3) so weak branches die early. A well-tuned ToT-RAG at depth 3 should fit within 20–25 LLM calls — comparable to a thorough ReAct agent.

---

**Q: How does retrieval differ between ReAct RAG and ToT-RAG?**

In ReAct, each retrieval query is informed by the previous observation — the search is sequential and linear. In ToT-RAG, each branch has its own retrieval query derived from the thought on that branch, not from a global conversation history. This means ToT-RAG can retrieve evidence that specifically supports or refutes a single hypothesis ("evidence for hypothesis A: retrieve 'drug interaction X'") rather than one general retrieval per turn. This targeted, hypothesis-specific retrieval is ToT-RAG's main precision advantage: the right branch gets the right evidence, instead of all branches sharing the same undirected retrieval result.
