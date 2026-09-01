# 49 — Auto-RAG & DeepRAG (Per-Step Retrieve-or-Reason Decisions)

> Two related architectures where the LLM decides, at *every step* of a multi-step reasoning process, whether to retrieve or to answer from its own parametric knowledge — rather than classifying query complexity once, up front.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Auto-RAG (multi-turn autonomous retrieval dialogue)
────────────────────────────────────────────────────
Query
  │
  ▼
┌─────────────────── Turn Loop (LLM ↔ Retriever) ───────────────────┐
│  LLM Planner: "What do I still need to know?"                      │
│       │                                                            │
│       ├─ Decide: retrieve again (issue new query)                  │
│       │      │                                                     │
│       │      ▼                                                     │
│       │  Retriever → new passages appended to dialogue history     │
│       │                                                            │
│       └─ Decide: sufficient info gathered → stop, emit final answer│
└──────────────────────────────────────────────────────────────────┘
  │
  ▼
Final Answer (autonomously decided iteration count, no fixed hop budget)


DeepRAG (retrieval-augmented reasoning as an MDP)
────────────────────────────────────────────────────
Query
  │
  ▼
Query Decomposer → atomic subquery 1, subquery 2, ...
  │
  ▼
┌────────────── Per-Subquery Decision (MDP state) ──────────────┐
│  Atomic Decision: retrieve external OR use parametric reasoning │
│       │                             │                          │
│       ▼                             ▼                          │
│  Retriever fetches passage     LLM answers subquery from memory │
│       │                             │                          │
│       └──────────► Update reasoning state ◄─────────────────────┘
│                            │                                    │
│                            ▼ (next subquery, repeat)             │
└──────────────────────────────────────────────────────────────────┘
  │
  ▼
Final Answer (synthesized from the full retrieval-narrative chain)
```

### Key Components

| Component | Responsibility |
|---|---|
| LLM Planner / Decision Policy | At each step, decides retrieve-vs-reason (Auto-RAG: continue-vs-stop; DeepRAG: retrieve-vs-parametric per atomic subquery) |
| Query Decomposer (DeepRAG) | Breaks the original question into atomic subqueries forming a "retrieval narrative" |
| Retriever | Executes retrieval only when the policy decides it's needed for the current step/subquery |
| Reasoning State Tracker | Carries forward accumulated answers/evidence across steps, conditioning the next decision |
| Stopping Policy (Auto-RAG) | Learned/self-determined criterion for ending the multi-turn dialogue with the retriever |
| MDP Formulation (DeepRAG) | Formalizes the sequence of retrieve/reason decisions as states, actions, and rewards for training/inference |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Auto-RAG reference | Auto-RAG (Yu, Zhang, Feng — ICT/CAS, 2024), fine-tuned open-source LLMs, code at `github.com/ictnlp/Auto-RAG` |
| DeepRAG reference | DeepRAG (Guan et al., 2025), MDP-based binary tree search for training data construction |
| Base models | Llama-family or Qwen-family open-source LLMs fine-tuned for the decision policy |
| Orchestration | LangGraph / custom agent loop for the multi-turn retriever dialogue |
| Retriever | Standard dense retriever (Contriever/DPR) or BM25 hybrid, called on-demand |
| Evaluation | HotpotQA, 2WikiMultiHopQA, MuSiQue, and single-hop QA sets (NQ, TriviaQA) for measuring adaptive iteration count |

---

## Q1. What is the core idea shared by Auto-RAG and DeepRAG, and how does it differ from Adaptive RAG's routing? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both **Auto-RAG** (Yu, Zhang, Feng, "Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models," arXiv:2411.19443, Nov 2024) and **DeepRAG** (Guan, Zeng, Meng, Xin, Lu, Lin, Han, Sun, Zhou, "DeepRAG: Thinking to Retrieve Step by Step for Large Language Models," arXiv:2502.01142, Feb 2025) let the model decide, **at every step of a multi-step reasoning process**, whether it should retrieve external information or rely on what it already knows/has gathered.

**Adaptive RAG (file 11) classifies complexity ONCE, up front:**

```
Adaptive RAG:
  Query → Complexity Classifier (one decision) → route to:
              no-retrieval | single-hop | multi-hop
  Once routed, the chosen strategy runs to completion.
```

**Auto-RAG / DeepRAG decide retrieve-vs-not AT EVERY STEP, during the reasoning itself:**

```
Auto-RAG / DeepRAG:
  Query → [decompose / start reasoning]
       → step 1: retrieve? → yes/no  → produces partial answer/evidence
       → step 2: retrieve? → yes/no  → produces partial answer/evidence
       → step 3: retrieve? → yes/no  → ...
       → stop when enough evidence has been accumulated
```

**Why this matters:** a single up-front complexity classification (Adaptive RAG) can't adapt mid-reasoning if a sub-question turns out to be easier or harder than the initial classification implied. Auto-RAG and DeepRAG instead treat "should I retrieve right now" as a decision made fresh at each reasoning step — so a single multi-hop question can mix retrieved and purely-parametric steps within the *same* answer, rather than being locked into one strategy for the whole query.

</details>

---

## Q2. How does Auto-RAG decide, turn by turn, when to retrieve and what to ask? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Auto-RAG frames the retrieval process as an autonomous **multi-turn dialogue between the LLM and the retriever**, expressed entirely in natural language, with no hand-authored prompting scaffold (no fixed "Thought/Action/Observation" template imposed by the developer — the reasoning-and-decision instructions are themselves synthesized and used to fine-tune the model).

```python
def auto_rag_loop(query: str, retriever, llm, max_turns: int = 10) -> str:
    """
    Simplified illustration of Auto-RAG's autonomous retrieval dialogue.
    The LLM has been fine-tuned to emit its own retrieval decisions and
    stopping decisions in natural language, without an externally imposed
    ReAct-style prompt template.
    """
    dialogue_history = [{"role": "user", "content": query}]

    for turn in range(max_turns):
        # LLM decides: do I need to retrieve, and if so, what's my query?
        decision = llm.generate(dialogue_history)
        # decision contains natural-language reasoning + either:
        #   a retrieval query, or
        #   a final answer + explicit stop signal

        if decision.is_final_answer:
            return decision.answer

        retrieved = retriever.search(decision.retrieval_query, k=5)
        dialogue_history.append({"role": "assistant", "content": decision.reasoning})
        dialogue_history.append({"role": "tool", "content": retrieved})

    return llm.generate(dialogue_history, force_answer=True)
```

**Key training detail:** Auto-RAG's authors synthesize training instructions by having a strong LLM autonomously plan and reason through iterative retrieval on training questions, producing decision-making traces (when to retrieve, what to query, when to stop) that are then used to fine-tune the target (often smaller, open-source) model. The result is that the fine-tuned model has *internalized* the retrieve/stop policy — at inference time, no external orchestration logic decides for it.

**Observed behavior:** Auto-RAG autonomously adjusts its number of retrieval iterations to the difficulty of the question — easy factual questions terminate in 1–2 turns, harder multi-hop questions take more turns — without any hop-count hyperparameter set by a human.

</details>

---

## Q3. How does DeepRAG formalize the retrieve-vs-reason decision as a Markov Decision Process, and how is it trained? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

DeepRAG decomposes a query into a sequence of **atomic subqueries** (its "retrieval narrative"), and at each subquery treats the retrieve-vs-parametric choice as an **action in an MDP**:

```
State  s_t  = (original query, subqueries answered so far, answers so far)
Action a_t  ∈ {RETRIEVE, PARAMETRIC}
              RETRIEVE   → call retriever on the current atomic subquery
              PARAMETRIC → let the LLM answer the subquery from memory alone
Reward      = terminal reward for final-answer correctness,
              shaped to penalize unnecessary/redundant retrieval calls
```

```python
def deeprag_step(state: dict, llm, retriever) -> dict:
    """One atomic decision step in DeepRAG's retrieval narrative."""
    subquery = llm.generate_next_subquery(state)          # decompose further
    action = llm.decide_action(state, subquery)            # RETRIEVE or PARAMETRIC

    if action == "RETRIEVE":
        passage = retriever.search(subquery, k=3)
        answer = llm.answer_subquery(subquery, context=passage)
    else:  # PARAMETRIC
        answer = llm.answer_subquery(subquery, context=None)

    state["history"].append({"subquery": subquery, "action": action, "answer": answer})
    return state

def deeprag(query: str, llm, retriever, max_steps: int = 6) -> str:
    state = {"query": query, "history": []}
    for _ in range(max_steps):
        state = deeprag_step(state, llm, retriever)
        if llm.is_sufficient(state):
            break
    return llm.synthesize_final_answer(state)
```

**Training procedure (two stages):**

1. **Binary tree search over decision sequences** — for each atomic subquery, explore both RETRIEVE and PARAMETRIC branches, propagate final-answer correctness back to label which sequence of decisions was actually necessary (i.e., find the *minimal* retrieval path that still reaches the correct answer).
2. **Imitation / policy fine-tuning** — train the model on these discovered minimal-retrieval decision sequences, then further calibrate the retrieve/parametric decision boundary against the model's own actual parametric knowledge (a subquery is only worth answering parametrically if the model, in practice, tends to get it right without retrieval).

**Reported result:** the paper reports a 26.4% accuracy improvement alongside improved retrieval efficiency, driven mainly by cutting out retrieval calls that the tree search shows were unnecessary — i.e., the model learns to trust its own parametric knowledge more often than an always-retrieve baseline would.

</details>

---

## Q4. Concretely, on the same multi-hop question, how would Auto-RAG's turn-based dialogue differ from DeepRAG's atomic-decision MDP? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Take: *"What is the birth year of the director of the movie that won Best Picture the year [X] was born?"*

**Auto-RAG's view — one continuous dialogue, decisions are turn-level and expressed in free-form natural language:**

```
Turn 1 (LLM): "I need to find who directed the Best Picture winner in [X]'s birth year.
               First I need [X]'s birth year." → retrieves "[X] birth year"
Turn 2 (LLM): "Got [X]'s birth year = 1990. Now I need the Best Picture winner of 1990."
               → retrieves "Best Picture winner 1990"
Turn 3 (LLM): "Winner was [Movie Y], directed by [Director Z]. Now I need [Director Z]'s
               birth year." → retrieves "[Director Z] birth year"
Turn 4 (LLM): "I have everything I need." → STOP, emits final answer
```
Each turn's retrieval query is generated by the model reasoning in natural language about what it still lacks; there's no explicit subquery decomposition step separate from the dialogue itself.

**DeepRAG's view — the question is decomposed into an explicit atomic-subquery plan first, and each subquery independently gets a RETRIEVE/PARAMETRIC label:**

```
Subquery 1: "What year was [X] born?"
  → action: RETRIEVE (model doesn't reliably know this) → 1990

Subquery 2: "Who won Best Picture in 1990?"
  → action: RETRIEVE → [Movie Y], directed by [Director Z]

Subquery 3: "What year was [Director Z] born?"
  → action: PARAMETRIC (model already knows this director well — no retrieval needed)
  → 1946 (from memory)
```

The key structural difference: DeepRAG explicitly separates "what atomic fact do I need next" (decomposition) from "should I retrieve for it" (action), and can skip retrieval on subquery 3 even mid-chain if the model is confident parametrically — whereas Auto-RAG's per-turn decision is entangled with its own free-form reasoning trace rather than a formally atomized subquery list.

</details>

---

## Q5. Both architectures make a per-step retrieve-or-not call — what happens when that call is wrong, and how would you detect/mitigate it in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Per-step retrieval decisions introduce two distinct failure modes, and they compound differently than in a single up-front router (Adaptive RAG):

**Failure mode 1 — False "PARAMETRIC" (skips retrieval when it shouldn't have):**

```
Subquery: "Who is the current CEO of [Company]?"
Model's confidence: high (it "knows" the answer from training)
Reality: training-cutoff-stale answer, company changed CEOs since

Result: confidently wrong answer for this subquery propagates into every
downstream subquery that depends on it — and because no retrieval happened,
there's no retrieved passage to cross-check against later.
```

This is strictly worse than Adaptive RAG's failure mode of "misclassified as no-retrieval," because in a multi-hop chain, one bad parametric answer early in the chain corrupts every subsequent step, and it happened *silently* — there's no artifact (like a bad retrieved passage) to inspect afterward.

**Failure mode 2 — False "RETRIEVE" (retrieves when parametric knowledge was fine):**

```
Subquery: "What is 15% of 200?"
Action: RETRIEVE (over-cautious policy)
Result: wasted latency/cost on a retrieval call for something the model
could answer perfectly well from reasoning alone — no correctness harm,
but erodes the whole efficiency benefit these architectures are built for.
```

**Mitigations:**

| Guard | Effect |
|---|---|
| Post-hoc consistency check on PARAMETRIC subqueries | Periodically spot-retrieve a sample of parametric-only answers in production and compare, to catch calibration drift (similar in spirit to Astute RAG's internal-vs-external cross-check) |
| Confidence thresholding, not binary classification | Require the action policy to emit a confidence score, not just a label, and force RETRIEVE below a threshold rather than trusting a hard classifier boundary |
| Time-sensitivity heuristics | Force RETRIEVE for subqueries containing volatility cues ("current," "latest," "as of," named entities with high update frequency), regardless of the policy's own confidence |
| Chain-level error tracking | Log which subquery in a chain a wrong final answer traces back to, to identify systematic PARAMETRIC-miscalibration on specific entity/fact types over time |

**Combining with Astute RAG:** since both Auto-RAG/DeepRAG and Astute RAG are fundamentally about "when can I trust the model's own knowledge vs. retrieved/external evidence," a natural production combination is to use DeepRAG-style per-subquery action decisions to control *when* to retrieve, and Astute RAG-style consolidation to *reconcile* the retrieved passage against the model's parametric answer whenever both are available (e.g., after a low-confidence PARAMETRIC decision, retrieve anyway and consolidate rather than trusting either source alone) — trading some of the latency savings for materially higher robustness on high-stakes subqueries.

</details>

---

## Real-World Applications

- **Open-domain multi-hop QA assistants** (HotpotQA/2WikiMultiHopQA-style deployments): variable-depth reasoning chains where retrieval depth should track question difficulty, not a fixed hop budget
- **Cost-sensitive RAG at scale**: DeepRAG-style skip-retrieval-when-confident policies materially cut retrieval/API cost on high-volume QA traffic where many subqueries are answerable parametrically
- **Research and fact-verification copilots**: Auto-RAG-style autonomous dialogue with a retriever, producing an interpretable, natural-language trace of what was looked up and why
- **Enterprise knowledge assistants with mixed fresh/stable knowledge**: routing volatile facts (pricing, org charts, policy versions) to retrieval while answering stable facts (definitions, historical data) parametrically, decided per-subquery rather than per-query
