# 42 — Search-R1 / Reasoning RAG (RL-Trained Search)

> An LLM is trained end-to-end with reinforcement learning to interleave its own reasoning tokens with self-issued search calls, learning *when* and *what* to retrieve purely from answer-correctness reward — no one hand-writes the retrieval policy.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
    │
    ▼
Policy LLM generates ──► <think> reasoning tokens </think>
    │
    ▼
Policy LLM decides to retrieve ──► <search> query </search>
    │
    ▼
Retriever executes the query, returns passages ──► <information> ... </information>
    │           (inserted into context, NOT produced by the LLM)
    ▼
Policy LLM continues ──► more <think>/<search> turns, as many as needed
    │
    ▼
Policy LLM emits <answer> final answer </answer>
    │
    ▼
Outcome Reward (exact-match / F1 against gold answer) ──► PPO / GRPO update
    (retrieved-token loss masking: gradient only flows through the
     LLM's own think/search/answer tokens, never the inserted passages)
```

### Key Components

| Component | Responsibility |
|---|---|
| Policy LLM | Single model that both reasons and decides when/what to search, trained end-to-end |
| Retriever (frozen) | Dense/sparse search engine invoked as an external tool; not trained, just called |
| Rollout Engine | Executes multi-turn think→search→observe trajectories during RL training and inference |
| Reward Function | Outcome-based (answer correctness, e.g. F1/EM); no process supervision on retrieval decisions |
| RL Trainer (PPO/GRPO) | Updates policy weights using rollout reward, with retrieved-token loss masking |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| RL algorithm | PPO, GRPO (Group Relative Policy Optimization) |
| Reference implementation | Search-R1 (PeterGriffinJin/Search-R1, GitHub) |
| RL training infra | veRL, OpenRLHF, TRL |
| Retriever backend | Dense retriever (e.g. E5) over Wikipedia dumps, or a live search API |
| Sibling frameworks | R1-Searcher, ReSearch, DeepRetrieval (same RL-for-search family) |

---

## Q1. What is Search-R1 and how does it differ from prompted Agentic RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Search-R1** (Jin et al., 2025, *"Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning"*, arXiv:2503.09516) trains an LLM with reinforcement learning to autonomously interleave reasoning (`<think>`) and search calls (`<search>`) during generation, optimizing purely for final-answer correctness. No human ever writes rules for *when* to search — the policy learns it from reward.

This is a fundamentally different mechanism from the prompted **Agentic RAG** described in file 04, which uses **ReAct-style or FLARE-style prompting**: a frozen, instruction-following LLM is *told* (via prompt engineering) to reason, then act, then observe, in a fixed loop. The retrieval policy in Agentic RAG lives entirely in the prompt; the model's weights never change.

```
Agentic RAG (file 04) — PROMPTED policy:
  System prompt: "Think step by step. If you need information, call search(query).
                   Otherwise, answer directly."
  Frozen LLM (e.g. Claude, GPT-4) follows this instruction each turn.
  Retrieval decisions = whatever the frozen model infers from the prompt.

Search-R1 — LEARNED policy:
  Base LLM (e.g. Qwen2.5-7B) is fine-tuned with RL.
  Reward = 1 if final answer matches gold, else 0.
  Over thousands of rollouts, gradient updates shape WHEN the model emits
  <search> vs continues reasoning — this becomes part of the model's weights,
  not a prompt instruction.
```

| Dimension | Agentic RAG (file 04, ReAct/FLARE) | Search-R1 (RL-trained) |
|---|---|---|
| Retrieval policy | Prompted (in-context instructions) | Learned (baked into model weights via RL) |
| Requires training run | No | Yes (PPO/GRPO over rollouts) |
| Generalizes retrieval timing | Only as well as the prompt generalizes | Learned from reward signal across many examples |
| Model needed | Any instruction-following LLM | A model you can fine-tune (open weights) |
| Reported gains | Depends on prompt engineering | Search-R1 reports +41% (Qwen2.5-7B) / +20% (Qwen2.5-3B) over RAG baselines on 7 QA benchmarks |

**Key insight:** Agentic RAG asks a frozen model to *follow* a retrieval strategy; Search-R1 makes the model *discover* its own retrieval strategy through trial and error, guided only by whether the final answer was correct.

</details>

---

## Q2. What does the training rollout actually look like, token by token? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A Search-R1 rollout is a single generation stream where the policy LLM's own tokens are interleaved with retriever output that gets *inserted*, not generated:

```
<think> The question asks about a treaty signed in 1919. I should search for it. </think>
<search> Treaty of Versailles signatories 1919 </search>
<information> [Retriever executes the query against the index]
  Doc 1: "The Treaty of Versailles was signed on 28 June 1919 by..."
  Doc 2: "Signatories included the Allied Powers and Germany..."
</information>
<think> The passages confirm Germany and the Allied Powers signed it.
  I now have enough to answer. </think>
<answer> The Treaty of Versailles (1919) was signed by the Allied Powers
  and Germany. </answer>
```

**Multi-turn behavior:** the model can emit multiple `<search>` blocks in sequence if the first retrieval was insufficient — this is what lets Search-R1 handle multi-hop questions (compare with the fixed decompose-then-retrieve pattern of file 19, Iterative Multi-Hop RAG) without any hand-coded stopping rule; the model itself learns when it has "enough."

**Pseudo-code for the rollout loop:**

```python
def rollout(question: str, policy_llm, retriever, max_turns: int = 4) -> str:
    context = f"Question: {question}\n"
    for turn in range(max_turns):
        # Policy generates until it hits </search>, </answer>, or max tokens
        segment = policy_llm.generate(context, stop=["</search>", "</answer>"])
        context += segment

        if segment.strip().endswith("</search>"):
            query = extract_between(segment, "<search>", "</search>")
            passages = retriever.search(query, k=3)
            info_block = f"<information>{format_passages(passages)}</information>\n"
            context += info_block          # inserted, not generated by the model
        elif segment.strip().endswith("</answer>"):
            return extract_between(segment, "<answer>", "</answer>")
    return "NO_ANSWER"
```

**Reward is computed only at the end**, comparing the extracted `<answer>` against the gold label (exact match or F1) — there is no intermediate reward for "good" search queries.

</details>

---

## Q3. Why does Search-R1 mask the retrieved-token loss during training, and what breaks if you don't? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

During RL fine-tuning, the policy gradient is computed over the *log-probability of the tokens the model generated*. The `<information>...</information>` block is not generated by the model — it's copied verbatim from the retriever's output and spliced into the context. If you don't mask it, the training loop will treat those retrieved tokens as if the model "chose" to produce them, and will:

```
Without masking:
  - The policy gradient tries to increase/decrease the log-probability of
    retrieved passage tokens (e.g. "The Treaty of Versailles was signed on...")
  - But the model never actually predicted those tokens — they're copy-pasted
  - This injects noisy, meaningless gradient signal tied to whatever documents
    happened to be retrieved, unrelated to the model's own decisions
  - Training becomes unstable; loss spikes correlate with long/short retrieved
    passages rather than actual policy quality

With retrieved-token loss masking:
  - Loss mask = 0 for all <information>...</information> tokens
  - Loss mask = 1 for <think>, <search>, <answer> tokens (the model's own output)
  - Gradient only updates the model's reasoning/search/answer behavior
  - Training signal is clean: "did MY decisions lead to a correct answer?"
```

```python
def compute_loss_mask(token_ids: list[int], info_start_id: int, info_end_id: int) -> list[int]:
    """1 = model-generated token (train on it), 0 = retrieved/inserted token (mask it)."""
    mask, inside_info = [], False
    for tok in token_ids:
        if tok == info_start_id:
            inside_info = True
        mask.append(0 if inside_info else 1)
        if tok == info_end_id:
            inside_info = False
    return mask

# loss = -sum(logprob[i] * mask[i] for i in range(len(tokens))) / sum(mask)
```

This is the same principle as masking prompt tokens in standard SFT — you only backpropagate through what the model is responsible for producing.

</details>

---

## Q4. What outcome-based reward function does Search-R1 use, and why not reward the search queries directly? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Search-R1 uses a deliberately simple **outcome reward**: compare the final extracted `<answer>` to the gold answer using exact match (EM) or F1, with a small format-compliance bonus/penalty for well-formed `<think>/<search>/<answer>` tags.

```python
def compute_reward(rollout_output: str, gold_answer: str) -> float:
    format_ok = has_valid_tags(rollout_output)          # <think>, <search>, <answer> well-formed
    if not format_ok:
        return -1.0                                       # format penalty
    predicted = extract_between(rollout_output, "<answer>", "</answer>")
    em = int(normalize(predicted) == normalize(gold_answer))
    return float(em)                                       # 0 or 1
```

**Why not reward the search queries directly (e.g. reward retrieval precision/recall)?**

- Building a "good query" labelset requires humans to annotate what an ideal search query looks like for every training question — expensive and subjective.
- Rewarding retrieval metrics directly can be gamed: the model learns to produce queries that maximize BM25/embedding overlap with a labeled "good" passage without that passage actually being useful for answering.
- Outcome-based reward is **self-supervising**: you only need (question, gold answer) pairs, which already exist in QA datasets — no retrieval-quality annotation needed.
- It naturally handles the multi-hop case: an intermediate search doesn't need to be individually "correct," it just needs to contribute to an eventually correct final answer, so the model is free to discover unconventional but effective query strategies.

**Trade-off:** with sparse, delayed reward (only 0/1 at the very end of a multi-turn trajectory), credit assignment is harder — GRPO variants (used by Search-R1 and R1-Searcher) address this by comparing multiple rollouts of the same question against each other (relative advantage) rather than relying on a learned value function like standard PPO.

</details>

---

## Q5. What are the failure modes of RL-trained search policies, and how do they compare across the Search-R1 family? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Failure mode 1 — reward hacking on retrieval count.** With a pure outcome reward, a model can learn to over-search (issue many redundant queries "just in case," inflating latency/cost) or under-search (guess from parametric memory when a lucky guess is rewarded, skipping retrieval on questions that actually needed it). Mitigations: add small per-search-call cost penalties, or format/turn-count regularization.

**Failure mode 2 — training instability over long multi-turn trajectories.** As the number of `<search>` turns grows, the effective trajectory length balloons, and sparse terminal reward makes credit assignment noisy. GRPO-style methods stabilize this by normalizing reward within a group of rollouts for the same prompt, rather than requiring a separate learned critic (as vanilla PPO does).

**Failure mode 3 — reliance on a frozen retriever's blind spots.** Because the retriever itself is not trained, the policy can only learn to phrase *queries* better — it cannot fix a fundamentally weak or stale index. If the retriever consistently fails on a class of questions, the RL policy will learn to route around it (answer from parametric knowledge) rather than fixing the underlying retrieval gap, which can silently increase hallucination on that class of questions.

**How the family compares:**

| Method | Distinctive mechanism | Reward signal |
|---|---|---|
| **Search-R1** | Multi-turn `<think>/<search>/<answer>`, retrieved-token masking | Outcome (EM/F1) + format |
| **R1-Searcher** | Two-stage RL: first learns to invoke search reliably, then optimizes answer quality | Stage 1: search-format reward; Stage 2: outcome reward |
| **ReSearch** | Frames search as a first-class reasoning-chain operation, elicits emergent reflection/self-correction without heuristics | Outcome-only, no supervised reasoning traces |
| **DeepRetrieval** | Trains the *query generator* itself (not just an answering agent) against real search engine APIs, rewarded by retrieval metrics (recall/relevance) | Retrieval-quality reward (recall@k), not final-answer reward |

**Combining with other bank architectures:** an RL-trained search policy like Search-R1 can be layered underneath a Corrective RAG (file 06) verification step — the learned policy decides *when* to retrieve, while a separate lightweight judge still checks *whether* the retrieved evidence is sufficient before allowing an `<answer>`, combining a learned retrieval trigger with an explicit correction loop.

</details>

---

## Real-World Applications

- **Open-domain multi-hop QA systems**: Search-R1-style training reported large gains (+41% Qwen2.5-7B, +20% Qwen2.5-3B) over prompted RAG baselines on seven QA benchmarks (HotpotQA, 2WikiMultihopQA, Musique, etc.)
- **Literature/biomedical search agents**: DeepRetrieval trains small (3B) models that outperform GPT-4o/Claude on PubMed and clinical-trial literature retrieval tasks by directly optimizing recall
- **Cost-sensitive search agents**: RL-trained policies that learn to minimize the number of search calls per query are attractive where each search call has a real dollar cost (paid search APIs)
- **Autonomous research assistants**: sibling frameworks (R1-Searcher, ReSearch) demonstrate the same RL-for-search recipe generalizing to reflection and self-correction behavior without hand-authored heuristics
