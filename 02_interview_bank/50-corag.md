# 50 — CoRAG (Chain-of-Retrieval Augmented Generation)

> Trains the model itself — via rejection-sampled retrieval chains — to dynamically reformulate its query at each step based on the evolving reasoning state, and exposes retrieval-chain length as a test-time compute scaling knob.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Training-time: building retrieval-chain data via rejection sampling
──────────────────────────────────────────────────────────────────
Existing RAG dataset (query → final answer only, no chain labels)
  │
  ▼
Sample MANY candidate retrieve-then-reason chains per question
  (query reformulation → retrieve → sub-answer → reformulate → retrieve → ...)
  │
  ▼
Reject chains that DON'T reach the correct final answer
Keep chains that DO reach the correct final answer
  │
  ▼
Fine-tune model on (query, kept retrieval chain, final answer) triples
  → model learns to autonomously reformulate queries mid-chain


Inference-time: test-time compute scaling
──────────────────────────────────────────────────────────────────
Query
  │
  ▼
┌───────────────── Retrieval Chain (learned, not prompted) ─────────────┐
│  Step 1: reformulate query given current state → retrieve → sub-answer │
│  Step 2: reformulate query given updated state  → retrieve → sub-answer│
│  Step 3: ... (chain length controllable at test time)                  │
└─────────────────────────────────────────────────────────────────────────┘
  │
  ▼
Decoding strategy controls compute:
  - greedy (1 chain, fixed length)
  - best-of-N sampled chains + reranking
  - longer chains for harder queries
  │
  ▼
Final Answer Generator (synthesizes from the full learned retrieval chain)
```

### Key Components

| Component | Responsibility |
|---|---|
| Rejection Sampler (training-time) | Samples many candidate retrieve-reformulate-reason chains per training question, discards ones that don't reach the correct answer |
| Chain-augmented Training Set | (query, kept retrieval chain, final answer) triples used to fine-tune the model |
| Query Reformulator (learned) | Model-internal capability — generates the next retrieval query conditioned on the evolving reasoning state, learned from data rather than prompted |
| Retriever | Executes retrieval for each reformulated query in the chain |
| Test-time Decoding Controller | Chooses chain length / sampling strategy (greedy, best-of-N, chain-length budget) to trade compute for accuracy |
| Final Answer Generator | Synthesizes the answer from the full retrieval chain the model produced |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Reference implementation | CoRAG (Wang, Chen, Yang, Huang, Dou, Wei — Microsoft Research, "Chain-of-Retrieval Augmented Generation," arXiv:2501.14342, Jan 2025; NeurIPS 2025) |
| Training data source | KILT benchmark tasks, augmented with sampled retrieval chains |
| Rejection sampling | Custom sampling + answer-match filtering over many candidate chains per question |
| Base models fine-tuned | Open-source LLMs (e.g. Llama-family) fine-tuned end-to-end on chain data |
| Retriever | Standard dense retriever (E5/Contriever-style) over the KILT corpus |
| Decoding strategies | Greedy chain decoding, best-of-N sampled chains with reranking, dynamic chain-length budgets |

---

## Q1. What is CoRAG, and how does it differ from IRCoT's interleaved retrieval-and-reasoning? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**CoRAG** (Chain-of-Retrieval Augmented Generation — Wang et al., Microsoft Research, arXiv:2501.14342, Jan 2025) trains an "o1-like" RAG model that retrieves and reasons step by step, **dynamically reformulating its query based on the evolving reasoning state**, before producing a final answer.

**IRCoT** (mentioned in file 19, `19-iterative-multihop-rag.md`) achieves interleaved retrieval-and-reasoning purely through **prompting**, on a **frozen, unmodified model**:

```
IRCoT:
  Frozen LLM + fixed prompt template
  → generate one CoT sentence → retrieve based on it → append retrieved text
  → generate next CoT sentence → retrieve → ... → answer
  No training/weight updates. The model's ability to interleave well
  depends entirely on prompt engineering and the base model's existing
  reasoning capability.
```

**CoRAG trains the model itself** to perform this reformulate-retrieve-reason loop, using retrieval chains constructed via rejection sampling as training signal:

```
CoRAG:
  Training: sample many candidate retrieval chains per question,
            keep only chains that reach the correct answer,
            fine-tune the model on those chains.
  Inference: the fine-tuned model NATIVELY reformulates its query at
             each step — this is a learned capability, not a prompted one.
```

**The practical difference:**

| Aspect | IRCoT | CoRAG |
|---|---|---|
| Mechanism | Prompting, frozen model | Fine-tuning on rejection-sampled chains |
| Query reformulation | Implicit, via whatever CoT sentence the frozen model happens to generate | Explicit learned skill, optimized against chains that provably reach correct answers |
| Model requirement | Works out-of-the-box with any capable LLM | Requires a training pipeline and a base model you can fine-tune |
| Test-time compute control | Not a first-class design feature | First-class: chain length / sampling strategy is an explicit scaling knob |

</details>

---

## Q2. How does CoRAG construct training data via rejection sampling when only final answers are labeled? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Most RAG datasets (including KILT tasks) only label the correct final answer — they don't come with ground-truth intermediate retrieval queries or reasoning chains. CoRAG solves this with **rejection sampling**: generate many candidate chains, keep only the ones that actually land on the correct answer.

```python
def sample_candidate_chain(query: str, llm, retriever, max_hops: int = 4) -> dict:
    """Sample one candidate retrieve-reformulate-reason chain."""
    state = {"original_query": query, "steps": []}
    current_query = query

    for hop in range(max_hops):
        # Model proposes next retrieval query given accumulated state
        reformulated = llm.sample_reformulation(state, current_query, temperature=0.8)
        passages = retriever.search(reformulated, k=5)
        sub_answer = llm.sample_subanswer(state, reformulated, passages, temperature=0.8)

        state["steps"].append({
            "query": reformulated, "passages": passages, "sub_answer": sub_answer
        })
        current_query = sub_answer  # condition next reformulation on this

        if llm.sample_stop_decision(state):
            break

    final_answer = llm.sample_final_answer(state)
    return {"chain": state["steps"], "final_answer": final_answer}


def build_corag_training_set(dataset: list[dict], llm, retriever,
                              n_samples: int = 20) -> list[dict]:
    """
    dataset: [{"query": ..., "gold_answer": ...}, ...]
    Returns kept chains: only those whose sampled final_answer matches gold.
    """
    training_examples = []
    for item in dataset:
        for _ in range(n_samples):
            candidate = sample_candidate_chain(item["query"], llm, retriever)
            if answers_match(candidate["final_answer"], item["gold_answer"]):
                training_examples.append({
                    "query": item["query"],
                    "chain": candidate["chain"],
                    "final_answer": candidate["final_answer"],
                })
                # keep multiple accepted chains per question if desired,
                # or just the first/shortest one found
    return training_examples

def answers_match(predicted: str, gold: str) -> bool:
    return predicted.strip().lower() == gold.strip().lower()  # or fuzzy/EM match
```

**Why rejection sampling rather than hand-labeling chains?** Hand-labeling gold retrieval-reformulation chains for every question in a large knowledge-intensive dataset (KILT spans multiple tasks) is infeasible at scale. Rejection sampling instead uses the *existing* answer labels as a filter: any chain sampled from a capable-enough teacher model that happens to arrive at the correct final answer is accepted as a plausible, useful training signal for *how* to get there — even though the chain itself was never hand-verified for perfect step-by-step correctness.

**Fine-tuning:** the target model is then trained (typically via standard supervised fine-tuning) on the kept (query, chain, final_answer) triples, learning to imitate the reformulation-retrieval-reasoning pattern end-to-end.

</details>

---

## Q3. What decoding strategies does CoRAG use at test time to control the retrieval chain, and how do they trade off compute vs. accuracy? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

At inference, CoRAG exposes chain length and sampling strategy as explicit, tunable levers — this is the "test-time compute scaling" analogue to reasoning-model scaling (more inference compute → better accuracy, up to diminishing returns).

```python
def corag_greedy(query: str, model, retriever, chain_length: int) -> str:
    """Cheapest: one deterministic chain of a fixed length."""
    state = {"steps": []}
    current_query = query
    for _ in range(chain_length):
        reformulated = model.reformulate(state, current_query, greedy=True)
        passages = retriever.search(reformulated, k=5)
        sub_answer = model.sub_answer(state, reformulated, passages, greedy=True)
        state["steps"].append({"query": reformulated, "sub_answer": sub_answer})
        current_query = sub_answer
    return model.final_answer(state)


def corag_best_of_n(query: str, model, retriever, chain_length: int, n: int) -> str:
    """More compute: sample N full chains, rerank, pick the best."""
    candidates = []
    for _ in range(n):
        candidates.append(corag_sample_one_chain(query, model, retriever, chain_length))
    return model.rerank_and_select(candidates)


def corag_adaptive_length(query: str, model, retriever, max_length: int = 6) -> str:
    """Let the model decide how many hops it actually needs, up to a cap."""
    state = {"steps": []}
    current_query = query
    for _ in range(max_length):
        reformulated = model.reformulate(state, current_query)
        passages = retriever.search(reformulated, k=5)
        sub_answer = model.sub_answer(state, reformulated, passages)
        state["steps"].append({"query": reformulated, "sub_answer": sub_answer})
        current_query = sub_answer
        if model.is_confident_to_stop(state):
            break
    return model.final_answer(state)
```

**Scaling behavior (as characterized in the paper):**

| Strategy | Compute cost | Accuracy effect |
|---|---|---|
| Greedy, short chain | Lowest | Fine for simple/single-hop queries; underperforms on multi-hop |
| Greedy, long chain | Medium | Better for multi-hop, but wastes compute on simple queries and risks chain drift if reformulation degrades over long chains |
| Best-of-N sampled chains + reranking | Highest | Best accuracy, especially on hard multi-hop tasks — reported >10 point EM improvement over strong baselines on multi-hop QA |
| Adaptive length (model decides to stop) | Query-dependent | Approaches best-of-N accuracy at a fraction of the compute, by only spending extra hops where the question actually needs them |

**Key point:** because chain length/sampling is a *decoding-time* choice, the same trained CoRAG model can be deployed at different latency/cost/accuracy operating points without retraining — analogous to how a reasoning model's "thinking budget" can be dialed up or down at inference time.

</details>

---

## Q4. Why can longer retrieval chains degrade rather than improve accuracy in CoRAG, and how does the paper address it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naively, one might expect "more retrieval hops = strictly better accuracy," but CoRAG's chains can drift or accumulate noise the same way any iterative multi-hop process can:

```
Chain drift example:
  Hop 1: reformulated query is accurate, retrieves good passage
  Hop 2: reformulation conditioned on a slightly imprecise sub-answer from hop 1
         → reformulated query drifts off-topic
  Hop 3: retrieves passages for the now off-topic query → irrelevant context
  Hop 4: final answer synthesized from an increasingly noisy chain
```

Each additional hop is generated conditioned on the model's *own* prior sub-answer — if an early sub-answer is subtly wrong, every downstream reformulation compounds that error, similar to the general error-accumulation risk in any iterative/multi-hop retrieval architecture (see file 19, `19-iterative-multihop-rag.md`, on stopping-criterion design for the same class of problem).

**How CoRAG mitigates this:**

- **Rejection sampling only keeps chains that reach the correct final answer** — so during training, the model is never taught to imitate a drifted chain that still happened to be sampled; only chains with a *correct outcome* survive into the training set, which implicitly biases the learned reformulation policy toward corrective, on-track behavior.
- **Best-of-N test-time sampling + reranking** directly compensates for the fact that any single sampled chain can drift — sampling several chains and reranking by (estimated) final-answer quality lets bad individual chains be filtered out at inference time rather than trusted blindly.
- **Adaptive stopping** avoids forcing the model into more hops than the question actually needs, which limits the number of opportunities for drift to occur in the first place.

**Practical implication for tuning in production:** chain length is not a free accuracy dial — past the point where the question's genuine multi-hop depth is satisfied, additional hops mostly add compute cost and drift risk rather than accuracy gains, so adaptive-length decoding tends to outperform a fixed long-chain-for-everything policy.

</details>

---

## Q5. How would you decide between IRCoT-style prompting and training a CoRAG model for a new multi-hop RAG system, and can the two be combined? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

This is fundamentally a build-vs-prompt tradeoff, and the right choice depends on scale, latency budget, and how much training infrastructure is available.

**When IRCoT-style prompting is the better choice:**

| Condition | Reason |
|---|---|
| No fine-tuning infrastructure / using a closed frontier model | IRCoT works with any capable frozen LLM via prompting alone |
| Rapidly evolving requirements | Prompt changes ship instantly; no retraining cycle |
| Low query volume | Doesn't amortize the upfront cost of building a rejection-sampled training set + fine-tuning pipeline |
| Need to swap base models frequently | A prompting scaffold transfers across model versions with minor tuning; a fine-tuned CoRAG checkpoint is tied to the model it was trained on |

**When training a CoRAG-style model is the better choice:**

| Condition | Reason |
|---|---|
| High query volume, latency/cost-sensitive | A model with an internalized reformulation policy needs less per-step prompt overhead and can be decoded greedily far more reliably than a prompted frozen model |
| Need reliable test-time compute scaling | CoRAG's chain-length/best-of-N knobs are trained-in and calibrated against real accuracy gains, rather than an ad hoc prompt hyperparameter |
| Domain-specific query reformulation patterns | Fine-tuning on rejection-sampled chains from your own domain data teaches reformulation behavior specific to your corpus, rather than relying on a frozen model's general-purpose reasoning |
| You control the base model weights | Fine-tuning is only viable with open-weights models (or a provider that supports custom fine-tuning) |

**Combining both:** in practice, IRCoT-style prompting is a reasonable way to *bootstrap* the rejection-sampling step itself — you can use a frozen, strongly-prompted model (IRCoT-style) as the chain-sampling policy during CoRAG's training-data construction phase, since it already produces reasonable interleaved retrieval-and-reasoning traces. Only chains that reach the correct final answer are then kept and used to fine-tune a smaller/cheaper target model, which — once trained — can be deployed without needing the elaborate IRCoT prompt scaffold at all. This lets you pay the prompting/frozen-model inference cost once, during offline data generation, in exchange for a cheaper, faster, natively-reformulating model in production.

</details>

---

## Real-World Applications

- **Enterprise multi-hop search assistants at scale**: fine-tuned CoRAG-style models reduce per-query latency/cost versus prompting-based interleaved retrieval (IRCoT) when query volume is high
- **Knowledge-intensive benchmarks (KILT tasks)**: CoRAG established new state-of-the-art results across a diverse set of knowledge-intensive tasks by training directly on rejection-sampled chains
- **Cost-tiered QA products**: exposing chain-length/sampling as a user- or product-tier-selectable knob (fast/cheap vs. thorough/expensive answers), the same way reasoning-effort tiers work for reasoning models
- **Domain-adapted retrieval assistants** (legal, medical, financial multi-hop research): fine-tuning the reformulation policy on domain-specific rejection-sampled chains to learn domain query-reformulation idioms a frozen general-purpose model wouldn't produce via prompting alone
