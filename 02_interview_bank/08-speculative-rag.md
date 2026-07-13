# 08 — Speculative RAG

> A smaller specialist model generates multiple draft answers; a larger generalist model selects and refines the best one — balancing quality and latency.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
                    ┌────────────────┐
                    │   User Query    │
                    └────────┬───────┘
                             ▼
                 ┌─────────────────────┐
                 │      Retriever       │  top-k chunks
                 └──────────┬──────────┘
                             ▼
                 ┌─────────────────────┐
                 │ Partition into m     │
                 │ document subsets     │
                 └──┬────┬────┬────┬───┘
                    ▼    ▼    ▼    ▼
                 ┌─────────────────────┐
                 │  Small Drafter LM    │  (parallel, one call per subset)
                 │  → draft + rationale │
                 └──────────┬──────────┘
                     draft_1 .. draft_m
                             ▼
                 ┌─────────────────────┐
                 │ Large Verifier LM     │  scores / selects / refines
                 └──────────┬──────────┘
                             ▼
                        Final Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| Retriever | Retrieves top-k chunks upstream of drafting |
| Document Partitioner | Splits retrieved chunks into m subsets (e.g., multi-perspective cluster sampling) |
| Small Drafter Model | Generates a candidate answer plus rationale per subset, in parallel |
| Large Verifier / Judge Model | Scores drafts via self-consistency and self-reflection, then selects or refines the best one |
| Draft Selector | Applies the scoring formula (drafter confidence × self-consistency × self-reflection) to pick the final draft |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Drafter model | Distilled or fine-tuned small model (e.g., Mistral-7B SFT, Llama-3-8B-Instruct) |
| Verifier model | GPT-4-class, Claude, Gemini-Ultra-class, or Llama-3-70B |
| Serving | vLLM with prefix caching for batched, parallel drafting |
| Retriever | Standard vector DB (Chroma, Pinecone, Qdrant) upstream of partitioning |

---

## Q1. What is Speculative RAG and what is the core insight behind it? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Speculative RAG** (Wang et al., 2024) is inspired by speculative decoding in LLMs. The core insight is:

> A small, specialized model is faster and often better at *drafting* answers over retrieved documents. A large, general model is better at *judging and refining* those drafts.

**Pipeline:**

1. Retrieved documents are **partitioned into subsets**.
2. A **small specialist RAG drafter** generates one candidate answer per subset.
3. A **large generalist LLM** receives all candidate answers (not the raw documents) and selects/refines the best one.

This reduces the number of tokens the large LLM must process — instead of reading all retrieved chunks, it only reads short candidate answers — improving both **quality** and **latency**.

```
Retrieved chunks (k=10)
  │
  ├── Subset 1 (2 chunks) ─► Small RAG drafter ─► Candidate answer 1 + rationale
  ├── Subset 2 (2 chunks) ─► Small RAG drafter ─► Candidate answer 2 + rationale
  ├── Subset 3 (2 chunks) ─► Small RAG drafter ─► Candidate answer 3 + rationale
  ├── Subset 4 (2 chunks) ─► Small RAG drafter ─► Candidate answer 4 + rationale
  └── Subset 5 (2 chunks) ─► Small RAG drafter ─► Candidate answer 5 + rationale
                                                          │
                                        ┌─────────────────┘
                                        ▼
                           Large generalist LLM
                           (reads only candidates, not raw chunks)
                                        │
                                        ▼
                                   Final Answer
```

**Headline results (Wang et al., 2024, arXiv:2407.08223):** evaluated on TriviaQA, MuSiQue, PubHealth, and ARC-Challenge, Speculative RAG improved accuracy by up to **12.97%** while reducing latency by **51%** (PubHealth) versus conventional RAG where the large model reads all retrieved chunks.

**Two properties worth calling out in an interview:**

1. The verifier performs **no retrieval pass of its own** — it never sees the raw chunks, only the drafts (each with a rationale that carries the evidence forward).
2. The verifier needs **no fine-tuning** — it scores drafts via self-consistency and self-reflection signals computed from its own conditional probabilities (see Q3). Only the small drafter is fine-tuned.

</details>

---

## Q2. How does document partitioning work in Speculative RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Rather than passing all retrieved chunks to a single model, Speculative RAG partitions them:

1. Retrieve top-k chunks (e.g., k=10).
2. Partition into m subsets of size k/m (e.g., 5 subsets of 2 chunks each).
3. Each subset is processed independently by the drafter model to produce a candidate answer with a rationale.

**Why partition?**
- Reduces context length per drafter call (faster inference).
- Different subsets may contain complementary evidence — multiple drafts capture this diversity.
- Prevents the "lost-in-the-middle" problem since each drafter sees fewer chunks.

**Partitioning strategies compared:**

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Contiguous rank-order** | Chunks 1–2 → subset 1, chunks 3–4 → subset 2, ... | Trivial; top subsets get best evidence | Draft 1 dominates; later drafts get weak evidence |
| **Random** | Shuffle, then split | Cheap diversity | Subsets may duplicate one viewpoint or miss key evidence entirely |
| **Cluster-sampled (paper's "multi-perspective sampling")** | k-means cluster chunk embeddings into `m` clusters; build each subset by sampling **one chunk per cluster** without replacement | Every draft sees the full topical spread; drafts differ in *which representative* of each topic they saw — maximizes useful diversity | Needs embeddings + clustering step (~ms, embeddings already exist from retrieval) |

The paper uses cluster sampling specifically so that drafts are **diverse but each individually complete** — every subset covers all perspectives, just through different documents. This is what makes the verifier's job meaningful: it compares genuinely different argument paths, not "good subset vs. leftover subset."

```python
from sklearn.cluster import KMeans
import numpy as np

def multi_perspective_partition(chunks, embeddings, m: int):
    """One chunk per cluster per subset (Wang et al., 2024)."""
    labels = KMeans(n_clusters=m, n_init="auto").fit_predict(np.array(embeddings))
    clusters = [[i for i, l in enumerate(labels) if l == c] for c in range(m)]

    subsets = []
    for _ in range(m):                      # build m subsets
        subset = []
        for cluster in clusters:            # one draw per topic cluster
            if cluster:
                subset.append(chunks[cluster.pop(0)])
        subsets.append(subset)
    return subsets
```

**Practical caveat:** deduplicate near-identical chunks *before* clustering — duplicates form their own cluster and end up in every subset, manufacturing false consensus among drafts (this also matters for robustness; see Q12).

</details>

---

## Q3. What is the role of the large generalist LLM in Speculative RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The large LLM acts as a **verifier and refiner**, not a reader of raw documents. It receives:

```
Candidate 1: [Answer A with rationale from subset 1]
Candidate 2: [Answer B with rationale from subset 2]
...
Candidate m: [Answer N with rationale from subset m]

Task: Select the most accurate, well-supported answer, or synthesize a better answer combining the candidates.
```

**Its responsibilities:**
1. **Judge faithfulness** — Which candidate is best grounded in evidence?
2. **Synthesize** — Combine complementary evidence from multiple candidates.
3. **Refine** — Improve fluency, completeness, or reasoning of the best candidate.

Because it only reads candidate answers (not raw chunks), the large model's context is much shorter — typically 3–5x fewer tokens than standard RAG.

**How the paper's verifier actually scores drafts:**

The verifier in Wang et al. doesn't generate a free-form critique — it computes scores from **conditional probabilities** in a single forward pass:

| Score | Question it answers | Computed as |
|---|---|---|
| **Drafter confidence** (ρ_draft) | How sure was the drafter? | Drafter's P(answer, rationale \| question, subset) — comes free with generation |
| **Self-consistency** (ρ_SC) | Does this draft hold together given just the question? | Verifier's P(answer, rationale \| question) |
| **Self-reflection** (ρ_SR) | Does the rationale actually support the answer? | Verifier's P("Yes" \| question, draft, reflection statement) |

The final score is the **product** of the three; the highest-scoring draft is returned.

```
score(draft_i) = ρ_draft,i × ρ_SC,i × ρ_SR,i
final answer   = draft with max score
```

**Why this is fast:** computing log-probs of an *existing* text is a **prefill-only** operation — no autoregressive decoding. Scoring 5 drafts of ~120 tokens each is one batched prefill over ~600 tokens, taking milliseconds even on a 70B model. This is a key reason the paper's 51% latency reduction holds despite involving a large model.

**Caveat for production:** logprob scoring requires access to token probabilities. With API-only verifiers that don't expose logprobs (most hosted frontier models), fall back to **generative selection** — prompt the verifier to pick/synthesize as in the template above. It costs extra output tokens and a decode pass, but works with any model (see Q6 on pairings).

</details>

---

## Q4. How does Speculative RAG compare to Speculative Decoding in standard LLM inference? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both share the **draft-then-verify** philosophy, but apply it differently:

| Aspect | Speculative Decoding | Speculative RAG |
|---|---|---|
| **What's drafted** | Token sequences | Full answers |
| **Drafter** | Small version of same model | Small specialist RAG model |
| **Verifier** | Large version of same model | Large generalist LLM |
| **Goal** | Reduce decoding latency | Reduce context length + improve accuracy |
| **Acceptance criterion** | Token probability alignment | Answer quality / faithfulness score |
| **Granularity** | Token-level | Answer-level |

In Speculative Decoding, the large model *accepts or rejects* the small model's tokens. In Speculative RAG, the large model *selects among* the small model's answers. The analogy is intentional — both exploit the asymmetry between fast drafting and accurate verification.

**The pairing-constraint difference (a common interview trap):**

Speculative decoding compares per-token probability distributions, so drafter and verifier must share a **tokenizer/vocabulary** (typically same model family — e.g., Llama-3-8B drafting for Llama-3-70B). Speculative RAG passes drafts across the boundary as **plain text**, so tokenizer compatibility is **irrelevant** — a Mistral-7B drafter can pair with a GPT-4-class or Gemini-Ultra-class API verifier across vendors. This is what makes Speculative RAG deployable with hosted frontier models, while speculative decoding is strictly a self-hosted inference optimization.

**Worked latency/cost example:**

Setup: k = 10 chunks × 300 tokens = 3,000 context tokens; final answer ≈ 150 tokens; m = 5 drafts of ~120 tokens each; subset = 2 chunks = 600 tokens. Illustrative rates: large model $10/M input, $30/M output; self-hosted 7B drafter ≈ $0.20/M tokens amortized.

| | Standard large-model RAG | Speculative RAG |
|---|---|---|
| Large-model input | 3,000 ctx + 200 prompt = 3,200 tok → **$0.0320** | 5 × 120 drafts + 200 prompt ≈ 800 tok → **$0.0080** |
| Large-model output | 150 tok → $0.0045 | 150 tok → $0.0045 (select/refine) |
| Drafter (5 parallel calls) | — | 5 × (600 + 120 + 80 prompt) = 4,000 tok → $0.0008 |
| **Cost per query** | **$0.0365** | **$0.0133 (~64% cheaper)** |
| Wall-clock (typical) | 3.2K-token prefill + 150-token decode, strictly sequential ≈ 2.5 s | max(5 parallel drafts ≈ 0.8 s) + short verifier call ≈ 0.7 s → **≈ 1.5 s** |

The latency win comes from two places: (1) drafts run **in parallel**, so drafting wall-clock ≈ one short 7B generation regardless of m, and (2) the verifier prefills ~800 tokens instead of 3,200 — and with logprob scoring (Q3) it doesn't even decode.

**Crossover condition** — speculative wins when the context the big model *would have read* exceeds what it reads instead:

```
C_big × T_ctx   >   m × C_small × (T_subset + T_draft)  +  C_big × (m × T_draft)
(standard input)    (drafter cost, ~negligible)            (verifier reads m drafts)
```

With T_ctx = 3,000 vs. m × T_draft = 600, big-model input shrinks ~4–5x. It **loses** when contexts are short (k ≤ 3 small chunks, T_ctx ≈ m × T_draft): the extra model hop, orchestration overhead, and verifier call cost more than they save — just run standard RAG.

</details>

---

## Q5. When would Speculative RAG outperform Agentic RAG, and when would it not? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Speculative RAG wins when:**
- Queries can be answered from a single retrieval round (no multi-hop needed).
- You have a large retrieved context and want to reduce large-LLM token costs.
- Latency matters — parallel drafting + single verification is faster than sequential agent loops.
- You have a domain-specific drafter model fine-tuned on your corpus.

**Agentic RAG wins when:**
- The query requires **multi-hop reasoning** (follow-up retrievals based on intermediate findings).
- The query type is **unknown at retrieval time** (agent decides which tool to call).
- The knowledge base doesn't contain all needed information (agent can fall back to web search).
- You need **interactive workflows** where the LLM takes real-world actions (not just Q&A).

**In practice:** Speculative RAG is best for a well-scoped, document-heavy Q&A product. Agentic RAG is better for open-ended assistants and workflows. Many production systems combine both — use Speculative RAG for the retrieval-heavy path and Agentic RAG for complex, multi-step queries.

**The other comparison interviewers probe: Speculative RAG vs. a reranker.** Both are "quality levers" you add after retrieval, but they operate at different levels:

| Dimension | Cross-encoder reranker | Speculative RAG |
|---|---|---|
| **Operates on** | Evidence — reorders/filters chunks | Answers — drafts and judges them |
| **Conflicting evidence** | Can't resolve; only ranks by relevance | Each draft argues from its subset; verifier adjudicates between conclusions |
| **Multi-perspective queries** | Collapses to one "most relevant" ordering | Diversifies across document subsets by design |
| **Added compute** | One cheap cross-encoder pass (~10–50 ms) | m drafter generations + verifier pass (~0.5–1.5 s) |
| **Failure mode** | Most-relevant chunk ≠ best answer | Verifier overhead wasted on easy queries |

**Speculative beats reranking when:** the query is ambiguous, sources conflict (e.g., two reports give different revenue figures), or the answer needs *synthesis* of complementary chunks. A reranker only changes the reading order of the same single generation — it cannot reason per evidence group or compare candidate conclusions.

**Reranking beats speculative when:** the query is a simple factoid with one obvious answer ("What year was the company founded?"). All m drafts converge on the same answer and the verifier merely confirms it — pure overhead. `rerank → top-3 → one small-model generation` is faster, cheaper, and just as accurate.

**They compose, not compete:** rerank first to clean the candidate pool, then partition the top-k for drafting. And route by query type — factoid classifier sends easy queries down the rerank path, open-ended ones down the speculative path.

</details>

---

## Q6. How do you select and fine-tune the small drafter model in Speculative RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The drafter model is typically a smaller version (or different model) specialized for fast, accurate answer extraction from limited context.

**Concrete drafter/verifier pairings:**

| Drafter (specialist, fine-tuned) | Verifier (generalist, zero-shot) | When to use |
|---|---|---|
| **Mistral-7B SFT** on (question, context, answer, rationale) tuples — the paper's M_Drafter | **Mistral-7B / Mixtral-8x7B Instruct** — the paper's M_Verifier | Reproduces the published setup; full logprob access for cheap prefill-only scoring |
| Mistral-7B or Llama-3-8B, fine-tuned | **GPT-4-class or Gemini-Ultra-class API** | Maximum verifier reasoning quality; no logprobs → use generative selection (Q3 caveat) |
| **Llama-3-8B-Instruct → SFT** | **Llama-3-70B-Instruct** | Fully open-weights, self-hosted, single model family simplifies serving and prompting |
| Qwen2.5-7B → SFT | Qwen2.5-72B | Multilingual corpora |

**What makes a good pairing:**

1. The drafter **must be RAG-fine-tuned.** Base instruct models draft verbose, weakly grounded answers; SFT on answer-plus-rationale-from-context data is what makes a 7B draft competitive with a 70B reading the full context.
2. The verifier needs **strong reasoning and good calibration** but **zero fine-tuning** — it only judges, never reads raw documents.
3. **Capability gap matters.** The verifier should be clearly stronger than the drafter, or its judgments add no signal over the drafter's own confidence.
4. **Tokenizer compatibility is irrelevant** — unlike speculative decoding (Q4), drafts cross the boundary as plain text, so cross-vendor and cross-family pairs work fine.
5. **Logprob access changes the economics.** Self-hosted verifiers can score drafts in a single prefill pass; API-only verifiers force generative selection, so budget the extra output tokens and decode latency.

**Selection criteria:**

| Criterion | Recommendation |
|-----------|----------------|
| **Base model** | Mistral 7B, Llama-3-8B, or a fine-tuned domain expert |
| **Latency target** | Should answer in <100ms per partition (on single GPU) |
| **Quality bar** | Answer accuracy >80% on validation set (can be loose; verifier refines) |
| **Fine-tuning** | Improves 3-5pp if trained on domain Q&A pairs |

**Fine-tuning recipe:**

```python
# 1. Prepare training data (domain Q&A pairs + retrieved context)
train_data = [
    {
        "question": "What is the company revenue?",
        "context": "Q3 2024 revenue was $10M, up 20%...",
        "answer": "Q3 2024 revenue: $10M",  # Extractive answer
        "rationale": "Extracted from financial report"
    },
    ...
]

# 2. Fine-tune using SFT (Supervised Fine-Tuning)
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B")

# Format data
formatted_data = [
    f"Question: {ex['question']}\nContext: {ex['context']}\nAnswer: {ex['answer']}"
    for ex in train_data
]

# Train
training_args = TrainingArguments(
    output_dir="drafter_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_data,
)
trainer.train()

# 3. Evaluate on validation set
# Metric: Answer exact match or BLEU against gold answers
```

**Typical improvements:**
- Base Mistral 7B: ~70% answer accuracy on in-domain data.
- +3pp fine-tuning on 5K examples.
- +2pp quantization (4-bit) without quality loss.

</details>

---

## Q7. Implement the Speculative RAG pipeline end-to-end in Python `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Here's a complete implementation:

```python
import asyncio
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SpeculativeRAG:
    def __init__(self, drafter_model="mistral-7b", verifier_model="gpt-4o-mini"):
        self.drafter = ChatOpenAI(model=drafter_model, temperature=0)
        self.verifier = ChatOpenAI(model=verifier_model, temperature=0)
        self.vectorstore = None
    
    def partition_documents(self, documents: List[str], num_subsets: int = 5) -> List[List[str]]:
        """Partition retrieved documents into subsets."""
        subset_size = max(1, len(documents) // num_subsets)
        partitions = [
            documents[i:i+subset_size]
            for i in range(0, len(documents), subset_size)
        ]
        return partitions[:num_subsets]  # Ensure exactly num_subsets
    
    async def draft_answer(self, query: str, documents: List[str]) -> dict:
        """Small model generates answer from one subset."""
        context = "\n".join(documents)
        prompt = f"""Question: {query}

Context:
{context}

Provide a concise answer with a brief rationale for your answer.

Answer: <answer>
Rationale: <why this is the best answer given the context>"""
        
        response = await asyncio.to_thread(self.drafter.invoke, prompt)
        
        # Parse response
        answer = response.content.split("Answer:")[1].split("Rationale:")[0].strip()
        rationale = response.content.split("Rationale:")[1].strip()
        
        return {
            "answer": answer,
            "rationale": rationale,
            "documents": documents
        }
    
    async def draft_all_candidates(self, query: str, documents: List[str], 
                                   num_drafters: int = 5) -> List[dict]:
        """Generate candidate answers in parallel."""
        partitions = self.partition_documents(documents, num_drafters)
        
        # Parallel drafting
        tasks = [self.draft_answer(query, part) for part in partitions]
        candidates = await asyncio.gather(*tasks)
        
        return candidates
    
    def verify_and_select(self, query: str, candidates: List[dict]) -> dict:
        """Large model selects and refines the best candidate."""
        
        candidates_text = "\n\n".join([
            f"Candidate {i+1}:\nAnswer: {c['answer']}\nRationale: {c['rationale']}"
            for i, c in enumerate(candidates)
        ])
        
        selection_prompt = f"""Question: {query}

Below are multiple candidate answers from different document subsets:

{candidates_text}

Task:
1. Evaluate which candidate is most accurate and well-supported.
2. If beneficial, synthesize a better answer from multiple candidates.
3. Provide your final answer.

Final Answer: <answer>
Reasoning: <explanation of which candidate(s) you selected and why>"""
        
        response = self.verifier.invoke(selection_prompt)
        
        # Parse response
        final_answer = response.content.split("Final Answer:")[1].split("Reasoning:")[0].strip()
        reasoning = response.content.split("Reasoning:")[1].strip()
        
        return {
            "final_answer": final_answer,
            "reasoning": reasoning,
            "num_candidates_considered": len(candidates)
        }
    
    async def query(self, query: str, k: int = 10) -> dict:
        """End-to-end pipeline."""
        
        # 1. Retrieve documents
        documents = self.vectorstore.similarity_search(query, k=k)
        doc_texts = [doc.page_content for doc in documents]
        
        # 2. Draft candidate answers in parallel
        candidates = await self.draft_all_candidates(query, doc_texts, num_drafters=5)
        
        # 3. Verify and select
        result = self.verify_and_select(query, candidates)
        
        return result

# Usage
rag = SpeculativeRAG()
rag.vectorstore = Chroma(...)

result = asyncio.run(rag.query("What is the company revenue?"))
print(result["final_answer"])
```

</details>

---

## Q8. How do you tune the number of subsets (m) and subset size in Speculative RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The number of partitions `m` trades off drafter parallelism, verification cost, and answer diversity.

```
Quality vs. Latency for different m values:

        Quality (Answer accuracy)
             ↑
           0.88 │           m=3  m=5  m=7
                │          ╱    ╱    ╱ (more drafts = better)
           0.85 │   ──────╱────╱────╱
                │  m=1           
           0.80 │ (no partitioning)
                │
                └────────────────────────► Latency
                 100ms  200ms  400ms  600ms

           m=1 (no partitioning): lowest latency, lower quality
           m=5 (balanced):        medium latency, good quality
           m=10 (exhaustive):     high latency, marginal quality gain
```

**Tuning approach:**

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class SpeculativeRAGConfig:
    num_subsets: int
    drafter_model: str
    verifier_model: str
    
class GridSearchTuning:
    def __init__(self, validation_set):
        self.validation_set = validation_set  # 100 test queries
    
    def evaluate_config(self, config: SpeculativeRAGConfig):
        """Measure quality and latency for this config."""
        
        accuracies = []
        latencies = []
        
        for query, gold_answer in self.validation_set:
            import time
            start = time.time()
            
            result = asyncio.run(rag.query(query, k=10))
            
            latency = time.time() - start
            
            # Compute accuracy (e.g., BLEU or exact match)
            accuracy = compute_similarity(result["final_answer"], gold_answer)
            
            accuracies.append(accuracy)
            latencies.append(latency)
        
        return {
            "config": config,
            "avg_accuracy": np.mean(accuracies),
            "p95_latency": np.percentile(latencies, 95),
            "cost_per_query": estimate_cost(config)
        }
    
    def find_pareto_frontier(self, configs):
        """Find configs on Pareto frontier (highest accuracy at each latency level)."""
        
        results = [self.evaluate_config(cfg) for cfg in configs]
        
        # Sort by latency
        results.sort(key=lambda x: x["p95_latency"])
        
        pareto_frontier = []
        max_accuracy = 0
        
        for result in results:
            if result["avg_accuracy"] >= max_accuracy:
                pareto_frontier.append(result)
                max_accuracy = result["avg_accuracy"]
        
        return pareto_frontier

# Example: Try m=3, 5, 7, 10
configs = [
    SpeculativeRAGConfig(num_subsets=m, drafter_model="mistral-7b", 
                        verifier_model="gpt-4o-mini")
    for m in [3, 5, 7, 10]
]

tuner = GridSearchTuning(validation_set)
pareto = tuner.find_pareto_frontier(configs)

# Output:
# m=3: 85% accuracy, 120ms latency
# m=5: 87% accuracy, 250ms latency ← Recommended (good accuracy/latency tradeoff)
# m=7: 87.5% accuracy, 380ms latency
# m=10: 88% accuracy, 600ms latency (too slow)
```

**Why returns diminish past m ≈ 3–5:**

| Step | Typical marginal gain | Why it shrinks |
|---|---|---|
| m: 1 → 3 | +3–5 pp | New subsets surface genuinely new evidence and argument paths |
| m: 3 → 5 | +1–2 pp | Subsets begin to overlap; drafts start repeating each other |
| m: 5 → 10 | <1 pp | Evidence coverage is saturated; the verifier reads near-duplicate drafts |

Meanwhile **cost grows linearly in m** (m drafter calls + m drafts in the verifier prompt), so the cost/quality curve bends hard after m ≈ 5.

**Practical rules of thumb:**

- Default to **m = 4–5**, tied to retrieval: `m ≈ k / subset_size` (k=10 with 2-chunk subsets → m=5 falls out naturally).
- **Pick m to fill a drafter GPU batch** — all m drafts run as one batched forward pass (see Q11), so m=4 or m=8 can be effectively free compared to m=5 on some serving stacks.
- **Adapt m per query:** high top-1 retrieval score and tight score gap → easy query, m=2 suffices; flat score distribution or conflicting sources → m=5.
- Re-tune m whenever you change the drafter, chunk size, or k — the saturation point moves with subset informativeness.

</details>

---

## Q9. How does Speculative RAG behave when no retrieved document is relevant? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A failure mode: all retrieved documents are off-topic. Speculative RAG can't save a bad retrieval round.

```
Retrieved docs:
├─ [Document 1] (irrelevant)
├─ [Document 2] (irrelevant)
└─ [Document 3] (irrelevant)

→ All drafters generate hallucinations or "I don't know"
→ Verifier picks "best of bad options"
→ Result: Hallucinated answer presented as confident
```

**Mitigation strategies:**

| Strategy | Implementation | Trade-off |
|----------|----------------|-----------|
| **Confidence threshold** | If all candidates score <0.5, abstain ("I don't know") | May miss valid answers from weak evidence |
| **Fallback to web search** | Detect all-irrelevant case, trigger web search | +500ms latency, higher cost |
| **Semantic diversity check** | If all candidates are too similar, flag for review | Human-in-the-loop latency |
| **Retrieval quality gating** | Filter retrieved docs before partitioning (CRAG style) | Adds evaluator overhead |

```python
class RobustSpeculativeRAG(SpeculativeRAG):
    def detect_retrieval_failure(self, candidates: List[dict], 
                               query: str, threshold=0.4) -> bool:
        """Check if retrieval was likely unsuccessful."""
        
        # Heuristic: If all drafts are very short or contain "I don't know"
        abstentions = sum(1 for c in candidates 
                         if len(c["answer"]) < 20 or "don't know" in c["answer"])
        
        if abstentions / len(candidates) > 0.6:
            return True  # Likely retrieval failure
        
        # Semantic similarity: if all candidates are nearly identical, 
        # no new information from partitioning
        similarities = []
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                sim = cosine_similarity(candidates[i]["answer"], 
                                      candidates[j]["answer"])
                similarities.append(sim)
        
        if all(sim > 0.9 for sim in similarities):
            return True  # All drafts redundant
        
        return False
    
    async def query_with_fallback(self, query: str, k: int = 10) -> dict:
        """Query with web search fallback."""
        
        # 1. Retrieve documents
        documents = self.vectorstore.similarity_search(query, k=k)
        doc_texts = [doc.page_content for doc in documents]
        
        # 2. Draft candidates
        candidates = await self.draft_all_candidates(query, doc_texts, num_drafters=5)
        
        # 3. Detect failure
        if self.detect_retrieval_failure(candidates, query):
            print("Retrieval quality low; falling back to web search")
            web_results = web_search_api.search(query, num_results=k)
            web_texts = [r["snippet"] for r in web_results]
            candidates = await self.draft_all_candidates(query, web_texts, num_drafters=5)
        
        # 4. Verify
        result = self.verify_and_select(query, candidates)
        return result
```

</details>

---

## Q10. What are the GPU/memory requirements for serving Speculative RAG in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Speculative RAG runs two models in parallel, each with different resource requirements.

**Resource table (single request):**

| Component | Model Size | VRAM | Batch Size | Total VRAM |
|-----------|-----------|------|-----------|-----------|
| Drafter | Mistral 7B | 14GB | 5 parallel | 14GB (shared across 5 subsets) |
| Verifier | GPT-4o-mini | API | N/A | N/A (or 8GB if self-hosted) |
| Vector DB | Chroma (in-memory) | ~2GB (for 100K embeddings) | N/A | 2GB |

**Single-GPU setup:**

```
H100 GPU (80GB VRAM):
├─ Drafter (14GB)
├─ Verifier/Router (4GB)
├─ Vector DB (2GB)
├─ Batch buffers (20GB)
└─ Headroom (40GB) ← Can serve ~4 concurrent requests

Recommended: H100 for >100 QPS; A100 (40GB) for <50 QPS
```

**Multi-GPU deployment (production):**

```
┌────────────────────────────────────┐
│  Load Balancer (incoming requests) │
└───────────┬────────────────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
[GPU 0: Drafter]  [GPU 1: Drafter]  (parallelism)
Mistral 7B        Mistral 7B
14GB each         14GB each
    │               │
    └───────┬───────┘
            ▼
   [GPU 2: Verifier]
   GPT-4o or Llama 70B
   16-40GB
            │
            ▼
   [Vector DB Server]
   Qdrant or Milvus
   (separate machine, CPU-based)
```

**Latency and throughput:**

```python
class SpeculativeRAGServer:
    def __init__(self, num_drafter_gpus=2, num_verifier_gpus=1):
        self.drafter_pool = GPUPool(num_drafter_gpus, "mistral-7b")
        self.verifier = GPUPool(num_verifier_gpus, "gpt-4o-mini")
        self.vectordb = RemoteVectorDB("qdrant://localhost:6333")
    
    def estimate_latency(self, k=10, num_subsets=5):
        """Estimate end-to-end latency."""
        
        # Retrieval: 10-50ms
        retrieval_time = 30
        
        # Drafting: parallel across num_subsets, but sequential per subset
        # Mistral 7B → ~50ms per subset (average)
        drafting_time = (num_subsets / self.drafter_pool.num_gpus) * 50
        
        # Verification: ~100ms
        verification_time = 100
        
        # Total
        total = retrieval_time + max(drafting_time, 50) + verification_time
        return total  # ~180-250ms expected
    
    def estimate_throughput(self, qps_budget=100):
        """How many concurrent requests can we serve?"""
        
        avg_latency_ms = self.estimate_latency()
        max_concurrent = qps_budget * (avg_latency_ms / 1000)  # Little's law
        
        return max_concurrent  # e.g., 100 QPS × 0.2s = 20 concurrent

# Example tuning
server = SpeculativeRAGServer(num_drafter_gpus=2, num_verifier_gpus=1)
print(f"Estimated latency: {server.estimate_latency()}ms")
print(f"Max throughput at 100 QPS: {server.estimate_throughput(100)} concurrent")
```

**Cost optimization:**

1. **Batch inference** — Group requests, run drafters in parallel.
2. **Quantization** — 4-bit Mistral 7B → 7GB VRAM (vs. 14GB), <5% accuracy loss.
3. **LoRA adapters** — Multiple domain-specific drafters sharing base weights.
4. **Caching** — Cache candidate answers for repeated queries.

**Recommended deployment:**

| Use Case | Setup | Cost/Month |
|----------|-------|-----------|
| <50 QPS | H100 (1 GPU) + API verifier | ~$3K |
| 50-200 QPS | 2× A100 (drafter + verifier) | ~$10K |
| >200 QPS | 4× H100 (2 drafters + 1 verifier + 1 reserve) | ~$20K |

</details>

---

## Q11. How do you architect and orchestrate a multi-GPU serving cluster for Speculative RAG to maximize drafter parallelism while keeping verifier utilization high? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

First, a precision point interviewers look for: in Speculative RAG the unit of work is a **whole draft answer**, not a token. So "drafter parallelism" means batching m draft generations, and "verifier utilization" means keeping the large model fed with scoring batches. Token-by-token acceptance belongs to speculative decoding, not here (see Q4).

**Cluster layout:**

```
            ┌─────────────────────────────────────────┐
            │  API gateway / load balancer            │
            └───────────────┬─────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │ Orchestrator               │
              │ retrieve → dedupe →        │
              │ partition into m subsets   │
              └──────┬──────────────┬──────┘
                     │ m drafts     │ scoring jobs
        ┌────────────▼────┐   ┌─────▼───────────────┐
        │ Drafter pool    │   │ Verifier pool       │
        │ 2-3× A100, vLLM │   │ 1× H100, 70B model  │
        │ Mistral-7B SFT  │   │ prefill-heavy work  │
        └─────────────────┘   └─────────────────────┘
```

**Why drafting batches almost for free:** the m drafts of one request share the same question and instruction prefix and differ only in the document subset. With prefix caching, the shared prefix is computed once; submit all m as a single batch and the drafting wall-clock ≈ one draft generation, not m.

```python
# Batched drafting with vLLM: m drafts in one call
from vllm import LLM, SamplingParams

drafter = LLM("org/mistral-7b-rag-drafter", enable_prefix_caching=True)

def draft_batch(question: str, subsets: list[list[str]]) -> list[str]:
    shared = f"Question: {question}\nInstructions: answer concisely with a rationale.\n"
    prompts = [shared + "Context:\n" + "\n".join(s) for s in subsets]
    params = SamplingParams(max_tokens=160, temperature=0.7)
    return [o.outputs[0].text for o in drafter.generate(prompts, params)]
```

**Keeping the verifier busy — it's prefill-bound:** with logprob scoring (Q3), verifying m drafts is one prefill over ~m × 120 tokens with **no decoding** — milliseconds on an H100. Drafters, by contrast, decode autoregressively (slow). So one verifier GPU can serve several drafter GPUs:

| Verifier mode | Verifier work per request | Drafter:verifier GPU ratio |
|---|---|---|
| Logprob scoring (select only) | Prefill ~800 tokens, decode 0 | **3–4 : 1** |
| Generative selection / refinement | Prefill ~800 + decode ~150 tokens | **~2 : 1** |

**Cross-request batching:** run continuous batching on both pools, and micro-batch verifier scoring jobs across requests (flush every ~10 ms or 16 jobs, whichever first). Because scoring jobs are tiny and uniform, the verifier batches far better than a standard generation workload.

**Optimization strategies:**

1. **Batch all m drafts of a request as one drafter call** — parallelism within a request is free batch width.
2. **Prefix caching** — shared question/instruction prefix computed once per request; KV reuse across the m drafts.
3. **Micro-batched verification** — group scoring jobs from concurrent requests into one verifier prefill.
4. **Adaptive m under load** — drop from m=5 to m=3 when the draft queue deepens; quality loss is small (Q8) and it sheds load exactly where it accumulates.

**Monitoring:**

- Drafter GPU utilization (target: >80%) and tokens/sec decoded.
- Verifier GPU utilization (target: >80%) — if low, widen the drafter:verifier ratio; if the verify queue backs up, narrow it.
- Queue depths (`draft_queue`, `verify_queue`) — the leading indicator of imbalance.
- Latency P95 (target: <1.5–2 s for interactive use), split by stage (retrieve / draft / verify).

</details>

---

## Q12. How can adversarial inputs exploit systematic disagreement between the drafter and verifier models, and what detection and mitigation strategies prevent this from degrading answer quality? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Framing first: drafts cross the drafter/verifier boundary as **whole text answers** — there is no token-level acceptance to exploit (that's speculative decoding). The real attack surface is the verifier's *selection signals*: agreement among drafts and plausibility of rationales.

**Attack 1 — corpus poisoning to manufacture consensus.** The verifier implicitly trusts agreement: drafts that converge on the same answer score well on self-consistency. An attacker who plants several near-duplicate poisoned documents in the KB gets the poisoned claim into **multiple subsets** → multiple agreeing (wrong) drafts → the verifier selects the poisoned answer with high confidence. Consensus ≠ truth.

```
Poisoned KB:
├─ doc_A: "Product X recall: refund denied per policy 7.2"  ← planted
├─ doc_A': near-duplicate of doc_A                          ← planted
├─ doc_A'': paraphrase of doc_A                             ← planted
└─ doc_B: genuine policy: "full refund within 30 days"

Partitioning spreads A/A'/A'' across 3 of 4 subsets
→ 3 drafts: "refund denied"   1 draft: "full refund"
→ Verifier: majority + coherent rationales → picks "refund denied" ✗
```

**Attack 2 — engineered disagreement.** Queries (or planted contradictory documents) crafted so every draft diverges. The verifier must pick among noise and its score margin becomes meaningless. The attacker can't choose *which* wrong answer wins, but degrades reliability — or, if you abstain on disagreement, mounts a denial-of-answer attack on targeted topics.

**Attack 3 — rationale gaming.** Drafts inherit text from their subsets. A planted document with authoritative-sounding fake citations yields rationales that pass the self-reflection check ("does the rationale support the answer?" — yes: internally coherent, externally false). The verifier never sees raw documents, so it cannot detect the fabricated provenance itself.

**Defences:**

1. **Dedupe before partitioning** — near-duplicate detection (MinHash or embedding similarity > 0.95) so one planted claim cannot occupy multiple subsets. Consensus then requires *independent* sources, which is exactly what makes it meaningful.

2. **Provenance-weighted scoring** — discount drafts whose evidence is single-source or freshly ingested:

   ```
   score'_i = score_i × source_diversity(subset_i) × min_trust(subset_i)
   ```

   where `source_diversity` counts distinct origins in the subset and `min_trust` reflects document age/review status.

3. **Verifier score calibration** — raw scores (products of probabilities, Q3) are *not* calibrated confidences; a 0.6 may mean 95% accuracy on one corpus and 60% on another. Calibrate on a held-out labeled set (temperature scaling or isotonic regression), then choose the abstention threshold τ from the reliability diagram — e.g., "calibrated score ≥ 0.7 ⇒ ≥ 90% empirical accuracy."

4. **All-drafts-disagree protocol** — measure pairwise agreement between extracted answers and never silently return a below-threshold winner:

   ```python
   def select_with_disagreement_guard(question, drafts, scores, tau=0.7):
       agreement = mean_pairwise_similarity([d.answer for d in drafts])
       best_draft, best_score = max(zip(drafts, scores), key=lambda x: x[1])

       if best_score >= tau:
           return best_draft                    # confident winner

       if agreement < 0.3:                      # all drafts disagree
           # Escalation ladder:
           # (a) re-retrieve with a reformulated query
           # (b) full large-model RAG over the raw chunks
           #     (the verifier reads documents this one time)
           # (c) abstain / human escalation
           return fallback_full_rag(question)

       return abstain("Low confidence; sources conflict on this question.")
   ```

5. **Disagreement monitoring** — log per-query draft-agreement entropy and verifier score margins. A sustained spike in disagreement on one topic cluster is the earliest signal of corpus poisoning (the attack changes the *distribution* of disagreement before anyone notices wrong answers). Alert at sustained disagreement rate > 30%; sample flagged queries for human review.

6. **Ingestion-side controls** — scan new KB documents for instruction-like text, anomalous duplication, and citation patterns before they become retrievable. Planted documents are the root cause; the verifier is the *last* line of defence, not the first.

**Defence-in-depth:** dedupe + provenance weighting break consensus manufacturing (Attack 1); calibration + the disagree-protocol prevent confident selection among noise (Attack 2); ingestion controls + monitoring catch the poisoning that rationale gaming depends on (Attack 3). An attacker must defeat all layers simultaneously.

</details>

---

## Real-World Applications

| Application | Domain | Why Speculative RAG Fits |
|---|---|---|
| High-throughput customer service bot (e.g., airline, telco) | Consumer / SaaS | Parallel drafting from a small model + verification by a large model cuts P95 latency by ~40% while keeping accuracy on par with full LLM retrieval |
| Real-time live chat support | E-commerce | Draft-then-verify pattern lets the system respond faster than single-model RAG at peak load without sacrificing answer quality |
| Financial news summarization at scale | Media / FinTech | Thousands of simultaneous queries benefit from speculative decoding; large model verifies only the drafts that need correction |
| Call center copilot (agent assist) | Telecom / BPO | Speculative RAG provides fast preliminary answers to human agents while a more capable model verifies claims in the background |
| Government citizen Q&A portal | Public sector | High daily query volume benefits from cost reduction via speculative drafting; verifier ensures compliance-sensitive answers are accurate |
