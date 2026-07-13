# 11 — Adaptive RAG

> Dynamically selects no-retrieval, single-hop, or multi-hop strategy based on query complexity at runtime.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
User Query
    │
    ▼
Query Complexity Classifier
    │
    ├─ Simple   → No-Retrieval Path  ─────────► Generator ──► Answer
    │             (direct LLM generation)
    │
    ├─ Moderate → Single-Hop Retriever ────────► Generator ──► Answer
    │             (embed → vector DB → top-k)
    │
    └─ Complex  → Multi-Hop Retrieval Loop ────► Generator ──► Answer
                  (retrieve → reason → retrieve → ... → answer)
```

### Key Components

| Component | Responsibility |
|---|---|
| Query Complexity Classifier | Scores query difficulty (simple/moderate/complex) to drive routing |
| Router | Applies calibrated thresholds to send the query down one of three paths |
| No-Retrieval Path | Answers directly from the LLM's parametric knowledge |
| Single-hop Retriever | One embed + vector search + generate pass |
| Multi-hop Retrieval Loop | Iteratively retrieves and reasons across sub-queries |
| Generator | Produces the final answer from the (optional) retrieved context |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Classifier | Fine-tuned T5-large / DistilBERT / DeBERTa, or a small prompted LLM classifier |
| Orchestration / routing | LangGraph, custom routing middleware |
| Vector DB (single/multi-hop) | Qdrant, Weaviate, Pinecone with HNSW ANN |
| Uncertainty escalation | FLARE-style token-probability monitoring |
| Generation | GPT-4/Claude/Llama variants sized per tier |

---

## Q1. What is Adaptive RAG and how does it differ from fixed-pipeline RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Adaptive-RAG** (Jeong et al., 2024, arXiv:2403.14403) is a retrieval strategy that dynamically routes queries to different retrieval depths based on a query-complexity classifier (a trained T5-large in the original paper), rather than applying a uniform pipeline to all queries.

**Fixed-pipeline RAG** executes the same retrieval strategy (e.g., always retrieve top-k, always do multi-hop) regardless of the query. **Adaptive RAG** uses a learned classifier to predict query complexity and routes accordingly:

- **No-retrieval path** — For simple queries (e.g., "What is X?") that the LLM can answer from parametric knowledge
- **Single-hop path** — For moderately complex queries requiring one retrieval step
- **Multi-hop path** — For complex reasoning queries requiring multiple retrieval and reasoning steps

This reduces latency and cost for simple queries while maintaining answer quality for hard questions. It is particularly effective when query complexity varies widely in production workloads.

</details>

---

## Q2. How does a query complexity classifier work in Adaptive RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A query complexity classifier predicts whether a given query is simple, moderate, or complex, using features like query length, linguistic markers, and semantic similarity to known simple/complex templates.

**Common approaches:**

1. **Keyword-based heuristics** — Flag queries with multi-hop keywords (e.g., "compare", "across") or multiple entities as complex.
2. **Supervised classifier** — Train a linear or neural model on labeled (query, complexity) pairs. Features can include:
   - Query length and token count
   - Presence of comparative/temporal/causal keywords
   - Named entity count
   - Embedding similarity to known simple vs. complex query templates
3. **LLM-as-judge** — Use a lightweight LLM or prompt to estimate complexity, balancing cost vs. accuracy.
4. **Confidence-based** — Let the LLM attempt to answer without retrieval and measure confidence. If confidence is below a threshold, escalate to retrieval.

**Training data** — Typically 500–2000 labeled (query, complexity, answer quality with/without retrieval) triplets from past user interactions or synthetic data.

**Silver labels (the Adaptive-RAG paper approach):** Instead of manual annotation, label each query by the *cheapest strategy that actually answered it correctly*. Run all three strategies offline on a benchmark set:

```
If no-retrieval output matches gold answer        → label A (simple)
Else if single-step retrieval matches gold answer → label B (moderate)
Else                                              → label C (complex)
```

For queries where no strategy succeeds, the paper falls back to dataset-bias labels (queries from single-hop datasets like SQuAD/NQ → B; from multi-hop datasets like HotpotQA/MuSiQue → C). The paper then fine-tunes a **T5-large** classifier on these silver labels — no human annotation needed, and labels reflect *your actual system's* capability, not abstract query difficulty.

The classifier runs *before* retrieval, so it must be fast (<10ms overhead to be practical).

</details>

---

## Q3. What are the three retrieval strategies in Adaptive RAG and when is each chosen? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The three strategies and their selection criteria are:

| Strategy | Trigger | Typical Queries | Benefit |
|---|---|---|---|
| **No-Retrieval** | Low complexity score (< 0.33) | "What is X?", "Define Y", factoid questions | Minimal latency, zero retrieval cost |
| **Single-Hop** | Moderate complexity (0.33–0.66) | "Where do X typically occur?", "Compare X and Y" | 1 retrieval round, faster than multi-hop |
| **Multi-Hop** | High complexity (> 0.66) | "How does X relate to Y in the context of Z?", reasoning chains | Multiple retrieval + reasoning steps, highest quality |

**Routing logic:**

1. Classifier produces a complexity score (0–1) for the input query.
2. Use threshold-based routing: if score < t1 → no-retrieval; if t1 ≤ score < t2 → single-hop; if score ≥ t2 → multi-hop.
3. Thresholds are tuned on a held-out validation set balancing latency, cost, and answer quality.

**Threshold tuning methodology:**

1. Hold out 500–1000 queries with gold answers.
2. **Precompute once, offline:** run all three strategies on every held-out query and record (quality, cost, latency) per strategy. This makes the sweep free — no re-running pipelines per threshold candidate.
3. Sweep (t1, t2) over a grid and compute expected quality/cost under each setting:

```python
import itertools

results = []
for t1, t2 in itertools.product(grid, grid):
    if t1 >= t2:
        continue
    route = lambda s: "none" if s < t1 else ("single" if s < t2 else "multi")
    quality = mean(q[route(score(q))].f1 for q in heldout)
    cost = mean(q[route(score(q))].cost for q in heldout)
    results.append((t1, t2, quality, cost))
```

4. Plot the **cost vs. quality frontier** (each (t1, t2) pair is a point; keep only Pareto-optimal points). Pick the knee of the curve, or the max-quality point under a cost/latency budget:

```
F1
0.86 │                          ●  (t1=0.2, t2=0.5)  ← quality-max
0.84 │              ●  (t1=0.3, t2=0.6)  ← knee, usually best
0.80 │      ●  (t1=0.4, t2=0.8)
0.74 │  ●  (t1=0.6, t2=0.9)  ← cost-min
     └──────────────────────────── Cost/query
       $0.005  $0.012  $0.020  $0.03
```

5. Re-run the sweep whenever the classifier is retrained or the query distribution shifts — thresholds tuned for one score distribution are stale after recalibration.

**Misclassification asymmetry:** routing a complex query to no-retrieval destroys answer quality; routing a simple query to multi-hop only wastes money. So bias t1 *low* (escalate when in doubt) and rely on cost controls rather than quality controls for the upper tier.

**End-to-end flow:**

```
User Query
    │
    ├─ Complexity Classifier
    │     │
    │     ├─ Score < 0.33 → Direct LLM generation (no retrieval)
    │     ├─ 0.33 ≤ Score < 0.66 → Retrieve 1x, then generate
    │     └─ Score ≥ 0.66 → Iterative multi-hop retrieval + generation
    │
    └─ Answer
```

</details>

---

## Q4. How is FLARE integrated into Adaptive RAG for uncertainty-triggered retrieval? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

FLARE (Forward-Looking Active Retrieval Augmented Generation) is a method that triggers retrieval *dynamically during generation*, based on the model's predictive uncertainty about upcoming tokens. Adaptive RAG can integrate FLARE to refine its initial routing decision or to escalate from no-retrieval to retrieval mid-generation.

**Integration approach:**

1. **Initial routing** — Complexity classifier routes to no-retrieval or single-hop as usual.
2. **During generation** — Monitor the LLM's confidence on each generated token.
3. **Uncertainty trigger** — If confidence drops below a threshold (e.g., next token probability < 0.5), pause generation and retrieve documents related to the low-confidence phrase.
4. **Augment context** — Append retrieved documents to the prompt and resume generation.

**Benefits:**

- Catches cases where the classifier underestimated complexity.
- Reduces unnecessary retrieval for simple questions (no-retrieval path is tried first).
- Enables late-stage correction if the LLM begins hallucinating.

**Example:** A query "Who won the 2024 World Cup?" is classified as simple. The LLM starts: "As of my training data..." but uncertainty spikes when generating the year. FLARE triggers retrieval of recent sports news and corrects the answer.

Combining Adaptive (upfront routing) + FLARE (runtime uncertainty) yields the best latency and quality balance.

</details>

---

## Q5. How do you train and evaluate a query complexity classifier for Adaptive RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Training data collection:**

1. Sample 500–2000 queries from your production logs or a representative dataset.
2. Label each query with a complexity class: Simple (0), Moderate (1), or Complex (2). Two complementary sources:
   - **Manual annotation** — Simple: facts, definitions, single-entity questions. Moderate: comparisons, aggregations, single-hop reasoning. Complex: multi-hop reasoning, temporal reasoning, constraint satisfaction.
   - **Silver labels from past system behavior** (cheaper, scales better) — replay logged queries through all three strategies offline and label each with the *cheapest strategy whose answer was correct* (exact match / F1 vs. gold, or LLM-judge vs. the accepted production answer). This is the Adaptive-RAG paper's labeling scheme and automatically reflects your LLM's parametric knowledge: a query is "simple" only if *your* model answers it without retrieval.
3. Deduplicate near-identical queries and stratify the train/test split by class — production logs skew heavily toward simple queries, and an unstratified split under-trains the complex class.

**Classifier architecture:**

A lightweight model such as:
- **Logistic regression** on handcrafted features (length, entity count, keyword TF-IDF).
- **Shallow neural net** (1–2 hidden layers) on query embeddings.
- **Fine-tuned small LM** (e.g., DistilBERT, MobileBERT) for more accuracy at higher cost.

For production, prefer simpler models with <10ms latency.

**Training procedure:**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    query_features, complexity_labels, test_size=0.2
)

clf = RandomForestClassifier(n_estimators=50, max_depth=5)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
```

**Calibration (do not skip this):**

Routing thresholds only mean something if the classifier's probabilities are calibrated — a raw softmax score of 0.9 from an overconfident neural classifier may correspond to only 70% empirical accuracy. Apply **temperature scaling** on a held-out calibration set:

```python
import numpy as np
from scipy.optimize import minimize_scalar

def nll(T, logits, labels):
    probs = np.exp(logits / T) / np.exp(logits / T).sum(axis=1, keepdims=True)
    return -np.log(probs[np.arange(len(labels)), labels]).mean()

T_opt = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded",
                        args=(val_logits, val_labels)).x
# Serve calibrated probs: softmax(logits / T_opt)
```

Verify with a reliability diagram or Expected Calibration Error (ECE) before and after. Then add a **confidence gate**: if the calibrated max-class probability is below a floor (e.g., 0.6), do not trust the prediction — route to the safe middle tier (single-hop) instead.

**Fallback behavior on misclassification:**

- **Complex → simple misroutes** (quality risk): catch at runtime with generation-confidence escape hatches — if the no-retrieval answer's token-level confidence is low or FLARE triggers (see Q4/Q6), escalate to retrieval rather than returning the answer.
- **Simple → complex misroutes** (cost risk): tolerate them; they only waste money. Cap multi-hop iterations and log them for retraining.
- **Classifier unavailable or low-confidence**: default everything to single-hop — it is the strategy with the best worst-case quality/cost balance.
- Log every escalation (predicted tier ≠ tier that finally produced the answer) — these are free hard-negative training examples for the next classifier version.

**Evaluation metrics:**

- **Accuracy** — Overall classification correctness.
- **Per-class precision/recall** — Ensure the classifier does not systematically mis-label complex queries as simple (dangerous for quality).
- **Latency** — Measure classifier inference time in the serving pipeline.
- **End-to-end RAG quality** — Log F1/NDCG/answer-quality metrics binned by predicted complexity; verify that simpler queries routed to no-retrieval maintain acceptable quality.

Monitor classifier performance quarterly and retrain as query patterns shift.

</details>

---

## Q6. How does confidence-based no-retrieval skipping reduce latency and cost? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Confidence-based no-retrieval skipping leverages the LLM's token-level confidence estimates to decide, on a per-query basis, whether retrieval is necessary. Queries the model feels confident about skip retrieval entirely, cutting latency by ~500ms–2s and eliminating embedding/vector DB costs.

**Implementation:**

1. **First-pass generation without retrieval** — Feed the query directly to the LLM and collect output tokens with their probabilities.
2. **Aggregate confidence** — Compute a single confidence score from the token probabilities:
   ```
   confidence = exp(mean(log(p_i)))  # geometric mean of token probs
   ```
   Or use the minimum token probability as a conservative lower bound.
3. **Confidence threshold** — If confidence > threshold (e.g., 0.7), return the answer. Otherwise, retrieve and regenerate.
4. **Empirical threshold tuning** — On a validation set, measure F1/BLEU for each threshold and pick the one maximizing F1 subject to a latency constraint.

**Cost and latency impact:**

| Scenario | Latency (ms) | Retrieval Cost | Confidence Score |
|----------|--------------|-----------------|------------------|
| No-retrieval (direct) | 100 | $0 | High (>0.75) |
| With retrieval | 1500 | $0.05 | Low (<0.75) |
| Skip rate (% of queries) | — | 30–50% reduction | Depends on query distribution |

For a 30% skip rate, total inference cost drops ~15% and median latency improves ~300ms.

**Trade-offs:**

- **False positives** — Low confidence on a query the model can actually answer (requires retrieval unnecessarily). Mitigation: conservative threshold.
- **False negatives** — High confidence on a query the model cannot answer well (skips retrieval when needed). Mitigation: use a mixture of confidence signals (token probs + semantic uncertainty).

Confidence-based skipping pairs well with the Adaptive RAG classifier: the classifier provides a coarse routing decision, and confidence gates fine-grained skipping within each tier.

</details>

---

## Q7. How do you implement self-consistency scoring to select among multi-hop retrieval candidates? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Self-consistency scoring evaluates multiple candidate retrieval sequences (e.g., different intermediate query formulations, different retrieve-and-think steps) and selects the trajectory that produces the most consistent and coherent intermediate reasoning, without requiring ground-truth labels.

**Approach:**

1. **Candidate generation** — For a complex query, generate k different chains of retrieval + reasoning:
   - Retrieval sequence A: retrieve for subquery A1 → reason → retrieve for A2 → answer.
   - Retrieval sequence B: retrieve for subquery B1 → reason → retrieve for B2 → answer.
   - ... (e.g., k=3).

2. **Self-consistency metrics** — Score each trajectory by:
   - **Semantic coherence** — Measure how well consecutive reasoning steps align (embeddings of consecutive reasoning strings are near each other in latent space).
   - **Document relevance agreement** — Score how much the retrieved documents at each step reinforce each other (use citation overlap, entity overlap, or embedding similarity).
   - **Answer stability** — If you regenerate the answer from each trajectory, how similar are the final answers? (e.g., BLEU or semantic similarity).

3. **Voting / aggregation** — Pick the trajectory with the highest aggregate score, or ensemble the answers from all trajectories.

**Example implementation:**

```python
from sentence_transformers import CrossEncoderModel

def score_trajectory(reasoning_steps, retrieved_docs):
    coherence_score = 0.0
    for i in range(len(reasoning_steps) - 1):
        # Cross-encoder score between consecutive reasoning steps
        score = cross_encoder.predict(
            [[reasoning_steps[i], reasoning_steps[i+1]]]
        )[0]
        coherence_score += score
    
    doc_relevance = sum(
        cross_encoder.predict([[step, doc] for doc in retrieved_docs])
        for step in reasoning_steps
    ) / len(retrieved_docs)
    
    return coherence_score + doc_relevance

best_trajectory = max(
    trajectories, 
    key=lambda t: score_trajectory(t['steps'], t['docs'])
)
return best_trajectory['answer']
```

**Benefits:**

- Reduces hallucination in multi-hop reasoning without requiring an oracle.
- Adapts to new query types (no task-specific fine-tuning needed).

**Overhead:** Generating k trajectories multiplies compute cost by k; typically k=2–3 is practical.

</details>

---

## Q8. What are the latency vs. accuracy trade-offs of each adaptive strategy tier? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Strategy | Latency (p50) | Cost/Query | Answer Quality (F1) | When to Prefer |
|----------|---------------|-----------|---------------------|-----------------|
| No-Retrieval | ~50ms | $0.00 | 0.65–0.75 (varies by LLM) | Simple factoid queries; cost-sensitive |
| Single-Hop | ~600ms | $0.01–0.02 | 0.75–0.85 | Moderate complexity; balance of speed/quality |
| Multi-Hop | ~2000ms | $0.05–0.10 | 0.85–0.95 | Complex reasoning; quality-critical |

**Empirical trade-off curve:**

For a typical production workload:

```
F1
│
0.95 │                  Multi-Hop ●
     │
0.85 │        Single-Hop ●
     │
0.75 │ No-Retrieval ●
     │
0.65 │
     └──────────────────────────── Latency (ms)
       0    600   1200   2000
```

**Optimal operating point:**

- Query complexity distribution heavily skews simple → weighted average latency favors no-retrieval path.
- If 50% simple, 30% moderate, 20% complex → median latency ≈ 0.5×50 + 0.3×600 + 0.2×2000 ≈ 650ms.

**Strategies to optimize each tier:**

- **No-retrieval** — Use a smaller, faster LLM (e.g., Llama 7B vs. 70B). Add a re-ranker to catch low-confidence predictions.
- **Single-hop** — Cache embeddings of frequently-retrieved documents. Use approximate nearest neighbor search (HNSW, Annoy).
- **Multi-hop** — Prune intermediate retrieval results aggressively. Use speculative decoding to speed up verifier LLM.

**Cost breakdown for 1M queries/month with 50/30/20 distribution:**

- No-retrieval (500k): $0.
- Single-hop (300k): $0.015 × 300k ≈ $4,500.
- Multi-hop (200k): $0.075 × 200k ≈ $15,000.
- **Total:** ~$19,500/month.

The no-retrieval path is the biggest lever for cost reduction.

</details>

---

## Q9. How do you evaluate an Adaptive RAG system using the metrics from the original Adaptive-RAG paper? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The Adaptive-RAG paper proposes a set of metrics to jointly evaluate routing accuracy, retrieval efficiency, and final answer quality:

**1. Routing Accuracy (RA):**

Measure how often the classifier correctly routes a query to the optimal strategy:

```
RA = (# correctly routed queries) / (total queries)
```

A query is "correctly routed" if:
- Assigned to no-retrieval AND the LLM can answer it without retrieval (confidence > threshold or F1 match).
- Assigned to single-hop AND retrieval+generation beats no-retrieval but multi-hop yields only marginal gains.
- Assigned to multi-hop AND multi-hop significantly outperforms single-hop.

Define "correctly" empirically using a small ground-truth validation set with oracle labels.

**2. Retrieval Efficiency (RE):**

Measure the fraction of queries that skipped retrieval:

```
RE = (# no-retrieval queries) / (total queries)
```

Higher RE = more cost savings. A system with 40% RE avoids 40% of retrieval calls.

**3. Answer Quality (AQ) — Per-tier:**

Report F1, BLEU, or ROUGE for each strategy tier:

```
F1_no_ret = F1(answers on no-retrieval queries)
F1_single = F1(answers on single-hop queries)
F1_multi = F1(answers on multi-hop queries)
```

Ensure no-retrieval F1 is still acceptable (not degraded due to incorrect routing).

**4. Overall Quality vs. Cost:**

Define a joint metric:

```
AdaptiveQuality = (1 - α) × AQ_overall + α × (1 - normalized_cost)
```

where α ∈ [0, 1] trades off answer quality vs. cost. Adaptive-RAG should maximize this metric relative to a fixed pipeline (single-hop or multi-hop baseline).

**Evaluation protocol:**

1. Collect ~500 queries with ground-truth answers.
2. Bin queries by oracle complexity (simple, moderate, complex).
3. For each query, run all three strategies (no-ret, single-hop, multi-hop) and measure quality.
4. Train the classifier to predict oracle complexity.
5. Evaluate:
   - Routing Accuracy = % of queries the classifier routes to the oracle-optimal tier.
   - Retrieval Efficiency = % routed to no-retrieval.
   - Quality per tier = F1/BLEU for each routed subset.
   - Overall cost and latency.

**Example results:**

```
Routing Accuracy: 87%
Retrieval Efficiency: 45%
F1 (no-retrieval): 0.72
F1 (single-hop): 0.81
F1 (multi-hop): 0.91
Overall F1: 0.82
Cost per query: $0.035 (vs. $0.10 for always-multi-hop)
```

</details>

---

## Q10. Design a production Adaptive RAG deployment that handles query routing at 500 QPS. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Architecture:**

```
User Query (500 QPS)
    │
    ├─ [Classification Service] (inference, batch size 32)
    │     │
    │     └─ Outputs: complexity score, confidence interval
    │
    ├─────────────────────────┬──────────────────────┬─────────────────┤
    │                        │                      │                 │
    ▼                        ▼                      ▼                 ▼
[Direct LLM]          [Retrieval SVC]       [Multi-Hop Orchestrator]
(No-Retrieval)        (Single-Hop)          (Multi-Step + Reasoning)
~100ms, $0/query      ~600ms, $0.015/q      ~2000ms, $0.075/q
    │                        │                      │                 │
    └────────────────────────┴──────────────────────┴─────────────────┘
                                    │
                                    ▼
                            [Response Aggregator]
                                    │
                                    ▼
                            [Client Response]
```

**Component design:**

1. **Classification Service:**
   - Model: Lightweight classifier (e.g., DistilBERT or logistic regression).
   - Batch size: 32 (allows 500 QPS with ~16ms latency per batch).
   - Container: GPU-optimized inference (NVIDIA Triton or vLLM).
   - Replicas: 2–3 for HA. Scaling rule: add replica if latency p99 > 50ms.

2. **Direct LLM Service (No-Retrieval Tier):**
   - Model: Smaller, faster LLM (e.g., Llama 7B) or a cached/quantized version of the main model.
   - Throughput: 500 tokens/sec × 32 batch size ≈ 16K tokens/sec. Use vLLM or TensorRT for fast batch inference.
   - Replicas: 1–2. Most queries route here, so this is the throughput bottleneck.

3. **Retrieval Service (Single-Hop Tier):**
   - Query embedding: Fast embedding model (e.g., BGE-small, <10ms).
   - Vector DB: Qdrant or Weaviate with approximate nearest neighbors (HNSW). Cache top-100 embeddings per day to reduce latency.
   - Batch retrieval: Retrieve for ~100 queries in parallel; typical response: <300ms.
   - Replicas: 2–3 to handle 30% of queries (~150 QPS).

4. **Multi-Hop Orchestrator (Complex Queries):**
   - Implement as a LangGraph workflow or custom agentic loop.
   - Parallel retrieval: Fan out multiple sub-queries to the Retrieval Service.
   - Result caching: Cache common sub-queries (e.g., "What is X?") across requests.
   - Replicas: 1–2 for the remaining ~100 QPS (20% of traffic).

**Load balancing:**

```
Load Balancer (nginx / AWS ALB)
    │
    ├─ Classify 500 QPS (round-robin to 2–3 classifiers)
    │
    └─ Route by predicted complexity:
        ├─ 50% → Direct LLM (2–3 replicas)
        ├─ 30% → Retrieval Service (2–3 replicas)
        └─ 20% → Multi-Hop (1–2 replicas)
```

**Latency SLO and monitoring:**

- **P50 latency target:** 400ms (30% on fast path, 70% blended).
- **P99 latency target:** 3000ms (multi-hop queries).
- **Metrics to log:**
  - Classifier latency, accuracy per bin.
  - Per-tier throughput, latency, cost/query.
  - Answer quality (F1) per tier.
  - Cache hit rate (for retrieval and multi-hop).

**Cost estimation (monthly, 1.3B queries at 500 QPS):**

- Classifier inference: $100 (GPU time, shared).
- Direct LLM: 650M queries × $0.0001/query (batch inference) ≈ $65K.
- Retrieval (embedding + vector DB): 390M queries × $0.015 ≈ $5.8K.
- Multi-hop (2–3 LLM calls per query): 260M queries × $0.06 ≈ $15.6K.
- **Total:** ~$87K/month (vs. $130K for always-multi-hop).

**Failure modes and recovery:**

- Classifier outage → default to single-hop for all queries (safe fallback).
- Retrieval timeout → escalate to multi-hop or return LLM-only answer with lower confidence flag.
- Multi-hop timeout (>5s) → return best-effort answer from intermediate steps.

</details>

---

## Q11. How do you quantify and reduce the cost of running a query-complexity classifier on every request, and when does routing overhead outweigh its savings? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Adaptive RAG pays the classifier cost on **every** query in exchange for savings on the *fraction* of queries it downgrades to cheaper tiers. The system is only worth running when:

```
C_classifier + E[cost of misroutes] < E[savings from cheaper routing]
```

**Cost of classifier options:**

| Classifier | Cost/query | Latency | Hardware | Routing accuracy (typical) |
|---|---|---|---|---|
| Logistic regression on features | ~$0.000001 | <1ms | CPU | 75–82% |
| Small BERT (DistilBERT-class) | ~$0.00001 | 5–10ms | CPU/GPU | 82–88% |
| T5-large (paper's choice) | ~$0.0001 | 20–50ms | GPU | 85–90% |
| LLM-based router (prompted small LLM) | $0.0002–0.001 | 300–800ms | API | 85–92% |

The LLM-based router is 100–1000x more expensive per query than a small BERT and adds visible latency to every request — it only makes sense for low-traffic, high-value workloads or for *bootstrapping labels* to train a cheap classifier.

**Amortization math — worked example (1M queries/month):**

Baseline: always-multi-hop at $0.075/query → **$75,000/month**.

Adaptive with a small BERT classifier, 50/30/20 routing (no-ret $0 / single-hop $0.015 / multi-hop $0.075):

```
Classifier:   1M × $0.00001                  =     $10
No-retrieval: 500K × $0.00                   =      $0
Single-hop:   300K × $0.015                  =  $4,500
Multi-hop:    200K × $0.075                  = $15,000
Total                                        ≈ $19,510/month  (74% savings)
```

The classifier itself is **0.05% of total spend** — its inference cost is noise. The real costs to watch are:

1. **Misroute cost** — at 87% routing accuracy, ~13% of queries are misrouted. Simple→complex misroutes waste money (~upper bound: 13% × $0.075 × 1M ≈ $9.7K worst case); complex→simple misroutes cost *quality*, which can dwarf dollar costs (bad answers, escalations, lost trust). Always quality-adjust the savings claim.
2. **GPU amortization** — a dedicated T4 at ~$0.50/hr serving the classifier is ~$360/month regardless of volume. At 1M queries/month that's $0.00036/query; at 100M it's negligible. Self-hosted classifiers have a *fixed floor*, API-based ones scale to zero.
3. **Maintenance** — labeling, retraining, threshold re-sweeps: budget engineer-hours, not just compute.

**When routing overhead outweighs savings:**

- **Homogeneous workloads** — if 95% of queries are genuinely complex (e.g., a research-assistant product), the classifier saves on only 5% of traffic; a fixed multi-hop pipeline is simpler and nearly as cheap.
- **Cheap downstream pipeline** — if single-hop costs $0.001/query, even perfect routing saves fractions of a cent; an LLM-based router would cost *more than the pipeline it gates*.
- **Strict latency budgets** — an LLM router adding 500ms to a 600ms pipeline is an 80% latency tax on every request, including the ones it doesn't help.
- **Low traffic** — below ~100K queries/month, fixed costs (GPU floor, retraining, monitoring) usually exceed the routed savings.

**Caching routing decisions:**

```python
import hashlib

def route_with_cache(query: str) -> str:
    # Exact-match cache on normalized query
    key = hashlib.sha256(normalize(query).encode()).hexdigest()
    if key in route_cache:
        return route_cache[key]                      # ~free

    # Semantic cache: reuse route for near-paraphrases
    emb = embed(query)                               # cheap model, ~1ms
    hit = semantic_cache.search(emb, min_sim=0.95)
    if hit:
        return hit.route

    route = classifier.predict(query)
    route_cache[key] = route
    semantic_cache.add(emb, route)
    return route
```

- Production query streams are repetitive: 20–40% exact/semantic hit rates are common, cutting classifier load proportionally.
- Cache the **route**, not the answer — the answer may be stale, but "this query shape needs single-hop" is stable.
- Set a TTL and **flush the cache when the classifier is retrained or thresholds change**, otherwise old routing decisions outlive the model that made them.

**Rule of thumb:** keep classifier cost under ~1% of average per-query pipeline cost. A small supervised model nearly always satisfies this; an LLM-as-router rarely does at scale.

</details>

---

## Q12. How can adversaries manipulate the query-complexity classifier to force misrouting, and what defenses keep Adaptive RAG robust? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The routing classifier is a new, externally reachable decision surface: every user-controlled query passes through it, and its output changes both *answer quality* and *money spent*. That makes it a target for two opposite attack classes.

**Attack 1: Downgrade attack (force no-retrieval → elicit hallucinations)**

The attacker phrases knowledge-dependent questions in the surface form of simple factoid queries, so the classifier routes them to the no-retrieval path and the LLM answers from (insufficient) parametric knowledge:

```
Honest query:      "What does our 2026 enterprise SLA say about refunds
                    for multi-region outages?"          → multi-hop (grounded)

Adversarial query: "Define the refund policy."          → no-retrieval
                   "What is the 2026 SLA refund rule?"  → no-retrieval (factoid shape)

LLM answers confidently from parametric guesswork → hallucinated policy text.
```

Why it matters: the attacker can harvest authoritative-sounding fabrications (to screenshot, to mislead other users in shared channels, or to probe what the base model "believes" about private topics), and the system skips exactly the grounding step that would have caught it. Classifiers trained on surface features (length, question words, entity count) are especially easy to steer this way.

**Attack 2: Complexity-inflation attack (cost / resource DoS)**

The opposite direction — stuff queries with multi-hop trigger features ("compare", "across", "considering", many named entities, nested clauses) so every query lands on the most expensive path:

```
"Compare X and Y across A, B, and C, considering D, E, and F,
 and how each evolved relative to G..."   → multi-hop, every time

Economics: cheap path $0.00/query vs. multi-hop $0.075/query (~50–100x).
10K adversarial queries/day × $0.075 ≈ $750/day ≈ $22K/month of attacker-
controlled spend — an economic DoS that also saturates multi-hop
orchestrator capacity and degrades latency for legitimate complex queries.
```

**Attack 3: Boundary probing**

Tier latencies differ by an order of magnitude (~100ms vs. ~2s), so response time is a **timing side channel** revealing which tier ran. An attacker can binary-search query phrasings to map the routing thresholds, then reliably sit just on the cheap side (for downgrades) or expensive side (for inflation) of the boundary.

**Defenses:**

**1. Calibrated confidence thresholds + fallback to the safe path**

Never act on a low-confidence routing decision, and make the *uncertain* default the grounded middle tier — not the cheap tier:

```python
def route(query: str) -> str:
    probs = calibrated_classifier.predict_proba(query)  # temperature-scaled
    label, conf = probs.argmax(), probs.max()

    if conf < 0.60:
        return "single_hop"          # uncertain → safe, grounded default
    if label == "simple" and conf < 0.80:
        return "single_hop"          # extra-strict bar for skipping retrieval
    return TIERS[label]
```

The asymmetric bar matters: a downgrade attack must now push the classifier to ≥0.8 calibrated confidence on "simple", not merely win the argmax.

**2. Policy overlay before the classifier**

Rules outrank the model. Queries touching dynamic, private, or high-stakes domains (policies, pricing, anything post-training-cutoff, anything matching tenant document namespaces) are **never eligible for no-retrieval**, regardless of classifier output. The downgrade attack then can't reach the vulnerable path at all for the content that matters.

**3. Cost guards against inflation**

- Per-user / per-tenant **budget caps and rate limits on the multi-hop tier** specifically (not just global QPS).
- Hard cap on hops and per-query token budget; degrade to single-hop with a "partial answer" flag rather than looping.
- Require account trust level (age, payment status) for unmetered multi-hop access; route anonymous traffic to capped tiers.

**4. Monitor routing-distribution drift**

A routing attack *is* a distribution shift. Track tier shares per window and per user segment against a trusted baseline, e.g., with Population Stability Index:

```python
def psi(baseline: dict, current: dict) -> float:
    return sum(
        (current[t] - baseline[t]) * math.log(current[t] / baseline[t])
        for t in ["none", "single", "multi"]
    )

# baseline = {"none": 0.50, "single": 0.30, "multi": 0.20}
# PSI > 0.2 on an hourly window → alert.
# Slice by user/tenant/IP: a single tenant at 95% multi-hop share is a
# stronger signal than a global shift.
```

Also alert on: spike in no-retrieval share for queries containing tenant-document entities (downgrade signature), and spike in escalations (FLARE/confidence triggers firing after a "simple" route — the classifier is being beaten).

**5. Adversarial training and consistency checks**

- Add attack-shaped examples (factoid-phrased private-knowledge questions labeled *complex*; keyword-stuffed simple questions labeled *simple*) to the classifier's training set each retraining cycle.
- **Paraphrase consistency test:** route 2–3 cheap paraphrases of suspicious queries; if routes disagree, take the most conservative route and log the query.

**6. Catch successful downgrades post-hoc**

Defense-in-depth for when the classifier is beaten anyway: on the no-retrieval path, run a lightweight groundedness/confidence check on the generated answer (token-probability floor, or FLARE-style uncertainty trigger from Q4/Q6) and escalate to retrieval before returning. The attacker must now fool the classifier *and* the generation-time gate.

**Attack → defense map:**

| Attack | Primary defense | Backstop |
|---|---|---|
| Downgrade (force no-retrieval) | High confidence bar + policy overlay | Generation-time confidence gate, escalate |
| Complexity inflation (cost DoS) | Per-tenant multi-hop budgets, hop caps | Routing-share monitoring per segment |
| Boundary probing | Calibrated thresholds, paraphrase consistency | Jitter/normalize response timing; rate-limit probers |

The principle: the classifier is an *optimization*, never a *safety control*. Quality and cost guarantees must hold even when the router gives the worst possible answer.

</details>

---

## Real-World Applications

| Application | Domain | Why Adaptive RAG Fits |
|---|---|---|
| General-purpose AI assistant (e.g., ChatGPT with browsing, Claude with tools) | Consumer / Enterprise | Query complexity varies enormously — a "what's 2+2?" needs no retrieval while "summarize this year's AI papers" needs deep multi-hop search |
| Enterprise help desk / IT self-service portal | Enterprise IT | Simple password-reset questions skip retrieval; complex "why is my VPN failing after the latest update?" routes to full agentic search |
| E-learning platform with adaptive tutoring | EdTech | Simple recall questions are answered from parametric knowledge; novel problem-solving routes to retrieved worked examples |
| Customer success platform (e.g., Zendesk AI) | SaaS / Support | Quick FAQs answered immediately; nuanced billing disputes route to policy retrieval and escalation logic |
| Internal developer tool (code Q&A + generation) | DevTools | Simple syntax questions need no retrieval; architecture-level questions trigger full codebase search and retrieval |
