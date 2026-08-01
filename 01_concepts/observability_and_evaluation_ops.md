# Observability & Evaluation Ops: Running RAG Evaluation in Production

> Offline metrics tell you a change is good in the lab; observability tells you it's still good on Tuesday at 3 PM with real users.

This file covers the *operational* side of evaluation: LLM-as-judge patterns, online metrics, tracing, and drift alerting. For the offline metrics themselves (Precision@k, Recall@k, NDCG, MRR, RAGAS, BERTScore), see [Evaluation Metrics](./evaluation_metrics.md). For the high-level observability design principle (what to instrument and the dashboard layout), see [System Design Principles](../00_overview/system_design_principles.md) — this file goes one level deeper into *how* those numbers get produced and acted on.

---

## What is Observability (in RAG)?

Observability is the practice of continuously monitoring a live RAG system's real-world behavior — tracing individual requests, tracking online quality signals, and alerting on drift — rather than relying only on offline benchmark scores computed before deployment. Offline metrics tell you a change looked good in the lab; observability tells you whether it's still holding up in production with real users and real data, days or weeks later.

---

## The Production Evaluation Loop

Offline evaluation is a gate; online evaluation is a feedback loop. A production RAG system needs both, connected.

```
┌────────────────────────────────────────────────────────────────┐
│                     OFFLINE (pre-deploy)                        │
│                                                                 │
│  Golden Dataset ──► RAGAS / Recall@k ──► CI Gate ──► Deploy    │
│       ▲                                                │        │
│       │  (promote hard cases into golden set)          │        │
└───────┼────────────────────────────────────────────────┼────────┘
        │                                                ▼
┌───────┼────────────────────────────────────────────────────────┐
│       │              ONLINE (post-deploy)                       │
│       │                                                         │
│  Failed Traces ◄── Alerting ◄── Drift Detection                │
│       ▲                              ▲                          │
│       │                              │                          │
│  Traces (every request) ──► Sampled LLM-as-judge scoring        │
│       ▲                              ▲                          │
│       │                              │                          │
│  User feedback (thumbs, regenerations, escalations)             │
└────────────────────────────────────────────────────────────────┘
```

**The loop:** every request emits a trace → a sample of traces gets judged → judge scores and user signals feed dashboards and alerts → failing traces get triaged and promoted into the golden dataset → the golden dataset gates the next deploy.

---

## LLM-as-Judge Patterns

Human labeling doesn't scale to production traffic. LLM judges do — but they're biased instruments that must be calibrated before you trust them.

### The Three Judging Patterns

| Pattern | Mechanism | Output | Best For | Cost |
|---|---|---|---|---|
| **Pointwise (rubric)** | Judge scores one answer against an explicit rubric | Score (e.g., 1–5) + justification | Continuous monitoring; absolute quality tracking | 1 call per sample |
| **Pairwise comparison** | Judge sees two answers (A vs. B), picks the better | Preference (A / B / tie) | A/B tests; comparing model or prompt versions | 1 call per pair (×2 with order swap) |
| **Jury of judges** | 3+ different models score independently; aggregate by vote or mean | Consensus score | High-stakes evals; reducing single-judge noise | 3–5× pointwise |

**Pointwise** is your workhorse for production monitoring (you can trend it over time). **Pairwise** is more reliable for *decisions* — judges are better at "which is better?" than "how good is this on a 1–5 scale?" — but produces no absolute trend line. **Jury-of-judges** trades cost for variance reduction: disagreement among jurors is itself a useful signal (route disagreements to humans).

### Known Judge Biases and Mitigations

| Bias | Symptom | Mitigation |
|---|---|---|
| **Position bias** | In pairwise, answer shown first (or last) wins more often regardless of quality | Run every comparison twice with order swapped; count only consistent verdicts, mark flips as ties |
| **Verbosity bias** | Longer answers score higher even when padded | Length-normalize or instruct the rubric to penalize unsupported padding; track score-vs-length correlation |
| **Self-preference bias** | A model rates its own outputs higher than other models' | Use a judge from a *different model family* than the generator |
| **Score clustering** | Judge gives 4/5 to almost everything | Use fewer, behaviorally-anchored score levels (e.g., 1/2/3 with explicit definitions per level) |
| **Sycophancy to phrasing** | Confident-sounding wrong answers score higher | Require the judge to verify claims against context *before* scoring (chain-of-verification) |

### Calibrating a Judge Before Trusting It

Never deploy a judge blind. Calibrate it against a small human-labeled set first:

1. Have humans label 50–200 traces with the same rubric the judge will use.
2. Run the judge on the same traces.
3. Measure agreement (Cohen's kappa or Spearman correlation with human scores).
4. **Kappa > 0.7:** trust the judge for trend monitoring. **0.4–0.7:** use only for triage (flag low scores for human review). **< 0.4:** fix the rubric or change the judge model.
5. Re-calibrate whenever you change the rubric, the judge model, or the domain.

### Code: Rubric-Based Faithfulness Judge

```python
FAITHFULNESS_RUBRIC = """You are evaluating whether an answer is faithful
to the provided context. Faithful = every factual claim in the answer is
directly supported by the context.

Context:
{context}

Question: {question}
Answer: {answer}

Steps:
1. List each factual claim in the answer.
2. For each claim, state SUPPORTED or UNSUPPORTED, quoting the
   supporting context span if SUPPORTED.
3. Output a final score:
   3 = all claims supported
   2 = minor unsupported detail, core answer supported
   1 = a central claim is unsupported (hallucination)

Output JSON: {{"claims": [...], "score": <1|2|3>, "reason": "..."}}"""

def judge_faithfulness(question, context, answer, judge_llm):
    """Pointwise rubric judge. Use a different model family than
    the generator to avoid self-preference bias."""
    prompt = FAITHFULNESS_RUBRIC.format(
        context=context, question=question, answer=answer
    )
    result = json.loads(judge_llm.generate(prompt, temperature=0))
    return result["score"], result["reason"]
```

Note the design choices: chain-of-verification (list claims first, score last), behaviorally-anchored 3-point scale (resists score clustering), temperature 0, and structured output for aggregation.

---

## Online Metrics: Measuring Quality on Live Traffic

You can't run Recall@k on live traffic — there are no relevance labels. Online quality is inferred from user behavior.

### Explicit vs. Implicit Signals

| Signal | Type | What It Indicates | Caveat |
|---|---|---|---|
| Thumbs up / down | Explicit | Direct quality judgment | <1–5% of users respond; heavily biased toward angry users |
| Regeneration rate | Implicit | First answer unsatisfying | Some regens are exploration, not dissatisfaction |
| Follow-up / rephrase rate | Implicit | Answer didn't resolve the question | Distinguish rephrases (bad) from natural follow-ups (fine) |
| Dwell time on answer | Implicit | Engagement (very short = bounce) | Long dwell can also mean "confusing answer" |
| Escalation-to-human rate | Implicit | RAG failed to deflect (support bots) | The single best business-aligned metric for support use cases |
| Citation click-through | Implicit | User trusts/verifies sources | Low CTR can mean "answer was sufficient" — interpret with care |
| Session abandonment | Implicit | User gave up | Strongest negative signal, but noisy |

**Rule of thumb:** never trust a single implicit signal. A *composite* (e.g., "no regeneration AND no rephrase AND no escalation within session") is a far more reliable proxy for a good answer.

### Why Online and Offline Metrics Diverge

This is a classic interview probe. Common causes:

1. **Distribution shift:** the golden dataset is cleaner, shorter, and better-spelled than real queries. Real users typo, paste logs, and ask out-of-scope questions.
2. **Offline metrics measure correctness; users measure usefulness.** A faithful, relevant answer that's three paragraphs of hedging gets a thumbs-down.
3. **Goodharting the eval set:** prompt tweaks tuned against a fixed golden set overfit to it.
4. **Missing-content cases:** offline sets only contain answerable questions; online traffic includes questions the corpus simply can't answer (see [retrieval failure](../03_failure_modes/02-retrieval_failure.md)).
5. **Latency:** offline evals ignore it; users abandon slow answers regardless of quality.

When they diverge, trust the online signal for *whether* there's a problem and the offline harness for *where* it is.

### Guardrail Metrics vs. Quality Metrics

| | Quality Metrics | Guardrail Metrics |
|---|---|---|
| **Purpose** | Maximize | Must not breach |
| **Examples** | Judge faithfulness score, thumbs-up rate, deflection rate | P95 latency, cost/query, error rate, PII-leak rate, toxicity rate, refusal rate |
| **Deploy decision** | "New version should improve these" | "New version must not regress these, even if quality improves" |
| **Alerting** | Trend dashboards, weekly review | Page on-call immediately |

A reranker that lifts faithfulness 3% but pushes P95 from 280ms to 600ms fails the guardrail and doesn't ship.

---

## Tracing & Tooling

Aggregate metrics tell you *that* quality dropped; traces tell you *why*. A RAG trace must capture every intermediate artifact so any answer can be replayed and debugged.

### Anatomy of a RAG Trace

```
trace_id: 7f3a-...        user_id (hashed)        timestamp
│
├─ span: query_processing
│   ├─ raw_query:       "how do i reset my plan?"
│   └─ rewritten_query: "how to reset subscription plan"
│
├─ span: retrieval                                 [22ms]
│   └─ chunks: [(id=doc_412#3, score=0.87),
│               (id=doc_089#1, score=0.81),
│               (id=doc_412#4, score=0.74), ...]   ← top-k IDs + scores
│
├─ span: reranking                                 [85ms]
│   └─ reranked: [(doc_089#1, 0.94), (doc_412#3, 0.71), ...]
│
├─ span: generation                                [1240ms]
│   ├─ final_prompt:    (full assembled prompt, incl. system + context)
│   ├─ model + params:  temperature, max_tokens
│   ├─ response:        "To reset your plan, go to..."
│   └─ token_usage:     in=1840, out=156
│
└─ span: evaluation (async, sampled)
    ├─ judge_faithfulness: 3/3
    ├─ judge_relevance:    2/3
    └─ user_feedback:      thumbs_up (arrives later, joined by trace_id)
```

Without the chunk IDs *and* scores at each stage, you cannot answer the most common debugging question: "was the right chunk retrieved and then lost in reranking, or never retrieved at all?" (See [reranker failure](../03_failure_modes/06-reranker_failure.md).)

### Tooling Comparison

| Tool | Hosting | RAG-Specific Features | Eval Integration | When to Choose |
|---|---|---|---|---|
| **LangSmith** | SaaS (self-host on enterprise) | Deep LangChain/LangGraph integration, prompt playground, datasets | Built-in LLM-as-judge evaluators, annotation queues | You're already on LangChain and want the least setup |
| **Arize Phoenix** | Open-source, self-host or SaaS | Embedding drift visualization (UMAP), retrieval analysis | Built-in evals, strong drift/embedding analysis | You care about embedding drift detection specifically |
| **Langfuse** | Open-source, self-host or SaaS | Framework-agnostic SDK, prompt management, cost tracking | Score API for custom judges, human annotation | You want open-source, framework-neutral, self-hosted |
| **TruLens** | Open-source library | "RAG triad" (context relevance, groundedness, answer relevance) feedback functions | Evals are the core product | You want opinionated eval functions more than tracing infra |
| **OpenTelemetry DIY** | Self-host (any OTel backend) | None out of the box — you define GenAI semantic-convention spans | Build your own | You have an existing observability stack (Grafana/Datadog/Jaeger) and strict data-residency needs |

Most are converging on OpenTelemetry GenAI semantic conventions as the wire format, which reduces lock-in: instrument once with OTel, export to whichever backend.

### RAGAS in CI as a Regression Gate

Run the offline harness on a golden dataset on every PR that touches prompts, chunking, embeddings, or retrieval config — same idea as the `should_deploy` gate in [Evaluation Metrics](./evaluation_metrics.md), wired into CI:

```yaml
# .github/workflows/rag-eval.yml
name: rag-regression-gate
on:
  pull_request:
    paths: ["prompts/**", "retrieval/**", "indexing/**"]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ragas datasets
      - run: python eval/run_golden_set.py --out scores.json
      - run: python eval/check_thresholds.py scores.json
```

```python
# eval/check_thresholds.py — fail the build on regression
THRESHOLDS = {           # floor values; tune against your baseline
    "faithfulness": 0.85,
    "answer_relevance": 0.80,
    "context_recall": 0.75,
}

scores = json.load(open(sys.argv[1]))
failures = [
    f"{m}: {scores[m]:.2f} < {floor:.2f}"
    for m, floor in THRESHOLDS.items() if scores[m] < floor
]
if failures:
    print("EVAL GATE FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
```

**Ops notes:** pin the judge model version (a judge upgrade silently shifts all scores); keep the golden set in version control next to the code; budget for judge nondeterminism by alerting on deltas > ~2–3%, not exact equality; refresh the golden set monthly with promoted production failures.

---

## Drift Detection & Alerting

RAG quality decays without any code change: the corpus goes stale, user query patterns shift, an upstream model gets swapped. Drift detection catches decay between deploys.

### Retrieval-Quality Drift Signals

| Signal | How to Compute | What Drift Means | Linked Failure Mode |
|---|---|---|---|
| **Mean top-k similarity drop** | Daily average of top-1 / mean top-5 retrieval scores | Queries are drifting away from the corpus, or embeddings degraded | [Embedding mismatch](../03_failure_modes/03-embedding_mismatch.md) |
| **Rising "no relevant docs" rate** | % of queries where top score < relevance floor | Users asking about content you don't have | [Retrieval failure](../03_failure_modes/02-retrieval_failure.md) |
| **Embedding distribution shift** | Population distance (e.g., centroid drift, PSI) between this week's query embeddings and the baseline window | New query topics, seasonality, or a new user segment | [Embedding mismatch](../03_failure_modes/03-embedding_mismatch.md) |
| **Index freshness lag** | `now - max(doc.indexed_at)` per source; indexing queue depth | Pipeline silently stopped; answers cite outdated facts | [Stale index problem](../03_failure_modes/04-stale_index_problem.md) |
| **Judge-score moving average** | 7-day rolling mean of sampled faithfulness/relevance scores vs. 28-day baseline | End-to-end quality regression from *any* cause | [Hallucination despite context](../03_failure_modes/01-hallucination_despite_context.md) |
| **Context-length creep** | P95 assembled-prompt tokens over time | Chunking or k changed; truncation risk rising | [Context window overflow](../03_failure_modes/05-context_window_overflow.md) |

### Alert Threshold Design

```
metric value
│   ████ baseline window (28d) ────────────────────────
│   ─────────── WARN: > 2σ from baseline for 24h ─────
│   ─────────── PAGE: > 3σ or absolute floor breached ─
│                                          ╲
│                                           ╲ ← drift
└──────────────────────────────────────────────► time
```

- **Alert on relative change, not just absolute floors.** A faithfulness drop from 0.92 → 0.85 is an incident even if 0.85 is "acceptable."
- **Require persistence** (e.g., 2σ for 24h) for WARN-level signals — judge scores are noisy daily.
- **Page immediately** on guardrail breaches (error rate, freshness lag > SLA, PII leak) — these don't need statistical smoothing.
- **Route alerts with the trace link attached.** An alert that says "faithfulness dropped" without sample traces just creates a triage chore.

### Example: Freshness Alert

```python
from datetime import datetime, timezone

def check_index_freshness(source: str, sla_hours: int = 24):
    """Alert if a source hasn't been re-indexed within SLA.
    Catches the silent-pipeline-death cause of the stale index problem."""
    last_indexed = index_metadata.get_last_indexed_at(source)
    lag_hours = (datetime.now(timezone.utc) - last_indexed).total_seconds() / 3600

    statsd.gauge(f"rag.index.freshness_lag_hours.{source}", lag_hours)
    if lag_hours > sla_hours:
        page(f"Index for '{source}' is {lag_hours:.0f}h stale "
             f"(SLA {sla_hours}h). Check ingestion queue.")
```

---

## Key Takeaways

1. **Connect the loop.** Traces → judged samples → alerts → triaged failures → golden set → CI gate. Each stage feeds the next; any missing stage breaks the cycle.
2. **Calibrate judges before trusting them.** Human-agreement kappa > 0.7, different model family than the generator, swapped orders for pairwise, pinned judge version.
3. **No single online signal is trustworthy.** Composite implicit signals (regeneration + rephrase + escalation) beat thumbs ratios.
4. **Log the full trace, not just Q&A.** Chunk IDs and scores at every stage are what let you localize a failure to retrieval, reranking, or generation.
5. **Alert on drift, not just floors.** Moving-average deltas with persistence requirements catch slow decay (stale index, query drift) that absolute thresholds miss.

---

## Interview Q&A

**Q: How would you know your RAG quality regressed after a deploy?** `[Advanced]`

The canonical answer has four layers — name all four, in order:

1. **Pre-deploy gate:** CI runs the golden-dataset eval (RAGAS + retrieval metrics); the deploy is blocked if any metric drops beyond threshold. Catches regressions from prompt/retrieval/chunking changes.
2. **Canary + guardrails:** roll out to 5–10% of traffic; compare guardrail metrics (latency, error rate, refusal rate) and judge scores canary-vs-control before full rollout.
3. **Online signals:** post-rollout, watch regeneration rate, rephrase rate, thumbs ratio, and escalation rate against the pre-deploy baseline window.
4. **Trace-level triage:** when a signal fires, pull the failing traces, localize the stage (retrieval vs. rerank vs. generation) using per-stage scores, and promote the failures into the golden set so the same regression is caught pre-deploy next time.

---

**Q: Can you just use an LLM judge to grade your answers?** `[Intermediate]`

Not unconditionally. Calibrate the judge against human labels first (Cohen's kappa or Spearman correlation on 50–200 human-labeled traces), use a judge from a different model family than the generator to avoid self-preference bias, and pin the judge model version — a silent provider-side judge upgrade shifts all scores without any change on your end.

---

**Q: Your thumbs-up rate is 92%. Is your RAG good?** `[Intermediate]`

Not necessarily — don't accept the number at face value. Explicit feedback like thumbs typically comes from under 5% of users and is heavily biased toward angry or highly engaged users, so a 92% rate says little about the silent majority. You need composite implicit signals (e.g., no regeneration AND no rephrase AND no escalation within session) to get a reliable read on quality.

---

**Q: Offline evals improved but users are complaining — what's going on?** `[Advanced]`

Don't blame the users. This is the classic online/offline divergence: the golden dataset is cleaner and narrower than live traffic (distribution shift), prompt tweaks may have overfit to the fixed golden set (Goodharting), or online traffic includes unanswerable questions that never appear offline. Check the unanswerable-question rate and compare the query distributions before concluding the model regressed.

---

**Q: How do you evaluate a RAG system without ground-truth labels?** `[Intermediate]`

"We can't" is the wrong answer. Reference-free LLM-as-judge techniques work because faithfulness only needs the context and answer (not a gold answer) to check whether claims are supported. Combine this with behavioral signals from live traffic (regeneration rate, escalation rate, dwell time) to triangulate quality without labeled data.

---

**Q: What do you log per request in a production RAG system?** `[Basic]`

More than "the question and answer." Log the full trace: the raw and rewritten query, retrieved chunk IDs and scores, reranked chunk IDs and scores, the final assembled prompt, model parameters and token usage, and any sampled judge scores. Without per-stage chunk IDs and scores you cannot localize a failure to retrieval, reranking, or generation.

---

**Q: Your judge scores dropped 5% overnight with no deploy — what do you check?** `[Advanced]`

Don't assume the system broke. The drop could come from the *judge* itself — the provider silently updated the underlying model — rather than from query or corpus drift. Score a fixed sentinel set daily (same inputs, same expected scores) so a shift in the sentinel scores isolates judge drift from genuine system drift.
