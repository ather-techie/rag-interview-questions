# RAGAS CI Harness

> A production-ready evaluation harness using RAGAS and custom judges, wired into CI to block regressions before deployment.

---

## Why RAGAS?

RAGAS (Retrieval Augmented Generation Assessment, Es et al., 2023) is the standard framework for evaluating RAG systems. Its key advantage: **no ground-truth answers required**. It measures four properties using the LLM itself as a judge, making it practical for systems where manually labeling hundreds of expected answers is infeasible.

| RAGAS Metric | What It Measures | Gold Labels Required? |
|-------------|-----------------|----------------------|
| **Faithfulness** | Is the answer grounded in the context? | No |
| **Answer Relevance** | Does the answer address the question? | No |
| **Context Precision** | Are retrieved chunks actually relevant? | Yes (relevant chunk IDs) |
| **Context Recall** | Were all relevant chunks retrieved? | Yes (relevant chunk IDs) |

---

## RAGAS Setup

```bash
pip install ragas langchain-anthropic
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import anthropic

# RAGAS uses LangChain under the hood — configure the judge model
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import AnthropicEmbeddings

judge_llm   = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-5"))
judge_embed = LangchainEmbeddingsWrapper(AnthropicEmbeddings())
```

---

## Building the Evaluation Dataset

RAGAS requires a HuggingFace `Dataset` with specific columns:

```python
def build_ragas_dataset(
    golden_samples: list[dict],
    retrieval_fn,
    generation_fn,
) -> Dataset:
    rows = []
    
    for sample in golden_samples:
        query = sample["query"]
        
        # Retrieve
        retrieved = retrieval_fn(query, k=5)
        contexts  = [r["text"] for r in retrieved]
        
        # Generate
        answer = generation_fn(query, contexts)
        
        row = {
            "question":          query,
            "answer":            answer,
            "contexts":          contexts,           # list of retrieved text strings
            "ground_truth":      sample.get("expected_answer", ""),  # optional
            "ground_truths":     [sample.get("expected_answer", "")],
        }
        rows.append(row)
    
    return Dataset.from_list(rows)


def run_ragas_eval(ragas_dataset: Dataset) -> dict:
    """Run all four RAGAS metrics."""
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embed,
    )
    return dict(result)
```

---

## Custom Judge Metrics

For domain-specific quality that RAGAS doesn't cover, add custom LLM-as-judge metrics:

```python
import json
from anthropic import Anthropic

client = Anthropic()

def custom_citation_quality(question: str, answer: str, contexts: list[str]) -> float:
    """Score how well claims in the answer are grounded in specific cited passages."""
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        system="""Score the citation quality of this answer from 0.0-1.0.
1.0 = every claim traceable to a provided context passage
0.5 = some claims grounded, some not
0.0 = no claims grounded in provided contexts

Output JSON: {"score": 0.0-1.0, "ungrounded_claims": ["..."]}""",
        messages=[{"role": "user", "content":
            f"Question: {question}\nContexts: {json.dumps(contexts)}\nAnswer: {answer}"}],
    )
    result = json.loads(resp.content[0].text)
    return result["score"]


def custom_completeness(question: str, answer: str) -> float:
    """Score how completely the answer addresses all aspects of the question."""
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=128,
        system='Score completeness 0.0-1.0. Output JSON: {"score": 0.0-1.0}',
        messages=[{"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}],
    )
    return json.loads(resp.content[0].text)["score"]
```

---

## Full CI Harness

```python
import sys
import json
from pathlib import Path
from datetime import datetime

THRESHOLDS = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.75,
    "context_precision": 0.70,
    "context_recall":    0.70,
    "citation_quality":  0.80,  # custom metric
}

REGRESSION_TOLERANCE = 0.03  # allow 3% drop from baseline before blocking


def run_full_eval_suite(
    golden_path: str,
    retrieval_fn,
    generation_fn,
    baseline_path: str = "08_evaluation/baseline_metrics.json",
    save_results: bool = True,
) -> dict:
    # Load golden dataset
    golden_samples = json.loads(Path(golden_path).read_text())["samples"]
    
    # Build RAGAS dataset
    print(f"Building eval dataset from {len(golden_samples)} samples...")
    ragas_dataset = build_ragas_dataset(golden_samples, retrieval_fn, generation_fn)
    
    # RAGAS metrics
    print("Running RAGAS evaluation...")
    ragas_metrics = run_ragas_eval(ragas_dataset)
    
    # Custom metrics (sample to control cost — 20% of queries)
    import random
    custom_sample = random.sample(golden_samples, max(10, len(golden_samples) // 5))
    custom_scores = []
    for s in custom_sample:
        retrieved = retrieval_fn(s["query"], k=5)
        answer    = generation_fn(s["query"], [r["text"] for r in retrieved])
        cq_score  = custom_citation_quality(s["query"], answer, [r["text"] for r in retrieved])
        custom_scores.append(cq_score)
    
    all_metrics = {
        **ragas_metrics,
        "citation_quality": sum(custom_scores) / len(custom_scores),
        "eval_timestamp":   datetime.utcnow().isoformat(),
        "n_samples":        len(golden_samples),
    }
    
    # Check absolute thresholds
    below_threshold = []
    for metric, threshold in THRESHOLDS.items():
        if metric in all_metrics and all_metrics[metric] < threshold:
            below_threshold.append(f"{metric}: {all_metrics[metric]:.3f} < {threshold}")
    
    # Check regression vs baseline
    regressions = []
    if Path(baseline_path).exists():
        baseline = json.loads(Path(baseline_path).read_text())
        for metric, baseline_val in baseline.items():
            if isinstance(baseline_val, float) and metric in all_metrics:
                drop = baseline_val - all_metrics[metric]
                if drop > REGRESSION_TOLERANCE:
                    regressions.append(f"{metric}: {baseline_val:.3f} → {all_metrics[metric]:.3f} (drop: {drop:.3f})")
    
    # Report
    print("\n=== Evaluation Results ===")
    for metric, value in all_metrics.items():
        if isinstance(value, float):
            status = "✓" if value >= THRESHOLDS.get(metric, 0) else "✗"
            print(f"  {status} {metric}: {value:.3f}")
    
    if below_threshold:
        print("\nBELOW THRESHOLD:")
        for f in below_threshold:
            print(f"  ✗ {f}")
    
    if regressions:
        print("\nREGRESSIONS vs BASELINE:")
        for r in regressions:
            print(f"  ✗ {r}")
    
    if save_results:
        results_path = f"08_evaluation/results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        Path(results_path).write_text(json.dumps(all_metrics, indent=2))
        print(f"\nResults saved to {results_path}")
    
    if below_threshold or regressions:
        print("\nEVALUATION FAILED — blocking deployment")
        sys.exit(1)
    
    print("\nAll checks passed.")
    return all_metrics
```

---

## Updating the Baseline

After a deliberate, validated improvement, update the baseline:

```python
def update_baseline(metrics: dict, baseline_path: str):
    """Update baseline after a deliberate improvement. Run manually, not in CI."""
    baseline = {k: v for k, v in metrics.items() if isinstance(v, float)}
    Path(baseline_path).write_text(json.dumps(baseline, indent=2))
    print(f"Baseline updated: {baseline}")

# Usage (run manually after a validated improvement):
# new_metrics = run_full_eval_suite(...)
# update_baseline(new_metrics, "08_evaluation/baseline_metrics.json")
```

---

## CI Configuration (GitHub Actions)

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation

on:
  pull_request:
    paths:
      - 'src/**'          # code changes
      - 'config/**'       # chunking/embedding config changes

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: pip install ragas langchain-anthropic datasets
      
      - name: Run RAG evaluation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python 08_evaluation/run_ci_eval.py
        
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: 08_evaluation/results_*.json
```

---

## Key Takeaways

1. **RAGAS is the standard** — use it for Faithfulness and Answer Relevance (no gold labels needed).
2. **Add custom metrics** for domain-specific quality dimensions RAGAS doesn't cover.
3. **Regression testing matters more than absolute thresholds** — a system at 0.82 faithfulness that drops to 0.79 needs investigation.
4. **Sample custom metrics** (20–30% of queries) to control LLM-as-judge costs.
5. **Update the baseline intentionally** — only when a validated improvement has been reviewed, not automatically.
