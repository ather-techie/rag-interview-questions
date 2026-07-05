# Golden Dataset Construction

> How to build a labeled evaluation set that actually measures what matters — from query sampling through annotation guidelines to inter-annotator agreement.

---

## Why a Golden Dataset?

Automated metrics (RAGAS Faithfulness, BERTScore, Recall@k) are only as good as the labeled data they're computed against. Without a golden dataset:

- You cannot measure Recall@k (you need to know which passages are relevant)
- You cannot detect regressions before deploying changes
- You cannot compare embedding models or chunking strategies objectively

A golden dataset is a set of (query, relevant_passage_ids, expected_answer) triples representative of your production query distribution.

---

## Step 1: Query Sampling

Sample queries that represent your actual production traffic. Use multiple sources:

```python
import random
from collections import Counter

def sample_queries_from_logs(
    query_logs: list[str],
    n: int = 200,
    strategy: str = "stratified",
) -> list[str]:
    """
    strategy:
      "random"     - uniform random sample
      "stratified" - preserve query-length and topic distribution
      "hard_cases" - queries where user clicked "not helpful" or re-queried immediately
    """
    if strategy == "random":
        return random.sample(query_logs, min(n, len(query_logs)))
    
    if strategy == "stratified":
        # Bin by length quartile
        short  = [q for q in query_logs if len(q.split()) <= 5]
        medium = [q for q in query_logs if 5 < len(q.split()) <= 15]
        long   = [q for q in query_logs if len(q.split()) > 15]
        
        n_short  = int(n * 0.3)
        n_medium = int(n * 0.5)
        n_long   = n - n_short - n_medium
        
        return (random.sample(short, min(n_short, len(short))) +
                random.sample(medium, min(n_medium, len(medium))) +
                random.sample(long, min(n_long, len(long))))
    
    if strategy == "hard_cases":
        # Queries followed by rephrasing (user dissatisfied with first answer)
        return [q for q in query_logs if was_rephrased(q)][:n]
    
    raise ValueError(f"Unknown strategy: {strategy}")
```

**Minimum viable probe set:** 50 queries. Production-grade: 200–500.

**Mandatory query types to include:**
- Simple factual (30%) — single-hop, single-passage answer
- Conceptual (30%) — requires understanding, not just matching
- Comparison/multi-aspect (20%) — requires multiple passages
- Edge cases (10%) — queries where the answer is NOT in the corpus
- Adversarial (10%) — queries that are similar to each other (probe for false retrieval)

---

## Step 2: Relevant Passage Labeling

For each query, identify which chunks in your index are relevant. Two approaches:

### Manual Annotation (Gold Standard)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Annotation:
    query: str
    relevant_chunk_ids: list[str]   # IDs of all relevant chunks
    primary_chunk_id: str           # single best chunk
    expected_answer: Optional[str]  # reference answer (optional but useful)
    notes: str = ""

# Annotation guidelines (give to annotators):
ANNOTATION_GUIDELINES = """
Mark a chunk as RELEVANT if:
  1. It contains information needed to answer the question
  2. A person would want to read it to answer the question

Mark a chunk as NOT RELEVANT if:
  1. It is topically related but doesn't help answer the question
  2. It contains the right keywords but not the right information

Mark a chunk as PRIMARY if:
  1. It is the single most useful chunk for answering the question
  2. If you could only show one chunk, this would be it

Notes:
  - A query can have 0 relevant chunks (unanswerable from corpus — mark it)
  - A query can have 1–10 relevant chunks (typical: 1–3)
  - When uncertain, err toward NOT RELEVANT (false positives hurt precision measurement)
"""
```

### LLM-Assisted Labeling (Scalable, Noisier)

```python
import anthropic
import json

client = anthropic.Anthropic()

LABELER_PROMPT = """You are labeling retrieval relevance for a RAG evaluation dataset.
Given a query and a passage, determine if the passage is relevant for answering the query.

Relevant means: a human answering the query would want to read this passage.

Output JSON: {"relevant": true/false, "confidence": "high/medium/low", "reason": "one sentence"}"""

def label_passage(query: str, passage: str) -> dict:
    resp = client.messages.create(
        model="claude-sonnet-5",     # use stronger model for labeling quality
        max_tokens=128,
        system=LABELER_PROMPT,
        messages=[{"role": "user", "content": f"Query: {query}\n\nPassage: {passage}"}],
    )
    return json.loads(resp.content[0].text)


def build_golden_labels_with_llm(
    queries: list[str],
    chunks: list[dict],
    top_k_candidates: int = 20,
    retrieval_fn=None,
) -> list[dict]:
    """
    For each query, retrieve top_k candidates, then label each with LLM.
    Only label the retrieved candidates (not all 1M+ chunks).
    """
    golden = []
    
    for query in queries:
        # Get candidate chunks (don't label everything — just plausible candidates)
        candidates = retrieval_fn(query, k=top_k_candidates)
        
        relevant_ids = []
        for chunk in candidates:
            label = label_passage(query, chunk["text"])
            if label["relevant"] and label["confidence"] in ("high", "medium"):
                relevant_ids.append(chunk["id"])
        
        golden.append({
            "query": query,
            "relevant_chunk_ids": relevant_ids,
            "n_relevant": len(relevant_ids),
            "is_unanswerable": len(relevant_ids) == 0,
        })
    
    return golden
```

---

## Step 3: Inter-Annotator Agreement

When using human annotators, measure agreement to ensure label quality.

```python
from itertools import combinations

def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Cohen's Kappa for two annotators (0 = chance, 1 = perfect agreement)."""
    n     = len(labels_a)
    agree = sum(a == b for a, b in zip(labels_a, labels_b))
    p_o   = agree / n  # observed agreement
    
    p_a = sum(labels_a) / n
    p_b = sum(labels_b) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)  # expected by chance
    
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def check_annotation_quality(annotations: list[dict]) -> dict:
    """Check agreement between annotators on a shared sample."""
    # Annotators should label the same 20% of queries independently
    shared = [a for a in annotations if a.get("double_annotated")]
    
    if not shared:
        return {"warning": "No double-annotated samples found"}
    
    kappas = []
    for ann_pair in combinations(set(a["annotator"] for a in shared), 2):
        labels_a = [1 if query_id in ann["relevant"] else 0
                    for query_id in shared
                    for ann in shared if ann["annotator"] == ann_pair[0]]
        labels_b = [1 if query_id in ann["relevant"] else 0
                    for query_id in shared
                    for ann in shared if ann["annotator"] == ann_pair[1]]
        kappas.append(cohen_kappa(labels_a, labels_b))
    
    avg_kappa = sum(kappas) / len(kappas)
    return {
        "avg_kappa": avg_kappa,
        "quality": "good" if avg_kappa > 0.7 else "acceptable" if avg_kappa > 0.5 else "poor",
        "recommendation": "" if avg_kappa > 0.7 else "Revisit annotation guidelines; resolve disagreements",
    }
```

**Kappa interpretation:** > 0.7 = good agreement; 0.5–0.7 = acceptable; < 0.5 = annotation guidelines need revision.

---

## Step 4: Dataset Format and Storage

```python
import json
from pathlib import Path

def save_golden_dataset(annotations: list[dict], path: str):
    """Save in a format compatible with RAGAS and custom eval harnesses."""
    dataset = {
        "version": "1.0",
        "created_at": "2026-07-05",
        "n_queries": len(annotations),
        "samples": [
            {
                "id":                   f"eval_{i:04d}",
                "query":               a["query"],
                "relevant_chunk_ids":  a["relevant_chunk_ids"],
                "primary_chunk_id":    a.get("primary_chunk_id"),
                "expected_answer":     a.get("expected_answer"),
                "query_type":          a.get("query_type", "unknown"),  # factual/conceptual/multi-hop
                "is_unanswerable":     a.get("is_unanswerable", False),
            }
            for i, a in enumerate(annotations)
        ],
    }
    Path(path).write_text(json.dumps(dataset, indent=2))


# Example golden dataset entry:
EXAMPLE_SAMPLE = {
    "id": "eval_0001",
    "query": "What is the difference between HNSW and IVF indexing?",
    "relevant_chunk_ids": ["chunk:vector_db:412", "chunk:vector_db:413", "chunk:benchmarks:77"],
    "primary_chunk_id": "chunk:vector_db:412",
    "expected_answer": "HNSW builds a multi-layer graph structure enabling O(log N) search; IVF pre-clusters vectors and searches only nearby clusters. HNSW is faster and more accurate for <100M vectors; IVF+PQ is better for billion-scale due to lower memory.",
    "query_type": "comparison",
    "is_unanswerable": False,
}
```

---

## Step 5: CI Integration

```python
import sys

BASELINE_METRICS_PATH = "08_evaluation/baseline_metrics.json"

def run_ci_eval(retrieval_fn, golden_dataset: list[dict], k: int = 5) -> dict:
    recalls, precisions = [], []
    
    for sample in golden_dataset:
        if sample["is_unanswerable"]:
            continue
        results    = retrieval_fn(sample["query"], k=k)
        result_ids = {r["doc_id"] for r in results}
        relevant   = set(sample["relevant_chunk_ids"])
        
        tp = len(result_ids & relevant)
        recalls.append(tp / len(relevant) if relevant else 0)
        precisions.append(tp / len(result_ids) if result_ids else 0)
    
    metrics = {
        f"recall_at_{k}":    sum(recalls) / len(recalls),
        f"precision_at_{k}": sum(precisions) / len(precisions),
    }
    return metrics


def assert_no_regression(new_metrics: dict, baseline_path: str, threshold: float = 0.02):
    baseline = json.loads(Path(baseline_path).read_text())
    failures = []
    
    for metric, baseline_val in baseline.items():
        new_val = new_metrics.get(metric, 0)
        if baseline_val - new_val > threshold:
            failures.append(f"{metric}: {baseline_val:.3f} → {new_val:.3f} (drop: {baseline_val-new_val:.3f})")
    
    if failures:
        print("REGRESSION DETECTED:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All metrics within threshold. OK to deploy.")
```

---

## Key Takeaways

1. **50 queries minimum** — fewer is statistically unreliable; 200 is production-grade.
2. **Stratify your query types** — don't over-represent easy queries; include edge cases and unanswerable queries.
3. **LLM labeling is fast but noisy** — spot-check 10% manually; re-label low-confidence samples.
4. **Measure inter-annotator agreement** — Kappa < 0.5 means your guidelines need revision, not that annotation is hard.
5. **Run eval in CI** — a 2% regression threshold blocks bad deployments without false-alarming on noise.
