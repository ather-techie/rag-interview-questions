# 08 — Evaluation

> Building and running the evaluation infrastructure that tells you whether your RAG system is actually good.

## Contents

| File | What It Covers |
|------|---------------|
| [01-golden-dataset-construction.md](01-golden-dataset-construction.md) | Query sampling, relevance labeling (manual + LLM-assisted), inter-annotator agreement, CI integration |
| [02-ragas-ci-harness.md](02-ragas-ci-harness.md) | RAGAS setup, custom LLM judges, full CI harness with regression detection, GitHub Actions config |

## Where to Start

1. **Understand the metrics**: [`01_concepts/evaluation_metrics.md`](../01_concepts/evaluation_metrics.md)
2. **Build your golden dataset**: [`01-golden-dataset-construction.md`](01-golden-dataset-construction.md)
3. **Wire it into CI**: [`02-ragas-ci-harness.md`](02-ragas-ci-harness.md)
4. **Production monitoring and drift**: [`01_concepts/observability_and_evaluation_ops.md`](../01_concepts/observability_and_evaluation_ops.md)

## The Eval Pyramid

```
                 Production Monitoring
                (online, continuous)
               ─────────────────────
              RAGAS / LLM-as-Judge
             (CI on every PR, ~200 samples)
            ─────────────────────────────
           Golden Dataset Recall@k
          (CI on every PR, ~100 samples)
         ──────────────────────────────────
        Unit tests: chunking, embedding dim checks
       (instant, every commit)
```

Each layer catches different failure modes. Don't skip layers — a fast unit test catches a broken embedding dimension; it won't catch a retrieval regression on a real query distribution.
