# Eval & Observability Tool Comparison

> Same three questions — "did we retrieve the right context?", "is the answer grounded?", "does it answer the question?" — five different toolkits. Picking one is mostly about how you want to run evaluation (CI pipeline vs. pytest vs. notebook) and what you're already standardized on.

> See [Evaluation Metrics](../01_concepts/evaluation_metrics.md) for the underlying metric definitions (Precision@k, Recall@k, Faithfulness, Context Precision/Recall) and the [Groundedness & Context Relevancy naming note](../01_concepts/evaluation_metrics.md#vendor-naming-groundedness--context-relevancy). See [Observability & Evaluation Ops](../01_concepts/observability_and_evaluation_ops.md) for LLM-as-judge methodology, judge calibration, and production tracing (LangSmith, Arize Phoenix, Langfuse) — not repeated here.

---

## Feature Matrix

| Tool | Primary use case | Requires gold labels? | Integration style | Current status |
|------|------------------|------------------------|--------------------|-----------------|
| **Ragas** | RAG-specific metrics (faithfulness, context precision/recall), CI regression gate | No (LLM-judged) for Faithfulness/Answer Relevance; yes for Context Precision/Recall | Standalone `evaluate()` over a HuggingFace `Dataset` | Actively maintained; core RAG eval standard |
| **TruLens** | Opinionated "RAG Triad" feedback functions + tracing | No (LLM/embedding-judged) | Wraps your app (chain/agent) and records feedback per call | Actively maintained |
| **DeepEval** | Unit-test-style LLM eval | No (LLM-judged) | Pytest-native (`assert_test`, `deepeval test run`) or standalone `evaluate()` | Actively maintained |
| **LlamaIndex eval** | Built-in evaluators for LlamaIndex-built RAG pipelines | No for Faithfulness/Relevancy; yes for Correctness/Retriever hit_rate & MRR | Python classes (`FaithfulnessEvaluator`, `RetrieverEvaluator`, ...) called directly on responses | Actively maintained, tied to LlamaIndex |
| **LangChain eval** | Criteria-based string evaluators | Depends on criteria (QA criteria needs a reference) | `load_evaluator(...)` from a now-legacy module | **Legacy/frozen** — see below |

---

## Ragas

Already covered in depth elsewhere in this repo — see [08_evaluation/02-ragas-ci-harness.md](../08_evaluation/02-ragas-ci-harness.md) for full setup code, the four core metrics, custom judge metrics, and a GitHub Actions CI harness, and [06_labs_py/04_ragas_evaluation.ipynb](../06_labs_py/04_ragas_evaluation.ipynb) for a runnable notebook version. Reach for Ragas when you want an LLM-judged, no-gold-label baseline wired into CI as a regression gate.

---

## TruLens

Already has a runnable example in this repo — see [03_failure_modes/01-hallucination_despite_context.md](../03_failure_modes/01-hallucination_despite_context.md) for `TruChain`/`Feedback`-based hallucination detection, and the [RAG Triad naming note](../01_concepts/evaluation_metrics.md#vendor-naming-groundedness--context-relevancy) for how its three feedback functions (Context Relevance, Groundedness, Answer Relevance) map onto Ragas's terminology. Reach for TruLens when you want opinionated, pre-built feedback functions plus tracing baked into the same tool, or when "RAG Triad" is the vocabulary your team/interviewer already uses.

---

## DeepEval

DeepEval (Confident AI) is a pytest-native LLM evaluation framework — its distinguishing feature is that evaluations run as ordinary test functions rather than a separate notebook/CLI step.

**Core RAG metrics** (`deepeval.metrics`):
- `FaithfulnessMetric`, `AnswerRelevancyMetric` — generation-side, mirror Ragas's Faithfulness/Answer Relevance
- `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric` — retrieval-side, mirror Ragas's Context Precision/Recall plus a rank-independent relevancy score (see the [Context Relevancy naming note](../01_concepts/evaluation_metrics.md#vendor-naming-groundedness--context-relevancy))

Test cases are built with `LLMTestCase` and run via either `assert_test` inside a pytest function, or the standalone `evaluate()` function for notebook/script use:

```python
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_rag_response():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="You have 30 days to get a full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."],
    )
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7), AnswerRelevancyMetric(threshold=0.7)])

# run with: deepeval test run test_rag_response.py
```

DeepEval also ships a `GEval` custom-metric class for rubric-based scoring (similar in spirit to the [G-Eval section](../01_concepts/evaluation_metrics.md) in the metrics reference) — its parameter-enum name has changed across DeepEval versions, so check your installed version's docs before hardcoding a specific enum in real code.

Reach for DeepEval when your team wants RAG eval to live in the same pytest suite as the rest of your test suite, with pass/fail assertions and thresholds rather than a separate reporting step.

---

## LlamaIndex Eval

If you're already building your RAG pipeline with LlamaIndex, its `llama_index.core.evaluation` module ships evaluators that plug directly into LlamaIndex response/retriever objects:

- `FaithfulnessEvaluator` — checks whether a response is grounded in its source nodes
- `RetrieverEvaluator` — scores a retriever directly against a labeled query set, using named metrics: `RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], retriever=...)`
- `RelevancyEvaluator`, `CorrectnessEvaluator` — also available (relevancy of response to query, and correctness against a reference answer respectively); consult the current LlamaIndex docs for exact method signatures before using them, since they weren't independently verified against a usage example here

```python
from llama_index.core.evaluation import FaithfulnessEvaluator, RetrieverEvaluator
from llama_index.llms.openai import OpenAI

evaluator = FaithfulnessEvaluator(llm=OpenAI(model="gpt-4", temperature=0.0))
eval_result = evaluator.evaluate_response(response=response)
print(eval_result.passing)

retriever_evaluator = RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], retriever=retriever)
retriever_evaluator.evaluate(query="query", expected_ids=["node_id1", "node_id2"])
```

Reach for LlamaIndex's evaluators when your retrieval and query pipeline are already built on LlamaIndex primitives — you avoid re-wiring your response/node objects into a separate framework's data format.

---

## LangChain Eval (legacy)

LangChain used to ship its own evaluation module (`langchain.evaluation`: `load_evaluator`, `EvaluatorType`, criteria-based string evaluators). As of LangChain v1, that module has been moved into the frozen legacy package `langchain_classic`, and LangChain's own documentation now points evaluation questions to **LangSmith** instead (already covered in [Observability & Evaluation Ops](../01_concepts/observability_and_evaluation_ops.md)).

```python
# Legacy path — still importable, no longer where LangChain's active development is:
from langchain_classic.evaluation import load_evaluator, EvaluatorType
evaluator = load_evaluator(EvaluatorType.QA)
result = evaluator.evaluate_strings(prediction=answer, input=question, reference=ground_truth)
```

Don't present "LangChain → evaluation chains" as an actively-recommended current pattern in an interview answer — the accurate framing is that LangChain has deprioritized its built-in eval module in favor of LangSmith. If you're already on LangChain/LangGraph, LangSmith is the tool to reach for, not `langchain.evaluation`/`langchain_classic`.

---

## Decision Criteria

| If... | Reach for |
|-------|-----------|
| You want a no-gold-label baseline wired into CI as a regression gate | **Ragas** |
| You want RAG eval to run as ordinary pytest tests with pass/fail assertions | **DeepEval** |
| You want pre-built, opinionated feedback functions plus tracing in one tool ("RAG Triad" vocabulary) | **TruLens** |
| Your pipeline is already built on LlamaIndex primitives | **LlamaIndex eval** (`FaithfulnessEvaluator`, `RetrieverEvaluator`, ...) |
| You're already on LangChain/LangGraph and want the least setup for tracing + eval | **LangSmith** (not the legacy `langchain.evaluation` module) |

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md). Vector database and orchestration-framework comparisons for this section are still planned — see [09_tools/README.md](./README.md).
