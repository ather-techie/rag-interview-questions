# Evaluation Metrics: Measuring Quality in RAG Systems

> You cannot improve what you cannot measure — the complete metric reference for RAG systems.

> See [Observability & Evaluation Ops](./observability_and_evaluation_ops.md) for LLM-as-judge, online metrics, tracing, and drift alerting in production.

---

## What are Evaluation Metrics (in RAG)?

Evaluation metrics are the quantitative measures used to judge whether a RAG system's retrieval and generation are actually good — things like Precision@k and Recall@k for whether the right chunks were retrieved, NDCG and MRR for whether they were ranked well, and answer-quality metrics like RAGAS or BERTScore for the final generated response. Without them, changes to chunking, retrieval, or prompting are just guesses; metrics turn "this feels better" into something you can measure and compare.

---

## The Two Evaluation Planes

RAG systems have two quality stages. You must measure both.

```
┌──────────────────────────────────────────────────────────────┐
│ Query: "What is RAG?"                                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  RETRIEVAL PLANE: "Did we get the right context?"            │
│  ├─ Embedding ──► Query: [0.5, 0.2, 0.1, ...]              │
│  ├─ Vector DB Search ──► Top-5 chunks                       │
│  └─ Metrics: Recall@5, Precision@5, NDCG@5                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Retrieved Context:                                       │ │
│  │  - "RAG stands for Retrieval-Augmented Generation"      │ │
│  │  - "It combines retrieval with generative models"       │ │
│  │  - "Applications in QA, summarization, ..."             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  GENERATION PLANE: "Did the LLM use the context correctly?"  │
│  ├─ Prompt Engineering ──► [Context] + [Query]             │
│  ├─ LLM ──► Generated Answer                               │
│  └─ Metrics: Faithfulness, Relevance, Hallucination Rate    │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Generated Answer:                                        │ │
│  │  "RAG is a technique where a system retrieves           │ │
│  │   relevant context and uses it to generate answers."    │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Why both matter:**
- Perfect retrieval + bad generation = wrong answer
- Poor retrieval + good generation = lucky correct answer (unlikely)
- Neither is sufficient alone

---

## Retrieval Metrics (Did we get the right context?)

For each metric: definition, formula (plaintext), range, interpretation, and code.

### Precision@k

**Definition:** What fraction of the top-k results are relevant?

**Formula (plaintext):** 
```
Precision@k = (number of relevant docs in top-k) / k
```

**Range:** [0, 1], where 1 = all top-k are relevant

**Interpretation:**
- 0.8 = 80% of top-5 results are relevant (1 out of 5 is noise)
- 0.5 = 50% of top-5 results are relevant (likely too much noise)

**Code:**

```python
def precision_at_k(retrieved_docs, relevant_docs, k=5):
    """Precision@k: fraction of top-k that are relevant."""
    top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_count / k
```

---

### Recall@k

**Definition:** What fraction of all relevant docs appear in the top-k?

**Formula:** 
```
Recall@k = (relevant docs in top-k) / (total relevant docs)
```

**Range:** [0, 1], where 1 = we found all relevant docs

**Interpretation:**
- 1.0 = We found 100% of the relevant documents
- 0.6 = We found 60% of relevant documents; 40% were missed

**Code:**

```python
def recall_at_k(retrieved_docs, relevant_docs, k=5):
    """Recall@k: fraction of all relevant docs found in top-k."""
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    total_relevant = len(relevant_docs)
    return relevant_in_top_k / total_relevant if total_relevant > 0 else 0
```

---

### MRR (Mean Reciprocal Rank)

**Definition:** Rank of the first relevant result (lower is better).

**Formula:**
```
MRR = 1 / rank_of_first_relevant
  If first relevant at rank 1: MRR = 1.0
  If first relevant at rank 3: MRR = 0.333
  If no relevant found: MRR = 0
```

**Interpretation:** How soon do we find the first correct answer? (speed to first hit)

**Code:**

```python
def mrr(retrieved_docs, relevant_docs):
    """Mean reciprocal rank: 1 / rank of first relevant."""
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0  # No relevant found
```

---

### NDCG@k (Normalized Discounted Cumulative Gain)

**Definition:** Rank quality considering that relevant results higher up are better.

**Formula (plaintext):**
```
DCG@k = sum over i=1 to k of:
  relevance(i) / log2(i + 1)
  
where relevance(i) is a score (e.g., 1 if relevant, 0 if not, or 1-5 for graded relevance)

NDCG@k = DCG@k / IDCG@k
  where IDCG@k is the DCG of the perfect ranking
```

**Intuition:** Logarithmic discount means position 1 is weighted much higher than position 10.

**Interpretation:**
- 0.85 = 85% of the ideal ranking quality
- 0.60 = 60% of ideal; room for improvement

**Code:**

```python
import numpy as np

def ndcg_at_k(retrieved_docs, relevant_docs, k=5):
    """NDCG@k: normalized ranking quality."""
    # DCG calculation
    dcg = 0.0
    for rank, doc in enumerate(retrieved_docs[:k], 1):
        relevance = 1 if doc in relevant_docs else 0
        dcg += relevance / np.log2(rank + 1)
    
    # IDCG: perfect ranking (all relevant first)
    idcg = 0.0
    for rank in range(1, min(len(relevant_docs) + 1, k + 1)):
        idcg += 1.0 / np.log2(rank + 1)
    
    return dcg / idcg if idcg > 0 else 0.0
```

---

### Hit Rate

**Definition:** Did at least one relevant document appear in the top-k?

**Formula:**
```
Hit@k = 1 if any(retrieved[:k] in relevant) else 0
```

**Interpretation:** Binary: either you found a good answer or you didn't. (Less nuanced than Precision/Recall/NDCG)

**Code:**

```python
def hit_rate_at_k(retrieved_docs, relevant_docs, k=5):
    """Hit@k: did we find at least one relevant doc?"""
    return 1 if any(doc in relevant_docs for doc in retrieved_docs[:k]) else 0
```

---

## Complete Retrieval Evaluation Harness

```python
class RetrievalEvaluator:
    def __init__(self, labeled_dataset):
        """Dataset: list of (query, [relevant_doc_ids])"""
        self.queries = [q for q, _ in labeled_dataset]
        self.relevant_sets = [r for _, r in labeled_dataset]
    
    def evaluate(self, retriever, k=5):
        metrics = {
            'precision': [],
            'recall': [],
            'mrr': [],
            'ndcg': [],
            'hit_rate': []
        }
        
        for query, relevant_docs in zip(self.queries, self.relevant_sets):
            # Retrieve
            retrieved = retriever.retrieve(query, k=k)
            retrieved_ids = [doc['id'] for doc in retrieved]
            
            # Compute metrics
            metrics['precision'].append(precision_at_k(retrieved_ids, relevant_docs, k))
            metrics['recall'].append(recall_at_k(retrieved_ids, relevant_docs, k))
            metrics['mrr'].append(mrr(retrieved_ids, relevant_docs))
            metrics['ndcg'].append(ndcg_at_k(retrieved_ids, relevant_docs, k))
            metrics['hit_rate'].append(hit_rate_at_k(retrieved_ids, relevant_docs, k))
        
        # Aggregate
        results = {}
        for metric, values in metrics.items():
            results[f'{metric}@{k}'] = np.mean(values)
        
        return results

# Usage
evaluator = RetrievalEvaluator(labeled_queries)
scores = evaluator.evaluate(my_retriever)
print(scores)
# Output: {'precision@5': 0.78, 'recall@5': 0.82, 'mrr': 0.91, 'ndcg@5': 0.85, 'hit_rate': 0.92}
```

---

## Generation Metrics (Did the LLM use context correctly?)

### Faithfulness (RAGAS)

**Definition:** Does each claim in the answer appear in the retrieved context?

**How RAGAS computes it:**
1. Extract claims from the answer (via LLM)
2. For each claim, ask: "Is this claim supported by the context?"
3. Aggregate: fraction of claims that are supported

**Interpretation:**
- 0.95 = 95% of claims are supported by context; low hallucination
- 0.60 = 60% of claims are supported; high hallucination risk
- 0.40 = Only 40% supported; dangerous (misinformation)

**Gotcha:** A model that quotes the context verbatim will have high faithfulness but may not answer the question well.

```python
from ragas.metrics import faithfulness
from ragas import evaluate

# Evaluate a sample
sample = {
    'question': "What is RAG?",
    'contexts': ["RAG is retrieval-augmented generation..."],
    'answer': "RAG combines retrieval and generation."
}

faith_score = faithfulness.score(sample)
print(faith_score)  # Output: 0.95
```

---

### Answer Relevance (RAGAS)

**Definition:** Does the answer address the question?

**How RAGAS computes it:**
1. Generate multiple alternative questions from the answer
2. Check if the original question is similar to these generated questions
3. Score based on similarity

**Interpretation:**
- 0.90 = Answer directly addresses the question
- 0.60 = Answer partially addresses it
- 0.30 = Answer doesn't address the question

---

### Context Precision / Context Recall (RAGAS)

**Context Precision:**
- **Definition:** Are the retrieved chunks actually useful for generating the answer?
- **Interpretation:** Did the LLM have to wade through irrelevant context?

**Context Recall:**
- **Definition:** Does the retrieved context contain all information needed to answer the question?
- **Interpretation:** Is context missing that would lead to incomplete answers?

---

## The RAGAS Framework

RAGAS (Retrieval-Augmented Generation Assessment) is the gold standard for LLM-based evaluation.

**Key insight:** RAGAS doesn't require gold-standard answers. It evaluates by prompting the LLM itself.

**The Four RAGAS Metrics:**
1. **Faithfulness** (0–1): Claims supported by context
2. **Answer Relevance** (0–1): Answer addresses question
3. **Context Precision** (0–1): Context is relevant
4. **Context Recall** (0–1): Context is complete

**Code: Full RAGAS Evaluation**

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)

# Prepare dataset
data = {
    'question': ["What is RAG?", "How do embeddings work?", ...],
    'answer': ["RAG is ...", "Embeddings map text to ...", ...],
    'contexts': [["RAG paper...", "Retrieval methods..."], [...], ...],
}

dataset = Dataset.from_dict(data)

# Evaluate
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevance, context_precision, context_recall]
)

print(scores)
# Output:
# {
#   'faithfulness': 0.81,
#   'answer_relevance': 0.88,
#   'context_precision': 0.75,
#   'context_recall': 0.72
# }
```

---

## BERTScore

**Definition:** Semantic similarity between generated answer and reference answer (using token-level embeddings).

**How it works:**
1. Embed each token in generated and reference answers
2. For each generated token, find most similar reference token
3. Compute F1 score over these similarities

**Interpretation:**
- 0.90 = High semantic similarity to reference
- 0.65 = Moderate; answer covers main concepts but phrasing differs
- 0.40 = Low; answer is quite different

**Code:**

```python
from bert_score import score

reference = "RAG is retrieval-augmented generation"
generated = "RAG combines retrieval and generation"

P, R, F1 = score([generated], [reference], lang="en", verbose=True)
print(f"Precision: {P[0]:.3f}, Recall: {R[0]:.3f}, F1: {F1[0]:.3f}")
# Output: Precision: 0.902, Recall: 0.845, F1: 0.873
```

---

## G-Eval (Liu et al., 2023)

**Concept:** Use an LLM as a judge with explicit rubrics.

**How it works:**
1. Define evaluation rubric (e.g., "Is the answer concise? Is it accurate?")
2. Prompt LLM with rubric + answer + question
3. LLM outputs a score (1–5)

**Advantage:** More flexible than RAGAS. You can define custom criteria.

```python
def g_eval(question: str, answer: str, llm, rubric: str):
    """Use LLM to evaluate answer against a rubric."""
    prompt = f"""
    Question: {question}
    Answer: {answer}
    
    Evaluate this answer on the following rubric:
    {rubric}
    
    Provide a score from 1-5 and brief justification.
    """
    
    response = llm.generate(prompt)
    score = extract_score_from_response(response)
    return score
```

---

## Metric Correlation and Gotchas

| Metric Pair | Expected Correlation | When They Diverge | What It Signals |
|---|---|---|---|
| Faithfulness + Relevance | High (>0.7) | Low faithfulness + high relevance | LLM understands but hallucinates |
| Recall + Faithfulness | High (>0.7) | High recall + low faithfulness | Context is abundant but answer is wrong |
| Precision + NDCG | High (>0.8) | Low precision + high NDCG | Top-1 is good but rest are noise |
| Context Precision + Hit Rate | Medium (>0.5) | High hit rate + low precision | Retrieved something relevant but mostly noise |

**Key Insight:** Faithfulness + Relevance divergence signals that your retrieval is good but your prompt needs work.

---

## Evaluation Datasets and Benchmarks

| Dataset | Domain | Size | What It Tests | Citation |
|---------|--------|------|--------------|----------|
| **TriviaQA** | General trivia | 110K Q&A | Open-domain QA retrieval | Joshi et al., 2017 |
| **Natural Questions** | Google search logs | 320K Q&A | Realistic user queries | Kwiatkowski et al., 2019 |
| **HotpotQA** | Wikipedia | 113K Q&A | Multi-hop reasoning | Yang et al., 2018 |
| **BEIR** | 18 diverse domains | 1.5M docs | Retrieval benchmark | Thakur et al., 2021 |
| **QuALITY** | Long documents | 5K QA | Long-context retrieval | Pang et al., 2022 |

---

## Building a Domain-Specific Evaluation Set

```python
def build_eval_set(domain: str, num_queries: int = 100):
    """Construct a labeled evaluation set for your domain."""
    
    # Step 1: Generate questions (using LLM)
    llm_questions = llm.generate(f"""
        Generate {num_queries} realistic questions for {domain}.
        Make them specific to this domain. Output one per line.
    """)
    
    # Step 2: For each question, retrieve candidate answers
    candidates = {}
    for question in llm_questions.split('\n'):
        docs = retriever.retrieve(question, k=10)
        candidates[question] = docs
    
    # Step 3: Manual curation (have domain expert label)
    gold_labels = {}
    for question, docs in candidates.items():
        relevant_indices = human_label(question, docs)
        gold_labels[question] = relevant_indices
    
    return gold_labels

# Output: dictionary of {question: [relevant_doc_indices]}
eval_set = build_eval_set("medical", num_queries=100)
```

---

## Evaluation in CI/CD

Treat evaluation as a regression test. Block deployments if metrics fall below thresholds.

```python
def should_deploy(new_model, baseline_metrics, threshold=0.02):
    """Check if new model meets quality bar."""
    new_metrics = evaluate(new_model, labeled_dataset)
    
    for metric_name, baseline_value in baseline_metrics.items():
        new_value = new_metrics[metric_name]
        delta = baseline_value - new_value
        
        if delta > threshold:
            print(f"FAIL: {metric_name} dropped {delta:.1%}")
            return False
    
    print("PASS: All metrics within threshold")
    return True

# In CI/CD:
# if should_deploy(new_model, baseline_metrics):
#   deploy()
# else:
#   fail_pr()
```

---

## Production Instrumentation

Monitor these metrics continuously on live traffic:

```python
import logging

def log_rag_metrics(query: str, retrieved_docs: list, answer: str):
    """Log metrics to monitoring system."""
    
    # Retrieval
    num_retrieved = len(retrieved_docs)
    avg_relevance_score = np.mean([doc['score'] for doc in retrieved_docs])
    
    # Generation
    answer_tokens = len(answer.split())
    
    # User feedback (thumbs up/down)
    user_feedback = collect_user_feedback()  # 1 = good, -1 = bad
    
    metrics = {
        'retrieved_count': num_retrieved,
        'avg_relevance': avg_relevance_score,
        'answer_length': answer_tokens,
        'user_feedback': user_feedback,
    }
    
    # Send to monitoring system
    for key, value in metrics.items():
        statsd.gauge(f'rag.{key}', value)
```

---

## Key Takeaways

1. **Measure both retrieval and generation.** They're independent; both must be good.
2. **Recall@5 and Faithfulness are your primary metrics.** Track them weekly.
3. **RAGAS is gold standard** for generation evaluation (no gold labels needed).
4. **Always have a labeled probe set.** 50–100 representative queries minimum.
5. **Divergence between metrics signals problems.** High recall + low faithfulness = fix the prompt, not retrieval.
