# 06 — Labs

> Hands-on Jupyter notebooks that turn Q&A knowledge into demonstrable, runnable skill. Each lab takes 30–60 minutes and includes a small bundled corpus.

## Contents

| Notebook | What You Build | Key Skills |
|----------|---------------|-----------|
| [01_naive_rag.ipynb](01_naive_rag.ipynb) | End-to-end naive RAG: chunk → embed → FAISS → Claude | Indexing pipeline, cosine retrieval, generation |
| [02_hybrid_rag.ipynb](02_hybrid_rag.ipynb) | BM25 + dense retrieval merged with RRF | `rank_bm25`, FAISS, RRF formula, comparison vs. each alone |
| [03_reranker_pipeline.ipynb](03_reranker_pipeline.ipynb) | Two-stage: bi-encoder → cross-encoder reranker | `CrossEncoder`, latency trade-off, rank change analysis |
| [04_ragas_evaluation.ipynb](04_ragas_evaluation.ipynb) | Full eval harness: RAGAS + custom judges + regression detection | Golden datasets, RAGAS 4 metrics, LLM-as-judge, Recall@k |
| [05_agentic_rag.ipynb](05_agentic_rag.ipynb) | ReAct & Plan-and-Execute agentic retrieval loops with tool use, stopping criteria, and prompt-injection guardrails | Provider-agnostic function-calling loops, multi-hop query decomposition, agent evaluation, injection defense |

## Prerequisites

### 1. Create a virtual environment

**Windows (PowerShell):**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name rag-labs --display-name "Python (rag-labs)"
```

Select the **Python (rag-labs)** kernel when opening a notebook (in VS Code: top-right kernel picker; in Jupyter: Kernel → Change Kernel).

### 4. Configure your LLM provider

Copy `.env.example` to `.env` and set `AI_PROVIDER`, `AI_MODEL`, and `AI_API_KEY` (Labs 02-05 read these via `ai_client.py`, which supports `claude`, `gemini`, `openai`, `azure`, and `ollama`). Lab 01 additionally needs `HF_TOKEN` (a HuggingFace Hub token), independent of `AI_PROVIDER`.

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

Then edit `.env` with your provider and key.

## Lab Progression

```
Lab 01 (Naive RAG)
  └── Learn: indexing + retrieval + generation baseline

Lab 02 (Hybrid RAG)
  └── Add: BM25 + RRF on top of Lab 01 dense index
  └── Why: exact-match queries that dense alone misses

Lab 03 (Reranker)
  └── Add: cross-encoder reranking on top of Lab 02 candidates
  └── Why: rank precision when fast retrieval makes ordering mistakes

Lab 04 (Evaluation)
  └── Measure: RAGAS + Recall@k on Labs 01–03
  └── Why: without measurement, improvement is guesswork

Lab 05 (Agentic RAG)
  └── Add: multi-step ReAct / Plan-and-Execute tool-use loops on top of Lab 01–03 retrieval
  └── Why: single-shot pipelines can't handle compound, multi-hop queries requiring reasoning + tool use
```

## Estimated Costs

| Lab | API Calls | Est. Cost |
|-----|-----------|-----------|
| 01 | ~10 Claude calls | ~$0.002 |
| 02 | ~5 LLM calls | ~$0.001 |
| 03 | ~5 LLM calls | ~$0.001 |
| 04 | ~50–80 RAGAS judge calls | ~$0.05 |
| 05 | ~25–40 LLM calls (multi-step loops + eval) | ~$0.01–0.02 |

Labs 02-05 use whichever model `AI_MODEL` is set to (see `.env`); costs above assume a small/cheap model like `gemini-2.5-flash-lite`. Lab 01 uses HuggingFace's Mistral-7B (or local Ollama), independent of `AI_PROVIDER`.
