# Cost Optimization in RAG Systems

> Reducing compute and API spend without degrading answer quality — the engineering discipline that makes RAG viable at production scale.

---

## What is Cost Optimization (in RAG)?

Cost optimization in RAG is the practice of reducing the compute and API spend accrued at each stage of a request — embedding, vector search, reranking, and generation — without degrading answer quality. It spans techniques like model tiering, prompt caching, quantization, and batching, and it matters because RAG systems make multiple paid calls per query, so per-request savings compound quickly at production traffic volumes.

---

## The RAG Cost Stack

A RAG request accrues cost at every stage. Understanding where money goes is the prerequisite to cutting it.

```
User Query
    │
    ├─ Embedding (query)        ← API call or GPU compute
    ├─ ANN Search               ← Vector DB compute / memory
    ├─ (Optional) Reranking     ← Cross-encoder inference
    ├─ LLM Generation           ← Largest cost: input tokens + output tokens
    │
    └─ Response

Typical cost breakdown (100K queries/day, production system):
  Embeddings:    ~5% of total cost
  Vector search: ~5% of total cost
  Reranking:     ~10% of total cost (if used)
  LLM input:     ~40% of total cost  ← main lever
  LLM output:    ~40% of total cost  ← main lever
```

---

## Lever 1: Prompt Caching (Input Token Reduction)

The highest-ROI optimization for most RAG systems. Cache the KV attention tensors for static parts of your prompt on the provider's infrastructure.

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_INSTRUCTIONS = "..." * 500   # static instructions (500 tokens)
REFERENCE_DOCS = "..." * 2000       # stable reference material (2000 tokens)

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": SYSTEM_INSTRUCTIONS,
            "cache_control": {"type": "ephemeral"}   # cache after first call
        },
        {
            "type": "text",
            "text": REFERENCE_DOCS,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": user_query}]  # variable — not cached
)
```

### Cache Economics

The dollar amounts change with every price update — what stays true is the *ratio*: cache reads cost a small fraction of a full-price input token, so the more of your prompt you can push before the cache breakpoint, the bigger the savings.

| Scenario | Without Caching | With Caching | Savings |
|----------|-----------------|--------------|---------|
| 2500 token static prefix, 50 token query | 2550 tokens at full input price | 50 tokens at full price + 2500 tokens at the cache-read rate (~0.1× full price on Anthropic's API) | ~94% input cost |
| 50 docs × 800 tokens each static | 40K input tokens at full price | 40K tokens at the cache-read rate + 50 uncached tokens at full price | ~98% input cost |

Check your provider's current pricing page for the exact cache-read and cache-write multipliers before modeling savings at scale — they vary by provider and can change independently of base token prices.

**Cache TTL:** Anthropic's default ephemeral cache entry lives 5 minutes, reset on each hit (`cache_control: {"type": "ephemeral"}`). Anthropic also offers an extended **1-hour TTL** (`cache_control: {"type": "ephemeral", "ttl": "1h"}`) for content that's reused less frequently but still worth caching — e.g. a reference-document prefix hit every 10–20 minutes rather than every few seconds. The 1-hour TTL costs more to *write* than the 5-minute default (roughly 2× base input price vs. ~1.25×), so it only pays off with enough reuse inside the hour to offset that premium; for bursty traffic with gaps longer than 5 minutes but shorter than an hour, it's usually a net win over repeatedly re-paying the 5-minute write cost. OpenAI's automatic prompt caching has no configurable TTL. Whichever TTL you pick, keep prompt prefixes byte-identical across requests — even whitespace differences bust the cache.

---

## Lever 2: Model Tiering (Right-Size Each Stage)

Not every RAG step needs the most capable model. Assign each stage to the cheapest model that meets its quality bar.

```
Routing/Classification  →  claude-haiku-4-5     (cheapest, fast)
Query rewriting         →  claude-haiku-4-5     (simple rewrite task)
Retrieval relevance check → claude-haiku-4-5   (binary yes/no)
Final answer generation →  claude-sonnet-5     (needs quality)
Complex reasoning       →  claude-opus-4-8     (only when needed)
```

### Example: Tiered Answer Generation

```python
def generate_answer(query: str, docs: list[str], complexity: str) -> str:
    MODEL_MAP = {
        "simple":  "claude-haiku-4-5",  # factual lookup, short answer
        "medium":  "claude-sonnet-5",   # explanation, multi-step
        "complex": "claude-opus-4-8",   # multi-document synthesis, reasoning
    }
    model = MODEL_MAP[complexity]
    
    response = client.messages.create(
        model=model,
        max_tokens=512 if complexity == "simple" else 1024,
        messages=[...]
    )
    return response.content[0].text
```

### Query Complexity Classifier

```python
CLASSIFY_PROMPT = """Classify this query as 'simple', 'medium', or 'complex'.
- simple: single fact lookup or definition
- medium: explanation, comparison, or multi-step answer
- complex: multi-document synthesis, reasoning chains, or ambiguous intent

Query: {query}
Reply with one word only."""

def classify_complexity(query: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",   # use cheap model for classification itself
        max_tokens=5,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}]
    )
    return resp.content[0].text.strip().lower()
```

---

## Lever 3: Context Window Management (Reduce Input Tokens)

The most common cost leak: passing more context than necessary to the LLM.

### Over-Retrieval Anti-Pattern

```python
# BAD: retrieve 20 chunks, pass all 20 to LLM
docs = retriever.retrieve(query, k=20)
prompt = build_prompt(query, docs)   # 20 × 800 tokens = 16,000 input tokens

# BETTER: retrieve 20, rerank, pass top-5 only
docs = retriever.retrieve(query, k=20)
reranked = reranker.rerank(query, docs)[:5]
prompt = build_prompt(query, reranked)   # 5 × 800 tokens = 4,000 input tokens
```

### Chunk-Level Compression

Use a small model to compress retrieved chunks before passing to the large model.

```python
COMPRESS_PROMPT = """Extract only the sentences from the passage that are
directly relevant to the question. Keep exact quotes. Discard the rest.

Question: {question}
Passage: {passage}"""

def compress_chunk(question: str, chunk: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",   # small model for compression
        max_tokens=200,
        messages=[{"role": "user", "content": COMPRESS_PROMPT.format(
            question=question, passage=chunk
        )}]
    )
    return resp.content[0].text
```

Typical result: 800-token chunks compressed to 100–200 relevant tokens. Net effect: 60–80% reduction in input tokens sent to the main model, at the cost of a small Haiku API call per chunk.

### Max Output Token Budget

Set `max_tokens` to the minimum needed for the task — don't leave it at 4096 for single-sentence answers.

```python
# Estimate required output length by query type
OUTPUT_BUDGET = {
    "factual":    128,
    "explain":    512,
    "summarize":  1024,
    "compare":    1024,
    "generate":   2048,
}
```

---

## Lever 4: Embedding Cost Reduction

### Use Smaller Embedding Models

| Model | Dimensions | Cost (API) | MTEB Score | Notes |
|---|---|---|---|---|
| text-embedding-3-large | 3072 | $0.13/1M tokens | 64.6 | Baseline |
| text-embedding-3-small | 1536 | $0.02/1M tokens | 62.3 | 85% cheaper, 97% quality |
| all-MiniLM-L6-v2 | 384 | $0 (self-hosted) | 56.3 | Free, ~87% quality |

For most production RAG systems, `text-embedding-3-small` or a self-hosted model is sufficient.

### Matryoshka / Dimension Truncation

Models that support Matryoshka Representation Learning (MRL) can truncate their output dimensions without retraining. Halving dimensions roughly halves storage and ANN search cost with minimal quality loss.

```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Your text here",
    dimensions=256   # truncate from 3072 to 256 — ~10x smaller
)
```

### Batch Embedding at Ingestion

Never embed one document at a time. Batch API calls to maximize throughput and minimize per-token overhead.

```python
from openai import OpenAI
import itertools

def batch_embed(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    client = OpenAI()
    all_embeddings = []
    for batch in itertools.batched(texts, batch_size):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=list(batch)
        )
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings
```

---

## Lever 5: Quantization for Self-Hosted Models

Running embedding models or small LLMs locally? Quantization reduces memory footprint and speeds up inference.

| Quantization | Memory Reduction | Quality Loss | When to Use |
|---|---|---|---|
| FP16 (half precision) | 50% vs FP32 | Negligible | Default for GPU inference |
| INT8 | 75% vs FP32 | < 1% quality | Standard production choice |
| INT4 (GPTQ / AWQ) | 87% vs FP32 | 2–5% quality | Memory-constrained deployments |
| Binary (1-bit) | 97% vs FP32 | Significant | Experimental, high-recall pre-filter only |

```python
from transformers import AutoModel
import torch

# Load embedding model in INT8 for 2× memory reduction
model = AutoModel.from_pretrained(
    "BAAI/bge-large-en-v1.5",
    load_in_8bit=True,          # requires bitsandbytes
    device_map="auto"
)
```

---

## Lever 6: Retrieval Architecture Choices

Different retrieval architectures have very different cost profiles for the same quality level.

| Architecture | Latency | Cost/Query | Quality | Notes |
|---|---|---|---|---|
| BM25 (sparse only) | 5–20ms | Near zero | Good for keyword queries | No GPU, CPU only |
| Dense (ANN, managed) | 10–50ms | $$$ (managed DB) | Good semantic | Pinecone/Weaviate cloud costs scale with vectors |
| Dense (self-hosted FAISS) | 10–50ms | $ (GPU/CPU) | Same as above | Fixed infra cost |
| Hybrid BM25 + dense | 20–80ms | $ | Best of both | RRF fusion, marginal extra cost |
| Reranking (API) | +100ms | $$ (API calls) | +5–15% NDCG | Cohere Rerank API priced per call |
| Reranking (self-hosted) | +50ms | $ (GPU) | Same | MiniLM, BGE-reranker |

**Rule of thumb:** Hybrid BM25 + dense self-hosted retrieval gives 80–90% of the quality of a fully managed stack at 20–30% of the cost.

---

## Lever 7: Asynchronous Batch Processing

For non-real-time RAG workloads (document summarization, scheduled report generation), use batch APIs instead of synchronous calls.

```python
# Anthropic Message Batches — up to 50% discount
import anthropic

client = anthropic.Anthropic()

# Submit a batch of 1000 requests
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"req_{i}",
            "params": {
                "model": "claude-sonnet-5",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": queries[i]}]
            }
        }
        for i in range(len(queries))
    ]
)
# Poll for completion; results delivered within 24 hours at ~50% cost
```

---

## Monitoring Cost per Query

Track cost at the query level to identify regressions and outliers.

```python
def track_cost(response, model: str) -> dict:
    # Per-1M-token prices below are illustrative — pull current rates from
    # your provider's pricing page (or the Models API) rather than hardcoding
    # them; they change over time and vary by intro/promotional pricing.
    PRICING = {
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},   # per 1M tokens
        "claude-sonnet-5":  {"input": 3.00, "output": 15.00},
        "claude-opus-4-8":  {"input": 5.00, "output": 25.00},
    }
    usage = response.usage
    prices = PRICING[model]
    
    input_cost  = usage.input_tokens  / 1_000_000 * prices["input"]
    output_cost = usage.output_tokens / 1_000_000 * prices["output"]
    
    # Deduct cache savings
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0)
    cache_savings = cache_read_tokens / 1_000_000 * (prices["input"] - prices["input"] * 0.1)
    
    return {
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": input_cost + output_cost - cache_savings,
        "cache_savings_usd": cache_savings,
    }
```

---

## Cost Optimization Priority Order

Apply these in order — each delivers diminishing returns once the previous is in place.

| Priority | Optimization | Typical Savings | Effort |
|----------|-------------|-----------------|--------|
| 1 | Prompt caching (static system prompt + docs) | 70–95% input cost | Low |
| 2 | Right-size output `max_tokens` | 10–40% output cost | Low |
| 3 | Model tiering (Haiku for classification/rewrite) | 30–60% total cost | Medium |
| 4 | Chunk compression before LLM | 50–80% input tokens | Medium |
| 5 | Semantic query cache (full response) | 20–60% total cost (hit rate dependent) | Medium |
| 6 | Smaller/quantized embedding models | 80–95% embedding cost | Low |
| 7 | Batch API for non-interactive workloads | 50% of remaining LLM cost | Low |
| 8 | Self-hosted retrieval (FAISS + BM25) | Eliminates managed vector DB fees | High |

---

## Interview Q&A

**Q: How would you reduce the cost of a RAG system that's spending $10K/month on LLM calls?** `[Advanced]`

Start with the highest-ROI levers: (1) enable prompt caching on the static system prompt and any fixed reference documents — this alone cuts 70–95% of input token cost on cached prefixes. (2) Set tight `max_tokens` budgets per query type. (3) Add a query classifier (Haiku, ~$0.001/call) to route simple factual queries to a cheaper model. (4) Add a semantic query cache for high-frequency repeated questions. (5) Compress retrieved chunks with a small model before passing to the large model. These five steps typically reduce cost by 60–80% with minimal quality impact.

---

**Q: What is model tiering in RAG and when is it safe to downgrade to a smaller model?** `[Intermediate]`

Model tiering assigns each pipeline stage to the cheapest model that meets the quality bar for that stage. It's safe to downgrade for: binary classification (is this relevant?), short query rewrites, simple factual questions with unambiguous answers, and extraction tasks with a fixed schema. Use larger models for: multi-document synthesis, complex reasoning chains, ambiguous queries, and generation tasks where nuance matters. Measure quality regression empirically — don't guess.

---

**Q: How does Anthropic prompt caching differ from application-level semantic caching?** `[Intermediate]`

Prompt caching (Anthropic/OpenAI) is server-side: the provider caches KV attention tensors for your static prompt prefix. You still run inference; you just pay less for the cached input tokens. Semantic caching is application-side: you skip the LLM call entirely when a semantically similar query is found in your own cache. Prompt caching helps with every call (even unique queries) as long as the prefix is shared. Semantic caching helps only for repeated queries but saves 100% of the LLM cost on a cache hit.

---

**Q: What is the risk of aggressively caching to reduce costs?** `[Intermediate]`

Stale responses: if the underlying documents change, cached answers reflect old information. For time-sensitive or rapidly-updated knowledge bases, TTL must be tuned to the document update frequency — or event-driven cache invalidation used. Additionally, semantic caching in multi-tenant systems risks cross-tenant data leakage if cache keys don't include a tenant scope.

---

## Related

- [Caching Strategies](./caching_strategies.md) — the other half of the cost story: prompt caching, semantic caching, and TTL tuning in depth
- [Fine-Tuning](./fine_tuning.md) — trading upfront training cost for lower per-request inference cost and quality gains
- [Reranking](./reranking.md) — the latency/cost tradeoffs of adding a cross-encoder reranking stage
