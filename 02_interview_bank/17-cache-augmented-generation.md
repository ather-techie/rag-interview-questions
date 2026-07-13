# 17 — Cache-Augmented Generation (CAG)

> Preloads the entire knowledge base into an LLM's KV cache at startup, eliminating retrieval latency entirely — a compelling alternative to RAG for bounded, stable corpora.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
SETUP / CACHE-BUILD TIME (once, or on corpus update)
─────────────────────────────────────────────────────
Entire Corpus (must fit in context window)
  → Corpus Preloader (clean, concatenate, optionally prepend ToC)
  → KV-Cache Builder (single forward pass computes key/value tensors
       for every corpus token, use_cache=True)
  → Cache Store (disk / GPU memory / provider-side prompt cache)

QUERY TIME (online, per request)
─────────────────────────────────
User Query
  → Cache Loader (restores precomputed KV cache — no recomputation)
  → Generator (LLM appends query tokens to cached prefix)
  → Answer (no retrieval step at all)
```

### Key Components

| Component | Responsibility |
|---|---|
| Corpus Preloader | Cleans, concatenates, and orders the full corpus so it fits the context window |
| KV-Cache Builder | Runs one forward pass over the corpus to precompute key/value attention tensors |
| Cache Store | Persists the KV cache (disk, GPU/CPU memory, or provider-side cache) between requests |
| Cache Loader | Restores the precomputed cache at query time, avoiding re-encoding the corpus |
| Generator | Produces the answer by attending to the cached corpus plus the new query tokens |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| KV caching (self-hosted) | HuggingFace `transformers` `past_key_values`, vLLM, SGLang |
| Provider-side caching | Anthropic prompt caching, OpenAI prompt caching |
| Long-context models | Claude 3.5 Sonnet, Gemini 1.5 Pro, GPT-4 Turbo (for corpus preload) |
| Memory management | GPU/CPU offload, INT8/INT4 KV-cache quantization for large caches |

---

## Q1. What is Cache-Augmented Generation and how does it differ from RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Cache-Augmented Generation (CAG)** preloads all relevant documents into the LLM's KV (key-value) cache before serving queries. At inference time, there is no retrieval step — the LLM generates directly from the pre-cached context.

**Standard RAG:**
```
Query → Retriever → top-k chunks → LLM → Answer
         ↑ Runtime retrieval: 100–500ms
```

**CAG:**
```
Startup: Load all documents → LLM precomputes KV cache → Save cache to disk
Query  → Load KV cache + Query → LLM → Answer
         ↑ No retrieval: < 10ms cache load
```

**Key difference:** RAG retrieves a relevant *subset* of the corpus at query time. CAG loads the *entire* corpus into context before query time and never retrieves anything.

**When CAG works:**
- The entire corpus fits in the LLM's context window (e.g., a product manual at 50K tokens fits in Claude's 200K window).
- The corpus is stable — documents don't change frequently.
- Queries span the full corpus (no single document is always relevant).

**When CAG fails:**
- Corpus is too large for the context window.
- Corpus changes frequently (cache must be rebuilt on every update).
- Cost of loading the full corpus per query is prohibitive (charged per token).

</details>

---

## Q2. How does KV cache precomputation work in CAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Modern transformer LLMs internally compute key-value pairs for every token during the attention layers. These KV pairs are normally computed at inference time for each new query. KV caching stores these precomputed values so they don't need to be recomputed on every request.

**Standard KV caching (session-level):**
- Within a single conversation, the LLM caches KV values for the previous tokens.
- The next token generation only needs to attend to these cached KV pairs.

**CAG's extended KV caching (corpus-level):**
1. **Precomputation:** Feed the entire corpus as the "prefix" to the LLM. Compute and save all KV pairs for these corpus tokens to disk.
2. **Persistence:** KV cache is a tensor (shape: `[num_layers, 2, num_heads, seq_len, head_dim]`). For a 7B model with a 100K-token corpus: approximately 8 GB in float16.
3. **At query time:** Load the precomputed KV cache + append the query tokens. Generate the answer — no recomputation of corpus KV pairs needed.

**Implementation with transformers (HuggingFace):**
```python
# Precompute KV cache for the corpus
inputs = tokenizer(full_corpus_text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    kv_cache = outputs.past_key_values  # The precomputed KV pairs

# Save to disk
torch.save(kv_cache, "corpus_kv_cache.pt")

# At query time
kv_cache = torch.load("corpus_kv_cache.pt")
query_inputs = tokenizer(query, return_tensors="pt")
with torch.no_grad():
    response = model.generate(
        **query_inputs,
        past_key_values=kv_cache,  # Inject precomputed cache
        max_new_tokens=512
    )
```

**Provider-level prompt caching (Anthropic, OpenAI):**
Claude and GPT-4 support server-side prompt caching via API. Marking a large context as cacheable achieves a similar effect — the provider's server caches the KV state for that prefix. This is the practical CAG implementation for closed-source models.

</details>

---

## Q3. What are the cost and latency trade-offs between CAG and RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Dimension | CAG | RAG |
|---|---|---|
| **Query latency** | Lowest — no retrieval step, KV cache loaded once | Medium — retrieval adds 100–500ms per query |
| **Startup latency** | High — must precompute KV cache (~seconds to minutes) | Low — retriever always ready |
| **Context window** | Hard limit — entire corpus must fit | No limit — retrieves any size corpus |
| **Per-query token cost** | High — pays for full corpus tokens every query (without caching) | Low — pays for top-k chunks only (~500–2000 tokens) |
| **Per-query token cost (with caching)** | Low — cache reads ~10% of write price | Low |
| **Freshness** | Poor — cache rebuild required on any corpus change | Good — add to index and query immediately |
| **Relevance** | N/A — LLM sees everything; must attend to relevant parts | Controlled — only relevant chunks in context |

**Cost comparison example (Claude 3.5 Sonnet, 50K-token corpus):**

Without caching:
- CAG: 50K input tokens × $3/1M = $0.15/query — extremely expensive
- RAG: 2K tokens × $3/1M = $0.006/query — 25× cheaper

With Anthropic prompt caching (90% cache read discount):
- CAG: 5K uncached + 45K cached = (5K × $3 + 45K × $0.30)/1M = $0.0285/query
- RAG: 2K × $3/1M = $0.006/query — still 5× more expensive

**When CAG wins on cost:** High-QPS systems where the KV cache stays warm, combined with a corpus small enough that full-context attention is cheap. For very small corpora (<5K tokens), CAG can be cheaper than RAG's embedding + retrieval infrastructure.

</details>

---

## Q4. When should you choose CAG over RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Decision framework:**

```
How large is your corpus?
  │
  ├─ > 50% of the LLM's context window → RAG (corpus too large for CAG)
  │
  └─ Fits in context window
       │
       How often does the corpus change?
         │
         ├─ Daily or more → RAG (frequent cache rebuilds negate CAG benefits)
         │
         └─ Weekly or less
              │
              What are your latency requirements?
                │
                ├─ Ultra-low latency required (< 100ms P95 including generation)
                │    → CAG (no retrieval step)
                │
                └─ Standard latency acceptable
                     │
                     Is relevance filtering needed?
                       │
                       ├─ Yes (only some docs are relevant per query)
                       │    → RAG (CAG makes LLM attend to everything)
                       │
                       └─ No (every query may touch any document)
                            → CAG is viable
```

**CAG sweet spots:**
- Product-specific chatbots backed by a single product manual (10K–100K tokens)
- Legal contract Q&A where the contract fits in context
- Medical device manuals, API reference documentation
- Internal policy Q&A for a small, stable policy set

**RAG sweet spots:**
- Large enterprise knowledge bases (millions of tokens)
- News/research corpora that update continuously
- Multi-tenant systems with per-tenant corpora of varying sizes
- Any corpus that won't fit in the largest available context window

</details>

---

## Q5. How does CAG handle the "lost-in-the-middle" problem? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The "lost-in-the-middle" problem — LLMs performing worse when relevant information appears in the middle of long contexts — applies equally to CAG, and may be more pronounced:

**Why CAG is more affected:**
- In RAG, you control which k chunks to include and can place the most relevant at the beginning/end.
- In CAG, the entire corpus is in context in document order. The relevant passage may be buried in the middle of 100K tokens.

**Mitigations for CAG:**

1. **Document ordering at cache build time**

Sort documents by predicted relevance to the most common query types. Place high-frequency reference documents at the beginning and end of the corpus:
```python
# Sort by query frequency (if known), else by document importance score
sorted_docs = sorted(corpus, key=lambda d: -importance_score(d))
# Interleave: most important at start and end
ordered_docs = sorted_docs[:N//2] + sorted_docs[N//2:][::-1]
```

2. **Instruction to focus**

Include an explicit instruction in the prompt to read the full context before answering:
```
"Read all provided documents carefully. The answer may be in any document.
Do not skip to the end."
```

3. **Section index in the corpus**

Prepend a short table of contents at the start of the preloaded corpus so the LLM knows what's in each document and can attend more accurately:
```
"Document 1: Product installation guide (pages 1-20)
Document 2: Troubleshooting guide (pages 21-45)
Document 3: API reference (pages 46-90)"
```

4. **Use models with strong long-context performance**

Claude 3.5 Sonnet, Gemini 1.5 Pro, and GPT-4 Turbo have been specifically evaluated for long-context needle-in-a-haystack performance. Choose a model with documented long-context accuracy.

</details>

---

## Q6. How do you handle corpus updates in a CAG system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Corpus updates are CAG's primary operational challenge. Unlike RAG (add to index = done), CAG requires rebuilding the KV cache on every meaningful update.

**Update strategies:**

**Strategy 1 — Scheduled full cache rebuild**

Rebuild the entire KV cache on a schedule (nightly, weekly) regardless of what changed:
```python
def rebuild_cache():
    corpus_text = load_all_documents()
    kv_cache = precompute_kv_cache(corpus_text)
    save_kv_cache(kv_cache, "corpus_kv_cache.pt")
    swap_live_cache(new_cache_path)  # Zero-downtime swap
```
- **When to use:** Corpus changes ≤ weekly; batch update is acceptable.
- **Downtime:** Rebuild time (seconds to minutes) before new content is available.

**Strategy 2 — Append-only corpus with partial invalidation**

Structure the corpus as a fixed base (rarely changes) + an append-only "recent updates" section:
```
KV Cache (stable base, 80% of corpus):
  Product manual v1.0, Core policies, Historical data

Live retrieval (recent updates, 20% of corpus):
  Policy updates this week, new product SKUs, recent announcements
```

Hybrid: use KV cache for the stable base + real-time RAG for the recent section. This limits cache rebuild frequency while keeping freshness for recent content.

**Strategy 3 — Provider-side caching (Anthropic, OpenAI)**

For closed-source models, prepend the corpus and use the provider's prompt caching. The provider caches the KV state server-side. Updates require sending the new corpus in the next request — the provider re-caches automatically.

**Staleness window:**

Define an SLA for how quickly updates must be reflected. CAG's staleness window = time between cache builds. If this is unacceptable, use RAG or the hybrid approach.

</details>

---

## Q7. How does CAG interact with multi-turn conversations? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Multi-turn conversations expand the context beyond the preloaded corpus, requiring careful management:

**Standard single-turn CAG:**
```
Context = [KV cache] + [single query]
```

**Multi-turn CAG:**
```
Turn 1: Context = [KV cache] + [Q1]
Turn 2: Context = [KV cache] + [Q1] + [A1] + [Q2]
Turn N: Context = [KV cache] + [Q1+A1+...+Q(N-1)+A(N-1)] + [QN]
```

**Problem:** The conversation history grows with each turn, eventually pushing total context beyond the window limit.

**Token budget:**
```
Total context window: 200K (Claude)
KV cache (corpus): 100K tokens
Reserved for conversation: 100K tokens
Avg turn (Q+A): 500 tokens → ~200 turns before overflow
```

**Mitigations:**

1. **Conversation summarization:** After every 10 turns, summarize the conversation history into a compact summary (500 tokens). Discard raw turns, keep summary.

2. **Sliding window:** Keep only the last N turns in context (e.g., last 5 turns). Sufficient for most conversational RAG use cases.

3. **Selective history:** Include only turns relevant to the current query (use a lightweight classifier to filter turns).

4. **Separate turn history from corpus cache:** Don't append conversation history to the KV-cached context; instead, pass conversation history as the "query" input (outside the cache prefix). This preserves the corpus cache across all turns.

</details>

---

## Q8. What is the GPU memory cost of CAG and how do you optimize it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

KV cache size is substantial for large corpora:

**KV cache size formula:**
```
KV cache size = 2 × num_layers × num_heads × head_dim × seq_len × bytes_per_element

For LLaMA 3 8B (float16):
  32 layers × 8 KV heads × 128 head_dim × seq_len × 2 bytes
  = 32 × 8 × 128 × seq_len × 2 = 524,288 bytes per token
  = 0.5 MB per token

For 50K-token corpus: 50,000 × 0.5 MB = 25 GB KV cache
(Does not fit on a single A100 80GB after model weights ~16 GB)
```

**Memory reduction strategies:**

1. **Quantized KV cache (INT8 or INT4)**

Quantize the KV cache values from float16 to int8 or int4:
- INT8: 50% size reduction (25 GB → 12.5 GB), minimal quality impact
- INT4: 75% size reduction (25 GB → 6.25 GB), small quality impact
- Libraries: vLLM's `--kv-cache-dtype fp8`, SGLang

2. **Multi-Query Attention (MQA) or Grouped-Query Attention (GQA)**

Models that use MQA/GQA have fewer KV heads → smaller KV cache. LLaMA 3 uses GQA (8 KV heads instead of 32) — already factored in above.

3. **Chunked prefill**

Process the corpus prefill in chunks to avoid GPU OOM during precomputation:
```python
# vLLM: process corpus in chunks
output = llm.generate(
    corpus_tokens,
    sampling_params=SamplingParams(max_tokens=0),  # Just prefill
    chunked_prefill=True
)
```

4. **Offload inactive KV cache to CPU/NVMe**

If the corpus KV cache doesn't fit on GPU VRAM, offload inactive layers to CPU memory or NVMe. Load layers on demand during generation. Trade-off: higher latency (~10–50ms per layer swap).

**Practical limits:**

For production CAG on consumer hardware (80 GB A100):
- Model weights (8B float16): ~16 GB
- Available for KV cache: ~60 GB
- Max corpus size at 0.5 MB/token: ~120K tokens
- At INT8: ~240K tokens

</details>

---

## Q9. How does CAG perform on the "needle-in-a-haystack" benchmark? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The "needle-in-a-haystack" (NIAH) benchmark tests a model's ability to find a specific fact hidden at various positions in a long context. It directly measures CAG quality.

**What NIAH tests:**
- A "needle" (specific fact, e.g., "The magic word is BANANA") is inserted at varying positions (10%, 50%, 90%) in a long document (the "haystack").
- The model is asked to recall the needle.
- Perfect recall = model can find facts at any position in its context.

**NIAH results for leading models (2024–2025):**

| Model | Context Window | NIAH Score (approx) |
|---|---|---|
| Claude 3.5 Sonnet | 200K | > 99% at ≤ 100K tokens |
| GPT-4 Turbo | 128K | > 99% at ≤ 64K tokens, degrades beyond |
| Gemini 1.5 Pro | 1M | > 99% at ≤ 500K tokens |
| LLaMA 3 70B | 128K | ~95% at 32K, degrades sharply beyond |

**Implications for CAG:**
- For corpora ≤ 100K tokens, Claude 3.5 Sonnet is a reliable CAG backend.
- Beyond 128K tokens, most models have degraded NIAH performance — retrieval errors become significant.
- The NIAH score sets a practical ceiling on CAG answer accuracy.

**NIAH as a production pre-deployment check:**

Before deploying CAG, run a NIAH evaluation on your specific corpus:
1. Insert known facts at positions 10%, 30%, 50%, 70%, 90% of the corpus.
2. Query for each fact.
3. If recall < 95% at any position, consider hybrid CAG+RAG or switch to pure RAG.

</details>

---

## Q10. Design a production CAG system for a product support chatbot backed by a 40K-token product manual. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
CORPUS: Product manual, release notes, FAQ (40K tokens total, stable)
TARGET: < 200ms P95 response latency, 99.9% uptime, weekly corpus updates

SETUP (run once or on corpus update)
─────────────────────────────────────
Manual + FAQs + Release notes
  → Clean and concatenate: 40K tokens
  → Prepend table of contents (500 tokens) for LLM navigation
  → Prepend system prompt (500 tokens)
  → Compute Anthropic prompt cache prefix (mark first 41K tokens as cacheable)
  → Warm cache: send one dummy request to populate server-side KV cache
  → Store corpus hash for staleness detection

QUERY PIPELINE
──────────────
User query (via API)
  → Safety check (PII detection, jailbreak detection)
  → Retrieve conversation history (last 5 turns, ~2,500 tokens)
  → Construct request:
       [Cached prefix: system prompt + ToC + manual] (41K tokens, cached)
       [Dynamic suffix: conversation history + current query] (~3K tokens)
  → Claude 3.5 Haiku (fast, cost-effective for support)
  → Response < 200ms P95 (cache hit eliminates prefill latency)
  → Store turn in conversation store

CORPUS UPDATE PIPELINE (weekly)
────────────────────────────────
New document version detected (content hash changed)
  → Regenerate concatenated corpus
  → Send request to Anthropic API with new corpus (triggers cache population)
  → Update stored corpus hash
  → No downtime — Anthropic caches the new prefix on next use

MONITORING
──────────
- Cache hit rate (target > 99%): If cache misses increase, check corpus hash
- P95 latency (target < 200ms): Alert if > 300ms
- Answer accuracy on 50-question golden set (weekly eval)
- User satisfaction (thumbs up/down)
- Corpus staleness: compare deployed corpus hash to latest document hash

COST ESTIMATE (10,000 queries/day)
───────────────────────────────────
Cached tokens (41K × 10K queries): 410M × $0.03/1M (cache read) = $12.30/day
Uncached dynamic tokens (3K × 10K queries): 30M × $0.80/1M = $24/day
Output tokens (500 × 10K queries): 5M × $4/1M = $20/day
Total: ~$56/day — vs. RAG at ~$20/day (2.8× more expensive but simpler)
```

</details>

---

## Q11. How do you combine CAG and RAG for a "hybrid" approach? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A hybrid CAG+RAG system uses CAG for the stable, high-importance portion of the knowledge base and RAG for the dynamic, long-tail portion:

**Architecture:**

```
Corpus split:
  CORE (40K tokens): Product manual, core policies, top-50 FAQ
    → KV-cached (CAG)
  DYNAMIC (unbounded): Daily news, support tickets, recent announcements
    → Standard vector index (RAG)

Query pipeline:
  User query
    │
    ├─ Always: include CORE KV cache in context
    │
    └─ Conditional: retrieve top-3 chunks from DYNAMIC index
       (only if query classifier predicts dynamic content needed)
    │
    ▼
  LLM receives: [core KV cache] + [0–3 dynamic chunks] + [query]
  → Answer with citations
```

**When to add dynamic RAG:**

Use a lightweight classifier trained to distinguish queries about stable content ("How do I install the product?") from queries about dynamic content ("What's the latest firmware version?"). Only pay the retrieval cost for dynamic queries.

**Benefits:**
- Core content: near-zero retrieval latency (KV cache)
- Dynamic content: always fresh (RAG)
- Context efficiency: dynamic chunks are only added when needed

**Complexity cost:**

This adds an extra system component (classifier + RAG pipeline). Only worth it if:
- A significant fraction of queries require dynamic content (> 20%)
- The dynamic corpus is too large to cache

</details>

---

## Q12. What are the security risks specific to CAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

CAG's security surface differs from RAG in important ways:

**Risk 1 — Full corpus exposure at generation time**

In RAG, only the top-k chunks (typically 2K–4K tokens) are in the LLM's context per query. In CAG, the entire corpus (40K–200K tokens) is in context. A prompt injection attack in ANY document in the corpus can trigger malicious behavior on ANY query.

- **Impact:** A single poisoned document in the corpus can affect all queries, not just queries that retrieve that document.
- **Mitigation:** Apply content validation to every document before it enters the corpus. CAG requires a higher security bar for corpus ingestion than RAG.

**Risk 2 — Data exfiltration via context repetition**

Because the full corpus is in context, prompt injection attempts to exfiltrate data via verbatim repetition ("Repeat everything before this point") have a much larger attack surface.

- **Mitigation:** Post-process model outputs to detect and block verbatim corpus repetition. Set `max_tokens` limits to prevent long verbatim outputs.

**Risk 3 — KV cache file security**

The KV cache file stored on disk contains the fully processed representation of all corpus documents. If an attacker accesses the KV cache file, they cannot directly extract the text, but could potentially probe it by running targeted queries through the model.

- **Mitigation:** Encrypt KV cache files at rest; apply the same access controls as the raw corpus.

**Risk 4 — Model version sensitivity**

A KV cache computed for model version X may not be compatible with model version X+1. If a model version update occurs and the cache is not rebuilt, queries may silently use the stale cache with a different model, producing incorrect or unpredictable results.

- **Mitigation:** Tie the KV cache version to the model checkpoint hash. Automatically invalidate cache on model version change.

</details>

---

## Real-World Applications

| Application | Domain | Why Cache-Augmented Generation Fits |
|---|---|---|
| High-traffic documentation assistant (e.g., cloud provider CLI docs) | DevTools / Cloud | Millions of queries against a stable API reference corpus — caching the KV representation eliminates repeated encoding costs at scale |
| Shared legal / compliance corpus assistant | Legal / Enterprise | Regulatory documents update quarterly; caching the stable corpus between updates slashes inference cost for thousands of daily users |
| LLM-powered customer service for a fixed product catalog | Retail / E-commerce | Product specs and FAQs are stable for weeks; KV-cached corpus means each query pays only for the dynamic user question, not the full context |
| Educational platform with fixed curriculum content | EdTech | Textbook chapters are stable for a semester; caching them lets the platform serve thousands of students at a fraction of the uncached cost |
| Enterprise "always-on" knowledge base copilot | Enterprise | Large internal wikis that change infrequently benefit from persistent KV caches — updates trigger selective cache invalidation, not full recompute |
