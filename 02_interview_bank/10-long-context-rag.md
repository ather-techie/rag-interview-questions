# 10 — Long-context RAG

> Leverages massive context windows (100K–1M tokens) to pass entire documents without chunking, trading compute for retrieval simplicity.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
        Query + minimally-chunked (or whole) documents
                            │
                            ▼
          Optional coarse filter / retrieval
          (BM25 + vector search narrows corpus
           to top-N relevant documents)
                            │
                            ▼
               Context Assembler
        (stuff full documents into prompt,
         bookend-order by relevance)
                            │
                            ▼
              Long-Context LLM Generator
         (Claude / Gemini / GPT — large window)
                            │
                    ┌───────┴───────┐
                    ▼               ▼
             Prompt Cache      Final Answer
          (reuse KV cache for
           repeated document
           context on next query)
```

### Key Components

| Component | Responsibility |
|---|---|
| Minimal Chunker (or none) | Splits only when a document exceeds the context window; otherwise passes whole documents |
| Coarse Document Selector | BM25/vector pre-filter narrows a large corpus to the top-N docs worth stuffing into context |
| Context Assembler | Orders and packs full documents into the prompt (bookend strategy to mitigate lost-in-the-middle) |
| Long-Context LLM | Generates the answer directly from the large, stuffed context |
| Prompt Cache | Caches the KV state of repeated document prefixes to cut cost/latency on follow-up queries |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Long-context models | Claude (200K–1M context), Gemini 1.5/2.x (1M+ context), GPT-4.1/4o |
| Prompt caching | Anthropic prompt caching, OpenAI prompt caching |
| Coarse retrieval | BM25 (Elasticsearch/Lucene), vector DB (Pinecone, Qdrant) |
| Compression (optional) | LLMLingua/LongLLMLingua, Selective Context, Recomp |

---

## Q1. What is Long-context RAG and how does it differ from chunk-based RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Long-context RAG** exploits the large context windows of modern LLMs (Gemini 1.5 Pro: 1M tokens; Claude 3: 200K; GPT-4o: 128K) to pass entire documents — or large portions of a corpus — directly into the prompt, eliminating the need for chunking and retrieval.

| Aspect | Chunk-based RAG | Long-context RAG |
|---|---|---|
| **Chunking** | Required | Not needed |
| **Retrieval step** | Required | Reduced or eliminated |
| **Context quality** | May miss context boundaries | Full document coherence preserved |
| **Compute cost** | Low (only top-k chunks) | High (full document per query) |
| **Latency** | Low | Higher (long prefill) |
| **Best for** | Large corpora | Small-medium corpora, complex documents |

</details>

---

## Q2. What is the "needle-in-a-haystack" problem and how does it relate to Long-context RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The **needle-in-a-haystack (NIAH)** test measures whether an LLM can find a specific fact ("needle") hidden within a large document ("haystack"). It evaluates:

- **Recall by position** — Does performance degrade when the needle is in the middle of the context?
- **Recall by depth** — Does performance degrade as the haystack grows longer?

**Relevance to Long-context RAG:**

Long-context RAG assumes the LLM can use all the context it's given — but NIAH tests reveal that most models have **degraded recall for information in the middle** of very long contexts (the "lost-in-the-middle" problem).

**Mitigation strategies:**
- Place the most critical documents at the **start or end** of the prompt.
- Use models specifically optimized for long context (Gemini 1.5, Claude 3 which use special attention mechanisms).
- Combine long-context with retrieval: do a coarse retrieval to narrow to 10–20 relevant documents, then pass all of them to the long-context LLM.

</details>

---

## Q3. When should you use Long-context RAG over traditional chunked RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Choose Long-context RAG when:**

1. **Document coherence matters** — Legal contracts, scientific papers, or financial reports where context spans the entire document.
2. **Small corpus** — Under ~50 documents; cheap to stuff everything in.
3. **Complex multi-part queries** — Questions that require synthesizing information from many sections of a document.
4. **Cross-document reasoning** — Comparing two contracts, finding contradictions across documents.
5. **Avoiding retrieval errors** — When chunking heuristics produce too many irrelevant or fragmented results.

**Stick with chunked RAG when:**
- Your corpus is millions of documents (context window can't scale to that).
- Latency and cost are critical — long-context inference is 10–100x more expensive per query.
- Most queries are narrow and answered by a small portion of your corpus.

</details>

---

## Q4. How do prompt compression techniques reduce the cost of Long-context RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Prompt compression reduces the number of tokens passed to the LLM while preserving the information needed to answer the query:

**Technique 1 — LLMLingua / LongLLMLingua:**
- Uses a small LM to score token importance for the given query.
- Drops low-importance tokens (filler words, redundant sentences).
- Achieves 3–20x compression with minimal accuracy loss.

**Technique 2 — Selective Context:**
- Computes self-information (surprisal) of each sentence.
- Removes low-surprisal (redundant) sentences.

**Technique 3 — Recomp:**
- Trains a compressor model to generate abstractive summaries of retrieved documents that are tailored to the query.

**Technique 4 — RAG-Token:**
- Rather than compressing the input, the model is trained to generate answers token-by-token where each token can attend to different retrieved passages.

**Result:** These techniques let you get the coherence benefits of long-context while reducing cost by 5–20x.

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")

compressed = compressor.compress_prompt(
    context_list,           # list of retrieved document strings
    instruction=query,      # keeps tokens relevant to this query
    target_token=1500,      # compress to ~1500 tokens (from e.g. 20K)
    rank_method="longllmlingua",
)
llm_answer = llm.invoke(compressed["compressed_prompt"])
```

</details>

---

## Q5. Design a hybrid system that combines retrieval with long-context to handle a 10,000-document legal corpus. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**The problem:** 10,000 legal documents × ~20K tokens each = 200M tokens total — far too large for any context window.

**Hybrid architecture:**

```
Stage 1 — Coarse Retrieval (fast, cheap)
  Query → BM25 + vector search → top 20 relevant documents
  [milliseconds, negligible cost]

Stage 2 — Long-context Synthesis (slow, expensive — but only for top 20)
  Top 20 docs (~400K tokens) → Claude 3 / Gemini 1.5 → Answer
  [seconds, moderate cost per query]
```

**Optimizations:**

1. **Document-level caching** — Cache the KV (key-value attention cache) of frequently accessed documents. On repeated access, skip recomputation of the document prefix.
   - Anthropic's **prompt caching** feature reduces cost by up to 90% for repeated context.
   
2. **Tiered retrieval** — Stage 1 narrows to 20 docs; Stage 2 optionally narrows to the 3 most relevant full documents for the costliest queries.

3. **Background indexing** — Run nightly BM25 + embedding updates as new documents arrive.

4. **Answer caching** — Cache answers for common query patterns (e.g., "what is the governing law clause in contract X?" is likely asked repeatedly).

**Cost estimate:** 20 docs × 20K tokens = 400K input tokens per query. At $3/1M tokens (with caching), that's ~$1.20/query — acceptable for high-value legal use cases.

</details>

---

## Q6. How does Anthropic's prompt caching reduce costs for repeated long-context queries? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Prompt caching** stores the KV (key-value) cache of a long document prefix so that repeated queries over the same document don't need to recompute it.

```
Query 1: "What is the termination clause?"
├─ Process full contract (~20K tokens) → compute KV cache
└─ Generate answer

Query 2: "What is the governing law?"
├─ Reuse cached KV for contract prefix
├─ Compute only new query tokens (~100)
└─ Generate answer (save 95% prefill compute)

Cost:
Query 1: $0.60 (full 20K tokens)
Query 2: $0.01 (only 100 new tokens + reuse cache)
Total: $0.61 (vs. $1.20 without caching)
→ 50% savings on repeated queries
```

**Implementation with Anthropic SDK:**

```python
import anthropic

client = anthropic.Anthropic()

def query_with_cache(query: str, document: str, model="claude-3-5-sonnet-20241022"):
    """Query using prompt caching for cost savings."""
    
    # First query: full cost, builds cache
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": "You are a legal expert answering questions about contracts."
            },
            {
                "type": "text",
                "text": f"Here is the contract you will analyze:\n\n{document}",
                "cache_control": {"type": "ephemeral"}  # Cache this prefix
            }
        ],
        messages=[
            {
                "role": "user",
                "content": query
            }
        ]
    )
    
    # Extract cache usage info
    usage = response.usage
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Cache creation tokens: {getattr(usage, 'cache_creation_input_tokens', 0)}")
    print(f"Cache read tokens: {getattr(usage, 'cache_read_input_tokens', 0)}")
    
    return response.content[0].text

# First call: ~20,000 tokens @ $3/1M = $0.06
answer1 = query_with_cache("What is the termination clause?", long_contract)

# Subsequent calls: ~100 new tokens + 19,900 cached @ $0.30/1M (10x cheaper) = $0.006
answer2 = query_with_cache("What is the governing law?", long_contract)

# Savings: ~80% on repeated queries
# Caches live for 5 minutes (ephemeral) or can be longer-lived with different configs
```

**Cost formula:**
```
Cost = (input_tokens × $3/1M) + (cache_creation_tokens × $3.75/1M) + (cache_read_tokens × $0.30/1M)

Typical:
- First query (cache miss): 20,000 input tokens → $0.060
- Second query (cache hit): 100 new tokens + 19,900 cached → $0.006
- Savings: 90% on repeated queries
```

</details>

---

## Q7. How does positional reordering of chunks mitigate the lost-in-the-middle problem? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Reordering chunks so most relevant ones are at the start or end (not middle) of the context improves LLM performance.

```
[Naive ordering] (by retrieval rank)

        Information density
             ↑
           0.9 │        Chunk 1 (most relevant)
               │
           0.7 │   ████ Chunk 2
               │   ████ Chunk 3
           0.5 │   ████ Chunk 4
               │   ████ Chunk 5
           0.3 │                    ← LLM attention suffers here
               │
           0.1 │   ████ Chunk 6
               └──────────────────────────────────→
                 Position in prompt

[Optimized ordering] (bookend strategy)

        Information density
             ↑
           0.9 │ ████ Chunk 1 (relevant)
               │         ...
           0.5 │    (noise/filler chunks)
               │         ...
           0.9 │ ████ Chunk 2 (relevant)
               └──────────────────────────────────→
                 Start        Middle        End
                 (good)       (bad)       (good)
```

**Implementation:**

```python
def reorder_chunks_bookend(chunks, query, num_keep_start=3, num_keep_end=3):
    """Reorder chunks: best at start/end, rest in middle."""
    
    from sentence_transformers import CrossEncoder
    
    # Score chunks using cross-encoder (more accurate than bi-encoder)
    cross_encoder = CrossEncoder('cross-encoder/mmarco-MiniLMv2-L12-H384')
    scores = cross_encoder.predict([(query, chunk) for chunk in chunks])
    
    # Rank chunks by relevance
    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    ranked_chunks = [chunks[i] for i in ranked_indices]
    
    # Bookend strategy: high-relevance at start/end, low-relevance in middle
    top_chunks = ranked_chunks[:num_keep_start]
    bottom_chunks = ranked_chunks[-num_keep_end:]
    middle_chunks = ranked_chunks[num_keep_start:-num_keep_end]
    
    # Arrange: start (best) + middle (rest) + end (best)
    reordered = top_chunks + middle_chunks + bottom_chunks
    
    return reordered

# Usage
original_chunks = retrieved_chunks  # [chunk1, chunk2, ...]
reordered = reorder_chunks_bookend(original_chunks, query, num_keep_start=2, num_keep_end=2)

# Combine with LLM
context = "\n---\n".join(reordered)
prompt = f"Answer: {query}\n\nContext:\n{context}"
answer = llm.invoke(prompt)
```

**Effect:**
- Default order: 65% accuracy
- Bookend reordering: 72% accuracy (+7pp)
- + Long-context LLM (Gemini 1.5): 78% accuracy

</details>

---

## Q8. How do you benchmark long-context RAG systems at scale? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Standard benchmarks aren't designed for long-context. Use specialized ones:

| Benchmark | Task | Corpus Size | Example |
|-----------|------|-------------|---------|
| **NIAH (Needle-in-a-haystack)** | Find a fact hidden in a long doc | 10K-100K tokens | "Find the ZIP code mentioned in paragraph 45" |
| **SCROLLS** | Multi-document QA at scale | 10K-100K tokens | Summarize main points across 20 documents |
| **LongBench** | 23 long-context tasks | 4K-6K average | Legal Q&A, financial analysis, multi-hop reasoning |
| **InfiniteBench** | Extreme length (up to 1M tokens) | 1M tokens | Find patterns in massive datasets |

```python
import numpy as np
from llmlingua import PromptCompressor

class LongContextBenchmark:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
    
    def evaluate_niah(self, num_docs=10, context_length=20000):
        """Needle-in-a-haystack: Can model find hidden fact?"""
        
        accuracies_by_position = {}
        
        for position_percentile in [0, 25, 50, 75, 100]:  # Where is the needle?
            position = int(context_length * position_percentile / 100)
            
            # Create document with hidden fact
            document = generate_haystack_document(context_length)
            document = insert_needle_at_position(document, position, 
                                               fact="The ZIP code is 12345")
            
            # Query
            query = "What is the ZIP code?"
            response = self.llm.invoke(f"{query}\n\n{document}")
            
            # Check if correct
            correct = "12345" in response
            accuracies_by_position[position_percentile] = float(correct)
        
        return accuracies_by_position  # {0: 0.95, 25: 0.85, 50: 0.45, 75: 0.85, 100: 0.95}
    
    def evaluate_multi_doc_qa(self, dataset):
        """Can model answer using information across multiple docs?"""
        
        scores = []
        
        for example in dataset:
            documents = example["documents"]  # List of docs
            query = example["query"]
            gold_answer = example["answer"]
            
            # Combine all documents
            context = "\n---\n".join(documents)
            
            response = self.llm.invoke(f"{query}\n\n{context}")
            
            # Compute similarity (BLEU, exact match, or semantic)
            similarity = compute_similarity(response, gold_answer)
            scores.append(similarity)
        
        return np.mean(scores)
    
    def evaluate_with_compression(self, documents, query, gold_answer):
        """Compare: raw long-context vs. compressed."""
        
        compressor = PromptCompressor()
        
        # Without compression
        raw_context = "\n".join(documents)
        raw_response = self.llm.invoke(f"{query}\n\n{raw_context}")
        raw_accuracy = float(compute_similarity(raw_response, gold_answer) > 0.5)
        raw_cost = estimate_cost(raw_context)
        
        # With compression
        compressed = compressor.compress_prompt(
            documents, instruction=query, target_token=1500
        )
        comp_response = self.llm.invoke(f"{query}\n\n{compressed['compressed_prompt']}")
        comp_accuracy = float(compute_similarity(comp_response, gold_answer) > 0.5)
        comp_cost = estimate_cost(compressed['compressed_prompt'])
        
        return {
            "raw_accuracy": raw_accuracy,
            "compressed_accuracy": comp_accuracy,
            "raw_cost": raw_cost,
            "compressed_cost": comp_cost,
            "compression_ratio": len(raw_context) / len(compressed['compressed_prompt'])
        }

# Typical results on LongBench
# Model                       Accuracy  Latency  Cost
# GPT-4 (4K context)          55%       1.2s     $0.30
# GPT-4o (128K)              72%       3.5s     $0.45
# Gemini 1.5 (1M)            78%       4.0s     $0.60
# + compression               76%       3.0s     $0.12  ← Recommended
```

</details>

---

## Q9. What chunking strategies serve as graceful fallbacks when a doc exceeds the context window? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

When a document exceeds the context window, fall back gracefully rather than truncating or splitting naively.

```
Strategy decision tree:

Document size < context window?
├─ YES → Use full long-context RAG (no chunking)
└─ NO → Is it mission-critical to answer from the whole doc?
    ├─ YES (legal, financial) → Chunk with semantic care
    │   ├─ [Option A] Hierarchical chunking (section → subsection)
    │   ├─ [Option B] Sliding window with overlap
    │   └─ [Option C] Smart split at paragraph/section boundaries
    └─ NO → Chunk aggressively, use traditional RAG
        ├─ [Option 1] Fixed-size chunks (512 tokens)
        └─ [Option 2] Sentence-level with deduplication
```

**Implementation:**

```python
class AdaptiveChunkingStrategy:
    def __init__(self, context_window_tokens=128000):
        self.context_window = context_window_tokens
        self.safety_margin = 10000  # Keep 10K tokens for prompt/answer
        self.available_context = context_window_tokens - self.safety_margin
    
    def choose_strategy(self, document: str, query: str) -> str:
        """Decide chunking strategy based on document size."""
        
        num_tokens = len(document.split())  # Rough estimate
        
        if num_tokens <= self.available_context:
            return "full_document"
        elif num_tokens <= self.available_context * 2:
            return "hierarchical"
        elif num_tokens <= self.available_context * 10:
            return "sliding_window"
        else:
            return "traditional_chunking"
    
    def chunk_hierarchically(self, document: str):
        """Preserve structure: section → subsection → paragraphs."""
        
        import re
        
        # Split by markdown headings
        sections = re.split(r'^(#{1,3} .+)$', document, flags=re.MULTILINE)
        
        chunks = []
        current_section = None
        
        for part in sections:
            if re.match(r'^#{1,3}', part):
                current_section = part
            else:
                # Keep section header + content together
                full_chunk = f"{current_section}\n{part}" if current_section else part
                if len(full_chunk) > 100:  # Min length
                    chunks.append(full_chunk)
        
        return chunks
    
    def chunk_with_smart_boundaries(self, document: str, target_chunk_size=2000):
        """Split at paragraph boundaries to preserve context."""
        
        paragraphs = document.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= target_chunk_size:
                current_chunk += f"\n\n{para}"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def process_document(self, document: str, query: str):
        """Route to appropriate chunking strategy."""
        
        strategy = self.choose_strategy(document, query)
        
        if strategy == "full_document":
            return [document]  # No chunking
        elif strategy == "hierarchical":
            return self.chunk_hierarchically(document)
        elif strategy == "sliding_window":
            return self.chunk_with_smart_boundaries(document, target_chunk_size=int(self.available_context * 0.8))
        else:  # traditional_chunking
            return self.chunk_with_smart_boundaries(document, target_chunk_size=1024)

# Usage
adapter = AdaptiveChunkingStrategy(context_window_tokens=128000)

strategy = adapter.choose_strategy(large_contract, query)
print(f"Using strategy: {strategy}")

if strategy == "full_document":
    context = large_contract
elif strategy == "hierarchical":
    chunks = adapter.chunk_hierarchically(large_contract)
    # Pass top-k most relevant chunks
    relevant_chunks = ranker.rank(chunks, query, k=5)
    context = "\n---\n".join(relevant_chunks)
else:
    # Fall back to traditional RAG
    chunks = adapter.process_document(large_contract, query)
    # Retrieve and answer normally
```

</details>

---

## Q10. How do you implement cost-aware routing between long-context and chunked RAG at runtime? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Route queries dynamically: long-context for small corpora / complex queries, chunked for large corpora / simple queries.

```python
from dataclasses import dataclass
from enum import Enum

class RAGMode(Enum):
    LONG_CONTEXT = "long_context"   # Full doc, no chunking
    CHUNKED = "chunked"             # Traditional RAG

@dataclass
class CostEstimate:
    mode: RAGMode
    estimated_tokens: int
    estimated_cost: float
    latency_ms: int

class CostAwareRouter:
    def __init__(self, 
                 long_context_price_per_1m=3.0,  # $ per million tokens
                 chunked_price_per_1m=0.5,       # $ per million tokens (cheaper LLM)
                 cost_budget=0.10):               # Max cost per query
        self.long_context_price = long_context_price_per_1m / 1e6
        self.chunked_price = chunked_price_per_1m / 1e6
        self.cost_budget = cost_budget
    
    def estimate_long_context_cost(self, documents: list[str], query: str) -> CostEstimate:
        """Estimate cost if we use long-context RAG."""
        
        # Full documents + query + output
        input_tokens = sum(len(doc.split()) for doc in documents) + len(query.split())
        expected_output_tokens = 200  # Average answer length
        
        total_tokens = input_tokens + expected_output_tokens
        cost = total_tokens * self.long_context_price
        latency_ms = int(total_tokens / 100)  # Rough estimate: 100 tokens/sec
        
        return CostEstimate(
            mode=RAGMode.LONG_CONTEXT,
            estimated_tokens=total_tokens,
            estimated_cost=cost,
            latency_ms=latency_ms
        )
    
    def estimate_chunked_cost(self, documents: list[str], query: str, k=5) -> CostEstimate:
        """Estimate cost if we use chunked RAG."""
        
        # Retrieve k chunks + query + output
        avg_chunk_size = (sum(len(doc.split()) for doc in documents) / len(documents)) if documents else 512
        retrieval_tokens = k * avg_chunk_size
        input_tokens = retrieval_tokens + len(query.split())
        expected_output_tokens = 200
        
        total_tokens = input_tokens + expected_output_tokens
        cost = total_tokens * self.chunked_price  # Cheaper LLM (e.g., gpt-3.5 vs gpt-4o)
        latency_ms = 500 + int(total_tokens / 50)  # Retrieval overhead ~500ms
        
        return CostEstimate(
            mode=RAGMode.CHUNKED,
            estimated_tokens=total_tokens,
            estimated_cost=cost,
            latency_ms=latency_ms
        )
    
    def decide_mode(self, documents: list[str], query: str, 
                   latency_budget_ms=None) -> RAGMode:
        """Route: long-context or chunked based on cost/latency."""
        
        long_ctx_estimate = self.estimate_long_context_cost(documents, query)
        chunked_estimate = self.estimate_chunked_cost(documents, query)
        
        # Decision logic
        if long_ctx_estimate.estimated_cost <= self.cost_budget:
            # Long-context is affordable
            if chunked_estimate.estimated_cost < long_ctx_estimate.estimated_cost * 0.2:
                # Chunked is much cheaper; use it
                return RAGMode.CHUNKED
            else:
                # Long-context not significantly more expensive; prefer quality
                return RAGMode.LONG_CONTEXT
        else:
            # Long-context exceeds budget; must use chunked
            return RAGMode.CHUNKED
    
    def query_with_adaptive_routing(self, documents: list[str], query: str) -> dict:
        """Execute query with adaptive routing."""
        
        mode = self.decide_mode(documents, query)
        
        print(f"Using mode: {mode.value}")
        
        if mode == RAGMode.LONG_CONTEXT:
            # Full context
            context = "\n---\n".join(documents)
            answer = self.llm_long_context.invoke(f"{query}\n\nContext:\n{context}")
            source = "full_document"
        else:
            # Retrieval
            from sentence_transformers import CrossEncoder
            ranker = CrossEncoder('cross-encoder/ms-marco-MiniLMv2-L12-H384')
            
            scores = ranker.predict([(query, doc) for doc in documents])
            top_indices = np.argsort(scores)[-5:][::-1]
            
            retrieved = [documents[i] for i in top_indices]
            context = "\n---\n".join(retrieved)
            answer = self.llm_chunked.invoke(f"{query}\n\nContext:\n{context}")
            source = "chunked_retrieval"
        
        return {
            "answer": answer,
            "mode": mode.value,
            "source": source
        }

# Example
router = CostAwareRouter(cost_budget=0.50)
result = router.query_with_adaptive_routing(corpus_docs, user_query)
# Output: Uses long-context if budget allows, falls back to chunked if needed
```

**Production tuning:**
- Monitor actual vs. estimated costs; adjust price models monthly.
- A/B test threshold (e.g., when does long-context become cost-prohibitive?).
- Pre-compute costs for frequent query patterns; cache decisions.
- For high-volume, low-cost workloads: default to chunked; reserve long-context for complex queries.


</details>
---

## Q11. How do you design and benchmark a compression pipeline (LLMLingua, Selective Context, Recomp) to reduce Long-context RAG token cost while maintaining answer quality? [Intermediate]

<details>
<summary>?? Show Answer</summary>

**Answer:**

**Compression pipeline stages:**

1. **Coarse filtering** - LLMLingua: identify least-relevant chunks via token importance scoring. Removes 30-50% of tokens.

2. **Fine-grained compression** - Recomp: train a compression model to summarize chunks while preserving key facts.

3. **Ranking** - Rank compressed chunks by relevance to query.

4. **Final assembly** - Pack top-k compressed chunks into final prompt (target: <8K tokens).

**Benchmark on your corpus:**

`python
def benchmark_compression(long_documents):
    for compression_ratio in [0.5, 0.6, 0.7, 0.8]:
        for method in [llmlingual, recomp, selective_context]:
            compressed = method(long_documents, ratio=compression_ratio)
            f1_score = evaluate(compressed)
            latency = measure_latency(compressed)
            cost = len(compressed) * COST_PER_TOKEN
            
            log_result(method, compression_ratio, f1_score, latency, cost)
`

**Optimal operating point:**

Typical: 60-70% compression with 5-10% F1 drop. Reduces token cost by 60-70%.

</details>

---

## Q12. How does a context window stuffing attack work against Long-context RAG, and what content-level, structural, and model-level controls prevent it? [Advanced]

<details>
<summary>?? Show Answer</summary>

**Answer:**

**Attack: Context window stuffing**

Attacker injects a large volume of irrelevant or malicious documents that "stuff" the context window, pushing aside relevant documents and forcing the model to attend to attacker-controlled content.

`
Query: "What is the revenue of Company X?"

Attacker stuffs context with 10K tokens of:
  "Company X is evil. Buy Company Y instead. [INJECTED_JAILBREAK]"

Original relevant doc (5K tokens) gets pushed out of context window.
LLM attends to injected content instead.
`

**Defences:**

1. **Content filtering** - Remove documents with low relevance scores (threshold >0.5 similarity) before feeding to LLM.

2. **Structural ordering** - Order documents by relevance, not insertion order. Place highest-confidence docs first in prompt.

3. **Compression** - Compress documents aggressively (60-80%) to fit more relevant content in fixed window.

4. **Sliding window with summary** - Keep summary of earlier content; only show detailed content for top-k docs.

5. **Adversarial input detection** - Flag documents with unusual length, duplicate content, or low-quality writing.

6. **Attention-based ranking** - Use cross-encoder or attention mechanism to re-rank documents within the context window.

7. **Model-level robustness** - Fine-tune LLM on stuffing attacks to ignore irrelevant injected content.

**Example control:**

Max 10 documents in context, each compressed to 500 tokens max. Filter by relevance >0.7. Total context: ~5K tokens.

Attacker can inject at most 10 docs � 500 tokens = 5K tokens. But these are low-relevance (0.3 score), filtered out before prompt construction.

Combining content, structural, and model-level controls prevents context stuffing.

</details>

---

## Real-World Applications

| Application | Domain | Why Long-Context RAG Fits |
|---|---|---|
| Legal contract review (e.g., Harvey AI, Ironclad AI) | Legal | Full contracts (50–200 pages) must be analyzed holistically; long-context window preserves cross-clause dependencies that chunk-based RAG fragments |
| Book or report analysis | Publishing / Research | Authors and analysts query against entire manuscripts or annual reports where section interdependencies matter |
| Large codebase Q&A (e.g., Sourcegraph Cody) | DevTools | Repository-level questions ("where does authentication flow through?") require reasoning across many files simultaneously |
| Clinical trial protocol analysis | Pharma / Regulatory | Trial protocols are long, structured documents; retrieving complete sections preserves eligibility criteria and procedure details intact |
| M&A document room assistant | Investment Banking | Due diligence corpora span thousands of pages; long-context windows allow the LLM to reason across related documents without losing thread |
