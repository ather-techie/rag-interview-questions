# WebGPT / Tool-Augmented LM

> An LLM trained — not just prompted — to issue retrieval actions as part of its generation process, treating web search or tool calls as first-class operations learned from human demonstrations and feedback.

---

## Definition

**WebGPT** (Nakano et al., OpenAI 2021) fine-tunes a language model to browse the web by learning an explicit action space (search, click, scroll, quote, done) from human demonstrations, then refines the policy with reinforcement learning from human feedback (RLHF). It is the formal precursor to agentic web RAG (#31) and modern tool-use patterns — but differs in a key dimension: **the retrieval policy is learned, not prompted**.

**Tool-Augmented LMs** (TALM, Parisi et al. 2022; Toolformer, Schick et al. 2023) generalize this: models learn when and how to call arbitrary APIs mid-generation via self-supervised fine-tuning, without explicit human labeling of every call.

---

## How It Works

### WebGPT Action Space

At each generation step, the model emits either a **text token** (regular generation) or one of five **browser actions**:

| Action | Signature | Purpose |
|---|---|---|
| `search(query)` | `<search>query text</search>` | Fetch top-10 web results |
| `click(n)` | `<click>3</click>` | Open result at index n |
| `scroll(dir)` | `<scroll>down</scroll>` | Page through a document |
| `quote(text)` | `<quote>exact text</quote>` | Save evidence for the answer |
| `done` | `<done/>` | Finalize answer with citations |

```
Input: "What is the population of Tokyo as of 2023?"

Model turn 1: <search>Tokyo population 2023</search>
→ Results returned: [result_0: "Tokyo - Wikipedia", result_1: "World Atlas: Tokyo", ...]

Model turn 2: <click>0</click>
→ Page content returned

Model turn 3: <quote>As of 2023, the Greater Tokyo Area has approximately 37.4 million people</quote>

Model turn 4: <done/>
→ Answer generated with quoted evidence
```

### Training Pipeline

```
Step 1 — Behavior Cloning (BC)
  Human demonstrators browse and answer questions.
  Fine-tune GPT-3 on (question, action_sequence, answer) triples.

Step 2 — RLHF
  For two WebGPT answers to the same question, humans choose which is better.
  Train a reward model on (answer_A, answer_B, preference) pairs.
  Fine-tune with PPO using the reward model signal.
```

### Toolformer: Self-Supervised Tool Use

Toolformer (Schick et al., 2023) removes the need for human demonstration data. The model learns to insert API calls by self-supervision:

```python
# Step 1: Sample candidate positions for API calls
text = "The Eiffel Tower is [POSITION] meters tall."
candidate_apis = ["calculator(height_lookup('Eiffel Tower'))"]

# Step 2: Check if the API call reduces perplexity on the continuation
perplexity_without = lm_perplexity("...is meters tall. It was built in 1889.")
perplexity_with    = lm_perplexity("...is 330 meters tall. It was built in 1889.")
# If perplexity_with << perplexity_without → keep this API call in training data

# Step 3: Fine-tune on filtered (text, API_calls) pairs
# Model learns: when an API call helps, emit it mid-generation
```

Toolformer APIs supported: Wikipedia search, calculator, calendar, QA model, machine translation.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
  │
  ▼
RLHF-trained Policy Model
  (decides next browser action: search / click / scroll / quote / done)
  │
  ▼
┌─────────────────────────────────────────┐
│         Browser Action Loop              │
│  ┌───────────────────────────────────┐  │
│  │ Action Executor                    │  │
│  │  search(query) → result list       │  │
│  │  click(n)      → page content      │  │
│  │  scroll(dir)   → more of page      │  │
│  │  quote(text)   → save as evidence  │  │
│  └──────────────┬──────────────────────┘  │
│                 │                          │
│                 ▼                          │
│         Citation Collector                 │
│  (tracks quoted passages + source doc)     │
│                 │                          │
│      loop back to Policy Model             │
│      until action == done                  │
└─────────────────────────────────────────┘
  │
  ▼
Answer Synthesizer
  (produces final answer with inline citations
   from the Citation Collector's evidence set)
```

### Key Components

| Component | Responsibility |
|---|---|
| **RLHF-trained Policy Model** | Decides, at each step, whether to emit text or issue a browser action (search/click/scroll/quote/done) |
| **Browsing Environment/Simulator** | Sandboxed, text-based web environment that executes actions and returns results/page content |
| **Action Executor** | Dispatches `search`, `click`, `scroll`, `quote` calls against the browsing environment |
| **Citation Collector** | Tracks each quoted passage alongside its source document for later attribution |
| **Answer Synthesizer** | Composes the final answer, citing the collected evidence once the model emits `done` |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| **RLHF training** | Reward model + PPO training stack |
| **Environment** | Sandboxed browser/search environment (text-based) |
| **Data collection** | Human preference/demonstration data collection pipeline |

---

## Architecture Comparison

| Dimension | WebGPT (#39) | Agentic Web RAG (#31) | Standard RAG |
|---|---|---|---|
| **Retrieval policy** | Learned via RLHF | Prompted via system prompt | Fixed, pre-defined |
| **Fine-tuning required** | Yes — full RLHF pipeline | No (prompting only) | No |
| **Action space** | Structured (search/click/quote) | Unstructured (any tool call) | Single retrieve-then-generate |
| **Citation handling** | Built-in (quote action) | Add-on (post-hoc) | Manual |
| **Generalizes to new tools** | No (must retrain for new actions) | Yes (add tool to prompt) | N/A |
| **Production cost** | High (fine-tuned model per domain) | Low (reuse base model) | Lowest |
| **Human data needed** | Yes (demonstrations + comparisons) | No | No |

---

## Retrieval Strategy: Learned vs. Prompted

The central interview distinction:

```
Prompted (Agentic Web RAG):
  System prompt: "You have access to a search tool. Use it when you need current information."
  Model: GPT/Claude 3.5 with tool_use API
  → Works immediately with any capable model; policy driven by in-context instructions

Learned (WebGPT / Toolformer):
  Fine-tuned model: GPT-3-WebGPT
  → Model has internalized when/how to search; doesn't rely on instruction-following quality
  → Advantage: more reliable on novel domains where prompting is ambiguous
  → Disadvantage: expensive to retrain; brittle to new tools
```

In 2024+, prompted tool use (agentic web RAG) dominates in production because:
1. Models improved dramatically at instruction following — the gap closed
2. Prompted systems are easier to update (change the prompt, not the model)
3. Fine-tuning cost is high; prompted system cost is low

WebGPT is primarily relevant today as the *conceptual precursor* and as evidence that retrieval policy can be learned.

---

## Modern Descendant: Gorilla

**Gorilla** (Patil et al., Berkeley 2023) applies the WebGPT idea specifically to API calling for ML model invocations:

```python
# Standard model: hallucinate API parameters
query = "Generate an image of a cat using Stable Diffusion"
# LLM output: stabilityai.generate(prompt="cat", version="wrong_version")  ← hallucinated

# Gorilla: retrieval-augmented fine-tuning for APIs
# Training: (query, retrieved_API_doc, correct_API_call) triples
# At inference: retrieve relevant API docs → generate correct call
query = "Generate an image of a cat using Stable Diffusion"
# Gorilla: from diffusers import StableDiffusionPipeline; pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")  ← correct
```

Gorilla is WebGPT's principle applied to a narrower, more tractable domain: API calls are verifiable (run the code; check if it works), making RL signal cheap to compute.

---

## Production Considerations

| Concern | WebGPT Pattern | Modern Alternative |
|---|---|---|
| **Latency** | Multi-step browsing = 3–10s | Parallel tool calls (agentic RAG) + caching |
| **Cost** | Fine-tuned model per domain | Prompt engineering on base model |
| **Citation quality** | Excellent (quote action is explicit) | Variable (post-hoc attribution) |
| **Hallucination on facts** | Low — model trained to quote | Depends on retrieval quality |
| **Maintenance** | High — retrain for new tools/domains | Low — update system prompt |

---

## Key Takeaways

1. **WebGPT established learned retrieval** — treating search as a structured action trained via RLHF rather than a hardcoded API call.
2. **Toolformer scaled this without human data** — self-supervised API call injection via perplexity reduction.
3. **Modern agentic web RAG (#31) superseded both** — instruction-tuned LLMs are now reliable enough that prompting works; fine-tuning is reserved for narrow API-calling tasks (Gorilla).
4. **The core insight persists**: treating retrieval as a *learned first-class operation*, not a bolted-on post-processing step, produces models that know *when not to search* (vs. prompting which can over- or under-trigger tool calls).

---

## Interview Q&A

**Q: What is the key difference between WebGPT and agentic web RAG (#31)?**

WebGPT *learns* its retrieval policy through RLHF — the model is fine-tuned on human-labeled browsing demonstrations and preference comparisons, so the decision of when and how to search is internalized as model weights. Agentic web RAG *prompts* an existing model with a tool definition and relies on the model's instruction-following ability to trigger the right tool calls. The advantage of the learned approach: consistent behavior on ambiguous queries where instruction-following is uncertain. The advantage of the prompted approach: no fine-tuning required, generalizes to new tools by updating the prompt. In 2024+, the improvement in instruction-following quality of models like GPT-4 and Claude largely closed the gap, making the prompted approach the practical default.

---

**Q: What is Toolformer and how does it differ from WebGPT?**

Toolformer (Schick et al., Meta 2023) teaches models to call APIs mid-generation without human demonstration data. It does this by: (1) sampling candidate positions in text where an API call *might* help, (2) checking whether including the API response reduces the model's perplexity on the subsequent tokens (if the calendar says "today is Monday", the word "Monday" becomes more predictable → keep the call; if the response doesn't help, discard it), and (3) fine-tuning on the filtered dataset of (text + helpful API calls). WebGPT requires human demonstrators and preference labelers; Toolformer is entirely self-supervised. The trade-off: WebGPT's human signal produces more reliable citation behavior; Toolformer is cheaper to train but the API call accuracy is lower.

---

**Q: When would you fine-tune a model for tool use (WebGPT-style) instead of using prompting (agentic RAG-style)?**

Fine-tune for tool use when: (1) **Reliability matters more than flexibility** — a fine-tuned model reliably calls the right API with correct parameters; prompting can fail on edge cases. (2) **The tool space is narrow and stable** — if you have exactly two tools (search, calculator) that won't change, fine-tuning amortizes. (3) **Verification signal is cheap** — Gorilla fine-tunes for API calls because correctness is verifiable (run the code). (4) **Latency budget is tight** — a fine-tuned smaller model (7B) can be faster than prompting a larger model. Use prompting when the tool space is evolving (new tools added monthly), latency is tolerable, or when you're prototyping before committing to fine-tuning cost.
