# 29 — Fusion-in-Decoder (FiD)

> The canonical retrieval *reader* architecture: encode each retrieved passage **independently** with the query (parallel, linear cost), then let the **decoder cross-attend over all encoded passages at once** to fuse evidence — sidestepping the quadratic blow-up of concatenating passages and scaling cleanly to 100+ passages.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
QUERY
  │
  ▼
┌───────────────────────────┐
│  Upstream Retriever          │  e.g., BM25 / DPR — trained and
│  (retriever-agnostic)        │  operated separately from FiD
└────────────┬──────────────── ┘
             │ top-k passages P1...Pk
             ▼
┌───────────────────────────────────────────┐
│  Per-Passage Encoder (parallel)               │
│   enc(q + P1) → H1                             │
│   enc(q + P2) → H2       (independent,          │
│   ...                       linear cost in k)    │
│   enc(q + Pk) → Hk                             │
└────────────┬──────────────────────────────────── ┘
             │ concatenate [H1;H2;...;Hk]
             ▼
┌───────────────────────────────────────┐
│  Decoder (joint cross-attention)         │
│   cross-attends over ALL encoded          │
│   passages at once → fuses evidence        │
└────────────┬────────────────────────────── ┘
             │
             ▼
     Generator Output (answer)
```

### Key Components

| Component | Responsibility |
|---|---|
| Upstream Retriever | Supplies top-k passages; independent of FiD and swappable (BM25, DPR, Contriever, ...) |
| Per-Passage Encoder | Encodes each (query+passage) pair independently and in parallel — linear cost in k |
| Concatenation step | Joins the encoded passage representations before decoding |
| Decoder (joint cross-attention) | Attends jointly over all passage encodings to fuse evidence into one generated answer |
| Generator output | Produces the final generative, synthesized answer |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Encoder-decoder backbone | T5, BART |
| Upstream retriever | DPR, BM25 (Pyserini / Elasticsearch), Contriever |
| ML framework | HuggingFace `transformers` |
| Efficiency variants | FiD-light, FiDO (FiD-Optimized) implementations |
| Reranking (pre-FiD pruning) | Cross-encoder rerankers to shrink k before passages reach FiD |

---

## Q1. What is Fusion-in-Decoder and what problem does it solve? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Fusion-in-Decoder (FiD)** (Izacard & Grave, 2020) is a sequence-to-sequence **reader** architecture for retrieval-augmented QA. It **encodes each retrieved passage separately** and then **fuses them in the decoder**, which attends over all encoded passages jointly to generate the answer.

**The problem it solves — how to feed *many* passages to a reader:**

Naively, you'd **concatenate** all retrieved passages into one long input and feed it to a standard seq2seq model. But:
- Transformer **self-attention is quadratic** in input length → concatenating k passages of length L costs O((kL)²). With many passages this explodes.
- Long concatenated inputs hit context limits and "lost-in-the-middle" degradation.

**FiD's fix:**
```
Concatenation (naive):    encode([P1 ; P2 ; ... ; Pk])  → O((kL)²)  ❌ blows up
Fusion-in-Decoder (FiD):  encode(P1), encode(P2), ... separately → O(k·L²)  ✅ linear in k
                          decoder cross-attends over ALL encoded passages → fuse
```

By encoding passages **independently**, the expensive self-attention is **per-passage** (O(L²) each, k of them = O(kL²) — linear in k, not quadratic). The **fusion** — combining evidence across passages — happens cheaply in the **decoder's cross-attention**.

**Result:** FiD scales to **many passages** (the original used up to ~100), and more passages → better answers, because the decoder can synthesize evidence spread across them.

</details>

---

## Q2. How exactly does FiD encode and fuse passages? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
INPUT: question q + k retrieved passages P1...Pk

ENCODER (independent, parallel):
  For each passage Pi:
    input_i = "question: {q}  title: {title_i}  context: {Pi}"
    H_i = Encoder(input_i)      # contextualized token representations
  → k separate encoded sequences H_1 ... H_k (computed in parallel)

FUSION (in the decoder):
  Concatenate the encoded representations: H = [H_1 ; H_2 ; ... ; H_k]
  Decoder generates the answer autoregressively, with CROSS-ATTENTION
  over the FULL concatenated H (all passages at once).
  → "fusion in the decoder": evidence from all passages is combined
     only at the cross-attention step, not in the encoder.
```

**The key architectural insight — *where* fusion happens:**
- **Encoder:** each passage is processed **in isolation** (no passage sees another) → cheap, parallel, linear in k.
- **Decoder:** sees **all passages jointly** via cross-attention → this is where evidence is *combined* to produce the answer.

**Why split it this way:**
- The **expensive** operation (self-attention) is confined to single passages (short).
- The **fusion** operation (cross-attention from the decoder) operates over the concatenated *encoded* representations — the decoder's queries are few (answer tokens), so attending over all passage tokens is affordable.

So FiD gets **the benefit of seeing all passages together** (joint reasoning over evidence) **without the cost of jointly self-attending over their concatenation**.

</details>

---

## Q3. Why does FiD scale better than concatenating passages into one input? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

It's a **computational-complexity** argument about where the quadratic cost lands.

**Concatenation approach:**
```
Input length = k·L  (k passages, each length L)
Encoder self-attention cost = O((k·L)²) = O(k²·L²)   ← quadratic in k
```
Doubling the number of passages **quadruples** encoder cost. This caps how many passages you can use.

**FiD approach:**
```
Encode each passage separately:  k × O(L²) = O(k·L²)   ← LINEAR in k
Decoder cross-attention over concatenated encodings:
  decoder has m answer tokens; attends over k·L encoded tokens
  → O(m · k·L), and m (answer length) is small → cheap
```
Doubling passages only **doubles** encoder cost (linear).

**The decisive difference:**
- Concatenation forces **every passage token to self-attend to every other passage token** — mostly wasted, since passages are independent evidence.
- FiD recognizes passages are **independent at encoding time**, so it skips cross-passage self-attention and only fuses where it matters: the **decoder's cross-attention**, which is cheap because answer tokens are few.

**Practical consequence:** FiD can use **10×–100× more passages** than concatenation at feasible cost — and since retrieval recall improves with more passages, this **directly improves answer quality**. The original FiD showed monotonic gains as passage count rises (e.g., up to 100 passages).

**Caveat (revisited in Q7/Q11):** the **decoder cross-attention** does grow with k·L, and at very large k it becomes the bottleneck — later work (FiD-light, FiDO) optimizes exactly this.

</details>

---

## Q4. How does FiD differ from REALM, RETRO, and Atlas? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

FiD is fundamentally a **reader architecture**, while the others are full retrieval-augmented *systems* (with their own retriever stories):

| | REALM (26) | RETRO (27) | Atlas (28) | Fusion-in-Decoder (29) |
|---|---|---|---|---|
| **What it primarily is** | End-to-end pretraining w/ learned retriever | LM w/ frozen retriever + CCA | Few-shot system (learned retriever + FiD) | **A reader architecture** |
| **Retriever** | Learned (latent var) | Frozen BERT | Learned Contriever (joint) | **Separate** (e.g., DPR), not part of FiD itself |
| **Key mechanism** | Marginalize over docs | Chunked cross-attention | Joint training (attn distillation) | **Encode separately, fuse in decoder** |
| **Where retrieval integrates** | Pretraining | During generation (per chunk) | Training | **At read time** (post-retrieval) |
| **Scope** | Full system | Full system | Full system | **Reader component** (retriever-agnostic) |

**FiD's distinct identity:**
1. **It's a *reader*, not a retriever or a training scheme.** FiD says nothing about *how* you retrieve — pair it with BM25, DPR, Contriever, anything. It defines **how the reader consumes many retrieved passages efficiently**.
2. **It's a *building block* the others use.** **Atlas's reader *is* FiD.** FiD is the component, Atlas is the system. RETRO uses a different integration (CCA during generation, per chunk); FiD fuses retrieved passages **once, at read time, in the decoder**.
3. **Encode-separately-fuse-in-decoder** is its signature trick — distinct from RETRO's chunked cross-attention (which retrieves *during* autoregressive generation) and from concatenation.

**One-liner:** FiD is the **efficient reader** — "encode each passage alone, fuse them in the decoder" — and it's the reader of choice that systems like Atlas plug in behind their retriever.

</details>

---

## Q5. Walk through FiD answering an open-domain question. `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
QUESTION: "What year was the Eiffel Tower completed?"

1. RETRIEVAL (external to FiD — e.g., DPR):
   Retrieve top-k passages (say k=50) from Wikipedia:
     P1: "The Eiffel Tower was completed in 1889 for the World's Fair..."
     P2: "Gustave Eiffel's company built the tower between 1887 and 1889..."
     ... (48 more)

2. FiD ENCODER (each passage independently, in parallel):
     enc("question: What year... title: Eiffel Tower context: P1") → H1
     enc("question: What year... title: Gustave Eiffel context: P2") → H2
     ... → H1..H50   (50 separate encodings, linear cost)

3. FiD DECODER (fusion):
     Concatenate [H1; H2; ...; H50]
     Decoder cross-attends over ALL of them while generating:
       → reads "1889" supported across P1 and P2
     Generates: "1889"

4. ANSWER: "1889"
```

**What the fusion buys here:**
- The answer "1889" is **corroborated across multiple passages** (P1, P2). The decoder, attending over all encodings at once, can **aggregate evidence** and weight agreement — more robust than reading a single passage.
- Had the answer required combining facts from **different** passages (e.g., "who built it and when"), FiD's joint decoder attention is what lets it **synthesize across passages** — the core capability concatenation also has but at far higher cost.

**The efficiency story:** encoding 50 passages separately is **50 × O(L²)** (cheap, parallelizable), whereas concatenating them would be **O((50L)²)** — FiD makes using 50 passages practical.

</details>

---

## Q6. Why does using more passages improve FiD's performance, and what's the limit? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Why more passages help (the original FiD finding):**

1. **Higher recall of the answer-bearing passage.** With more retrieved passages, the probability that *at least one* contains the answer rises. FiD's decoder can find and use it even if it's ranked low.

2. **Evidence aggregation.** Multiple passages may each contain *partial* or *corroborating* evidence. The decoder's joint cross-attention **combines** them — improving robustness and enabling multi-passage synthesis.

3. **Noise tolerance.** FiD learns to **ignore** irrelevant passages and attend to useful ones, so adding passages rarely hurts and often helps (up to a point). The original showed near-monotonic gains from 1 → 100 passages.

**The limits / diminishing returns:**

1. **Decoder cross-attention cost grows with k·L.** While *encoding* is linear in k, the **decoder must cross-attend over all k·L encoded tokens** — at large k this becomes the **dominant cost and bottleneck** (the target of FiD-light/FiDO optimizations).

2. **Diminishing recall gains.** Beyond a point, extra passages are increasingly likely to be irrelevant; recall plateaus while cost keeps rising.

3. **Distraction risk.** Very many low-quality passages can dilute attention, though FiD is relatively robust to this.

**Practical sweet spot:** the original FiD benefited up to ~100 passages on open-domain QA, but the **optimal k** depends on retriever quality and latency budget — a strong retriever needs fewer passages. **Reranking to prune to the most useful passages** before FiD is a common way to get the recall benefit without paying for 100 encodings.

**The fundamental trade-off:** more passages → better recall/answers, but **decoder cross-attention cost scales with total retrieved tokens** → tune k to your latency budget.

</details>

---

## Q7. What are FiD's main limitations and how did later work address them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Limitations:**

1. **Decoder cross-attention is the bottleneck.** Encoding is linear in k, but the **decoder cross-attends over all k·L encoded tokens at every generation step**. At large k, this **memory-bandwidth-bound** cross-attention dominates inference cost.

2. **Memory footprint.** Storing k separate encoded passage representations (each L tokens) consumes significant memory, limiting k on a given GPU.

3. **Inference latency.** The per-step cross-attention over many passages makes generation slower than a single-passage reader.

4. **Retriever-dependent.** FiD only reads what it's given; a weak retriever caps its quality (FiD doesn't fix retrieval).

5. **No retrieval during generation.** Unlike RETRO/FLARE, FiD retrieves **once** up front; it can't fetch new evidence mid-answer.

**How later work addressed them:**

| Improvement | What it does |
|---|---|
| **FiD-light** | Reduces the number of encoded tokens the decoder attends to (compress encoder output) → big speedup with small quality loss |
| **FiDO** (FiD-Optimized) | Re-balances the architecture toward the bottleneck — reduces decoder cross-attention cost, reallocates compute; major latency gains |
| **Reranking before FiD** | Prune to the top most-useful passages so k is smaller without losing recall |
| **Encoder output compression / caching** | Cache passage encodings (they're query-dependent in FiD, so this is limited) or compress them |

**Core lesson:** FiD's elegance (linear-in-k *encoding*) shifted the bottleneck to **decoder cross-attention over total tokens** — so the optimization frontier (FiD-light, FiDO) targets *that* cross-attention, not the encoder.

</details>

---

## Q8. How do you evaluate FiD, and against what baselines? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Benchmarks:** open-domain QA — **Natural Questions (NQ)**, **TriviaQA**, **WebQuestions** (the original FiD datasets). Metric: **exact-match (EM)** answer accuracy.

**Baselines / comparisons:**

| Comparison | What it isolates |
|---|---|
| **Extractive readers** (e.g., DPR reader, BERT-span) | Generative fusion vs span extraction |
| **Concatenation reader** (same passages, concatenated input) | FiD's efficiency/scaling benefit at equal passage count |
| **Closed-book seq2seq** (T5 without retrieval) | Value of retrieval/reading vs parametric memory |
| **Single-passage reader** | The benefit of fusing *many* passages |

**FiD-specific evaluation moves:**
1. **Scaling curve: EM vs number of passages k.** The signature FiD result — performance rising as k grows (1 → 100). This demonstrates the fusion benefit and identifies your task's sweet-spot k.
2. **Cost vs k.** Pair the quality curve with **latency/memory vs k** to choose k under a budget (and to motivate FiD-light/FiDO if the decoder bottleneck bites).
3. **Retriever ablation.** Fix the reader, vary the retriever (BM25 vs DPR vs Contriever) — since FiD's ceiling is set by retrieval recall.
4. **Robustness to irrelevant passages.** Inject distractors; measure degradation (FiD is relatively robust).

**Principle:** FiD's headline is **"more passages → better answers,"** so the **EM-vs-k scaling curve (with its cost counterpart)** is the central evaluation, alongside isolating the **reader** from the **retriever**.

</details>

---

## Q9. Design considerations for using FiD as the reader in a production RAG system. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
GOAL: a generative reader that fuses many retrieved passages efficiently
      for open-domain / knowledge QA.

ARCHITECTURE
────────────
- Retriever (your choice): DPR / Contriever / hybrid — FiD is retriever-agnostic.
- Reader: T5-based FiD — encode each (query+passage) separately, fuse in decoder.
- Optional reranker between retriever and FiD to prune passages.

KEY DECISIONS
─────────────
1. Number of passages k:
   - More k → better recall/answers but decoder cross-attention cost ↑.
   - Use a RERANKER to pass FiD the top (e.g., 10–20) most useful passages
     instead of a raw top-100 → most of the recall benefit, far less cost.
2. Passage length L: shorter passages → cheaper per-passage encoding;
   balance against losing context.
3. Latency budget → drives k. If the decoder cross-attention bottleneck
   bites, adopt FiD-light / FiDO style optimizations.
4. Encoder parallelism: encode the k passages in parallel (independent) —
   exploit FiD's core property for throughput.

WHEN FiD vs OTHER READERS / INFERENCE-TIME RAG
───────────────────────────────────────────────
- FiD shines when you want to fuse MANY passages generatively and a single
  long-context prompt would be too costly/long.
- Modern long-context LLMs can sometimes just take concatenated passages in
  the prompt — simpler, no special reader — but cost grows with context and
  "lost-in-the-middle" hurts; FiD remains attractive when fusing large k
  efficiently matters and you control the reader.

OPS / MONITORING
────────────────
- EM/F1 vs k curve; decoder cross-attention latency vs k; retrieval recall@k;
  GPU memory per request (k×L encodings); distractor robustness.
```

The central production lever is **k + a reranker**: use the reranker to feed FiD a small set of high-value passages so you capture the fusion benefit without paying the decoder-cross-attention cost of hundreds of passages.

</details>

---

## Q10. What is FiD's lasting influence on RAG architectures? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

1. **The default generative reader for retrieval QA.** FiD established **"encode passages separately, fuse in the decoder"** as the standard way to build a reader that consumes many retrieved passages. It became a **building block** — most prominently, **Atlas's reader is FiD**.

2. **Decoupled the reader from the retriever.** FiD is **retriever-agnostic**, reinforcing the modular RAG view: retrieval and reading are separate, swappable components. You can upgrade the retriever without touching the reader and vice versa.

3. **Showed that *more passages help* — quantitatively.** The EM-vs-k scaling curve made "retrieve more, read more, answer better" a concrete, measurable principle, motivating higher-recall retrieval.

4. **Identified the cross-attention bottleneck**, spawning an optimization line (**FiD-light, FiDO**) and informing how efficient long-context fusion is engineered.

5. **Generative > extractive for open-domain QA.** FiD helped shift the field from span-extraction readers to **generative** readers that synthesize answers across passages — the norm today.

**Relation to modern practice:** with strong long-context LLMs, many systems now just **stuff retrieved passages into the prompt** (a form of "fusion in the context window") rather than using a dedicated FiD reader. But this is conceptually FiD's descendant — and FiD's **efficiency insight** (don't pay quadratic cost to read many independent passages) still matters wherever you fuse large numbers of passages. FiD is the architecture that made **reading many passages practical**, and its modular, generative, scale-with-passages philosophy is now standard.

</details>

---

## Q11. What is FiD's cost and latency profile, and where is the bottleneck? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
ENCODING (cheap, parallel):
  k passages × O(L²) self-attention each = O(k·L²)  → LINEAR in k.
  Passages are independent → encode them in PARALLEL (high throughput).

DECODING (the bottleneck):
  At EACH generated token, the decoder CROSS-ATTENDS over the concatenated
  encoded passages = k·L encoded tokens.
  → cost grows with k·L; memory-bandwidth-bound; this DOMINATES at large k.

MEMORY:
  Must hold k separate encoded representations (k·L token vectors) → large
  footprint, limits how big k can be on a given GPU.
```

**Where the cost concentrates:** counter-intuitively **not** the encoder (FiD's whole point is linear encoding) but the **decoder's cross-attention over all passage tokens**, plus the **memory** to store k encodings.

**Optimizations:**
1. **Rerank → smaller k.** Feed FiD only the top most-useful passages (biggest practical lever).
2. **FiD-light / FiDO.** Compress encoder outputs / rebalance compute to shrink decoder cross-attention — large speedups.
3. **Shorter passages (L).** Reduces both encoding and the k·L the decoder attends over.
4. **Parallel encoding.** Exploit passage independence for throughput.
5. **Quantization / efficient attention kernels** for the decoder cross-attention.

**Framing:** FiD trades the **encoder's** quadratic-in-k cost (eliminated → linear) for a **manageable decoder-cross-attention cost** that nonetheless becomes the bottleneck at high k — so production tuning centers on **controlling k** (via reranking) and adopting **FiD-light/FiDO** when needed.

</details>

---

## Q12. What are the security and robustness considerations for FiD? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

FiD reads whatever passages it's given, so most risks enter through the **retrieved passages** — and FiD's *fusion* adds a few twists:

**1. Prompt injection via retrieved passages.**
Each encoded passage feeds the decoder; a **poisoned passage** containing injected instructions can influence generation. With **many** passages, the attack surface is larger (more documents, more chances one is malicious).
- *Mitigation:* treat passage content as untrusted data; sanitize; spotlight/mark retrieved text; source-trust scoring before retrieval.

**2. Evidence-conflict / majority manipulation.**
Because FiD **fuses across passages** and benefits from corroboration, an attacker who injects **multiple** agreeing poisoned passages can **outvote** correct evidence (the fusion that aids robustness can be gamed).
- *Mitigation:* deduplicate near-identical passages; source diversity/trust weighting; detect coordinated/duplicated content.

**3. Distraction / denial via irrelevant passages.**
Flooding retrieval with irrelevant or adversarial passages can dilute the decoder's attention and degrade answers (and inflate cost).
- *Mitigation:* rerank/filter before FiD; cap k; monitor passage relevance.

**4. Faithfulness / hallucination.**
As a **generative** reader, FiD can produce answers not grounded in any passage, especially when the answer isn't retrieved.
- *Mitigation:* faithfulness checks (answer entailed by attended passages); abstain on weak retrieval; attribution/citation of the supporting passage.

**5. Access control & PII leakage.**
FiD will read — and can copy from — any passage retrieved, including documents a user shouldn't see or PII.
- *Mitigation:* ACL-filtered retrieval / per-tenant indexes; PII scrubbing pre-indexing; output filtering.

**6. Cost-based DoS.**
Since cost scales with k·L, an attacker forcing large-k retrieval (e.g., via crafted queries) can drive up compute.
- *Mitigation:* hard caps on k and passage length; per-request budgets.

</details>

---

## Real-World Applications

| Application | Domain | Why Fusion-in-Decoder Fits |
|---|---|---|
| Open-domain question answering | Search / Knowledge | Efficiently fuses many retrieved passages to synthesize and corroborate answers |
| The reader behind larger RAG systems (e.g., Atlas) | ML platform / R&D | A modular, retriever-agnostic generative reader that scales with passage count |
| Knowledge-intensive assistants needing evidence synthesis | Enterprise | Decoder fusion aggregates partial evidence spread across multiple documents |
| Long-tail / high-recall QA over large corpora | Enterprise search | Using many passages raises answer recall without quadratic concatenation cost |
| Research on efficient retrieval-augmented readers | Academia | The canonical architecture (and bottleneck) that FiD-light / FiDO optimize |
