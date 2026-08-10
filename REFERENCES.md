# References

Full citations with links for every paper referenced in this repo, grouped by topic. Each entry includes a one-line note on why (and when) to read it.

> Reading strategy: you don't need to read papers to pass most RAG interviews — the Q&A files summarize what matters. Read the **Foundations** papers if you're interviewing for research-adjacent roles; skim the architecture papers for whichever RAG types your target role emphasizes.

---

## Foundations

| Paper | Link | Why read it |
|---|---|---|
| Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | The original RAG paper; defines retrieve-then-generate. Used in [`02_interview_bank/01-naive-rag.md`](./02_interview_bank/01-naive-rag.md) |
| Karpukhin et al., 2020 — *Dense Passage Retrieval for Open-Domain Question Answering* | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) | DPR — the bi-encoder dense retrieval recipe everything else builds on |
| Reimers & Gurevych, 2019 — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* | [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) | Why bi-encoders exist; the sentence-transformers library behind most embedding fine-tuning |
| Gao et al., 2023 — *Retrieval-Augmented Generation for Large Language Models: A Survey* | [arXiv:2312.10997](https://arxiv.org/abs/2312.10997) | The survey that popularized the Naive → Advanced → Modular taxonomy used in this repo |

## Architecture papers (by interview bank section)

| Paper | Link | Bank section |
|---|---|---|
| Yao et al., 2022 — *ReAct: Synergizing Reasoning and Acting in Language Models* | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) | [`04-agentic-rag.md`](./02_interview_bank/04-agentic-rag.md) — the reason/act loop agentic RAG is built on |
| Jiang et al., 2023 — *Active Retrieval Augmented Generation* (FLARE) | [arXiv:2305.06983](https://arxiv.org/abs/2305.06983) | [`04-agentic-rag.md`](./02_interview_bank/04-agentic-rag.md) — generate, detect uncertainty, re-retrieve |
| Edge et al., 2024 — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) | [`05-graph-rag.md`](./02_interview_bank/05-graph-rag.md) — Microsoft GraphRAG: KG + community detection |
| Yan et al., 2024 — *Corrective Retrieval Augmented Generation* (CRAG) | [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) | [`06-corrective-rag.md`](./02_interview_bank/06-corrective-rag.md) — retrieval evaluator + web-search fallback |
| Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) | [`07-self-rag.md`](./02_interview_bank/07-self-rag.md) — reflection tokens, trained retrieval decisions |
| Wang et al., 2024 — *Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting* | [arXiv:2407.08223](https://arxiv.org/abs/2407.08223) | [`08-speculative-rag.md`](./02_interview_bank/08-speculative-rag.md) — small drafter + large verifier |
| Radford et al., 2021 — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | [`09-multimodal-rag.md`](./02_interview_bank/09-multimodal-rag.md) — the joint text-image embedding space |
| Faysse et al., 2024 — *ColPali: Efficient Document Retrieval with Vision Language Models* | [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) | [`09-multimodal-rag.md`](./02_interview_bank/09-multimodal-rag.md) — late-interaction retrieval over page images, no OCR |
| Liu et al., 2023 — *Lost in the Middle: How Language Models Use Long Contexts* | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) | [`10-long-context-rag.md`](./02_interview_bank/10-long-context-rag.md) — why stuffing context fails; U-shaped attention |
| Jiang et al., 2023 — *LLMLingua: Compressing Prompts for Accelerated Inference of LLMs* | [arXiv:2310.05736](https://arxiv.org/abs/2310.05736) | [`10-long-context-rag.md`](./02_interview_bank/10-long-context-rag.md) — prompt compression for token cost |
| Jeong et al., 2024 — *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity* | [arXiv:2403.14403](https://arxiv.org/abs/2403.14403) | [`11-adaptive-rag.md`](./02_interview_bank/11-adaptive-rag.md) — query-complexity classifier routing |
| Trivedi et al., 2023 — *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions* (IRCoT) | [arXiv:2212.10509](https://arxiv.org/abs/2212.10509) | [`19-iterative-multihop-rag.md`](./02_interview_bank/19-iterative-multihop-rag.md) — interleave CoT with retrieval |
| Press et al., 2022 — *Measuring and Narrowing the Compositionality Gap in Language Models* (Self-Ask) | [arXiv:2210.03350](https://arxiv.org/abs/2210.03350) | [`19-iterative-multihop-rag.md`](./02_interview_bank/19-iterative-multihop-rag.md) — explicit follow-up sub-questions |
| Shao et al., 2023 — *Enhancing Retrieval-Augmented LMs with Iterative Retrieval-Generation Synergy* (ITER-RETGEN) | [arXiv:2305.15294](https://arxiv.org/abs/2305.15294) | [`19-iterative-multihop-rag.md`](./02_interview_bank/19-iterative-multihop-rag.md) — generated answer as next-round query |
| Gutiérrez et al., 2024 — *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models* | [arXiv:2405.14831](https://arxiv.org/abs/2405.14831) | [`20-hipporag.md`](./02_interview_bank/20-hipporag.md) — Personalized PageRank over an LLM-built KG |
| Packer et al., 2023 — *MemGPT: Towards LLMs as Operating Systems* | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) | [`21-memory-conversational-rag.md`](./02_interview_bank/21-memory-conversational-rag.md) — OS-style tiered memory paging |
| Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE) | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | [`22-hyde-rag.md`](./02_interview_bank/22-hyde-rag.md) — embed an LLM-generated hypothetical answer |
| Jiang et al., 2023 — *Active Retrieval Augmented Generation* (FLARE) | [arXiv:2305.06983](https://arxiv.org/abs/2305.06983) | [`23-flare-rag.md`](./02_interview_bank/23-flare-rag.md) — confidence-gated, forward-looking active retrieval |
| Liang et al., 2024 — *KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation* | [arXiv:2409.13731](https://arxiv.org/abs/2409.13731) | [`24-kag.md`](./02_interview_bank/24-kag.md) — logical-form reasoning + KG/text mutual indexing |
| Li et al., 2024 — *GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of LLMs* | [arXiv:2406.14550](https://arxiv.org/abs/2406.14550) | [`25-graphreader-gnn-rag.md`](./02_interview_bank/25-graphreader-gnn-rag.md) — agentic graph-of-notes traversal |
| Mavromatis & Karypis, 2024 — *GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning* | [arXiv:2405.20139](https://arxiv.org/abs/2405.20139) | [`25-graphreader-gnn-rag.md`](./02_interview_bank/25-graphreader-gnn-rag.md) — GNN-retrieved reasoning subgraphs for KGQA |
| Guu et al., 2020 — *REALM: Retrieval-Augmented Language Model Pre-Training* | [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) | [`26-realm.md`](./02_interview_bank/26-realm.md) — end-to-end learned retriever via masked-LM pre-training |
| Borgeaud et al., 2021 — *Improving Language Models by Retrieving from Trillions of Tokens* (RETRO) | [arXiv:2112.04426](https://arxiv.org/abs/2112.04426) | [`27-retro.md`](./02_interview_bank/27-retro.md) — chunked cross-attention over a trillion-token datastore |
| Izacard et al., 2022 — *Atlas: Few-shot Learning with Retrieval Augmented Language Models* | [arXiv:2208.03299](https://arxiv.org/abs/2208.03299) | [`28-atlas.md`](./02_interview_bank/28-atlas.md) — jointly-trained Contriever + FiD for few-shot knowledge tasks |
| Izacard & Grave, 2020 — *Leveraging Passage Retrieval with Generative Models for Open Domain QA* (Fusion-in-Decoder) | [arXiv:2007.01282](https://arxiv.org/abs/2007.01282) | [`29-fusion-in-decoder.md`](./02_interview_bank/29-fusion-in-decoder.md) — encode passages separately, fuse in the decoder |

## Retrieval & indexing

| Paper | Link | Why read it |
|---|---|---|
| Robertson & Zaragoza, 2009 — *The Probabilistic Relevance Framework: BM25 and Beyond* | [DOI:10.1561/1500000019](https://doi.org/10.1561/1500000019) | The definitive BM25 reference behind sparse/hybrid retrieval |
| Formal et al., 2021 — *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval* | [arXiv:2109.10086](https://arxiv.org/abs/2109.10086) | Learned sparse retrieval — the modern half of hybrid search. See [`01_concepts/retrieval_strategies.md`](./01_concepts/retrieval_strategies.md) |
| Khattab & Zaharia, 2020 — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction* | [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) | Late interaction — the middle ground between bi- and cross-encoders |
| Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE) | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | Hypothetical document embeddings for query rewriting. See [`02-advanced-rag.md`](./02_interview_bank/02-advanced-rag.md) |
| Zheng et al., 2023 — *Take a Step Back: Evoking Reasoning via Abstraction in LLMs* | [arXiv:2310.06117](https://arxiv.org/abs/2310.06117) | Step-back prompting for query abstraction. See [`01_concepts/retrieval_strategies.md`](./01_concepts/retrieval_strategies.md) |
| Cormack et al., 2009 — *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* | [DOI:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114) | RRF — the standard way to merge dense + sparse result lists |
| Malkov & Yashunin, 2016 — *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs* | [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) | The index behind most vector DBs. See [`01_concepts/vector_databases.md`](./01_concepts/vector_databases.md) |
| Sun et al., 2023 — *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents* (RankGPT) | [arXiv:2304.09542](https://arxiv.org/abs/2304.09542) | LLM listwise reranking. See [`01_concepts/reranking.md`](./01_concepts/reranking.md) |
| Kusupati et al., 2022 — *Matryoshka Representation Learning* | [arXiv:2205.13147](https://arxiv.org/abs/2205.13147) | Truncatable embeddings — why `text-embedding-3` lets you cut dimensions. See [`01_concepts/embeddings.md`](./01_concepts/embeddings.md) |
| Wang et al., 2021 — *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval* | [arXiv:2112.07577](https://arxiv.org/abs/2112.07577) | Synthetic training data for embedding fine-tuning. See [`01_concepts/fine_tuning.md`](./01_concepts/fine_tuning.md) |
| 2025 — *Training-Free Adaptive Retrieval Gating for Efficient RAG* (TARG) | [arXiv:2511.09803](https://arxiv.org/abs/2511.09803) | Training-free retrieve-or-not decision via token-entropy/logit-margin uncertainty on a no-context draft. See [`02_interview_bank/11-adaptive-rag.md`](./02_interview_bank/11-adaptive-rag.md) |

## Evaluation & benchmarks

| Paper | Link | Why read it |
|---|---|---|
| Es et al., 2023 — *RAGAS: Automated Evaluation of Retrieval Augmented Generation* | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | Reference-free RAG evaluation — the de facto standard. See [`01_concepts/evaluation_metrics.md`](./01_concepts/evaluation_metrics.md) |
| Liu et al., 2023 — *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* | [arXiv:2303.16634](https://arxiv.org/abs/2303.16634) | LLM-as-judge with chain-of-thought rubrics |
| Thakur et al., 2021 — *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models* | [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) | The retrieval generalization benchmark |
| Muennighoff et al., 2022 — *MTEB: Massive Text Embedding Benchmark* | [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) | How embedding models are compared; read before trusting any leaderboard number |
| Saad-Falcon et al., 2023 — *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems* | [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) | Fine-tuned judges + prediction-powered inference for calibrated scores. See [`08_evaluation/03-llm-judge-frameworks.md`](./08_evaluation/03-llm-judge-frameworks.md) |
| Ru et al., 2024 — *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation* | [arXiv:2408.08067](https://arxiv.org/abs/2408.08067) | Claim-level entailment checking for retriever-vs-generator diagnosis. See [`08_evaluation/03-llm-judge-frameworks.md`](./08_evaluation/03-llm-judge-frameworks.md) |
| Liu et al., 2025 — *Judge as A Judge: Improving the Evaluation of RAG through the Judge-Consistency of LLMs* (ConsJudge) | [arXiv:2502.18817](https://arxiv.org/abs/2502.18817) | Training judges via cross-judgment consistency + DPO, no human labels needed. See [`08_evaluation/03-llm-judge-frameworks.md`](./08_evaluation/03-llm-judge-frameworks.md) |
| Tamber et al., 2025 — *Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards* (FaithJudge) | [arXiv:2505.04847](https://arxiv.org/abs/2505.04847) | Human-guided (few-shot) LLM judge for hallucination, built on Vectara's Hallucination Leaderboard. See [`08_evaluation/03-llm-judge-frameworks.md`](./08_evaluation/03-llm-judge-frameworks.md) |
| Katsis et al., 2025 — *MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems* | [arXiv:2501.03468](https://arxiv.org/abs/2501.03468) | Human-authored multi-turn RAG benchmark; single-turn benchmarks miss this failure surface. See [`01_concepts/evaluation_metrics.md`](./01_concepts/evaluation_metrics.md) |

## Security

| Paper | Link | Why read it |
|---|---|---|
| Greshake et al., 2023 — *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* | [arXiv:2302.12173](https://arxiv.org/abs/2302.12173) | Indirect prompt injection — the core RAG attack surface. See [`01_concepts/prompt_injection_risks.md`](./01_concepts/prompt_injection_risks.md) |
| Hines et al., 2024 — *Defending Against Indirect Prompt Injection Attacks With Spotlighting* | [arXiv:2403.14720](https://arxiv.org/abs/2403.14720) | Marking untrusted content so the model can tell instructions from data |
| Zou et al., 2024 — *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models* | [arXiv:2402.07867](https://arxiv.org/abs/2402.07867) | USENIX Security 2025. Multi-document corpus poisoning to force a targeted wrong answer. See [`01_concepts/prompt_injection_risks.md`](./01_concepts/prompt_injection_risks.md) |
| Zhang et al., 2025 — *Practical Poisoning Attacks against Retrieval-Augmented Generation* (CorruptRAG) | [arXiv:2504.03957](https://arxiv.org/abs/2504.03957) | Single-document poisoning — cheaper, stealthier variant of PoisonedRAG's threat model. See [`01_concepts/prompt_injection_risks.md`](./01_concepts/prompt_injection_risks.md) |
| Liang et al., 2025 — *SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model* | [arXiv:2501.18636](https://arxiv.org/abs/2501.18636) | ACL 2025. Benchmark spanning noise/conflict/injection/DoS attacks across 14 RAG components. See [`01_concepts/prompt_injection_risks.md`](./01_concepts/prompt_injection_risks.md) |
