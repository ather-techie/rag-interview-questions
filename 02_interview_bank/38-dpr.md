# DPR — Dense Passage Retrieval

> The foundational bi-encoder architecture that established learned dense retrieval — the basis for every modern RAG embedding pipeline.

---

## What is DPR?

Dense Passage Retrieval (Karpukhin et al., Facebook AI, 2020) is the architecture that replaced sparse BM25 retrieval with *learned dense vectors* for open-domain question answering. DPR trains two separate BERT encoders — one for questions and one for passages — using contrastive learning on (question, positive passage, negative passages) triplets. The result: a retrieval system that understands *semantic intent* rather than just keyword overlap.

Every modern RAG embedding model (BGE, E5, Contriever, GTE, Nomic-Embed) is a descendant of DPR. Understanding DPR is understanding the foundations of dense retrieval.

---

## Architecture

```
DPR = Two Independent Encoders + Inner Product Similarity

┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│  Question Encoder (BERT-base)   │   │  Passage Encoder (BERT-base)    │
│                                 │   │                                 │
│  Input: "What is RAG?"          │   │  Input: "RAG combines retrieval │
│                                 │   │  with generative models..."     │
│  [CLS] token → d-dim vector     │   │  [CLS] token → d-dim vector     │
│  q = E_Q(question)              │   │  p = E_P(passage)               │
└────────────────┬────────────────┘   └────────────────┬────────────────┘
                 │                                      │
                 └──────────────┬───────────────────────┘
                                │
                         sim(q, p) = q · p   (inner product)
                         
Retrieval: find top-k passages maximizing q · p_i
```

Key design decision: **two separate encoders** (not a single cross-encoder). This enables offline indexing — you can pre-compute all passage embeddings and store them in FAISS before any query arrives.

### Key Components

| Component | Responsibility |
|---|---|
| **Question Encoder (BERT-base)** | Encodes the incoming query into a d-dim dense vector at inference time |
| **Passage Encoder (BERT-base)** | Encodes every corpus passage into a d-dim dense vector, run offline/in advance |
| **Offline Passage Index** | Stores all pre-computed passage embeddings for fast lookup at query time |
| **Inner-Product Retriever** | Computes q · p over the index and returns the top-k highest-scoring passages |
| **Downstream Reader/Generator** | Consumes retrieved passages to extract an answer span or generate a response |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| **Vector index** | FAISS (`IndexFlatIP` for exact inner-product search) |
| **Model library** | HuggingFace `transformers` — `DPRQuestionEncoder`, `DPRContextEncoder` |
| **Base checkpoints** | BERT-base (uncased) checkpoints for both encoders |

---

## Training Objective

DPR is trained with the InfoNCE (in-batch negative) loss:

```
For each training sample (q, p+, p1-, p2-, ..., pn-):
  sim(q, p+)  = dot product with positive passage (should be HIGH)
  sim(q, pi-) = dot product with negative passages (should be LOW)

Loss = -log [ exp(sim(q, p+)) / Σ exp(sim(q, pi)) ]
           = cross entropy over batch, treating in-batch passages as negatives
```

The "in-batch negatives" trick: every other passage in the same training batch is a free negative — at batch size 128, you get 127 negatives per positive without any additional sampling.

```python
import torch
import torch.nn.functional as F

def dpr_loss(q_vecs: torch.Tensor, p_vecs: torch.Tensor) -> torch.Tensor:
    """
    q_vecs: [batch, dim] question embeddings
    p_vecs: [batch, dim] passage embeddings (i-th passage is positive for i-th question)
    """
    # Similarity matrix: [batch, batch]
    sim_matrix = torch.matmul(q_vecs, p_vecs.T)  # q_i · p_j
    
    # Diagonal entries are positive pairs
    # Off-diagonal entries are in-batch negatives
    labels = torch.arange(q_vecs.size(0), device=q_vecs.device)
    
    # NLL loss: maximize diagonal, minimize off-diagonal
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
```

---

## Hard Negatives vs. In-Batch Negatives

In-batch negatives are easy negatives (randomly sampled passages). DPR's actual training uses **hard negatives** — passages that are semantically similar to the question but *do not* answer it. Hard negatives force the model to learn fine-grained distinctions.

```
Easy negative: "RAG" question → negative from a cooking recipe passage
Hard negative: "RAG" question → negative from a passage about "retrieval" in information retrieval theory
               (topically related, but not the answer)
```

Sources of hard negatives:
1. **BM25 top-k** — keyword-matching passages that don't semantically answer the question
2. **In-batch from similar questions** — passages retrieved by similar questions in the same batch
3. **Random negatives** from the corpus

```python
def sample_hard_negatives(question: str, positive_id: str, bm25_index, corpus, n: int = 7) -> list[str]:
    """Use BM25 to find passages that match keywords but aren't the positive."""
    bm25_results = bm25_index.search(question, k=100)
    negatives = [
        corpus[r["id"]] for r in bm25_results
        if r["id"] != positive_id
    ][:n]
    return negatives
```

---

## Full DPR Training Pipeline

```python
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

class DPREncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        return output.last_hidden_state[:, 0, :]  # [batch, 768]


class DPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.question_encoder = DPREncoder()
        self.passage_encoder  = DPREncoder()
    
    def encode_question(self, input_ids, attention_mask) -> torch.Tensor:
        return self.question_encoder(input_ids, attention_mask)
    
    def encode_passage(self, input_ids, attention_mask) -> torch.Tensor:
        return self.passage_encoder(input_ids, attention_mask)


# Training step
def train_step(model, batch, optimizer):
    q_ids, q_mask  = batch["question_input_ids"], batch["question_attention_mask"]
    p_ids, p_mask  = batch["passage_input_ids"],  batch["passage_attention_mask"]
    
    q_vecs = model.encode_question(q_ids, q_mask)
    p_vecs = model.encode_passage(p_ids, p_mask)
    
    loss   = dpr_loss(q_vecs, p_vecs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## Offline Indexing

The two-encoder design allows offline indexing of all passages before any query arrives:

```python
import faiss
import numpy as np
from tqdm import tqdm

def build_dpr_index(passages: list[str], passage_encoder, tokenizer, batch_size: int = 256) -> faiss.Index:
    all_embeddings = []
    
    for i in tqdm(range(0, len(passages), batch_size)):
        batch   = passages[i: i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            embs = passage_encoder(**encoded)  # [batch, 768]
        
        all_embeddings.append(embs.cpu().numpy())
    
    all_embeddings = np.vstack(all_embeddings).astype("float32")
    
    # Build FAISS inner product index
    dim   = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact inner product (for normalized: equivalent to cosine)
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)
    
    return index


def dpr_retrieve(question: str, question_encoder, tokenizer, faiss_index, passages, k: int = 5) -> list[str]:
    encoded = tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        q_vec = question_encoder(**encoded).cpu().numpy().astype("float32")
    
    faiss.normalize_L2(q_vec)
    scores, indices = faiss_index.search(q_vec, k)
    
    return [passages[i] for i in indices[0]]
```

---

## DPR vs. BM25 vs. Modern Embedding Models

| Dimension | BM25 | DPR (2020) | BGE / E5 / Nomic (2023+) |
|-----------|------|-----------|--------------------------|
| **Matching type** | Exact keyword | Semantic (bi-encoder) | Semantic (bi-encoder, larger) |
| **Training** | None (heuristic) | Contrastive on NQ/TriviaQA | Contrastive + instruction tuning |
| **Model size** | None | 2× BERT-base (~220M params) | 110M–7B params |
| **Out-of-domain** | Good | Moderate | Strong |
| **Code / technical text** | Poor | Moderate | Strong (domain fine-tuned) |
| **Multilingual** | Limited | No | Yes (mE5, LaBSE) |
| **Zero-shot QA (NQ)** | ~38% top-20 | ~78% top-20 | ~85%+ top-20 |

---

## DPR's Legacy: What It Established

1. **Bi-encoder = offline indexability** — the two-encoder design is the foundational insight enabling billion-document RAG
2. **Hard negatives are critical** — in-batch negatives alone produce mediocre models; hard negatives define modern training
3. **Inner product = retrieval metric** — normalized dot product (= cosine) became the universal retrieval similarity
4. **Separate question/passage encoders** — empirically better than a shared encoder because queries and passages have different distributional properties

Every modern embedding model fine-tunes or extends the DPR training recipe.

---

## Key Takeaways

1. **DPR = two BERT encoders trained with contrastive loss** on question-passage pairs — the architecture behind all dense retrieval.
2. **In-batch negatives + hard negatives** are what make the training work; understanding this explains why fine-tuning embedding models is powerful.
3. **Offline indexing** (pre-compute all passage embeddings) is DPR's killer feature — it makes billion-scale retrieval practical.
4. **Modern models (BGE, E5, Nomic) are DPR++ ** — larger, instruction-tuned, multilingual, but the same bi-encoder + contrastive recipe.
5. **DPR underperforms BM25 on exact match queries** — hybrid search (DPR + BM25 + RRF) is better than either alone.

---

## Interview Q&A

**Q: What is DPR and why was it a breakthrough for open-domain QA?**

Dense Passage Retrieval (Karpukhin et al., 2020) replaced sparse BM25 retrieval with two BERT encoders trained end-to-end with contrastive loss on question-passage pairs. The breakthrough was showing that a bi-encoder (separate encoders for questions and passages) trained on curated QA pairs could significantly outperform BM25 on open-domain QA benchmarks — ~78% top-20 recall on Natural Questions vs. BM25's ~38% — while keeping queries fast via offline indexing. It established the two key ideas that all modern RAG relies on: (1) semantic retrieval via learned dense vectors is more powerful than keyword matching; (2) offline pre-computation of passage embeddings makes billion-scale dense retrieval practical.

---

**Q: Why does DPR use two separate encoders rather than one shared encoder?**

Empirically, separate encoders outperform a shared encoder. The intuition: questions and passages come from different distributions — questions are typically short, interrogative, and keyword-sparse; passages are declarative, longer, and informationally dense. A shared encoder must produce vectors from both distributions in the same space, which creates a tension during training. Separate encoders can specialize: the question encoder learns to represent *intent*; the passage encoder learns to represent *answer-relevant content*. The cost is double the parameter count (~220M instead of ~110M for BERT-base), but the accuracy gain justifies it. Modern models like BGE and E5 follow the same pattern with instruction-tuned asymmetric encoders (different prompts for queries vs. passages).

---

**Q: What are hard negatives in DPR training and why are they important?**

Hard negatives are passages that are topically related to the question (retrieved by BM25 or a previous dense retriever) but do not actually answer it. In-batch negatives, by contrast, are random passages that happen to be in the same training batch — they're usually obviously irrelevant. Training on only in-batch negatives produces a model that can distinguish "cooking recipe" from "RAG paper" but struggles to distinguish "a passage that mentions retrieval in an unrelated context" from "the passage that actually answers the question." Hard negatives force the model to learn fine-grained distinctions, which is precisely what production retrieval requires. Modern training recipes (BGE, E5) combine BM25 hard negatives, mined hard negatives from a previous model iteration, and in-batch negatives for the strongest training signal.
