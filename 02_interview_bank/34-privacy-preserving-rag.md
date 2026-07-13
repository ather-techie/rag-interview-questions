# Privacy-Preserving RAG

> How to build RAG systems that retrieve relevant information without exposing the corpus, user queries, or embeddings to untrusted parties.

---

## What is Privacy-Preserving RAG?

Privacy-preserving RAG addresses a gap that multi-tenancy and ACL controls don't cover: in standard RAG, the retrieval server sees every query and every retrieved document. In regulated industries (healthcare, finance, legal), this creates risks:
- A vendor-hosted vector DB sees raw patient queries
- Embeddings can be inverted to approximate the original text
- Query logs reveal user intent even when document access is controlled

Privacy-preserving RAG uses a combination of techniques — differential privacy, federated retrieval, secure computation, and on-device processing — to reduce what any single party can learn.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
INGESTION (offline, per silo)
  Raw Documents
        │
        ▼
  Anonymizer (NER + regex PII scrub)
        │
        ▼
  Embedder ──► Local Vector Index

QUERY (online)
  Client Query
        │
        ▼
  On-device Embedder (local model, raw text never leaves the client)
        │
        ▼
  DP Noise Injector (calibrated Gaussian/Laplace noise, budget ε)
        │
        ▼
  k-Anonymity Query Obfuscator (optional: real query + k−1 dummy queries)
        │
        ▼
  Federated Retrieval Coordinator
        │
        ├─► Silo A index ─┐
        ├─► Silo B index ─┤── merge via Reciprocal Rank Fusion (RRF)
        └─► Silo C index ─┘
        │
        ▼
  Ranked Results (server never saw raw query text or the full corpus)
```

### Key Components

| Component | Responsibility |
|---|---|
| On-device Embedder | Converts the query to a vector locally so raw query text never reaches the server |
| DP Noise Injector | Adds calibrated noise to the query embedding to defeat vec2text-style inversion attacks |
| Query Obfuscator | Wraps the real query with k−1 dummy queries so the server cannot tell which result was wanted |
| Federated Retrieval Coordinator | Dispatches the query to per-silo indexes in parallel and merges ranked lists via RRF without seeing raw content |
| Anonymizer (ingestion-time) | Scrubs PII from documents via NER/regex before they are embedded and indexed |
| Trust / Access Layer | Enforces per-tenant and per-silo authorization on top of the privacy techniques above |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| On-device embedding | sentence-transformers, ONNX Runtime (mobile/edge inference) |
| PII anonymization | spaCy NER, regex pattern libraries, Microsoft Presidio |
| Differential privacy | Custom Laplace/Gaussian noise implementation, Opacus (DP-trained models) |
| Federated retrieval | Custom async coordinator (asyncio), gRPC between silo services |

---

## Threat Model

| Threat | Example | Mitigation |
|--------|---------|-----------|
| **Query exposure** | Vendor sees raw user query | Query obfuscation, on-device embedding |
| **Corpus exposure** | Retrieval API reveals document content | Federated retrieval, blind retrieval |
| **Embedding inversion** | Embeddings reconstructed to approximate text | Differential privacy on embeddings |
| **Membership inference** | Attacker infers if a document is in the corpus | DP training for embedding models |
| **Cross-tenant leakage** | Tenant A's query retrieves Tenant B's data | Per-tenant index isolation (see multi-tenancy guide) |

---

## Technique 1: On-Device Embedding

Embed the query locally (on the client device) before sending it to the retrieval server. The server only sees a vector, not the raw query text.

```
Without on-device embedding:
  Client ──► "What is my HIV test result?"  ──► Retrieval Server (sees query)

With on-device embedding:
  Client ──► [0.21, -0.14, 0.88, ...]  ──────► Retrieval Server (sees only vector)
```

```python
# Client-side (on device, never leaves the device)
from sentence_transformers import SentenceTransformer

LOCAL_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")  # small enough for on-device

def embed_locally(query: str) -> list[float]:
    emb = LOCAL_MODEL.encode(query, normalize_embeddings=True)
    return emb.tolist()

# Only the vector is sent to the server
query_vector = embed_locally("What is my HIV test result?")
results = retrieval_server.search(query_vector, k=5)  # server never sees raw text
```

**Limitation:** The vector itself can be approximately inverted with vec2text-style attacks if the embedding model and architecture are known. Add differential privacy noise to the embedding before sending for stronger guarantees.

---

## Technique 2: Differential Privacy on Embeddings

Add calibrated Laplace or Gaussian noise to the query embedding before transmission. The retrieval quality degrades slightly (controlled by the privacy budget ε), but the server cannot reliably invert the noisy vector.

```python
import numpy as np

def privatize_embedding(
    emb: np.ndarray,
    epsilon: float = 1.0,    # privacy budget; smaller = more private, less accurate
    sensitivity: float = 1.0  # L2 sensitivity of normalized embeddings ≈ 1.0
) -> np.ndarray:
    """Add Gaussian noise calibrated to (epsilon, delta=1e-5)-DP."""
    delta   = 1e-5
    sigma   = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise   = np.random.normal(0, sigma, size=emb.shape)
    noisy   = emb + noise
    # Re-normalize so cosine similarity still works
    return noisy / np.linalg.norm(noisy)


# Privacy-accuracy trade-off for epsilon values:
#   epsilon = 0.1 → strong privacy, ~15% recall drop
#   epsilon = 1.0 → moderate privacy, ~5% recall drop  ← common production choice
#   epsilon = 10  → weak privacy, ~0.5% recall drop

def private_retrieval(query: str, epsilon: float = 1.0, k: int = 5):
    emb       = embed_locally(query)
    noisy_emb = privatize_embedding(np.array(emb), epsilon=epsilon)
    return retrieval_server.search(noisy_emb.tolist(), k=k)
```

---

## Technique 3: Federated Retrieval

In federated retrieval, each data silo (e.g., each hospital, each department) runs its own local retrieval index. A central coordinator aggregates ranked lists without ever seeing the documents or queries.

```
Traditional (centralized):
  All Documents ──► Central Index ──► All Queries routed here

Federated:
  Hospital A index ──┐
  Hospital B index ──┤──► Coordinator (merges ranked lists via RRF)
  Hospital C index ──┘
  
  Each index sees only its own queries (routed by the coordinator)
  Coordinator sees only ranked doc IDs + scores — no raw content
```

```python
import asyncio
from typing import NamedTuple

class RankedResult(NamedTuple):
    silo_id: str
    doc_id: str
    score: float

async def federated_retrieve(
    query_vector: list[float],
    silos: list,  # each has a .search(vector, k) method
    k: int = 5,
) -> list[RankedResult]:
    """Query each silo in parallel; merge results with RRF."""
    
    async def query_silo(silo):
        results = await asyncio.to_thread(silo.search, query_vector, k * 2)
        return [(silo.id, r["doc_id"], r["score"]) for r in results]
    
    all_results = await asyncio.gather(*[query_silo(s) for s in silos])
    
    # Reciprocal Rank Fusion across silos
    rrf_scores = {}
    for silo_results in all_results:
        for rank, (silo_id, doc_id, _) in enumerate(silo_results):
            key = (silo_id, doc_id)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
    
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [RankedResult(silo_id=k[0], doc_id=k[1], score=v) for k, v in merged[:k]]
```

---

## Technique 4: Anonymization Before Indexing

Scrub PII from documents before embedding and storing. This reduces the risk of a retrieval result directly exposing sensitive data.

```python
import re
import spacy

nlp = spacy.load("en_core_web_sm")

PII_PATTERNS = {
    "SSN":   r"\b\d{3}-\d{2}-\d{4}\b",
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "Phone": r"\b(\+1)?\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
}

def anonymize_text(text: str) -> str:
    # Regex-based PII removal
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{label}]", text)
    
    # NER-based entity removal (names, organizations, locations)
    doc = nlp(text)
    for ent in reversed(doc.ents):  # reversed to preserve character offsets
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}:
            text = text[:ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]
    
    return text


def build_private_index(documents: list[str], embed_fn) -> list[dict]:
    return [
        {"id": i, "vector": embed_fn(anonymize_text(doc)), "original": doc}
        for i, doc in enumerate(documents)
    ]
```

---

## Technique 5: Query Obfuscation (k-Anonymity Queries)

Send the real query embedding alongside k−1 dummy queries, so the server cannot determine which result the client actually wanted.

```python
import numpy as np

def obfuscate_query(
    real_emb: np.ndarray,
    k_anon: int = 4,
    noise_scale: float = 0.1,
) -> tuple[list[np.ndarray], int]:
    """Return k query vectors; real query is at a random index."""
    dummies = [real_emb + np.random.normal(0, noise_scale, real_emb.shape)
               for _ in range(k_anon - 1)]
    # normalize dummies
    dummies = [d / np.linalg.norm(d) for d in dummies]
    
    real_index = np.random.randint(0, k_anon)
    queries = dummies[:real_index] + [real_emb] + dummies[real_index:]
    return queries, real_index


def k_anonymous_retrieve(query: str, k_anon: int = 4) -> list:
    real_emb = np.array(embed_locally(query))
    queries, real_idx = obfuscate_query(real_emb, k_anon=k_anon)
    
    # Send all k queries to server; only use results for the real one
    all_results = [retrieval_server.search(q.tolist(), k=5) for q in queries]
    return all_results[real_idx]
```

---

## Privacy-Accuracy Trade-off Summary

| Technique | Privacy Guarantee | Retrieval Impact | Complexity |
|-----------|------------------|-----------------|-----------|
| On-device embedding | Query text hidden from server | None | Low |
| DP noise (ε=1.0) | Embedding ≈ uninvertible | ~5% recall drop | Low |
| Federated retrieval | Each silo sees only its own docs | Moderate (RRF merge) | High |
| Anonymization before indexing | PII not stored in index | Minor (entity loss) | Medium |
| k-Anonymity queries | Query identity hidden | None (k× server load) | Low |

---

## Compliance Considerations

| Regulation | Requirement | RAG Implication |
|-----------|-------------|----------------|
| **HIPAA** | PHI must be protected at rest and in transit | Anonymize before indexing; encrypt vectors at rest |
| **GDPR Art. 17** | Right to erasure | Must be able to delete all vectors derived from a document |
| **GDPR Art. 25** | Privacy by design | On-device embedding as default; no raw query logging |
| **CCPA** | User data opt-out | Per-user semantic memory must be deletable on request |

---

## Key Takeaways

1. **On-device embedding is the easiest first step** — a small ONNX model on the client hides raw queries at near-zero cost.
2. **DP noise is the main tool against embedding inversion** — ε = 1.0 is a practical starting point; tune against your recall SLA.
3. **Federated retrieval is complex but necessary** when data *cannot* leave its silo (hospital-to-hospital, cross-org).
4. **Anonymize before indexing, not after** — post-hoc PII removal from a deployed index is error-prone and auditable gaps remain.
5. **Log minimization is as important as encryption** — if query logs are never written, they can never leak.

---

## Interview Q&A

**Q: What is the difference between multi-tenant ACL-based RAG and privacy-preserving RAG?**

Multi-tenant ACL controls *who can access what* — the server is trusted and enforces access rules. Privacy-preserving RAG addresses the case where *the server itself is untrusted* (or must not see certain data). Example: a SaaS retrieval service run by a vendor should not see raw patient queries even if it correctly enforces tenant isolation. ACL + encryption handles cross-tenant leakage; on-device embedding + DP handles what the vendor can infer. In practice, production systems need both: ACL for authorization, privacy techniques for data minimization.

---

**Q: How does differential privacy apply to query embeddings?**

The embedding of a query is a high-dimensional vector. An adversary with access to the embedding and knowledge of the embedding model can use vec2text-style inversion attacks to approximately reconstruct the original text. Differential privacy adds calibrated Gaussian noise to the embedding before it leaves the client, such that the noisy vector is (ε, δ)-indistinguishable from the embedding of any neighboring query (differing in one word). The server's retrieval result changes slightly (because the query vector moved), but the client's privacy is protected with a formal guarantee. The privacy-accuracy trade-off is controlled by ε: smaller ε means more noise and lower recall; ε=1.0 is a common production starting point with ~5% recall degradation.

---

**Q: How do you implement the right-to-erasure (GDPR Art. 17) requirement in a RAG system?**

Every vector stored in the index must be traceable to its source document. Implement this with a `source_doc_id` metadata field on every vector at ingestion time. To erase a document: (1) query the vector DB for all vectors with `source_doc_id = X`; (2) delete those vectors by ID; (3) delete the raw document and its parsed chunks from the document store; (4) delete any per-user semantic memory entries that reference that document. The hard part is semantic memory — if a user's memory contains "the report mentioned X" derived from the deleted document, that entry is also subject to erasure. Maintain a mapping from `source_doc_id` to derived memory IDs to make this tractable.
