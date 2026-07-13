# 33 — Verifiable / Citation RAG

> Every claim in the generated answer is linked to a specific retrieved passage — and that link is verified, not assumed.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
    │
    ▼
Retriever (fetches candidate source chunks)
    │
    ▼
Citation-aware Generator (produces answer with inline citation markers per claim)
    │
    ▼
Attribution / Entailment Verifier (checks each cited claim against its source chunk via NLI or LLM-judge)
    │
    ▼
Citation Renderer (formats verified citations; flags or removes unsupported claims)
```

### Key Components

| Component | Responsibility |
|---|---|
| Retriever | Fetches candidate source chunks relevant to the query |
| Citation-aware Generator | Produces the answer with inline citation markers tied to specific passages |
| Attribution/Entailment Verifier | Checks whether the cited passage actually entails the paired claim |
| Citation Renderer | Formats verified citations in the final output, or flags/removes unsupported claims |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Retriever + Generator stack | Standard dense retriever paired with an LLM generator |
| NLI verification model | `cross-encoder/nli-deberta-v3-base`, `bart-large-mnli` |
| LLM-as-judge verification | Cheap model (e.g. Claude Haiku) prompted for SUPPORTED / NOT_SUPPORTED verdicts |
| Evaluation benchmark | ALCE (attribution scoring benchmark, Gao et al., 2023) |
| Post-processing | Citation-formatting and unsupported-claim flagging post-processor |

---

## Q1. What is Verifiable RAG and why is "citing a source" not the same as "grounding a claim"? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Verifiable RAG** (also called Citation RAG or Attributed RAG) extends standard RAG with two additional requirements:
1. Every factual claim in the output is paired with a citation pointing to a specific retrieved passage
2. That citation is **verified** — the passage actually supports the claim

**The grounding gap — why standard RAG citations often fail:**

In standard RAG, asking the LLM to "cite your sources" often produces hallucinated or inaccurate citations:

```
Hallucinated citation:
  Claim:    "RAG reduces hallucinations by 37%."
  Citation: [Source 2]
  Reality:  Source 2 discusses RAG architecture — never mentions 37%
```

```
Correct citation:
  Claim:    "RAG reduces hallucinations by 37%."
  Citation: [Source 2, paragraph 3]
  Verification: Source 2 paragraph 3 says "...reduced hallucination rate by 37%..."
  → Supported ✓
```

**Three levels of citation quality:**

| Level | Description | Failure Mode |
|-------|-------------|--------------|
| **Source-level** | Answer cites a document | LLM may cite document that doesn't support the claim |
| **Passage-level** | Answer cites a specific chunk | LLM may misattribute which sentence supports the claim |
| **Span-level** | Answer cites the exact span | Highest precision; requires span extraction |

**When Verifiable RAG is required:**

- Medical and legal contexts where incorrect citations create liability
- Research assistants where users follow citations to verify claims
- Enterprise compliance reporting where traceability is audited
- Financial analysis where specific figures must trace to specific documents

</details>

---

## Q2. How do you generate passage-level citations in a RAG response? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Method 1: In-context citation instructions**

Instruct the LLM to place inline citation markers after each claim, then map them back to retrieved passages.

```python
from anthropic import Anthropic

client = Anthropic()

def generate_with_citations(query: str, passages: list[dict]) -> dict:
    """
    passages: list of {"id": int, "text": str, "source": str}
    Returns: {"answer": str, "citations": list[{"claim": str, "passage_id": int}]}
    """
    passages_block = "\n\n".join(
        f"[{p['id']}] {p['text']}"
        for p in passages
    )
    
    prompt = f"""You are a precise research assistant. Answer the question using
the numbered passages below. After each factual claim in your answer, insert
a citation in brackets like [1] or [2] referencing the passage number.
Only cite passages that directly support the specific claim.
Do not add a citation if no passage supports the claim — instead, omit the claim.

Passages:
{passages_block}

Question: {query}

Answer (with inline citations):"""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Example output:
# "RAG was introduced in 2020 [1] and has been shown to reduce hallucinations
#  significantly [3]. The retrieval component typically uses a bi-encoder
#  architecture [2]."
```

**Method 2: Structured citation output**

Force structured output with explicit claim → citation mappings:

```python
import json

STRUCTURED_PROMPT = """Answer the question. For each factual claim, output JSON:
{{
  "claims": [
    {{"claim": "exact sentence from your answer", "passage_ids": [1, 3]}}
  ],
  "answer": "full answer text with [1],[2] inline markers"
}}

Passages:
{passages}

Question: {query}"""

def generate_structured_citations(query: str, passages: list[dict]) -> dict:
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": STRUCTURED_PROMPT.format(
            passages="\n".join(f"[{p['id']}] {p['text']}" for p in passages),
            query=query
        )}]
    )
    return json.loads(resp.content[0].text)
```

</details>

---

## Q3. How do you verify that a citation actually supports a claim? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Generating citations is easy; verifying them requires a separate **attribution verification** step.

**Method 1: NLI-based verification (Natural Language Inference)**

Use an NLI model to check whether the cited passage *entails* the claim.

```python
from transformers import pipeline

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

def verify_citation(claim: str, cited_passage: str) -> dict:
    """Returns entailment/neutral/contradiction + confidence."""
    result = nli(f"{cited_passage} [SEP] {claim}")[0]
    return {
        "claim": claim,
        "passage": cited_passage,
        "label": result["label"],      # ENTAILMENT / NEUTRAL / CONTRADICTION
        "confidence": result["score"],
        "supported": result["label"] == "ENTAILMENT" and result["score"] > 0.7
    }

# Example:
claim = "RAG reduces hallucinations by 37%."
passage = "In our experiments, RAG-equipped models showed a 37% reduction in hallucination rate."
result = verify_citation(claim, passage)
# → {"label": "ENTAILMENT", "confidence": 0.94, "supported": True}
```

**Method 2: LLM-as-judge verification**

```python
VERIFY_PROMPT = """Does the following passage support the claim?
Answer with 'SUPPORTED', 'NOT_SUPPORTED', or 'PARTIALLY_SUPPORTED'.

Claim: {claim}

Passage: {passage}

Verdict:"""

def llm_verify_citation(claim: str, passage: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",   # cheap model for binary verification
        max_tokens=10,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(
            claim=claim, passage=passage
        )}]
    )
    return resp.content[0].text.strip()
```

**Full verification pipeline:**

```python
def verifiable_rag(query: str, passages: list[dict]) -> dict:
    # Step 1: Generate answer with citations
    raw = generate_structured_citations(query, passages)
    
    # Step 2: Verify each citation
    verified_claims = []
    for item in raw["claims"]:
        claim = item["claim"]
        evidence = []
        for pid in item["passage_ids"]:
            passage_text = next(p["text"] for p in passages if p["id"] == pid)
            verdict = verify_citation(claim, passage_text)
            evidence.append(verdict)
        
        # Claim is supported if at least one cited passage entails it
        supported = any(e["supported"] for e in evidence)
        verified_claims.append({
            "claim": claim,
            "supported": supported,
            "evidence": evidence,
        })
    
    return {
        "answer": raw["answer"],
        "verified_claims": verified_claims,
        "unsupported_claims": [c for c in verified_claims if not c["supported"]],
    }
```

**What to do with unsupported claims:**

1. **Remove them:** Regenerate the answer without the unsupported claims
2. **Flag them:** Show the answer with a warning on flagged claims
3. **Retrieve more:** Trigger additional retrieval to find supporting evidence
4. **Abstain:** If the key claim cannot be verified, don't answer

</details>

---

## Q4. How do you evaluate a Citation RAG system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Citation RAG requires metrics beyond standard RAG evaluation because it has an additional attribution quality dimension.

**Metric 1: Citation Precision** — what fraction of generated citations actually support their paired claim

```python
def citation_precision(verified_claims: list[dict]) -> float:
    if not verified_claims:
        return 0.0
    supported = sum(1 for c in verified_claims if c["supported"])
    return supported / len(verified_claims)
```

**Metric 2: Citation Recall** — what fraction of verifiable claims in the answer have at least one citation

```python
def citation_recall(claims: list[dict]) -> float:
    """claims: list with 'has_citation' and 'is_factual' flags."""
    factual_claims = [c for c in claims if c["is_factual"]]
    if not factual_claims:
        return 1.0
    cited = sum(1 for c in factual_claims if c.get("has_citation", False))
    return cited / len(factual_claims)
```

**Metric 3: Attribution F1** — harmonic mean of citation precision and recall

**Metric 4: Claim Faithfulness** — across all supported claims, does the answer accurately reflect what the passage says (no distortion)?

```python
FAITHFULNESS_PROMPT = """On a scale of 1-5, how faithfully does the claim
represent the meaning of the passage? 5 = exact paraphrase, 1 = distortion.

Claim: {claim}
Passage: {passage}
Score (1-5):"""
```

**ALCE benchmark** (Gao et al., 2023) provides:
- Automatic citation evaluation using NLI-based attribution scoring
- Human-annotated citation quality labels for calibration
- Three sub-tasks: ASQA (open-domain), QAMPARI (multi-answer), ELI5 (explanations)

**Production monitoring:**

```python
# Track citation quality per query in production
metrics = {
    "citation_precision": compute_citation_precision(response),
    "unsupported_claim_rate": len(response["unsupported_claims"]) / len(response["verified_claims"]),
    "citation_coverage": len(response["claims_with_citations"]) / len(response["all_claims"]),
}
```

</details>

---

## Q5. What is the difference between attribution and hallucination detection in RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

These are related but distinct problems:

```
Hallucination detection: "Did the LLM invent a fact not present in any source?"
Attribution verification: "Does this specific citation support this specific claim?"

Relationship:
  ─ A claim can be hallucinated AND incorrectly cited  (worst case)
  ─ A claim can be real but incorrectly cited          (citation error, not hallucination)
  ─ A claim can be correctly cited but misrepresented  (faithfulness error)
  ─ A claim can be correct and correctly cited         (ideal case)
```

**Hallucination detection approach:**

Check whether any retrieved passage supports the claim — regardless of which passage the LLM cited.

```python
def detect_hallucination(claim: str, all_passages: list[str]) -> bool:
    """A claim is hallucinated if NO passage in the retrieved set supports it."""
    for passage in all_passages:
        result = verify_citation(claim, passage)
        if result["supported"]:
            return False   # at least one passage supports it → not hallucinated
    return True   # no passage supports it → hallucinated
```

**Attribution verification approach:**

Check whether the *specific cited* passage supports the claim.

```python
def check_attribution(claim: str, cited_passage: str) -> bool:
    """Attribution fails if the cited passage doesn't support the claim,
    even if another passage would."""
    return verify_citation(claim, cited_passage)["supported"]
```

**Why both matter:**
- A hallucination detector that passes a correctly-attributed claim but misses a hallucinated uncited claim will underreport hallucinations
- An attribution checker that only checks cited passages won't catch hallucinated claims that happen to have no citation at all

In production Verifiable RAG, run both: (1) verify all citations, (2) check all uncited factual claims against the full retrieved set.

</details>

---

## Real-World Applications

- **Perplexity.ai and Bing Copilot**: Inline citations with hover-to-verify passage display
- **Legal research (Lexis AI, Harvey)**: Every statement in a legal brief must be attributable to a cited case or statute — attribution verification is a compliance requirement
- **Medical literature assistants**: Claims about drug interactions or clinical outcomes must cite specific study passages
- **ALCE benchmark**: Stanford benchmark specifically for evaluating attribution in long-form answers (Gao et al., 2023)
- **Enterprise compliance reporting**: Audit trails require traceability from every output claim back to a source document
