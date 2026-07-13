# SURGE: Schema-Grounded RAG (Structured Output with Grounding)

> A RAG architecture that generates structured, schema-conformant output (JSON, tables, structured reports) where each field is explicitly grounded — traceable to a specific retrieved passage — and the grounding is verified before delivery.

---

## Definition

**SURGE** (Structured and Grounded generation) extends standard RAG to domains where the output must be machine-readable and auditable: compliance reports, financial data extraction, medical summarization, API responses, and database population. The two extensions over standard RAG:

1. **Schema-constrained generation** — output is forced to conform to a predefined schema (no free-form text that might miss required fields or hallucinate structure).
2. **Per-field grounding** — each field value is linked to the specific passage that supports it; a validation pass confirms the field is entailed by the cited passage before the response is returned.

Without grounding: a RAG system might correctly format a JSON response but populate `annual_revenue: "$4.2B"` from a hypothetical company merged in training data rather than from the retrieved document.

---

## Architecture

```
Query + Schema
      │
      ▼
  Retrieval (dense/hybrid)
      │
      ▼
  [doc_1, doc_2, ..., doc_k]
      │
      ▼
  Structured Generation (LLM)
  ┌─────────────────────────────────────┐
  │ "Extract the following fields.      │
  │  For each field, cite the passage   │
  │  verbatim. Leave null if not found."│
  │                                     │
  │ Output: {                           │
  │   "revenue": {                      │
  │     "value": "$4.2B",               │
  │     "source_passage": "...",        │
  │     "doc_id": "annual_report_2023", │
  │     "confidence": "high"            │
  │   }, ...                            │
  │ }                                   │
  └─────────────────────────────────────┘
      │
      ▼
  Grounding Validation (NLI)
  For each field: does source_passage ENTAIL value?
      │
      ├─ ENTAILMENT → keep field
      ├─ NEUTRAL    → flag as low-confidence
      └─ CONTRADICTION → null field, log conflict
      │
      ▼
  Validated Structured Output
```

### Key Components

| Component | Responsibility |
|---|---|
| **Schema Definer** | Specifies the target JSON schema (fields, types, nullability) the output must conform to |
| **Retriever** | Fetches candidate passages (dense/hybrid) relevant to the query/schema fields |
| **Constrained Extractor** | Uses tool-use/function-calling to populate schema fields only from retrieved passages |
| **Per-field NLI Grounding Validator** | Checks whether each cited passage entails its extracted field value |
| **Structured Output Assembler** | Nulls out unentailed/contradicted fields and assembles the final validated object |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| **Schema-constrained generation** | OpenAI/Claude `tool_use`/function-calling with JSON schema |
| **Schema validation** | Pydantic |
| **Grounding validation** | NLI models (e.g. `bart-large-mnli`, `cross-encoder/nli-deberta-v3-base`) |

---

## Schema-Constrained Generation

Use Anthropic's `tool_use` to force schema-conformant output. The model cannot produce malformed JSON because the tool call validates at the API layer:

```python
import json
import anthropic
from typing import Optional

client = anthropic.Anthropic()

EXTRACTION_TOOL = {
    "name": "extract_company_profile",
    "description": "Extract structured company data from the provided passages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "company_name": {
                "type": "string",
                "description": "Legal company name"
            },
            "founded_year": {
                "type": ["integer", "null"],
                "description": "Year founded; null if not found in passages"
            },
            "annual_revenue": {
                "type": ["string", "null"],
                "description": "Most recent annual revenue with unit (e.g. '$4.2B')"
            },
            "headquarters": {
                "type": ["string", "null"],
                "description": "City, Country"
            },
            "employee_count": {
                "type": ["string", "null"],
                "description": "Approximate headcount (e.g. '~12,000')"
            },
            "field_sources": {
                "type": "object",
                "description": "For each non-null field, the verbatim passage it was extracted from",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["company_name", "field_sources"]
    }
}


def structured_extract(query: str, passages: list[dict]) -> dict:
    """
    passages: list of {"id": str, "text": str}
    Returns validated structured extraction.
    """
    # Build context with passage IDs so the model can cite them
    context = "\n\n".join(
        f"[{p['id']}] {p['text']}" for p in passages
    )

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_company_profile"},
        system=(
            "Extract the requested fields from the provided passages. "
            "Only extract information explicitly stated in the passages — "
            "never infer or hallucinate values. "
            "Set a field to null if the information is absent."
        ),
        messages=[{
            "role": "user",
            "content": f"Passages:\n{context}\n\nQuery: {query}"
        }]
    )

    tool_call = next(
        b for b in response.content if b.type == "tool_use"
    )
    return tool_call.input
```

---

## Per-Field Grounding Validation

After schema-constrained extraction, validate that each cited passage actually supports its claimed value:

```python
from sentence_transformers import CrossEncoder

# NLI cross-encoder trained on SNLI/MNLI
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

LABELS = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]


def validate_grounding(extracted: dict, passages: dict[str, str]) -> dict:
    """
    extracted: output of structured_extract (tool_call.input)
    passages: {passage_id: text} lookup
    Returns extracted with grounding verdicts added.
    """
    validated = dict(extracted)
    grounding_verdicts = {}

    field_sources = extracted.get("field_sources", {})

    for field, source_text in field_sources.items():
        field_value = extracted.get(field)
        if not field_value:
            continue

        # Premise: the retrieved passage.  Hypothesis: the extracted field value.
        premise   = source_text
        hypothesis = f"The {field} is {field_value}."

        scores = nli_model.predict([(premise, hypothesis)])
        label  = LABELS[scores.argmax()]
        conf   = float(scores.max())

        grounding_verdicts[field] = {
            "label":      label,
            "confidence": round(conf, 3),
        }

        # Null out contradicted or low-confidence fields
        if label == "CONTRADICTION":
            validated[field] = None
            grounding_verdicts[field]["action"] = "nulled (contradiction)"
        elif label == "NEUTRAL" and conf < 0.6:
            grounding_verdicts[field]["action"] = "flagged (low confidence)"

    validated["_grounding_verdicts"] = grounding_verdicts
    return validated


def surge_pipeline(query: str, retrieval_fn, k: int = 5) -> dict:
    """Full SURGE pipeline: retrieve → extract → validate → return."""
    passages = retrieval_fn(query, k=k)  # [{id, text}, ...]
    extracted = structured_extract(query, passages)
    validated = validate_grounding(extracted, {p["id"]: p["text"] for p in passages})
    return validated
```

---

## Multi-Document Structured Synthesis

For reports that aggregate across many documents (e.g., a competitive landscape table):

```python
from dataclasses import dataclass

@dataclass
class CompetitorProfile:
    name: str
    revenue: str | None
    employees: str | None
    key_product: str | None
    source_doc: str

def build_competitive_table(
    competitors: list[str],
    retrieval_fn,
) -> list[CompetitorProfile]:
    """
    Fan-out: one retrieval + extraction per competitor.
    Fan-in: merge into a comparative table.
    """
    import asyncio

    async def extract_one(name: str) -> CompetitorProfile:
        passages = retrieval_fn(f"{name} company profile revenue employees", k=5)
        extracted = structured_extract(f"Extract profile for {name}", passages)
        validated = validate_grounding(extracted, {p["id"]: p["text"] for p in passages})
        return CompetitorProfile(
            name=name,
            revenue=validated.get("annual_revenue"),
            employees=validated.get("employee_count"),
            key_product=validated.get("key_product"),
            source_doc=passages[0]["id"] if passages else "unknown",
        )

    return asyncio.run(
        asyncio.gather(*[extract_one(c) for c in competitors])
    )
```

---

## Comparison: SURGE vs Related Architectures

| Architecture | Output Format | Grounding | Schema Enforced? | Validation |
|---|---|---|---|---|
| **Standard RAG** | Free-form prose | Implicit | No | None |
| **Verifiable/Citation RAG (#33)** | Prose with inline citations | Explicit, passage-level | No | NLI on claims |
| **Structured RAG (#12)** | SQL query results | None (SQL is ground truth) | Database schema | Schema validation only |
| **SURGE (#40)** | Schema-conformant JSON/table | Explicit, field-level | Yes (tool_use / JSON mode) | NLI per field |
| **Table-Aware RAG (#36)** | Prose about tables | None explicit | No | None |

The distinguishing property of SURGE: **every output field has a cited passage, and that citation is NLI-verified**. Standard citation RAG verifies claims in prose; SURGE verifies individual fields in a structured object.

---

## When to Use SURGE

| Use Case | Why SURGE? |
|---|---|
| Contract clause extraction | Each clause value must be traceable to exact contract language |
| Financial data ETL (PDF → database) | Revenue/EBITDA values must be auditable |
| Medical record summarization | ICD codes, dosages must cite source notes |
| Regulatory compliance reports | Auditors need passage-level traceability |
| Competitive intelligence tables | Each cell must be sourced; null > hallucinated |

**Do not use** when: output is inherently conversational/advisory (a recommendation letter doesn't need field-level grounding), or when the schema is unknown at query time.

---

## Failure Mode: Schema Over-Fit

The most common SURGE failure: designing a schema that forces the model to populate fields that are absent in the corpus, leading to hallucinated low-confidence values. Mitigations:

1. **Make every field nullable** — `"type": ["string", "null"]`. Never require a field the corpus might not contain.
2. **Add a `found_in_passages: bool` field** — explicit model self-report.
3. **NLI threshold as a null gate** — any field with NEUTRAL NLI confidence < 0.6 is set to null rather than returned.
4. **Log null rates** — if `employee_count` is null on 80% of queries, either expand retrieval or remove the field.

---

## Key Takeaways

1. **Schema-constrained generation eliminates structural hallucination** — the output can't hallucinate extra fields or malformed values when using `tool_use`/JSON mode.
2. **NLI validation closes the grounding loop** — extracted values are verified against cited passages before delivery.
3. **Nulls are better than guesses** — in structured extraction, a null field is an honest signal; a plausible-but-wrong value is a data quality bug.
4. **SURGE is the right pattern for ETL, compliance, and analytics** — anywhere downstream systems ingest RAG output programmatically.
5. **Distinction from Verifiable/Citation RAG (#33)**: citation RAG verifies prose claims; SURGE verifies field values in a typed schema.

---

## Interview Q&A

**Q: How does SURGE differ from standard RAG with a JSON output prompt?**

A standard RAG system asked to "output JSON" can still produce structurally valid but semantically hallucinated values — it has no mechanism to distinguish "I found this in the passage" from "I'm generating a plausible-sounding value." SURGE adds two layers: (1) schema enforcement via `tool_use` (not just a prompt instruction) — the API layer validates the schema before the response is returned; (2) per-field grounding verification — an NLI model checks that the cited passage actually entails the claimed value. The NLI step converts "grounding" from a model aspiration to a programmatic gate. A field that the NLI model rates as NEUTRAL or CONTRADICTION is nulled out, making the system fail-closed rather than fail-hallucinating.

---

**Q: What NLI model would you use for grounding validation in production, and what threshold would you set?**

`cross-encoder/nli-deberta-v3-base` (SentenceTransformers) is the standard choice — small enough to run inline (~180M params, ~5ms/pair on CPU), trained on NLI tasks, and well-calibrated. For production: ENTAILMENT confidence ≥ 0.7 to keep a field, ENTAILMENT confidence 0.5–0.7 to flag it as low-confidence, below 0.5 or CONTRADICTION to null it. Calibrate thresholds on a labeled validation set: randomly sample 200 extraction results, manually verify which fields are correct, and find the threshold that maximizes F1 between "keep" decisions and actual correctness. At 0.7 a typical system achieves ~95% precision at the cost of ~20% recall (nulling some correct fields). In regulated domains (medical, legal), raise precision by increasing the threshold to 0.8.

---

**Q: How do you handle a field that is not mentioned anywhere in the retrieved documents?**

The schema must make the field nullable, and the system prompt must explicitly instruct the model to return null rather than infer or hallucinate. In code: `"type": ["string", "null"]` in the tool schema. The NLI validation step provides a second safety net: if the model returns a non-null value but `field_sources` is empty or the NLI scores NEUTRAL with low confidence, null the field programmatically. Log the null rate per field — a field that is null on >50% of queries either needs better retrieval (add query variations for that field), a schema redesign (remove or rephrase the field), or indicates the corpus genuinely lacks that information.
