# Table-Aware RAG (TAT-QA / OmniTab)

> Specialized retrieval and reading over semi-structured tables — row/column-aware chunking, hybrid text-table retrieval, and answer extraction from numerical data.

---

## What is Table-Aware RAG?

Standard RAG treats every chunk as a bag of words. Tables are fundamentally different: they encode structured relationships between rows, columns, and cell values that prose-oriented embedding models don't understand well. Table-Aware RAG uses specialized techniques at every stage — parsing, chunking, embedding, and answer extraction — to handle tabular data correctly.

Distinct from Structured RAG (architecture #12), which routes queries to a relational database via SQL: Table-Aware RAG handles tables *embedded in documents* (PDFs, HTML reports, spreadsheets) where writing SQL queries is not feasible.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Document (PDF / HTML / spreadsheet)
        │
        ▼
  Table Extractor  (detects & isolates tables from surrounding text)
        │
        ├───────────────────────────────┐
        ▼                               ▼
  Row/Column Linearizer          Text-to-SQL Router
  (Markdown or per-row chunks)   (for queryable structured stores)
        │                               │
        ▼                               │
  Table-aware Embedder                  │
        │                               │
        ▼                               │
  Hybrid Retriever (table + text) ◄─────┘
        │
        ▼
  Generator (reasons over table structure, shows arithmetic steps)
```

### Key Components

| Component | Responsibility |
|---|---|
| Table Extractor | Detects and isolates tables from surrounding prose in PDFs/HTML (pdfplumber, BeautifulSoup) |
| Linearizer / SQL Router | Converts a table into retrievable text units (full Markdown or per-row chunks), or routes structured queries to SQL when a live table/database is available |
| Table-aware Embedder | Produces embeddings that capture row/column structure rather than flattened tokens |
| Hybrid Retriever | Merges table-chunk and text-chunk results, boosting table results for numerical/comparison queries |
| Generator | Consumes the retrieved table + text context and performs or shows arithmetic explicitly |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Table extraction | Camelot, pdfplumber, unstructured.io, Azure Document Intelligence |
| Table-aware encoders | TAPAS, OmniTab, TAT-QA-style hybrid text+table encoders |
| Structured-query path | LangChain SQL Agent, direct Text-to-SQL over the source table/database |
| Retrieval / embedding | sentence-transformers (bge), vector DB with `chunk_type` metadata |

---

## Why Standard RAG Fails on Tables

```
Table from a financial report:
┌─────────────────┬──────────┬──────────┬──────────┐
│ Region          │ Q1 2025  │ Q2 2025  │ Q3 2025  │
├─────────────────┼──────────┼──────────┼──────────┤
│ North America   │ $142M    │ $158M    │ $171M    │
│ Europe          │ $87M     │ $93M     │ $101M    │
│ Asia Pacific    │ $45M     │ $52M     │ $64M     │
└─────────────────┴──────────┴──────────┴──────────┘

Query: "What was Europe's revenue growth from Q1 to Q3?"

Standard RAG failure:
  1. Table serialized as flat text loses row/column structure
  2. Embedding sees "Europe 87M 93M 101M" with no structural context
  3. Retrieval finds the table but LLM can't do the arithmetic from plain text
  4. Answer: wrong or hallucinated

Table-Aware RAG:
  1. Table parsed and linearized as structured Markdown
  2. Each row embedded with column headers prepended
  3. Query retrieves correct rows + headers
  4. LLM given structured table context → computes: (101-87)/87 = +16%
```

---

## Stage 1: Table Extraction and Parsing

```python
import pdfplumber
import pandas as pd

def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """Extract all tables from a PDF with surrounding context."""
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            page_text   = page.extract_text() or ""
            
            for table_raw in page_tables:
                if not table_raw or len(table_raw) < 2:
                    continue
                
                headers = table_raw[0]
                rows    = table_raw[1:]
                df      = pd.DataFrame(rows, columns=headers)
                
                tables.append({
                    "page":     page_num + 1,
                    "headers":  headers,
                    "dataframe": df,
                    "markdown": df.to_markdown(index=False),
                    # Surrounding text as context for what the table is about
                    "context_before": page_text[:500],
                })
    
    return tables
```

### HTML Table Extraction

```python
from bs4 import BeautifulSoup
import pandas as pd

def extract_tables_from_html(html: str) -> list[dict]:
    soup   = BeautifulSoup(html, "html.parser")
    tables = []
    
    for table in soup.find_all("table"):
        # Get caption / preceding heading as context
        caption = table.find("caption")
        context = caption.get_text() if caption else ""
        
        # Find preceding h2/h3 heading
        for sibling in table.previous_siblings:
            if sibling.name in ("h1", "h2", "h3", "h4"):
                context = sibling.get_text() + " " + context
                break
        
        try:
            df = pd.read_html(str(table))[0]
            tables.append({
                "context": context.strip(),
                "markdown": df.to_markdown(index=False),
                "dataframe": df,
            })
        except Exception:
            continue
    
    return tables
```

---

## Stage 2: Table Linearization and Chunking

### Full Table Linearization (Markdown)

For small tables (< 20 rows), serialize the entire table as Markdown with a title prefix:

```python
def linearize_table(table: dict) -> str:
    """Convert table to a retrievable text chunk."""
    title   = table.get("context", "Table")
    md      = table["markdown"]
    return f"Table: {title}\n\n{md}"
```

### Row-Level Chunking (Large Tables)

For large tables (100+ rows), embed each row as a separate chunk with headers prepended:

```python
def chunk_table_by_row(table: dict) -> list[str]:
    """One chunk per row, each with column headers for context."""
    df      = table["dataframe"]
    context = table.get("context", "")
    chunks  = []
    
    for _, row in df.iterrows():
        # Format: "Region: North America | Q1 2025: $142M | Q2 2025: $158M | Q3 2025: $171M"
        row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
        chunks.append(f"[{context}] {row_text}")
    
    return chunks
```

### Hybrid: Table Summary + Row Chunks

```python
def chunk_table_hybrid(table: dict, summary_fn) -> list[str]:
    """Summary chunk for retrieval + row chunks for precise answer extraction."""
    summary = summary_fn(table["markdown"])  # LLM-generated summary of what table contains
    rows    = chunk_table_by_row(table)
    # Return summary first (good for broad queries) + row chunks (good for specific lookups)
    return [f"[TABLE SUMMARY] {summary}"] + rows
```

---

## Stage 3: Embedding Tables

Standard sentence embedding models underperform on tabular text. Options:

```python
from sentence_transformers import SentenceTransformer

# Option A: General-purpose model (baseline)
TEXT_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Option B: Table-aware model (better for numerical data)
# OmniTab and TAPAS produce table-specific representations
# For production: fine-tune bge on (query, table_row) pairs from your domain

def embed_table_chunk(chunk: str, model=TEXT_MODEL) -> list[float]:
    return model.encode(chunk, normalize_embeddings=True).tolist()
```

**Metadata to store with table vectors:**
```python
{
    "id":          "doc:annual-report-2025:table:3:row:7",
    "vector":      [...],
    "metadata": {
        "doc_id":     "annual-report-2025",
        "table_idx":  3,
        "row_idx":    7,
        "chunk_type": "table_row",  # vs. "text", "table_summary"
        "context":    "Q3 2025 Revenue by Region",
    }
}
```

---

## Stage 4: Hybrid Table + Text Retrieval

At query time, retrieve from both text chunks and table chunks, then merge:

```python
def table_aware_retrieve(query: str, vector_db, k: int = 5) -> list[dict]:
    query_emb = embed_table_chunk(query)
    
    # Retrieve from all chunk types; filter tables separately if needed
    results = vector_db.query(
        vector=query_emb,
        top_k=k * 2,
        include_metadata=True,
    )
    
    # Separate text and table results for different weighting
    text_results  = [r for r in results if r["metadata"]["chunk_type"] == "text"]
    table_results = [r for r in results if r["metadata"]["chunk_type"].startswith("table")]
    
    # For numerical / comparison queries, boost table results
    if is_numerical_query(query):
        # Promote table results: give them rank boost in RRF
        merged = rrf_merge(table_results[:k], text_results[:k], table_boost=1.5)
    else:
        merged = rrf_merge(text_results, table_results)
    
    return merged[:k]


def is_numerical_query(query: str) -> bool:
    numerical_signals = ["how much", "percentage", "growth", "compare", "highest", "lowest",
                         "average", "total", "revenue", "cost", "increase", "decrease"]
    q_lower = query.lower()
    return any(s in q_lower for s in numerical_signals)
```

---

## Stage 5: Generating Answers from Tables

When context includes table chunks, structure the prompt to help the LLM reason arithmetically:

```python
import anthropic

client = anthropic.Anthropic()

TABLE_SYSTEM_PROMPT = """You are answering questions that may require reading tables and doing arithmetic.
When tables are provided:
1. Identify the relevant rows and columns
2. Show the numbers you are using
3. Show any arithmetic steps explicitly
4. State units clearly (%, $M, etc.)"""

def generate_table_aware_answer(query: str, retrieved_chunks: list[dict]) -> str:
    # Separate table and text context
    table_context = "\n\n".join(
        c["metadata"]["text"] for c in retrieved_chunks
        if c["metadata"]["chunk_type"].startswith("table")
    )
    text_context = "\n\n".join(
        c["metadata"]["text"] for c in retrieved_chunks
        if c["metadata"]["chunk_type"] == "text"
    )
    
    user_content = f"""Question: {query}

{"Table Data:\n" + table_context if table_context else ""}
{"Background Text:\n" + text_context if text_context else ""}

Please answer the question using the data above."""
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        system=TABLE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text
```

---

## TAT-QA and OmniTab Architectures

### TAT-QA (Table-and-Text Question Answering)

TAT-QA (Zhu et al., 2021) is a benchmark and model architecture for questions that require *both* table and text evidence. Its key insight: some questions require reasoning across text paragraphs *and* table cells simultaneously.

```
Query: "What was the largest revenue increase between Q1 and Q3 for any region?"

Evidence needed:
  - Table: all Q1 and Q3 values
  - Text: "Revenue growth is measured at constant currency"

TAT-QA model components:
  1. Hybrid encoder: RoBERTa on text + TAPAS on table
  2. Reasoning type classifier: extractive / arithmetic / counting
  3. Span extractor (extractive) OR arithmetic program generator
```

### OmniTab

OmniTab (Jiang et al., 2022) pre-trains on (natural text, table) pairs scraped from Wikipedia to create a joint table-text understanding model. Key differences from TAPAS:

| | TAPAS | OmniTab |
|---|---|---|
| Training | Single-table supervision | Large-scale table-text pairs |
| Scope | Classification / aggregation | Generative QA |
| Use in RAG | Table retrieval encoding | Full QA with table context |

---

## Table-Aware RAG vs. Structured RAG

| Dimension | Table-Aware RAG (#36) | Structured RAG (#12) |
|-----------|----------------------|---------------------|
| **Data source** | Tables embedded in documents (PDF, HTML) | Relational databases |
| **Query interface** | Natural language → table retrieval → LLM | Natural language → SQL → DB |
| **When applicable** | Unstructured reports, exported spreadsheets | Queryable live databases |
| **Arithmetic accuracy** | Moderate (LLM-based, verify) | High (SQL is exact) |
| **Schema required?** | No | Yes |

---

## Key Takeaways

1. **Linearize tables as Markdown** for the LLM context — it preserves structure better than plain text.
2. **Row-level chunking with headers** is essential for large tables — the LLM needs column context for every row.
3. **Store `chunk_type` metadata** to distinguish table rows from text — enables modality-boosted retrieval.
4. **Detect numerical queries** and boost table recall — standard semantic similarity underweights tables for number questions.
5. **Show arithmetic steps in the prompt** — instruct the LLM explicitly; do not let it silently compute.

---

## Interview Q&A

**Q: Why does standard RAG underperform on tables, and how does table-aware RAG fix it?**

Standard RAG serializes tables as flat text, losing structural relationships. "North America 142M Q1" without column headers is ambiguous — the embedding model sees unrelated tokens. At generation, the LLM cannot reliably do arithmetic over improperly structured numbers embedded in prose. Table-aware RAG fixes this at three levels: (1) parsing: extract tables as DataFrames to preserve row/column structure; (2) chunking: embed each row with its column headers prepended ("Region: North America | Q1 2025: $142M") so similarity search is structure-aware; (3) generation: pass the full table as Markdown to the LLM with an explicit arithmetic-reasoning prompt, not as a flattened string.

---

**Q: How would you handle a 500-row financial table in a RAG system?**

Full-table linearization is infeasible — 500 rows of Markdown would overflow the context window and dilute the relevant signal. Instead: (1) embed each row as a separate chunk with column headers as prefix (500 chunks per table); (2) embed a table-level summary ("Q3 2025 revenue by region and product line, 500 rows, covering Jan–Sep 2025") as an additional chunk; (3) at query time, retrieve the most semantically similar rows plus the summary chunk; (4) reconstruct a mini-table from the top-k rows and pass that to the LLM. For aggregation queries ("total Q3 revenue"), retrieve all rows matching the filter and compute the aggregation in Python before generating the answer — don't ask the LLM to sum 500 numbers.

---

**Q: What is the difference between TAPAS and a standard retriever for table QA?**

A standard dense retriever (BGE, E5) is trained on text-text pairs — it produces a single vector for a table row that captures token-level semantics but not tabular structure (row identity, column type, aggregation scope). TAPAS is a BERT-variant fine-tuned specifically on (natural language question, table) pairs with a cell-level annotation objective: it learns to select which cells are relevant and what aggregation operation (SUM, COUNT, AVERAGE) to apply. In a Table-Aware RAG pipeline, TAPAS is most useful as a reader (answer extraction from a retrieved table) rather than a retriever (finding the right table), because it requires the full table as input and doesn't produce a retrieval vector. The practical RAG design: use dense retrieval to find the right table chunks, then pass them to a TAPAS-style or LLM-based reader for answer extraction.
