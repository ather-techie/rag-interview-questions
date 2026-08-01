# Document Ingestion & Parsing: Getting Knowledge Into Your Pipeline

> Most RAG failures happen before the LLM. Bad ingestion means bad retrieval — no amount of clever reranking fixes garbage input.

---

## What is Document Ingestion?

Document ingestion is the pipeline step that takes raw source files — PDFs, HTML pages, Word docs, scanned images — and converts them into clean, structured text that a RAG system can actually chunk and embed. It handles parsing file formats, extracting text from layouts like tables and multi-column pages, and normalizing the result into a consistent representation. Because every downstream step depends on this text being accurate, ingestion quality puts a ceiling on how good retrieval can ever be.

---

## Why Ingestion Is the Foundation

The retrieval and generation layers are only as good as what's been indexed. Ingestion covers the full pipeline from raw data sources to clean, structured text ready for chunking:

```
Raw Sources
  │
  ▼
Extraction ── (PDF parsers, web scrapers, API connectors, DB queries)
  │
  ▼
Cleaning ──── (deduplication, encoding fixes, HTML stripping, PII removal)
  │
  ▼
Metadata ──── (source, page, section, document type, date)
  │
  ▼
Structured Text ── ready for Chunking → Embedding → Vector DB
```

A broken step anywhere in this chain silently degrades every downstream retrieval.

---

## Data Sources and Extraction Methods

| Source | Extraction Method | Python Tools |
|--------|------------------|--------------|
| PDF | PDF parsers / OCR | `pypdf`, `pdfplumber`, `pymupdf` |
| Scanned PDF / Image | OCR engine | `pytesseract`, `easyocr`, `paddleocr` |
| Word doc (.docx) | XML-based document parsing | `python-docx`, `docx2txt` |
| Website / HTML | Web scraping | `beautifulsoup4`, `scrapy`, `playwright` |
| Database | SQL queries | `SQLAlchemy`, direct DB drivers |
| REST API | HTTP requests | `requests`, `httpx` |
| Notion | Official SDK | `notion-client` |
| CSV / Excel | DataFrame parsing | `pandas` |
| Markdown / Code | Direct read | `pathlib` + text parsing |

**Note on Word docs:** `docx2txt` is the simplest option — it returns body, table, and header/footer text in one call, but discards structure. `python-docx` gives structured access (iterate `document.paragraphs`, `document.tables`, and each section's `header`/`footer`), which is needed if you want to preserve table structure or separate body text from boilerplate headers/footers. Neither library renders tracked changes; a `.docx` with unaccepted revisions will silently include both the original and edited text unless you first flatten revisions (e.g., via `docx.settings` or a conversion step in Word/LibreOffice).

---

## PDF Parser Comparison

The three most common PDF libraries serve different use cases:

| Library | PyPI Name | Best For | Speed | Dependency |
|---------|-----------|----------|-------|------------|
| pypdf | `pypdf` | Splits, merges, crops PDFs; basic text extraction; password-protected PDFs | Fast | Pure Python (no C) |
| pdfplumber | `pdfplumber` | Table extraction to DataFrames; coordinate-level text positioning | Medium | Built on `pdfminer.six` |
| PyMuPDF | `pymupdf` | Fastest full-document text + image extraction; layout preservation | Fastest | MuPDF C library |

```
Throughput benchmark (approximate):
pypdf      ──  fast, ~0.024s/page
pdfplumber ──  ~0.10s/page
pymupdf    ──  ~0.003–0.01s/page (fastest)

Rule of thumb:
  High-volume pipeline (>10k docs/hr)  → pymupdf
  Table-heavy documents                → pdfplumber
  Pure Python / no C deps              → pypdf
  Scanned / image PDFs                 → OCR (pytesseract / easyocr)
```

**Note on PyMuPDF imports:** `import fitz` is the legacy module name; `import pymupdf` works in version ≥ 1.24.0. Both resolve to the same package.

---

## Text Cleaning Techniques

Raw extracted text always requires cleaning before it's useful:

| Problem | Technique | Tool |
|---------|-----------|------|
| Encoding mojibake (â€™ instead of ') | Unicode fix | `ftfy` |
| HTML tags left in text | Tag stripping | `BeautifulSoup.get_text()` |
| Duplicate documents | Content hash | `hashlib` SHA-256 |
| PII (emails, SSNs, phone numbers) | Entity detection + redaction | `presidio-analyzer` + `presidio-anonymizer` |
| Header / footer noise | Heuristic filtering (repeated short lines) | Custom regex |
| Excessive whitespace | Normalize | `re.sub(r'\s+', ' ', text)` |
| Language detection | Per-document classification | `langdetect` (55 languages) |

**Key principle:** Preserve `source`, `page_number`, and `section` as metadata during cleaning — do not discard them. These fields become pre-filters during retrieval.

---

## OCR Pipeline for Scanned Documents

When documents are scanned images, text must be extracted via OCR before any parsing:

```
Scanned PDF / Image
  │
  ▼
Image Preprocessing
  ├─ Deskew (correct tilt)
  ├─ Binarize (convert to black/white)
  └─ Denoise
  │
  ▼
OCR Engine
  ├─ pytesseract ── best for high-res typed text; needs Tesseract binary installed
  ├─ easyocr ────── DL-based; 80+ languages; minimal setup; balanced accuracy/speed
  └─ paddleocr ──── best for structured docs (tables, forms); superior bounding-box accuracy
  │
  ▼
Confidence Filtering
  └─ Discard or flag pages below threshold (e.g., pytesseract conf < 60)
  │
  ▼
Clean Text
```

---

## VLM-Based Parsing and the Modern Document-AI Stack

Traditional OCR + layout-detection pipelines (above) treat text extraction and layout understanding as separate, hand-tuned steps. An increasingly common alternative in production RAG ingestion: feed a rendered page image directly to a vision-language model (VLM) and let it read the page the way a person would.

**VLM-based parsing**

Instead of bounding-box detection → OCR per region → table reconstruction, a VLM (e.g., GPT-4o, Claude, Gemini, or open-weight models like Qwen2-VL / Qwen2.5-VL) takes a page image plus a prompt and returns structured output — usually Markdown that preserves headings, tables, and reading order — in a single pass.

```
Traditional OCR pipeline:
  Page Image → Layout Detection → OCR per region → Table Reconstruction → Text
  (several specialized models chained together, each a separate failure point)

VLM pipeline:
  Page Image → VLM (single pass) → Structured Markdown / JSON
  (one model call; layout and tables handled via natural-language reasoning, not bounding boxes)
```

- **Strengths:** noticeably more robust than classical OCR on complex multi-column layouts, merged/nested tables, handwriting, and low-quality scans, because the model reasons about structure semantically rather than geometrically. Can also describe charts/figures in text, which OCR cannot.
- **Trade-offs:** higher per-page cost and latency than `pytesseract`/`easyocr`; outputs can hallucinate values not actually on the page, so numeric-heavy extractions (financial tables) need spot-checking; less deterministic than rule-based OCR; born-digital PDFs must still be rasterized to an image first, an unnecessary step if a text layer already exists.
- **When to reach for it:** scanned documents with dense tables, handwriting, checkboxes/forms, or irregular layouts where classical OCR keeps failing. For clean, born-digital PDFs, a text-layer extractor (`pypdf`/`pymupdf`) remains cheaper and more reliable.

**Modern document-AI stack**

Beyond calling a VLM directly, several tools now package layout detection, OCR, table extraction, and (increasingly) VLM options behind one API, reducing the custom `pypdf`/`pdfplumber` glue code a team has to maintain:

| Tool | Type | Best For | vs. DIY pypdf/pdfplumber pipeline |
|------|------|----------|-----------------------------------|
| `unstructured` (unstructured.io) | Open-source Python library | Partitioning many formats (PDF, DOCX, PPTX, HTML, images, email) into typed elements (`Title`, `NarrativeText`, `Table`) via `fast`/`hi_res`/`ocr_only` strategies | One consistent API across many formats instead of a different library per format; heavier dependency footprint and slower on the `hi_res` path |
| `LlamaParse` (LlamaIndex) | Hosted parsing API | Complex PDFs with embedded tables and charts; clean Markdown/JSON output; offers an agentic multimodal parsing mode for hard documents | Handles layout and table reconstruction as a service, at usage-based cost, instead of hand-tuned local extraction code |
| `Docling` (IBM) | Open-source toolkit | Local/offline conversion of PDF, DOCX, PPTX, XLSX, HTML, and images to Markdown/JSON, using purpose-built layout (DocLayNet) and table-structure (TableFormer) models | Preserves reading order and table structure better than a plain text-layer extraction, runs entirely on a laptop with no API key, but adds a heavier ML dependency stack than `pypdf` |
| Azure AI Document Intelligence | Managed cloud API | Enterprise OCR + layout extraction (text, tables, key-value pairs, selection marks), with prebuilt models for invoices, receipts, IDs | Fully managed and scalable (batch analysis, SLA) at the cost of per-page pricing and a cloud dependency; less customizable than an OSS pipeline |

**Decision guide:**
```
Clean, born-digital PDF, no tables        → pypdf / pymupdf (cheapest, fastest)
Table-heavy digital PDF                   → pdfplumber, or unstructured/Docling if many formats are involved
Scanned, handwritten, or irregular layout → VLM-based parsing or LlamaParse
Need one library across many file types   → unstructured or Docling
Enterprise/regulated, need a managed SLA  → Azure Document Intelligence
```

---

## Incremental Ingestion: Handling Updates Without Full Re-Embedding

Re-embedding your entire corpus on every document update is expensive. The standard approach:

```
Incoming Document
  │
  ├─ Compute SHA-256 content hash
  │
  ├─ Lookup hash in fingerprint store
  │     ├─ Match found → skip (unchanged)
  │     └─ No match → proceed
  │
  ├─ Upsert document (update if doc_id exists, insert if new)
  │
  └─ Soft-delete stale versions (mark deleted, apply TTL-based eviction)
```

This pattern is covered in detail in [04-stale_index_problem.md](../03_failure_modes/04-stale_index_problem.md).

---

## Key Takeaways

1. **Ingestion failures are silent.** A corrupted chunk produces a valid embedding and gets indexed — you only discover the failure when retrieval returns wrong answers.
2. **Preserve metadata at extraction time.** `source`, `page`, and `section` fields enable pre-filtering that can double retrieval precision.
3. **Route scanned PDFs to OCR explicitly.** Don't let an empty text extraction pass silently into the pipeline.
4. **Use content hashing for incremental ingestion.** Re-embedding unchanged documents wastes compute and introduces no benefit.
5. **Deduplicate before embedding, not after.** Near-duplicate chunks waste index space and bias retrieval toward over-represented content.

---

## Interview Q&A

---

### Q1. What are the main data sources in a RAG pipeline and how do you extract text from each? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

RAG systems ingest documents from multiple source types, each requiring a different extraction strategy:

| Source | Extraction Method | Tools |
|--------|-----------------|-------|
| **PDF** | PDF parser (text layer) | `pypdf`, `pdfplumber`, `pymupdf` |
| **Scanned PDF / Image** | OCR | `pytesseract`, `easyocr`, `paddleocr` |
| **Website** | Web scraping | `beautifulsoup4`, `scrapy`, `playwright` |
| **Database** | SQL queries | `SQLAlchemy` + DB driver |
| **REST API** | HTTP requests | `requests`, `httpx` |
| **Notion** | Official SDK | `notion-client` |
| **CSV / Excel** | DataFrame read | `pandas` |

The extraction method must match the source: a web scraper on a PDF returns nothing; a PDF parser on a JS-rendered site misses all dynamic content.

```python
# PDF extraction example
import pymupdf  # or: import fitz (legacy alias, same package)

def extract_pdf_text(path: str) -> list[dict]:
    doc = pymupdf.open(path)
    pages = []
    for page_num, page in enumerate(doc):
        pages.append({
            "text": page.get_text(),
            "metadata": {
                "source": path,
                "page": page_num + 1,
                "total_pages": len(doc)
            }
        })
    return pages
```

**Production note:** Always validate that text was actually extracted (len > 0) before proceeding. A blank extraction means the PDF is scanned — route it to the OCR branch.

</details>

---

### Q2. How do you choose between pypdf, pdfplumber, and PyMuPDF for a production RAG pipeline? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The three libraries target different priorities:

| Library | Use When | Avoid When |
|---------|----------|------------|
| `pypdf` | You need pure Python (no compiled deps), basic text, or PDF manipulation (split/merge/crop) | High-volume throughput or table-heavy docs |
| `pdfplumber` | Documents contain tables you need as structured data | Speed is critical (it's the slowest) |
| `pymupdf` | You need maximum throughput and layout-aware extraction | You can't install C dependencies |

**Decision rule:**
```
Does the PDF have tables you must extract as structured data?
  └─ Yes → pdfplumber

Is throughput critical (>10k docs/hour)?
  └─ Yes → pymupdf

Need pure Python (no C deps, e.g., restricted environments)?
  └─ Yes → pypdf

Is the PDF scanned (no text layer)?
  └─ Yes → None of the above; use OCR (pytesseract, easyocr, or paddleocr)
```

**PyMuPDF import note:** Use `import pymupdf` (version ≥ 1.24.0) or `import fitz` (legacy alias). Both refer to the same installed package.

```python
# Fastest path: PyMuPDF
import pymupdf

doc = pymupdf.open("report.pdf")
text = "\n".join(page.get_text() for page in doc)

# Table path: pdfplumber
import pdfplumber

with pdfplumber.open("report.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()  # returns list[list[list[str]]]
```

</details>

---

### Q3. Why is ingestion quality the #1 failure point in RAG systems? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Every downstream component — embeddings, vector search, reranking, LLM generation — operates on the text produced by ingestion. If that text is corrupted, the LLM has no way to recover it.

```
Bad ingestion produces:
  ├─ Corrupt text ────────► Meaningless embeddings → wrong retrieval
  ├─ Missing metadata ────► No pre-filtering → low precision
  ├─ Duplicate content ───► Biased retrieval → same answer retrieved 3x
  └─ Lost table data ─────► LLM answers from missing information
```

**Concrete failure examples:**

| Ingestion Failure | Downstream Effect |
|------------------|-------------------|
| OCR outputs "Rev€nue" for "Revenue" | Semantic search misses all revenue-related queries |
| Header "Page 4 — Company Confidential" not stripped | Every chunk includes irrelevant boilerplate; embeddings polluted |
| Tables converted to garbled linear text | LLM cannot answer numerical questions correctly |
| Duplicate policy docs not deduplicated | RAG retrieves the same paragraph multiple times, wasting context |
| Source metadata discarded | Metadata filtering impossible; precision drops |

The principle is simple: **garbage in, garbage out**. No retrieval or generation optimization compensates for corrupted text.

</details>

---

### Q4. What text cleaning steps should you apply after extraction and before chunking? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Cleaning transforms raw extracted text into structured, usable input. Apply these in order:

**Step 1 — Fix encoding issues**
```python
import ftfy

text = ftfy.fix_text(raw_text)
# Converts: "â€™" → "'"
# Fixes mojibake, surrogate pairs, wrong codecs
```

**Step 2 — Strip HTML/XML tags**
```python
from bs4 import BeautifulSoup

text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
```

**Step 3 — Remove boilerplate (headers, footers, watermarks)**
```python
import re

# Remove lines that are short repeated patterns (headers/footers)
lines = text.split("\n")
text = "\n".join(line for line in lines if len(line.strip()) > 20)
```

**Step 4 — Normalize whitespace**
```python
text = re.sub(r'\s+', ' ', text).strip()
```

**Step 5 — Remove PII (if required)**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

results = analyzer.analyze(text=text, language="en")
text = anonymizer.anonymize(text=text, analyzer_results=results).text
# Replaces: "John Smith, SSN 123-45-6789" → "<PERSON>, SSN <US_SSN>"
```

**Step 6 — Deduplication** (document level, before adding to index)
```python
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

**Critical:** Do NOT discard metadata (source, page, section) during cleaning — these fields enable precision pre-filtering at retrieval time.

</details>

---

### Q5. Why is metadata extraction during ingestion critical for RAG retrieval quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Metadata unlocks **pre-filtering** — narrowing the search space before vector similarity runs. This improves precision and reduces hallucination.

**Minimum viable metadata schema:**
```python
{
    "source": "Q4-2024-annual-report.pdf",
    "page": 12,
    "section": "Revenue Analysis",
    "doc_type": "financial_report",
    "department": "finance",
    "date": "2024-12-01",
    "language": "en"
}
```

**How metadata enables precision retrieval:**
```
User query: "2024 finance report revenue"

Without metadata → vector search across all 500k chunks
  └─ Returns: legal, HR, tech, and finance docs mixed

With metadata pre-filter:
  doc_type = "financial_report" AND date >= "2024-01-01"
  └─ Narrows to 8,000 finance chunks → much higher precision
```

**Where to extract metadata:**
- `source`: full file path or URL
- `page`: from the parser (PDF page number, HTML URL anchor)
- `section`: from document structure (headings, chapter titles)
- `doc_type`: classify at ingestion time (regex pattern or classifier)
- `date`: from file metadata or document headers

**Performance note:** Use **pre-filtering** (apply metadata filter before vector search), not post-filtering, to guarantee the top-K result pool is not depleted before ranking.

</details>

---

### Q6. What are the most common parsing failures when ingesting documents for RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Failure | Cause | Detection | Fix |
|---------|-------|-----------|-----|
| **Broken OCR** | Scanned PDF with low resolution or complex layout | Extract text → check `len(text) < threshold` per page | Re-process with higher-quality OCR; set confidence threshold |
| **Encoding mojibake** | PDF encoded in non-UTF-8; Windows-1252 / Latin-1 artifacts | Look for `â€`, `Ã©`, replacement chars `?` | `ftfy.fix_text()` |
| **Tables converted to garbage** | PDF parser reads table cells left-to-right without structure | Tables become a wall of numbers with no column headers | Use `pdfplumber.extract_tables()` instead |
| **Header/footer noise** | "Page 4 — CONFIDENTIAL" repeated in every chunk | Short repetitive lines at top/bottom of each page | Heuristic filter: drop lines < 20 chars that repeat across pages |
| **Duplicate content** | Same document ingested twice; near-duplicate policy versions | Identical chunks returned by retrieval | SHA-256 hash deduplication at ingestion |
| **Image-only pages** | PDF pages that are images, not text | `page.get_text()` returns empty string | Route empty pages to OCR pipeline |
| **Encoding errors on DB export** | CSV exported with wrong locale settings | `UnicodeDecodeError` or `?` characters | `pd.read_csv(path, encoding='utf-8-sig')` or detect encoding with `chardet` |

**Detection pattern:**
```python
def validate_extracted_text(text: str, source: str) -> bool:
    if len(text.strip()) < 50:
        print(f"WARNING: Near-empty extraction from {source}")
        return False
    # Check for common mojibake signatures
    if "â€" in text or "Ã©" in text:
        print(f"WARNING: Possible encoding issue in {source}")
    return True
```

</details>

---

### Q7. How do you handle multilingual document corpora in a RAG ingestion pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Multilingual ingestion requires decisions at three levels:

**1. Encoding normalization (always do this first)**
```python
import ftfy
import unicodedata

def normalize_text(text: str) -> str:
    text = ftfy.fix_text(text)
    # NFC normalization: canonical composition (é as single char, not e + combining accent)
    text = unicodedata.normalize("NFC", text)
    return text
```

**2. Language detection**
```python
from langdetect import detect, LangDetectException

def get_language(text: str) -> str:
    try:
        return detect(text)  # returns ISO 639-1 code: "en", "fr", "de", etc.
    except LangDetectException:
        return "unknown"
```
`langdetect` supports 55 languages. For higher accuracy at the cost of more memory, `lingua-py` supports 75 languages with better confidence scoring.

**3. Embedding model selection**

Language affects which embedding model to use:

| Corpus | Recommended Model |
|--------|------------------|
| English only | `text-embedding-3-small`, `BGE-large-en` |
| Multilingual | `multilingual-e5-large`, `paraphrase-multilingual-mpnet-base-v2` |
| Cross-lingual (query in EN, docs in FR) | `multilingual-e5-large` (trained on cross-lingual pairs) |

**Key warning:** An English-only embedding model will degrade silently on French or German documents — cosine similarity scores drop, retrieval recall falls, and no error is thrown.

**Metadata to store:** Add `"language": lang_code` to each document's metadata. This enables per-language pre-filtering and model routing at query time.

</details>

---

### Q8. How do you build a production OCR pipeline for scanned document ingestion? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A production OCR pipeline adds preprocessing, engine selection, confidence filtering, and fallback logic:

```
Scanned PDF Page
  │
  ▼
Convert to image (if PDF)
  │  pdf2image.convert_from_path()
  ▼
Image preprocessing
  ├─ Deskew (cv2.warpAffine on detected rotation angle)
  ├─ Binarize (cv2.threshold with OTSU method)
  └─ Denoise (cv2.fastNlMeansDenoising)
  │
  ▼
OCR engine
  ├─ pytesseract → get_data() returns per-word confidence scores
  ├─ easyocr    → readtext() returns (bbox, text, confidence) tuples
  └─ paddleocr  → best for structured layouts; returns bounding boxes
  │
  ▼
Confidence filtering
  └─ If mean confidence < threshold (e.g., 60 for tesseract, 0.6 for easyocr):
       → Flag page for human review or discard
  │
  ▼
Post-OCR text cleaning (ftfy, whitespace normalization)
```

**Pytesseract confidence filtering:**
```python
import pytesseract
from PIL import Image

def ocr_with_confidence(image_path: str, min_conf: int = 60) -> str:
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = [
        data["text"][i]
        for i in range(len(data["text"]))
        if int(data["conf"][i]) >= min_conf and data["text"][i].strip()
    ]
    return " ".join(words)
```

**EasyOCR example:**
```python
import easyocr

reader = easyocr.Reader(["en"], gpu=False)

def ocr_easyocr(image_path: str, min_conf: float = 0.6) -> str:
    results = reader.readtext(image_path)
    # Each result: (bbox, text, confidence)
    return " ".join(text for _, text, conf in results if conf >= min_conf)
```

**Engine selection guide:**
- `pytesseract`: best for clean, high-resolution typed text; free, no GPU needed
- `easyocr`: balanced accuracy + speed; 80+ languages; works on GPU or CPU
- `paddleocr`: best for structured layouts (tables, forms); superior bounding-box accuracy for multilingual docs

</details>

---

### Q9. How do you extract tables from PDFs and make them usable for LLM context? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Tables are among the hardest PDF elements to extract — most parsers read cells left-to-right, destroying structure. The two reliable approaches are:

**Approach 1: pdfplumber (best for digital PDFs with text layer)**
```python
import pdfplumber

def extract_tables_as_markdown(pdf_path: str) -> list[dict]:
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()  # returns list[list[list[str | None]]]
            for table in tables:
                if not table:
                    continue
                md = table_to_markdown(table)
                results.append({
                    "text": md,
                    "metadata": {"source": pdf_path, "page": page_num + 1, "type": "table"}
                })
    return results

def table_to_markdown(table: list) -> str:
    if not table or not table[0]:
        return ""
    header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
    separator = "| " + " | ".join("---" for _ in table[0]) + " |"
    rows = [
        "| " + " | ".join(str(c or "") for c in row) + " |"
        for row in table[1:]
    ]
    return "\n".join([header, separator] + rows)
```

**Approach 2: camelot-py (for complex lattice or stream tables)**
```
pip install camelot-py[cv]
```
`camelot` uses two modes: `lattice` (tables with visible borders) and `stream` (borderless, whitespace-separated). It returns tables as pandas DataFrames.

**Why Markdown output?**

LLMs understand Markdown tables better than raw CSV or space-delimited text. Converting tables to Markdown before embedding preserves column headers in the vector representation and makes the LLM's task easier during generation.

**Key consideration:** Chunk tables as a single unit — never split a table across chunks. A table split mid-row loses its meaning entirely.

</details>

---

### Q10. How do you design an incremental ingestion pipeline that handles new, updated, and deleted documents? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Full re-embedding on every update is prohibitively expensive at scale. The standard pattern uses content hashing for change detection:

```python
import hashlib
from datetime import datetime

def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class IncrementalIngester:
    def __init__(self, fingerprint_store: dict, vector_db):
        # fingerprint_store: {doc_id: content_hash}
        self.fingerprints = fingerprint_store
        self.vector_db = vector_db

    def ingest(self, doc_id: str, text: str, metadata: dict):
        new_hash = compute_content_hash(text)
        existing_hash = self.fingerprints.get(doc_id)

        if existing_hash == new_hash:
            return  # Unchanged — skip re-embedding

        if existing_hash:
            # Updated — delete old vectors, insert new
            self.vector_db.delete(filter={"doc_id": doc_id})

        # New or updated document — embed and store
        chunks = self.chunk(text)
        self.vector_db.upsert(chunks, metadata=metadata)
        self.fingerprints[doc_id] = new_hash

    def delete(self, doc_id: str):
        self.vector_db.delete(filter={"doc_id": doc_id})
        self.fingerprints.pop(doc_id, None)
```

**Handling deletions — two approaches:**

| Approach | Mechanism | Use When |
|----------|-----------|----------|
| Hard delete | Remove from vector DB immediately | Document must not be retrievable |
| Soft delete + TTL | Mark as `deleted=True` in metadata; filter at query time; evict after TTL | Audit trail required; gradual rollout |

**Async ingestion at scale:**

For high-volume pipelines, push documents into a queue (e.g., Kafka, SQS) and process with worker threads. This decouples ingestion latency from the upstream source and allows retries on failure.

See also: [04-stale_index_problem.md](../03_failure_modes/04-stale_index_problem.md) for a full treatment of index staleness patterns.

</details>

---

### Q11. How do you detect and remove duplicate content during ingestion? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Duplicate content in the index causes retrieval to return the same information multiple times, wasting context window and skewing relevance scores. Three levels of deduplication:

**Level 1: Exact deduplication (cheapest)**
```python
import hashlib

seen_hashes: set[str] = set()

def is_duplicate(text: str) -> bool:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False
```

**Level 2: Near-duplicate detection (MinHash + LSH)**

Exact hashing misses documents that are 95% identical (e.g., v1 and v2 of a policy with minor edits). Use `datasketch` for probabilistic near-duplicate detection:

```python
from datasketch import MinHash, MinHashLSH

# Build LSH index during ingestion
lsh = MinHashLSH(threshold=0.85, num_perm=128)

def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in {text[i:i+5] for i in range(len(text) - 4)}:
        m.update(shingle.encode("utf-8"))
    return m

def is_near_duplicate(doc_id: str, text: str) -> bool:
    m = get_minhash(text)
    result = lsh.query(m)  # returns doc_ids with Jaccard similarity >= threshold
    if result:
        return True
    lsh.insert(doc_id, m)
    return False
```

`datasketch` requires Python ≥ 3.9 and NumPy ≥ 1.11.

**Level 3: Semantic deduplication (most expensive)**

After embedding, compute cosine similarity between new document's embedding and existing embeddings. Discard if similarity exceeds threshold (e.g., 0.97). Use only for small corpora or as a final pass — it requires an embedding call per document.

**When to use each:**

```
Exact hash        → Always; zero cost; catches identical re-uploads
MinHash LSH       → When corpus has many document versions / copies
Semantic          → When documents are paraphrases of each other (e.g., translated duplicates)
```

</details>

---

### Q12. How do you design production error handling for a document ingestion pipeline? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Production ingestion pipelines must isolate failures so a single bad document doesn't block the entire batch.

**Error taxonomy:**

| Error Type | Examples | Handling |
|-----------|---------|---------|
| **Parse errors** | Corrupted PDF, unsupported format | Log + move to dead-letter; continue batch |
| **Encoding errors** | Non-UTF-8 content, mojibake | Attempt `ftfy` fix; fallback to `errors='replace'`; log if still failing |
| **Extraction empty** | Scanned PDF with no OCR | Route to OCR branch; if OCR also fails, move to dead-letter |
| **Embedding errors** | API rate limit, model timeout | Exponential backoff retry (3 attempts); then dead-letter |
| **Vector DB write errors** | Connection timeout, quota exceeded | Retry with backoff; circuit breaker on sustained failure |

**Idempotent pipeline design:**
```python
def ingest_document(doc_id: str, path: str) -> None:
    try:
        text = extract_text(path)          # may raise ParseError
        if not text.strip():
            text = ocr_fallback(path)      # scanned PDF fallback
        text = clean_text(text)
        chunks = chunk_text(text)
        embed_and_store(doc_id, chunks)    # may raise EmbeddingError
    except ParseError as e:
        send_to_dead_letter(doc_id, path, reason=str(e))
    except EmbeddingError as e:
        retry_queue.push(doc_id, path, attempt=1)
    except Exception as e:
        send_to_dead_letter(doc_id, path, reason=f"unexpected: {e}")
```

**Dead-letter queue pattern:**
- Store failed documents with: `doc_id`, `source_path`, `failure_reason`, `timestamp`, `attempt_count`
- Alert on queue depth > threshold (e.g., >1% of daily volume)
- Reprocess after root cause is fixed (make processing idempotent so re-runs are safe)

**Key principle:** A partial failure should never silently corrupt the index. Either the document is ingested correctly, or it goes to the dead-letter queue with a logged reason.

</details>
