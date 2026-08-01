# Chunking Strategies: How You Split Documents Determines What Gets Retrieved

> How you split documents determines what can be retrieved — chunking is the most underrated RAG design decision.

---

## What is Chunking?

Chunking is the process of splitting a document into smaller, self-contained pieces before it gets embedded and indexed, so retrieval can return a focused, relevant snippet instead of an entire document. Every RAG pipeline chunks somewhere between ingestion and embedding — the only question is how. The size and boundaries you choose directly shape what a query can find: chunk too large and you dilute relevance with unrelated text; chunk too small and you lose the surrounding context needed to make sense of the snippet.

---

## Why Chunking Exists

Embedding models truncate long sequences. Most are trained on context windows of 512–8192 tokens. When you pass a longer document, two things happen:

1. **Truncation:** The model cuts off text beyond the window
2. **Degradation:** Embeddings of truncated text lose information about content that was cut

This forces a choice: either accept information loss or split documents into chunks.

The second forcing function is **retrieval granularity**:

```
Large chunks (e.g., full pages)
  ├─ Pros: High recall (more context returned)
  └─ Cons: Low precision (irrelevant text mixed with relevant)

Small chunks (e.g., sentences)
  ├─ Pros: High precision (only relevant snippet returned)
  └─ Cons: Low recall (might miss context that spans chunks)
```

**Precision/Recall/Latency Triangle:**

```
         Precision
            ▲
            │     Small Chunks
            │      ●●●
            │     ●   ●
            │    ●     ●
            │   ●       ●
      Medium│  ●         ●
       Chunk● ●           ●  ← Trade-off frontier
            ●               ●
            ●     Large      ●
            ●    Chunks      ●
            └─────────────────────► Recall
```

Your job as a system designer is to pick a point on this frontier based on your latency and quality requirements.

---

## Strategy Catalog: 9 Approaches

### 1. Fixed-Size with Overlap

**Mechanism:** Split document every N characters; overlap previous M characters.

**Parameters:** chunk_size (typical: 256–1024), overlap (typical: 50–100)

**Pros:** Simple, predictable, deterministic
**Cons:** Splits mid-sentence, no semantic awareness
**When to use:** Baseline; when no better option is available

```python
def chunk_fixed_size(text: str, size: int = 512, overlap: int = 100) -> list[str]:
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks
```

**Example impact on retrieval:** A naive fixed-size split can cut a sentence in half at exactly the wrong point. Take: *"Flood damage is excluded unless the policyholder purchased the flood rider."* If the boundary falls mid-sentence, chunk A ends up with "Flood damage is excluded" and chunk B starts with "unless the policyholder purchased the flood rider." A similarity search for "is flood damage covered?" may retrieve only chunk A — its embedding is a strong match for "flood damage excluded" — and never surface chunk B's exception, giving a misleadingly incomplete (even incorrect) answer. This is the single biggest argument for overlap or a boundary-aware strategy over raw fixed-size splitting.

---

### 2. Sentence Boundary

**Mechanism:** Split on sentence boundaries (detected via regex or NLTK), not mid-sentence.

**Parameters:** max_chunk_sentences (typical: 5–15)

**Pros:** No mid-sentence breaks; more semantically coherent
**Cons:** Chunk sizes vary wildly (short sentences vs. long sentences); still naive
**When to use:** General documents (news, articles, blogs)

```python
import nltk
from nltk.tokenize import sent_tokenize

def chunk_by_sentence(text: str, max_sentences: int = 10) -> list[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    
    for sent in sentences:
        current_chunk.append(sent)
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

---

### 3. Paragraph / Section Boundary

**Mechanism:** Split on paragraph breaks (`\n\n`) or markdown headers (`##`, `###`).

**Parameters:** None (determined by document structure)

**Pros:** Respects document semantics (sections are designed by author)
**Cons:** Paragraphs vary in size; some are empty or tiny; no control
**When to use:** Structured documents (papers, documentation, markdown)

```python
def chunk_by_paragraph(text: str) -> list[str]:
    # Split on double newline (paragraph boundary)
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def chunk_by_markdown_header(markdown_text: str) -> list[str]:
    """Split markdown on top-level headers."""
    chunks = []
    current = []
    
    for line in markdown_text.split('\n'):
        if line.startswith('# '):  # Top-level header
            if current:
                chunks.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    
    if current:
        chunks.append('\n'.join(current))
    return chunks
```

---

### 4. Recursive Character Splitting (LangChain)

**Mechanism:** Try splitting on natural boundaries in order: `\n\n`, `\n`, ` `, then characters. Stop when chunk is small enough.

**Parameters:** chunk_size, chunk_overlap, separators (list of boundaries to try)

**Pros:** Semantically smart; respects document structure; doesn't split mid-word
**Cons:** Slightly slower (recursive calls); parameters require tuning
**When to use:** Default choice for most text documents (news, blogs, docs)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(document_text)
```

**How it works:** Tries to split on `\n\n` first; if chunks are still >1000 chars, splits on `\n`; if still too large, splits on spaces; finally, character-level split as last resort.

---

### 5. Semantic Chunking

**Mechanism:** Embed every sentence. Split when embedding similarity drops below threshold (sign of topic change).

**Parameters:** similarity_threshold (typical: 0.5–0.7)

**Pros:** Preserves semantic cohesion; respects topic boundaries
**Cons:** Expensive (embed every sentence); requires embedding model at index-time
**When to use:** High-precision retrieval where semantic coherence is critical (medical, legal)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def chunk_semantic(text: str, threshold: float = 0.6) -> list[str]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = sent_tokenize(text)
    
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i], embeddings[i-1])
        
        if similarity < threshold:  # Topic change detected
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

---

### 6. Document-Aware Chunking

**Mechanism:** Respect document structure (HTML tags, markdown headers, code blocks). Don't split mid-code-block or mid-table.

**Parameters:** Structure-specific (tag-aware)

**Pros:** Preserves semantic structure; perfect for code, tables, mixed-media
**Cons:** Requires document-type-specific logic
**When to use:** Code repositories, documentation, mixed-content PDFs

```python
def chunk_code_aware(text: str) -> list[str]:
    """For code: split by function/class, not lines."""
    chunks = []
    current = []
    in_function = False
    
    for line in text.split('\n'):
        is_definition = line.startswith('def ') or line.startswith('class ')
        is_dedent = bool(line) and not line[0].isspace() and not is_definition
        
        if is_definition:
            if current:
                chunks.append('\n'.join(current))
            current = [line]
            in_function = True
        elif in_function and is_dedent:
            # A non-indented, non-blank line means the previous function/class ended.
            chunks.append('\n'.join(current))
            current = [line]
            in_function = False
        else:
            current.append(line)
    
    if current:
        chunks.append('\n'.join(current))
    return chunks
```

---

### 7. Hierarchical Chunking (Parent-Child)

**Mechanism:** Create two levels: small chunks for retrieval precision + large parent chunks for LLM context. Store both; retrieve child, return parent.

**Parameters:** small_chunk_size, large_parent_size

**Pros:** Combines precision (small chunk retrieval) with context (large parent generation)
**Cons:** Index size doubles; requires storing relationships
**When to use:** Long documents where context matters (books, theses, long-form articles)

```python
class HierarchicalChunker:
    def __init__(self, small_size: int = 256, large_size: int = 1024):
        self.small_size = small_size
        self.large_size = large_size
    
    def chunk(self, text: str) -> list[dict]:
        # Create large chunks (parents)
        parent_chunks = self._chunk_fixed(text, self.large_size, overlap=100)
        
        results = []
        for parent_id, parent_text in enumerate(parent_chunks):
            # Create small chunks (children) within each parent
            child_chunks = self._chunk_fixed(parent_text, self.small_size, overlap=50)
            
            for child_id, child_text in enumerate(child_chunks):
                results.append({
                    'text': child_text,
                    'parent_id': parent_id,
                    'parent_text': parent_text,
                    'child_id': child_id
                })
        return results
    
    def _chunk_fixed(self, text: str, size: int, overlap: int) -> list[str]:
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunks.append(text[i:i+size])
        return chunks
```

**Retrieval + Generation Flow:**

```
Query → Embed → Retrieve Small Chunks (top-5) → Fetch Parent Chunks → Generate
        (precision)                  (context)
```

---

### 8. Agentic Chunking

**Mechanism:** Use an LLM to decide chunk boundaries. Prompt the LLM: "Where would a human naturally break this text?"

**Parameters:** None (LLM decides)

**Pros:** Most semantically intelligent; handles complex documents
**Cons:** Expensive (O(document_length / window_size) LLM calls); slow
**When to use:** Critical documents (legal contracts, research papers); small corpus

```python
def chunk_agentic(text: str, llm_client) -> list[str]:
    """Use an LLM to decide chunk boundaries."""
    # Slide a window over the document
    window_size = 2000  # tokens
    overlap = 500
    chunks = []
    
    for i in range(0, len(text), window_size - overlap):
        window = text[i:i+window_size]
        lines = window.split('\n')
        
        # Ask LLM where to split
        prompt = f"""Given this text, suggest 2-3 natural break points where a human would split into separate chunks. Reply with line numbers only, one per line.

Text:
{window}

Break points (line numbers):"""
        
        response = llm_client.complete(prompt)
        
        # Parse the LLM's response into a sorted list of valid line-number break points
        break_lines = sorted({
            int(tok) for tok in response.split()
            if tok.strip().isdigit() and 0 < int(tok) < len(lines)
        })
        
        # Split this window's lines at the suggested break points
        start = 0
        for break_line in break_lines:
            chunks.append('\n'.join(lines[start:break_line]))
            start = break_line
        chunks.append('\n'.join(lines[start:]))
        
    return chunks
```

---

### 9. Late Chunking

*Introduced by JinaAI (2024). Requires an embedding model that exposes token-level outputs.*

**Mechanism:** All 8 strategies above embed chunks independently — each chunk is encoded without knowledge of what surrounds it. Late Chunking inverts this: embed the **entire document first** with a long-context embedding model, then pool the resulting token-level embeddings into chunk-sized windows.

**Parameters:** chunk_boundaries (character spans, determined after embedding), max document length (bounded by the embedding model's context window, e.g., 8,192 tokens for JinaAI v3)

**Pros:** Each chunk's embedding reflects full-document context (entity names, dates, titles mentioned elsewhere in the doc); no separate contextualization step needed
**Cons:** Requires a token-level embedding model; document must fit in the model's context window; any edit forces re-embedding the whole document
**When to use:** Corpora where chunks frequently omit necessary context (entity names/dates/titles that only appear at the top of the document) and queries use document-level vocabulary

```
Standard chunking:
  Document → [Chunk 1] → Embed → vector_1
             [Chunk 2] → Embed → vector_2   ← Chunk 2 has no context from Chunk 1

Late Chunking:
  Document → Full-document embedding (token-level) → [token_1, token_2, ..., token_N]
           → Pool tokens for Chunk 1 window → vector_1  ← Contains cross-chunk context
           → Pool tokens for Chunk 2 window → vector_2  ← Also contains full-doc context
```

**Why it preserves cross-chunk context:** Standard chunking splits "The growth rate improved to 23%" into a chunk that doesn't mention *which company* or *which quarter* — forcing the embedding model to work with a decontextualized fragment. In Late Chunking, when encoding "The growth rate improved to 23%", the model has already attended to the full document including "Acme Corp Q3 2024" — so the token embeddings for this passage reflect the company and quarter, even though those words aren't in the chunk.

**Requirements:**

| Requirement | Detail |
|---|---|
| **Embedding model** | Must expose token-level outputs (not just the [CLS] pooled vector) |
| **Compatible models** | JinaAI Embeddings v3, `nomic-embed-text`, long-context bi-encoders |
| **Context window** | Document must fit in the embedding model's context (JinaAI v3: 8,192 tokens) |

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "jinaai/jina-embeddings-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

def late_chunk_embed(document: str, chunk_boundaries: list[tuple[int, int]]):
    """
    chunk_boundaries: list of (start_char, end_char) for each chunk
    Returns: list of embeddings, one per chunk
    """
    inputs = tokenizer(document, return_tensors="pt",
                       return_offsets_mapping=True, truncation=True,
                       max_length=8192)
    offset_mapping = inputs.pop("offset_mapping")[0]  # (num_tokens, 2)

    with torch.no_grad():
        outputs = model(**inputs)
    
    token_embeddings = outputs.last_hidden_state[0]  # (num_tokens, hidden_dim)
    
    chunk_embeddings = []
    for start_char, end_char in chunk_boundaries:
        # Find token indices that fall within this chunk's character span
        mask = (offset_mapping[:, 0] >= start_char) & (offset_mapping[:, 1] <= end_char)
        chunk_tokens = token_embeddings[mask]
        # Mean pool over the chunk's tokens
        chunk_emb = chunk_tokens.mean(dim=0)
        chunk_embeddings.append(chunk_emb.numpy())
    
    return chunk_embeddings
```

---

## Comparison Table: All 9 Strategies

| Strategy | Chunk Size Control | Semantic Coherence | Index Size | Latency | Implementation Complexity | When to Use |
|----------|-------------------|-------------------|-----------|---------|--------------------------|------------|
| Fixed-size | Full | Low | Small | Fast | 1 line | Baseline, any document |
| Sentence | Loose | Medium | Medium | Fast | Simple regex | General documents |
| Paragraph | Loose | High | Medium | Fast | Simple split | Structured documents |
| Recursive | Full | High | Medium | Fast | Moderate | **Default choice** |
| Semantic | Full | Highest | Medium | Slow | Moderate (need embedder) | High-precision retrieval |
| Document-Aware | Varies | Highest | Medium | Moderate | High (type-specific) | Code, tables, mixed media |
| Hierarchical | Full | High | 2x | Fast | Moderate (need relationships) | Long documents + context matters |
| Agentic | N/A | Highest | Medium | Slowest | High (need LLM) | Critical documents, small corpus |
| Late Chunking | Full | Highest (full-doc context) | Medium | Slow (index); fast (query) | High (needs token-level embedder) | Chunks lose context without doc-level info |

---

## The Parent-Child Retrieval Pattern

Hierarchical chunking enables a powerful pattern: retrieve small chunks for precision, return large chunks for context.

```
Corpus
  │
  ├─ Document 1
  │  ├─ [Parent: Full chapter on "Deep Learning"]
  │  │  ├─ [Child 1: "Introduction to neural networks"]
  │  │  ├─ [Child 2: "Backpropagation algorithm"]
  │  │  └─ [Child 3: "Training deep networks"]
  │
  └─ Document 2
     └─ ...

Query: "How does backpropagation work?"
  │
  ├─ Embed query, search vector DB
  │  └─ Top-5 results: Child 2 from Doc 1, Child 2.1, Child 2.2, ...
  │
  ├─ Fetch parent chunks for each child
  │  └─ Parent: Full chapter on "Deep Learning" (provides full context)
  │
  └─ Pass parent + question to LLM for generation
     └─ Answer (informed by both precision + context)
```

**Why this beats single-level chunking:**
- Single small chunks: LLM loses context (answer is fragmented)
- Single large chunks: Retrieval precision drops (might retrieve irrelevant parts)
- Parent-child: Retrieval is precise (child), generation has context (parent)

---

## Chunking Parameters and Their Effect

| Parameter | Effect on Precision | Effect on Recall | Effect on Index Size | Effect on Latency |
|-----------|-------------------|-----------------|-------------------|------------------|
| Smaller chunk_size | ↑ (fewer irrelevant words) | ↓ (must retrieve more) | ↑ (more chunks) | ↓ (faster retrieval) |
| Larger chunk_size | ↓ (more noise) | ↑ (more context) | ↓ (fewer chunks) | ↑ (slower retrieval) |
| Larger overlap | ↑ (boundary effects reduced) | ↑ (redundancy helps) | ↑↑ (more chunks) | ↑ (more chunks to store/search, slightly slower) |
| Smaller overlap | ↓ (boundary effects) | ↓ (gaps between chunks) | ↓ (fewer chunks) | ↓ (fewer chunks, faster but risks gaps) |

**Calibration Method:** Grid search over chunk_size values against a labeled probe set.

```python
def calibrate_chunking(document: str, labeled_queries: list, embedding_model) -> dict:
    """Find optimal chunk_size by grid search."""
    best_params = None
    best_recall = 0
    
    for chunk_size in [256, 512, 1024, 2048]:
        # Chunk the document
        chunks = chunk_fixed_size(document, size=chunk_size, overlap=chunk_size//4)
        
        # Embed chunks
        chunk_embeddings = embedding_model.encode(chunks)
        
        # For each labeled query, check if relevant chunks are in top-k
        recall_at_5 = 0
        for query_text, relevant_chunks in labeled_queries:
            query_emb = embedding_model.encode(query_text)
            similarities = np.dot(chunk_embeddings, query_emb)
            top_5 = np.argsort(-similarities)[:5]
            
            if any(chunk_id in relevant_chunks for chunk_id in top_5):
                recall_at_5 += 1
        
        recall_at_5 /= len(labeled_queries)
        
        if recall_at_5 > best_recall:
            best_recall = recall_at_5
            best_params = {'chunk_size': chunk_size}
    
    return best_params
```

---

## Chunking for Non-Standard Documents

### Code Files

**Don't:** Split by line count
**Do:** Split by function or class boundaries

```python
def chunk_python_code(code: str) -> list[str]:
    """Split Python by function/class definitions."""
    tree = ast.parse(code)
    chunks = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            chunk_start = node.lineno - 1
            chunk_end = node.end_lineno
            chunks.append('\n'.join(code.split('\n')[chunk_start:chunk_end]))
    
    return chunks
```

### Tables (CSV, Markdown)

**Don't:** Split mid-row
**Do:** Preserve table structure; embed row-by-row OR as whole unit

```python
def chunk_csv_table(csv_text: str) -> list[str]:
    """Each row is a chunk; include header with each row."""
    lines = csv_text.strip().split('\n')
    header = lines[0]
    chunks = [header]  # Header alone for retrieval
    
    for row in lines[1:]:
        # Include header + row for context
        chunks.append(header + '\n' + row)
    
    return chunks
```

### PDFs with Complex Layouts

**Use:** pdfplumber or pypdfium2 to respect layout; avoid raw text extraction

```python
import pdfplumber

def chunk_pdf_complex_layout(pdf_path: str) -> list[str]:
    """Extract text while respecting page layout."""
    chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Respect page structure
            text = page.extract_text()
            # Split by structural elements (tables, text blocks)
            chunks.append(text)
    
    return chunks
```

### Multilingual Documents

**Gotcha:** Character-level tokenizers differ. BPE (used by many models) handles some languages poorly.

```python
def chunk_multilingual(text: str, language: str, tokenizer) -> list[str]:
    """Chunk respecting language tokenization."""
    # Use language-aware tokenizer
    if language == 'zh':  # Chinese: no spaces between words
        tokens = list(text)  # Character-level
    elif language == 'ja':  # Japanese: use MeCab or janome
        import janome
        tokens = [t.surface for t in janome.Tokenizer().tokenize(text)]
    else:  # European: word-level
        tokens = text.split()
    
    # Group into chunks by token count (not character count)
    chunks = []
    current = []
    for token in tokens:
        current.append(token)
        if len(current) >= 100:  # 100 tokens
            chunks.append(' '.join(current))
            current = []
    
    if current:
        chunks.append(' '.join(current))
    return chunks
```

---

## Common Mistakes

1. **Splitting Mid-Sentence (with fixed-size chunking)**
   - Problem: Context is lost
   - Fix: Use sentence or recursive splitting

2. **Ignoring Overlap Entirely**
   - Problem: Information at chunk boundaries is lost
   - Fix: Always use overlap of roughly 10-20% of chunk size

3. **Using Same Chunk Size for All Document Types**
   - Problem: Code needs function-level splits, text needs sentence-level
   - Fix: Detect document type; use type-specific chunking

4. **Not Measuring Chunk Quality**
   - Problem: You don't know if your chunking helps or hurts
   - Fix: Run NDCG@5 on a labeled probe set; measure impact of chunk_size changes

5. **Hierarchical Chunking Without Measuring Value**
   - Problem: Doubles index size; only worth it if improves quality
   - Fix: A/B test parent-child vs. single-level retrieval

6. **Chunking at Index-Time Only**
   - Problem: Can't tune later without re-indexing entire corpus
   - Fix: Store chunk boundaries in metadata; support re-chunking without re-embedding

---

## Key Takeaways

1. **Start with Recursive Character Splitting.** It handles 90% of cases.
2. **Chunk size matters most.** More than any other parameter. Calibrate on your corpus.
3. **Measure retrieval quality on a labeled probe set.** Don't guess chunk sizes.
4. **Parent-child chunking is worth it for long documents.** But measure the value first.
5. **Document-aware chunking is critical for code and tables.** Don't apply naive chunking.

---

## Interview Q&A

**Q: How do you chunk code files vs. prose documents?** `[Intermediate]`

Prose uses linguistic boundaries (sentences, paragraphs); code has semantic boundaries defined by its AST (Abstract Syntax Tree). For code: (1) **Function/method level** is the natural chunk unit — one function per chunk preserves the callable signature, docstring, and body as a unit. (2) **Class level** for small classes; split large classes by methods. (3) Use language-specific parsers (`tree-sitter`, Python's `ast` module) to extract exact boundary positions rather than splitting on newlines — a newline mid-expression is not a semantic boundary. (4) Preserve the full function signature even when the body is truncated (for long functions). (5) For file-level context (imports, class hierarchy), prepend a "file header" context block to each chunk similar to Contextual Retrieval. Never use fixed-character chunking for code — it will split in the middle of function signatures and break semantics.

---

**Q: How would you chunk a 200-page PDF financial report with mixed text, tables, and charts?** `[Advanced]`

A mixed-content PDF requires content-type-aware chunking rather than a single strategy: (1) **Detect content types** — use pdfplumber or PyMuPDF to identify text regions, table bounding boxes, and image regions separately. (2) **Text sections**: chunk by section headers (H1/H2 boundaries if detectable); within sections, use paragraph or sentence boundaries with 20% overlap. (3) **Tables**: extract with camelot or pdfplumber's table extractor; serialize as Markdown (column headers + rows); store as single chunks keyed by table title or position. Do not mix table rows with surrounding prose — they have different embedding characteristics. (4) **Charts/figures**: use a current multimodal/vision-capable LLM (e.g., GPT-4o, Claude, Gemini) to generate a text description of each chart; embed the description, not the image. (5) **Cross-reference metadata**: tag each chunk with page number, section title, and document name — financial reports are often queried by section ("page 47, liquidity risk").

---

## Related

- [Embeddings](./embeddings.md)
- [Retrieval Strategies](./retrieval_strategies.md)
- [Cost Optimization](./cost_optimization.md)
- [Document Ingestion and Parsing](./document_ingestion_and_parsing.md)
- [Vector Databases](./vector_databases.md)
