# 46 — VisRAG

> A vision-language model embeds and reads document *pages as images* directly for both retrieval and generation — skipping OCR, layout parsing, and text extraction entirely.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
        PDF / Slide / Scanned Document
                    │
                    ▼
        Render each page as an IMAGE
        (no OCR, no layout parser, no
         table/figure extraction step)
                    │
                    ▼
        VLM Page Encoder (embeds the
        whole page image into a vector —
        text, tables, figures, layout,
        all captured jointly)
                    │
                    ▼
        Page-Image Vector Store (ANN
        index over page-image embeddings)
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   Query (text)          Top-k Page Images
        │                       │
        └───────────┬───────────┘
                    ▼
        VLM Generator (reads the
        retrieved page IMAGES directly,
        conditioned on the text query)
                    │
                    ▼
                 Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| Page Renderer | Converts each document page to an image (PDF→PNG/JPEG); no parsing library involved |
| VLM Page Encoder | Embeds the raw page image (layout, text, tables, figures together) into a single dense vector |
| Page-Image Vector Store | ANN index over page-image embeddings, queried with a text (or image) query |
| VLM Generator | A vision-language model that reads the retrieved page images directly and produces the answer — generation is also image-conditioned, not just retrieval |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Reference implementation | [OpenBMB/VisRAG](https://github.com/openbmb/visrag) (open-source, MiniCPM-V based) |
| VLM encoders/generators | MiniCPM-V, GPT-4o (vision), Gemini 1.5/2.x (vision), Qwen2-VL |
| Page rendering | `pdf2image`, `PyMuPDF` (`fitz`) — render pages to images, no text extraction |
| Related retrieval-only method | ColPali / ColQwen2 (late-interaction, patch-level page embeddings — see file 30) |
| Vector store | FAISS, Qdrant, Milvus (same ANN infra as text RAG, just embedding page images instead of text chunks) |

---

## Q1. What is VisRAG, and how does it differ from standard Multi-modal RAG (file 09) and document parsing (OCR/layout extraction)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**VisRAG** ("VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents," Yu, Tang, Xu, Cui, Ran, Yan, Liu, Wang, Han, Liu & Sun, 2024, [arXiv:2410.10594](https://arxiv.org/abs/2410.10594)) treats each document page as an **image**, end to end. A vision-language model (VLM) embeds the page image for retrieval, and the same class of VLM reads the retrieved page images directly to generate the answer. There is no OCR step, no layout parser, no table-structure model, and no text-extraction pipeline anywhere in the loop.

**Contrast with document ingestion/parsing** (`01_concepts/document_ingestion_and_parsing.md`): the standard pipeline — Unstructured, LlamaParse, Docling, Tesseract/`pytesseract` — exists specifically to convert PDFs/scans into clean text or Markdown before a text embedder ever sees the content. Every one of those tools can lose information: a table's structure can be flattened wrong, a figure's caption can get disassociated from the figure, reading order on a multi-column layout can get scrambled. VisRAG's entire premise is that this loss is avoidable — feed the model the pixels and let the VLM's own visual understanding do what OCR + layout reconstruction used to do.

**Contrast with Multi-modal RAG (file 09):** file 09's architecture keeps *separate* encoders per modality (text encoder, image encoder via CLIP/SigLIP, table-to-text summarizer, ASR for audio) feeding into a unified or per-modality vector store — modalities are extracted and handled independently, then reconciled. VisRAG instead collapses "text + table + figure + layout" into **one modality: the page image**, embedded and read by a single VLM. It's a narrower but deeper bet specifically on documents (PDFs, slides, scanned reports), not a general framework for arbitrary text/image/audio/video mixtures.

```
Standard text RAG:     PDF ─OCR/parse─► text chunks ─embed─► vector store ─► text-only LLM
Multi-modal RAG (09):  PDF ─(split by modality)─► [text|image|table|audio] encoders ─► unified store ─► multimodal LLM
VisRAG:                PDF ─render page as image─► VLM embeds PAGE IMAGE ─► VLM reads PAGE IMAGE ─► answer
                              (no parsing step exists anywhere in this pipeline)
```

</details>

---

## Q2. How is VisRAG's retrieval index actually built and queried, given that documents are never converted to text? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Retrieval works the same way as text RAG structurally (embed → index → ANN search), but the thing being embedded is a rendered page image, not a text chunk. The VLM used for embedding is trained (often via contrastive learning, à la CLIP-style objectives) so that a *text query* embedding lands close to the *page-image* embeddings of pages that answer it.

```python
from PIL import Image
import fitz  # PyMuPDF
import faiss
import numpy as np

def render_pdf_pages(pdf_path: str, dpi: int = 150) -> list[Image.Image]:
    """Render each PDF page directly to an image — no text extraction at all."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return pages

def embed_page_images(vlm_encoder, page_images: list[Image.Image]) -> np.ndarray:
    """VLM embeds each rendered page image as a single dense vector."""
    return vlm_encoder.encode_images(page_images)  # shape: (n_pages, dim)

# Build the index
page_images = render_pdf_pages("quarterly_report.pdf")
page_embeddings = embed_page_images(vlm_encoder, page_images)

index = faiss.IndexFlatIP(page_embeddings.shape[1])
index.add(page_embeddings.astype(np.float32))

def retrieve_pages(query: str, k: int = 3) -> list[Image.Image]:
    q_emb = vlm_encoder.encode_text([query])   # same embedding space as page images
    scores, idx = index.search(q_emb.astype(np.float32), k)
    return [page_images[i] for i in idx[0]]
```

**Why this can outperform text-based retrieval on visually rich documents:** a table with merged cells, a chart with an axis label, or a figure with an embedded caption are all represented faithfully in a rendered page image. An OCR pipeline has to *reconstruct* structure that a VLM's visual encoder can perceive directly — no reconstruction step means no reconstruction errors.

</details>

---

## Q3. How does VisRAG generate an answer once it has retrieved page images — and how is that different from ColPali? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Generation, like retrieval, is fully image-conditioned: the retrieved page images are passed as image inputs directly to a VLM generator alongside the text query — the VLM reads the page visually rather than reading extracted text.

```python
def visrag_answer(query: str, vlm_generator, k: int = 3) -> str:
    top_pages = retrieve_pages(query, k=k)   # list of PIL Images, no text anywhere

    # Multimodal prompt: images + text query, no OCR'd text in the prompt
    content = []
    for page_img in top_pages:
        content.append({"type": "image", "image": page_img})
    content.append({"type": "text", "text": f"Answer the question using the page images above.\n\nQuestion: {query}"})

    response = vlm_generator.generate(messages=[{"role": "user", "content": content}])
    return response
```

**Contrast with ColPali** (referenced in files 09 and 30): ColPali is a **retrieval-only** method. It uses a late-interaction, patch-level scheme (like ColBERT's MaxSim, but over image patches instead of text tokens) to retrieve the most relevant page images extremely precisely. But once ColPali hands back its top-k pages, something else (typically a standard text-based LLM, after an OCR pass, or a separate VLM) still has to *read* those pages to generate an answer.

```
ColPali:  Query ──► patch-level late-interaction retrieval ──► top-k page images
                                                                       │
                                                          (retrieval stops here —
                                                           generation is a SEPARATE step,
                                                           often still needs OCR/VLM reader)

VisRAG:   Query ──► VLM retrieves page images ──► SAME CLASS OF VLM reads the
                                                    images and generates the answer
                    (retrieval AND generation are both image-native, end-to-end)
```

**In short:** ColPali answers "which pages should I retrieve?" with unusually high precision (page→patch-level scoring). VisRAG answers the full question "how do I retrieve *and* generate over page images without ever touching parsed text?" They're compatible, not mutually exclusive — a production pipeline could use ColPali-style late interaction for the retrieval stage and a VisRAG-style VLM for the generation stage.

</details>

---

## Q4. What training objective and data does VisRAG use, and does it require fine-tuning? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

VisRAG's retriever is trained with a contrastive objective — the same family as dense text retrievers (e.g. in-batch negatives, InfoNCE-style loss) — except one side of the pair is a rendered page image instead of a text passage:

```python
# Conceptual contrastive training objective for the VisRAG retriever
# (VLM encoder produces embeddings for both text queries and page images)

def contrastive_loss(query_embs, page_image_embs, temperature=0.02):
    """
    query_embs:      (batch, dim) — text query embeddings
    page_image_embs: (batch, dim) — corresponding positive page-image embeddings
    In-batch negatives: every other page image in the batch acts as a negative.
    """
    sims = query_embs @ page_image_embs.T / temperature
    labels = torch.arange(len(query_embs))          # diagonal = positive pairs
    return cross_entropy(sims, labels)
```

- **Base model**: the paper builds on an existing open VLM (MiniCPM-V family) rather than training a VLM from scratch — the encoder is fine-tuned for the retrieval task specifically.
- **Training data**: synthetic (query, page-image) pairs generated from document collections, plus existing document VQA / retrieval datasets.
- **Generation side**: the VLM generator can be used off-the-shelf (zero-shot) once given retrieved page images, or further instruction-tuned on QA-over-document-images data for better answer quality.

**Fine-tuning requirement in practice:** off-the-shelf VLMs (GPT-4o, Gemini, Claude with vision) can serve as a VisRAG-style *generator* with no fine-tuning — they already accept page images and answer questions about them zero-shot. The retrieval side benefits most from fine-tuning, since general-purpose VLM embeddings are not necessarily optimized for the "does this page image answer this text query" contrastive objective the way a purpose-built retriever is.

</details>

---

## Q5. When does VisRAG lose to a good OCR+text pipeline, and how would you combine the two approaches? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

VisRAG's advantage — no information loss from parsing — is not free. It trades one failure mode for another set of tradeoffs:

| Dimension | OCR/layout-parsing text RAG | VisRAG |
|---|---|---|
| Complex tables, multi-column layout, figures with captions | Risk of structural loss during parsing | Preserved faithfully as pixels |
| Long, dense prose pages | Cheap to embed/search as text; very mature tooling | Page-image embeddings can be less precise for pure long-text retrieval; more expensive |
| Retrieval/embedding cost | Low — text tokens are cheap to embed and index | Higher — image embeddings and VLM inference cost more per page |
| Exact-match / keyword search (BM25) | Trivial — works out of the box on extracted text | Not directly possible — no text index exists unless one is built in parallel |
| Explainability / highlighting exact cited span | Easy — can highlight the exact extracted text span | Harder — "citation" is a whole page image, not a precise span, unless a secondary grounding step is added |
| Scanned/handwritten documents where OCR fails badly | OCR pipeline fails outright, garbage text | VLM can often still read the page correctly (the failure case VisRAG solves for) |

**Where VisRAG clearly wins:** visually dense documents — financial reports with nested tables, scientific papers with figures/equations, scanned forms, slide decks — exactly the documents that break `01_concepts/document_ingestion_and_parsing.md`'s pipeline (per that guide: table-heavy PDFs need `pdfplumber` or `unstructured`/`Docling`, scanned/irregular layouts need VLM-based parsing or `LlamaParse` — VisRAG is essentially pushing that "VLM-based parsing" escape hatch all the way through retrieval too, rather than treating it as a one-off tool for hard documents).

**Where a text pipeline still wins:** plain-text-heavy corpora (long-form articles, legal contracts without complex tables), where OCR/parsing is already lossless or near-lossless, and where BM25/keyword search and cheap embedding costs matter more than layout fidelity.

**Hybrid approach in production:**

```
Route by document type:
  Plain-text-dominant docs   → standard OCR/parse → text RAG (cheap, precise citations)
  Visually-dense docs        → VisRAG (page-image retrieval + VLM generation)
  Mixed corpora              → dual-index: text index (BM25 + dense) AND page-image
                                index (VisRAG) queried together, results merged/reranked
```

A dual-index hybrid also mitigates VisRAG's citation-precision weakness: use the text index (where available) to produce an exact quoted span for the user-facing citation, while relying on the page-image index/VLM to catch cases where the text extraction silently failed or lost structure.

</details>

---

## Real-World Applications

- **Financial report / 10-K analysis**: tables of quarterly figures and charts retrieved and read without risking a broken table-to-text conversion
- **Scientific paper QA**: figures, equations, and multi-column layouts preserved for the VLM to reason over directly
- **Scanned legacy document archives**: government/insurance/legal scans where OCR historically fails on poor scan quality or handwriting
- **Slide-deck and presentation search**: retrieving the right slide by its visual layout and embedded chart, not just any text that happens to be on it
- **OpenBMB's VisRAG reference implementation**: open-source, MiniCPM-V-based, demonstrating a 20–40% end-to-end gain over traditional text-based RAG pipelines on multi-modality document benchmarks
