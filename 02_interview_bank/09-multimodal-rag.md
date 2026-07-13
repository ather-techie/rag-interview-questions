# 09 — Multi-modal RAG

> Retrieves across multiple data types — text, images, tables, audio — using multi-modal embeddings and vision-language models.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
     Multi-modal documents (text / image / table / audio / video)
                            │
        ┌──────────┬────────┼────────┬───────────┐
        ▼          ▼        ▼        ▼           ▼
      Text       Image    Table    Audio       Video
    Encoder     Encoder  Encoder  Encoder    (frames + ASR)
        │          │        │        │           │
        └──────────┴────────┴────────┴───────────┘
                            ▼
              Unified / per-modality vector store
                            │
                            ▼
                Cross-modal Retriever (query in
                any modality: text / image / etc.)
                            │
                            ▼
                 Multimodal LLM Generator
                (text + images inline in prompt)
                            │
                            ▼
                     Answer + citations
```

### Key Components

| Component | Responsibility |
|---|---|
| Text Encoder | Embeds text chunks (e.g., text-embedding-3, BGE) |
| Image Encoder | Embeds images into a shared or dedicated space (CLIP / SigLIP) |
| Table/Audio Encoder | Converts tables to text summaries and transcribes audio (Whisper) before embedding |
| Multi-vector / Unified Vector Store | Stores per-modality embeddings and supports cross-modal ANN search |
| Cross-modal Retriever | Accepts a query in one modality and retrieves relevant items across modalities |
| Multimodal Generator | Synthesizes an answer from interleaved text/image context, citing sources |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Cross-modal embeddings | CLIP, SigLIP, ImageBind, ColPali / ColQwen2 |
| Vector stores | Weaviate, Milvus, Qdrant (multi-vector support) |
| Document parsing | Unstructured.io, LlamaParse, Azure Document Intelligence |
| Multimodal generation | GPT-4o, Claude (vision), Gemini 1.5/2.x |

---

## Q1. What is Multi-modal RAG and what data types does it support? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Multi-modal RAG** extends RAG beyond plain text to support retrieval and generation across heterogeneous data types:

| Modality | Examples | Embedding Model |
|---|---|---|
| Text | Documents, PDFs, web pages | text-embedding-3, BGE |
| Images | Photos, diagrams, screenshots | CLIP, SigLIP, OpenCLIP |
| Tables | CSV, spreadsheets, HTML tables | Table2Text + text embeddings |
| Audio | Transcripts, voice notes | Whisper → text → text embeddings |
| Video | Keyframes + transcripts | CLIP (frames) + text (transcript) |

The key challenge is **cross-modal retrieval** — a text query should be able to retrieve relevant images, and an image query should retrieve relevant text.

</details>

---

## Q2. How does CLIP enable cross-modal retrieval? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**CLIP (Contrastive Language-Image Pretraining)** by OpenAI jointly trains text and image encoders so that semantically related text-image pairs are close in the same embedding space:

```
"a dog playing fetch" ──[text encoder]──► vector [0.2, 0.8, ...]
[photo of dog with ball] ─[image encoder]► vector [0.21, 0.79, ...]
                                             ↑ close in cosine space
```

**For Multi-modal RAG:**
1. At indexing time: embed all images with the CLIP image encoder; embed all text with the CLIP text encoder.
2. At query time: embed the user's text query with the CLIP text encoder.
3. Run ANN search across the combined index — returns both relevant text chunks AND relevant images.
4. Pass top results (images + text) to a vision-language model (GPT-4V, LLaVA, Gemini) for generation.

**Limitations:** CLIP was trained on web images; performance degrades on domain-specific diagrams, medical imaging, or technical schematics without fine-tuning. Two structural weaknesses matter in interviews: the text encoder has a **77-token limit** (long queries/captions get truncated), and CLIP is **weak on dense text inside images** (slides, scanned pages, charts with labels) — it sees text as texture, not content.

**How CLIP compares to the broader multimodal embedding landscape:**

| Approach | Architecture | Strengths | Weaknesses | Best for | Storage / latency |
|---|---|---|---|---|---|
| **CLIP (ViT-L/14)** | Dual-encoder trained with contrastive (softmax) loss; one vector per image or text | Fast single-vector ANN search; huge ecosystem; strong zero-shot on natural images | 77-token text limit; weak on dense in-image text; web-photo bias | Natural images, product photos, short captions | 1 vector per image (~768d); cheapest queries |
| **SigLIP / SigLIP-2** | Dual-encoder with **sigmoid loss** (pairwise, no batch-wide softmax normalization) | Sigmoid loss scales better with batch size → better zero-shot accuracy than CLIP at the same size; SigLIP-2 adds improved localization and multilingual support | Still single-vector; still degrades on document-style images | Drop-in CLIP upgrade for general image retrieval | Same profile as CLIP — 1 vector per image |
| **ColPali / ColQwen2** | VLM backbone (PaliGemma / Qwen2-VL) emits **multi-vector embeddings over page-image patches**; ColBERT-style late interaction (MaxSim) at query time | SOTA document/PDF retrieval; **no OCR/layout/chunking pipeline** — index page screenshots directly; captures text + tables + figures + layout jointly | **Heavy multi-vector storage** (~1K patch vectors per page); MaxSim scoring costs more than one ANN lookup; needs a multi-vector-capable store (Vespa, Qdrant) | PDFs, slides, scanned docs where layout and embedded text carry meaning | 100–1000× vectors per page vs CLIP; mitigate with binary quantization + patch pooling; higher query latency |
| **Captioning-based indexing** | Multimodal LLM (GPT-4o, Claude) writes a caption/description per image; embed that **text** with your existing text embedding model | Simple; reuses the mature text stack (hybrid search, rerankers, filters); captions are human-auditable; prompt can be domain-tuned ("describe this schematic, list labeled components") | **Lossy** — you can only retrieve what the caption mentions; VLM cost per image at index time; caption quality bounds recall | Teams with a working text RAG stack adding moderate image volume | Text-vector storage only (cheapest); indexing cost = 1 VLM call per image |

**Rule of thumb:** natural images → SigLIP; PDF/slide/scanned-document retrieval → ColPali/ColQwen2; low image volume on an existing text pipeline → captioning-based indexing; CLIP remains the baseline everyone benchmarks against.

</details>

---

## Q3. How do you handle tables and structured data in a Multi-modal RAG system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Tables require special treatment because raw HTML/CSV doesn't embed well. Common approaches:

**Option 1 — Table-to-text:**
Convert tables to natural language summaries using an LLM:
> "The table shows Q3 revenue by region. North America: $2.1B (+8%), EMEA: $1.4B (+12%)..."
Then embed the summary as text.

**Option 2 — Table as image:**
Render the table as a PNG and embed with CLIP. Useful for complex visual layouts.

**Option 3 — Structured retrieval:**
Don't embed tables at all — route table queries to a Text-to-SQL pipeline that queries the underlying database directly.

**Option 4 — Hybrid:**
Embed the table header + column names for retrieval. At generation time, pass the full raw table to the LLM in the context.

**Best practice:** Use Option 3 for large, queryable datasets and Option 1 for small reference tables in documents.

</details>

---

## Q4. What are the key challenges in building a production Multi-modal RAG system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Challenge | Description | Solution |
|---|---|---|
| **Index heterogeneity** | Text, image, table embeddings live in different spaces | Use a shared embedding space (CLIP) or separate indexes with a fusion layer |
| **Chunking images** | Images don't chunk like text | Extract regions of interest (ROI) or use the full image |
| **PDF parsing** | PDFs mix text, tables, figures | Use specialized parsers (Unstructured.io, Azure Document Intelligence) |
| **VLM context limits** | Can't pass 20 images to GPT-4V | Rerank to top 2–3 images before generation |
| **Latency** | Image encoding is slower than text | Pre-compute image embeddings offline; use caching |
| **Evaluation** | Hard to evaluate image-grounded answers | Use VQA benchmarks; human eval for production |

</details>

---

## Q5. How would you architect a Multi-modal RAG system for a technical documentation chatbot that includes diagrams and code? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Ingestion pipeline:**
```
PDF/HTML docs
  ├── Text blocks    → text-embedding-3 → Pinecone (text index)
  ├── Code blocks    → CodeBERT / text-embedding → Pinecone (code index)
  ├── Diagrams/figs  → CLIP image encoder → Pinecone (image index)
  └── Tables         → LLM → text summary → Pinecone (text index)
```

**Query pipeline:**
```
User query
  ├── Text retrieval    → top-5 text chunks
  ├── Image retrieval   → top-2 diagrams (CLIP text encoder query)
  └── Code retrieval    → top-3 code snippets
          │
          ▼
     Reranker (filter irrelevant results)
          │
          ▼
  GPT-4V / Gemini 1.5 Pro (with images inline in prompt)
          │
          ▼
     Answer with text + referenced diagrams
```

**Key choices:**
- Use **GPT-4V or Gemini** — they handle interleaved text and image inputs natively.
- Store **original image URLs** alongside embeddings so the UI can display the source diagram.
- Build a **citation layer** — each part of the answer should cite which chunk or diagram it came from.
- Use **Unstructured.io** or **LlamaParse** for robust PDF parsing that preserves layout context.

**Hybrid text+image retrieval — query both indexes, fuse with weighted RRF, prompt a vision LLM:**

```python
import numpy as np
from sentence_transformers import SentenceTransformer

clip_model = SentenceTransformer("clip-ViT-L-14")        # queries the image index
text_model = SentenceTransformer("all-MiniLM-L6-v2")     # queries the text-chunk index

def weighted_rrf(ranked_lists: list[list[str]], weights: list[float], k: int = 60):
    """Merge ranked ID lists with weighted Reciprocal Rank Fusion."""
    scores = {}
    for ids, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def hybrid_retrieve(query: str, image_index, text_index, k: int = 5):
    # 1. CLIP *text* encoder against the image index (cross-modal search)
    q_clip = clip_model.encode(query)
    image_hits = image_index.search(q_clip, k=10)        # ranked image IDs

    # 2. Text embedding model against the text-chunk index
    q_text = text_model.encode(query)
    text_hits = text_index.search(q_text, k=10)          # ranked chunk IDs

    # 3. Fuse on RANKS, not raw scores — CLIP-space and text-space
    #    cosine similarities are not on a comparable scale.
    fused = weighted_rrf([text_hits, image_hits], weights=[0.6, 0.4])
    return [lookup(doc_id) for doc_id in fused[:k]]      # docs with type/url/text metadata

def answer(query: str, results: list[dict]) -> str:
    # 4. Assemble an interleaved multimodal prompt for a vision LLM
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:"}]
    for r in results:
        if r["type"] == "image":
            content.append({"type": "image_url", "image_url": {"url": r["url"]}})
            content.append({"type": "text",
                            "text": f"(Figure from {r['source']}, page {r['page']})"})
        else:
            content.append({"type": "text", "text": r["text"]})
    content.append({"type": "text",
                    "text": "Answer using only the context above. "
                            "Cite each chunk or figure you rely on."})
    return vision_llm.chat(messages=[{"role": "user", "content": content}])

results = hybrid_retrieve("How does the auth flow handle token refresh?",
                          image_index, text_index)
print(answer("How does the auth flow handle token refresh?", results))
```

Tuning notes: the RRF weights (0.6/0.4) are query-time knobs — raise the image weight for diagram-heavy corpora; enforce a per-modality quota (e.g., always include the top image) if text floods the fused list.

</details>

---

## Q6. How do you index and retrieve video content in a multimodal RAG system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Video requires decomposition into frames and transcripts before embedding and retrieval.

```
Video file (.mp4, .mov, etc.)
    │
    ├─ [Decompose into frames]
    │  ├─ Sample frames every N seconds (e.g., every 5s)
    │  └─ Embed frame-N with CLIP image encoder
    │      → Stored in image index with timestamp metadata
    │
    ├─ [Extract transcript]
    │  ├─ Whisper: speech-to-text
    │  └─ Embed transcript chunks with text-embedding-3
    │      → Stored in text index with timestamp metadata
    │
    └─ [Link frames to transcript]
       Join frame embeddings with corresponding transcript chunks
       (e.g., frame at 2:30s links to transcript "00:02:30 - ...user clicks...")
```

**Retrieval:**

```python
class VideoRAG:
    def index_video(self, video_path: str):
        """Index a video for multi-modal retrieval."""
        
        # Extract frames
        frames = extract_frames_every_n_seconds(video_path, interval=5)
        frame_embeddings = [clip_image_encoder(f) for f in frames]
        
        # Extract audio and transcribe
        audio = extract_audio(video_path)
        transcript = whisper.transcribe(audio)  # List of (timestamp, text) pairs
        
        # Embed transcript
        transcript_chunks = [chunk for ts, chunk in transcript]
        transcript_embeddings = [text_embedding_model(chunk) for chunk in transcript_chunks]
        
        # Store with cross-references
        for i, (frame_emb, frame) in enumerate(zip(frame_embeddings, frames)):
            timestamp = i * 5  # seconds
            self.vectorstore.add(
                embedding=frame_emb,
                metadata={"type": "video_frame", "video": video_path, "timestamp": timestamp},
                doc=frame
            )
        
        for ts, (trans_emb, text) in enumerate(zip(transcript_embeddings, transcript_chunks)):
            self.vectorstore.add(
                embedding=trans_emb,
                metadata={"type": "video_transcript", "video": video_path, "timestamp": ts},
                doc=text
            )
    
    def retrieve_video_content(self, query: str, k=5):
        """Retrieve both frames and transcript for a query."""
        
        # Text query → search both text and image indices
        text_results = self.vectorstore.similarity_search(query, k=k)
        
        # Get frames at same timestamps
        frame_results = []
        for result in text_results:
            if result.metadata["type"] == "video_transcript":
                timestamp = result.metadata["timestamp"]
                # Find corresponding frame
                frame = get_frame_at_timestamp(timestamp)
                frame_results.append({
                    "text": result.page_content,
                    "image": frame,
                    "timestamp": timestamp
                })
        
        return frame_results

# Usage
video_rag = VideoRAG()
video_rag.index_video("tutorial.mp4")
results = video_rag.retrieve_video_content("How do I set up the system?")
# Returns: [{"text": "First, open...", "image": <PIL image>, "timestamp": 12}, ...]
```

</details>

---

## Q7. What is early fusion vs late fusion for multimodal retrieval? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Three strategies for combining multi-modal data in retrieval:

**Early Fusion — Combine before retrieval:**
```
Text embedding     Image embedding
      ↓                    ↓
      └─ Concatenate ─→ Single combined embedding
                             │
                             ▼
                      ANN search (single index)
                             │
                             ▼
                    Top-k results (mixed types)
```

**Late Fusion — Retrieve separately, merge results:**
```
Text query          Image query (from image)
      ↓                    ↓
Text retrieval      Image retrieval
(text index)       (image index)
      ↓                    ↓
 Top-k text        Top-k images
      └──────┬─────────────┘
             ▼
      RRF or learned ranking
             ▼
     Final merged top-k
```

**Hierarchical / two-stage fusion — cheap broad retrieval, then expensive multimodal precision:**
```
Stage 1 (cheap, recall-oriented)
  Text retrieval over captions / OCR text / page text → top-100 candidates
             ▼
Stage 2 (expensive, precision-oriented)
  Multimodal rerank of candidates only:
    - VLM or cross-encoder scores (query, image) pairs, OR
    - page-level → patch-level (ColPali: retrieve pages, then
      MaxSim over patch vectors to score/rerank)
             ▼
       Final top-k (3–5)
```

**Comparison:**

| Aspect | Early Fusion | Late Fusion | Hierarchical / Two-stage |
|--------|-------------|-----------|--------------------------|
| **Index size** | Single large index | Multiple smaller indexes | Cheap first-stage index + heavy second-stage artifacts (patch vectors, rerank model) |
| **Query latency** | Single ANN search | Parallel searches, then merge | Two sequential stages — highest latency, but stage 2 only touches ~100 candidates |
| **Recall** | Bounded by shared-space quality (CLIP misses what it can't embed) | Best raw recall — each modality searched with its strongest model | Bounded by stage-1 recall: if the caption/OCR misses it, the reranker never sees it |
| **Infra complexity** | Lowest — one index, one encoder pair | Medium — per-modality indexes + fusion layer to tune | Highest — two pipelines, candidate handoff, rerank serving |
| **Cross-modal search** | Native (embeddings share space) | Requires cross-modal bridge | Stage 1 is usually text-only; stage 2 supplies the cross-modal signal |
| **Score handling** | One score scale | Scores NOT comparable across modalities — must use rank fusion or normalize | Stage 2 produces a single comparable score for all candidates |
| **Ranking control** | Fixed at indexing time | Tunable at query time (RRF weights) | Tunable at both stages (candidate count k1, rerank cutoff) |

**Two failure modes interviewers probe:**

1. **Modality imbalance (text dominates):** in a shared space, text-to-text similarities are systematically higher than text-to-image similarities (the "modality gap" — image and text embeddings occupy separate cones in CLIP space). A naive merged index returns mostly text. Mitigations: retrieve per-modality quotas (e.g., always take top-2 images), apply a per-modality score offset/temperature, or use late fusion with explicit weights.
2. **Score normalization across modalities:** cosine scores from different encoders (CLIP space vs. a text embedding model) live on different scales and distributions — never sum raw scores. Use rank-based fusion (RRF), per-modality z-score/min-max normalization, or a learned fusion layer trained on click/relevance data.

**Early fusion example (CLIP-based):**
```python
# All modalities embed to same CLIP space
text_embedding = clip_text_encoder("What is this diagram?")
image_embedding = clip_image_encoder(diagram_image)
# Embeddings are comparable; single ANN index
```

**Late fusion example (FAISS hybrid):**
```python
# Separate indexes
text_results = faiss_text_index.search(query_embedding, k=10)
image_results = faiss_image_index.search(clip_image_query, k=10)
# Merge via RRF
merged = reciprocal_rank_fusion([text_results, image_results], weights=[0.6, 0.4])
```

**Recommendation:**
- Use **early fusion** when modalities are naturally comparable (text + images in CLIP space).
- Use **late fusion** for heterogeneous data (structured tables + unstructured text + images).
- Use **hierarchical/two-stage** when precision matters and a VLM rerank is affordable — or when using ColPali-style page→patch retrieval over document corpora.

</details>

---

## Q8. How do you parse and embed PDFs containing mixed text, tables, and images? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

PDFs are notoriously tricky; layout, fonts, and embedded objects break naive text extraction.

```python
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import io

class PDFMultimodalParser:
    def parse_pdf(self, pdf_path: str) -> dict:
        """Extract text, tables, and images from PDF."""
        
        # Use Unstructured.io to preserve layout
        elements = partition_pdf(
            pdf_path,
            extract_images_in_pdf=True,  # Extract images
            infer_table_structure=True,  # Parse table structure
            strategy="hi_res",  # Use model-based layout analysis (slower but better)
        )
        
        parsed = {
            "text_chunks": [],
            "tables": [],
            "images": []
        }
        
        for element in elements:
            # Classify by element type
            if element.category == "Table":
                # Tables: convert to text summary + keep raw for SQL queries
                table_summary = self.summarize_table(element.text)
                parsed["tables"].append({
                    "raw_html": element.text,
                    "summary": table_summary,
                    "metadata": {"page": element.metadata.page_number}
                })
            
            elif element.category == "Figure" or element.category == "Image":
                # Images: extract and store
                image_data = element.metadata.image_base64  # If available
                if image_data:
                    parsed["images"].append({
                        "base64": image_data,
                        "metadata": {"page": element.metadata.page_number}
                    })
            
            else:  # Text, heading, etc.
                parsed["text_chunks"].append({
                    "text": element.text,
                    "type": element.category,
                    "metadata": {"page": element.metadata.page_number}
                })
        
        return parsed
    
    def summarize_table(self, table_html: str) -> str:
        """Convert table to natural language summary."""
        
        prompt = f"""Summarize this table in 1-2 sentences:
{table_html}

Summary:"""
        
        summary = llm.invoke(prompt)
        return summary.content
    
    def embed_multimodal_pdf(self, pdf_path: str, vectorstore):
        """Parse, embed, and index PDF elements."""
        
        parsed = self.parse_pdf(pdf_path)
        
        # Embed text chunks
        for chunk in parsed["text_chunks"]:
            embedding = text_embedding_model.embed(chunk["text"])
            vectorstore.add(
                embedding=embedding,
                metadata={**chunk["metadata"], "type": "text", "source": pdf_path},
                text=chunk["text"]
            )
        
        # Embed table summaries
        for table in parsed["tables"]:
            embedding = text_embedding_model.embed(table["summary"])
            vectorstore.add(
                embedding=embedding,
                metadata={**table["metadata"], "type": "table", "source": pdf_path, "raw_html": table["raw_html"]},
                text=table["summary"]
            )
        
        # Embed images
        for image in parsed["images"]:
            image_obj = Image.open(io.BytesIO(base64.b64decode(image["base64"])))
            embedding = clip_image_encoder(image_obj)
            vectorstore.add(
                embedding=embedding,
                metadata={**image["metadata"], "type": "image", "source": pdf_path},
                image=image_obj
            )
        
        print(f"Indexed {len(parsed['text_chunks'])} text, {len(parsed['tables'])} tables, {len(parsed['images'])} images")
```

</details>

---

## Q9. What evaluation frameworks exist for multimodal RAG? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Evaluating multi-modal RAG is harder than text-only because answers span images, text, and structured data.

**Metric taxonomy:**

| Category | Metrics | Tools |
|----------|---------|-------|
| **Visual QA** | Accuracy on image captioning/VQA | NVSA, MSCOCO Captions, Flickr30K |
| **Cross-modal retrieval** | Recall@k for image-text pairs | MSCOCO (text→image), Flickr (image→text) |
| **Table QA** | Exact match, F1 on table queries | WikiTableQuestions, SPIDER (SQL) |
| **Citation accuracy** | % of claims cited to source (text/image/table) | Custom annotation |
| **Semantic consistency** | Does answer match all retrieved modalities? | LLM-based consistency scoring |

```python
from torchmetrics.retrieval import RetrievalRecall
import numpy as np

class MultimodalRAGEvaluator:
    def __init__(self, test_set):
        # test_set: [(query, gold_text, gold_images, gold_table)]
        self.test_set = test_set
    
    def evaluate_cross_modal_recall(self, system):
        """Can text queries retrieve relevant images?"""
        
        recalls = []
        for query, _, gold_images, _ in self.test_set:
            # System retrieves
            retrieved = system.retrieve(query)
            
            # Check if any retrieved image is in gold_images
            retrieved_images = [r for r in retrieved if r["type"] == "image"]
            gold_image_ids = set(img["id"] for img in gold_images)
            
            hit = any(img["id"] in gold_image_ids for img in retrieved_images)
            recalls.append(float(hit))
        
        return np.mean(recalls)
    
    def evaluate_citation_accuracy(self, system):
        """Are claims properly cited to their sources?"""
        
        correct_citations = 0
        total_claims = 0
        
        for query, _, _, _ in self.test_set:
            answer = system.answer(query)  # Generates answer + cites sources
            
            # Parse citations from answer
            # E.g., "Revenue was $10M [Text:chunk_5, Image:diagram_2]"
            claims_with_citations = extract_claims_and_citations(answer)
            
            for claim, citations in claims_with_citations:
                total_claims += 1
                
                # Verify each citation is valid
                all_valid = all(
                    verify_claim_in_source(claim, citation)
                    for citation in citations
                )
                
                if all_valid:
                    correct_citations += 1
        
        return correct_citations / total_claims if total_claims > 0 else 0
    
    def evaluate_consistency(self, system):
        """Does answer align with all retrieved modalities?"""
        
        consistency_scores = []
        for query, _, _, _ in self.test_set:
            retrieved = system.retrieve(query)
            answer = system.answer(query)
            
            # LLM judges: "Is the answer consistent with all retrieved text/images/tables?"
            consistency_prompt = f"""Query: {query}
Answer: {answer}

Retrieved text: {[r['text'] for r in retrieved if r['type'] == 'text']}
Retrieved images: [show images]
Retrieved table: [table summary]

Rate consistency (0-1): Are all claims consistent with retrieved content?"""
            
            score = float(llm.invoke(consistency_prompt).content)
            consistency_scores.append(score)
        
        return np.mean(consistency_scores)

# Benchmark datasets
benchmarks = {
    "MSCOCO Captions": ("Image captioning", 5K test images),
    "Flickr30K": ("Image-text retrieval", 1K queries),
    "WikiTableQuestions": ("Table QA", 22K questions),
    "NVSA": ("Visual semantic alignment", 7K image-text pairs),
}
```

</details>

---

## Q10. How do you fine-tune CLIP for domain-specific multimodal retrieval? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

CLIP is trained on web images; fine-tuning on domain data improves retrieval significantly (5–20pp gains).

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW

class CLIPDomainFineTuning:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def prepare_contrastive_dataset(self, domain_image_text_pairs):
        """
        Args:
            domain_image_text_pairs: List of (image_path, matching_text_captions)
        
        For each image, create positive (image, caption) pairs and 
        negative pairs from other images.
        """
        
        training_data = []
        
        for i, (image_path, captions) in enumerate(domain_image_text_pairs):
            image = Image.open(image_path)
            
            # Positive pairs: this image with all its captions
            for caption in captions:
                training_data.append({
                    "image": image,
                    "text": caption,
                    "label": 1  # positive
                })
            
            # Negative pairs: this image with captions from other images
            for j in range(max(0, i-2), min(len(domain_image_text_pairs), i+3)):
                if i != j:
                    _, other_captions = domain_image_text_pairs[j]
                    for caption in other_captions[:1]:  # Limit negatives
                        training_data.append({
                            "image": image,
                            "text": caption,
                            "label": 0  # negative
                        })
        
        return training_data
    
    def fine_tune(self, training_data, num_epochs=3, batch_size=32):
        """Contrastive loss fine-tuning."""
        
        from torch.utils.data import DataLoader, Dataset
        
        class ContrastiveDataset(Dataset):
            def __init__(self, data, processor):
                self.data = data
                self.processor = processor
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                image_inputs = self.processor(
                    images=example["image"],
                    return_tensors="pt"
                )
                text_inputs = self.processor(
                    text=example["text"],
                    return_tensors="pt"
                )
                
                return {
                    "image_pixel_values": image_inputs["pixel_values"].squeeze(),
                    "text_input_ids": text_inputs["input_ids"].squeeze(),
                    "label": example["label"]
                }
        
        dataset = ContrastiveDataset(training_data, self.processor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in loader:
                # Forward pass
                image_outputs = self.model.vision_model(
                    pixel_values=batch["image_pixel_values"]
                )
                text_outputs = self.model.text_model(
                    input_ids=batch["text_input_ids"]
                )
                
                # Project to embedding space
                image_embeds = self.model.visual_projection(image_outputs.pooler_output)
                text_embeds = self.model.text_projection(text_outputs.pooler_output)
                
                # Contrastive loss (simplified)
                batch_labels = batch["label"]
                logits = torch.matmul(image_embeds, text_embeds.T) / 0.07
                loss = torch.nn.functional.cross_entropy(logits, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")
    
    def evaluate_domain_performance(self, domain_test_pairs):
        """Measure improvement on domain-specific queries."""
        
        recalls = []
        
        for image, captions in domain_test_pairs:
            # Encode query image
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_embeds = self.model.vision_model(**image_inputs).pooler_output
            image_embeds = self.model.visual_projection(image_embeds)
            
            # Encode text options
            text_inputs = self.processor(text=captions, return_tensors="pt", padding=True)
            text_embeds = self.model.text_model(**text_inputs).pooler_output
            text_embeds = self.model.text_projection(text_embeds)
            
            # Rank: top-k captions should be the correct ones
            logits = torch.matmul(image_embeds, text_embeds.T)
            top_k_indices = torch.argsort(logits[0], descending=True)[:5]
            
            # Recall@5: first gold caption in top 5?
            recall = 1.0 if 0 in top_k_indices else 0.0
            recalls.append(recall)
        
        return np.mean(recalls)

# Pitfalls and mitigation
pitfalls = {
    "Overfitting to domain": {
        "problem": "Model memorizes training pairs, fails on new images",
        "solution": "Data augmentation (crop, rotate), validation set monitoring"
    },
    "Catastrophic forgetting": {
        "problem": "Fine-tuning on domain data destroys general capabilities",
        "solution": "Use LoRA (low-rank adaptation) instead of full fine-tuning; mix general + domain data"
    },
    "Class imbalance": {
        "problem": "Some domain categories have few examples",
        "solution": "Weighted sampling, contrastive hard-negative mining"
    }
}

# Typical results:
# ├─ Base CLIP on domain: 45% recall@5
# ├─ + Fine-tune 5K domain pairs: 72% recall@5  (+27pp)
# └─ + LoRA (20 adapters): 74% recall@5, retains 98% general performance
```

**Recommendation:**
Use **LoRA fine-tuning** over full fine-tuning to avoid forgetting. Target 5–10K domain image-caption pairs for a 15–20pp gain.


</details>
---

## Q11. How do you estimate and reduce the cost of CLIP inference at index time and multi-modal embedding storage at scale for a production Multi-modal RAG system? [Intermediate]

<details>
<summary>?? Show Answer</summary>

**Answer:**

**Cost breakdown for Multi-modal RAG:**

| Component | Cost | Notes |
|-----------|------|-------|
| CLIP inference (index time) | .001-0.005/image | Batch processing cheaper |
| Embedding storage (768 dims) | .001/1K embeddings/month | Quantization cuts this 4-10x |
| Multimodal retrieval latency | 100-200ms | Acceptable for most use cases |

**Cost optimization:**

1. **Batch CLIP inference** - Process images in batches (100-1000 per batch) instead of individually. Reduces cost by 50-70%.

2. **Image quantization** - Quantize CLIP embeddings from float32 to int8 or binary, cutting storage by 4-10x.

3. **Selective indexing** - Only index high-value images (>100 views/month), skip low-traffic images.

4. **Lazy embedding** - Embed on-demand for rarely-queried images instead of upfront.

5. **Region-based indexing** - Index only relevant image regions (via bounding boxes), not entire image.

**Example cost structure (1M images):**

Baseline: 1M � .003 inference + 1M � 768 � 4 bytes = .5K indexing + .7K storage = .2K/month.

Optimized (batching + quantization + selective): 30% image indexing, int8 storage = .5K + .2K = .7K/month (76% savings).

</details>

---

## Q12. How do adversarial image inputs and cross-modal injection attacks threaten Multi-modal RAG, and what defences apply at the embedding, retrieval, and generation layers? [Advanced]

<details>
<summary>?? Show Answer</summary>

**Answer:**

**Attack 1: Adversarial image embedding**

Attacker crafts an adversarial image (imperceptible noise added to legitimate image) that embeds near many innocent queries but contains hidden malicious text overlay.

**Attack 2: Cross-modal injection**

Attacker injects text into images (via OCR-detectable overlay) that, when retrieved, influences the LLM's answer.

**Defences:**

1. **Embedding robustness** - Fine-tune CLIP on adversarial examples to resist evasion.

2. **Image sanitization** - Detect and remove text overlays, anomalous regions via image quality checks.

3. **Multi-modal consistency** - Verify that text and images describe the same concept (cross-modal coherence check).

4. **Confidence thresholding** - Flag low-confidence retrievals (embedding similarity <0.85) for human review.

5. **OCR pre-filtering** - Extract and validate any OCR text from images before use.

6. **Ensemble retrieval** - Combine image retrieval with text retrieval; require agreement.

7. **Adversarial perturbation analysis** - Test if image embeddings are stable under small perturbations.

Combining these defences prevents adversarial images from poisoning the retrieval layer.

</details>

---

## Real-World Applications

| Application | Domain | Why Multimodal RAG Fits |
|---|---|---|
| E-commerce visual + text search (e.g., Pinterest, Amazon visual search) | Retail | Users upload a photo to find similar products; multimodal retrieval matches image embeddings against catalogue images + product descriptions |
| Radiology report assistant | Healthcare | Retrieves similar chest X-rays and their associated radiologist reports to assist with differential diagnosis |
| Document understanding / invoice processing (e.g., AWS Textract-based apps) | Finance / Operations | Mixed PDFs contain tables, charts, and text; multimodal RAG retrieves the right page-image and interprets layout-sensitive content |
| Manufacturing defect detection Q&A | Industrial / QA | Engineers query visual defect libraries: "show me welds with this type of crack" retrieves annotated image-text pairs |
| Educational content platform (textbook Q&A) | EdTech | Students ask questions about diagrams or charts; image-text retrieval fetches the relevant figure and its surrounding explanation |
