# Multimodal Embeddings: Aligning Text, Images, and Beyond

> How cross-modal embedding models work, which models to use, and how to build RAG systems that retrieve and reason across text and images.

---

## What Are Multimodal Embeddings?

A multimodal embedding model maps inputs from different modalities (text, image, audio, video) into a *shared vector space* where semantically related content is close regardless of its modality. The canonical example: a photo of a golden retriever and the text "golden retriever dog" should be near-neighbors in the embedding space, even though one is a tensor of pixels and the other is a sequence of tokens.

This shared space is what enables cross-modal retrieval: a text query can retrieve relevant images, an image can retrieve relevant text passages, or a mixed query (image + caption) can retrieve related documents.

---

## The Core Models

### CLIP (Contrastive Language-Image Pre-training)

OpenAI's 2021 model trained on 400M (image, alt-text) pairs with a contrastive loss: pull matching pairs together, push non-matching pairs apart.

```
Training:
  (image₁, text₁) → [similar]
  (image₁, text₂) → [dissimilar] ← pushed apart in embedding space

Architecture:
  Image Encoder: Vision Transformer (ViT-B/32, ViT-L/14, ViT-H/14)
  Text Encoder:  Transformer (GPT-2 style)
  Both produce 512/768/1024-dim normalized vectors
```

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed an image
image = Image.open("diagram.png")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_emb = model.get_image_features(**inputs)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)  # normalize

# Embed a text query
texts  = ["architecture diagram", "bar chart", "photograph of a person"]
inputs = processor(text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    text_embs = model.get_text_features(**inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

# Cross-modal similarity
similarities = (text_embs @ image_emb.T).squeeze()  # [3]
print(dict(zip(texts, similarities.tolist())))
```

### ImageBind

Meta's 2023 model that binds *six* modalities into a single shared space: image, text, audio, depth, thermal, IMU (inertial). The key trick: train all modalities to be aligned with images, and transitivity gives you cross-modal alignment between non-image pairs (e.g., audio ↔ text via the image anchor).

```python
from imagebind import data, ModalityType
from imagebind.models import imagebind_model

model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()

inputs = {
    ModalityType.TEXT:  data.load_and_transform_text(["a dog barking"], device="cpu"),
    ModalityType.AUDIO: data.load_and_transform_audio_data(["bark.wav"], device="cpu"),
    ModalityType.VISION: data.load_and_transform_vision_data(["dog.jpg"], device="cpu"),
}

with torch.no_grad():
    embeddings = model(inputs)
# embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO], etc.
# All in the same 1024-dim space
```

### BLIP-2 / LLaVA (Vision-Language Models)

Rather than a shared embedding space, these models produce *tokens* from the vision encoder that a language model can attend to. They enable captioning, VQA, and document understanding but are less suited to embedding-based retrieval — the output is generated text, not a fixed vector.

| Use Case | Use CLIP/ImageBind | Use BLIP-2/LLaVA |
|----------|-------------------|-----------------|
| Cross-modal retrieval (image→text, text→image) | ✓ | — |
| Image captioning / description generation | — | ✓ |
| Visual QA ("what does this chart say?") | — | ✓ |
| Large-scale vector index over images | ✓ | — |

---

## Embedding Fusion Strategies

### Early Fusion

Combine raw inputs before encoding. Only works for modalities with compatible representations (e.g., text + structured metadata).

```
[Image pixels + Text caption] → Single multimodal encoder → Vector
```

Pros: single model, simpler pipeline. Cons: requires modality-aware architecture; can't reuse standard encoders.

### Late Fusion (Recommended for RAG)

Encode each modality separately, then combine the resulting vectors.

```
Image  → Image encoder  → image_emb  ─┐
Text   → Text encoder   → text_emb   ─┤─ combine → final_emb
                                       │
Combination strategies:
  - Average: (image_emb + text_emb) / 2
  - Weighted: α·image_emb + (1-α)·text_emb  [α tuned per domain]
  - Concatenation + linear projection: W · [image_emb; text_emb]
```

```python
def embed_document_with_image(text: str, image_path: str, alpha: float = 0.5) -> np.ndarray:
    text_emb  = clip_text_encoder(text)
    image_emb = clip_image_encoder(Image.open(image_path))
    # Weighted average in the shared CLIP space
    fused = alpha * image_emb + (1 - alpha) * text_emb
    return fused / np.linalg.norm(fused)
```

### Asymmetric Late Fusion (Different Encoders per Modality)

Use a domain-tuned text encoder (e.g., a fine-tuned BGE model) for text and CLIP's vision encoder for images. Align the spaces with a learned projection layer.

```python
class CrossModalProjector(torch.nn.Module):
    def __init__(self, text_dim: int = 768, image_dim: int = 512, shared_dim: int = 512):
        super().__init__()
        self.text_proj  = torch.nn.Linear(text_dim, shared_dim)
        self.image_proj = torch.nn.Linear(image_dim, shared_dim)
    
    def forward(self, text_emb=None, image_emb=None):
        if text_emb is not None:
            return torch.nn.functional.normalize(self.text_proj(text_emb), dim=-1)
        return torch.nn.functional.normalize(self.image_proj(image_emb), dim=-1)
```

---

## Building a Multimodal RAG Pipeline

```
INDEXING:
  For each document:
    ├─ Extract text chunks     → text_emb  (BGE/E5/CLIP text encoder)
    ├─ Extract images/figures  → image_emb (CLIP image encoder)
    └─ Extract tables          → table_text → text_emb
    Store all vectors in the same index (with modality metadata)

RETRIEVAL:
  Query:
    ├─ If text query  → text_emb → ANN search → text + image results
    ├─ If image query → image_emb → ANN search → text + image results
    └─ If mixed       → fuse embs  → ANN search → mixed results

GENERATION:
    Assemble context:
    ├─ Text chunks: include as text
    ├─ Images: include as base64 or pass to vision model for caption
    └─ Tables: include as Markdown
    → Generate answer with LLM (vision model if images included)
```

```python
import anthropic
import base64

def multimodal_rag(query: str, image_query_path: str = None) -> str:
    client = anthropic.Anthropic()
    
    # 1. Embed query (text + optional image)
    query_emb = embed_query(query, image_path=image_query_path)
    
    # 2. Retrieve top-k mixed results
    results = vector_db.search(query_emb, k=5, include_modality=True)
    
    # 3. Build context message (handle both text and image results)
    content = []
    for r in results:
        if r["modality"] == "text":
            content.append({"type": "text", "text": r["content"]})
        elif r["modality"] == "image":
            img_data = base64.standard_b64encode(open(r["path"], "rb").read()).decode()
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": img_data,
            }})
    
    content.append({"type": "text", "text": f"\nQuestion: {query}"})
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text
```

---

## Retrieval Metrics for Multimodal

Standard text RAG metrics (Recall@k, NDCG) apply, but you need modality-aware variants:

| Metric | What It Measures |
|--------|-----------------|
| **Text-Text Recall@k** | Standard: are relevant text chunks retrieved? |
| **Text-Image Recall@k** | Cross-modal: does a text query retrieve relevant images? |
| **Image-Text Recall@k** | Cross-modal: does an image query retrieve relevant text? |
| **CLIP Score** | Cosine similarity between image and its retrieved text description (0–1) |
| **R@1, R@5, R@10** | Standard image-text retrieval benchmark format (MS-COCO, Flickr30k) |

```python
def cross_modal_recall_at_k(queries, ground_truth_images, index, k=5):
    hits = 0
    for query, gt_image_id in zip(queries, ground_truth_images):
        results = index.search(embed_text(query), k=k)
        if gt_image_id in [r["id"] for r in results]:
            hits += 1
    return hits / len(queries)
```

---

## Embedding Space Alignment: The Key Challenge

Different encoders produce embeddings in *incompatible spaces* — you cannot directly compare a CLIP vector with a BGE vector. Alignment strategies:

1. **Use a single multi-modal model (CLIP, ImageBind)** — both modalities land in the same space by design.
2. **Fine-tune a projection** — train a linear layer to map your text encoder's space into CLIP's image space using paired (image, description) examples.
3. **MAGICLENS / VLP alignment** — use a vision-language pre-training stage to align your custom encoders.

Rule of thumb: for most RAG use cases, CLIP is sufficient. Only invest in custom alignment if CLIP's retrieval quality is insufficient for your domain (medical imaging, satellite imagery, legal diagrams).

---

## Model Comparison

| Model | Modalities | Embedding Dim | Best For | Limitation |
|-------|-----------|--------------|---------|-----------|
| CLIP ViT-B/32 | Image + Text | 512 | General-purpose cross-modal retrieval | Weak on dense text in images |
| CLIP ViT-L/14 | Image + Text | 768 | Higher quality, same API | 2× memory vs. ViT-B/32 |
| ImageBind | 6 modalities | 1024 | Audio+image+text alignment | Large model; harder to deploy |
| SigLIP | Image + Text | 768 | Better recall than CLIP on zero-shot | Google proprietary training data |
| E5-mistral (text-only with vision projection) | Text + Image (via projection) | 4096 | High-quality text, added vision | Complex deployment |

---

## Key Takeaways

1. **CLIP is the default** for image-text retrieval — widely supported, easy to deploy.
2. **Late fusion** (encode separately, combine vectors) is simpler and more flexible than early fusion for RAG.
3. **Keep modality as metadata** in your vector index — it enables modality-filtered retrieval.
4. **Pass images directly to the LLM** (Claude's vision API) rather than captioning them first when answer quality matters — captioning loses information.
5. **Alignment matters** — mixing vectors from different encoders without a projection layer will silently produce wrong retrieval results.

---

## Interview Q&A

**Q: How does CLIP's training enable cross-modal retrieval without labeled image-text pairs?**

CLIP is trained with a contrastive objective on 400M web-scraped (image, alt-text) pairs — these are naturally co-occurring, not manually labeled. The loss function pulls each image embedding close to its paired text embedding and pushes it away from the other 255 images in the same batch (InfoNCE/NT-Xent loss). After training, both image and text encoders produce vectors in the same 512-dim space, so cosine similarity is directly comparable across modalities. The result: you can embed a text query and retrieve images purely by ANN search, no classification head required.

---

**Q: What is "late fusion" vs. "early fusion" and when should you use each in multimodal RAG?**

Early fusion combines raw inputs before encoding (requires a modality-aware architecture that can process both at once), while late fusion encodes each modality independently and then combines the resulting vectors. Late fusion is strongly preferred for RAG because: (1) you can reuse best-in-class encoders for each modality (e.g., BGE for text, CLIP for images) rather than one compromised joint model; (2) you can index text and image chunks in the same vector database and serve mixed retrieval with a single ANN query; (3) if you add a new modality later (e.g., audio), you add a new encoder without retraining the entire system. Early fusion is worth considering only when modalities are tightly coupled and cannot be meaningfully understood in isolation.

---

**Q: How would you index a corpus of 10,000 PDF documents that contain both text and embedded figures?**

Parse each PDF (PyMuPDF/pdfplumber) to extract: (a) text blocks → chunk by section → embed with a text encoder; (b) figures/images → extract as PNG → embed with CLIP image encoder; (c) tables → linearize to Markdown → embed with text encoder. Store all vectors in a single index with a `modality` metadata field (`"text"`, `"image"`, `"table"`). At query time, embed the query as text, search the entire index (all modalities), and let ANN similarity decide which chunks to retrieve. For image results, either pass the raw image to Claude's vision API or run BLIP-2/LLaVA to generate a caption for inclusion in the text prompt. Flag any figure with no surrounding caption text for manual review — those are the hardest for a text query to surface via semantic similarity alone.
