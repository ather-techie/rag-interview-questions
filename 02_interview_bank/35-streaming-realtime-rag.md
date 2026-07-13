# Streaming / Real-Time RAG

> How to keep a RAG index continuously up-to-date using event streams and CDC pipelines, reducing the freshness window from hours/days to seconds.

---

## What is Streaming RAG?

Standard RAG indexes are built in batch: documents are processed, embedded, and loaded into the vector DB on a schedule (hourly, nightly). During the interval between batches, new documents are invisible to retrieval — the system answers questions about an outdated world.

Streaming RAG replaces the batch pipeline with a continuous pipeline driven by event streams. Every document creation, update, or deletion triggers an immediate index update, bringing the freshness window down to seconds.

```
Batch RAG (standard):
  Documents ──► Batch Pipeline (nightly) ──► Vector Index (stale)

Streaming RAG:
  Documents ──► Event Stream ──► Streaming Pipeline (seconds) ──► Vector Index (fresh)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                               │
│  ├─ Database (Postgres CDC / Debezium)                      │
│  ├─ APIs (webhooks, polling)                                │
│  └─ File systems (S3 event notifications)                   │
└─────────────────────────────┬───────────────────────────────┘
                              │  events (create/update/delete)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  EVENT STREAM (Kafka / Kinesis / Pub/Sub)                   │
│  Topic: rag-document-events                                 │
│  Message: {doc_id, operation, content, timestamp}           │
└─────────────────────────────┬───────────────────────────────┘
                              │  consume
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STREAM PROCESSOR (Kafka Consumer / Faust / Bytewax)        │
│  ├─ Parse & validate                                        │
│  ├─ Chunk text                                              │
│  ├─ Embed (batch within window)                             │
│  └─ Upsert / delete vectors                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  VECTOR INDEX (Qdrant / Weaviate / Pinecone)                │
│  Always reflects latest document state                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Responsibility |
|---|---|
| CDC / Event Source Connector | Captures create/update/delete events from databases, APIs, or file systems (e.g., Debezium reading the Postgres WAL) |
| Stream Processor | Consumes the event topic, parses/validates payloads, chunks text, and triggers embedding |
| Incremental Embedder | Embeds only the changed chunks (micro-batched) instead of re-embedding the full corpus |
| Live Index Upserter | Applies delete-then-insert or versioned upserts so the vector index never serves stale chunks |
| Freshness-aware Retriever | Tracks per-document freshness lag and can enforce a max-staleness SLO on query results |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| CDC | Debezium, AWS DMS, Postgres logical replication |
| Event streaming | Kafka, AWS Kinesis, Google Pub/Sub, Redis Streams (lighter-weight alternative) |
| Stream processing | Kafka Streams, Apache Flink, Spark Structured Streaming, Faust/Bytewax |
| Vector index (incremental upsert) | Qdrant, Weaviate, Pinecone |

---

## Implementation: Kafka Producer (Document Side)

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def publish_document_event(doc_id: str, operation: str, content: str = None):
    """
    operation: "create" | "update" | "delete"
    """
    event = {
        "doc_id":    doc_id,
        "operation": operation,
        "content":   content,  # None for deletes
        "timestamp": time.time(),
    }
    producer.send("rag-document-events", event)
    producer.flush()

# Called by your CMS / database trigger / webhook handler
publish_document_event("doc:12345", "update", "New policy effective 2026-07-01...")
publish_document_event("doc:9999",  "delete")
```

---

## Implementation: Stream Processor (Indexing Side)

```python
from kafka import KafkaConsumer
from sentence_transformers import SentenceTransformer
import json
import time

EMBED_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def process_event(event: dict, vector_db):
    doc_id    = event["doc_id"]
    operation = event["operation"]
    
    if operation == "delete":
        # Delete all vectors for this document
        vector_db.delete(filter={"doc_id": {"$eq": doc_id}})
        return
    
    if operation in ("create", "update"):
        # On update: delete old vectors first, then re-insert
        if operation == "update":
            vector_db.delete(filter={"doc_id": {"$eq": doc_id}})
        
        chunks     = chunk_text(event["content"])
        embeddings = EMBED_MODEL.encode(chunks, normalize_embeddings=True, batch_size=32)
        
        vectors = [
            {
                "id":     f"{doc_id}:chunk:{i}",
                "values": emb.tolist(),
                "metadata": {
                    "doc_id":    doc_id,
                    "chunk_idx": i,
                    "text":      chunk,
                    "timestamp": event["timestamp"],
                },
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        vector_db.upsert(vectors=vectors)


def run_indexing_consumer(vector_db):
    consumer = KafkaConsumer(
        "rag-document-events",
        bootstrap_servers=["localhost:9092"],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="rag-indexing-group",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    
    print("Streaming indexer started.")
    for message in consumer:
        event = message.value
        try:
            process_event(event, vector_db)
            print(f"Processed {event['operation']} for {event['doc_id']}")
        except Exception as e:
            print(f"Error processing {event['doc_id']}: {e}")
            # In production: send to dead-letter queue (DLQ) for retry
```

---

## Micro-Batching for Embedding Efficiency

Calling the embedding model once per document is expensive. Micro-batch events within a short window before embedding:

```python
import asyncio
from collections import defaultdict

class MicroBatchIndexer:
    def __init__(self, vector_db, batch_size: int = 32, flush_interval: float = 0.5):
        self.vector_db      = vector_db
        self.batch_size     = batch_size
        self.flush_interval = flush_interval
        self.pending        = []
        self._lock          = asyncio.Lock()
    
    async def enqueue(self, event: dict):
        async with self._lock:
            self.pending.append(event)
            if len(self.pending) >= self.batch_size:
                await self._flush()
    
    async def _flush(self):
        if not self.pending:
            return
        batch, self.pending = self.pending[:], []
        
        # Gather all chunks across the batch
        all_chunks  = []
        all_meta    = []
        for event in batch:
            if event["operation"] == "delete":
                self.vector_db.delete(filter={"doc_id": {"$eq": event["doc_id"]}})
                continue
            if event["operation"] == "update":
                self.vector_db.delete(filter={"doc_id": {"$eq": event["doc_id"]}})
            
            chunks = chunk_text(event.get("content", ""))
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({"doc_id": event["doc_id"], "chunk_idx": i, "text": chunk})
        
        if not all_chunks:
            return
        
        # Single batch embedding call
        embeddings = EMBED_MODEL.encode(all_chunks, normalize_embeddings=True, batch_size=64)
        vectors = [
            {
                "id":     f"{m['doc_id']}:chunk:{m['chunk_idx']}",
                "values": emb.tolist(),
                "metadata": m,
            }
            for m, emb in zip(all_meta, embeddings)
        ]
        self.vector_db.upsert(vectors=vectors)
    
    async def periodic_flush(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            async with self._lock:
                await self._flush()
```

---

## Change Data Capture (CDC) with Debezium

For databases (Postgres, MySQL), use CDC to capture row-level changes without application-level instrumentation:

```yaml
# Debezium connector config (Kafka Connect)
name: postgres-rag-connector
config:
  connector.class: io.debezium.connector.postgresql.PostgresConnector
  database.hostname: postgres
  database.port: 5432
  database.user: replicator
  database.password: secret
  database.dbname: content_db
  table.include.list: public.documents
  transforms: ExtractNewDocumentState
  transforms.ExtractNewDocumentState.type: io.debezium.transforms.ExtractNewRecordState
  transforms.ExtractNewDocumentState.add.fields: op,ts_ms
  topic.prefix: cdc
```

Debezium publishes a message to `cdc.public.documents` for every `INSERT`, `UPDATE`, or `DELETE`. The RAG indexer consumes this topic — no application code changes required on the document side.

---

## Freshness Metrics and SLOs

Track freshness as a first-class SLO:

```python
from dataclasses import dataclass
import time

@dataclass
class FreshnessMetrics:
    doc_id: str
    source_updated_at: float   # when document changed in source system
    indexed_at: float           # when vector DB was updated

    @property
    def freshness_lag_seconds(self) -> float:
        return self.indexed_at - self.source_updated_at


# Alert if p95 freshness lag exceeds SLO
FRESHNESS_SLO_SECONDS = 30  # 30-second freshness guarantee

def check_freshness_slo(metrics: list[FreshnessMetrics]) -> bool:
    lags = sorted(m.freshness_lag_seconds for m in metrics)
    p95  = lags[int(len(lags) * 0.95)]
    return p95 <= FRESHNESS_SLO_SECONDS
```

---

## Handling Out-of-Order Events

Events may arrive out of order (network delays, Kafka partition rebalancing). Use a timestamp-based deduplication window:

```python
from functools import lru_cache

@lru_cache(maxsize=10_000)
def get_last_indexed_timestamp(doc_id: str) -> float:
    """Returns the timestamp of the last successfully indexed version."""
    result = vector_db.fetch(ids=[f"{doc_id}:chunk:0"])
    if result and result["vectors"]:
        return result["vectors"][f"{doc_id}:chunk:0"]["metadata"].get("timestamp", 0)
    return 0

def should_process(event: dict) -> bool:
    last_ts = get_last_indexed_timestamp(event["doc_id"])
    return event["timestamp"] > last_ts  # skip stale events
```

---

## Streaming RAG vs. Batch RAG Trade-offs

| Dimension | Batch RAG | Streaming RAG |
|-----------|-----------|--------------|
| **Freshness** | Hours–days | Seconds |
| **Infrastructure** | Simple (cron job) | Complex (Kafka, stream processor) |
| **Cost** | Low (offline embedding) | Higher (real-time embedding, always-on consumer) |
| **Error isolation** | Easy (re-run batch) | Harder (DLQ, retry, ordering) |
| **Best for** | Stable corpora, low-update frequency | News, financial data, live support docs |

---

## Key Takeaways

1. **The event stream is the authoritative update channel** — Kafka / Kinesis decouples document production from index consumption.
2. **Micro-batching within the consumer** is critical for embedding efficiency — never embed one chunk at a time.
3. **Delete before upsert on updates** — the index must not contain stale chunks from the prior version.
4. **CDC (Debezium) is the zero-code option** for database-backed content — it captures changes without modifying application code.
5. **Freshness lag is an SLO, not just a nice-to-have** — instrument and alert on it or users won't trust the system.

---

## Interview Q&A

**Q: How does streaming RAG differ from standard batch-indexed RAG?**

In batch RAG, documents are ingested on a schedule (hourly, nightly) so the index always lags behind the source of truth by at least one batch interval. Streaming RAG replaces the batch job with a continuous event-driven pipeline: every document mutation (create/update/delete) publishes an event to a stream (Kafka/Kinesis), and a stream processor consumes these events, re-embeds the affected chunks, and upserts them into the vector DB within seconds. The trade-off is infrastructure complexity: streaming requires managing a message broker, stream processor, dead-letter queues, and ordering guarantees that don't exist in a simple batch job. The payoff is freshness: queries issued after a document update immediately see the updated content rather than waiting for the next batch.

---

**Q: What is Change Data Capture (CDC) and how does it apply to streaming RAG?**

CDC is a technique for capturing row-level database mutations (INSERT/UPDATE/DELETE) and publishing them as a stream of events, without requiring application code changes. Tools like Debezium connect to a database's replication log (WAL in Postgres), read every committed change, and publish it to Kafka. A RAG indexer can subscribe to the CDC topic and process mutations as they happen. This is the preferred approach when the source documents live in a relational database: you don't need to modify the application that writes documents, you just add a CDC connector and let the indexer react. Debezium supports Postgres, MySQL, MongoDB, SQL Server, and Oracle.

---

**Q: How do you handle document updates in a streaming RAG index without creating duplicate or stale chunks?**

Each document's chunks are identified by a `doc_id` metadata field. On an update event: (1) first delete all vectors with `doc_id = X` from the index; (2) then re-chunk and re-embed the new content; (3) upsert the new vectors. This guarantees no stale chunks from the prior version remain. The main risk is partial failure: if the process crashes after deleting old vectors but before inserting new ones, the document is temporarily invisible. Mitigate by using a two-phase approach: write new vectors with a `pending` flag, swap the flag to `active` atomically, then delete the old vectors. Alternatively, keep a small freshness lag (5–10 seconds) acceptable and use an idempotent upsert-only approach: write new chunks with a `version_ts` timestamp field and filter queries to prefer the highest timestamp.
