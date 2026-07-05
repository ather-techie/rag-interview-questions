# Conversational Memory Architecture

> How to persist, compress, and retrieve conversation history so RAG systems maintain coherent multi-turn sessions without overflowing the context window.

---

## The Problem

A RAG system answers a single query well. A *conversational* RAG system must also remember what was said three turns ago, track which documents were already cited, resolve anaphoric references ("what about *that* approach?"), and handle topic shifts — all while the context window fills up.

Conversational memory is not a single technique; it's a stack of four layers with different time horizons:

```
┌────────────────────────────────────────┐
│  WORKING MEMORY (in-prompt)            │ ← last N turns, verbatim
├────────────────────────────────────────┤
│  EPISODIC SUMMARY (compressed turns)  │ ← rolling summary of older turns
├────────────────────────────────────────┤
│  SEMANTIC MEMORY (vector store)        │ ← per-user fact store, long-lived
├────────────────────────────────────────┤
│  KNOWLEDGE BASE (shared corpus)        │ ← unchanged RAG document index
└────────────────────────────────────────┘
```

Each layer answers a different question:
- **Working memory**: "What did we just say?"
- **Episodic summary**: "What has this session been about?"
- **Semantic memory**: "What do I know about this user's history?"
- **Knowledge base**: "What do my documents say?"

---

## Layer 1: Working Memory — Sliding Window

Keep the last N complete turns verbatim in the context. Simple and reliable, but it hits the context limit linearly with conversation length.

```python
from collections import deque

class SlidingWindowMemory:
    def __init__(self, max_turns: int = 10):
        self.turns = deque(maxlen=max_turns)
    
    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
    
    def to_messages(self) -> list[dict]:
        return list(self.turns)
    
    def token_count(self) -> int:
        return sum(len(t["content"].split()) * 1.3 for t in self.turns)  # rough estimate
```

**When sliding window alone works:** single-session QA where sessions are short (<10 turns) and each turn is independent. It fails when users refer back to early turns that have scrolled off the window.

---

## Layer 2: Episodic Summary — Compress and Carry

When the sliding window is full (or token budget is approached), compress the oldest N turns into a summary and carry that summary forward. This reduces token cost while preserving the semantic gist.

```python
import anthropic

client = anthropic.Anthropic()

COMPRESSION_PROMPT = """Summarize the following conversation segment concisely (2-4 sentences).
Preserve: key questions asked, topics discussed, decisions made, and any user preferences revealed.
Discard: filler phrases, repeated confirmations, verbatim code blocks (note they were shown, not the code)."""

def compress_turns(turns: list[dict]) -> str:
    text = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"{COMPRESSION_PROMPT}\n\n<conversation>\n{text}\n</conversation>",
        }],
    )
    return resp.content[0].text


class CompressAndCarryMemory:
    def __init__(self, window_turns: int = 6, compress_after: int = 10):
        self.summary: str = ""
        self.recent_turns: deque = deque(maxlen=window_turns)
        self.compress_after = compress_after
        self.total_turns = 0
    
    def add(self, role: str, content: str):
        self.recent_turns.append({"role": role, "content": content})
        self.total_turns += 1
        
        if self.total_turns % self.compress_after == 0:
            older = list(self.recent_turns)[:-3]  # keep last 3 turns verbatim
            if older:
                new_summary = compress_turns(older)
                self.summary = (self.summary + "\n" + new_summary).strip() if self.summary else new_summary
    
    def to_messages(self) -> list[dict]:
        messages = []
        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[Earlier conversation summary: {self.summary}]",
            })
            messages.append({"role": "assistant", "content": "Understood."})
        messages.extend(self.recent_turns)
        return messages
```

---

## Layer 3: Semantic Memory — Per-User Vector Store

Store notable facts, preferences, and decisions from past sessions in a vector index keyed to the user. At the start of each session, retrieve the most relevant memories for the current query.

```python
import json
import numpy as np

class SemanticMemoryStore:
    """Per-user long-term memory: store and retrieve facts across sessions."""
    
    def __init__(self, user_id: str, embed_fn, vector_db):
        self.user_id = user_id
        self.embed  = embed_fn
        self.db     = vector_db
        self.namespace = f"memory:{user_id}"
    
    def store(self, fact: str, metadata: dict = None):
        emb = self.embed(fact)
        self.db.upsert(
            namespace=self.namespace,
            vectors=[{
                "id": f"{self.user_id}:{hash(fact)}",
                "values": emb.tolist(),
                "metadata": {"fact": fact, **(metadata or {})},
            }]
        )
    
    def retrieve(self, query: str, k: int = 3) -> list[str]:
        emb = self.embed(query)
        results = self.db.query(
            namespace=self.namespace,
            vector=emb.tolist(),
            top_k=k,
            include_metadata=True,
        )
        return [r["metadata"]["fact"] for r in results["matches"]]
    
    def extract_and_store(self, conversation_turn: str):
        """Use an LLM to extract memorable facts from a turn and store them."""
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system='Extract notable facts about the user from this text. '
                   'Output a JSON array of strings, one fact per item. '
                   'Return [] if nothing notable. Focus on: preferences, constraints, domain knowledge, decisions.',
            messages=[{"role": "user", "content": conversation_turn}],
        )
        facts = json.loads(resp.content[0].text)
        for fact in facts:
            self.store(fact)
```

---

## Layer 4: MemGPT Paging Model

MemGPT (Packer et al., 2023) models memory explicitly as a hierarchy:

```
┌─────────────────────────────────────┐
│  MAIN CONTEXT (in-window, fast)     │ ← active working set
│  ├─ System prompt                   │
│  ├─ Working memory (user facts)     │
│  ├─ Conversation buffer (recent)    │
│  └─ Scratch pad                     │
├─────────────────────────────────────┤
│  EXTERNAL STORAGE (out-of-window)   │ ← paged in/out on demand
│  ├─ Archival memory (vector store)  │
│  └─ Recall memory (conversation DB) │
└─────────────────────────────────────┘
```

The model itself calls special memory functions (`memory_append`, `archival_memory_search`, `conversation_search`) to page information in and out — treating memory management as just another tool-use problem.

```python
MEMGPT_TOOLS = [
    {
        "name": "memory_append",
        "description": "Write a new fact to working memory (in-context).",
        "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
    },
    {
        "name": "archival_memory_search",
        "description": "Search long-term archival memory for relevant facts.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "archival_memory_insert",
        "description": "Store a fact into long-term archival memory.",
        "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
    },
]
```

---

## Session Boundary Detection

A session boundary is when a user starts a significantly new topic. Detecting it lets you:
- Flush the episodic summary and start fresh
- Store the session summary in semantic memory
- Reset the sliding window

```python
def is_new_session(prev_query: str, current_query: str, threshold: float = 0.4) -> bool:
    prev_emb    = embed(prev_query)
    current_emb = embed(current_query)
    similarity  = np.dot(prev_emb, current_emb)
    return float(similarity) < threshold


class SessionManager:
    def __init__(self, memory: CompressAndCarryMemory, semantic: SemanticMemoryStore):
        self.memory   = memory
        self.semantic = semantic
        self.last_query = ""
    
    def handle_turn(self, user_query: str, response: str):
        if self.last_query and is_new_session(self.last_query, user_query):
            # Store session gist in long-term memory
            session_summary = self.memory.summary
            if session_summary:
                self.semantic.store(f"Previous session: {session_summary}")
            # Reset working + episodic memory
            self.memory = CompressAndCarryMemory()
        
        self.memory.add("user", user_query)
        self.memory.add("assistant", response)
        self.last_query = user_query
```

---

## Anaphora Resolution

When a user says "tell me more about *that*", the model must resolve "that" to a prior referent. Two strategies:

**Strategy 1 — Query rewriting**: use a cheap LLM to rewrite the query with explicit references before embedding.

```python
REWRITE_PROMPT = """Given the conversation history and the user's latest message,
rewrite the latest message as a standalone, self-contained question that contains
all necessary context from history. Output only the rewritten question."""

def resolve_anaphora(history: list[dict], current_query: str) -> str:
    history_text = "\n".join(f"{t['role']}: {t['content']}" for t in history[-6:])
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system=REWRITE_PROMPT,
        messages=[{"role": "user", "content": f"History:\n{history_text}\n\nLatest: {current_query}"}],
    )
    return resp.content[0].text
```

**Strategy 2 — Entity tracking**: maintain a running entity list (document names, concepts, entities) and inject it into every retrieval query.

---

## Memory Layer Comparison

| Layer | Scope | Cost | Retention | Best For |
|-------|-------|------|-----------|---------|
| Sliding window | Last N turns verbatim | O(N × tokens) | Session | Short sessions, exact quote recall |
| Episodic summary | Current session compressed | O(1) tokens | Session | Medium sessions, topic continuity |
| Semantic memory | Multi-session user facts | Vector DB lookup | Persistent | Personalisation, user preferences |
| MemGPT paging | All of the above unified | Tool calls per turn | Persistent | Long-lived agent assistants |

---

## Key Takeaways

1. **Use all four layers for production assistants** — they address different time horizons and failure modes.
2. **Compress-and-carry is the pragmatic default** — it handles sessions of arbitrary length without unbounded token growth.
3. **Anaphora resolution via query rewriting is cheap and effective** — a Haiku call costs less than a failed retrieval.
4. **Session boundaries are opportunities** — detect them, store a session summary in semantic memory, and reset working memory.
5. **MemGPT is worth studying** even if you don't deploy it — it makes the memory trade-offs explicit and gives a vocabulary for discussing them in interviews.

---

## Interview Q&A

**Q: What is the "sliding window" memory problem and how does compress-and-carry solve it?**

A sliding window keeps the last N turns verbatim in the context. When the window fills, older turns are simply dropped — the system loses all memory of early conversation context. Compress-and-carry solves this by periodically compressing older turns into a short summary (1–3 sentences) and prepending that summary to the context instead of the raw turns. The cost of carrying the summary is O(summary tokens) rather than O(N × average_turn_tokens), so the context window doesn't grow linearly with session length. The trade-off: the summary loses precision (exact quotes, nuanced phrasing), so compress-and-carry is best combined with a sliding window over the most recent few turns for fidelity on immediate context.

---

**Q: How does MemGPT differ from standard conversational memory approaches?**

Standard approaches manage memory *outside* the model — the application layer decides what goes into the context and what gets archived. MemGPT gives the model explicit memory management *tools* (append to working memory, search archival memory, insert into archival memory) and lets the model decide when to invoke them, just like it decides when to call a retrieval tool. This has two benefits: (1) the model can reason about *what* is worth remembering ("this constraint will matter later") in a way a simple sliding window can't; (2) memory management is transparent in the model's reasoning trace. The downside is higher latency (memory tool calls add round trips) and the risk of the model mismanaging memory if the tool prompts are poorly specified.

---

**Q: How would you implement cross-session personalization in a conversational RAG assistant?**

After each session ends (detected by session boundary or explicit logout), extract notable user facts using a lightweight LLM pass ("what did we learn about this user's preferences, constraints, or domain knowledge?") and store those facts in a per-user vector store (namespaced by user ID). At the start of each new session, retrieve the top-k most relevant memories for the opening query and inject them into the system prompt as a "user context" section. Example facts worth storing: preferred verbosity, domain expertise level, prior decisions ("user decided not to use HyDE"), recurring entities they ask about. Avoid storing full conversation transcripts — they're expensive and violate minimal-data principles; store semantic summaries only.
