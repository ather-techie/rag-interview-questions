# 32 — Few-Shot Example RAG (PEARL / Example-Augmented Prompting)

> Retrieves demonstration examples (query→answer pairs) instead of documents — teaches the LLM the expected output format and reasoning pattern via in-context examples.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
New Query
    │
    ▼
Example Retriever (search vector store of (query, answer) demonstration pairs)
    │
    ▼
Top-k Similar Demonstrations selected
    │
    ▼
Prompt Assembler (builds few-shot prompt: demonstrations + new query)
    │
    ▼
Generator (LLM produces the answer, conditioned on the retrieved examples)
```

### Key Components

| Component | Responsibility |
|---|---|
| Demonstration Store | Vector DB of (query, answer) pairs, indexed on the query side |
| Example Retriever | Finds the top-k demonstrations most semantically similar to the new query |
| Prompt Assembler | Formats retrieved demonstrations and the new query into a single few-shot prompt |
| Generator | LLM that produces the answer, following the pattern shown by the retrieved examples rather than raw documents |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Embedding model | sentence-transformers (e.g. `all-MiniLM-L6-v2`) |
| Vector store | FAISS, Chroma |
| Example selection | MMR (Maximal Marginal Relevance) for relevance + diversity |
| Automated optimization | DSPy for automated example selection and prompt optimization |

---

## Q1. What is Few-Shot Example RAG and how does it differ from standard document RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Standard RAG** retrieves *documents* — passages that contain factual information the LLM uses to answer the query.

**Few-Shot Example RAG** retrieves *examples* — prior (query, answer) pairs that show the LLM the expected output format, style, and reasoning pattern for the current query. The LLM then generates a new answer that follows the demonstrated pattern.

```
Standard RAG:
  User: "What is the capital of France?"
  Retrieved: [Document about France containing "Paris is the capital..."]
  LLM uses document as reference → "The capital of France is Paris."

Few-Shot Example RAG:
  User: "Translate 'hello' to Spanish."
  Retrieved examples:
    - ("Translate 'dog' to Spanish", "perro")
    - ("Translate 'house' to Spanish", "casa")
    - ("Translate 'water' to Spanish", "agua")
  LLM sees pattern: translate single word → single word translation → "hola"
```

**When few-shot example retrieval outperforms document retrieval:**

| Scenario | Standard RAG | Few-Shot Example RAG |
|----------|--------------|---------------------|
| Factual Q&A | ✓ Excellent | ✗ Not applicable |
| Code generation (follow project style) | Limited | ✓ Excellent |
| Structured output (follow schema) | Possible | ✓ Excellent |
| Few-shot classification | Poor | ✓ Excellent |
| Chain-of-thought tasks | Limited | ✓ Excellent |
| Text-to-SQL (follow naming conventions) | Limited | ✓ Excellent |

**The key insight:** For tasks where the LLM needs to see *what good output looks like* rather than *what the facts are*, retrieving examples is more informative than retrieving documents.

</details>

---

## Q2. How is the example datastore built and queried in PEARL? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**PEARL** (Prompting via Example-based and Adaptive Retrieval from a Library) introduced a systematic approach to building and querying an example library.

**Building the example datastore:**

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example library: list of (query, answer) pairs
examples = [
    {"query": "Write a Python function to reverse a string",
     "answer": "def reverse_string(s: str) -> str:\n    return s[::-1]"},
    {"query": "Write a Python function to check if a number is prime",
     "answer": "def is_prime(n: int) -> bool:\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"},
    # ... thousands more examples
]

# Embed the QUERY side of each example (not the answer)
queries = [ex["query"] for ex in examples]
embeddings = model.encode(queries, normalize_embeddings=True)

# Build FAISS index over query embeddings
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
```

**Why embed the query side?**

You retrieve examples whose *questions* are semantically similar to the current question. An example whose question matches the current question will demonstrate a similar reasoning pattern.

**Querying at inference time:**

```python
def retrieve_examples(current_query: str, k: int = 3) -> list[dict]:
    q_emb = model.encode([current_query], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), k)
    return [examples[i] for i in indices[0]]

def few_shot_rag(query: str) -> str:
    retrieved = retrieve_examples(query, k=3)
    
    # Format as few-shot prompt
    few_shot_block = ""
    for ex in retrieved:
        few_shot_block += f"Q: {ex['query']}\nA: {ex['answer']}\n\n"
    
    prompt = f"{few_shot_block}Q: {query}\nA:"
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Output with retrieved examples:**

```
Q: Write a Python function to reverse a string
A: def reverse_string(s: str) -> str:
       return s[::-1]

Q: Write a Python function to check if a number is prime
A: def is_prime(n: int) -> bool:
       ...

Q: Write a Python function to count vowels in a string
A: [LLM follows the pattern: function signature, docstring style, implementation]
```

</details>

---

## Q3. How do you select which examples to retrieve to maximize in-context learning? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naive nearest-neighbor retrieval can return near-identical examples. Good example selection requires balancing **relevance** and **diversity**.

**Anti-pattern: retrieving near-duplicates**

```
Query: "Sort a list of integers in Python"
Retrieved:
  - "Sort a list of numbers in Python"   ← near-duplicate
  - "Sort a Python list"                  ← near-duplicate  
  - "Sort integers using Python"          ← near-duplicate

All 3 examples look the same → no additional information for the LLM
```

**Better: Diverse but relevant examples**

```
Query: "Sort a list of integers in Python"
Retrieved:
  - "Sort a list of strings by length"     ← sorts, different key function
  - "Sort a list of dicts by a field"       ← sorts, complex key
  - "Sort a list of integers in reverse"   ← sorts integers, different parameter
```

**MMR-based example selection:**

```python
def select_examples_mmr(
    query: str,
    candidate_examples: list[dict],
    k: int = 3,
    lambda_: float = 0.7
) -> list[dict]:
    """Maximal Marginal Relevance for example selection."""
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    cand_embs = model.encode([ex["query"] for ex in candidate_examples],
                              normalize_embeddings=True)
    
    selected = []
    selected_embs = []
    
    while len(selected) < k:
        scores = []
        for i, (ex, emb) in enumerate(zip(candidate_examples, cand_embs)):
            if i in [candidate_examples.index(s) for s in selected]:
                continue
            relevance = float(q_emb @ emb)
            if not selected_embs:
                redundancy = 0.0
            else:
                redundancy = max(float(emb @ s_emb) for s_emb in selected_embs)
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            scores.append((i, mmr_score))
        
        best_idx = max(scores, key=lambda x: x[1])[0]
        selected.append(candidate_examples[best_idx])
        selected_embs.append(cand_embs[best_idx])
    
    return selected
```

**Coverage-based selection for chain-of-thought:**

For tasks requiring multi-step reasoning, select examples that cover different *reasoning patterns*:

```python
# Cluster examples by reasoning type first, then retrieve one from each cluster
# E.g., for math word problems:
# Cluster 1: single arithmetic step
# Cluster 2: multi-step with unit conversion
# Cluster 3: word problem requiring extraction of key values
```

**Ordering matters:**

Research (Min et al., 2022) shows that placing the most similar example **last** (closest to the actual query) yields the best performance — the LLM's attention pattern means recent context is weighed more heavily.

</details>

---

## Q4. How is Few-Shot Example RAG used for text-to-SQL generation? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Text-to-SQL is one of the highest-value applications: retrieve SQL examples that match the schema and query pattern, so the LLM learns the exact table names, column names, and join patterns in use.

```python
SQL_EXAMPLES = [
    {
        "question": "How many orders were placed last month?",
        "sql": "SELECT COUNT(*) FROM orders WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND created_at < DATE_TRUNC('month', CURRENT_DATE)"
    },
    {
        "question": "What is the total revenue by product category?",
        "sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.id GROUP BY p.category ORDER BY revenue DESC"
    },
    # ...
]

def text_to_sql(question: str, schema: str) -> str:
    examples = retrieve_examples(question, k=3)
    
    few_shot = "\n".join(
        f"-- Question: {ex['question']}\n{ex['sql']}\n"
        for ex in examples
    )
    
    prompt = f"""You are a SQL expert. Given a database schema and example queries,
write a SQL query for the new question.

Schema:
{schema}

Example queries:
{few_shot}

-- Question: {question}
-- SQL:"""
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Why this outperforms a static few-shot prompt:**
- Static prompts use the same 3–5 examples for every query → examples may be irrelevant
- Dynamic retrieval finds examples with the same JOIN patterns, aggregations, or filters as the current question
- When the schema has hundreds of tables, retrieved examples implicitly teach which tables are relevant

</details>

---

## Q5. How do you combine document RAG and example RAG in the same pipeline? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The two types of retrieval are complementary and can be combined in the same context window.

```
User Query
    │
    ├──► Document Retrieval    → "What are the facts?" (retrieved passages)
    │
    └──► Example Retrieval     → "What should the output look like?" (demonstrations)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Prompt structure:                                    │
│                                                     │
│ [Few-shot examples — teach format/style]            │
│ Example 1: Q + A                                    │
│ Example 2: Q + A                                    │
│                                                     │
│ [Retrieved documents — provide facts]               │
│ Document 1: relevant passage                        │
│ Document 2: relevant passage                        │
│                                                     │
│ [Current query]                                     │
└─────────────────────────────────────────────────────┘
```

**Implementation:**

```python
def hybrid_rag(query: str) -> str:
    # Retrieve both types
    doc_results = doc_retriever.retrieve(query, k=3)
    example_results = retrieve_examples(query, k=2)
    
    # Build combined prompt
    examples_block = ""
    for ex in example_results:
        examples_block += f"Q: {ex['query']}\nA: {ex['answer']}\n\n"
    
    docs_block = "\n\n".join(
        f"[Document {i+1}]: {doc.page_content}"
        for i, doc in enumerate(doc_results)
    )
    
    prompt = f"""Examples of the expected answer format:
{examples_block}
---
Reference documents (use these for factual content):
{docs_block}
---
Question: {query}
Answer (following the format shown in the examples):"""
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Use cases for combined retrieval:**

- **Technical documentation with style guide**: Documents provide API facts; examples show the expected explanation format
- **Medical report generation**: Documents provide clinical guidelines; examples show the expected SOAP note format
- **Code review**: Documents provide security best practices; examples show what a good code review comment looks like

</details>

---

## Real-World Applications

- **GitHub Copilot**: Retrieves similar code snippets from the open codebase as few-shot context for code completion
- **Text-to-SQL systems** (Salesforce DAIL-SQL, DIN-SQL): Dynamic example retrieval boosts SQL accuracy by 5–15% over static few-shot
- **Customer support bots**: Retrieve past solved tickets as examples for consistent tone and resolution format
- **Medical coding**: Retrieve similar clinical notes with correct ICD codes as examples for coding new notes
- **Structured extraction**: Retrieve examples of correctly-formatted JSON extractions to guide schema-constrained generation
