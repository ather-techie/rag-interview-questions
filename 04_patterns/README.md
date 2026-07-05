# 04 — Patterns

> Reusable RAG design patterns that cut across the architectures in [`02_interview_bank/`](../02_interview_bank/).

## Contents

| File | Pattern | When to Use |
|------|---------|-------------|
| [01-router-fallback.md](01-router-fallback.md) | Router + Fallback | Route queries to cheapest retriever; fall back to expensive tiers only on failure |
| [02-fan-out-fan-in.md](02-fan-out-fan-in.md) | Fan-Out / Fan-In | Decompose multi-aspect queries into sub-queries, retrieve in parallel, merge with RRF |
| [03-migration-path.md](03-migration-path.md) | Migration Path | Upgrade Naive → Advanced → Modular → Agentic incrementally with forcing functions |
| [04-anti-patterns.md](04-anti-patterns.md) | Anti-Patterns | Five most common RAG design mistakes and how to avoid them |

## How to Use These Patterns

These patterns are **not architectures** (those are in `02_interview_bank/`). They are *design decisions* that sit on top of any architecture:

- A **Modular RAG** system can use the Router + Fallback pattern to pick its retrieval strategy
- An **Advanced RAG** system can use Fan-Out to improve recall on multi-aspect queries
- Any system benefits from the Migration Path framework when deciding which upgrade to make next
- The Anti-Patterns apply to all architectures

## Related Resources

- Architecture selection guidance: [`cheatsheets/CHEATSHEET.md`](../cheatsheets/CHEATSHEET.md)
- Evaluation: [`08_evaluation/`](../08_evaluation/)
- Production system design principles: [`00_overview/system_design_principles.md`](../00_overview/system_design_principles.md)
