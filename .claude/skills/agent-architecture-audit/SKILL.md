---
description: Audit this project's LLM pipeline against the 12-layer agent architecture framework. Run after major architecture changes or when diagnosing response quality bugs.
---

Perform a structured agent architecture audit of this project using the 12-layer framework below.

## The 12-Layer Stack

Each layer is a potential failure point. Audit all 12 before declaring the system healthy.

1. **System Prompt** — persona, constraints, tone, language rules
2. **Session History** — what message objects the LLM actually receives (HumanMessage/AIMessage sequence, consecutive message merging, format consistency)
3. **Long-Term Memory** — retrieval from vector store / user_memories; deduplication; staleness
4. **Distillation** — summarization or compression of older context; what gets dropped
5. **Active Recall** — RAG retrieval; what triggers it; ranking and cutoff
6. **Tool Selection** — how the LLM picks tools; whether it can get stuck in a loop
7. **Tool Execution** — actual tool call invocation; error handling; timeouts
8. **Tool Interpretation** — how tool results are fed back; truncation; hallucination on failure
9. **Answer Shaping** — post-processing before delivery (strip_markdown, language correction, etc.)
10. **Platform Rendering** — Telegram message limits, parse_mode, character escaping
11. **Hidden Repair Loops** — second LLM passes (e.g. apply_language_correction); are they visible in history?
12. **Persistence** — what survives restarts; checkpointer vs. custom table; serialization constraints

## 5 Common Failure Patterns to Watch For

1. **History/delivery inconsistency** — response mutated after being stored
2. **Dead code branches** — conditional logic that always evaluates false because a state key was removed
3. **Tool discipline gaps** — LLM can invoke tools without code-level gating; tool_use_failed not handled
4. **Concurrent write races** — async tasks without per-key locks
5. **Hidden repair loops with no history trace** — second LLM pass applied but not stored

## 4-Phase Audit Workflow

### Phase 1 — Inventory
- Map all LLM call sites (grep for `ainvoke`, `agenerate`, `llm.`, `ChatGroq`)
- Map all state keys read/written at each node
- Identify all post-processing steps between LLM output and final delivery

### Phase 2 — Trace a Request
- Follow a single message through all 12 layers manually
- Note what each node reads from state vs. what it writes
- Check for keys that are read but may no longer be written (dead reads)

### Phase 3 — Stress Cases
- Forwarded message with media (does ingester run?)
- Reply to a message outside the recent window (does context_builder handle it?)
- Consecutive messages from same user (does HumanMessage merging happen?)
- Bot restart mid-conversation (does thread_history survive?)
- Concurrent requests from same user (are there race conditions?)

### Phase 4 — Severity Classification and Report
- **HIGH**: Data loss, silent wrong answers, security boundary violation
- **MEDIUM**: Inconsistency visible to users, dead code that hides bugs, races under load
- **LOW**: Minor inefficiency, cosmetic inconsistency, missing observability

Output a findings table (Layer | Finding | Severity | Status) and a prioritised fix list with rationale.
