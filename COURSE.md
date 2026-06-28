# AxiomCart — Multi-Agent AI Assistant Course

> **Build a production-grade multi-agent voice-enabled shopping assistant in 4 stages, ~3 hours.**

This course builds the **AxiomCart AI assistant** progressively — each stage is a self-contained, runnable system that introduces one new concept. By Stage 4, you have the complete final product.

---

## 🗺️ Course Map

```
modules/
├── stage1/   ←  The Toolkit           (~30 min)  LLM clients, RAG, @tool
├── stage2/   ←  The Agent Loop        (~35 min)  StateGraph, ReAct, two specialists
├── stage3/   ←  The Orchestrator      (~50 min)  Structured output, Send(), synthesis
└── stage4/   ←  Complete AxiomCart    (~35 min)  HITL, MemorySaver, REPL, Voice
```

Final architecture (built in Stage 3, extended in Stage 4):

```
START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                      └─ support_agent ──↗
                           (HITL in Stage 4)
```

---

## ⚡ Quick Start

```bash
# 0. Install uv (one-time, macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 1. Clone and enter the project
cd axiomcart-ai-assistant

# 2. Create virtualenv and install all dependencies
uv venv .venv --python 3.11
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-...       ← required
#   LLM_MODEL=gpt-4o            ← or gpt-3.5-turbo for lower cost

# 4. Run a quick smoke test (Module 1)
uv run python modules/stage1/test_stage1.py
```

---

## Stage 1 — The Toolkit ⏱ ~30 min

**One idea:** Before the agents, build their tools.

```bash
# Run the full Module 1 concept walkthrough
uv run python modules/stage1/test_stage1.py
```

**Concepts:** `.env` + API clients · `LLM_MODEL` from env · RAG (embeddings → vector store → similarity search) · `@tool` decorator

📖 [Stage 1 README](modules/stage1/README.md)

---

## Stage 2 — The Agent Loop ⏱ ~35 min

**One idea:** One agent = LLM in a model ⇄ tools loop.

```bash
# Run the full Module 2 concept walkthrough
uv run python modules/stage2/test_stage2.py
```

**Concepts:** `AgentState` TypedDict · `operator.add` reducer · `StateGraph` nodes + edges · `should_continue` conditional edge · `bind_tools()`

📖 [Stage 2 README](modules/stage2/README.md)

---

## Stage 3 — The Orchestrator ⏱ ~50 min

**One idea:** A manager classifies queries and dispatches to specialists — in parallel.

```bash
# Run the full Module 3 concept walkthrough
uv run python modules/stage3/test_stage3.py
```

**Concepts:** `with_structured_output(Pydantic)` · `Send()` parallel dispatch · `Command(update=..., goto=...)` · custom state reducer · response synthesis

📖 [Stage 3 README](modules/stage3/README.md)

---

## Stage 4 — Complete AxiomCart ⏱ ~35 min

**One idea:** Add memory, graceful fallbacks, and voice — making it production-ready.

```bash
# Run the full Module 4 concept walkthrough
uv run python modules/stage4/test_stage4.py

# Full interactive REPL
uv run python -m modules.stage4.main

# Voice mode (requires microphone)
uv run python -m modules.stage4.main --voice
```

**Concepts:** `MemorySaver` checkpointer · `thread_id` session management · `interrupt()` + `Command(resume=...)` · Whisper STT · OpenAI TTS

📖 [Stage 4 README](modules/stage4/README.md)

---

## Concept Progression

| Stage | New Concept | "Aha" Moment |
|---|---|---|
| 1 | `@tool` + RAG + `LLM_MODEL` env var | Tools work without any graph |
| 2 | ReAct loop | LLM decides when to call tools |
| 3 | `Send()` + `with_structured_output` | Two agents run simultaneously |
| 4 | `interrupt()` + `MemorySaver` | Graph pauses and resumes mid-execution |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY is missing` | Copy `.env.example` → `.env`, add your key |
| `ModuleNotFoundError: modules` | Run from the project root: `cd axiomcart-ai-assistant` |
| `gpt-3.5-turbo json_schema warning` | Expected — auto-falls back to `function_calling`. Not an error. |
| HITL not triggering | Ensure `MemorySaver` is in `stage4/graph.py`'s `compile(checkpointer=memory)` |
| Voice: no audio | Check microphone permissions; install `sounddevice` and `soundfile` |

---

## What's in `src/`?

The `src/` folder is the **final production version** of the same system — same architecture, with minor refinements (one-file-per-module, imports cleaned up, voice.py extracted). Compare your stage files with `src/` once you've completed all modules.
