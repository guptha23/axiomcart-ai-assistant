"""
state.py — Full AxiomCart state for Stage 3 (multi-agent graph).

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 3 CONCEPT: State for Multi-Agent Graphs                  ║
║                                                                  ║
║  Stage 2 used AgentState with just one field: messages.         ║
║  Stage 3 needs a richer state to coordinate multiple agents:    ║
║                                                                  ║
║    messages          — full conversation history                 ║
║    user_query        — the current customer query (plain str)   ║
║    tasks             — routing decisions from the orchestrator  ║
║    requires_synthesis — should the synthesizer merge results?   ║
║    agent_results     — collected outputs from each agent        ║
║    final_answer      — the response returned to the user        ║
║                                                                  ║
║  New concept: CUSTOM REDUCERS                                    ║
║  The agent_results field uses a custom reducer rather than      ║
║  operator.add.  This lets parallel agent writes accumulate       ║
║  correctly and lets the orchestrator "reset" stale results       ║
║  from the previous turn by returning an empty list.             ║
║                                                                  ║
║  New concept: PYDANTIC STRUCTURED OUTPUT                         ║
║  ClassificationResult is a Pydantic model, not a TypedDict.     ║
║  The LLM fills it in with llm.with_structured_output().          ║
║  This gives us a typed, validated Python object instead of      ║
║  free text we'd need to parse ourselves.                         ║
╚══════════════════════════════════════════════════════════════════╝

📖 Docs:
  - Custom reducers       → https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
  - with_structured_output → https://python.langchain.com/docs/concepts/structured_outputs/
  - Pydantic BaseModel    → https://docs.pydantic.dev/latest/concepts/models/
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.messages import AnyMessage
from pydantic import BaseModel, Field


# ── Custom reducer for agent_results ────────────────────────────────
#
# Why not operator.add?
#
# operator.add always appends.  But on a new conversation turn, the
# orchestrator must RESET agent_results so stale results from the
# previous turn don't bleed into the new synthesizer call.
#
# Convention: returning an EMPTY list signals "reset".
#             returning a NON-EMPTY list signals "append".
#
def agent_results_reducer(current: list[dict], update: list[dict]) -> list[dict]:
    """Append agent results, or reset on empty list."""
    if not update:          # orchestrator sends [] to clear previous turn
        return []
    return current + update  # agent nodes send [{"source":..., "response":...}]


# ── Pydantic models for structured LLM output ───────────────────────
#
# The orchestrator uses llm.with_structured_output(ClassificationResult).
# Instead of returning free text, the LLM fills in these fields.
# Pydantic validates that the types are correct and required fields exist.

class AgentTask(BaseModel):
    """A single task assigned to one specialist agent."""

    agent: Literal["product_agent", "support_agent"] = Field(
        description="Which specialist agent handles this task"
    )
    task_description: str = Field(
        description="Clear description of what the agent should do"
    )


class ClassificationResult(BaseModel):
    """The orchestrator's full routing decision."""

    tasks: List[AgentTask] = Field(
        description="One or two tasks — one per agent that should be invoked"
    )
    requires_synthesis: bool = Field(
        description="True when multiple agents contribute and results must be merged"
    )
    reasoning: str = Field(
        description="One-line explanation of why this routing was chosen"
    )


# ── Main graph state ─────────────────────────────────────────────────

class AxiomCartState(TypedDict):
    """State that flows through the entire multi-agent graph.

    Field summary:
      messages          — full conversation (additive, never replaced)
      user_query        — the raw text of the current customer message
      tasks             — routing decisions from the orchestrator LLM
      requires_synthesis — flag set by the orchestrator
      agent_results     — one entry per agent that ran (custom reducer)
      final_answer      — assembled reply returned to the user
    """

    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Routing metadata — set by orchestrator_node
    tasks: list[AgentTask]
    requires_synthesis: bool

    # Accumulated agent outputs — uses custom reducer (reset + append)
    agent_results: Annotated[list[dict], agent_results_reducer]

    # Final text returned to the caller
    final_answer: str


class WorkerInput(TypedDict):
    """Payload delivered to each agent worker via Send().

    This is intentionally flat (no nested Pydantic) so Send() can
    serialize it without surprises.  Agents receive this instead of
    the full AxiomCartState.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    task_description: str
