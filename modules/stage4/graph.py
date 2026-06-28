"""
graph.py — Stage 4: Full AxiomCart graph with MemorySaver and HITL.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 4 CONCEPT: MemorySaver + Checkpointing                   ║
║                                                                  ║
║  Stage 3 had no memory — each invoke() was independent.          ║
║  Stage 4 adds a MemorySaver checkpointer.                        ║
║                                                                  ║
║  What MemorySaver does:                                          ║
║    After every node execution, LangGraph serialises the entire   ║
║    graph state to memory (keyed by thread_id).                   ║
║    On the next invoke() with the same thread_id, it restores     ║
║    the state from the checkpoint.                                ║
║                                                                  ║
║  This unlocks two capabilities:                                  ║
║    1. Multi-turn memory — the agent remembers prior conversation  ║
║    2. HITL pause/resume — interrupt() stores the paused state;  ║
║       Command(resume=...) restores it and continues              ║
║                                                                  ║
║  Thread ID:                                                       ║
║    {"configurable": {"thread_id": "session-abc"}}               ║
║    One thread_id = one conversation session.                     ║
║    Different thread_ids = different independent conversations.   ║
╚══════════════════════════════════════════════════════════════════╝

📖 Docs:
  - MemorySaver  → https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
  - Persistence  → https://langchain-ai.github.io/langgraph/concepts/persistence/
  - thread_id    → https://langchain-ai.github.io/langgraph/concepts/persistence/#threads
"""

from __future__ import annotations

from typing import Literal

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from modules.stage1.config import get_logger, llm
from modules.stage3.state import AxiomCartState, ClassificationResult, WorkerInput
from modules.stage3.nodes import (
    orchestrator_node,
    product_agent,
    synthesizer_node,
    build_context,
)
# Import the HITL-enabled support subgraph from stage4
from modules.stage4.nodes import support_subgraph_hitl
from modules.stage2.nodes import SUPPORT_PROMPT

logger = get_logger("graph")


# ── HITL-aware support_agent node ───────────────────────────────────
# This replaces the Stage 3 support_agent with one that uses the
# interrupt-enabled subgraph.

def support_agent_hitl(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the HITL-enabled sales-support agent."""
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Support Agent (HITL)  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = support_subgraph_hitl.invoke({"messages": [
        SystemMessage(content=SUPPORT_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "sales_support", "response": answer}]},
        goto="synthesizer",
    )


def build_graph():
    """Build the full Stage 4 graph: Stage 3 + MemorySaver + HITL support agent."""
    builder = StateGraph(AxiomCartState)

    builder.add_node("orchestrator",  orchestrator_node)
    builder.add_node("product_agent", product_agent)
    builder.add_node("support_agent", support_agent_hitl)   # ← HITL version
    builder.add_node("synthesizer",   synthesizer_node)

    builder.add_edge(START, "orchestrator")
    builder.add_edge("synthesizer", END)

    # ── MemorySaver checkpointer ─────────────────────────────────
    # This is the only change vs Stage 3 graph.py — adding a checkpointer
    # to compile() enables:
    #   • Multi-turn memory (conversation history persists across invocations)
    #   • interrupt() pause/resume (state is checkpointed before the pause)
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("Stage 4 graph compiled (with MemorySaver)")
    return graph


axiomcart_graph = build_graph()
