"""
nodes.py — Stage 4: adds HITL interrupt to the support agent.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 4 CONCEPT: Human-in-the-Loop (HITL)                      ║
║                                                                  ║
║  The Problem:                                                    ║
║    A customer asks "Where is my order?" without providing an     ║
║    order ID.  The support agent can't call get_order_status()    ║
║    without it.  We need to pause, ask the user, then resume.     ║
║                                                                  ║
║  The Naive Solution (don't do this):                             ║
║    The agent sends back "I need your order ID" and the next      ║
║    message from the user has the ID.  Works for single-turn      ║
║    but loses all tool-call context across turns.                 ║
║                                                                  ║
║  The LangGraph Solution: interrupt()                             ║
║    interrupt() literally PAUSES THE ENTIRE GRAPH at the exact    ║
║    point it's called.  The caller gets back a dict with a        ║
║    special "__interrupt__" key containing the question.          ║
║    When the user answers, invoke(Command(resume=answer)) resumes  ║
║    the graph from EXACTLY where it paused.  No state is lost.    ║
║                                                                  ║
║  Timeline:                                                        ║
║    invoke("Where is my order?")                                  ║
║          ↓                                                       ║
║    support_model detects: no order ID, no tools called yet       ║
║          ↓                                                       ║
║    interrupt("Could you provide your order ID?")                 ║
║          ↓  ← GRAPH IS FULLY PAUSED HERE                        ║
║    caller sees result["__interrupt__"][0].value                  ║
║          ↓                                                       ║
║    user answers "ORD102"                                         ║
║          ↓                                                       ║
║    invoke(Command(resume="ORD102"))                              ║
║          ↓                                                       ║
║    support_model resumes → appends HumanMessage("ORD102")        ║
║    → LLM now has order ID → calls get_order_status → answer      ║
╚══════════════════════════════════════════════════════════════════╝

📖 Docs:
  - interrupt()   → https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt
  - Command       → https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command
  - HITL concepts → https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
  - MemorySaver   → https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

# Re-use Stage 2 state and tool bindings
from modules.stage2.state import AgentState
from modules.stage2.nodes import (
    SUPPORT_PROMPT,
    support_tools,
    support_tools_map,
    should_continue,
    support_tools_node,
)
from modules.stage1.config import get_logger, llm

logger = get_logger("nodes.hitl")


# ═══════════════════════════════════════════════════════════════════
#  Modified support_model_node with HITL
#
#  The key insight: we only interrupt when BOTH of these are true:
#    1. The LLM made no tool calls (it's asking for info, not acting)
#    2. No tools have been called yet in this subgraph run
#       (i.e. we're not mid-conversation after a prior tool call)
#
#  If tools have already run, the agent is responding after having
#  gathered data — no need to interrupt again.
# ═══════════════════════════════════════════════════════════════════

def support_model_node_hitl(state: AgentState) -> dict:
    """Call the support LLM. Interrupt if it needs missing info.

    interrupt() pauses the entire graph and surfaces a question
    to the caller.  The graph resumes when Command(resume=...) is
    passed to the next invoke() call.
    """
    # Bind tools to the support LLM (same as Stage 2)
    support_llm = llm.bind_tools(support_tools)

    response = support_llm.invoke(state["messages"])
    logger.info("[support-hitl] model  tool_calls=%s", bool(response.tool_calls))

    if not response.tool_calls:
        # Check if any tools have already been called in this run
        any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])

        if not any_tools_called:
            # The LLM is asking for missing info (no order ID, no email).
            # Pause the graph and surface the question to the caller.
            logger.info("[support-hitl] HITL: pausing graph to collect user info")
            user_reply = interrupt(response.content)
            # When the graph resumes, user_reply contains the answer.
            # We append it as a HumanMessage so the LLM sees it on next pass.
            logger.info("[support-hitl] HITL: resumed with %r", user_reply)
            return {"messages": [response, HumanMessage(content=str(user_reply))]}

    return {"messages": [response]}


def support_should_continue_hitl(state: AgentState) -> str:
    """Route after support model.

    Extra case vs Stage 2: if the last message is a HumanMessage
    (the user just answered via HITL), loop back to model so the LLM
    can now call the appropriate tool.
    """
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"           # HITL resume path — loop back
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build the HITL-enabled support subgraph ─────────────────────────
_sb_hitl = StateGraph(AgentState)
_sb_hitl.add_node("model", support_model_node_hitl)
_sb_hitl.add_node("tools", support_tools_node)
_sb_hitl.add_edge(START, "model")
_sb_hitl.add_conditional_edges("model", support_should_continue_hitl)
_sb_hitl.add_edge("tools", "model")

# IMPORTANT: compile with checkpointer=None here — the parent graph
# supplies the MemorySaver checkpointer.  Subgraphs should not have
# their own checkpointers unless you intentionally want nested state.
support_subgraph_hitl = _sb_hitl.compile()
