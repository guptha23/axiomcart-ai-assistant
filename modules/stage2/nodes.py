"""
nodes.py — Two specialist agent subgraphs.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 2 CONCEPT: The ReAct Loop (Reason + Act)                 ║
║                                                                  ║
║  A LangGraph agent is just an LLM in a loop:                    ║
║                                                                  ║
║    START                                                         ║
║      ↓                                                           ║
║    model ── has tool_calls? ──YES──→ tools ──→ model             ║
║      │                                           (loop)          ║
║      └── no tool_calls ───────────────────────→ END             ║
║                                                                  ║
║  On each pass through the model node the LLM sees the full      ║
║  message history — including prior tool calls and their results. ║
║  It decides to either call another tool or produce a final       ║
║  response.  LangGraph manages this cycle natively.              ║
║                                                                  ║
║  This stage builds TWO specialists:                              ║
║    product_subgraph  – searches the catalog (1 tool)            ║
║    support_subgraph  – looks up orders + escalates (2 tools)    ║
║                                                                  ║
║  Same skeleton.  Different system prompt + different tools       ║
║  = completely different agent behaviour.                         ║
╚══════════════════════════════════════════════════════════════════╝

Key LangGraph concepts introduced here:
  • StateGraph    — the graph builder
  • add_node      — register a Python function as a graph node
  • add_edge      — unconditional transitions
  • add_conditional_edges — routing logic (model → tools or END)
  • compile()     — seal the graph into an executable Pregel runner

📖 Docs:
  - StateGraph     → https://langchain-ai.github.io/langgraph/reference/graphs/
  - ReAct pattern  → https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation
  - ToolNode       → https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode
  - bind_tools     → https://python.langchain.com/docs/concepts/tool_calling/#binding-tools-to-llms
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from modules.stage2.state import AgentState

# Re-use config and tools from Stage 1 so we don't duplicate code.
# In a real project all stages share the same src/ package.
from modules.stage1.config import get_logger, llm
from modules.stage1.data import SUPPORT_POLICIES
from modules.stage1.tools import escalate_to_human, get_order_status, search_product_catalog

logger = get_logger("nodes")


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
#
#  This is where agent specialisation lives.  Two agents, same base
#  LLM, same subgraph structure — but completely different behaviour
#  because the system prompt sets the persona, rules, and tool usage.
#
#  Think of it like a job description given to a contractor on day 1.
# ═══════════════════════════════════════════════════════════════════

PRODUCT_PROMPT = """\
You are the Product Discovery Agent for AxiomCart.

ROLE: Help customers find and learn about products.  You also handle
general conversation (greetings, thanks, chitchat).

TOOLS:
  search_product_catalog – semantic search over our product database

GUIDELINES:
- For greetings or general chat, respond warmly WITHOUT calling tools.
- For product questions, ALWAYS search the catalog first.
- Highlight key features and prices in your response.
- If no matching product is found, say so honestly.
- Keep responses concise and helpful.
"""

SUPPORT_PROMPT = f"""\
You are the Sales Support Agent for AxiomCart.

ROLE: Handle order enquiries and escalate issues to human agents.

TOOLS:
  get_order_status   – look up an order by order ID or customer email
  escalate_to_human  – create a ticket for human support

POLICIES:
{SUPPORT_POLICIES}

GUIDELINES:
- If the customer has NOT provided an order ID or email, ask for it
  BEFORE calling any tools.
- Be empathetic and professional.
- Only escalate when the customer explicitly asks for a human OR
  the issue cannot be resolved with available tools.
"""


# ═══════════════════════════════════════════════════════════════════
#  TOOL BINDINGS
#
#  llm.bind_tools(tools) attaches a JSON schema for each tool to
#  every API call.  The LLM can then respond with a tool_call object
#  instead of plain text.
#
#  We create separate bound LLMs so each agent only "sees" its own
#  tools — the product agent cannot accidentally call get_order_status.
# ═══════════════════════════════════════════════════════════════════

product_tools     = [search_product_catalog]
product_tools_map = {t.name: t for t in product_tools}   # fast lookup by name

support_tools     = [get_order_status, escalate_to_human]
support_tools_map = {t.name: t for t in support_tools}

product_llm = llm.bind_tools(product_tools)  # product LLM sees only search
support_llm = llm.bind_tools(support_tools)  # support LLM sees order + escalation


# ═══════════════════════════════════════════════════════════════════
#  ROUTING LOGIC
#
#  After the model node runs we ask: did the LLM decide to call a tool?
#    YES → go to "tools" node
#    NO  → the LLM produced a final answer → go to END
#
#  This is called a conditional edge.  It's just a function that
#  reads state and returns the name of the next node.
# ═══════════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """Route after the model node.

    Returns "tools" if the LLM wants to call a tool,
    returns END if the LLM produced a final text response.
    """
    last_message = state["messages"][-1]
    # AIMessage.tool_calls is a list — non-empty means the LLM wants action
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ═══════════════════════════════════════════════════════════════════
#  PRODUCT AGENT SUBGRAPH
#
#  Subgraph topology:
#
#    START → model ─┬─ "tools" → tools → model  (loop)
#                   └─  END                      (done)
#
#  Two nodes:  model (calls the LLM)  and  tools (executes tool calls)
# ═══════════════════════════════════════════════════════════════════

def product_model_node(state: AgentState) -> dict:
    """Invoke the product LLM and return its response as a new message."""
    response = product_llm.invoke(state["messages"])
    logger.info("[product] model  tool_calls=%s", bool(response.tool_calls))
    # Returning {"messages": [response]} triggers the operator.add reducer:
    # the response is APPENDED to state["messages"], not replacing it.
    return {"messages": [response]}


def product_tools_node(state: AgentState) -> dict:
    """Execute all tool calls from the last AIMessage and return ToolMessages."""
    last = state["messages"][-1]
    results = []
    for tool_call in last.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        logger.info("[product] tools  %s(%s)", name, args)
        tool_fn = product_tools_map.get(name)
        output = tool_fn.invoke(args) if tool_fn else f"Unknown tool: {name}"
        # ToolMessage links the result back to the tool call via tool_call_id
        results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {"messages": results}


# Build the product subgraph
_pb = StateGraph(AgentState)
_pb.add_node("model", product_model_node)
_pb.add_node("tools", product_tools_node)
_pb.add_edge(START, "model")
_pb.add_conditional_edges("model", should_continue)  # model → tools or END
_pb.add_edge("tools", "model")                       # always loop back to model
product_subgraph = _pb.compile()


# ═══════════════════════════════════════════════════════════════════
#  SUPPORT AGENT SUBGRAPH
#
#  Identical structure to the product subgraph.
#  Different LLM binding (support_llm) + different tool map.
#  The system prompt drives the different behaviour.
# ═══════════════════════════════════════════════════════════════════

def support_model_node(state: AgentState) -> dict:
    """Invoke the support LLM and return its response as a new message."""
    response = support_llm.invoke(state["messages"])
    logger.info("[support] model  tool_calls=%s", bool(response.tool_calls))
    return {"messages": [response]}


def support_tools_node(state: AgentState) -> dict:
    """Execute all tool calls from the last AIMessage and return ToolMessages."""
    last = state["messages"][-1]
    results = []
    for tool_call in last.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        logger.info("[support] tools  %s(%s)", name, args)
        tool_fn = support_tools_map.get(name)
        output = tool_fn.invoke(args) if tool_fn else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {"messages": results}


# Build the support subgraph — same pattern, different tools
_sb = StateGraph(AgentState)
_sb.add_node("model", support_model_node)
_sb.add_node("tools", support_tools_node)
_sb.add_edge(START, "model")
_sb.add_conditional_edges("model", should_continue)
_sb.add_edge("tools", "model")
support_subgraph = _sb.compile()
