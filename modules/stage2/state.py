"""
state.py — LangGraph state definitions for Stage 2.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 2 CONCEPT: State — the shared memory of a graph          ║
║                                                                  ║
║  In LangGraph, every node reads from and writes to a single     ║
║  shared State object.  Think of it as the whiteboard that all   ║
║  participants in a meeting can see and write on.                 ║
║                                                                  ║
║  State is defined as a Python TypedDict.                        ║
║  Each field has a "reducer" — a function that determines how    ║
║  updates from different nodes are merged.                        ║
║                                                                  ║
║  Stage 2 uses a minimal AgentState with a single field:         ║
║    messages: list[AnyMessage]                                    ║
║  The reducer is operator.add — each node appends new messages   ║
║  rather than replacing the whole list.                           ║
╚══════════════════════════════════════════════════════════════════╝

Why TypedDict over a regular class?
  • LangGraph can serialize/deserialize TypedDicts for checkpointing.
  • Type hints document exactly what each node expects.
  • No __init__ boilerplate — just declare the fields.

📖 Docs:
  - LangGraph State   → https://langchain-ai.github.io/langgraph/concepts/low_level/#state
  - Reducers          → https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
  - Annotated         → https://langchain-ai.github.io/langgraph/concepts/low_level/#using-annotated-type
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage


class AgentState(TypedDict):
    """
    Minimal state for a single-agent model ⇄ tools subgraph.

    Fields
    ──────
    messages : list[AnyMessage]
        The full conversation as a list of LangChain message objects.
        The reducer is operator.add, meaning each node's return value
        is APPENDED to the existing list rather than replacing it.

        Message types you'll see here:
          SystemMessage  – initial instructions (agent persona + rules)
          HumanMessage   – the customer's question
          AIMessage      – the LLM's response (may contain tool_calls)
          ToolMessage    – the result of a tool execution

        Example sequence for a product query:
          [SystemMessage("You are the Product Agent…"),
           HumanMessage("Show me headphones under 15000"),
           AIMessage(tool_calls=[{name: "search_product_catalog", …}]),
           ToolMessage("Found: Sony XM5…"),
           AIMessage("Here are some great headphones for you: …")]
    """

    # Annotated[T, reducer] — the reducer controls how updates are merged.
    # operator.add means: new_messages = current_messages + returned_messages
    messages: Annotated[list[AnyMessage], operator.add]
