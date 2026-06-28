"""
test_stage2.py — Run all Stage 2 concepts interactively.

Run from the project root:
    python3 stage2/test_stage2.py

What this file demonstrates:
  Concept 1 — AgentState: the messages list and how it grows
  Concept 2 — ReAct loop: LLM decides when to call tools (and when NOT to)
  Concept 3 — Two specialists: same subgraph structure, different behaviour
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.disable(logging.INFO)

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from modules.stage2.nodes import (
    PRODUCT_PROMPT,
    SUPPORT_PROMPT,
    product_subgraph,
    support_subgraph,
)


# ── Helpers ──────────────────────────────────────────────────────────

def banner(title: str) -> None:
    width = 64
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)

def section(label: str) -> None:
    print(f"\n── {label} {'─' * (60 - len(label))}")

def observe(note: str) -> None:
    print(f"   👁  {note}")

def print_message_sequence(messages: list) -> None:
    """Print each message in a result as a labelled, readable sequence."""
    for i, m in enumerate(messages):
        msg_type = type(m).__name__
        if isinstance(m, SystemMessage):
            print(f"  [{i}] {msg_type:14s} → (system prompt — {len(m.content)} chars)")
        elif isinstance(m, HumanMessage):
            print(f"  [{i}] {msg_type:14s} → {m.content[:70]}")
        elif isinstance(m, AIMessage):
            if getattr(m, "tool_calls", None):
                calls = [f"{tc['name']}({list(tc['args'].values())})" for tc in m.tool_calls]
                print(f"  [{i}] {msg_type:14s} → TOOL CALLS: {calls}")
            else:
                print(f"  [{i}] {msg_type:14s} → {m.content[:70]}")
        elif isinstance(m, ToolMessage):
            print(f"  [{i}] {msg_type:14s} → {m.content[:70]}...")
        else:
            print(f"  [{i}] {msg_type:14s} → {str(m)[:70]}")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 1: AgentState — Messages as a Running Log
#  Goal: show that state["messages"] accumulates the full conversation.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 1 — AgentState: Messages Build Up Over Time")

print("""
  AgentState has one field: messages: Annotated[list[AnyMessage], operator.add]
  Each node APPENDS to it (never replaces it).
  After a full run, messages is the complete conversation transcript.
""")

section("Run a product query and inspect every message")
result = product_subgraph.invoke({"messages": [
    SystemMessage(content=PRODUCT_PROMPT),
    HumanMessage(content="Show me Sony headphones"),
]})
print_message_sequence(result["messages"])
observe(f"Total messages in state: {len(result['messages'])}")
observe("Notice SystemMessage→HumanMessage→AIMessage(tool_call)→ToolMessage→AIMessage(final)")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 2: The ReAct Loop — LLM Decides When to Act
#  Goal: show the LLM calling a tool for a product query, but NOT
#        calling any tool for a simple greeting.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 2 — ReAct Loop: LLM Decides When to Call Tools")

print("""
  The conditional edge 'should_continue' checks: did the LLM want to call a tool?
    → YES: go to tools node → loop back to model
    → NO:  go to END

  The LLM decides. You don't write routing rules.
""")

section("Query WITH a tool call — 'Do you have wireless headphones?'")
r_tool = product_subgraph.invoke({"messages": [
    SystemMessage(content=PRODUCT_PROMPT),
    HumanMessage(content="Do you have wireless headphones?"),
]})
print_message_sequence(r_tool["messages"])
tool_calls_made = sum(1 for m in r_tool["messages"] if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
observe(f"Tool calls made: {tool_calls_made}  — LLM searched the catalog before answering.")

section("Query WITHOUT a tool call — 'Hi! How are you?'")
r_no_tool = product_subgraph.invoke({"messages": [
    SystemMessage(content=PRODUCT_PROMPT),
    HumanMessage(content="Hi! How are you?"),
]})
print_message_sequence(r_no_tool["messages"])
tool_calls_made = sum(1 for m in r_no_tool["messages"] if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
observe(f"Tool calls made: {tool_calls_made}  — LLM responded directly. No tool needed for a greeting.")
observe("Same agent, same tools bound — the LLM chose not to call any.")

section("Final answer for each")
print(f"  Product query  → {r_tool['messages'][-1].content[:100]}")
print(f"  Greeting query → {r_no_tool['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 3: Two Specialists — Same Skeleton, Different Soul
#  Goal: show that the support agent has completely different behaviour
#        even though it uses the identical subgraph structure.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 3 — Two Specialists: Same Structure, Different Behaviour")

print("""
  product_subgraph and support_subgraph share:
    • Identical StateGraph topology
    • Identical conditional edge logic (should_continue)

  They differ only in:
    • System prompt  (PRODUCT_PROMPT vs SUPPORT_PROMPT)
    • Bound tools    (search_product_catalog vs get_order_status + escalate_to_human)

  Same skeleton.  Different soul.
""")

section("Product agent — asked about an order (not its domain)")
r_wrong = product_subgraph.invoke({"messages": [
    SystemMessage(content=PRODUCT_PROMPT),
    HumanMessage(content="What is the status of order ORD102?"),
]})
print(f"  Product agent says: {r_wrong['messages'][-1].content[:150]}")
observe("Product agent doesn't have get_order_status — it will say it can't help with orders.")

section("Support agent — asked about the same order (its domain)")
r_right = support_subgraph.invoke({"messages": [
    SystemMessage(content=SUPPORT_PROMPT),
    HumanMessage(content="What is the status of order ORD102?"),
]})
print_message_sequence(r_right["messages"])
print(f"\n  Final answer: {r_right['messages'][-1].content[:200]}")
observe("Support agent has get_order_status — it called it and returned the real data.")

section("Support agent — asked to search products (not its domain)")
r_wrong2 = support_subgraph.invoke({"messages": [
    SystemMessage(content=SUPPORT_PROMPT),
    HumanMessage(content="Show me all Sony headphones"),
]})
print(f"  Support agent says: {r_wrong2['messages'][-1].content[:150]}")
observe("Support agent doesn't have search_product_catalog — it redirects or apologises.")


# ════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════

banner("STAGE 2 COMPLETE ✅")
print("""
  You have:
    ✅  Seen AgentState messages accumulate across model→tools→model
    ✅  Watched the LLM decide when to call a tool (and when not to)
    ✅  Built two specialist agents with the same structure but
        different prompts and tools

  Next → Stage 3: add an orchestrator to route between agents in parallel.
  Run:   python3 stage3/test_stage3.py
""")
