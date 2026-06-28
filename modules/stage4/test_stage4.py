"""
test_stage4.py — Run all Stage 4 concepts interactively.

Run from the project root:
    python3 stage4/test_stage4.py

What this file demonstrates:
  Concept 1 — MemorySaver: multi-turn conversation memory
  Concept 2 — interrupt(): graph pauses for missing info, resumes with answer
  Concept 3 — Full session: the complete AxiomCart experience
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.disable(logging.INFO)

from langchain.messages import HumanMessage
from langgraph.types import Command
from modules.stage4.graph import axiomcart_graph


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

def invoke(query: str, thread_id: str) -> dict:
    return axiomcart_graph.invoke(
        {"messages": [HumanMessage(content=query)], "user_query": query},
        {"configurable": {"thread_id": thread_id}},
    )


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 1: MemorySaver — Multi-Turn Conversation Memory
#  Goal: show that the graph remembers prior turns within one session,
#        enabling natural follow-up questions.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 1 — MemorySaver: Multi-Turn Memory")

print("""
  Without MemorySaver: each invoke() is stateless — the agent forgets
    everything between turns.

  With MemorySaver + a consistent thread_id: state is checkpointed
    after every node.  The next invoke() restores it.  The agent
    remembers the full conversation.
""")

section("Turn 1 — Ask about Sony products")
thread = "memory-demo"
r1 = invoke("What Sony products do you carry?", thread)
print(f"  Turn 1 answer: {r1['final_answer'][:150]}")

section("Turn 2 — Follow-up using 'the headphones' (no explicit name)")
r2 = invoke("What is the price of the headphones you just mentioned?", thread)
print(f"  Turn 2 answer: {r2['final_answer'][:200]}")
observe("Turn 2 answered correctly — the agent knew 'the headphones' referred to Sony XM5 from Turn 1.")
observe("This works because MemorySaver restored the messages list (including Turn 1) before Turn 2 ran.")

section("Same query in a NEW thread — no memory")
r_fresh = invoke("What is the price of the headphones you just mentioned?", "fresh-thread")
print(f"  Fresh thread answer: {r_fresh['final_answer'][:200]}")
observe("With a fresh thread_id, the agent has no context — it asks 'which headphones?' or gives a generic answer.")

section("How MemorySaver is added — just one line in graph.py")
print("""
  # Stage 3 (no memory):
  graph = builder.compile()

  # Stage 4 (with memory):
  from langgraph.checkpoint.memory import MemorySaver
  memory = MemorySaver()
  graph  = builder.compile(checkpointer=memory)
  
  That's the only change.  Same graph, same nodes, same edges.
""")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 2: interrupt() — Human-in-the-Loop Pause/Resume
#  Goal: show the graph literally pausing mid-execution, surfacing
#        a question, and resuming from exactly where it left off.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 2 — interrupt(): Graph Pauses and Resumes")

print("""
  When a customer asks "Where is my order?" without an order ID,
  the support agent can't call get_order_status() — it needs the ID first.

  interrupt(question) pauses the ENTIRE graph.
  The caller sees result["__interrupt__"] with the question.
  invoke(Command(resume=answer)) resumes from exactly where it paused.
""")

section("Step 1: invoke with no order ID — graph will pause")
hitl_thread = "hitl-demo"
r_paused = axiomcart_graph.invoke(
    {"messages": [HumanMessage(content="Where is my order?")],
     "user_query": "Where is my order?"},
    {"configurable": {"thread_id": hitl_thread}},
)

if "__interrupt__" in r_paused and r_paused["__interrupt__"]:
    question = r_paused["__interrupt__"][0].value
    print(f"  ✅ Graph paused.")
    print(f"  Agent asks: \"{question[:100]}\"")
    print(f"  final_answer in this result: {repr(r_paused.get('final_answer', 'EMPTY'))}")
    observe("The graph is now frozen. State is checkpointed in MemorySaver.")
    observe("result['final_answer'] is empty — the graph hasn't reached synthesizer_node yet.")
else:
    print(f"  Result: {r_paused.get('final_answer', '')[:150]}")
    observe("The agent answered directly — it may have had enough context. Try a vaguer query.")

section("Step 2: provide the order ID — graph resumes")
r_resumed = axiomcart_graph.invoke(
    Command(resume="ORD102"),
    {"configurable": {"thread_id": hitl_thread}},
)
print(f"  ✅ Graph resumed.")
print(f"  Final answer: {r_resumed.get('final_answer', '')[:250]}")
observe("The graph resumed from inside support_model_node, appended 'ORD102' as a HumanMessage,")
observe("then the LLM called get_order_status('ORD102') and produced the final answer.")

section("What interrupt() looks like in code")
print("""
  def support_model_node_hitl(state):
      response = support_llm.invoke(state["messages"])

      if not response.tool_calls:
          any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])
          if not any_tools_called:
              # ← GRAPH PAUSES HERE
              user_reply = interrupt(response.content)
              # ← GRAPH RESUMES HERE when Command(resume=...) is called
              return {"messages": [response, HumanMessage(content=user_reply)]}

      return {"messages": [response]}
""")
observe("interrupt() is a LangGraph primitive — not a try/except, not a callback. A first-class pause.")

section("Full HITL scenario: ask order status, then escalate")
hitl_thread2 = "hitl-demo-2"
r1 = axiomcart_graph.invoke(
    {"messages": [HumanMessage(content="I need help with my order")],
     "user_query": "I need help with my order"},
    {"configurable": {"thread_id": hitl_thread2}},
)
if "__interrupt__" in r1 and r1["__interrupt__"]:
    print(f"  Pause 1 — Agent asks: \"{r1['__interrupt__'][0].value[:80]}\"")
    r2 = axiomcart_graph.invoke(Command(resume="ORD103"), {"configurable": {"thread_id": hitl_thread2}})
    print(f"  After resume: {r2.get('final_answer', '')[:200]}")
else:
    print(f"  Direct answer: {r1.get('final_answer', '')[:200]}")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 3: Full Session Simulation
#  Goal: simulate a realistic multi-turn session that exercises
#        memory, HITL, and the full graph together.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 3 — Complete Session Simulation")

print("""
  Simulating a real customer session with all features active:
    Turn 1: product question        (product_agent, no HITL)
    Turn 2: follow-up question      (uses memory from Turn 1)
    Turn 3: order question, no ID   (support_agent → HITL → resume)
    Turn 4: mixed query             (both agents + synthesizer)
""")

session = "full-session-demo"

section("Turn 1 — 'Do you have noise-cancelling headphones?'")
r = invoke("Do you have noise-cancelling headphones?", session)
print(f"  Answer: {r['final_answer'][:180]}")

section("Turn 2 — 'Which one has the best battery life?' (follow-up)")
r = invoke("Which one has the best battery life?", session)
print(f"  Answer: {r['final_answer'][:180]}")
observe("Agent answered about headphones from Turn 1 — that's multi-turn memory working.")

section("Turn 3 — 'Where is my order?' (HITL triggers)")
r = axiomcart_graph.invoke(
    {"messages": [HumanMessage(content="Where is my order?")],
     "user_query": "Where is my order?"},
    {"configurable": {"thread_id": session}},
)
if "__interrupt__" in r and r["__interrupt__"]:
    print(f"  [PAUSE] Agent asks: {r['__interrupt__'][0].value[:80]}")
    r = axiomcart_graph.invoke(Command(resume="ORD102"), {"configurable": {"thread_id": session}})
    print(f"  [RESUME] Answer: {r.get('final_answer', '')[:200]}")
else:
    print(f"  Answer: {r.get('final_answer', '')[:200]}")

section("Turn 4 — Mixed: 'ORD102 is late. Any Sony headphone alternatives?'")
r = invoke(
    "Given ORD102 is delayed, can you show me alternative Sony headphones?",
    session,
)
print(f"  Answer: {r['final_answer'][:300]}")
observe("Mixed query: both agents ran in parallel. Synthesizer merged the results.")


# ════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════

banner("STAGE 4 COMPLETE ✅  —  COURSE COMPLETE 🎉")
print("""
  You have:
    ✅  Seen MemorySaver carry context across multiple turns
    ✅  Watched interrupt() pause the graph for missing info
    ✅  Used Command(resume=...) to continue from where the graph stopped
    ✅  Run a complete multi-turn session with all features combined

  You've built the full AxiomCart AI assistant from scratch:
    Stage 1  →  LLM + RAG + @tool
    Stage 2  →  ReAct agent loop + two specialists
    Stage 3  →  Orchestrator + parallel dispatch + synthesis
    Stage 4  →  HITL + memory + REPL + voice

  To run the interactive REPL:
    python3 -m stage4.main

  To run with voice (requires microphone):
    python3 -m stage4.main --voice
""")
