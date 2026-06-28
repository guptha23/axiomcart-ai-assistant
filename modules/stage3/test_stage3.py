"""
test_stage3.py — Run all Stage 3 concepts interactively.

Run from the project root:
    python3 stage3/test_stage3.py

What this file demonstrates:
  Concept 1 — Structured output: ClassificationResult from the LLM
  Concept 2 — Send() + parallel dispatch: single vs dual agent routing
  Concept 3 — Full graph end-to-end: product, support, and mixed queries
  Concept 4 — Graph topology: print the Mermaid diagram
"""

import os
import sys
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.disable(logging.INFO)

from langchain.messages import HumanMessage
from modules.stage1.config import llm
from modules.stage3.state import ClassificationResult
from modules.stage3.graph import axiomcart_graph


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

def run_graph(query: str, thread_id: str) -> dict:
    return axiomcart_graph.invoke(
        {"messages": [HumanMessage(content=query)], "user_query": query},
        {"configurable": {"thread_id": thread_id}},
    )


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 1: Structured LLM Output
#  Goal: show that with_structured_output() makes the LLM return a
#        typed Python object instead of free text.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 1 — Structured LLM Output (Pydantic)")

print("""
  Problem: if we ask the LLM "which agent should handle this query?",
           it might return anything — "product_agent", "the product one",
           "both agents", etc.  Parsing that is fragile.

  Solution: with_structured_output(ClassificationResult) forces the LLM
           to fill a Pydantic schema.  We get a typed Python object back.
""")

classifier = llm.with_structured_output(ClassificationResult)

test_queries = [
    ("Product only",  "Do you have Sony headphones under 15000?"),
    ("Support only",  "What is the status of my order ORD102?"),
    ("Mixed query",   "My order ORD102 is late. Also show me headphone alternatives."),
    ("Greeting",      "Hi! What can you help me with?"),
]

def classify(query: str) -> ClassificationResult:
    """Use the same rich prompt as orchestrator_node for reliable results."""
    prompt = (
        f'Analyse this customer query and decide which agent(s) should handle it.\n\n'
        f'QUERY: "{query}"\n\n'
        'AGENTS:\n'
        '  product_agent – product searches, recommendations, general conversation\n'
        '  support_agent – order status, complaints, human escalation\n\n'
        'RULES:\n'
        '1. Greetings / chitchat             → product_agent only\n'
        '2. Product-only queries             → product_agent only\n'
        '3. Order/support-only queries       → support_agent only\n'
        '4. Mixed (product + order/support)  → BOTH agents, requires_synthesis=true\n'
    )
    return classifier.invoke(prompt)

section("Classification results for 4 query types")
for label, query in test_queries:
    result = classify(query)
    agents = [t.agent for t in result.tasks]
    print(f"\n  [{label}]")
    print(f"    Query      : {query[:60]}")
    print(f"    → Agents   : {agents}")
    print(f"    → Synthesis: {result.requires_synthesis}")
    print(f"    → Reasoning: {result.reasoning[:80]}")

observe("The LLM returns ClassificationResult(tasks=[...], requires_synthesis=...) — no parsing needed.")
observe("For mixed queries, requires_synthesis=True triggers the synthesizer node.")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 2: Send() — Parallel Dispatch
#  Goal: show that both agents actually run for a mixed query,
#        and demonstrate timing to make parallelism concrete.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 2 — Send(): Parallel Dispatch")

print("""
  Command(goto=[Send("product_agent", ...), Send("support_agent", ...)])
  tells LangGraph to invoke both nodes simultaneously.

  The agent_results field accumulates results from both using the
  custom reducer in state.py.  The synthesizer runs once both have written.
""")

section("Timing: sequential estimate vs actual parallel execution")

# Time individual agents as proxies for sequential cost
t0 = time.time()
r_product_only = run_graph("Show me Sony headphones", "time-product")
t_product = time.time() - t0

t0 = time.time()
r_support_only = run_graph("Status of order ORD102?", "time-support")
t_support = time.time() - t0

t0 = time.time()
r_mixed = run_graph(
    "My order ORD102 is delayed. Also show me Sony headphone alternatives.",
    "time-mixed",
)
t_mixed = time.time() - t0

print(f"  Product-only query : {t_product:.1f}s  (1 agent)")
print(f"  Support-only query : {t_support:.1f}s  (1 agent)")
print(f"  Mixed query        : {t_mixed:.1f}s  (2 agents in parallel)")
print(f"  Sequential would be ≈ {t_product + t_support:.1f}s  (sum of both)")
observe("Mixed query takes ~max(product, support) time, not their sum — that's parallelism.")

section("agent_results field — what the synthesizer sees")
results = r_mixed.get("agent_results", [])
for r in results:
    print(f"\n  Source : {r['source']}")
    print(f"  Answer : {r['response'][:120]}...")
observe(f"{len(results)} agent(s) contributed results. Synthesizer merged them into one reply.")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 3: Full Graph — All Three Query Types
#  Goal: show the complete end-to-end pipeline for product, support,
#        and mixed queries.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 3 — Full Graph End-to-End")

print("""
  START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                        └─ support_agent ──↗

  The graph routes automatically based on the query.
""")

section("Query 1 — Product only: 'Show me MacBooks'")
r = run_graph("Show me MacBooks", "s3-e2e-1")
print(f"  Final answer: {r['final_answer'][:200]}")
print(f"  Agents used : {[t.agent for t in r.get('tasks', [])]}")

section("Query 2 — Support only: 'Where is order ORD101?'")
r = run_graph("Where is order ORD101?", "s3-e2e-2")
print(f"  Final answer: {r['final_answer'][:200]}")
print(f"  Agents used : {[t.agent for t in r.get('tasks', [])]}")

section("Query 3 — Mixed: 'ORD102 delayed. Show me headphone alternatives.'")
r = run_graph(
    "My order ORD102 is delayed. Can you show me similar headphone options?",
    "s3-e2e-3",
)
print(f"  Final answer:\n  {r['final_answer'][:350].replace(chr(10), chr(10) + '  ')}")
print(f"  Agents used : {[t.agent for t in r.get('tasks', [])]}")
observe("Synthesizer merged two separate responses into one coherent reply above.")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 4: Graph Topology
#  Goal: show the Mermaid diagram so learners can see the structure visually.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 4 — Graph Topology (Mermaid Diagram)")

print("""
  axiomcart_graph.get_graph().draw_mermaid() returns the graph as a
  Mermaid diagram string.  Paste it into https://mermaid.live to render it.
""")
mermaid = axiomcart_graph.get_graph().draw_mermaid()
print(mermaid)
observe("Each box is a node. Edges show the flow. Notice the fan-out from orchestrator.")


# ════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════

banner("STAGE 3 COMPLETE ✅")
print("""
  You have:
    ✅  Seen with_structured_output() return typed ClassificationResult objects
    ✅  Watched Send() dispatch two agents in parallel
    ✅  Verified the synthesizer merges multi-agent responses
    ✅  Run all three routing paths end-to-end
    ✅  Printed the live graph topology as a Mermaid diagram

  Next → Stage 4: add memory (MemorySaver), HITL (interrupt), and a REPL.
  Run:   python3 stage4/test_stage4.py
""")
