"""
nodes.py — All four graph nodes for Stage 3.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 3 CONCEPT: The Orchestrator Pattern                       ║
║                                                                  ║
║  Full graph topology:                                            ║
║                                                                  ║
║    START → orchestrator ─┬─ product_agent ──→ synthesizer → END  ║
║                          └─ support_agent ──↗                   ║
║                                                                  ║
║  NEW CONCEPTS in this stage:                                     ║
║                                                                  ║
║  1. Structured LLM Output                                        ║
║     llm.with_structured_output(ClassificationResult) makes the  ║
║     LLM return a typed Pydantic object rather than free text.   ║
║                                                                  ║
║  2. Send() — Parallel Dispatch                                   ║
║     Instead of returning a single next node, the orchestrator   ║
║     returns a list of Send() objects — one per agent to invoke. ║
║     LangGraph fans these out and runs them in parallel.          ║
║                                                                  ║
║  3. Command — Updating State + Routing Together                  ║
║     Command(update={...}, goto=[...]) is how a node can both     ║
║     write state fields AND specify the next node(s) in one call. ║
║                                                                  ║
║  4. Response Synthesis                                           ║
║     When two agents contribute, the synthesizer node calls the  ║
║     LLM to merge both answers into one coherent reply.          ║
║     When only one agent ran, it passes the answer through.       ║
╚══════════════════════════════════════════════════════════════════╝

📖 Docs:
  - with_structured_output → https://python.langchain.com/docs/concepts/structured_outputs/
  - LangGraph Send()       → https://langchain-ai.github.io/langgraph/concepts/low_level/#send
  - Command                → https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command
"""

from __future__ import annotations

from typing import Literal

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command, Send

from modules.stage3.state import AxiomCartState, ClassificationResult, WorkerInput

# Re-use Stage 2 subgraphs — same product and support agents
from modules.stage2.nodes import (
    PRODUCT_PROMPT,
    SUPPORT_PROMPT,
    product_subgraph,
    support_subgraph,
)
from modules.stage1.config import get_logger, llm

logger = get_logger("nodes")


# ── Utility: format prior conversation as context ────────────────────

def build_context(messages: list[AnyMessage]) -> str:
    """Format prior Human/AI turns as a brief context block."""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    if not parts:
        return ""
    return "PRIOR CONVERSATION:\n" + "\n".join(parts) + "\n\n"


# ═══════════════════════════════════════════════════════════════════
#  NODE 1: Orchestrator
#
#  Responsibility:
#    Read the user query → classify it → dispatch to agent(s).
#
#  Two new ideas here:
#
#  A. Structured output:
#       classifier = llm.with_structured_output(ClassificationResult)
#       result = classifier.invoke(prompt)   # returns ClassificationResult
#     The LLM is forced to fill in a validated Pydantic schema.
#     No parsing, no regex, no risk of missing fields.
#
#  B. Send() for parallel dispatch:
#       goto = [
#         Send("product_agent", {...}),
#         Send("support_agent", {...}),
#       ]
#     Both nodes start in parallel.  Results accumulate in
#     agent_results via the custom reducer in state.py.
# ═══════════════════════════════════════════════════════════════════

def orchestrator_node(
    state: AxiomCartState,
) -> Command[Literal["product_agent", "support_agent", "synthesizer"]]:
    """Classify the query and fan out to the right agent(s)."""

    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        user_query = state["messages"][-1].content

    logger.info("Orchestrator  query=%r", user_query)

    # ── Step A: Ask the LLM to classify the query ──────────────────
    prompt = (
        f'Analyse this customer query and decide which agent(s) should handle it.\n\n'
        f'QUERY: "{user_query}"\n\n'
        'AGENTS:\n'
        '  product_agent – product searches, recommendations, general conversation\n'
        '  support_agent – order status, complaints, human escalation\n\n'
        'RULES:\n'
        '1. Greetings / chitchat             → product_agent only\n'
        '2. Product-only queries             → product_agent only\n'
        '3. Order/support-only queries       → support_agent only\n'
        '4. Mixed (product + order/support)  → BOTH agents, requires_synthesis=true\n'
    )

    # with_structured_output() wraps the LLM call so it returns
    # a ClassificationResult object, not plain text.
    classifier = llm.with_structured_output(ClassificationResult)
    try:
        classification = classifier.invoke(prompt)
    except Exception:
        logger.exception("Classification failed — defaulting to product_agent")
        from modules.stage3.state import AgentTask
        classification = ClassificationResult(
            tasks=[AgentTask(agent="product_agent", task_description=user_query)],
            requires_synthesis=False,
            reasoning="Fallback: classification error",
        )

    logger.info(
        "  routing=%s  synthesis=%s  reason=%s",
        [t.agent for t in classification.tasks],
        classification.requires_synthesis,
        classification.reasoning,
    )

    # ── Step B: Build Send() targets ───────────────────────────────
    # Each Send("node_name", payload) queues one parallel invocation.
    # The payload becomes the WorkerInput the agent node receives.
    targets: list[Send] = []
    for task in classification.tasks:
        targets.append(Send(task.agent, {
            "messages": state.get("messages", []),
            "user_query": user_query,
            "task_description": task.task_description,
        }))

    if not targets:
        # Safety net: if classification returned no tasks, go to synthesizer
        targets = [Send("synthesizer", {})]

    # ── Step C: Return Command ──────────────────────────────────────
    # Command(update=..., goto=...) does two things in one call:
    #   update  — writes fields into AxiomCartState
    #   goto    — sets the next node(s) to execute
    return Command(
        update={
            "tasks": classification.tasks,
            "requires_synthesis": classification.requires_synthesis,
            "user_query": user_query,
            "agent_results": [],   # ← empty list triggers the reset in agent_results_reducer
        },
        goto=targets,
    )


# ═══════════════════════════════════════════════════════════════════
#  NODE 2: Product Agent
#
#  Wraps the product_subgraph from Stage 2 as a top-level graph node.
#  Receives a WorkerInput, runs the subgraph, stores the answer.
# ═══════════════════════════════════════════════════════════════════

def product_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the product-discovery agent and forward its answer."""
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Product Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = product_subgraph.invoke({"messages": [
        SystemMessage(content=PRODUCT_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "product_discovery", "response": answer}]},
        goto="synthesizer",
    )


# ═══════════════════════════════════════════════════════════════════
#  NODE 3: Support Agent
#
#  Same pattern as product_agent.  Wraps support_subgraph.
# ═══════════════════════════════════════════════════════════════════

def support_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the sales-support agent and forward its answer."""
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Support Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = support_subgraph.invoke({"messages": [
        SystemMessage(content=SUPPORT_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "sales_support", "response": answer}]},
        goto="synthesizer",
    )


# ═══════════════════════════════════════════════════════════════════
#  NODE 4: Synthesizer
#
#  Receives all agent_results and produces the final_answer.
#
#  Single-agent path: len(results) == 1 → pass through directly.
#  Multi-agent path:  len(results) >= 2 → LLM merges into one reply.
#
#  This keeps the user experience seamless — they always get one
#  coherent answer regardless of how many agents ran behind the scenes.
# ═══════════════════════════════════════════════════════════════════

def synthesizer_node(state: AxiomCartState) -> dict:
    """Merge agent responses into one final answer."""
    results    = state.get("agent_results", [])
    user_query = state.get("user_query", "")

    if not results:
        logger.warning("Synthesizer received no agent results")
        return {"final_answer": "Sorry, I couldn't process that request."}

    if len(results) == 1:
        # One agent ran — no merging needed
        logger.info("Synthesizer  single-agent pass-through")
        return {"final_answer": results[0]["response"]}

    # Multiple agents ran — merge with LLM
    logger.info("Synthesizer  merging %d responses", len(results))
    parts = "\n\n".join(
        f"[{r['source'].upper()}]:\n{r['response']}" for r in results
    )
    merge_prompt = (
        f"You are combining responses from multiple specialist agents.\n\n"
        f"CUSTOMER QUERY: {user_query}\n\n"
        f"AGENT RESPONSES:\n{parts}\n\n"
        "Write one coherent reply that addresses every part of the customer's "
        "query. Be concise. Speak as 'AxiomCart Assistant'."
    )
    merged = llm.invoke(merge_prompt)
    return {"final_answer": merged.content}
