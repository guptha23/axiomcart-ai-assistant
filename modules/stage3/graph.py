"""
graph.py — Build and compile the full AxiomCart multi-agent graph.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 3 CONCEPT: Compiling the Graph                            ║
║                                                                  ║
║  After defining nodes and edges, compile() seals the graph into  ║
║  an executable Pregel runner.  Post-compilation you can:         ║
║    • invoke()        — run the graph synchronously               ║
║    • stream()        — stream events as they happen              ║
║    • get_graph()     — inspect / visualise the topology          ║
║                                                                  ║
║  Key edges in this graph:                                        ║
║    START → orchestrator (always)                                 ║
║    orchestrator → product_agent / support_agent (via Send())     ║
║    product_agent → synthesizer (always)                          ║
║    support_agent → synthesizer (always)                          ║
║    synthesizer → END (always)                                    ║
║                                                                  ║
║  NOTE: The orchestrator → agents edges are handled by Command    ║
║  (goto=Send(...)) inside orchestrator_node, not by add_edge().   ║
║  This is how dynamic dispatch works in LangGraph.                ║
╚══════════════════════════════════════════════════════════════════╝

📖 Docs:
  - StateGraph.compile() → https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.compile
  - Graph visualisation  → https://langchain-ai.github.io/langgraph/how-tos/visualization/
"""

from langgraph.graph import END, START, StateGraph

from modules.stage1.config import get_logger
from modules.stage3.nodes import (
    orchestrator_node,
    product_agent,
    support_agent,
    synthesizer_node,
)
from modules.stage3.state import AxiomCartState

logger = get_logger("graph")


def build_graph():
    """Create, wire, and compile the Stage 3 multi-agent graph.

    Topology:
        START → orchestrator ─┬─ product_agent ──→ synthesizer → END
                              └─ support_agent ──↗
    """
    builder = StateGraph(AxiomCartState)

    # Register nodes — each is a Python function
    builder.add_node("orchestrator",  orchestrator_node)
    builder.add_node("product_agent", product_agent)
    builder.add_node("support_agent", support_agent)
    builder.add_node("synthesizer",   synthesizer_node)

    # Static edges
    builder.add_edge(START,        "orchestrator")
    builder.add_edge("synthesizer", END)

    # NOTE: orchestrator → agents edges are dynamic (via Send() in Command)
    # so we do NOT add_edge for them here.
    # product_agent → synthesizer and support_agent → synthesizer are handled
    # by Command(goto="synthesizer") inside each node function.

    graph = builder.compile()
    logger.info("Stage 3 graph compiled")
    return graph


# Module-level singleton — imported by tests and Stage 4
axiomcart_graph = build_graph()
