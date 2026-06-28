"""
test_stage1.py — Run all Stage 1 concepts interactively.

Run from the project root:
    python3 stage1/test_stage1.py

What this file demonstrates:
  Concept 1 — LLM client: basic invocation
  Concept 2 — RAG: embed the catalog, run a semantic search
  Concept 3 — @tool: all three tools called standalone (no graph needed)
"""

import os
import sys
import logging

# ── Make sure project root is on sys.path ──────────────────────────
# This lets you run: python3 stage1/test_stage1.py from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Suppress noisy log output so test sections are readable
logging.disable(logging.INFO)

from modules.stage1.config import llm, embeddings
from modules.stage1.rag import product_vectorstore
from modules.stage1.tools import search_product_catalog, get_order_status, escalate_to_human


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


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 1: LLM Connection
#  Goal: verify the OpenAI client works and understand what ChatOpenAI is.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 1 — LLM Connection")

print("""
  ChatOpenAI wraps the OpenAI chat API.
  .invoke() sends a message and returns an AIMessage object.
  .content is the plain-text response.
""")

section("Basic LLM call")
user_prompt = "In one sentence, what is an AI assistant?"
response = llm.invoke(user_prompt)
print(f"  User Prompt  : {user_prompt}")
print(f"  Response : {response.content}")
print(f"  Type     : {type(response).__name__}")

observe("The response is an AIMessage, not a raw string.")
observe("Try changing the prompt to see how temperature=0.3 affects creativity.")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 2: RAG — Retrieval-Augmented Generation
#  Goal: show that semantic search finds products by *meaning*, not keywords.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 2 — RAG: Semantic Search")

print("""
  The product catalog is embedded into a vector store at startup.
  At query time, we embed the search query and return the products
  whose vectors are most similar (cosine distance).
""")

section("How many products are indexed?")
count = product_vectorstore._collection.count()
print(f"  Products in vector store : {count}")
observe(f"Each of the {count} products was embedded to a 1536-dimensional vector.")

user_prompt = "wireless headphones"
section(f"Semantic search — {user_prompt}")
docs = product_vectorstore.similarity_search(user_prompt, k=3)
for i, doc in enumerate(docs, 1):
    name  = doc.metadata["name"]
    price = doc.metadata["price"]
    print(f"  [{i}] {name}  ₹{price:,}")
observe(f"The query '{user_prompt}' matched audio products by meaning — no keyword overlap needed.")

user_prompt = "budget earbuds under 2000"
section(f"Semantic search — {user_prompt}")
docs2 = product_vectorstore.similarity_search(user_prompt, k=2)
for i, doc in enumerate(docs2, 1):
    name  = doc.metadata["name"]
    price = doc.metadata["price"]
    print(f"  [{i}] {name}  ₹{price:,}")
observe(f"The query '{user_prompt}' matched boAt Airdopes at ₹1,299 — price semantics work too.")

section("What does a raw embedding look like?")
user_prompt = "wireless headphones"
sample_vector = embeddings.embed_query(user_prompt)
print(f"  Dimensions : {len(sample_vector)}")
print(f"  First 5    : {[round(v, 4) for v in sample_vector[:5]]}")
observe("1536 numbers represent the meaning of 'wireless headphones' in a high-dimensional space.")


# ════════════════════════════════════════════════════════════════════
#  CONCEPT 3: @tool — Wrapping Functions for the LLM
#  Goal: show that tools work completely standalone — no graph needed.
#        The LLM will call these later; here we call them directly.
# ════════════════════════════════════════════════════════════════════

banner("CONCEPT 3 — The @tool Decorator")

print("""
  @tool reads the function's docstring + type hints and generates a
  JSON schema that describes it to the LLM.  But the function is still
  a plain Python function — you can .invoke() it directly for testing.
""")

section("Tool schema — what the LLM sees")
schema = search_product_catalog.args_schema.model_json_schema()
print(f"  Name   : {search_product_catalog.name}")
print(f"  Desc   : {search_product_catalog.description[:80]}...")
print(f"  Args   : {list(schema.get('properties', {}).keys())}")
observe("The LLM sees this schema and decides when/how to call the tool.")

section("Tool 1: search_product_catalog — product query")
result = search_product_catalog.invoke({"query": "Sony headphones"})
lines = result.strip().split("\n")
for line in lines[:8]:
    print(f"  {line}")
print("  ...")
observe("Tool returns raw text — the LLM will format this into a natural response.")

section("Tool 2: get_order_status — by order ID")
result = get_order_status.invoke({"identifier": "ORD102"})
print(f"  {result.replace(chr(10), chr(10) + '  ')}")
observe("Try 'ORD-102', 'ord102', or '102' — normalise_order_id handles all of them.")

section("Tool 2: get_order_status — by email")
result = get_order_status.invoke({"identifier": "rahul.sharma@example.com"})
print(f"  {result.replace(chr(10), chr(10) + '  ')}")
observe("The tool accepts either an order ID OR an email — the LLM picks the right form.")

section("Tool 3: escalate_to_human — create a support ticket")
result = escalate_to_human.invoke({
    "order_id": "ORD102",
    "issue_summary": "Package delayed, customer requesting refund",
    "priority": "high",
})
print(f"  {result.replace(chr(10), chr(10) + '  ')}")
observe("Ticket ID is random — in production this would write to a database.")


# ════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════

banner("STAGE 1 COMPLETE ✅")
print("""
  You have:
    ✅  Connected to GPT-3.5-turbo via ChatOpenAI
    ✅  Built a ChromaDB vector store from the product catalog
    ✅  Run semantic searches that return results by meaning
    ✅  Called all three @tool functions standalone

  Next → Stage 2: wire these tools into a LangGraph agent loop.
  Run:   python3 stage2/test_stage2.py
""")
