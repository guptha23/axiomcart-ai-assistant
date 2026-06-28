"""
rag.py — Build a vector store from the product catalog.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 1 CONCEPT: RAG — Retrieval-Augmented Generation           ║
║                                                                  ║
║  The problem:                                                    ║
║    An LLM knows a lot about the world, but nothing about         ║
║    *your* product catalog.  You could paste the catalog into     ║
║    every prompt — but that wastes tokens and hits context limits. ║
║                                                                  ║
║  The solution: RAG                                               ║
║    1. EMBED   each product description → a vector of numbers     ║
║               (a point in high-dimensional space)                ║
║    2. STORE   all vectors in a vector database (ChromaDB)        ║
║    3. SEARCH  at query time: embed the user's question and       ║
║               return the products whose vectors are closest      ║
║                                                                  ║
║  Think of it as Google Search for your data, powered by          ║
║  meaning rather than keywords.                                   ║
╚══════════════════════════════════════════════════════════════════╝

How embeddings capture meaning
───────────────────────────────
  "wireless headphones"  →  [0.12, -0.45, 0.78, ...]   (1536 numbers)
  "Bose QuietComfort 45" →  [0.11, -0.43, 0.81, ...]   (very similar!)
  "table fan"            →  [-0.62, 0.33, -0.11, ...]  (very different)

  Similarity is measured with cosine distance:
    cos(θ) close to 1  → very similar meaning
    cos(θ) close to 0  → unrelated

📖 Docs:
  - OpenAI Embeddings → https://platform.openai.com/docs/guides/embeddings
  - LangChain Chroma  → https://python.langchain.com/docs/integrations/vectorstores/chroma/
  - ChromaDB          → https://docs.trychroma.com/
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from modules.stage1.config import embeddings, get_logger
from modules.stage1.data import PRODUCT_CATALOG

logger = get_logger("rag")


# ── Step 1: Convert product dicts into LangChain Documents ──────────
#
# A Document has two parts:
#   page_content  — the text that gets embedded (what the LLM searches)
#   metadata      — structured fields for filtering (not embedded)
#
# We deliberately put all human-readable information in page_content
# so the embedding captures it.  Price, brand, category etc. also go
# in metadata for future filtering (e.g. "under ₹5000").
#
def _build_documents() -> list[Document]:
    """Convert every catalog entry into a LangChain Document."""
    docs: list[Document] = []
    for p in PRODUCT_CATALOG:
        # Flatten all fields into a readable string — this is what gets embedded.
        # Rich text = richer semantic representation.
        content = (
            f"Product: {p['name']}\n"
            f"Brand: {p['brand']}\n"
            f"Category: {p['category']}\n"
            f"Price: ₹{p['price']}\n"
            f"Rating: {p['rating']}/5\n"
            f"Features: {', '.join(p['features'])}\n"
            f"Description: {p['description']}\n"
            f"Colors: {', '.join(p['colors'])}\n"
            f"In Stock: {'Yes' if p['in_stock'] else 'No'}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "id": p["id"],
                    "name": p["name"],
                    "brand": p["brand"],
                    "category": p["category"],
                    "price": p["price"],
                    "rating": p["rating"],
                },
            )
        )
    return docs


# ── Step 2: Build the vector store ─────────────────────────────────
#
# Chroma.from_documents():
#   1. Takes each Document's page_content
#   2. Calls the embeddings model (text-embedding-3-small) for each one
#   3. Stores the (vector, metadata, content) triples in memory
#
# collection_name is just a label — useful if you have multiple stores.
#
def build_vectorstore() -> Chroma:
    """Create an in-memory ChromaDB collection from the product catalog."""
    docs = _build_documents()
    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="axiomcart_products",
    )
    logger.info("Vector store ready  (%d products indexed)", len(docs))
    return store


# ── Step 3: Module-level singleton ──────────────────────────────────
# Created once when the module is first imported.
# Every importer shares the same store — no duplicate API calls.
product_vectorstore = build_vectorstore()
