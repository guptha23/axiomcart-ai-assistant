"""
config.py — Foundation: Logger, API clients, and settings.

╔══════════════════════════════════════════════════════════════════╗
║  STAGE 1 CONCEPT: The Power Supply                               ║
║                                                                  ║
║  Before any agent can think, it needs three things:              ║
║    1. A connection to the LLM  (ChatOpenAI)                      ║
║    2. A way to embed text      (OpenAIEmbeddings)                ║
║    3. A raw API client         (openai.OpenAI)                   ║
║                                                                  ║
║  This file is imported by everything else.  It must have         ║
║  zero dependencies on other project modules.                     ║
╚══════════════════════════════════════════════════════════════════╝

📖 Docs:
  - OpenAI Python SDK     → https://platform.openai.com/docs/libraries/python
  - langchain-openai      → https://python.langchain.com/docs/integrations/chat/openai/
  - python-dotenv         → https://pypi.org/project/python-dotenv/
"""

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

# ── Step 1: Load .env ────────────────────────────────────────────────
# python-dotenv reads key=value pairs from a .env file and injects them
# into os.environ.  This runs BEFORE we read any env vars.
#
# Why .env files?
#   • Secrets (API keys) never get committed to git.
#   • Different environments (dev / staging / prod) can have different keys.
#   • One file to change, all modules see the updated value.
load_dotenv()


# ── Step 2: Build a reusable logger ─────────────────────────────────
# A logger is better than print() because:
#   • You can see WHICH module logged a message (name column).
#   • You can filter by level (INFO vs WARNING vs ERROR).
#   • In production you'd swap StreamHandler for a file or cloud sink.
#
def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format.

    Usage:
        logger = get_logger(__name__)   # pass module name
        logger.info("Vector store ready  (%d docs)", len(docs))
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                # timestamp | module name (padded) | level | message
                "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger("config")


# ── Step 3: Validate the API key ────────────────────────────────────
# Fail fast — better to crash here with a clear message than fail
# silently inside a nested tool call 10 layers deep.
#
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
    logger.error(
        "OPENAI_API_KEY is missing or is a placeholder. "
        "Copy .env.example → .env and add your real key."
    )
    sys.exit(1)


# ── Step 4: Create the LLM clients ──────────────────────────────────
#
# openai_client  — raw OpenAI Python SDK client.
#   Used directly for: TTS (text-to-speech), Whisper (speech-to-text).
#   Why raw?  LangChain doesn't wrap TTS/Whisper, so we use the SDK.
#
# llm  — LangChain ChatOpenAI wrapper around the chat completion API.
#   Why LangChain?  It gives us message types (HumanMessage / AIMessage),
#   tool binding (.bind_tools()), and structured output (.with_structured_output()).
#   temperature=0.3 means slightly creative but mostly consistent answers.
#
# embeddings  — converts text to a vector of floats (meaning as numbers).
#   Used by RAG to index the product catalog and to search it.
#   text-embedding-3-small is fast, cheap, and good enough for this scale.
#
# ── Step 5: Read model from environment ─────────────────────────────
# LLM_MODEL can be set in .env to switch between models without touching code.
#   LLM_MODEL=gpt-3.5-turbo   ← affordable for development / testing
#   LLM_MODEL=gpt-4o           ← production-grade reasoning
#
# EMBEDDINGS_MODEL controls the embedding model for RAG.
# text-embedding-3-small is fast, cheap, and accurate enough for this scale.
LLM_MODEL        = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
llm        = ChatOpenAI(model=LLM_MODEL, temperature=0.3)
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

logger.info(
    "OpenAI clients initialised  (model: %s, embeddings: %s)",
    LLM_MODEL, EMBEDDINGS_MODEL,
)
