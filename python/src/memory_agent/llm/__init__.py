# python/src/memory_agent/llm/__init__.py
"""
LLM providers for the Cogito Agent.

Available providers:
- GroqLLM: Cloud-based Groq API (requires GROQ_API_KEY)
- LlamaCppLLM: Local inference with llama.cpp (requires GGUF model)

Usage:
    from memory_agent.llm import GroqLLM, LlamaCppLLM

    # Cloud provider
    llm = GroqLLM(model="llama-3.1-8b-instant")

    # Local provider
    llm = LlamaCppLLM(model_path="~/models/mistral-7b-instruct.gguf")
"""

import logging

from .base import BaseLLM, GenerationConfig

logger = logging.getLogger(__name__)

try:
    from .groq import GroqLLM
except ImportError as e:
    logger.debug(f"[memory_agent:llm] GroqLLM not available:\n---\n{e}\n---\n")
    GroqLLM = None

try:
    from .llama_cpp import LlamaCppLLM
except ImportError as e:
    logger.debug(f"[memory_agent:llm] LlamaCppLLM not available:\n---\n{e}\n---\n")
    LlamaCppLLM = None

__all__ = [
    "BaseLLM",
    "GenerationConfig",
    "GroqLLM",
    "LlamaCppLLM",
]
