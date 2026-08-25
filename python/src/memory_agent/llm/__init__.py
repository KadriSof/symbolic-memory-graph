"""
LLM Providers Package

Provides abstract and concrete LLM implementations:
- BaseLLM: Abstract interface
- GenerationConfig: Generation parameters
- GroqLLM: Groq API client
- LlamaCppLLM: Local llama.cpp client
"""

import logging
from .base import BaseLLM, GenerationConfig
from .summarization import summarize_conversation

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    from .groq import GroqLLM
except ImportError as e:
    logger.debug(f"GroqLLM not available: {e}")
    GroqLLM = None

try:
    from .llama_cpp import LlamaCppLLM
except ImportError as e:
    logger.debug(f"LlamaCppLLM not available: {e}")
    LlamaCppLLM = None

try:
    from .openrouter import OpenRouterLLM
except ImportError as e:
    logger.debug(f"OpenRouter not available: {e}")
    OpenRouterLLM = None

__all__ = [
    "BaseLLM",
    "GenerationConfig",
    "GroqLLM",
    "LlamaCppLLM",
    "summarize_conversation",
]
