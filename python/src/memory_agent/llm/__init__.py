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

__all__ = [
    "BaseLLM",
    "GenerationConfig",
    "GroqLLM",
    "LlamaCppLLM",
]
