"""
Memory Agent Package

Provides the Cogito Agent — a symbolic memory-augmented agent
that thinks through the graph, not just the prompt.

Components:
- CogitoAgent: Main agent with symbolic memory
- ReactAgent: Fallback ReAct agent
- Core types: Entity, Relation, KnowledgeDelta, Confidence
- LLM providers: Groq, LlamaCpp
- Tools: Tool abstraction
"""

# Agents
from .agents.cogito_agent import CogitoAgent
from .agents.react_agent import ReactAgent
from .agents.base import BaseAgent

# Core Types
from .core import (
    # Prompts
    Prompts,
    # State
    AgentState,
    CogitoState,
    # Types
    Entity,
    Relation,
    KnowledgeDelta,
    ComprehensionResult,
    ReasoningResult,
    # Schemas
    EntitySchema,
    RelationSchema,
    ComprehensionSchema,
    ConsolidationSchema,
)

# LLM
from .llm import BaseLLM, GenerationConfig, GroqLLM, LlamaCppLLM

# Tools
from .tools import Tool

# Utils
from .utils import (
    JSONExtractor,
    JSONRepair,
    JSONSanitizer,
    StructuredOutputParser,
    parse_structured_output,
    parse_structured_output_factory,
)

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    # Prompts
    "Prompts",
    # Agents
    "CogitoAgent",
    "ReactAgent",
    "BaseAgent",
    # Core Types
    "AgentState",
    "CogitoState",
    # DTOs
    "Entity",
    "Relation",
    "ComprehensionResult",
    "KnowledgeDelta",
    "ReasoningResult",
    # LLM Models
    "EntitySchema",
    "RelationSchema",
    "ComprehensionSchema",
    "ConsolidationSchema",
    # LLM
    "BaseLLM",
    "GenerationConfig",
    "GroqLLM",
    "LlamaCppLLM",
    # Tools
    "Tool",
    # Utils
    "JSONExtractor",
    "JSONRepair",
    "JSONSanitizer",
    "StructuredOutputParser",
    "parse_structured_output",
    "parse_structured_output_factory",
]

# Package Info
__author__ = "Mohamed Sofiene KADRI"
__description__ = "Symbolic memory-augmented agent for LLMs"
