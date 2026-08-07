"""
Core Primitives for the Cogito Agent.

Provides foundational types and state management for the agent's cognitive architecture.
"""

from .state import AgentState, CogitoState
from .types import (
    Entity,
    Relation,
    KnowledgeDelta,
    Confidence,
    ConfidenceScored,
    ComprehensionResult,
    ReasoningResult,
    ConsolidationResult,
)
from .prompts import Prompts

__all__ = [
    # State
    "AgentState",
    "CogitoState",
    # Types
    "Entity",
    "Relation",
    "KnowledgeDelta",
    "Confidence",
    "ConfidenceScored",
    "ComprehensionResult",
    "ReasoningResult",
    "ConsolidationResult",
    # Prompts
    "Prompts",
]
