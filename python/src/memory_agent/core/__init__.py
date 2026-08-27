"""
Core Primitives for the Cogito Agent.

Provides foundational types and state management for the agent's cognitive architecture.
"""

from .prompts import Prompts
from .state import AgentState, CogitoState
from .memory import WorkingMemory, SymbolicMemory, EpisodicMemory
from .orchestrator import (
    ProcedureNode,
    TransitionEdge,
    WorkflowOrchestrator,
    CogitoStatus,
)
from .types import (
    Entity,
    Relation,
    KnowledgeDelta,
    ComprehensionResult,
    KnowledgeDelta,
    ReasoningResult,
)
from .schemas import (
    EntitySchema,
    RelationSchema,
    ComprehensionSchema,
    ConsolidationSchema,
)
from .prompts import Prompts

__all__ = [
    # Prompts
    "Prompts",
    # State
    "AgentState",
    "CogitoState",
    # Memory
    "WorkingMemory",
    "SymbolicMemory",
    "EpisodicMemory",
    # Orchestrator
    "ProcedureNode",
    "TransitionEdge",
    "WorkflowOrchestrator",
    "CogitoStatus"
    # Types
    "Entity",
    "Relation",
    "KnowledgeDelta",
    "ComprehensionResult",
    "ReasoningResult",
    # Schemas
    "EntitySchema",
    "RelationSchema",
    "ComprehensionSchema",
    "ConsolidationSchema",
]
