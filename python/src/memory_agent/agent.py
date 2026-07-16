"""
Cogito Agent - Symbolic Memory-Augmented Agent

The Cogito Patter extends ReAct by making Symbolic memory intrinsic to reasoning.
The agent thinks through the graph, not just the prompt.
"""

import json
import logging

from datetime import datetime

from enum import Enum
from dataclasses import dataclass, field
from typing import Any

from memory_graph import (
    MemoryGraph,
    Node,
    Edge,
    EdgeType,
    SymmetricConnections,
    AsymmetricConnections,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathType(Enum):
    """The reasoning path the agent can take."""

    REACT = "react"  # Simple ReAct: think -> act -> respond
    COGITO = "cogito"  # Full Cogito: comprehend -> retrieve -> consolidate -> reason -> update -> respond


class Confidence(Enum):
    """Confidence levels for knowledge."""

    CERTAIN = 1.0
    HIGH = 0.9
    MEDIUM = 0.7
    LOW = 0.5
    SPECULATIVE = 0.3


@dataclass
class Primitive:
    """A graph primitive extracted from text."""

    primitive_type: str  # "entity", "relation", "attribute"
    data: dict[str, Any]
    confidence: float = 0.7
    source: str = "llm_extraction"


@dataclass
class intent:
    """The intent and complexity of a user query."""

    action: str  # "ask", "task", "clarify", "correct"
    entities: list[str] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    requires_tools: bool = False
    complexity: str = "simple"  # "simple", "moderate", "complex"


@dataclass
class ReasoningResult:
    """Result of the reasoning step."""

    reasoning_trace: str
    solution: str
    tool_results: list[dict[str, Any]] | None = None
    confidence: float = 1.0
