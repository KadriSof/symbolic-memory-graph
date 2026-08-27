"""
Core data types for the Cogito Agent.

Internal Domain Models (DTOs): Defines the primitive types used throughout the agent's cognitive pipeline.
"""

import re

from dataclasses import dataclass, field
from typing import Any

from memory_graph import Node, Edge, EdgeType


# HELPER FUNCS:
def normalize_id(text: str) -> str:
    """Deterministically convert a label into a valid graph ID."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# Entity
@dataclass
class Entity:
    """
    An entity extracted from text.

    Attributes:
        id: Unique identifier for the entity
        label: Human-readable label
        type: Entity type (optional)
        metadata: Additional metadata
        confidence: Confidence score for this entity
    """

    label: str
    type: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    # Auto-generated ID
    id: str = field(init=False)

    def __post_init__(self):
        """Auto-generate ID from label if not provided."""
        self.id = normalize_id(self.label)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "metadata": self.metadata,
            "confidence": self.confidence,
        }

    def to_node(self) -> Node:
        """Convert to a MemoryGraph Node."""
        metadata = self.metadata.copy()
        if self.type:
            metadata["type"] = self.type
        metadata["confidence"] = self.confidence
        return Node(self.id, self.label, metadata)


# Relation
@dataclass
class Relation:
    """
    A relation between two entities.

    Attributes:
        source: Source entity ID
        target: Target entity ID
        label: Relation label
        direction: "symmetric" or "asymmetric"
        weight: Relation weight (0.0 to 1.0)
        confidence: Confidence score for this relation
        metadata: Additional metadata
    """

    label: str
    source_label: str
    target_label: str
    direction: str = "asymmetric"
    weight: float = 0.5
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    # Auto-generated IDs
    source: str = field(init=False)
    target: str = field(init=False)
    edge_id: str = field(init=False)

    def __post_init__(self):
        # Normalize source/target to match Entity IDs
        self.source = normalize_id(self.source_label)
        self.target = normalize_id(self.target_label)
        self.edge_id = f"{self.source}::{normalize_id(self.label)}::{self.target}"

        if self.direction not in ["symmetric", "asymmetric"]:
            self.direction = "asymmetric"
        if not 0.0 <= self.weight <= 1.0:
            self.weight = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "direction": self.direction,
            "weight": self.weight,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_edge(self) -> Edge:
        """Convert to MemoryGraph Edge."""
        if self.direction == "symmetric":
            edge_type = EdgeType.SYMMETRIC
            connections = {self.source, self.target}
        else:
            edge_type = EdgeType.ASYMMETRIC
            connections = (self.source, self.target)

        edge_id = f"{self.source}::{self.label}::{self.target}"
        metadata = self.metadata.copy()
        metadata["confidence"] = self.confidence
        return Edge(edge_id, self.label, edge_type, connections, self.weight, metadata)


# Knowledge Delta
@dataclass
class KnowledgeDelta:
    """
    What changed in the knowledge graph.

    Represents the difference between the previous state and the new state.
    """

    new_entities: list[Entity] = field(default_factory=list)
    new_relations: list[Relation] = field(default_factory=list)
    modified_entities: list[Entity] = field(default_factory=list)
    modified_relations: list[Relation] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source: str = "llm"

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(
            self.new_entities
            or self.new_relations
            or self.modified_entities
            or self.modified_relations
        )

    def to_nodes(self) -> list[Node]:
        """Convert new entities to C++ Node objects."""
        return [e.to_node() for e in self.new_entities]

    def to_edges(self) -> list[Edge]:
        """Convert new relations to C++ Edge objects."""
        return [r.to_edge() for r in self.new_relations]

    def to_modified_nodes(self) -> list[Node]:
        """Convert modified entities to C++ Node objects."""
        return [e.to_node() for e in self.modified_entities]

    def to_modified_edges(self) -> list[Edge]:
        """Convert modified relations to C++ Edge objects."""
        return [r.to_edge() for r in self.modified_relations]

    @classmethod
    def empty(cls) -> "KnowledgeDelta":
        """Create an empty delta."""
        return cls()


# Result Types
@dataclass
class ComprehensionResult:
    """Result of the unified comprehension step."""

    goal: str
    plan: list[str]

    intent: str
    path: str
    context: str
    entities: list[Entity]
    relations: list[Relation]
    confidence: float = 0.5

    def is_cogito(self) -> bool:
        """Check if the path is COGITO."""
        return self.path.upper() == "COGITO"

    def is_react(self) -> bool:
        """Check if the path is REACT."""
        return self.path.upper() == "REACT"

    @classmethod
    def fallback(cls, query: str) -> "ComprehensionResult":
        """Create a fallback result for REACT path."""
        return cls(
            goal=query,
            plan=[],
            intent="ask",
            path="REACT",
            context="general",
            entities=[],
            relations=[],
            confidence=0.0,
        )


@dataclass
class ReasoningResult:
    """Result of the reasoning step."""

    reasoning_trace: str
    solution: str
    confidence: float = 0.5
    tool_results: list[dict[str, Any]] | None = None
