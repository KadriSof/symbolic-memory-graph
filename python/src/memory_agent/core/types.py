"""
Core data types for the Cogito Agent.

Internal Domain Models (DTOs): Defines the primitive types used throughout the agent's cognitive pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

from memory_graph import Node, Edge


# Confidence
@dataclass
class Confidence:
    """
    Confidence scoring for knowledge.

    Attributes:
        score: Confidence value (0.0 to 1.0)
        source: Origin of the confidence ("llm", "tool", "user", "inference")
        explanation: Optional reasoning for the confidence score
        timestamp: When the confidence was recorded
    """

    score: float
    source: str = "llm"
    explanation: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate confidence score."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0, got {self.score}"
            )

    @classmethod
    def high(cls, source: str = "llm", explanation: str | None = None) -> "Confidence":
        """Create a high confidence score (0.9)."""
        return cls(score=0.9, source=source, explanation=explanation)

    @classmethod
    def medium(
        cls, source: str = "llm", explanation: str | None = None
    ) -> "Confidence":
        """Create a medium confidence score (0.7)."""
        return cls(score=0.7, source=source, explanation=explanation)

    @classmethod
    def low(cls, source: str = "llm", explanation: str | None = None) -> "Confidence":
        """Create a low confidence score (0.5)."""
        return cls(score=0.5, source=source, explanation=explanation)

    @classmethod
    def speculative(
        cls, source: str = "inference", explanation: str | None = None
    ) -> "Confidence":
        """Create a speculative confidence score (0.3)."""
        return cls(score=0.3, source=source, explanation=explanation)

    @classmethod
    def certain(
        cls, source: str = "user", explanation: str | None = None
    ) -> "Confidence":
        """Create a certain confidence score (1.0)."""
        return cls(score=1.0, source=source, explanation=explanation)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "source": self.source,
            "explanation": self.explanation,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Confidence":
        """Create from dictionary."""
        return cls(
            score=data["score"],
            source=data.get("source", "llm"),
            explanation=data.get("explanation"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class ConfidenceScored:
    """A value with an associated confidence score."""

    value: Any
    confidence: Confidence

    def __post_init__(self) -> None:
        """Ensure confidence is a Confidence object."""
        if not isinstance(self.confidence, Confidence):
            if isinstance(self.confidence, (int, float)):
                self.confidence = Confidence(score=self.confidence)
            else:
                raise TypeError(
                    f"confidence must be Confidence or number, got {type(self.confidence)}"
                )

    @property
    def score(self) -> float:
        """Get the confidence score."""
        return self.confidence.score


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

    id: str
    label: str
    type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: Confidence = field(default_factory=lambda: Confidence(score=0.7))

    def __post_init__(self) -> None:
        """Ensure confidence is a Confidence object."""
        if not isinstance(self.confidence, Confidence):
            self.confidence = Confidence(score=self.confidence)

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.confidence, Confidence):
            confidence = self.confidence.to_dict()
        else:
            confidence = self.confidence

        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "metadata": self.metadata,
            "confidence": confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create from dictionary."""
        confidence = data.get("confidence")
        if isinstance(confidence, dict):
            confidence = Confidence.from_dict(confidence)
        else:
            confidence = Confidence(score=confidence or 0.7)

        return cls(
            id=data["id"],
            label=data["label"],
            type=data.get("type"),
            metadata=data.get("metadata", {}),
            confidence=confidence,
        )

    def to_node(self) -> Node:
        """Convert to a MemoryGraph Node."""
        metadata = self.metadata.copy()
        if self.type:
            metadata["type"] = self.type
        if self.confidence.score:
            metadata["confidence"] = self.confidence.score
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

    source: str
    target: str
    label: str
    direction: str = "asymmetric"
    weight: float = 1.0
    confidence: Confidence = field(default_factory=lambda: Confidence(score=0.7))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize fields."""
        if self.direction not in ["symmetric", "asymmetric"]:
            raise ValueError(
                f"direction must be 'symmetric' or 'asymmetric', got {self.direction}"
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be between 0.0 and 1.0, got {self.weight}")
        if not isinstance(self.confidence, Confidence):
            self.confidence = Confidence(score=self.confidence)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "direction": self.direction,
            "weight": self.weight,
            "confidence": self.confidence.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        confidence = data.get("confidence")
        if isinstance(confidence, dict):
            confidence = Confidence.from_dict(confidence)
        else:
            confidence = Confidence(score=confidence or 0.7)

        return cls(
            source=data["source"],
            target=data["target"],
            label=data["label"],
            direction=data.get("direction", "asymmetric"),
            weight=data.get("weight", 1.0),
            confidence=confidence,
            metadata=data.get("metadata", {}),
        )

    def is_symmetric(self) -> bool:
        """Check if the relation is symmetric."""
        return self.direction == "symmetric"


# Knowledge Delta
@dataclass
class KnowledgeDelta:
    """
    What changed in the knowledge graph.

    Represents the difference between the previous state and the new state.
    """

    new_nodes: list[Node] = field(default_factory=list)
    new_edges: list[Edge] = field(default_factory=list)
    modified_nodes: list[Node] = field(default_factory=list)
    modified_edges: list[Edge] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    confidence: Confidence = field(default_factory=lambda: Confidence(score=0.7))
    source: str = "llm"

    def __post_init__(self) -> None:
        """Ensure confidence is a Confidence object."""
        if not isinstance(self.confidence, Confidence):
            self.confidence = Confidence(score=self.confidence)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(
            self.new_nodes
            or self.new_edges
            or self.modified_nodes
            or self.modified_edges
        )

    @property
    def total_additions(self) -> int:
        """Get the total number of additions."""
        return len(self.new_nodes) + len(self.new_edges)

    @property
    def total_modifications(self) -> int:
        """Get the total number of modifications."""
        return len(self.modified_nodes) + len(self.modified_edges)

    def merge(self, other: "KnowledgeDelta") -> "KnowledgeDelta":
        """Merge two deltas."""
        return KnowledgeDelta(
            new_nodes=self.new_nodes + other.new_nodes,
            new_edges=self.new_edges + other.new_edges,
            modified_nodes=self.modified_nodes + other.modified_nodes,
            modified_edges=self.modified_edges + other.modified_edges,
            conflicts=self.conflicts + other.conflicts,
            gaps=self.gaps + other.gaps,
            confidence=Confidence(
                score=(self.confidence.score + other.confidence.score) / 2
            ),
            source=f"{self.source}+{other.source}",
        )

    @classmethod
    def empty(cls) -> "KnowledgeDelta":
        """Create an empty delta."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "new_nodes": [n.to_json() for n in self.new_nodes],
            "new_edges": [e.to_json() for e in self.new_edges],
            "modified_nodes": [n.to_json() for n in self.modified_nodes],
            "modified_edges": [e.to_json() for e in self.modified_edges],
            "conflicts": self.conflicts,
            "gaps": self.gaps,
            "confidence": self.confidence.to_dict(),
            "source": self.source,
        }


# Result Types
@dataclass
class ComprehensionResult:
    """Result of the unified comprehension step."""

    reconstructed_query: str
    intent: str
    path: str
    context: str
    entities: list[Entity]
    relations: list[Relation]
    confidence: Confidence = field(default_factory=lambda: Confidence(score=0.7))

    def __post_init__(self) -> None:
        """Ensure confidence is a Confidence object."""
        if not isinstance(self.confidence, Confidence):
            self.confidence = Confidence(score=self.confidence)

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
            reconstructed_query=query,
            intent="ask",
            path="REACT",
            context="",
            entities=[],
            relations=[],
            confidence=Confidence(score=0.3),
        )


@dataclass
class ReasoningResult:
    """Result of the reasoning step."""

    reasoning_trace: str
    solution: str
    confidence: Confidence = field(default_factory=lambda: Confidence(score=0.7))
    tool_results: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Ensure confidence is a Confidence object."""
        if not isinstance(self.confidence, Confidence):
            self.confidence = Confidence(score=self.confidence)


@dataclass
class ConsolidationResult:
    """Result of the consolidation step."""

    current_knowledge: str
    new_information: list[dict[str, Any]]
    gaps: list[str]
    updates: list[dict[str, Any]]
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    confidence: Confidence = field(default_factory=lambda: Confidence(score=0.7))

    def __post_init__(self) -> None:
        """Ensure confidence is a Confidence object."""
        if not isinstance(self.confidence, Confidence):
            self.confidence = Confidence(score=self.confidence)
