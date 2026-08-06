"""
Agent State Management.

Provides clean, focused state tracking for the Cogito Agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentState:
    """
    Core state shared by all agents.

    Attributes:
        messages: Conversation history
        thoughts: Reasoning traces
        tool_calls: Record of tool executions
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def add_thought(self, thought: str) -> None:
        """Add a reasoning thought."""
        self.thoughts.append(thought)

    def add_tool_call(self, tool_name: str, kwargs: dict[str, Any]) -> None:
        """Record a tool call."""
        self.tool_calls.append(
            {
                "tool": tool_name,
                "kwargs": kwargs,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def last_message(self) -> dict[str, str] | None:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def reset(self) -> None:
        """Reset the state."""
        self.messages.clear()
        self.thoughts.clear()
        self.tool_calls.clear()


@dataclass
class CogitoState(AgentState):
    """
    Extended state for the Cogito Agent.

    Adds symbolic memory tracking, mode management, and confidence history.
    """

    # Mode tracking
    current_mode: str = "REACT"
    mode_history: list[dict[str, str]] = field(default_factory=list)

    # Context tracking
    active_context: str | None = None
    context_format: str = "linearized"
    session_entities: list[str] = field(default_factory=list)

    # Memory tracking
    graph_snapshots: list[dict[str, Any]] = field(default_factory=list)

    # Confidence
    confidence_threshold: float = 0.6
    confidence_history: list[float] = field(default_factory=list)

    # Mode Management
    def set_mode(self, new_mode: str, reason: str) -> None:
        """Switch mode with a reason for traceability."""
        self.mode_history.append(
            {
                "from": self.current_mode,
                "to": new_mode,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.current_mode = new_mode

    def is_cogito_mode(self) -> bool:
        """Check if currently in COGITO mode."""
        return self.current_mode == "COGITO"

    # Context Management
    def set_context(self, context: str) -> None:
        """Set the active context."""
        self.active_context = context

    def set_context_format(self, format: str) -> None:
        """Set the context representation format."""
        valid_formats = ["linearized", "triples", "json", "cypher", "narrative"]
        if format in valid_formats:
            self.context_format = format
        else:
            raise ValueError(f"Invalid format: {format}. Choose from {valid_formats}")

    def add_session_entity(self, entity_id: str) -> None:
        """Add an entity to the session tracking."""
        if entity_id not in self.session_entities:
            self.session_entities.append(entity_id)

    # Memory Tracking
    def add_graph_snapshot(self, nodes: int, edges: int) -> None:
        """Record a graph snapshot for traceability."""
        self.graph_snapshots.append(
            {
                "nodes": nodes,
                "edges": edges,
                "timestamp": datetime.now().isoformat(),
            }
        )

    # Confidence Management
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for updates."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))

    def add_confidence(self, confidence: float) -> None:
        """Add a confidence score to history."""
        self.confidence_history.append(max(0.0, min(1.0, confidence)))

    # Serialization
    def to_dict(self) -> dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        return {
            "messages": self.messages,
            "thoughts": self.thoughts,
            "tool_calls": self.tool_calls,
            "current_mode": self.current_mode,
            "mode_history": self.mode_history,
            "active_context": self.active_context,
            "context_format": self.context_format,
            "session_entities": self.session_entities,
            "graph_snapshots": self.graph_snapshots,
            "confidence_threshold": self.confidence_threshold,
            "confidence_history": self.confidence_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogitoState":
        """Create a state from a dictionary."""
        state = cls()
        state.messages = data.get("messages", [])
        state.thoughts = data.get("thoughts", [])
        state.tool_calls = data.get("tool_calls", [])
        state.current_mode = data.get("current_mode", "REACT")
        state.mode_history = data.get("mode_history", [])
        state.active_context = data.get("active_context", None)
        state.context_format = data.get("context_format", "linearized")
        state.session_entities = data.get("session_entities", [])
        state.graph_snapshots = data.get("graph_snapshots", [])
        state.confidence_threshold = data.get("confidence_threshold", 0.6)
        state.confidence_history = data.get("confidence_history", [])
        return state
