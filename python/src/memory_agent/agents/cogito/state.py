# src/memory_agent/agents/cogito/state.py

import time

from enum import StrEnum, auto
from dataclasses import dataclass, field

from typing import Any

from ..base import BaseState, AgentStatus, StateDict
from ..react import ReactStateProtocol
from .memory import WorkingMemory


class CognitivePhase(StrEnum):
    COMPREHEND = auto()
    RECALL = auto()
    REASON = auto()
    REACT = auto()
    REQUEST = auto()
    REPLY = auto()


# COGITO STATE
@dataclass
class CogitoState(BaseState, ReactStateProtocol):
    """
    Runtime state of the Cogito cognitive agent.

    CogitoState owns the execution state of the cognitive process
    and the current-turn WorkingMemory.

    Long-term symbolic and episodic memory remain owned by CogitoAgent.
    """

    # Cognitive execution
    current_step: int = 0
    current_phase: CognitivePhase = CognitivePhase.COMPREHEND

    # Current-turn cognitive workspace
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)

    # Execution metrics
    turn_start_time: float = 0.0
    total_execution_time: float = 0.0
    total_tool_time: float = 0.0

    # Error recovery
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3
    last_error: str | None = None

    # Clarification lifecycle
    awaiting_clarification: bool = False

    # Turn management
    def start_turn(self, reset_working_memory: bool = True) -> None:
        """Begin a new cognitive turn."""

        self.current_turn += 1
        self.current_phase = CognitivePhase.COMPREHEND
        self.turn_start_time = time.perf_counter()
        self.awaiting_clarification = False

        if reset_working_memory:
            self.working_memory.reset()

        self.reset_errors()

    def resume_after_clarification(self, response: str) -> None:
        """Resume reasoning after the user answered a clarification request."""
        self.working_memory.clarification_response = response
        self.working_memory.next_action = None
        self.working_memory.final_answer = ""

        self.awaiting_clarification = False
        self.current_phase = CognitivePhase.REASON

        self.current_turn = 0
        self.turn_start_time = time.perf_counter()

        self.reset_errors()

    def end_turn(self) -> None:
        """Finish the current execution turn and record duration."""

        if self.turn_start_time:
            self.total_execution_time += time.perf_counter() - self.turn_start_time

        self.turn_start_time = 0.0

    def increment_step(self) -> None:
        self.current_step += 1

    def record_thought(
        self, thought: str, metadata: dict[str, Any] | None = None
    ) -> None:
        del metadata
        self.working_memory.reasoning = thought

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str | None = None,
        duration: float | None = None,
    ) -> None:
        """Record tool execution metadata in working memory."""

        self.current_phase = CognitivePhase.REACT

        calls = self.working_memory.retrieved_knowledge.setdefault(
            "tool_calls",
            [],
        )

        calls.append(
            {
                "tool": tool_name,
                "args": dict(args),
                "result": result,
                "duration": duration,
            }
        )

        if duration is not None:
            self.total_tool_time += duration

    def record_observation(
        self, observation: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a ReAct observation."""

        self.current_phase = CognitivePhase.REACT

        observations = self.working_memory.retrieved_knowledge.setdefault(
            "observations",
            [],
        )

        observations.append(
            {
                "content": observation,
                "metadata": dict(metadata or {}),
            }
        )

    def record_result(self, result: str) -> None:
        """Record the result of a ReAct execution."""
        self.working_memory.retrieved_knowledge["react_result"] = result

    def record_error(self, error: str) -> None:
        """Record an execution error."""
        self.last_error = error
        self.consecutive_errors += 1

    def reset_errors(self) -> None:
        """Reset consecutive error tracking."""
        self.consecutive_errors = 0
        self.last_error = None

    def should_abort(self) -> bool:
        """Return True when consecutive errors exceed the configured limit."""
        return self.consecutive_errors >= self.max_consecutive_errors

    # Lifecycle
    def complete(self) -> None:
        self.status = AgentStatus.COMPLETED

    def fail(self) -> None:
        self.status = AgentStatus.ERROR

    # Serialization
    def to_dict(self) -> StateDict:
        """Serialize Cogito runtime state."""

        data = super().to_dict()

        data.update(
            {
                "current_phase": self.current_phase.value,
                "current_step": self.current_step,
                "working_memory": self.working_memory.to_dict(),
                "turn_start_time": self.turn_start_time,
                "total_execution_time": self.total_execution_time,
                "total_tool_time": self.total_tool_time,
                "consecutive_errors": self.consecutive_errors,
                "max_consecutive_errors": self.max_consecutive_errors,
                "last_error": self.last_error,
                "awaiting_clarification": self.awaiting_clarification,
            }
        )

        return data

    @classmethod
    def from_dict(cls, data: StateDict) -> "CogitoState":
        """Deserialize Cogito runtime state."""

        state = super().from_dict(data)

        if not isinstance(state, cls):
            raise TypeError(f"Expected {cls.__name__}, " f"got {type(state).__name__}")

        state.current_step = data.get("current_step", 0)
        state.current_phase = CognitivePhase(
            data.get(
                "current_phase",
                CognitivePhase.COMPREHEND.value,
            )
        )

        if "working_memory" in data:
            state.working_memory = WorkingMemory.from_dict(data["working_memory"])

        state.turn_start_time = data.get("turn_start_time", 0.0)
        state.total_execution_time = data.get(
            "total_execution_time",
            0.0,
        )
        state.total_tool_time = data.get(
            "total_tool_time",
            0.0,
        )

        state.consecutive_errors = data.get(
            "consecutive_errors",
            0,
        )
        state.max_consecutive_errors = data.get(
            "max_consecutive_errors",
            3,
        )
        state.last_error = data.get("last_error")
        state.awaiting_clarification = data.get("awaiting_clarification", False)

        return state

    # Reset
    def reset(self, keep_metadata: bool = False) -> None:
        """Reset the complete Cogito runtime state."""

        super().reset(keep_metadata=keep_metadata)

        self.current_phase = CognitivePhase.COMPREHEND
        self.current_step = 0

        self.working_memory.reset()

        self.turn_start_time = 0.0
        self.total_execution_time = 0.0
        self.total_tool_time = 0.0

        self.consecutive_errors = 0
        self.last_error = None

        self.awaiting_clarification = False
