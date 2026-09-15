# src/memory_agent/agents/react/state.py

import time
import json
import copy

from datetime import datetime

from collections import deque
from dataclasses import dataclass, field

from typing import Any, Protocol, runtime_checkable

from ..base import Message, MessageRole, AgentStatus, BaseState


# TOOL CALL RECORD & TRAJECTORY STEP
@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any]
    result: str | None
    timestamp: str
    turn: int
    step: int
    duration: float | None = None


@dataclass
class TrajectoryStep:
    turn: int
    step: int

    thought: str | None = None
    action: str | None = None
    action_args: dict[str, Any] | None = None
    observation: str | None = None
    result: str | None = None
    error: str | None = None

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        parts = [f"Turn {self.turn}, Step {self.step}:"]

        if self.thought:
            parts.append(f"Thought: {self.thought}")

        if self.action:
            parts.append(f"Action: {self.action}")

        if self.action_args:
            parts.append(f"Input: {json.dumps(self.action_args, ensure_ascii=False)}")

        if self.observation:
            parts.append(f"Observation: {self.observation}")

        if self.result:
            parts.append(f"Result: {self.result}")

        if self.error:
            parts.append(f"Error: {self.error}")

        return " | ".join(parts)


# REACT STATE
@dataclass
class ReactState(BaseState):
    """
    ReAct-specific execution state.
    Implements both BaseState and ReactProtocol.
    """

    # Memory
    thoughts: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    observations: deque = field(default_factory=deque)

    max_observations: int = 30

    # ReAct execution
    current_step: int = 0
    trajectory: list[TrajectoryStep] = field(default_factory=list)

    # Performance
    turn_start_time: float | None = None
    total_execution_time: float = 0.0
    total_tool_time: float = 0.0

    # Error recovery
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3
    last_error: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations = deque(
            self.observations,
            maxlen=self.max_observations,
        )

    # Turn / Step Management
    def start_turn(self) -> None:
        """Begin a new agent invocation."""
        self.current_turn += 1
        self.current_step = 0
        self.turn_start_time = time.perf_counter()

    def end_turn(self) -> None:
        """Complete the current agent invocation."""
        if self.turn_start_time is not None:
            self.total_execution_time += time.perf_counter() - self.turn_start_time

        self.turn_start_time = None

    def increment_step(self) -> None:
        """Advance to the next ReAct reasoning step."""
        self.current_step += 1

    # Trajectory
    def _get_or_create_trajectory_step(self) -> TrajectoryStep:
        """Return the trajectory entry for the current step."""
        if self.trajectory:
            last = self.trajectory[-1]

            if last.turn == self.current_turn and last.step == self.current_step:
                return last

        step = TrajectoryStep(
            turn=self.current_turn,
            step=self.current_step,
        )

        self.trajectory.append(step)

        return step

    # Thought Management
    def record_thought(
        self,
        thought: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a reasoning thought."""
        self.thoughts.append(
            {
                "content": thought,
                "timestamp": datetime.now().isoformat(),
                "turn": self.current_turn,
                "step": self.current_step,
                **(metadata or {}),
            }
        )

        trajectory_step = self._get_or_create_trajectory_step()
        trajectory_step.thought = thought

    def retrieve_recent_thoughts(
        self,
        limit: int = 5,
    ) -> list[str]:
        """Get recent thought contents."""
        return [thought["content"] for thought in self.thoughts[-limit:]]

    # Tool Call Management
    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str | None = None,
        duration: float | None = None,
    ) -> None:
        """Record a tool execution."""
        self.tool_calls.append(
            ToolCallRecord(
                tool=tool_name,
                args=copy.deepcopy(args),
                result=result,
                timestamp=datetime.now().isoformat(),
                turn=self.current_turn,
                step=self.current_step,
                duration=duration,
            )
        )

        if duration is not None:
            self.total_tool_time += duration

        trajectory_step = self._get_or_create_trajectory_step()
        trajectory_step.action = tool_name
        trajectory_step.action_args = args

    def retrieve_tool_calls(
        self,
        tool_name: str | None = None,
    ) -> list[ToolCallRecord]:
        """Get tool call history, optionally filtered by tool."""
        if tool_name:
            return [call for call in self.tool_calls if call.tool == tool_name]

        return self.tool_calls

    # Observation Management
    def record_observation(
        self,
        observation: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a tool observation."""
        entry = {
            "turn": self.current_turn,
            "step": self.current_step,
            "observation": observation,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        self.observations.append(entry)

        trajectory_step = self._get_or_create_trajectory_step()
        trajectory_step.observation = observation

    def retrieve_observations(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve recent observations."""
        observations = list(self.observations)

        if limit is not None:
            observations = observations[-limit:]

        return observations

    # Result / Error Management
    def record_result(self, result: str) -> None:
        """Record the final result for the current step."""
        trajectory_step = self._get_or_create_trajectory_step()
        trajectory_step.result = result

    def record_error(self, error: str) -> None:
        """Record an execution error."""
        self.consecutive_errors += 1
        self.last_error = error

        trajectory_step = self._get_or_create_trajectory_step()
        trajectory_step.error = error

    def reset_errors(self) -> None:
        """Reset error recovery state."""
        self.consecutive_errors = 0
        self.last_error = None

    def should_abort(self) -> bool:
        """Return whether execution should be aborted."""
        return self.consecutive_errors >= self.max_consecutive_errors

    # Trajectory Retrieval
    def get_trajectory_summary(
        self,
        limit: int = 5,
    ) -> str:
        """Return a compact representation of recent trajectory steps."""
        if not self.trajectory:
            return "No trajectory yet."

        recent = self.trajectory[-limit:]

        return "\n".join(step.summary() for step in recent)

    # Persistence
    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        data = super().to_dict()

        data.update(
            {
                "thoughts": self.thoughts,
                "tool_calls": [call.__dict__ for call in self.tool_calls],
                "observations": list(self.observations),
                "max_observations": self.max_observations,
                "current_turn": self.current_turn,
                "current_step": self.current_step,
                "trajectory": [step.__dict__ for step in self.trajectory],
                "total_execution_time": self.total_execution_time,
                "total_tool_time": self.total_tool_time,
                "consecutive_errors": self.consecutive_errors,
                "max_consecutive_errors": self.max_consecutive_errors,
                "last_error": self.last_error,
            }
        )

        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ReactState":
        """Deserialize state."""
        state = super().from_dict(data)

        if not isinstance(state, ReactState):
            raise TypeError("Expected ReactState")

        state.thoughts = data.get("thoughts", [])
        state.tool_calls = [
            ToolCallRecord(**call) for call in data.get("tool_calls", [])
        ]
        state.observations = deque(
            data.get("observations", []),
            maxlen=state.max_observations,
        )
        state.current_turn = data.get("current_turn", 0)
        state.current_step = data.get("current_step", 0)
        state.trajectory = [
            TrajectoryStep(**step) for step in data.get("trajectory", [])
        ]
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
        state.last_error = data.get("last_error")

        return state

    # Reset
    def reset(
        self,
        keep_metadata: bool = False,
    ) -> None:
        """Reset the complete ReAct execution state."""
        super().reset(keep_metadata)

        self.thoughts.clear()
        self.tool_calls.clear()
        self.observations.clear()
        self.trajectory.clear()

        self.current_turn = 0
        self.current_step = 0

        self.turn_start_time = None
        self.total_execution_time = 0.0
        self.total_tool_time = 0.0

        self.consecutive_errors = 0
        self.last_error = None
