"""
ReAct Agent - Production Implementation

Reasoning + Acting pattern with robust tool execution, state management,
error recovery, and observability.

Key improvements over base implementation:
- Proper stateful execution with token management
- Structured parsing with validation
- Comprehensive error recovery
- Performance monitoring
- Streaming support
- Clean separation of concerns
"""

from __future__ import annotations

import re
import time
import json
import copy

from enum import Enum
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field

from typing import Any

from .base import (
    AgentStatus,
    BaseAgent,
    BaseState,
    AgentError,
    MessageRole,
    ToolExecutionError,
)

from ..llm.base import BaseLLM
from ..tools import Tool


# DATA MODELS
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

    Tracks:
    - reasoning thoughts
    - tool executions
    - observations
    - ReAct trajectory
    - turn/step counters
    - execution timing
    - error recovery
    """

    # Memory
    thoughts: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    observations: deque = field(default_factory=deque)

    max_observations: int = 30

    # ReAct execution
    current_turn: int = 0
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
        self.status = AgentStatus.RUNNING

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
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

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

        trajectory_step = self._get_or_create_trajectory_step()
        trajectory_step.action = tool_name
        trajectory_step.action_args = args

    def retrieve_tool_calls(
        self,
        tool_name: str | None = None,
    ) -> list[ToolCallRecord]:
        """Get tool call history, optionally filtered by tool."""
        if tool_name is None:
            return list(self.tool_calls)

        return [call for call in self.tool_calls if call.tool == tool_name]

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

        if limit is None:
            return observations

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        return observations[-limit:]

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

        self.status = AgentStatus.ERROR

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


# Constants and Markers:
class ReactMarker(str, Enum):
    """Standardized ReAct markers."""

    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    ACTION_INPUT = "ACTION_INPUT"
    OBSERVATION = "OBSERVATION"
    FINAL_ANSWER = "FINAL_ANSWER"

    @property
    def open_tag(self) -> str:
        return f"<{self.value}>"

    @property
    def close_tag(self) -> str:
        return f"</{self.value}>"

    @classmethod
    def extract(cls, text: str, marker: "ReactMarker") -> str | None:
        """Extract content between markers."""
        pattern = (
            rf"{re.escape(marker.open_tag)}" rf"(.*?)" rf"{re.escape(marker.close_tag)}"
        )
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @classmethod
    def has_marker(cls, text: str, marker: "ReactMarker") -> bool:
        """Check if marker exists in text."""
        return marker.open_tag in text and marker.close_tag in text


# PARSED ACTION:
@dataclass(frozen=True)
class ParsedAction:
    """Type-safe representation of a parsed ReAct action."""

    thought: str
    action: str
    action_input: dict[str, Any]
    raw_text: str

    @classmethod
    def from_text(cls, text: str) -> "ParsedAction | None":
        """
        Parsed ReAct response into structured action.

        Returns None if parsing fails.
        """

        # Extract thought
        thought = ReactMarker.extract(text, ReactMarker.THOUGHT)
        if thought is None:
            return None

        # Extract action
        action = ReactMarker.extract(text, ReactMarker.ACTION)
        if action is None:
            return None

        # Extract action input with robust JSON parsing
        action_input_raw = ReactMarker.extract(text, ReactMarker.ACTION_INPUT)
        if action_input_raw is None:
            return None

        try:
            # Robust JSON extraction
            action_input = cls._parse_json(action_input_raw)
        except json.JSONDecodeError:
            return None

        return cls(
            thought=thought,
            action=action.strip(),
            action_input=action_input,
            raw_text=text,
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        value = json.loads(text.strip())

        if not isinstance(value, dict):
            raise json.JSONDecodeError("Action_INPUT must be a JSON object", text, 0)

        return value

    @property
    def is_final_answer(self) -> bool:
        """Check if this is a final answer action."""
        return self.action.lower() in {"final_answer", "final answer"}


# REACT AGENT
class ReactAgent(BaseAgent):
    """
    ReAct Agent implementation.

    Features:
    - Structured state management
    - Comprehensive error recovery
    - Performance monitoring
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool] | None = None,
        config: dict[str, Any] | None = None,
        state: ReactState | None = None,
    ) -> None:
        """
        Initialize ReAct Agent.

        Args:
            llm: Language model instance
            tools: List of available tools
            config: Configuration dict with keys:
                - max_steps: Maximum reasoning steps (default: 10)
                - max_retries: Max retries per step (default: 2)
                - debug: Enable debug logging (default: False)
        """
        super().__init__(llm, tools, config, state)

        # ReAct specific config
        self.max_steps = self.config.get("max_steps", 10)
        self.max_retries = self.config.get("max_retries", 3)

        # Performance tracking
        self._tool_times: dict[str, list[float]] = {}

    def _initialize_state(self) -> BaseState:
        return ReactState()

    @property
    def state(self) -> ReactState:
        """Get state with correct type."""
        state = super().state

        if not isinstance(state, ReactState):
            raise TypeError("ReactAgent requires ReactState")

        return state

    @classmethod
    def _state_class(cls) -> type[BaseState]:
        return ReactState

    # PROMPT CONSTRUCTION:
    def _build_system_prompt(self) -> str:
        """
        Build comprehensive system prompt with tool descriptions.

        Includes:
        - Role definition
        - Tool descriptions
        - Format specification
        - Examples
        - Error handling guidance
        """
        has_tools = bool(self.tools)

        # Tool section
        if has_tools:
            tool_descriptions = self._format_tool_descriptions()
            tool_section = self._build_tool_section(tool_descriptions)
        else:
            tool_section = self._build_no_tool_section()

        # Format specification
        format_section = self._build_format_section(has_tools)

        # Examples section
        example_section = self._build_examples(has_tools)

        # Rules
        rules_section = self._build_rules(has_tools)

        return f"""
        You are a ReAct (Reasoning + Acting) agent designed to solve problems through iterative reasoning and tool use.

        ## YOUR CAPABILITIES
        {tool_section}

        ## REACT FORMAT
        {format_section}

        ## EXAMPLES
        {example_section}

        ## RULES
        {rules_section}

        ## RESPONSE FORMAT
        {self._build_response_format()}

        Now, process the user's request using the ReAct framework.
        """

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt."""
        descriptions = []
        for tool in self.tools.values():
            tool_schema = tool.schema()
            desc = f"   - {tool_schema['name']}: {tool_schema['description']}"

            if "input_schema" in tool_schema:
                desc += f"\n    Input Schema: {json.dumps(tool_schema['input_schema'], indent=2)}"

            descriptions.append(desc)

        return "\n".join(descriptions)

    def _build_tool_section(self, descriptions: str) -> str:
        """Build tool section for prompt."""
        return f"""
        You have access to the following tools:
        {descriptions}

        When you need to use a tool:
        1. First reason about why you need it
        2. Then specify the exact tool to use
        3. Provide valid JSON arguments
        4. Wait for the observation before continuing
        """

    def _build_no_tool_section(self) -> str:
        """Build section for no-tool scenario."""
        return """
        You have NO tools available for this task.

        IMPORTANT: 
        - You must rely solely on your knowledge and reasoning
        - Do not attempt to use tools or mention tool names
        - If you lack information, state this honestly
        - Never fabricate information or results
        """

    def _build_format_section(self, has_tools: bool) -> str:
        """Build format specification."""
        if has_tools:
            return """
            Format your responses using these markers:

            <THOUGHT>concise reasoning summary / decision rationale</THOUGHT>
            <ACTION>The tool name to execute</ACTION>
            <ACTION_INPUT>{"parameter": "value"}</ACTION_INPUT>

            After executing the tool, you'll receive an observation. You can then:
            - Continue with another action
            - Or provide the final answer

            To provide the final answer:
            <THOUGHT>I now have enough information</THOUGHT>
            <FINAL_ANSWER>Your complete answer to the user</FINAL_ANSWER>
            """
        else:
            return """
            Format your responses using these markers:

            <THOUGHT>concise reasoning summary / decision rationale</THOUGHT>
            <FINAL_ANSWER>final answer to the user</FINAL_ANSWER>

            Since you have no tools, you must directly answer the user's question.
            """

    def _build_examples(self, has_tools: bool) -> str:
        """Build examples for the prompt."""
        if has_tools:
            return """
            Example with tools:
            User: "What's the weather in Tunis?"
            <THOUGHT>I need to check the current weather in Tunis using the weather tool</THOUGHT>
            <ACTION>get_weather</ACTION>
            <ACTION_INPUT>{"city": "Tunis"}</ACTION_INPUT>

            [After receiving observation]
            <THOUGHT>The weather in Tunis is sunny and 37°C. I have enough information to answer</THOUGHT>
            <FINAL_ANSWER>The current weather in Tunis is sunny with a temperature of 37°C.</FINAL_ANSWER>
            """
        else:
            return """
            Example without tools:
            User: "Who wrote the novel '1984'?"
            <THOUGHT>This is a literature question about the author of 1984</THOUGHT>
            <FINAL_ANSWER>George Orwell wrote the novel '1984'.</FINAL_ANSWER>

            Example when you don't know:
            User: "What is the current stock price of AAPL?"
            <THOUGHT>I don't have access to real-time stock data without tools</THOUGHT>
            <FINAL_ANSWER>I cannot provide current stock prices as I don't have access to real-time market data. Please check a financial website for this information.</FINAL_ANSWER>
            """

    def _build_rules(self, has_tools: bool) -> str:
        """Build rules section."""
        rules = [
            "1. Provide a concise decision rationale",
            "2. Never invent or hallucinate information",
            "3. Use exactly one action per response",
        ]

        if has_tools:
            rules.extend(
                [
                    "4. Always use valid JSON for <ACTION_INPUT>",
                    "5. Wait for observations before making new decisions",
                    "6. If a tool fails, inspect the error observation and recover when possible.",
                ]
            )
        else:
            rules.append(
                "4. Provide complete, accurate answers based on your knowledge"
            )

        return "\n".join(rules)

    def _build_response_format(self) -> str:
        """Build final response format guidance."""
        return """
        Your response must contain either:
        1. A <THOUGHT> and <ACTION> with <ACTION_INPUT> (if tools are available)
        2. A <THOUGHT> and <FINAL_ANSWER> 

        DO NOT include anything outside these markers.
        """

    # CORE EXECUTION:
    def _process_query_impl(self, query: str) -> str:
        start_time = time.perf_counter()

        self.logger.info(
            "Starting ReAct execution: %s",
            query[:100],
        )

        self.state.start_turn()

        try:
            for _ in range(self.max_steps):
                self.state.increment_step()

                try:
                    result = self._execute_step()

                    if result is not None:
                        self.state.record_result(result)
                        self._log_performance(start_time)
                        return result

                    self.state.reset_errors()

                except AgentError as exc:
                    error = str(exc)
                    self.state.record_error(str(exc))

                    observation = self._format_error_observation(error)

                    self.logger.warning(
                        f"Step {self.state.current_step} failed: {exec}"
                    )

                    self.state.record_message(
                        MessageRole.TOOL,
                        observation,
                        metadata={
                            "turn": self.state.current_turn,
                            "step": self.state.current_step,
                            "error": True,
                        },
                    )

            raise AgentError(f"Agent exceeded maximum steps ({self.max_steps})")

        finally:
            self.state.end_turn()

    def _execute_step(self) -> str | None:
        step = self.state.current_step
        self.on_step_start(step)

        response = self._reason()

        self.state.record_message(
            MessageRole.ASSISTANT,
            response,
            metadata={
                "turn": self.state.current_turn,
                "step": step,
            },
        )

        parsed = self._parse_response(response)

        if parsed is None:
            raise AgentError(self._format_parse_error(response))

        self.state.record_thought(parsed.thought)

        if parsed.is_final_answer:
            answer = self._extract_final_answer(response)

            if not answer:
                raise AgentError("Final answer marker was present but empty.")

            return answer

        observation = self._execute_action(parsed)

        self.state.record_observation(observation)

        self.state.record_message(
            MessageRole.TOOL,
            observation,
            metadata={
                "tool": parsed.action,
                "turn": self.state.current_turn,
                "step": step,
            },
        )

        self.state.status = AgentStatus.RUNNING

        self.on_step_end(
            step,
            observation,
        )

        return None

    # REASONING
    def _reason(self) -> str:
        """
        Get reasoning from LLM.

        Args:
            step: Current step number

        Returns:
            LLM response string
        """
        messages = super()._prepare_messages()
        self.state.status = AgentStatus.THINKING

        # Call LLM with retries
        try:
            response = self._llm_call(messages)

            self._log_debug(f"LLM response:\n{response}")

            if not response.strip():
                raise AgentError("LLM returned and empty response")

            return response

        except AgentError as e:
            self.logger.warning(f"LLM call failed: {e}")
            raise

    # RESPONSE PARSING
    def _parse_response(self, response: str) -> ParsedAction | None:
        """
        Parse LLM response into structured action.

        Handles:
        - Standard ReAct format
        - Final answer detection
        - Malformed responses
        - Missing markers
        """
        thought = ReactMarker.extract(response, ReactMarker.THOUGHT)
        final_answer = ReactMarker.extract(response, ReactMarker.FINAL_ANSWER)

        if final_answer is not None:
            if thought is None:
                return None

            return ParsedAction(
                thought=thought,
                action="final_answer",
                action_input={},
                raw_text=response,
            )

        # Action
        try:
            parsed = ParsedAction.from_text(response)
            if parsed:
                self.logger.info(
                    f"Parsed action: {parsed.action} with {parsed.action_input}"
                )
                return parsed

        except Exception as e:
            self.logger.warning(f"Failed to parse response: {e}")

        return None

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        answer = ReactMarker.extract(response, ReactMarker.FINAL_ANSWER)

        if answer is None:
            raise AgentError("Missing FINAL_ANSWER marker.")

        return answer

    def _format_error_observation(self, error: str) -> str:
        return (
            f"Tool/execution error: {error}\n"
            "Reconsider the previous action and continue if recovery is possible."
        )

    def _format_parse_error(self, response: str) -> str:
        """Format parsing error for feedback."""
        return f"""
        <OBSERVATION>Error: Could not parse your response.

        Your response: {response[:200]}

        Please use the correct format:
        1. Start with <THOUGHT> your reasoning </THOUGHT>
        2. Then use either:
        - <ACTION>tool_name</ACTION> with <ACTION_INPUT>{{"param": "value"}}</ACTION_INPUT>
        - Or <FINAL_ANSWER>your answer</FINAL_ANSWER>

        Make sure all markers are properly closed.
        </OBSERVATION>
        """

    # ACTION EXECUTION
    def _execute_action(self, parsed: ParsedAction) -> str:
        """
        Execute a parsed action with robust handling.

        Args:
            parsed: Parsed action to execute

        Returns:
            Observation string
        """
        # persist in state
        self.state.status = AgentStatus.ACTING

        observation = self._execute_tool(parsed.action, parsed.action_input)

        return observation

    # TOOL EXECUTION
    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """
        Execute a tool with validation and error handling.

        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments

        Returns:
            Tool execution result as string

        Raises:
            ToolExecutionError: If tool execution fails
        """
        tool = self.tools.get(tool_name)

        if tool is None:
            error = (
                f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
            )
            raise ToolExecutionError(error)

        self.on_tool_start(tool_name, args)
        start_time = time.perf_counter()

        try:
            result = tool.execute(args)
            result_str = self._stringify_tool_result(result)

        except Exception as exc:
            duration = time.perf_counter() - start_time
            self.state.record_tool_call(tool_name, args, None, duration)
            self._track_tool_performance(tool_name, duration)
            self.on_tool_end(tool_name, str(exc))
            raise ToolExecutionError(
                f"Tool '{tool_name}' execution failed: {exc}"
            ) from exc

        duration = time.perf_counter() - start_time
        self.state.record_tool_call(tool_name, args, result_str, duration)
        self._track_tool_performance(tool_name, duration)
        self.on_tool_end(tool_name, result_str)

        return result_str

    # HELPERS
    def _stringify_tool_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result

        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    def _track_tool_performance(self, tool_name: str, duration: float) -> None:
        """Track tool execution performance."""
        self.state.total_tool_time += duration

        if tool_name not in self._tool_times:
            self._tool_times[tool_name] = []

        self._tool_times[tool_name].append(duration)

    def _log_performance(self, start_time: float) -> None:
        """Log performance metrics."""
        total_time = time.perf_counter() - start_time

        self.logger.info(
            f"Performance Summary:\n"
            f"+ Total time: {total_time:.2f}s\n"
            f"+ Steps: {self.state.current_step}\n"
            f"+ Execution time: {self.state.total_execution_time:.2f}s\n"
            f"+ Tool execution time: {self.state.total_tool_time:.2f}s\n"
            f"+ LLM calls: {self._llm_call_count}\n"
            f"+ Total executed: {len(self.state.tool_calls)}"
        )

        if self._tool_times:
            stats = []
            for tool, times in self._tool_times.items():
                avg = sum(times) / len(times)
                stats.append(f"{tool}: {len(times)} calls, avg {avg:.2f}s")
            self.logger.info(f"Tools stats: {', '.join(stats)}")

    # SERIALIZATION
    def __repr__(self) -> str:
        return (
            f"ReactAgent("
            f"status={self.state.status.value}, "
            f"steps={self.state.current_step}/{self.max_steps}, "
            f"tools={len(self.tools)}, "
            f"errors={self.state.consecutive_errors})"
        )


# FACTORY FUNCTION
def create_react_agent(
    llm: BaseLLM, tools: list[Tool] | None = None, **kwargs
) -> ReactAgent:
    """
    Convenience factory for creating ReAct agents.

    Args:
        llm: Language model
        tools: List of tools
        **kwargs: Additional config

    Returns:
        Configured ReactAgent
    """
    config = {
        "max_steps": kwargs.get("max_steps", 10),
        "max_retries": kwargs.get("max_retries", 2),
        "timeout": kwargs.get("timeout", 30),
        "debug": kwargs.get("debug", False),
        "token_budget": kwargs.get("token_budget", 8000),
        "streaming": kwargs.get("streaming", False),
        **kwargs.get("config", {}),
    }

    return ReactAgent(llm=llm, tools=tools, config=config)


# USAGE EXAMPLE
if __name__ == "__main__":
    # Example setup
    from pydantic import BaseModel, Field
    from ..llm import OpenRouterLLM

    # from ..llm.groq import GroqLLM
    from ..llm.base import GenerationConfig

    # Configure LLM
    config = GenerationConfig(max_tokens=2048, temperature=0.2)
    # llm = GroqLLM(config=config)
    llm = OpenRouterLLM(model="dots-studio/dots-3-note-preview:free", config=config)

    # Define tools
    class WeatherToolArgs(BaseModel):
        city: str = Field(..., description="The city to fetch weather for")

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        weather_data = {
            "Paris": "Sunny, 25°C",
            "London": "Cloudy, 18°C",
            "Tokyo": "Rainy, 22°C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

    class CalcToolArgs(BaseModel):
        expression: str = Field(..., description="Mathematical operation expression")

    def calculate(expression: str) -> float:
        """Safe mathematical calculation."""
        # Simple safe eval
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            raise ValueError("Invalid characters in expression")
        return eval(expression)

    # Create tools
    weather_tool = Tool(
        name="get_weather",
        description="Get current weather for a city. Input: {{'city': 'city_name'}}",
        args_model=WeatherToolArgs,
        fn=get_weather,
    )

    calc_tool = Tool(
        name="calculate",
        description="Perform mathematical calculations. Input: {{'expression': '2+2'}}",
        args_model=CalcToolArgs,
        fn=calculate,
    )

    # Create agent
    agent = create_react_agent(
        llm=llm, tools=[weather_tool, calc_tool], max_steps=15, debug=True
    )

    # Execute
    try:
        response = agent.process_query("What's the weather in Tokyo?")
        print(f"Response: {response}")

        response = agent.process_query("What's the result of '9.2 - 6.3' operation?")
        print(f"Response: {response}")

        response = agent.process_query(
            "What is the answer to the life, the universe, and everything?"
        )
        print(f"Response: {response}")

        # Save state
        agent.save_state("agent_state.json")

    except Exception as e:
        print(f"Error: {e}")
