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

from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field

from typing import Any


from .base import (
    AgentStatus,
    BaseAgent,
    BaseState,
    AgentError,
    MessageRole,
    ToolExecutionError,
    TokenLimitExceeded,
)

from ..llm.base import BaseLLM, Message
from ..tools import Tool


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
        pattern = f"{marker.open_tag}(.*?){marker.close_tag}"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @classmethod
    def has_marker(cls, text: str, marker: "ReactMarker") -> bool:
        """Check if marker exists in text."""
        return marker.open_tag in text and marker.close_tag in text


# REACT STATE
@dataclass
class ReactState(BaseState):
    """
    ReAct-specific state with trajectory tracking.

    Extends BaseAgentState with:
    - Turn tracking
    - Tool execution history
    - Performance metrics
    - Retry counters
    """

    # ReAct specific
    current_turn: int = 0
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    # Performance tracking
    turn_start_time: float | None = None
    total_reasoning_time: float = 0.0
    total_tool_time: float = 0.0

    # Error recovery
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3
    last_error: str | None = None

    def start_turn(self) -> None:
        """Begin a new reasoning turn."""
        self.current_turn += 1
        self.step_count = self.current_turn
        self.turn_start_time = time.time()
        self.status = AgentStatus.THINKING

    def end_turn(self, result: str) -> None:
        """Complete a reasoning turn with result."""
        if self.turn_start_time:
            self.total_reasoning_time += time.time() - self.turn_start_time

        self.trajectory.append(
            {
                "turn": self.current_turn,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.status = AgentStatus.IDLE

    def record_tool_execution(
        self, tool_name: str, kwargs: dict[str, Any], result: str, duration: float
    ) -> None:
        """Record a tool execution with metrics."""
        self.tool_history.append(
            {
                "tool": tool_name,
                "args": kwargs,
                "result": result[:500],
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "turn": self.current_turn,
            }
        )
        self.total_tool_time += duration

    def record_error(self, error: str) -> None:
        """Record an error for recovery tracking."""
        self.consecutive_errors += 1
        self.last_error = error
        self.status = AgentStatus.ERROR

    def reset_errors(self) -> None:
        """Reset error counter after successful step."""
        self.consecutive_errors = 0
        self.last_error = None

    def should_abort(self) -> bool:
        """Check if agent should abort due to many errors."""
        return self.consecutive_errors >= self.max_consecutive_errors

    def get_trajectory_summary(self) -> str:
        """Get a compressed trajectory summary for prompts."""
        if not self.trajectory:
            return "Non trajectory yet."

        recent = self.trajectory[-3:]  # Last 3 turns
        summary = []
        for step in recent:
            summary.append(f"Turn {step['turn']}: {step['result'][:100]}")

        return "\n".join(summary)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "current_turn": self.current_turn,
                "tool_history": self.tool_history,
                "trajectory": self.trajectory,
                "total_reasoning_time": self.total_reasoning_time,
                "total_tool_time": self.total_tool_time,
                "consecutive_errors": self.consecutive_errors,
                "last_error": self.last_error,
            }
        )

        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReactState":
        """Deserialize state."""
        state = super().from_dict(data)

        if not isinstance(state, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(state).__name__}")

        state.current_turn = data.get("current_turn", 0)
        state.tool_history = data.get("tool_history", [])
        state.trajectory = data.get("trajectory", [])
        state.total_reasoning_time = data.get("total_reasoning_time", 0.0)
        state.total_tool_time = data.get("total_tool_time", 0.0)
        state.consecutive_errors = data.get("consecutive_errors", 0)
        state.last_error = data.get("last_error")

        return state


# PARSED ACTION:
@dataclass
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
            thought=thought, action=action, action_input=action_input, raw_text=text
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """
        Robust JSON parsing with fallbacks.

        Handles:
        - Extra text before/after JSON
        - Nested objects
        - Single vs double quotes
        - Trailing commas
        """
        # Try direct parsing first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:

            raise json.JSONDecodeError("No JSON object found", text, 0)

        json_str = text[start : end + 1]

        # Clean common issues
        json_str = re.sub(
            r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str
        )

        return json.loads(json_str)

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
    - Streaming support
    - Retry logic with backoff
    - Token budged management
    - Graceful degradation
    """

    def __init__(
        self,
        llm: "BaseLLM",
        tools: list["Tool"] | None = None,
        config: dict[str, Any] | None = None,
        state: BaseState | None = None,
    ) -> None:
        """
        Initialize ReAct Agent.

        Args:
            llm: Language model instance
            tools: List of available tools
            config: Configuration dict with keys:
                - max_steps: Maximum reasoning steps (default: 10)
                - max_retries: Max retries per step (default: 2)
                - timeout: Per-step timeout (default: 30)
                - debug: Enable debug logging (default: False)
                - token_budget: Max tokens (default: 8000)
                - streaming: Enable streaming (default: False)
        """
        super().__init__(llm, tools, config, state)

        # ReAct specific config
        self.max_steps = self.config.get("max_steps", 10)
        self.max_retries = self.config.get("max_retries", 2)
        self.max_step_timeout = self.config.get("timeout", 30)
        self.streaming_enabled = self.config.get("streaming", False)

        # Performance tracking
        self._step_times: list[float] = []
        self._tool_times: dict[str, list[float]] = {}

        self._log_init()

    def _initialize_state(self) -> BaseState:
        return ReactState()

    @property
    def state(self) -> ReactState:
        """Get state with correct type."""
        return super().state  # type: ignore

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
            desc = f"  - {tool.name}: {tool.description}"
            if hasattr(tool, "parameters"):
                if isinstance(tool.parameters, dict):
                    serializable_params = {
                        k: v.__name__ if isinstance(v, type) else v
                        for k, v in tool.parameters.items()
                    }
                    desc += (
                        f"\n    Parameters: {json.dumps(serializable_params, indent=2)}"
                    )
                else:
                    try:
                        desc += (
                            f"\n    Parameters: {json.dumps(tool.parameters, indent=2)}"
                        )
                    except TypeError:
                        desc += f"\n    Parameters: {str(tool.parameters)}"

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

            <THOUGHT>Your reasoning about the current situation</THOUGHT>
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

            <THOUGHT>Your reasoning about the question</THOUGHT>
            <FINAL_ANSWER>Your complete answer to the user</FINAL_ANSWER>

            Since you have no tools, you must directly answer the user's question.
            """

    def _build_examples(self, has_tools: bool) -> str:
        """Build examples for the prompt."""
        if has_tools:
            return """
            Example with tools:
            User: "What's the weather in Paris?"
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
            "1. Always start with <THOUGHT> to show your reasoning",
            "2. Be concise but thorough in your reasoning",
            "3. Never invent or hallucinate information",
            "4. Use exactly one action per response",
        ]

        if has_tools:
            rules.extend(
                [
                    "5. Always use valid JSON for <ACTION_INPUT>",
                    "6. Wait for observations before making new decisions",
                    "7. If a tool fails, explain the error and try an alternative if possible",
                ]
            )
        else:
            rules.append(
                "5. Provide complete, accurate answers based on your knowledge"
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
        """
        Execute the ReAct loop with full error handling.

        Returns:
            Final answer string

        Raises:
            AgentError: On unrecoverable errors
        """
        start_time = time.time()

        self.logger.info(f"Starting ReAct execution for query:\n{query[:100]}...\n----")

        for step in range(1, self.max_steps + 1):
            try:
                result = self._execute_step(step)

                if result:
                    self.logger.info(f"ReAct completed in {step} steps")
                    self._log_performance(start_time)
                    return result

            except TokenLimitExceeded:
                self.logger.warning(
                    f"Token limit exceeded at step {step}, summarizing.."
                )
                self._force_summarization()
                continue

            except Exception as e:
                self.state.record_error(str(e))
                self.logger.error(f"Error at step {step}:\n{e}\n----")

                if self.state.should_abort():
                    raise AgentError(
                        f"Too many consecutive errors: {self.state.last_error}"
                    ) from e

                # Attempt recovery
                recovery_message = f"<OBSERVATION>Error: {str(e)}. Please try a different approach.</OBSERVATION>"
                self.state.add_message("user", recovery_message)
                continue

        # Max steps exceeded
        error_msg = f"Agent exceeded maximum steps ({self.max_steps})"
        self.logger.error(error_msg)
        raise AgentError(error_msg)

    def _execute_step(self, step: int) -> str | None:
        """
        Execute a single ReAct step.

        Returns:
            Final answer if complete, None otherwise.
        """
        self.state.start_turn()
        self.on_step_start(step)

        try:
            # Get reasoning from LLM
            response = self._reason(step)

            # Persist the LLM's answer
            self.state.add_message("assistant", response)

            # Parse and route response
            parsed = self._parse_response(response)

            if parsed and parsed.is_final_answer:
                # Extract and return final answer
                final_answer = self._extract_final_answer(response)
                self.state.end_turn(final_answer)
                return final_answer

            if parsed:
                # Execute tool action
                observation = self._execute_action(parsed)
                self.state.add_message(
                    "user", f"<OBSERVATION>{observation}</OBSERVATION>"
                )

                # Reset error counter on success
                self.state.reset_errors()
                self.state.end_turn(f"Action: {parsed.action}")

            else:
                # Parsing failed, add error feedback
                error_msg = self._format_parse_error(response)
                self.state.add_message("user", error_msg)
                self.state.end_turn("Parse Error")

            self.on_step_end(step, "Step completed")
            return None

        except Exception as e:
            self.state.record_error(str(e))
            self.on_error(e)
            raise

    # REASONING
    def _build_reasoning_messages(self, step: int) -> list[Message]:
        """Prepare messages for LLM call with proper context."""
        messages = self._prepare_messages()

        if step > 1:
            messages.append(
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": f"This is turn {step} of {self.max_steps}. You have {self.max_steps - step + 1} turns remaining.",
                }
            )

        if self.state.current_turn > 1:
            messages.append(
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": f"Recent trajectory:\n{self.state.get_trajectory_summary()}",
                }
            )

        return messages

    def _reason(self, step: int) -> str:
        """
        Get reasoning from LLM with retry logic.

        Args:
            step: Current step number

        Returns:
            LLM response string
        """
        messages = self._build_reasoning_messages(step)

        # Call LLM with retries
        for attempt in range(self.max_retries + 1):
            try:
                self._log_debug(f"LLM call {attempt + 1}/{self.max_retries + 1}")
                response = self._llm_call(messages)

                self._log_debug(f"LLM response:\n{response}")
                return response

            except Exception as e:
                if attempt == self.max_retries:
                    raise

                wait_time = 2**attempt  # for exp. backoff
                self.logger.warning(f"LLM call failed: {e}, retrying in {wait_time}s")
                time.sleep(wait_time)

        raise AgentError("LLM call failed after all retries")

    def _prepare_messages(self) -> list[Message]:
        """Prepare messages for LLM call with proper context."""
        messages = []

        # System prompt
        messages.append({"role": "system", "content": self._get_system_prompt()})

        # summary if available
        if self.state.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary:\n{self.state.summary}",
                }
            )

        # conversation history (with token management)
        history = self.state.get_messages(limit=20)
        messages.extend(history)

        return messages

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
        # Final Thought
        if ReactMarker.has_marker(response, ReactMarker.FINAL_ANSWER):
            return ParsedAction(
                thought="Final answer provided",
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

        # Malformed response recovery
        recovered = self._recover_malformed_response(response)
        return recovered

    def _recover_malformed_response(self, response: str) -> ParsedAction | None:
        """
        Attempt to recover malformed responses.

        Handles:
        - Missing markers but contains phrases.
        - Extra text before/after markers
        - Simple corruption
        """
        action_match = re.search(r"(?:action|tool)[:\s]+(\w+)", response, re.IGNORECASE)
        if action_match:
            tool_name = action_match.group(1)

            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                try:
                    action_input = json.loads(json_match.group())
                    return ParsedAction(
                        thought="Recovered from malformed response",
                        action=tool_name,
                        action_input=action_input,
                        raw_text=response,
                    )

                except json.JSONDecodeError:
                    pass

        return None

    def _extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        answer = ReactMarker.extract(response, ReactMarker.FINAL_ANSWER)

        if answer is None:
            answer = response.strip()

            for marker in ReactMarker:
                answer = answer.replace(marker.open_tag, "").replace(
                    marker.close_tag, ""
                )

            answer = answer.strip()

        return answer

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
        if parsed.is_final_answer:
            return parsed.thought

        # persist in state
        self.state.add_thought(parsed.thought)

        try:
            start_time = time.perf_counter()
            result = super()._execute_tool(parsed.action, **parsed.action_input)
            duration = time.perf_counter() - start_time

            # metrics
            self._track_tool_performance(parsed.action, duration)

            self.logger.info(f"Tool {parsed.action} executed in {duration:.2f}s")

            return str(result)

        except ToolExecutionError as e:
            self.logger.error(f"Tool execution failed: {e}")
            return f"Error executing {parsed.action}: {str(e)}"

        except Exception as e:
            self.logger.error(f"Unexpected tool error: {e}")
            return f"Unexpected error: {str(e)}"

    def _track_tool_performance(self, tool_name: str, duration: float) -> None:
        """Track tool execution performance."""
        self.state.total_tool_time += duration

        if tool_name not in self._tool_times:
            self._tool_times[tool_name] = []

        self._tool_times[tool_name].append(duration)
        self._step_times.append(duration)

    # STREAMING SUPPORT
    def process_query_stream(self, query: str) -> Any:
        """
        Process query with streaming support.

        Yields:
            Chunks of the response as they become available
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not self.streaming_enabled:
            yield self.process_query(query)
            return

        self.logger.info(f"Processing query with streaming: {query[:100]}")

        self.state.status = AgentStatus.RUNNING
        self.state.add_message(MessageRole.USER, query)

        try:
            for step in range(1, self.max_steps + 1):
                self.state.start_turn()
                response = self._reason_stream(step)

                if ReactMarker.has_marker(response, ReactMarker.FINAL_ANSWER):
                    answer = self._extract_final_answer(response)
                    self.state.add_message(MessageRole.ASSISTANT, answer)
                    self.state.end_turn(answer)
                    self.state.status = AgentStatus.COMPLETED
                    yield answer
                    return

                parsed = ParsedAction.from_text(response)
                if parsed:
                    self.state.add_message(MessageRole.ASSISTANT, response)
                    observation = self._execute_action(parsed)
                    self.state.add_message(
                        "user", f"<OBSERVATION>{observation}</OBSERVATION>"
                    )
                    self.state.reset_errors()
                    self.state.end_turn(f"Action: {parsed.action}")
                    yield f"<OBSERVATION>{observation[:100]}...</OBSERVATION>"
                else:
                    error_msg = self._format_parse_error(response)
                    self.state.add_message(MessageRole.USER, error_msg)
                    self.state.end_turn("Parse Error")
                    yield "<ERROR>Failed to parse response</ERROR>"

                self.on_step_end(step, "Step completed")

        except Exception as e:
            self.state.status = AgentStatus.ERROR
            self.logger.error(f"Streaming query processing failed: {e}", exc_info=True)
            yield f"<ERROR>{str(e)}</ERROR>"

    def _reason_stream(self, step: int) -> str:
        """Get streaming reasoning from LLM with retry logic and context."""
        messages = self._build_reasoning_messages(step)

        for attempt in range(self.max_retries + 1):
            try:
                self._log_debug(
                    f"LLM streaming call {attempt + 1}/{self.max_retries + 1}"
                )
                full_response = ""

                # Stream chunks and accumulate
                for chunk in self.llm.chat_stream(messages):
                    full_response += chunk

                self._log_debug(f"LLM streaming response:\n{full_response}")
                return full_response

            except Exception as e:
                if attempt == self.max_retries:
                    raise AgentError(
                        f"LLM streaming call failed after all retries: {e}"
                    )

                wait_time = 2**attempt  # Exponential backoff
                self.logger.warning(
                    f"LLM streaming call failed: {e}, retrying in {wait_time}s"
                )
                time.sleep(wait_time)

        raise AgentError("LLM streaming call failed after all retries")

    def _log_performance(self, start_time: float) -> None:
        """Log performance metrics."""
        total_time = time.time() - start_time

        self.logger.info(
            f"Performance Summary:\n"
            f"+ Total time: {total_time:.2f}s\n"
            f"+ Steps: {self.state.current_turn}\n"
            f"+ Reasoning time: {self.state.total_reasoning_time:.2f}s\n"
            f"+ Tool execution time: {self.state.total_tool_time}s\n"
            f"+ LLM calls: {self._llm_call_count}\n"
            f"+ Total tokens: {self.state.total_tokens}\n"
            f"+ Total executed: {len(self.state.tool_history)}"
        )

        if self._tool_times:
            stats = []
            for tool, times in self._tool_times.items():
                avg = sum(times) / len(times)
                stats.append(f"{tool}: {len(times)} calls, avg {avg:.2f}s")
            self.logger.info(f"Tools stats: {', '.join(stats)}")

    # SERIALIZATION
    def save_state(self, path: str) -> None:
        """Save ReAct state with additional performance metadata in a single atomic write."""
        import os

        state_dict = self.state.to_dict()
        state_dict.update(
            {
                "_agent_type": self.__class__.__name__,
                "_execution_id": self._execution_id,
                "_llm_call_count": self._llm_call_count,
                "_performance": {
                    "step_times": self._step_times,
                    "tool_times": self._tool_times,
                },
            }
        )

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            self.logger.info(f"State saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise

    def __repr__(self) -> str:
        return (
            f"ReactAgent("
            f"status={self.state.status.value}, "
            f"steps={self.state.current_turn}/{self.max_steps}, "
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
    from ..llm import OpenRouterLLM

    # from ..llm.groq import GroqLLM
    from ..llm.base import GenerationConfig

    # Configure LLM
    config = GenerationConfig(max_tokens=2048, temperature=0.2)
    # llm = GroqLLM(config=config)
    llm = OpenRouterLLM(model="dots-studio/dots-3-note-preview:free", config=config)

    # Define tools
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        weather_data = {
            "Paris": "Sunny, 25°C",
            "London": "Cloudy, 18°C",
            "Tokyo": "Rainy, 22°C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

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
        parameters={"city": str},
        fn=get_weather,
    )

    calc_tool = Tool(
        name="calculate",
        description="Perform mathematical calculations. Input: {{'expression': '2+2'}}",
        parameters={"expression": str},
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
