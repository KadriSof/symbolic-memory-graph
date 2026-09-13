# src/memory_agent/agents/react/engine.py

import time
import json

from typing import Any, Callable, Protocol, runtime_checkable

from ..base import Message, MessageRole, AgentStatus, AgentError, ToolExecutionError
from .parser import ReactMarker, ParsedAction
from ...tools import Tool


# REACT PROTOCOL
@runtime_checkable
class ReactStateProtocol(Protocol):
    """
    Execution-state contract required by ReactEngine.

    Implementations may represent ReAct execution state differently.
    ReactState provides the native ReAct implementation, while other
    agent states (e.g. CogitoState) may implement this protocol when
    ReAct execution is embedded into their cognitive trajectory.
    """

    status: AgentStatus
    current_turn: int
    current_step: int

    def start_turn(self) -> None: ...
    def end_turn(self) -> None: ...
    def increment_step(self) -> None: ...

    def record_message(
        self,
        role: str | MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def retrieve_messages(
        self,
        limit: int | None = None,
    ) -> list[Message]: ...

    def record_thought(
        self,
        thought: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str | None = None,
        duration: float | None = None,
    ) -> None: ...

    def record_observation(
        self,
        observation: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def record_result(self, result: str) -> None: ...

    def record_error(self, error: str) -> None: ...

    def reset_errors(self) -> None: ...


# REACT ENGINE
class ReactEngine:
    """
    Encapsulates the ReAct Pattern logic.

    Responsibilities:
    - Building system prompts
    - Parsing LLM responses
    - Executing actions/tools
    - Managing the ReAct loop
    - Error handling and recovery
    """

    def __init__(
        self,
        tools: dict[str, Tool] | None = None,
        max_steps: int = 10,
        logger: Any = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the ReAct Engine.

        Args:
            tools: Dictionary of available tools
            max_steps: Maximum reasoning steps
            max_retries: Maximum retries per step
            logger: Logger instance
            debug: Enable debug logging
        """
        self.tools = tools if tools is not None else {}
        self.max_steps = max_steps
        self.logger = logger
        self.debug = debug

        # Performance tracking
        self._tool_times: dict[str, list[float]] = {}
        self.total_tool_time: float = 0.0

    # PROMPT CONSTRUCTION
    def build_system_prompt(self) -> str:
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

        tools_section = self._build_tools_section(has_tools)
        format_section = self._build_format_section(has_tools)
        examples_section = self._build_examples(has_tools)
        rules_section = self._build_rules(has_tools)

        return f"""
        You are a ReAct (Reasoning + Acting) agent designed to solve problems through iterative reasoning and tool use.

        ## YOUR CAPABILITIES
        {tools_section}

        ## REACT FORMAT
        {format_section}

        ## EXAMPLES
        {examples_section}

        ## RULES
        {rules_section}

        ## RESPONSE FORMAT
        Your response must contain either:
        1. A <THOUGHT> and <ACTION> with <ACTION_INPUT> (if tools are available)
        2. A <THOUGHT> and <FINAL_ANSWER> 

        DO NOT include anything outside these markers.


        Now, process the user's request using the ReAct framework.
        """

    def _build_tools_section(self, has_tools: bool) -> str:
        """Build tool section for prompt."""
        if has_tools:
            descriptions = "\n\n".join(repr(tool) for tool in self.tools.values())

            return f"""
            You have access to the following tools:
            {descriptions}

            When you need to use a tool:
            1. First reason about why you need it
            2. Then specify the exact tool to use
            3. Provide valid JSON arguments
            4. Wait for the observation before continuing
            """

        else:
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
            Format your responses using these markers. Every marker MUST be closed:

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
            Format your responses using these markers. Every marker MUST be closed:

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
                    "5. EVERY tag must have a matching closing tag (e.g. </ACTION_INPUT>)",
                    "6. Wait for observations before making new decisions",
                    "7. If a tool fails, inspect the error observation and recover when possible.",
                ]
            )
        else:
            rules.append(
                "4. Provide complete, accurate answers based on your knowledge"
            )

        return "\n".join(rules)

    def format_error_observation(self, error: str) -> str:
        """Format error for observation feedback."""
        return f"""
        Tool/execution error: {error}
        Reconsider the previous action and continue if recovery is possible.
        """

    def format_parse_error(self, response: str) -> str:
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

    # RESPONSE PARSING
    def parse_response(self, response: str) -> ParsedAction | None:
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

        try:
            parsed = ParsedAction.from_text(response)
            if parsed:
                self.logger.info(
                    f"Parsed action: {parsed.action} with {parsed.action_input}"
                )
                return parsed

        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")

        return None

    def extract_final_answer(self, response: str) -> str:
        """Extract final answer from response."""
        answer = ReactMarker.extract(response, ReactMarker.FINAL_ANSWER)

        if answer is None:
            raise AgentError("Missing FINAL_ANSWER marker.")

        return answer

    # ACTION EXECUTION
    def execute_action(self, parsed: ParsedAction, state: ReactStateProtocol) -> str:
        """
        Execute a parsed action with robust handling.

        Args:
            parsed: Parsed action to execute
            state: Current ReactState for recording

        Returns:
            Observation string
        """
        return self._execute_tool(parsed.action, parsed.action_input, state)

    def _execute_tool(
        self, tool_name: str, args: dict, state: ReactStateProtocol
    ) -> str:
        """
        Execute a tool with validation and error handling.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            state: Current ReactState

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

        start_time = time.perf_counter()

        try:
            result = tool.execute(args)
            result_str = self._stringify_tool_result(result)

        except Exception as exc:
            duration = time.perf_counter() - start_time
            state.record_tool_call(tool_name, args, None, duration)
            self._track_tool_performance(tool_name, duration)
            raise ToolExecutionError(
                f"Tool '{tool_name}' execution failed: {exc}"
            ) from exc

        duration = time.perf_counter() - start_time
        state.record_tool_call(tool_name, args, result_str, duration)
        self._track_tool_performance(tool_name, duration)

        return result_str

    def _stringify_tool_result(self, result: Any) -> str:
        """Convert tool result to string."""
        if isinstance(result, str):
            return result

        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    def _track_tool_performance(self, tool_name: str, duration: float) -> None:
        """Track tool execution performance."""
        if tool_name not in self._tool_times:
            self._tool_times[tool_name] = []

        self.total_tool_time += duration
        self._tool_times[tool_name].append(duration)

    def get_tool_performance(self) -> dict[str, dict[str, Any]]:
        """Get tool performance statistics."""
        stats = {}
        for tool, times in self._tool_times.items():
            stats[tool] = {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
            }
        return stats

    # CORE EXECUTION LOOP
    def execute(
        self,
        query: str,
        state: ReactStateProtocol,
        llm_caller: Callable,
        max_steps: int | None = None,
    ) -> str:
        """
        Execute the ReAct loop.

        Args:
            query: User query
            state: Current ReactState
            llm_caller: Function that takes messages and returns LLM response
            max_steps: Optional override for max steps

        Returns:
            Final answer string

        Raises:
            AgentError: If execution fails
        """
        max_steps = max_steps or self.max_steps

        self.logger.info(f"Starting ReAct execution: {query[:100]}...")

        state.start_turn()
        state.status = AgentStatus.RUNNING

        try:
            for _ in range(max_steps):
                state.increment_step()

                try:
                    result = self._execute_step(state, llm_caller)

                    if result is not None:
                        state.record_result(result)
                        return result

                    state.reset_errors()

                except AgentError as exc:
                    error = str(exc)
                    state.record_error(error)
                    state.status = AgentStatus.ERROR

                    observation = self.format_error_observation(error)

                    self.logger.error(f"Step {state.current_step} failed: {error}")

                    state.record_message(
                        MessageRole.TOOL,
                        observation,
                        metadata={
                            "turn": state.current_turn,
                            "step": state.current_step,
                            "error": True,
                        },
                    )

            raise AgentError(f"Agent exceeded maximum steps ({max_steps})")

        finally:
            state.end_turn()

    def _execute_step(
        self,
        state: ReactStateProtocol,
        llm_caller: Callable,
    ) -> str | None:
        """
        Execute a single ReAct step.

        Returns:
            Final answer if complete, None otherwise

        Raises:
            AgentError: If step fails
        """
        step = state.current_step

        response = self._reason(state, llm_caller)

        state.record_message(
            MessageRole.ASSISTANT,
            response,
            metadata={
                "turn": state.current_turn,
                "step": step,
            },
        )

        parsed = self.parse_response(response)

        if parsed is None:
            # TODO: Verify if we need to return the 'format_parse_error' output as answer
            raise AgentError(self.format_parse_error(response))

        state.record_thought(parsed.thought)

        if parsed.is_final_answer:
            answer = self.extract_final_answer(response)

            if not answer:
                raise AgentError("Final answer marker was present but empty.")

            return answer

        observation = self.execute_action(parsed, state)

        state.record_observation(observation)

        state.record_message(
            MessageRole.TOOL,
            observation,
            metadata={
                "tool": parsed.action,
                "turn": state.current_turn,
                "step": step,
            },
        )

        state.status = AgentStatus.RUNNING

        return None

    def _reason(self, state: ReactStateProtocol, llm_caller: Callable) -> str:
        """
        Get reasoning from LLM via the provided caller.

        Args:
            state: Current ReactState
            llm_caller: Function that takes messages and returns LLM response

        Returns:
            LLM response string

        Raises:
            AgentError: If LLM call fails
        """
        state.status = AgentStatus.THINKING

        try:
            # Prepare messages with system prompt
            messages = self._prepare_messages(state)

            response = llm_caller(messages)

            self.logger.info(f"LLM response:\n{response}")

            if not response.strip():
                raise AgentError("LLM returned an empty response")

            return response

        except AgentError as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    def _prepare_messages(self, state: ReactStateProtocol) -> list[dict]:
        """Prepare messages for LLM call."""
        messages = [
            {"role": MessageRole.SYSTEM.value, "content": self.build_system_prompt()},
            *state.retrieve_messages(),
        ]

        return messages
