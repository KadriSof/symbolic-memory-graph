"""
ReAct Agent Implementation

Implements the Reasoning + Acting pattern with tool support.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass, field

from typing import Any

from .base import BaseAgent, BaseAgentState
from ..llm import BaseLLM
from ..tools import Tool

MARKER_THOUGHT = "THOUGHT"
MARKER_TOOL = "TOOL"
MARKER_TOOL_INPUT = "TOOL-INPUT"
MARKER_TOOL_OUTPUT = "TOOL-OUTPUT"
MARKER_FINAL_ANSWER = "FINAL-ANSWER"


logger = logging.getLogger(__name__)


@dataclass
class ReactState(BaseAgentState):
    """
    State for the ReAct agent.

    Extends BaseAgentState with ReAct-specific field.
    """

    current_turn: int = 0
    tool_output_history: list[str] = field(default_factory=list)

    def add_tool_output(self, tool_output: str) -> None:
        """Record tool output to the history"""
        self.tool_output_history.append(tool_output)

    def increment_turn(self) -> None:
        self.current_turn += 1


# ReAct Agent:
class ReactAgent(BaseAgent):
    """
    ReAct Agent — Reasoning + Acting pattern.

    The agent iteratively reasons about the query, decides on actions,
    executes tools, and observes results until it has enough information
    to provide a final answer.

    Example:
        >>> agent = ReactAgent(llm=llm, tools=[weather_tool])
        >>> response = agent.process_query("What's the weather in Paris?")
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(llm, tools, config)
        self.max_steps = self.config.get("max_steps", 5)

    def _initialize_state(self) -> ReactState:
        """Create a ReactState instance"""
        return ReactState()

    def _get_system_prompt(self) -> str:
        """Unified ReAct prompt with or without tools."""
        return self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        Construct tool-aware system prompt.
        """
        has_tools = bool(self.tools)

        if has_tools:
            tool_descriptions = "\n".join(
                f"  - {t.name}: {t.description}" for t in self.tools.values()
            )
            tool_section = f"""
            You have access to the following tools:
            {tool_descriptions}

            When you need to use a tool, follow this structure:
            <{MARKER_THOUGHT}> reason about the current situation </{MARKER_THOUGHT}>
            <{MARKER_TOOL}> the tool to execute </{MARKER_TOOL}>
            <{MARKER_TOOL_INPUT}> {{"key": "value"}} </{MARKER_TOOL_INPUT}>

            <{MARKER_TOOL_OUTPUT}>: result returned by the tool
            """
        else:
            tool_section = """
            You have NO tools available.

            ## IMPORTANT: You CANNOT use <TOOL> or <TOOL-INPUT> markers.
            - You must respond directly with <FINAL-ANSWER> after thinking.
            - If you lack information, state this honestly.
            - NEVER invent or hallucinate information.
            """

        return f"""
        You are a ReAct reasoning agent.

        {tool_section}

        When you have enough information to answer the user, respond with:
        <{MARKER_THOUGHT}> I now know the final answer </{MARKER_THOUGHT}>
        <{MARKER_FINAL_ANSWER}> the final response to the user </{MARKER_FINAL_ANSWER}>

        ## RULES
        1. Keep thoughts concise and task-oriented
        2. Never invent tool observations
        3. Tool Input must always be valid JSON (if tools are available)
        4. If no tools are available, skip TOOL and TOOL-INPUT
        5. Provide direct, truthful answers without tool references when no tools exist

        ## EXAMPLE WITH TOOLS
        Question: What is the weather in Paris?
        <{MARKER_THOUGHT}> I need to get the weather for Paris</{MARKER_THOUGHT}>
        <{MARKER_TOOL}>get_weather</{MARKER_TOOL}>
        <{MARKER_TOOL_INPUT}>{{"city": "Paris"}}</{MARKER_TOOL_INPUT}>
        <{MARKER_TOOL_OUTPUT}>: Sunny, 25°C</{MARKER_TOOL_OUTPUT}>
        <{MARKER_THOUGHT}> I now have the weather information</{MARKER_THOUGHT}>
        <{MARKER_FINAL_ANSWER}> The weather in Paris is sunny with a temperature of 25°C.</{MARKER_FINAL_ANSWER}>

        ## EXAMPLE WITHOUT TOOLS
        Question: Who wrote Romeo and Juliet?
        <{MARKER_THOUGHT}> This is a literary knowledge question about Shakespeare</{MARKER_THOUGHT}>
        <{MARKER_FINAL_ANSWER}> William Shakespeare wrote Romeo and Juliet.</{MARKER_FINAL_ANSWER}>

        ## EXAMPLE WITHOUT TOOLS (missing information)
        Question: What is the current stock price of Tesla?
        <{MARKER_THOUGHT}> I don't have access to real-time data without tools</{MARKER_THOUGHT}>
        <{MARKER_FINAL_ANSWER}> I cannot provide current stock prices as I have no access to real-time market data.</{MARKER_FINAL_ANSWER}>

        Now, respond to the user's question using the format above.
        """

    def process_query(
        self,
        query: str,
    ) -> str:
        """
        Execute ReAct loop.

        Returns:
            Final assistant answer.
        """
        self._log_debug(f"Processing query: {query[:100]}")

        self.state = self._initialize_state()
        self.state.add_message(role="system", content=self._get_system_prompt())
        self.state.add_message(role="user", content=query)

        for _ in range(self.max_steps):
            self.state.increment_turn()
            response = self._reason()
            final_answer = self._route_response(response=response)

            if final_answer is not None:
                return final_answer

        raise RuntimeError(f"[ReActAgent] Agent exceeded max_steps={self.max_steps}.")

    def _reason(self) -> str:
        """
        Delegate reasoning to the LLM
        Serializes the conversation state and sends it to the LLM
        for the next reasoning step.

        Args:
            state (AgentState): current agent state containing conversation history.

        Returns:
            (str) LLM response string.
        """
        messages = self.state.messages.copy()
        response = self._llm_call(messages=messages)

        if self.debug:
            print(
                f"[ReActAgent][DEBUG] LLM Prompt:\n{json.dumps(self.state.messages, indent=2)}\n---"
            )
            print(f"[ReActAgent][DEBUG] LLM Response:\n{response}\n---")

        self.state.add_message(role="assistant", content=response)
        return response

    def _route_response(self, response: str) -> str | None:
        """
        Route model output.

        Args:
            state (AgentState): the agent state.
            response (str): the model response.

        Returns:
            (str) the final answer string if execution should terminate,
            None if the loop should continue.
        """
        # 1. No tools trajectory:
        if not self.tools:
            if "<{MARKER_FINAL_ANSWER}>" in response:
                answer = self._extract_block(
                    text=response,
                    start=f"<{MARKER_FINAL_ANSWER}>",
                    end=f"</{MARKER_FINAL_ANSWER}>",
                )
            else:
                answer = response.strip()

            self.state.add_message(role="assistant", content=answer)
            return answer

        # 2. Tool execution trajectory:
        if f"<{MARKER_TOOL}>" in response:
            self._handle_tool_path(response=response)
            return None

        # 3. Final Answer trajectory:
        if f"<{MARKER_FINAL_ANSWER}>" in response:
            return self._handle_final_answer(response=response)

        # 4. Self-healing fallback if neither marker is present
        error_msg = (
            f"<{MARKER_TOOL_OUTPUT}>Error: Your response did not contain a valid "
            f"<{MARKER_TOOL}> or <{MARKER_FINAL_ANSWER}> marker. "
            "Please ensure you follow the ReAct trajectory format exactly."
            f"</{MARKER_TOOL_OUTPUT}>"
        )
        self.state.add_message(role="tool", content=error_msg)
        return None

    def _handle_final_answer(
        self,
        response: str,
    ) -> str:
        """
        Extract and persist final answer.

        Parses the final answer from the response and adds it to state.

        Args:
            state (AgentState): current agent state.
            response (str): model response containing "Final Answer".

        Returns:
            (str) extracted answer string.
        """
        answer = self._extract_block(
            text=response,
            start=f"<{MARKER_FINAL_ANSWER}>",
            end=f"</{MARKER_FINAL_ANSWER}>",
        )
        self.state.add_message(role="assistant", content=answer)
        return answer

    def _handle_tool_path(
        self,
        response: str,
    ) -> None:
        """
        Parse and execute tool action.

        Extracts thought, action, and action input from response,
        then executes the requested tool.

        Args:
            state (AgentState): current agent state.
            response (str): model response containing Thought/Action/Action Input.

        Returns:
            None to indicate loop should continue.

        Raises:
            ValueError: if response format is invalid.
        """
        parsed = self._parse_action(text=response)

        if parsed is not None:

            thought, tool_name, tool_input = parsed

            # Record intent
            self.state.add_thought(thought)
            self.state.add_tool_call(tool_name, tool_input)

            try:
                tool_output = self._execute_tool(tool_name, tool_input)
                output_message = (
                    f"<{MARKER_TOOL_OUTPUT}>{tool_output}</{MARKER_TOOL_OUTPUT}>"
                )
            except Exception as e:
                # Do we add the tool output marker here?
                output_message = f"<{MARKER_TOOL_OUTPUT}>Tool execution error: {str(e)}</{MARKER_TOOL_OUTPUT}>"

            self.state.add_message(role="user", content=output_message)

        else:
            # SELF-HEALING FALLBACK: Prevents crashing on bad JSON/formatting
            error_msg = (
                f"<{MARKER_TOOL_OUTPUT}>Error: Your Tool Input was not valid JSON or markers were missing. "
                f"Ensure you use <{MARKER_TOOL_INPUT}> with valid, complete JSON enclosed in curly braces {{}}."
                f"</{MARKER_TOOL_OUTPUT}>"
            )
            self.state.add_message(role="tool", content=error_msg)

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Any:
        """
        Execute tool.

        Looks up and executes the requested tool with provided arguments.

        Args:
            tool_name (str): name of the tool to execute.
            tool_args (dict[str, Any]): dictionary of arguments for the tool.

        Returns:
            (Any) result from tool execution.

        Raises:
            ValueError: if tool_name doesn't exist in available tools.:
        """
        tool = self.tools.get(tool_name)

        if tool is None:
            raise ValueError(f"[ReActAgent] Unknown tool: {tool_name}")

        return tool.execute(**tool_args)

    def _parse_action(self, text: str) -> tuple[str, str, dict[str, Any]] | None:
        """
        Parse action block using special markers.
        Returns (thought, action, action_input) or None if parsing fails.
        """
        try:
            thought = self._extract_block(
                text, f"<{MARKER_THOUGHT}>", f"</{MARKER_THOUGHT}>"
            ).strip()

            tool = self._extract_block(
                text, f"<{MARKER_TOOL}>", f"</{MARKER_TOOL}>"
            ).strip()

            # Get everything after the action input marker
            tool_input_raw = text.split(f"<{MARKER_TOOL_INPUT}>", 1)[1].strip()

            if self.debug:
                print(
                    f"[ReActAgent][DEBUG] Raw Action Input snippet: {repr(tool_input_raw[:100])}"
                )

            # ROBUST JSON EXTRACTION: Find first '{' and LAST '}' to handle nested objects
            start_idx = tool_input_raw.find("{")
            end_idx = tool_input_raw.rfind("}")  # rfind gets the LAST occurrence

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = tool_input_raw[start_idx : end_idx + 1]
                tool_input = json.loads(json_str)
            else:
                # Fallback: strip hallucinated trailing text and try again
                fallback_str = tool_input_raw
                for marker in [
                    f"<{MARKER_TOOL_OUTPUT}>",
                    f"<{MARKER_THOUGHT}>",
                    f"<{MARKER_FINAL_ANSWER}>",
                ]:
                    if marker in fallback_str:
                        fallback_str = fallback_str.split(marker, 1)[0].strip()
                        break
                tool_input = json.loads(fallback_str)

            return thought, tool, tool_input

        except (IndexError, json.JSONDecodeError, AttributeError) as e:
            if self.debug:
                print(
                    f"[ReActAgent][DEBUG] Failed to parse ReAct response: {text}\n---"
                )
                print(f"[ReActAgent][DEBUG] Error: {e}\n---")

            return None

    @staticmethod
    def _extract_block(text: str, start: str, end: str) -> str:
        """Extract substring between markers safely."""
        try:
            return text.split(start, 1)[1].split(end, 1)[0]
        except IndexError:
            return ""


if __name__ == "__main__":
    from ..llm.groq import GroqLLM
    from ..llm.base import GenerationConfig
    from ..tools import Tool

    config = GenerationConfig(max_tokens=2048, temperature=0.2)
    groq_llm = GroqLLM(config=config)

    def get_weather(city: str) -> str:
        """Mock weather tool for testing"""
        return f"Sunny and 30°C in {city}"

    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        fn=get_weather,
    )

    agent = ReactAgent(llm=groq_llm, tools=[weather_tool], config={"debug": True})

    user_query = "What is the weather in Tunis today?"
    output = agent.process_query(user_query)

    print(f"Agent output: {output}")

    # Running the agent without any tool:
    base_agent = ReactAgent(llm=groq_llm, tools=[], config={"debug": True})
    output = base_agent.process_query(
        "What's the answer to the meaning of the universe, life, and everything?"
    )
    print(f"Base Agent output: {output}")
