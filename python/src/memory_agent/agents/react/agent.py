"""
ReAct Agent - Refactored Implementation

Components:
1. ReactProtocol: Defines the ReAct state contract
2. ReactState: ReAct state implementation (extends BaseState)
3. ReactEngine: Encapsulates the ReAct Pattern logic
4. ReactAgent: Implements BaseAgent, powered by ReactState and ReactEngine
"""

import time

from typing import Any

from ..base import (
    BaseAgent,
    BaseState,
    AgentError,
)

from .state import ReactState
from .engine import ReactEngine
from ...llm.base import BaseLLM
from ...tools import Tool


# REACT AGENT
class ReactAgent(BaseAgent):
    """
    ReAct Agent implementation.

    Powered by ReactState and ReactEngine.
    The LLM is owned by the agent (passed to engine via llm_caller).

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
                - max_retries: Max retries per step (default: 3)
                - debug: Enable debug logging (default: False)
            state: Pre-existing state (optional)
        """
        super().__init__(llm, tools, config, state)

        # ReAct specific config
        self.max_steps = self.config.get("max_steps", 10)
        self.max_retries = self.config.get("max_retries", 3)

        # Create engine with tool registry
        self._engine = ReactEngine(
            tools=self.tools,
            max_steps=self.max_steps,
            logger=self.logger,
            debug=self.config.get("debug", False),
        )

    # STATE MANAGEMENT
    def _initialize_state(self) -> BaseState:
        """Initialize ReactState."""
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

    # PROMPT DELEGATION
    def _build_system_prompt(self) -> str:
        """Delegate to engine for prompt construction."""
        return self._engine.build_system_prompt()

    # CORE EXECUTION
    def _process_query_impl(self, query: str) -> str:
        """
        Core query processing implementation.

        Delegates to ReactEngine for the actual ReAct loop.
        """
        start_time = time.perf_counter()

        self.logger.info(
            "Starting ReAct execution: %s",
            query[:100],
        )

        # Define LLM caller that uses the agent's LLM
        def llm_caller(messages: list[dict]) -> str:
            import json

            self.logger.debug(
                f"[llm_caller] messages:\n----\n{json.dumps(messages, indent=2)}\n----\n"
            )
            response = self._llm_call(messages)
            self.logger.debug(f"[llm_caller] respone:\n----\n{repr(response)}\n----\n")
            return response

        try:
            result = self._engine.execute(
                query=query,
                state=self.state,
                llm_caller=llm_caller,
                max_steps=self.max_steps,
            )

            self._log_performance(start_time)
            return result

        except AgentError:
            self.logger.error("Query processing failed", exc_info=True)
            raise

    # PERFORMANCE LOGGING
    def _log_performance(self, start_time: float) -> None:
        """Log per-turn and cumulative performance metrics."""
        turn_time = time.perf_counter() - start_time
        tool_stats = self._engine.get_tool_performance()

        self.logger.info(
            f"Performance Summary:\n"
            f"+ Turn time: {turn_time:.2f}s\n"
            f"+ Steps: {self.state.current_step}\n"
            f"+ Cumulative execution time: {self.state.total_execution_time:.2f}s\n"
            f"+ Cumulative tool time: {self._engine.total_tool_time:.2f}s\n"
            f"+ LLM calls: {self._llm_call_count}\n"
            f"+ Total tools executed: {len(self.state.tool_calls)}"
        )

        if tool_stats:
            stats_str = ", ".join(
                f"{tool}: {stats['count']} calls, avg {stats['avg']:.2f}s"
                for tool, stats in tool_stats.items()
            )
            self.logger.info(f"Tools stats: {stats_str}")

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
        "max_retries": kwargs.get("max_retries", 3),
        "timeout": kwargs.get("timeout", 30),
        "debug": kwargs.get("debug", True),
        "token_budget": kwargs.get("token_budget", 8000),
        "streaming": kwargs.get("streaming", False),
        **kwargs.get("config", {}),
    }
    print(f"[create_react_agent] Passed configuration:\n----\n{config}\n----\n")

    return ReactAgent(llm=llm, tools=tools, config=config)


# USAGE EXAMPLE
if __name__ == "__main__":
    import os
    from pydantic import BaseModel, Field
    from ...llm import OpenRouterLLM, LlamaCppLLM
    from ...llm.base import GenerationConfig

    # Configure LLM
    # config = GenerationConfig(max_tokens=2048, temperature=0.2)
    # llm = OpenRouterLLM(model="dots-studio/dots-3-note-preview:free", config=config)

    model_path = os.path.expanduser("~/models/Qwen3.5-2B-Q5_K_M.gguf")

    config = GenerationConfig(temperature=0.1, max_tokens=8192)
    llm = LlamaCppLLM(
        model_path=model_path,
        config=config,
    )

    # Define tools
    class WeatherToolArgs(BaseModel):
        city: str = Field(..., description="The city to fetch weather for")

    def get_weather(city: str) -> str:
        weather_data = {
            "Paris": "Sunny, 25°C",
            "London": "Cloudy, 18°C",
            "Tokyo": "Rainy, 22°C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

    class CalcToolArgs(BaseModel):
        expression: str = Field(..., description="Mathematical operation expression")

    def calculate(expression: str) -> float:
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
