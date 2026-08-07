"""
Base Agent Abstraction

Defines the core interfaces for all agents in the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from datetime import datetime
from typing import Any

from ..llm.base import BaseLLM
from ..tools import Tool


# Base Agent State
@dataclass
class BaseAgentState:
    """
    Core state shared by all agent implementations.

    Attributes:
        messages: Conversation history
        thoughts: Reasoning traces
        tool_calls: Record of tool executions
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Record a new message"""
        self.messages.append({"role": role, "content": content})

    def add_thought(self, thought: str) -> None:
        """Record a reasoning thought"""
        self.thoughts.append(thought)

    def add_tool_call(self, tool_name: str, kwargs: dict[str, Any]) -> None:
        """Record a tool call"""
        self.tool_calls.append(
            {
                "tool": tool_name,
                "kwargs": kwargs,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def last_message(self) -> dict[str, str] | None:
        """Retrieve the last message in the conversation"""
        return self.messages[-1] if self.messages else None

    def reset(self) -> None:
        """Reset state"""
        self.messages.clear()
        self.thoughts.clear()
        self.tool_calls.clear()


# Base Agent
class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Defines the core interface that all agents must implement.

    Attributes:
        llm: Language model interface
        tools: Dictionary of available tools
        state: Agent state
        config: Configuration options
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            llm: Language model interface
            tools: List of available tools
            config: Configuration options
        """
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.config = config or {}
        self.state = self._initialize_state()
        self.max_steps = self.config.get("max_steps", 10)
        self.debug = self.config.get("debug", False)

    @abstractmethod
    def _initialize_state(self) -> BaseAgentState:
        """Create the agent-specific state interface"""
        raise NotImplementedError

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        raise NotImplementedError

    @abstractmethod
    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response

        Args:
            query: The user's input

        Returns:
            The agent's response
        """
        raise NotImplementedError

    # Core Utilities
    def _llm_call(self, messages: list[dict[str, Any]]) -> str:
        """Call the LLM with a prompt"""
        response = self.llm.chat(messages=messages)
        return response

    def _log_init(self) -> None:
        """Log initialization details."""
        import logging

        logger = logging.getLogger(__name__)
        _CLSNAME = self.__class__.__name__
        logger.info(f"[{_CLSNAME}] Initialized")
        logger.info(
            f"[{_CLSNAME}] Tools: {list(self.tools.keys()) if self.tools else 'None'}"
        )
        logger.info(f"[{_CLSNAME}] Max steps: {self.max_steps}")

    def _log_debug(self, message: str) -> None:
        """Log debug message if enabled"""
        if self.debug:
            import logging

            logging.getLogger(__name__).debug(f"[DEBUG] {message}")

    # TODO: Add "save_state" and "load_state" logic (for later)
