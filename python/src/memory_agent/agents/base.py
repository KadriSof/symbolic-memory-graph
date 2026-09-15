"""
Base Agent Abstraction

Defines the core interfaces for all agents in the system.
"""

from abc import ABC, abstractmethod

import json
import uuid
import logging

from enum import Enum
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field

from typing import Any

from ..llm import BaseLLM
from ..tools import Tool


# Exceptions:
class AgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class TokenLimitExceeded(AgentError):
    """Raised when context window is exceeded."""

    pass


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""

    pass


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""

    pass


# Types:
class MessageRole(str, Enum):
    """Standardized message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

    # REACT STATUS
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"

    # COGITO STATUS
    PERCEIVE = "perceive"
    RECALL = "recall"
    ORIENT = "orient"
    CONSOLIDATE = "consolidate"


# Type aliases:
Message = dict[str, Any]
StateDict = dict[str, Any]


# Core State Management:
@dataclass
class BaseState:
    """
    Core state shared by all agent implementations.

    Attributes:
        current_turn: Conversation turn index
        messages: Conversation history with token tracking
        metadata: Arbitrary key-value store for extensions
        status: Current execution status
        max_messages: Maximum messages before compression
    """

    # Core conversation
    current_turn: int = 0
    messages: deque = field(default_factory=deque)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.IDLE

    # Configuration
    max_messages: int = 50

    def __post_init__(self) -> None:
        self.messages = deque(self.messages, maxlen=self.max_messages)

    # Message Management
    def record_message(
        self,
        role: str | MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a new message.

        Args:
            role: Message role (system/user/assistant/tool)
            content: Message content
            metadata: Optional additional data (tool_name, etc.)
        """
        if isinstance(role, MessageRole):
            role = role.value

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **({} if metadata is None else metadata),
        }

        self.messages.append(message)

    def retrieve_messages(self, limit: int | None = None) -> list[Message]:
        """Get recent messages, optionally limited."""
        messages = list(self.messages)

        if limit is None:
            return messages

        if limit < 0:
            raise ValueError("limit must be non-negative")

        return messages[-limit:] if limit else messages

    def retrieve_last_message(self) -> Message | None:
        """Retrieve the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def retrieve_last_user_message(self) -> Message | None:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg["role"] == MessageRole.USER.value:
                return msg

        return None

    def compact_messages(self, keep_last: int = 8) -> None:
        """Replace older conversation history with the last n messages."""
        keep_last = min(keep_last, self.max_messages)

        if len(self.messages) <= keep_last:
            return

        recent = list(self.messages)[-keep_last:]

        self.messages = deque(
            recent,
            maxlen=self.max_messages,
        )

    # State Serialization
    def to_dict(self) -> StateDict:
        """
        Serialize state to dictionary for persistence.

        Returns:
            JSON-serializable state dictionary
        """
        return {
            "state_type": self.__class__.__name__,
            "messages": list(self.messages),
            "metadata": self.metadata,
            "status": self.status.value,
            "max_messages": self.max_messages,
        }

    @classmethod
    def from_dict(cls, data: StateDict) -> "BaseState":
        """Deserialize state from dictionary."""
        state = cls(max_messages=data.get("max_messages", 50))

        state.messages = deque(data.get("messages", []), maxlen=state.max_messages)
        state.metadata = data.get("metadata", {})
        state.status = AgentStatus(data.get("status", AgentStatus.IDLE.value))

        return state

    def reset(self, keep_metadata: bool = False) -> None:
        """
        Reset state to initial configuration.

        Args:
            keep_metadata: Preserve metadata and configuration
        """
        self.messages.clear()
        self.status = AgentStatus.IDLE

        if not keep_metadata:
            self.metadata.clear()


# BASE AGENT ABSTRACT CLASS
class BaseAgent(ABC):
    """
    Abstract base class for all agents - Production Grade.

    Provides:
    - Robust error handling and recovery
    - Token budget management
    - Streaming support
    - Performance monitoring
    - Extensible hooks
    - Clean resource management
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Tool] | None = None,
        config: dict[str, Any] | None = None,
        state: BaseState | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            llm: Language model interface
            tools: List of available tools
            config: Configuration options
            state: Pre-existing state (for resuming)
            logger: Logger instance (auto-created if None)

        Raises:
            ValueError: If required dependencies are missing
        """

        # Core dependencies
        self.llm: BaseLLM = llm
        self.tools: dict[str, Tool] = {}
        self._validate_tools(tools)

        # Configuration with defaults
        self.config = {
            "max_steps": 10,
            "max_retries": 3,
            "debug": False,
            **(config or {}),
        }
        self._validate_config()

        # State management (lazy init)
        self._state: BaseState | None = state
        self._cached_system_prompt: str | None = None
        self._execution_id: str = str(uuid.uuid4())

        # Health monitoring
        self._last_llm_call_time: datetime | None = None
        self._llm_call_count: int = 0

        self.logger = logger if logger is not None else self._create_logger()
        self._log_init()

    # PROPERTIES & STATE MANAGEMENT
    @property
    def state(self) -> BaseState:
        """Lazy-initialize state."""
        if self._state is None:
            self._state = self._initialize_state()

        return self._state

    @classmethod
    def _state_class(cls) -> type[BaseState]:
        return BaseState

    def _deserialize_state(self, data: StateDict) -> BaseState:
        """Deserialize state using the concrete agent's state type."""
        return self._state_class().from_dict(data)

    @property
    def execution_id(self) -> str:
        """Get unique execution ID."""
        return self._execution_id

    @property
    def is_running(self) -> bool:
        """Check if agent is currently executing."""
        return self.state.status not in {
            AgentStatus.IDLE,
            AgentStatus.COMPLETED,
            AgentStatus.ERROR,
        }

    # ABSTRACT METHODS
    @abstractmethod
    def _initialize_state(self) -> BaseState:
        """
        Create the agent-specific state.
        Override this to add custom state fields.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the agent.
        Called once and cached.
        """
        raise NotImplementedError

    @abstractmethod
    def _process_query_impl(self, query: str) -> str:
        """
        Core query processing implementation.

        This is where the actual agent logic (ReAct, Plan-Execute, etc.)
        should be implemented.
        """
        raise NotImplementedError

    # PUBLIC API
    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response.

        Args:
            query: The user's input

        Returns:
            The agent's response

        Raises:
            AgentError: If processing fails
            AgentTimeoutError: If execution times out
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        self.state.status = AgentStatus.RUNNING
        self.state.record_message(MessageRole.USER, query)

        try:
            result = self._execute_with_retry(query)
            self.state.record_message(MessageRole.ASSISTANT, result)
            self.state.status = AgentStatus.COMPLETED

            return result

        except Exception as exc:
            self.state.status = AgentStatus.ERROR
            self.on_error(exc)
            self.logger.error("Query processing failed", exc_info=True)
            raise AgentError(f"Failed to process query: {exc}") from exc

    # EXECUTION & RETRY LOGIC
    def _execute_with_retry(self, query: str) -> str:
        """Execute with retry logic."""
        max_retries = self.config.get("max_retries", 3)
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                return self._process_query_impl(query)

            except AgentError as exc:
                last_error = exc

                if attempt == max_retries:
                    raise

                self.logger.warning(
                    f"[BaseAgent] Agent execution failed (attempt {attempt}/{max_retries}"
                    f"Error:\n----\n{exc}\n----\n"
                )

        raise AgentError(
            f"Execution failed after {max_retries} attempts"
        ) from last_error

    def process_query_stream(self, query: str):
        """Fallback streaming interface.

        Subclasses should override this when the underlying LLM
        supports incremental generation.
        """
        yield self.process_query(query)

    # TOOL MANAGEMENT
    def add_tool(self, tool: "Tool") -> None:
        """Dynamically add a tool at runtime."""
        if tool.name in self.tools:
            raise ValueError(f"[BaseAgent] Tool already registered: {tool.name}")

        self.tools[tool.name] = tool
        self.logger.info(f"Added tool: {tool.name}")

    def remove_tool(self, tool_name: str) -> bool:
        """Remove tool by name."""
        if tool_name not in self.tools:
            return False

        del self.tools[tool_name]
        self.logger.info(f"Removed tool: {tool_name}")
        return True

    # LLM Interface
    def _llm_call(self, messages: list[Message]) -> str:
        """
        Call the LLM with robust error handling.

        Args:
            messages: List of messages

        Returns:
            LLM response

        Raises:
            TokenLimitExceeded: If context window is exceeded
        """
        try:
            # Make the call
            self._last_llm_call_time = datetime.now()
            self._llm_call_count += 1

            response = self.llm.chat(messages=messages)

            return response or ""

        except Exception:
            self.logger.error("[BaseAgent] LLM call failed", exc_info=True)
            raise

    def _get_system_prompt(self) -> str:
        """Get cached system prompt."""
        if self._cached_system_prompt is None:
            self._cached_system_prompt = self._build_system_prompt()

        return self._cached_system_prompt

    def _prepare_messages(self) -> list[Message]:
        """Prepare messages for LLM call."""
        messages = [
            {"role": MessageRole.SYSTEM.value, "content": self._get_system_prompt()},
            *self.state.retrieve_messages(limit=8),
        ]

        return messages

    def _build_context(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "messages": self._prepare_messages(),
            "state": self.state,
            "metadata": self.state.metadata,
            **kwargs,
        }

    # Serialization & Persistence
    def save_state(self, path: str) -> None:
        """
        Save agent state to disk.

        Args:
            path: File path for state dump
        """
        import os

        try:
            state_dict = self.state.to_dict()
            state_dict.update(
                {
                    "_agent_type": self.__class__.__name__,
                    "_execution_id": self._execution_id,
                    "_llm_call_count": self._llm_call_count,
                }
            )

            directory = os.path.dirname(path)

            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(path, "w") as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"State saved to {path}")

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise

    def load_state(self, path: str) -> None:
        """
        Load agent state from disk.

        Args:
            path: File path for state dump
        """

        try:
            with open(path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)

            saved_agent_type = state_dict.get("_agent_type")

            if (
                saved_agent_type is not None
                and saved_agent_type != self.__class__.__name__
            ):
                raise ValueError(
                    f"State belongs to {saved_agent_type}, "
                    f"not {self.__class__.__name__}"
                )

            self._state = self._deserialize_state(state_dict)
            self._execution_id = state_dict.get("_execution_id", self._execution_id)
            self._llm_call_count = state_dict.get("_llm_call_count", 0)

            self.logger.info(f"State loaded from {path}")

        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            raise

    # Hooks
    def on_step_start(self, step: int) -> None:
        """Hook called before each reasoning step."""
        pass

    def on_step_end(self, step: int, result: str) -> None:
        """Hook called after each reasoning step."""
        pass

    def on_tool_start(self, tool_name: str, kwargs: dict[str, Any]) -> None:
        """Hook called before tool execution."""
        pass

    def on_tool_end(self, tool_name: str, result: str) -> None:
        """Hook called after tool execution."""
        pass

    def on_error(self, error: Exception) -> None:
        """Hook called when an error occurs."""
        pass

    # Utilities
    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.config["max_steps"] <= 0:
            raise ValueError("[BaseAgent] max_steps must be positive")

        if self.config["max_retries"] <= 0:
            raise ValueError("[BaseAgent] max_retries must be positive")

    def _validate_tools(self, tools: list[Tool] | None = None) -> None:
        """Validate tools."""
        for tool in tools or []:
            if tool.name in self.tools:
                raise ValueError(f"[BaseAgent] Duplicate tool name: {tool.name}")

            self.tools[tool.name] = tool

    def _create_logger(self) -> logging.Logger:
        """Create logger instance."""
        logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.DEBUG if self.config.get("debug") else logging.INFO)
        logger.propagate = False

        return logger

    def _log_init(self) -> None:
        """Log initialization details."""
        self.logger.info(f"[{self.__class__.__name__}] Initialized")
        self.logger.info(f"Tools: {list(self.tools.keys()) or 'None'}")
        self.logger.info(f"Max steps: {self.config.get('max_steps')}")
        self.logger.info(f"Execution ID: {self._execution_id}")

    def _log_debug(self, message: str) -> None:
        """Log debug message if enabled."""
        if self.config.get("debug", False):
            self.logger.debug(message)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if exc_type is not None:
            self.state.status = AgentStatus.ERROR
            self.on_error(exc_val)
        else:
            self.state.status = AgentStatus.IDLE

        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"status={self.state.status.value}, "
            f"messages={len(self.state.messages)}, "
            f"tools={len(self.tools)})"
        )
