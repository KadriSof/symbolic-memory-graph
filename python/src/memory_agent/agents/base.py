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
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    ERROR = "error"


# Type aliases:
Message = dict[str, Any]
StateDict = dict[str, Any]


# Core State Management:
@dataclass
class BaseState:
    """
    Core state shared by all agent implementations.
    Immutable-safe with controlled mutation methods.

    Attributes:
        messages: Conversation history with token tracking
        thoughts: Reasoning traces with timestamps
        tool_calls: Record of tool executions with metadata
        observations: Compressed observation store
        summary: Rolling summary of interactions
        metadata: Arbitrary key-value store for extensions
        status: Current execution status
        step_count: Number of reasoning steps taken
        total_tokens: Running token count
        max_messages: Maximum messages before compression
        max_observations: Maximum stored observations
    """

    # Core conversation
    messages: deque = field(default_factory=lambda: deque(maxlen=50))
    thoughts: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Memory Management
    observations: deque = field(default_factory=lambda: deque(maxlen=30))
    summary: str = ""
    summary_token_count: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.IDLE
    step_count: int = 0
    total_tokens: int = 0

    # Configuration
    max_messages: int = 50
    max_observations: int = 30
    summary_threshold: int = 4000
    compression_enabled: bool = True

    @property
    def context_tokens(self) -> int:
        return sum(int(message.get("tokens", 0)) for message in self.messages)

    # Message Management
    def add_message(
        self,
        role: str | MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a new message with token tracking.

        Args:
            role: Message role (system/user/assistant/tool)
            content: Message content
            metadata: Optional additional data (tool_name, etc.)
        """
        if isinstance(role, MessageRole):
            role = role.value

        msg_tokens = self._estimate_tokens(content)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tokens": msg_tokens,
            **({} if metadata is None else metadata),
        }

        self.messages.append(message)
        self.total_tokens += message["tokens"]

        # Trigger compression when required
        if self.total_tokens > self.summary_threshold:
            self._requires_compression = True

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get recent messages, optionally limited."""
        messages = list(self.messages)
        return messages[-limit:] if limit else messages

    def get_last_message(self) -> Message | None:
        """Retrieve the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def get_last_user_message(self) -> Message | None:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg["role"] == MessageRole.USER.value:
                return msg

        return None

    # Thought Management
    def add_thought(self, thought: str, metadata: dict[str, Any] | None = None) -> None:
        """Record a reasoning thought with timestamp."""
        self.thoughts.append(
            {
                "content": thought,
                "timestamp": datetime.now().isoformat(),
                "step": self.step_count,
                **(metadata or {}),
            }
        )

    def get_recent_thoughts(self, limit: int = 5) -> list[str]:
        """Get recent thought contents."""
        return [t["content"] for t in self.thoughts[-limit:]]

    # Tool Call Management
    def add_tool_call(
        self, tool_name: str, kwargs: dict[str, Any], result: str | None = None
    ) -> None:
        """Record a tool call with full context."""
        self.tool_calls.append(
            {
                "tool": tool_name,
                "kwargs": kwargs,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "step": self.step_count,
            }
        )

    def get_tool_history(self, tool_name: str | None = None) -> list[dict[str, Any]]:
        """Get tool call history, optionally filtered by tool."""
        if tool_name:
            return [tc for tc in self.tool_calls if tc["tool"] == tool_name]
        return self.tool_calls

    # Observation Management
    def add_observation(
        self,
        thought: str,
        action: str,
        observation: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an observation with compression.

        The observation is stored in a compact format to preserve details
        while minimizing memory footprint.
        """
        max_len = 2000
        truncated_obs = (
            observation
            if len(observation) <= max_len
            else f"{observation[:max_len]}... [truncated, {len(observation) - max_len} chars]"
        )
        obs_tokens = self._estimate_tokens(observation)

        entry = {
            "thought": thought,
            "action": action,
            "observation": truncated_obs,
            "step": self.step_count,
            "tokens": obs_tokens,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        self.observations.append(entry)
        self.step_count += 1

    def get_observations(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get observations, optionally with compression restored."""
        obs = list(self.observations)
        if limit:
            obs = obs[-limit:]

        return obs

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Count tokens in a text string using a simple regex-based tokenizer.
        Splits on whitespace and punctuation, similar to common LLM tokenizers.

        Args:
            text: Input text string.

        Returns:
            Number of tokens.
        """
        import re

        tokens = re.findall(r"\w+|\S", text)
        return len(tokens)

    def compact_messages(self, keep_last: int = 8) -> None:
        """Replace older conversation history with the current summary."""
        if len(self.messages) <= keep_last:
            return

        recent = list(self.messages)[-keep_last:]

        self.messages = deque(
            recent,
            maxlen=self.max_messages,
        )

    # Summary Management
    def requires_summarization(self) -> bool:
        """Check if state needs summarization."""
        return self.context_tokens > self.summary_threshold

    def set_summary(self, summary: str, token_count: int) -> None:
        """Update the rolling summary."""
        self.summary = summary
        self.summary_token_count = token_count

    def get_context_summary(self) -> str:
        """Get the current summary for prompt injection."""
        return self.summary if self.summary else "No summary available."

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
            "thoughts": self.thoughts,
            "tool_calls": self.tool_calls,
            "observations": list(self.observations),
            "summary": self.summary,
            "summary_token_count": self.summary_token_count,
            "metadata": self.metadata,
            "status": self.status.value,
            "step_count": self.step_count,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: StateDict) -> "BaseState":
        """Deserialize state from dictionary."""
        state = cls()

        state.messages = deque(data.get("messages", []), maxlen=state.max_messages)
        state.thoughts = data.get("thoughts", [])
        state.tool_calls = data.get("tool_calls", [])
        state.observations = deque(
            data.get("observations", []), maxlen=state.max_observations
        )
        state.summary = data.get("summary", "")
        state.summary_token_count = data.get("summary_token_count", 0)
        state.metadata = data.get("metadata", {})
        state.status = AgentStatus(data.get("status", AgentStatus.IDLE.value))
        state.step_count = data.get("step_count", 0)
        state.total_tokens = data.get("total_tokens", 0)

        return state

    def reset(self, keep_metadata: bool = False) -> None:
        """
        Reset state to initial configuration.

        Args:
            keep_metadata: Preserve metadata and configuration
        """
        self.messages.clear()
        self.thoughts.clear()
        self.tool_calls.clear()
        self.observations.clear()
        self.summary = ""
        self.summary_token_count = 0
        self.step_count = 0
        self.total_tokens = 0
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
        llm: "BaseLLM",
        tools: list["Tool"] | None = None,
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
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.logger = logger or self._create_logger()

        # Configuration with defaults
        self.config = {
            "max_steps": 10,
            "max_retries": 3,
            "debug": False,
            "token_budget": 8000,
            "summarization_threshold": 0.7,
            **(config or {}),
        }

        # State management (lazy init)
        self._state: BaseState | None = state
        self._cached_system_prompt: str | None = None
        self._execution_id: str = str(uuid.uuid4())

        # Health monitoring
        self._last_llm_call_time: datetime | None = None
        self._llm_call_count: int = 0

        self._log_init()
        self._validate_config()

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
        return self.state.status == AgentStatus.RUNNING

    # ABSTRACT METHODS
    @abstractmethod
    def _initialize_state(self) -> BaseState:
        """
        Create the agent-specific state.
        Override this to add custom state fields.
        """
        return BaseState()

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
        self.state.add_message(MessageRole.USER, query)

        try:
            result = self._execute_with_retry(query)
            self.state.add_message(MessageRole.ASSISTANT, result)
            self.state.status = AgentStatus.COMPLETED
            return result

        except Exception as e:
            self.state.status = AgentStatus.ERROR
            self.logger.error(f"Query processing failed: {e}", exc_info=True)
            raise AgentError(f"Failed to process query: {e}") from e

    def process_query_stream(self, query: str) -> Any:
        """
        Process a query with streaming support.

        Args:
            query: User input

        Yields:
            Stream chunks (implementation-specific)
        """
        # Override in subclasses for streaming
        yield self.process_query(query)

    def add_tool(self, tool: "Tool") -> None:
        """Dynamically add a tool at runtime."""
        self.tools[tool.name] = tool
        self.logger.info(f"Added tool: {tool.name}")

    def remove_tool(self, tool_name: str) -> None:
        """Remove tool by name."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"Removed tool: {tool_name}")

    # EXECUTION & RETRY LOGIC
    def _execute_with_retry(self, query: str) -> str:
        """Execute with retry logic."""
        max_retries = self.config.get("max_retries", 3)
        last_error = None

        for attempt in range(max_retries):
            try:
                return self._process_query_impl(query)

            except TokenLimitExceeded as e:
                self.logger.warning(
                    f"Token limit exceeded (attempt {attempt + 1}/{max_retries}."
                    f"Forcing summarization..."
                )
                self._force_summarization()
                last_error = str(e)
                continue

            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Execution failed after {max_retries} attempt: {e}"
                    )
                    raise

                self.logger.warning(
                    f"Failed after retrying {max_retries} times. Error:\n{e}\n---"
                )
                last_error = str(e)
                continue

        raise AgentError(f"All retries failed: {last_error}")

    def _force_summarization(self) -> None:
        """Force summarization of current state."""
        from ..llm import summarize_conversation

        if not self.state.messages:
            return

        try:
            messages = list(self.state.messages)
            summary = summarize_conversation(messages, self.llm, max_tokens=500)
            summary_tokens = self.state._estimate_tokens(summary)

            self.state.set_summary(summary, summary_tokens)
            self.state.compact_messages(keep_last=8)

            self.logger.info(
                "Conversation compacted: "
                f"kept={len(self.state.messages)} "
                f"context_tokens={self.state.context_tokens}"
            )

        except Exception as e:
            self.logger.warning(f"Summarization failed: {e}")

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
            # Check token budget
            context_tokens = self._estimate_messages_tokens(messages)
            max_budget = self.config.get("token_budget", 8000)

            if context_tokens > max_budget:
                if self.state.requires_summarization():
                    self._force_summarization()

                raise TokenLimitExceeded(
                    f"Token budget exceeded: {context_tokens} > {max_budget}"
                )

            # Make the call
            self._last_llm_call_time = datetime.now()
            self._llm_call_count += 1

            response = self.llm.chat(messages=messages)

            return response

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}", exc_info=True)
            raise

    def _get_system_prompt(self) -> str:
        """Get cached system prompt."""
        if self._cached_system_prompt is None:
            self._cached_system_prompt = self._build_system_prompt()
        return self._cached_system_prompt

    # Tool Execution
    def _execute_tool(self, tool_name: str, **kwargs) -> str:
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
        if tool_name not in self.tools:
            error = (
                f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
            )
            self.logger.error(error)
            raise ToolExecutionError(error)

        tool = self.tools[tool_name]

        try:
            self.logger.debug(f"Executing tool: {tool_name} with {kwargs}")

            result = tool.execute(**kwargs)
            result_str = str(result)

            self.state.add_tool_call(tool_name, kwargs, result_str)

            return result_str

        except Exception as e:
            error_msg = f"Tool '{tool_name}' execution failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ToolExecutionError(error_msg) from e

    # Token & Context Management
    def _estimate_messages_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for messages."""
        return sum(self.state._estimate_tokens(str(msg)) for msg in messages)

    def _build_context(self, **kwargs) -> dict[str, Any]:
        """
        Build the full context for LLM calls.

        Override in subclasses to customize context composition.
        """
        return {
            "messages": self._prepare_messages(),
            "summary": self.state.get_context_summary(),
            "state": self.state,
            "metadata": self.state.metadata,
            **(kwargs),
        }

    def _prepare_messages(self) -> list[Message]:
        """Prepare messages for LLM call."""
        messages = [
            {"role": MessageRole.SYSTEM.value, "content": self._get_system_prompt()}
        ]

        # Add summary if available
        if self.state.summary:
            messages.append(
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": f"Previous context summary:\n{self.state.summary}",
                }
            )

        # Add conversation history
        messages.extend(self.state.get_messages())

        return messages

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
                json.dump(state_dict, f, indent=2)

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
    @staticmethod
    def _generate_execution_id() -> str:
        """Generate unique execution ID."""
        import uuid

        return str(uuid.uuid4())[:8]

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.config.get("max_steps", 0) <= 0:
            raise ValueError("max_steps must be positive")
        if self.config.get("token_budget", 0) <= 0:
            raise ValueError("token_budget must be positive")

    def _create_logger(self) -> logging.Logger:
        """Create logger instance."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def _log_init(self) -> None:
        """Log initialization details."""
        self.logger.info(f"[{self.__class__.__name__}] Initialized")
        self.logger.info(f"Tools: {list(self.tools.keys()) or 'None'}")
        self.logger.info(f"Max steps: {self.config.get('max_steps')}")
        self.logger.info(f"Token budget: {self.config.get('token_budget')}")
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
        self.state.status = AgentStatus.IDLE
        if exc_type:
            self.on_error(exc_val)
        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"status={self.state.status.value}, "
            f"steps={self.state.step_count}, "
            f"tools={len(self.tools)})"
        )
