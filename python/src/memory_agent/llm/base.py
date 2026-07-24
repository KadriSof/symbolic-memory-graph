"""
Base LLM abstractions for the Cogito Agent.

Design goals:
- provider-agnostic
- lightweight
- synchronous-first
- easy to extend
- supports structured output for agent steps
- no framework coupling

Any future provider should implement:
- chat(messages) -> str
- structured_output(messages, schema) -> dict
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# Type aliases for clarity
RoleType = Literal["system", "user", "assistant", "tool"]
ROLES = ("system", "user", "assistant", "tool")
Message = dict[str, str]
Messages = list[Message]


@dataclass
class GenerationConfig:
    """
    Runtime generation parameters for LLM providers.

    Attributes:
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling threshold
        frequency_penalty: Penalty for repeated tokens
        presence_penalty: Penalty for new topics
        seed: Random seed for reproducibility
    """

    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None

    # Provider-specific overrides
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    The Cogito Agent uses this interface to interact with any LLM.
    Providers should implement:
    1. chat() - for natural language interaction
    2. structured_output() - for structured data extraction

    Example:
        >>> llm = GroqLLM(model="mixtral-8x7b-32768")
        >>> response = llm.chat([{"role": "user", "content": "Hello!"}])
        >>> data = llm.structured_output(
        ...     messages=[{"role": "user", "content": "Extract entities"}],
        ...     schema={"type": "object", "properties": {"entities": {"type": "array"}}}
        ... )
    """

    def __init__(
        self, model: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> None:
        """
        Initialize the LLM provider.

        Args:
            model: Model identifier (provider-specific)
            config: Generation configuration
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.config = config or GenerationConfig()
        self._provider_kwargs = kwargs

    @abstractmethod
    def chat(self, messages: Messages) -> str:
        """
        Generate a chat completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response as a string

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is 2+2?"}
            ... ]
            >>> response = llm.chat(messages)
            "2+2 equals 4"
        """
        raise NotImplementedError

    @abstractmethod
    def structured_output(
        self, messages: Messages, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Generate structured output (JSON) from messages.

        This is used for entity extraction, consolidation, and other agent steps
        that require structured data.

        Args:
            messages: List of message dicts with 'role' and 'content'
            schema: JSON Schema describing the expected output structure
            **kwargs: Additional provider-specific arguments

        Returns:
            Parsed JSON response as a dictionary

        Example:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "entities": {"type": "array", "items": {"type": "string"}},
            ...         "relations": {"type": "array", "items": {"type": "object"}}
            ...     }
            ... }
            >>> result = llm.structured_output(messages, schema)
            {"entities": ["Geralt", "Ciri"], "relations": [{"source": "Geralt", "target": "Ciri"}]}
        """
        raise NotImplementedError

    def with_config(self, **kwargs: Any) -> "BaseLLM":
        """
        Create a copy with updated configuration.

        This is useful for one-off requests with different parameters.

        Example:
            >>> creative_llm = llm.with_config(temperature=0.9, max_tokens=4096)
        """
        new_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            frequency_penalty=kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            ),
            presence_penalty=kwargs.get(
                "presence_penalty", self.config.presence_penalty
            ),
            seed=kwargs.get("seed", self.config.seed),
            extra=kwargs.get("extra", self.config.extra.copy()),
        )

        return self.__class__(
            model=kwargs.get("model", self.model),
            config=new_config,
            **self._provider_kwargs,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model} config={self.config}>"
