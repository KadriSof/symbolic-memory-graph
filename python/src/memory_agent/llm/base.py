"""
Base LLM abstractions for the Cogito Agent.
Compact, focused, and leveraging the robust JSON parser.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Type, TypeVar, Optional

from pydantic import BaseModel

from memory_agent.utils.parser import StructuredOutputParser

RoleType = Literal["system", "user", "assistant", "tool"]
Message = dict[str, str]
Messages = list[Message]

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Runtime generation parameters."""

    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    Two core methods:
    1. chat() - natural language response
    2. get_structured() - structured output with automatic parsing
    """

    def __init__(
        self, model: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> None:
        self.model = model
        self.config = config or GenerationConfig()
        self._provider_kwargs = kwargs

        # One parser instance, reused for all structured calls
        self._parser = StructuredOutputParser(
            fallback_on_error=True,
            log_errors=True,
        )

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sanitize messages to only include keys accepted by LLM APIs (Groq/OpenAI).
        
        Removes internal state metadata like 'timestamp', 'tokens', 'step', etc.
        that were added by BaseState.add_message() for tracking purposes.
        
        Args:
            messages: List of message dictionaries from agent state
            
        Returns:
            List of cleaned message dictionaries safe for LLM API calls
        """
        allowed_keys = {"role", "content", "name", "tool_calls", "tool_call_id"}
        
        sanitized = []
        for msg in messages:
            clean_msg = {k: v for k, v in msg.items() if k in allowed_keys}
            sanitized.append(clean_msg)
            
        return sanitized

    @abstractmethod
    def chat(self, messages: Messages) -> str:
        """Generate a chat completion."""
        raise NotImplementedError

    @abstractmethod
    def chat_stream(self, messages: Messages) -> str:
        """Generate a chat stream."""
        raise NotImplementedError

    @abstractmethod
    def structured_output(
        self, messages: Messages, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Generate structured output using provider's native support.

        If not supported, raise NotImplementedError to trigger fallback.
        """
        raise NotImplementedError

    def get_structured(
        self,
        messages: Messages,
        model: Type[T],
        **kwargs: Any,
    ) -> Optional[T]:
        """
        Get structured output with automatic fallback to chat+parse.

        Uses the robust JSON parser we already built.
        """
        messages = self._sanitize_messages(messages)

        try:
            # Try native structured output first
            try:
                schema = model.model_json_schema()

            except AttributeError:
                # Fallback for Pydantic v1 (if there are anyone there still using it --)
                schema = model.schema() if hasattr(model, "schema") else {}  # type: ignore

            raw = self.structured_output(messages, schema, **kwargs)
            return self._parser.parse_dict(raw, model)

        except NotImplementedError:
            # Fallback: chat + parse
            logger.debug(f"{self.__class__.__name__}: Using chat+parse fallback")
            response = self.chat(messages, **kwargs)
            return self._parser.parse(response, model)

        except Exception as e:
            logger.error(f"Structured output failed: {e}")
            return None

    def with_config(self, **kwargs: Any) -> "BaseLLM":
        """Create a copy with updated configuration."""
        from dataclasses import replace

        new_config = replace(self.config, **kwargs)
        return self.__class__(
            model=kwargs.get("model", self.model),
            config=new_config,
            **self._provider_kwargs,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model}>"
