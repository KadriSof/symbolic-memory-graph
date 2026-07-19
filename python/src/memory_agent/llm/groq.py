"""
Groq LLM client for the Cogito Agent.

Supports:
- Chat completions
- Structured output (JSON mode)
- Configuration via environment variables
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq as GroqClient
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLM, GenerationConfig, Messages

load_dotenv()

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """
    Groq API wrapper for the Cogito Agent.

    Provides a clean interface to Groq's LLM API with support for:
    - Chat completions
    - JSON/structured output
    - Configurable generation parameters

    Environment variables:
        GROQ_API_KEY: Required for API authentication
        GROQ_MODEL: Default model (overrides constructor)

    Example:
        >>> llm = GroqLLM(model="llama-3.1-8b-instant")
        >>> response = llm.chat([{"role": "user", "content": "Hello!"}])
        >>> data = llm.structured_output(
        ...     messages=[{"role": "user", "content": "Extract entities"}],
        ...     schema={"type": "object", "properties": {"entities": {"type": "array"}}}
        ... )
    """

    def __init__(
        self,
        model: str | None = None,
        config: GenerationConfig | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Groq LLM client.

        Args:
            model: Model identifier (defaults to GROQ_MODEL env or llama-3.1-8b-instant)
            config: Generation configuration
            api_key: Groq API key (defaults to GROQ_API_KEY env)
            **kwargs: Additional provider-specific arguments

        Raises:
            ValueError: If GROQ_API_KEY is not provided or found in environment
        """
        # Resolve model from environment or default
        if model is None:
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        super().__init__(model=model, config=config, **kwargs)

        # Resolve API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "[GroqLLM] Missing GROQ_API_KEY environment variable.\n"
                "Please set it in your .env file or pass it to the constructor."
            )

        # Initialize the client
        self.client = GroqClient(api_key=self.api_key)
        self.logger = logging.getLogger(f"[GroqLLM:{self.model}]")

        self.logger.info(f"Initialized with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat(self, messages: Messages) -> str:
        """
        Generate a chat completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response as a string

        Raises:
            Exception: On API errors
        """
        try:
            self.logger.debug(f"Chat request: {len(messages)} messages")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                seed=self.config.seed,
                **self._provider_kwargs,
            )

            content = response.choices[0].message.content
            if content is None:
                content = ""

            self.logger.debug(f"Chat response: {len(content)} characters")
            return content

        except Exception as e:
            self.logger.error(f"Chat generation failed: {str(e)}")
            raise

    def structured_output(
        self,
        messages: Messages,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate structured output (JSON) from messages.

        Uses Groq's JSON mode to ensure valid JSON output.

        Args:
            messages: List of message dicts with 'role' and 'content'
            schema: JSON Schema describing the expected output structure
            **kwargs: Additional provider-specific arguments

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            ValueError: If response is not valid JSON
            Exception: On API errors
        """
        try:
            self.logger.debug(f"Structured output request: {len(messages)} messages")

            # Add system instruction for JSON output
            system_prompt = (
                "You must respond with valid JSON only. "
                "Do not include any explanatory text outside the JSON object. "
                "The JSON must conform to the provided schema."
            )

            # Inject system message if not already present
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": system_prompt}] + messages
            else:
                # Update existing system message
                for m in messages:
                    if m.get("role") == "system":
                        m["content"] = system_prompt + "\n\n" + m["content"]

            # Add schema instruction
            schema_prompt = f"\n\nExpected JSON schema:\n{json.dumps(schema, indent=2)}"
            messages[-1]["content"] = messages[-1]["content"] + schema_prompt

            # Call API with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                seed=self.config.seed,
                response_format={"type": "json_object"},
                **self._provider_kwargs,
                **kwargs,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from Groq API")

            # Parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"Failed to parse JSON response: {content[:200]}..."
                )
                raise ValueError(f"Invalid JSON response: {str(e)}")

            self.logger.debug(f"Structured output: {len(result)} keys")
            return result

        except Exception as e:
            self.logger.error(f"Structured output failed: {str(e)}")
            raise

    def __repr__(self) -> str:
        return f"<GroqLLM model={self.model} config={self.config}>"
