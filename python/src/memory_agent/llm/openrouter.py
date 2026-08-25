"""
Open Router LLM client for the Cogito Agent.

Supports:
- Chat completions
- Structured output (JSON mode)
- Configuration via environment variables
"""

from __future__ import annotations

import os
import json
import httpx
import logging

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from typing import Any, Generator

from ..utils.parser import JSONRepair
from .base import BaseLLM, GenerationConfig, Messages

load_dotenv()

logger = logging.getLogger(__name__)


class OpenRouterLLM(BaseLLM):
    """
    Open Router API wrapper for the Cogito Agent.

    Provides a clean interface to Open Router's LLM API with support for:
    - Chat completions
    - JSON/structured output
    - Configurable generation parameters

    Environment variables:
        OPENROUTER_API_KEY: Required for API authentication
        OPENROUTER_MODEL: Default model (overrides constructor)

    Example:
        >>> llm = OpenRouterLLM(model="openai/gpt-4o-mini")
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
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Open Router LLM client.

        Args:
            model: Model identifier (defaults to OPENROUTER_MODEL env or openai/gpt-4o-mini)
            config: Generation configuration
            api_key: Open Router API key (defaults to OPENROUTER_API_KEY env)
            site_url: Your site URL for Open Router rankings (optional)
            site_name: Your site name for Open Router rankings (optional)
            **kwargs: Additional provider-specific arguments

        Raises:
            ValueError: If OPENROUTER_API_KEY is not provided or found in environment
        """
        # Resolve model from environment or default
        if model is None:
            model = os.getenv(
                "OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
            )

        super().__init__(model=model, config=config, **kwargs)

        # Resolve API key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "[OpenRouterLLM] Missing OPENROUTER_API_KEY environment variable.\n"
                "Please set it in your .env file or pass it to the constructor."
            )

        # Optional site metadata for Open Router rankings
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", None)
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", None)

        # Initialize the HTTP client
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self.logger = logging.getLogger(f"[OpenRouterLLM:{self.model}]")

        self.logger.info(f"Initialized with model: {self.model}")

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests, including optional site metadata."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def _sanitize_messages(self, messages: Messages) -> Messages:
        """Ensure messages are in the correct format for Open Router."""
        sanitized = []
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Invalid message format: {msg}")
            sanitized.append({"role": msg["role"], "content": msg["content"]})
        return sanitized

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat(self, messages: Messages, **kwargs: Any) -> str:
        """
        Generate a chat completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response as a string

        Raises:
            Exception: On API errors
        """
        messages = self._sanitize_messages(messages)

        try:
            self.logger.debug(f"Chat request: {len(messages)} messages")

            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
            }

            if self.config.seed is not None:
                params["seed"] = self.config.seed

            params.update(kwargs)

            response = self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self._get_headers(),
                json=params,
            )

            response_data = response.json()

            response.raise_for_status()

            # ✅ FIX: Check for OpenRouter API error payloads
            if "error" in response_data:
                raise Exception(f"OpenRouter API error: {response_data['error']}")

            # ✅ FIX: Safely access choices
            choices = response_data.get("choices", [])

            if not choices:
                raise Exception("OpenRouter returned empty choices array")

            content = choices[0]["message"].get("content", "")
            content = response.json()["choices"][0]["message"]["content"]

            self.logger.debug(f"Chat response: {len(content)} characters")
            return content

        except Exception as e:
            self.logger.error(f"Chat generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat_stream(
        self,
        messages: Messages,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming chat completion.

        OpenRouter returns streaming responses as Server-Sent Events (SSE).
        Each JSON payload is sent as:

            data: {...}

        and the stream terminates with:

            data: [DONE]
        """
        messages = self._sanitize_messages(messages)

        try:
            self.logger.debug(f"Streaming chat request: {len(messages)} messages")

            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stream": True,
            }

            if self.config.seed is not None:
                params["seed"] = self.config.seed

            params.update(kwargs)

            with self.client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self._get_headers(),
                json=params,
            ) as response:

                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    # httpx may return str or bytes depending on configuration.
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")

                    # OpenRouter uses SSE:
                    #     data: {...}
                    if line.startswith("data:"):
                        line = line[5:].strip()

                    # End of stream.
                    if line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping invalid SSE payload: {line!r}")
                        continue

                    choices = chunk.get("choices", [])

                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    output = delta.get("content")

                    if output:
                        yield output

        except Exception as e:
            self.logger.error(f"Streaming chat generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def structured_output(
        self,
        messages: Messages,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate structured output (JSON) from messages.

        Uses Open Router's JSON mode to ensure valid JSON output.

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

            safe_messages = [{**m} for m in messages]

            # Add system instruction for JSON output
            system_prompt = (
                "You must respond with valid JSON only. "
                "Do not include any explanatory text outside the JSON object. "
                "The JSON must conform to the provided schema."
            )

            # Inject system message if not already present
            has_system = any(m.get("role") == "system" for m in safe_messages)
            if not has_system:
                safe_messages = [
                    {"role": "system", "content": system_prompt}
                ] + safe_messages
            else:
                # Update existing system message
                for m in safe_messages:
                    if m.get("role") == "system":
                        m["content"] = f"{system_prompt}\n\n{m['content']}"

            # Add schema instruction
            schema_prompt = f"\n\nExpected JSON schema:\n{json.dumps(schema, indent=2)}"
            safe_messages[-1] = {
                **safe_messages[-1],
                "content": safe_messages[-1]["content"] + schema_prompt,
            }

            # Call API with JSON mode
            params = {
                "model": self.model,
                "messages": safe_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "response_format": {"type": "json_object"},
            }

            if self.config.seed is not None:
                params["seed"] = self.config.seed

            params.update(kwargs)

            response = self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=self._get_headers(),
                json=params,
            )

            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            if content is None:
                raise ValueError("Empty response from Open Router API")

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                self.logger.debug(
                    "[OpenRouterLLM:structured_output] Native JSON parse failed, attempting robust repair..."
                )

                repaired = JSONRepair.repair_and_validate(text=content)
                if repaired:
                    self.logger.debug(
                        "[OpenRouterLLM:structured_output] Successfully repaired JSON"
                    )
                    return json.loads(repaired)

                raise ValueError(
                    "[OpenRouterLLM:structured_output] Invalid JSON and repair failed. Triggering chat fallback..."
                )

        except ValueError:
            # We raise ValueError so the BaseLLM class catches it and triggers the chat fallback
            raise

        except Exception as e:
            self.logger.error(
                f"[OpenRouterLLM:structured_output] Open Router API request failed:\n{str(e)}\n---"
            )
            raise

    def __repr__(self) -> str:
        return f"<OpenRouterLLM model={self.model} config={self.config}>"


if __name__ == "__main__":
    from memory_agent.llm.base import GenerationConfig

    # Initialize the client
    config = GenerationConfig(temperature=0.7, max_tokens=2048)
    llm = OpenRouterLLM(
        model="cohere/north-mini-code:free",
        config=config,
        site_url="https://my-site.com",
        site_name="My App",
    )

    # Chat completion
    response = llm.chat([{"role": "user", "content": "Hello!"}])
    print(response)

    # Streaming
    for chunk in llm.chat_stream([{"role": "user", "content": "Tell me a story."}]):
        print(chunk, end="", flush=True)

    # Structured output
    schema = {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    }
    data = llm.structured_output(
        messages=[
            {
                "role": "user",
                "content": "Extract entities from this text: Apple, Banana",
            }
        ],
        schema=schema,
    )
    print(data)
