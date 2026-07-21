"""
Local llama.cpp client for the Cogito Agent.

Supports:
- Local inference with GGUF models
- Chat completions
- Structured output (via JSON schema prompting)
- Configurable via model path and parameters
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from llama_cpp import Llama

from .base import BaseLLM, GenerationConfig, Messages

logger = logging.getLogger(__name__)


class LlamaCppLLM(BaseLLM):
    """
    Local llama.cpp wrapper for the Cogito Agent.

    Provides a clean interface to llama.cpp with support for:
    - Chat completions using GGUF models
    - JSON/structured output (via prompting)
    - Configurable context size and threads

    Example:
        >>> llm = LlamaCppLLM(
        ...     model_path="~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        ...     n_ctx=4096,
        ...     n_threads=8
        ... )
        >>> response = llm.chat([{"role": "user", "content": "Hello!"}])
        >>> data = llm.structured_output(
        ...     messages=[{"role": "user", "content": "Extract entities"}],
        ...     schema={"type": "object", "properties": {"entities": {"type": "array"}}}
        ... )
    """

    def __init__(
        self,
        model_path: str,
        model: str | None = None,
        config: GenerationConfig | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LlamaCpp LLM client.

        Args:
            model_path: Path to the GGUF model file
            model: Model name (defaults to model_path basename)
            config: Generation configuration
            n_ctx: Context window size
            n_threads: Number of CPU threads (defaults to auto)
            n_batch: Batch size for processing
            verbose: Enable verbose logging
            **kwargs: Additional llama.cpp parameters
        """
        # Resolve model name
        if model is None:
            model = os.path.basename(model_path).replace(".gguf", "")

        super().__init__(model=model, config=config, **kwargs)

        # Expand user path
        self.model_path = os.path.expanduser(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch

        self.logger = logging.getLogger(f"[LlamaCppLLM:{self.model}]")

        # Validate model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"[LlamaCppLLM] Model file not found: {self.model_path}"
            )

        # Initialize llama.cpp
        try:
            self.client = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                verbose=verbose,
                **self._provider_kwargs,
            )
            self.logger.info(f"Initialized with model: {self.model}")
            self.logger.info(f"  Context size: {n_ctx}")
            self.logger.info(f"  Threads: {n_threads if n_threads else 'auto'}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def chat(self, messages: Messages) -> str:
        """
        Generate a chat completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response as a string

        Raises:
            Exception: On inference errors
        """
        try:
            self.logger.debug(f"Chat request: {len(messages)} messages")

            response = self.client.create_chat_completion(
                messages=messages,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                seed=self.config.seed,
                stream=False,
                **self._provider_kwargs,
            )

            # Extract text from response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            if content is None:
                content = ""

            self.logger.debug(f"Chat response: {len(content)} characters")
            return content.strip()

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

        Since llama.cpp doesn't have native JSON mode, we use prompt engineering
        to encourage valid JSON output.

        Args:
            messages: List of message dicts with 'role' and 'content'
            schema: JSON Schema describing the expected output structure
            **kwargs: Additional provider-specific arguments

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            ValueError: If response is not valid JSON
            Exception: On inference errors
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

            # For llama.cpp, we also add a prefix to guide the output
            messages.append({"role": "assistant", "content": "```json\n"})

            # Call inference
            response = self.client.create_chat_completion(
                messages=messages,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                seed=self.config.seed,
                stream=False,
                **self._provider_kwargs,
                **kwargs,
            )

            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            if content is None:
                content = ""

            # Clean up the response (remove markdown blocks if present)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"Failed to parse JSON response: {content[:200]}..."
                )
                # Try to extract JSON from the response
                import re

                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON response: {str(e)}")
                else:
                    raise ValueError(f"Invalid JSON response: {str(e)}")

            self.logger.debug(f"Structured output: {len(result)} keys")
            return result

        except Exception as e:
            self.logger.error(f"Structured output failed: {str(e)}")
            raise

    def __repr__(self) -> str:
        return f"<LlamaCppLLM model={self.model} path={self.model_path} config={self.config}>"
