"""
Llama.cpp LLM client for local inference.

Supports:
- Chat completions
- Streaming chat completions
- Structured output (JSON mode, grammar-constrained)

Uses llama-cpp-python's `create_chat_completion` API, which applies the
model's built-in chat template automatically. Do NOT build prompts by hand —
hardcoded templates (e.g. `[INST]`) cause the model to echo the prompt
instead of generating a response.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Generator

from dotenv import load_dotenv
from llama_cpp import Llama, LlamaGrammar
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.parser import JSONRepair
from .base import BaseLLM, GenerationConfig, Messages

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


# Stop tokens by model family. Add more as needed.
DEFAULT_STOP_TOKENS: dict[str, list[str]] = {
    "qwen2": ["<|im_end|>", "<|endoftext|>"],
    "qwen": ["<|im_end|>", "<|endoftext|>"],
    "chatml": ["<|im_end|>", "<|im_start|>"],
    "granite": ["<|end_of_text|>", "<|start_of_role|>"],
    "llama-3": ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
    "llama-2": ["[INST]", "[/INST]"],
    "mistral": ["[INST]", "[/INST]"],
    "gemma": ["<end_of_turn>", "<start_of_turn>"],
}


class LlamaCppLLM(BaseLLM):
    """
    Llama.cpp API wrapper for local LLM inference.

    Provides a clean interface to Llama.cpp with support for:
    - Chat completions (via create_chat_completion — uses model's chat template)
    - Streaming chat completions
    - JSON/structured output (via response_format / grammar)

    Environment variables:
        LLAMA_CPP_MODEL_PATH: Path to the GGUF model file
        LLAMA_CPP_N_CTX: Maximum context length (default: 8192)
        LLAMA_CPP_N_THREADS: Number of CPU threads (default: 4)
        LLAMA_CPP_CHAT_FORMAT: Optional chat format override (e.g. "qwen2")
    """

    def __init__(
        self,
        model_path: str | None = None,
        config: GenerationConfig | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        chat_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Llama.cpp LLM client.

        Args:
            model_path: Path to the GGUF model file (defaults to LLAMA_CPP_MODEL_PATH env)
            config: Generation configuration
            n_ctx: Maximum context length (defaults to LLAMA_CPP_N_CTX env or 8192)
            n_threads: Number of CPU threads (defaults to LLAMA_CPP_N_THREADS env or 4)
            chat_format: Optional chat format override (defaults to LLAMA_CPP_CHAT_FORMAT
                         env or the model's embedded template)
            **kwargs: Additional Llama constructor arguments
        """
        # Resolve model path
        if model_path is None:
            model_path = os.getenv("LLAMA_CPP_MODEL_PATH")
            if not model_path:
                raise ValueError(
                    "[LlamaCppLLM] Missing LLAMA_CPP_MODEL_PATH environment variable.\n"
                    "Please set it in your .env file or pass it to the constructor."
                )

        # Resolve context length and threads
        self.n_ctx = n_ctx or int(os.getenv("LLAMA_CPP_N_CTX", "8192"))
        self.n_threads = n_threads or int(os.getenv("LLAMA_CPP_N_THREADS", "4"))

        # Resolve chat format (may be None — llama.cpp will use the embedded template)
        chat_format = chat_format or os.getenv("LLAMA_CPP_CHAT_FORMAT") or None

        super().__init__(model=model_path, config=config, **kwargs)

        # Initialize the Llama.cpp client
        self.client = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            chat_format=chat_format,
            verbose=False,
        )

        self.logger = logging.getLogger(f"[LlamaCppLLM:{model_path}]")
        self.logger.info(f"Initialized with model: {model_path}")

        # Resolve the effective chat format for stop-token selection
        self._chat_format = (
            chat_format or self._detect_chat_format_from_metadata() or "chatml"
        )
        self._stop_tokens = self._resolve_stop_tokens()

        self.logger.debug(
            f"chat_format={self._chat_format}, stop_tokens={self._stop_tokens}"
        )

    # PUBLIC API
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def chat(self, messages: Messages, **kwargs: Any) -> str:
        """
        Generate a chat completion from messages.

        Uses `create_chat_completion`, which applies the model's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response as a string

        Raises:
            Exception: On inference errors
        """
        messages = self._sanitize_messages(messages)

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
                stop=self._stop_tokens,
                **kwargs,
            )

            content = response["choices"][0]["message"]["content"] or ""  # type: ignore
            self.logger.debug(f"Chat response: {len(content)} characters")
            return content

        except Exception as e:
            self.logger.error(f"Chat generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def chat_stream(
        self, messages: Messages, **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Generate a streaming chat completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Yields:
            Response content chunks (strings)

        Raises:
            Exception: On inference errors
        """
        messages = self._sanitize_messages(messages)

        try:
            self.logger.debug(f"Streaming chat request: {len(messages)} messages")

            stream = self.client.create_chat_completion(
                messages=messages,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                seed=self.config.seed,
                stop=self._stop_tokens,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})  # type: ignore
                content = delta.get("content")
                if content:
                    yield content

        except Exception as e:
            self.logger.error(f"Streaming chat generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def structured_output(
        self,
        messages: Messages,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate structured output (JSON) from messages.

        Uses the model's chat template plus either grammar-constrained
        sampling (preferred) or `response_format={"type": "json_object"}`.

        Args:
            messages: List of message dicts with 'role' and 'content'
            schema: JSON Schema describing the expected output structure
            **kwargs: Additional arguments

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            ValueError: If response is not valid JSON
            Exception: On inference errors
        """
        messages = self._sanitize_messages(messages)

        # Prepend a system message instructing JSON-only output
        system_prompt = (
            "You must respond with valid JSON only. "
            "Do not include any explanatory text outside the JSON object. "
            "The JSON must conform to the provided schema."
        )

        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages = [
                (
                    {**m, "content": f"{system_prompt}\n\n{m['content']}"}
                    if m.get("role") == "system"
                    else m
                )
                for m in messages
            ]

        # Build a grammar to constrain sampling (preferred)
        grammar: LlamaGrammar | None = None
        try:
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        except Exception as e:
            self.logger.debug(
                f"[structured_output] Grammar construction failed, "
                f"falling back to response_format: {e}"
            )

        try:
            completion_kwargs: dict[str, Any] = dict(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                seed=self.config.seed,
                stop=self._stop_tokens,
                **kwargs,
            )

            if grammar is not None:
                completion_kwargs["grammar"] = grammar
            else:
                completion_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.create_chat_completion(**completion_kwargs)
            content = response["choices"][0]["message"]["content"] or ""  # type: ignore

            if not content:
                raise ValueError("Empty response from Llama.cpp")

            # Attempt native parse, then repair
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                self.logger.debug(
                    "[structured_output] Native JSON parse failed, "
                    "attempting robust repair..."
                )

            repaired = JSONRepair.repair_and_validate(text=content)
            if repaired:
                self.logger.debug("[structured_output] Successfully repaired JSON")
                return json.loads(repaired)

            raise ValueError("[structured_output] Invalid JSON and repair failed.")

        except ValueError:
            raise

        except Exception as e:
            self.logger.error(
                f"[structured_output] Llama.cpp request failed:\n{str(e)}\n---"
            )
            raise

    # INTERNAL HELPERS
    def _detect_chat_format_from_metadata(self) -> str | None:
        """
        Try to infer the chat format from GGUF metadata.

        Returns None if it cannot be inferred.
        """
        try:
            metadata = getattr(self.client, "metadata", {}) or {}
            template = metadata.get("tokenizer.chat_template") or ""
            template_lower = template.lower()

            if "<|im_start|>" in template_lower:
                return "chatml"
            if "<|eot_id|>" in template_lower:
                return "llama-3"
            if "[inst]" in template_lower:
                return "llama-2"
            if "<start_of_turn>" in template_lower:
                return "gemma"
            if "<|start_of_role|>" in template_lower:
                return "granite"
        except Exception:
            pass

        return None

    def _resolve_stop_tokens(self) -> list[str]:
        """
        Return a stop-token list appropriate for the active chat format.
        Falls back to a generic set.
        """
        if self._chat_format in DEFAULT_STOP_TOKENS:
            return DEFAULT_STOP_TOKENS[self._chat_format]

        # Try a partial match
        fmt = (self._chat_format or "").lower()
        for key, tokens in DEFAULT_STOP_TOKENS.items():
            if key in fmt:
                return tokens

        # Generic fallback — safe for most modern chat models
        return ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>"]

    def __repr__(self) -> str:
        return (
            f"<LlamaCppLLM model_path={self.model} "
            f"chat_format={self._chat_format} config={self.config}>"
        )


# Manual test
if __name__ == "__main__":
    import time

    from memory_agent.llm.base import GenerationConfig

    start_time = time.perf_counter()

    model_path = os.path.expanduser("~/models/qwen2-0.5b-instruct-q5_0.gguf")

    config = GenerationConfig(temperature=0.2, max_tokens=256)
    llm = LlamaCppLLM(model_path=model_path, config=config)

    # Sanity check — must NOT echo the prompt
    print("--- Single-turn sanity ---")
    resp = llm.chat([{"role": "user", "content": "Reply with only the word: OK"}])
    print(f"Response: {resp!r}")

    # Multi-turn check
    print("\n--- Multi-turn ---")
    resp = llm.chat(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    )
    print(f"Response: {resp!r}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTime: {elapsed:.2f}s")
