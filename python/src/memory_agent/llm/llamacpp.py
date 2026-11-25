"""
Llama.cpp LLM client for local inference and llama-server HTTP communication.

Supports:
- Chat completions (Local and HTTP)
- Streaming chat completions (Local and HTTP)
- Structured output (JSON mode, grammar-constrained) (Local and HTTP)
- System One Decision Routing (HTTP only, native to /v1/systemone)

Method Naming Convention:
- chat, chat_stream, structured_output: Uses the local `llama_cpp.Llama` client.
- chat_http, chat_stream_http, structured_output_http, systemone_http: Uses `httpx` to communicate with `llama-server`.
"""

import os
import json
import httpx
import logging

from typing import Any, Generator

from dotenv import load_dotenv
from llama_cpp import Llama, LlamaGrammar
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.parser import JSONRepair
from .base import BaseLLM, GenerationConfig, Messages

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


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
    def __init__(
        self,
        model_path: str | None = None,
        config: GenerationConfig | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        chat_format: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        if model_path is None:
            model_path = os.getenv("LLAMA_CPP_MODEL_PATH")

        if model_path:
            model_path = os.path.expanduser(model_path)

        self.base_url = (base_url or os.getenv("LLAMA_SERVER_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLAMA_SERVER_API_KEY", "")

        if model_path is None and not self.base_url:
            raise ValueError(
                "[LlamaCppLLM] Provide either a model path (local inference) "
                "or a base_url (llama-server HTTP inference). Set "
                "LLAMA_CPP_MODEL_PATH / LLAMA_SERVER_URL in your .env file "
                "or pass them to the constructor."
            )

        self.model_name = model_path or "llama-server"
        self.n_ctx = n_ctx or int(os.getenv("LLAMA_CPP_N_CTX", "8192"))
        self.n_threads = n_threads or int(os.getenv("LLAMA_CPP_N_THREADS", "4"))
        self._chat_format = chat_format or os.getenv("LLAMA_CPP_CHAT_FORMAT") or None

        super().__init__(model=self.model_name, config=config, **kwargs)

        self._model_path = model_path
        self._client: Llama | None = None
        self._chat_format = self._chat_format or "chatml"
        self._stop_tokens = self._resolve_stop_tokens()
        self.http_client = httpx.Client(timeout=120.0)

        mode = "HTTP-only" if not model_path else "local + HTTP"
        self.logger = logging.getLogger(f"[LlamaCppLLM:{self.model_name}]")
        self.logger.info(
            f"Initialized ({mode}) with model: {self.model_name or 'None'} "
            f"and HTTP base_url: {self.base_url or 'None'}"
        )

    @property
    def client(self) -> Llama | None:
        if self._model_path is None:
            return None
        if self._client is None:
            if not os.path.isfile(self._model_path):
                raise ValueError(
                    f"[LlamaCppLLM] Model path does not exist: {self._model_path}"
                )
            try:
                self._client = Llama(
                    model_path=self._model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    chat_format=self._chat_format,
                    verbose=False,
                )
            except ValueError as e:
                raise ValueError(
                    f"[LlamaCppLLM] Failed to load local model {self._model_path}: {e}"
                ) from e

            detected = self._detect_chat_format_from_metadata()
            if detected and self._chat_format == "chatml":
                self._chat_format = detected
                self._stop_tokens = self._resolve_stop_tokens()
        return self._client

    @property
    def _http_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # LOCAL INFERENCE METHODS
    def _require_local(self) -> Llama:
        client = self.client
        if client is None:
            raise ValueError(
                "[LlamaCppLLM] No local model configured; this method requires "
                "a model path. Set LLAMA_CPP_MODEL_PATH or pass model_path."
            )
        return client

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat(self, messages: Messages, **kwargs: Any) -> str:
        messages = self._sanitize_messages(messages)
        client = self._require_local()
        try:
            response = client.create_chat_completion(
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
            return response["choices"][0]["message"]["content"] or ""  # type: ignore
        except Exception as e:
            self.logger.error(f"Local chat generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat_stream(
        self, messages: Messages, **kwargs: Any
    ) -> Generator[str, None, None]:
        messages = self._sanitize_messages(messages)
        client = self._require_local()
        try:
            stream = client.create_chat_completion(
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
            self.logger.error(f"Local streaming chat generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def structured_output(
        self, messages: Messages, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        messages = self._sanitize_messages(messages)
        client = self._require_local()

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

        grammar: LlamaGrammar | None = None
        try:
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        except Exception as e:
            self.logger.debug(
                f"[structured_output] Grammar construction failed, falling back: {e}"
            )

        try:
            completion_kwargs = dict(
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
                completion_kwargs["grammar"] = grammar  # type: ignore
            else:
                completion_kwargs["response_format"] = {"type": "json_object"}  # type: ignore

            response = client.create_chat_completion(**completion_kwargs)  # type: ignore
            content = response["choices"][0]["message"]["content"] or ""  # type: ignore

            if not content:
                raise ValueError("Empty response from Llama.cpp")

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                repaired = JSONRepair.repair_and_validate(text=content)
                if repaired:
                    return json.loads(repaired)
                raise ValueError("[structured_output] Invalid JSON and repair failed.")
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(
                f"[structured_output] Llama.cpp request failed:\n{str(e)}\n---"
            )
            raise

    # HTTP LLAMA-SERVER METHODS
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat_http(self, messages: Messages, **kwargs: Any) -> str:
        messages = self._sanitize_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "seed": self.config.seed,
            "stop": self._stop_tokens,
            **kwargs,
        }
        response = self.http_client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._http_headers,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def chat_stream_http(
        self, messages: Messages, **kwargs: Any
    ) -> Generator[str, None, None]:
        messages = self._sanitize_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "seed": self.config.seed,
            "stop": self._stop_tokens,
            "stream": True,
            **kwargs,
        }
        with self.http_client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._http_headers,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def structured_output_http(
        self, messages: Messages, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        messages = self._sanitize_messages(messages)

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

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "seed": self.config.seed,
            "stop": self._stop_tokens,
            "response_format": {"type": "json_object"},
            **kwargs,
        }

        try:
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
            payload["grammar"] = str(grammar)
        except Exception:
            pass

        response = self.http_client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._http_headers,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""

        if not content:
            raise ValueError("Empty response from llama-server")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            repaired = JSONRepair.repair_and_validate(text=content)
            if repaired:
                return json.loads(repaired)
            raise ValueError("[structured_output_http] Invalid JSON and repair failed.")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def systemone_http(
        self, state: str, questions: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Query a decision model via the native /v1/systemone endpoint.
        This is the optimal and intended way to use decision models (e.g., Laya, Kev-4B) in llama.cpp.
        """
        payload = {
            "model": self.model_name,
            "state": state,
            "questions": questions,
            **kwargs,
        }
        response = self.http_client.post(
            f"{self.base_url}/v1/systemone",
            json=payload,
            headers=self._http_headers,
        )
        response.raise_for_status()
        return response.json()

    # INTERNAL HELPERS
    def _detect_chat_format_from_metadata(self) -> str | None:
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
        if self._chat_format in DEFAULT_STOP_TOKENS:
            return DEFAULT_STOP_TOKENS[self._chat_format]

        fmt = (self._chat_format or "").lower()
        for key, tokens in DEFAULT_STOP_TOKENS.items():
            if key in fmt:
                return tokens

        return ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>"]

    def __repr__(self) -> str:
        return (
            f"<LlamaCppLLM model={self.model_name} "
            f"base_url={self.base_url or 'None'} "
            f"chat_format={self._chat_format} config={self.config}>"
        )
