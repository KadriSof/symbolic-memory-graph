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

from __future__ import annotations

import json
import logging
import os
from typing import Any, Generator

import httpx
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

    # =========================================================================
    # LOCAL INFERENCE METHODS
    # =========================================================================

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
            return response["choices"][0]["message"]["content"] or ""
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
                messages=messages,
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
                delta = chunk["choices"][0].get("delta", {})
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
                completion_kwargs["grammar"] = grammar
            else:
                completion_kwargs["response_format"] = {"type": "json_object"}

            response = client.create_chat_completion(**completion_kwargs)
            content = response["choices"][0]["message"]["content"] or ""

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

    # =========================================================================
    # HTTP LLAMA-SERVER METHODS
    # =========================================================================

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

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

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


# ==============================================================================
# USAGE EXAMPLE & ROUTING LOGIC TEST HARNESS
# Benchmarks System One routing against a running llama-server endpoint via
# the native /v1/systemone decision model endpoint.
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import time
    from typing import Any

    DESC_REPLY = (
        "Answer directly from memory or stable general knowledge; no tool is needed."
    )
    DESC_REACT = "Use a listed available tool that can obtain the required information."
    DESC_REQUEST = "Ask the user for essential missing information that no available tool can obtain."

    ROUTE_DESCS = {"REPLY": DESC_REPLY, "REACT": DESC_REACT, "REQUEST": DESC_REQUEST}

    INSTRUCTION = (
        "Choose exactly the best route for handling the request now, "
        "considering the request, relevant memory, and available tools. "
        "REPLY means answer directly from memory or stable general knowledge. "
        "REACT means a listed available tool is required and can fulfill the "
        "request. REQUEST means essential information is missing and no "
        "available tool can obtain it."
    )

    # id | expected | user request | relevant memory | available tools
    CASES = [
        (
            "R01",
            "REPLY",
            "What is the capital of France?",
            "The capital of France is Paris.",
            "None",
        ),
        (
            "R02",
            "REPLY",
            "What does BFS stand for in graph theory?",
            "BFS stands for Breadth-First Search.",
            "None",
        ),
        (
            "R03",
            "REPLY",
            "What is the purpose of a hash table?",
            "A hash table stores key-value pairs.",
            "None",
        ),
        (
            "Q01",
            "REQUEST",
            "Why did my program crash?",
            "No source code or error message provided.",
            "None",
        ),
        (
            "Q02",
            "REQUEST",
            "Which of my two implementations is faster?",
            "Neither implementation provided.",
            "None",
        ),
        (
            "Q03",
            "REQUEST",
            "What did I name the function in my unshared source file?",
            "The source file is not present.",
            "None",
        ),
        (
            "T01",
            "REACT",
            "What is the weather in Tunis right now?",
            "No current weather information available.",
            "weather - returns current weather",
        ),
        (
            "T02",
            "REACT",
            "Search the web for the latest release of the Linux kernel.",
            "No information available.",
            "web_search - searches the live web",
        ),
        (
            "T03",
            "REACT",
            "Read the contents of config.toml in the current project.",
            "The file contents are not available.",
            "file_read - reads a local file",
        ),
    ]

    ORDERS = [
        ("REPLY", "REACT", "REQUEST"),
        ("REQUEST", "REPLY", "REACT"),
        ("REACT", "REQUEST", "REPLY"),
    ]

    def normalize_choice(raw: str) -> str | None:
        if not raw:
            return None
        if raw == "REPLY" or raw == DESC_REPLY or raw.startswith("Answer directly"):
            return "REPLY"
        if (
            raw == "REACT"
            or raw == DESC_REACT
            or raw.startswith("Use a listed available tool")
        ):
            return "REACT"
        if raw == "REQUEST" or raw == DESC_REQUEST or raw.startswith("Ask the user"):
            return "REQUEST"
        return None

    def run_case_routing(llm: "LlamaCppLLM", state: str) -> dict[str, Any]:
        """
        Run one case against the /v1/systemone endpoint three times (cyclic option
        permutations) and return the majority route plus per-route vote counts.
        """
        votes = {"REPLY": 0, "REACT": 0, "REQUEST": 0}
        round_log: list[str] = []
        valid_rounds = 0

        for order in ORDERS:
            criteria = {label: ROUTE_DESCS[label] for label in order}
            questions = {
                "decision": {
                    "type": "choice",
                    "instructions": INSTRUCTION,
                    "criteria": criteria,
                }
            }
            try:
                # Use the native /v1/systemone endpoint designed for decision models
                result = llm.systemone_http(state=state, questions=questions)
                answer = result["answers"]["decision"]
            except Exception as e:
                round_log.append(f"  [{'/'.join(order)}] ERROR: {e}")
                continue

            mapped = normalize_choice(answer.get("choice", ""))
            if mapped is None:
                round_log.append(
                    f"  [{'/'.join(order)}] ERROR: unrecognized choice: {answer.get('choice', '')!r}"
                )
                continue

            votes[mapped] += 1
            valid_rounds += 1

            # Log probabilities if the decision model provides them
            probs = answer.get("probabilities", {})
            prob_str = " ".join([f"{k}={v:.3f}" for k, v in probs.items()])
            round_log.append(f"  [{'/'.join(order)}] choice={mapped} ({prob_str})")

        tie_break = ("REPLY", "REACT", "REQUEST")
        route = (
            max(tie_break, key=lambda r: (votes[r], -tie_break.index(r)))
            if valid_rounds > 0
            else "NONE"
        )

        return {
            "valid": 1 if valid_rounds > 0 else 0,
            "route": route,
            "votes": votes,
            "log": "\n".join(round_log) + "\n",
        }

    # =====================================================================
    # CLI & benchmark run
    # =====================================================================

    parser = argparse.ArgumentParser(
        description="System One Cogito Routing Benchmark (llama-server HTTP)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080"),
        help="Base URL of the running llama-server (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("LLAMA_SERVER_API_KEY", ""),
        help="Optional API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="laya",
        help="Model name reported in the HTTP payload",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max tokens per request"
    )
    args = parser.parse_args()

    # HTTP-only LlamaCppLLM: no local GGUF is loaded; all inference goes to the llama-server endpoint.
    llm = LlamaCppLLM(
        base_url=args.base_url,
        api_key=args.api_key or None,
        config=GenerationConfig(
            temperature=args.temperature, max_tokens=args.max_tokens
        ),
    )
    llm.model_name = args.model

    # Health check, mirroring the bash script's pre-flight probe.
    try:
        health = llm.http_client.get(f"{llm.base_url}/health")
        health.raise_for_status()
    except Exception as e:
        raise SystemExit(f"ERROR: llama-server unavailable at {llm.base_url}\n  {e}")

    print("=" * 60)
    print(" SYSTEM ONE — COGITO ROUTING BENCHMARK")
    print(" Method: cyclic option rotation; majority vote vs ground truth")
    print("=" * 60)
    print(f"Endpoint: {llm.base_url}/v1/systemone")
    print("Cases:    9 (3 per expected route)")
    print("Rounds:   3 calls per case (up to 27 requests)\n")

    total = correct = valid = failed = 0
    per_route = {"REPLY": [0, 0], "REQUEST": [0, 0], "REACT": [0, 0]}

    for case_id, expected, task, memory, available_tools in CASES:
        state = f"USER REQUEST:\n{task}\n\nAVAILABLE TOOLS:\n{available_tools}\n\nRELEVANT MEMORY:\n{memory}"

        start_ms = time.perf_counter()
        result = run_case_routing(llm, state)
        latency_ms = int((time.perf_counter() - start_ms) * 1000)

        total += 1
        per_route[expected][1] += 1

        if result["valid"] == 0:
            failed += 1
            print(
                f"{case_id:<4} | INVALID | expected={expected:<7} actual={'NONE':<7} | {latency_ms:5d} ms"
            )
            print(result["log"], end="")
            continue

        valid += 1
        actual = result["route"]
        if actual == expected:
            correct += 1
            per_route[expected][0] += 1
            status = "PASS"
        else:
            status = "MISS"

        print(
            f"{case_id:<4} | {status:<7} | expected={expected:<7} actual={actual:<7} | {latency_ms:5d} ms"
        )
        votes = result["votes"]
        print(
            f"     votes     REPLY={votes['REPLY']}  REACT={votes['REACT']}  REQUEST={votes['REQUEST']}"
        )
        print(result["log"], end="")

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    if total:
        print(f"End-to-end accuracy: {100 * correct / total:.1f}% ({correct}/{total})")
        print(f"Valid-response rate:  {100 * valid / total:.1f}% ({valid}/{total})")
    if valid > 0:
        print(
            f"Conditional accuracy:  {100 * correct / valid:.1f}% ({correct}/{valid} valid cases)"
        )

    for route in ("REPLY", "REQUEST", "REACT"):
        hits, cases = per_route[route]
        if cases > 0:
            print(f"{route:<7} accuracy: {100 * hits / cases:.1f}% ({hits}/{cases})")

    print(f"\nCases: {total} | Valid: {valid} | Invalid: {failed}")
