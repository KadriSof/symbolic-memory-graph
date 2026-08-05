"""
JSON Parser and Sanitizer for LLM Structured Output.
FULLY GENERIC - handles ANY structure, ANY model, ANY nesting.
"""

import re
import json
import logging
from datetime import datetime
from typing import Any, Optional, Callable, Type, TypeVar
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class JSONExtractor:
    """Extracts JSON from LLM responses with optimized multi-strategy extraction."""

    @staticmethod
    def extract(text: str) -> str | None:
        """Extract JSON using multiple strategies with logging."""
        if not text or not text.strip():
            logger.debug("Empty response text")
            return None

        # Strategy 1: Raw JSON
        cleaned = text.strip()
        if JSONExtractor._is_valid(cleaned):
            logger.debug("Extracted raw JSON")
            return cleaned

        # Strategy 2: Code blocks & Markdown fences
        patterns = [
            r"```(?:json|JSON)?\s*\n?(.*?)\n?```",
            r"`\s*(\{.*?\}|\[.*?\])\s*`",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                candidate = match.group(1).strip()
                if JSONExtractor._is_valid(candidate):
                    return candidate
                repaired = JSONRepair.repair_and_validate(candidate)
                if repaired:
                    return repaired

        # Strategy 3: Balanced structures
        candidate = JSONExtractor._extract_balanced(text)
        if candidate and JSONExtractor._is_valid(candidate):
            logger.debug("Extracted balanced structure")
            return candidate

        # Strategy 4: Key-value fallback
        candidate = JSONExtractor._extract_key_value_pairs(text)
        if candidate:
            logger.debug("Extracted key-value pairs")
            return candidate

        logger.warning("No valid JSON found in response")
        return None

    @staticmethod
    def _is_valid(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def _extract_balanced(text: str) -> str | None:
        stack = []
        start_idx = -1
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char in "{[":
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char in "}]":
                if stack and (
                    (stack[-1] == "{" and char == "}")
                    or (stack[-1] == "[" and char == "]")
                ):
                    stack.pop()
                    if not stack:
                        candidate = text[start_idx : i + 1]
                        if JSONExtractor._is_valid(candidate):
                            return candidate
                else:
                    stack = []
                    start_idx = -1
        return None

    @staticmethod
    def _extract_key_value_pairs(text: str) -> str | None:
        result = {}
        pattern = re.compile(
            r'["\']?([^"\'=:]+)["\']?\s*[:=]\s*["\']?([^"\'\n]+)["\']?'
        )

        for match in pattern.finditer(text):
            k, v = match.group(1).strip(), match.group(2).strip()
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            elif v.lower() == "null":
                v = None
            elif v.isdigit():
                v = int(v)
            elif re.match(r"^\d+\.\d+$", v):
                v = float(v)
            result[k] = v

        return json.dumps(result) if result else None


class JSONRepair:
    """Repairs common JSON formatting issues with robust handling."""

    @staticmethod
    def repair(text: str) -> str:
        if not text:
            return text

        repaired = text
        repaired = re.sub(r"^\s*```(?:json|JSON)?\s*\n?", "", repaired)
        repaired = re.sub(r"\n?\s*```\s*$", "", repaired)
        repaired = re.sub(
            r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', repaired
        )
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', repaired)

        def _replace_python_literals(m):
            val = m.group(1)
            return ":null" if val == "None" else f":{val.lower()}"

        repaired = re.sub(
            r":\s*(True|False|None)\b", _replace_python_literals, repaired
        )
        repaired = re.sub(
            r":\s*(?!true\b|false\b|null\b)([a-zA-Z_][a-zA-Z0-9_\s]*?)(?=\s*[,}\]])",
            r':"\1"',
            repaired,
        )

        try:
            repaired.encode("utf-8").decode("utf-8")
        except UnicodeDecodeError:
            repaired = repaired.encode("utf-8", errors="replace").decode("utf-8")

        return repaired

    @staticmethod
    def repair_and_validate(text: str) -> str | None:
        if not text:
            return None
        repaired = JSONRepair.repair(text)
        return repaired if JSONExtractor._is_valid(repaired) else None


class JSONSanitizer:
    """Generic JSON sanitizer - NO project-specific logic."""

    def __init__(
        self,
        type_conversions: dict[str, Callable[[Any], Any]] | None = None,
        default_values: dict[str, Any] | None = None,
        remove_nulls: bool = True,
        raise_on_missing: bool = False,
        max_depth: int = 10,
    ):
        self.type_conversions = type_conversions or {}
        self.default_values = default_values or {}
        self.remove_nulls = remove_nulls
        self.raise_on_missing = raise_on_missing
        self.max_depth = max_depth

    def sanitize(self, data: Any, depth: int = 0) -> Any:
        if depth > self.max_depth:
            logger.warning(f"Max recursion depth ({self.max_depth}) exceeded")
            return data

        if isinstance(data, dict):
            return self._sanitize_dict(data, depth)
        elif isinstance(data, list):
            return [self.sanitize(item, depth + 1) for item in data]
        else:
            return self._sanitize_value(data)

    def _sanitize_dict(self, data: dict[str, Any], depth: int) -> dict[str, Any]:
        out = {}

        for key, value in data.items():
            if value is None:
                if self.raise_on_missing and key not in self.default_values:
                    raise ValueError(f"Missing required value for '{key}'")
                if self.remove_nulls:
                    continue
                value = self.default_values.get(key)

            if isinstance(value, (dict, list)):
                out[key] = self.sanitize(value, depth + 1)
            else:
                out[key] = self._sanitize_field(key, value)

        for key, value in self.default_values.items():
            if key not in out:
                out[key] = value

        return out

    def _sanitize_field(self, key: str, value: Any) -> Any:
        if key in self.type_conversions:
            try:
                value = self.type_conversions[key](value)
            except Exception as e:
                logger.warning(f"Failed to convert field '{key}': {e}")
                if key in self.default_values:
                    return self.default_values[key]
                raise
        return self._sanitize_value(value)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value.encode("utf-8", errors="replace").decode("utf-8")
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)


class StructuredOutputParser:
    """
    FULLY GENERIC parser - handles ANY structure, ANY model, ANY nesting.

    Automatically extracts nested data that matches the schema.
    """

    def __init__(
        self,
        sanitizer: JSONSanitizer | None = None,
        fallback_on_error: bool = True,
        log_errors: bool = True,
    ):
        self.sanitizer = sanitizer or JSONSanitizer()
        self.fallback_on_error = fallback_on_error
        self.log_errors = log_errors

    def parse(
        self,
        response: str,
        schema: Type[T],
        context: str = "",
    ) -> T | None:
        """
        Parse LLM response into ANY model.

        Automatically handles:
        - Nested responses (extracts data that matches schema)
        - Arrays (extracts first matching item)
        - Flat responses (direct validation)
        """
        if not response:
            self._log_error("Empty response", context)
            if not self.fallback_on_error:
                raise ValueError("Empty response")
            return None

        try:
            # Extract JSON
            json_str = JSONExtractor.extract(response)
            if not json_str:
                json_str = JSONRepair.repair_and_validate(response)

            if not json_str:
                self._log_error("No JSON found or repaired in response", context)
                if not self.fallback_on_error:
                    raise ValueError("No JSON found in response")
                return None

            # Parse to dict
            data = json.loads(json_str)
            return self._parse_data(data, schema, context)

        except json.JSONDecodeError as e:
            self._log_error(f"JSON decode error: {e}", context)
            if not self.fallback_on_error:
                raise ValueError(f"JSON decode error: {e}") from e
            return None
        except Exception as e:
            self._log_error(f"Unexpected error: {e}", context)
            if not self.fallback_on_error:
                raise
            return None

    def parse_dict(
        self,
        data: dict[str, Any],
        schema: Type[T],
        context: str = "",
    ) -> T | None:
        """Parse already-parsed dict with automatic extraction."""
        return self._parse_data(data, schema, context)

    def _parse_data(
        self,
        data: Any,
        schema: Type[T],
        context: str = "",
    ) -> T | None:
        """
        Core parsing logic - tries multiple strategies to match ANY structure.
        """
        # Strategy 1: Try to validate directly
        result = self._try_validate(data, schema)
        if result is not None:
            return result

        # Strategy 2: If data is dict, try to find nested matches
        if isinstance(data, dict):
            for key, value in data.items():
                result = self._try_validate(value, schema)
                if result is not None:
                    logger.debug(f"Found match in '{key}'")
                    return result

                if isinstance(value, list):
                    for item in value:
                        result = self._try_validate(item, schema)
                        if result is not None:
                            logger.debug(f"Found match in '{key}[...]'")
                            return result

        # Strategy 3: If data is list, try each item
        if isinstance(data, list):
            for item in data:
                result = self._try_validate(item, schema)
                if result is not None:
                    logger.debug("Found match in list item")
                    return result

        # Strategy 4: Try to find ANY object that has all required fields
        result = self._find_by_required_fields(data, schema)
        if result is not None:
            return result

        # All strategies failed
        error_msg = "No data matches schema structure"
        self._log_error(error_msg, context)

        # Raise exception if fallback is disabled
        if not self.fallback_on_error:
            raise ValueError(error_msg)

        return None

    def _try_validate(self, data: Any, schema: Type[T]) -> T | None:
        """Try to validate data against schema."""
        if not isinstance(data, dict):
            return None

        try:
            # Sanitize first
            sanitized = self.sanitizer.sanitize(data)

            # Validate (Pydantic v1/v2 compatible)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(sanitized)
            elif hasattr(schema, "parse_obj"):
                return schema.parse_obj(sanitized)
            elif isinstance(sanitized, dict):
                return schema(**sanitized)  # type: ignore
            else:
                return sanitized  # type: ignore

        except (ValidationError, TypeError, ValueError):
            return None
        except Exception:
            return None

    def _find_by_required_fields(self, data: Any, schema: Type[T]) -> T | None:
        """
        Find any dict in the data that has all required fields of the schema.
        """
        if not isinstance(data, dict):
            return None

        # Get required field names from schema
        required_fields = self._get_required_fields(schema)
        if not required_fields:
            return None

        # Recursively search for dict with all required fields
        return self._search_required_fields(data, required_fields, schema)

    def _search_required_fields(
        self,
        data: Any,
        required_fields: set[str],
        schema: Type[T],
    ) -> T | None:
        """Recursively search for required fields."""
        if isinstance(data, dict):
            # Check if this dict has all required fields
            if required_fields.issubset(data.keys()):
                result = self._try_validate(data, schema)
                if result is not None:
                    return result

            # Search nested
            for value in data.values():
                result = self._search_required_fields(value, required_fields, schema)
                if result is not None:
                    return result

        elif isinstance(data, list):
            for item in data:
                result = self._search_required_fields(item, required_fields, schema)
                if result is not None:
                    return result

        return None

    def _get_required_fields(self, schema: Type[T]) -> set[str]:
        """Get required field names from a Pydantic model."""
        try:
            if hasattr(schema, "model_fields"):
                # Pydantic v2
                return {
                    name
                    for name, field in schema.model_fields.items()
                    if field.is_required()
                }
            elif hasattr(schema, "__fields__"):
                # Pydantic v1
                return {
                    name for name, field in schema.__fields__.items() if field.required  # type: ignore
                }
            else:
                # Fallback: use annotations
                return set(getattr(schema, "__annotations__", {}).keys())
        except Exception:
            return set()

    def parse_or_fallback(
        self,
        response: str,
        schema: Type[T],
        fallback_factory: Callable[[], T],
        context: str = "",
    ) -> T:
        """Parse or return fallback on failure."""
        result = self.parse(response, schema, context)
        if result is None:
            return fallback_factory()
        return result

    def _log_error(self, message: str, context: str) -> None:
        if self.log_errors:
            if context:
                logger.error(f"[{context}] {message}")
            else:
                logger.error(message)


# Convenience Functions
def parse_structured_output(
    response: str,
    schema: Type[T],
    fallback: T | None = None,
    sanitizer: JSONSanitizer | None = None,
    context: str = "",
) -> T | None:
    parser = StructuredOutputParser(sanitizer=sanitizer)
    result = parser.parse(response, schema, context)
    return result if result is not None else fallback


def parse_structured_output_factory(
    response: str,
    schema: Type[T],
    fallback_factory: Callable[[], T],
    sanitizer: JSONSanitizer | None = None,
    context: str = "",
) -> T:
    parser = StructuredOutputParser(sanitizer=sanitizer)
    return parser.parse_or_fallback(response, schema, fallback_factory, context)


__all__ = [
    "JSONExtractor",
    "JSONRepair",
    "JSONSanitizer",
    "StructuredOutputParser",
    "parse_structured_output",
    "parse_structured_output_factory",
]
