import re
import json

from enum import Enum
from dataclasses import dataclass

from typing import Any


# REACT MARKERS & PARSED ACTION
class ReactMarker(str, Enum):
    """Standardized ReAct markers."""

    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    ACTION_INPUT = "ACTION_INPUT"
    OBSERVATION = "OBSERVATION"
    FINAL_ANSWER = "FINAL_ANSWER"

    @property
    def open_tag(self) -> str:
        return f"<{self.value}>"

    @property
    def close_tag(self) -> str:
        return f"</{self.value}>"

    @classmethod
    def extract(cls, text: str, marker: "ReactMarker") -> str | None:
        """Extract content between markers."""
        pattern = (
            rf"{re.escape(marker.open_tag)}" rf"(.*?)" rf"{re.escape(marker.close_tag)}"
        )
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @classmethod
    def has_marker(cls, text: str, marker: "ReactMarker") -> bool:
        """Check if marker exists in text."""
        return marker.open_tag in text and marker.close_tag in text


@dataclass
class ParsedAction:
    """Type-safe representation of a parsed ReAct action."""

    thought: str
    action: str
    action_input: dict[str, Any]
    raw_text: str

    @classmethod
    def from_text(cls, text: str) -> "ParsedAction | None":
        """Parse ReAct response into structured action."""
        thought = ReactMarker.extract(text, ReactMarker.THOUGHT)
        if thought is None:
            return None

        action = ReactMarker.extract(text, ReactMarker.ACTION)
        if action is None:
            return None

        action_input_raw = ReactMarker.extract(text, ReactMarker.ACTION_INPUT)
        if action_input_raw is None:
            return None

        try:
            action_input = cls._parse_json(action_input_raw)
        except json.JSONDecodeError:
            return None

        return cls(
            thought=thought,
            action=action.strip(),
            action_input=action_input,
            raw_text=text,
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        value = json.loads(text.strip())

        if not isinstance(value, dict):
            raise json.JSONDecodeError("ACTION_INPUT must be a JSON object", text, 0)

        return value

    @property
    def is_final_answer(self) -> bool:
        """Check if this is a final answer action."""
        return self.action.lower() in {"final_answer", "final answer"}
