import json

from dataclasses import dataclass
from pydantic import BaseModel, ValidationError

from typing import Callable, Any


@dataclass
class Tool:
    """Simple callable tool abstraction."""

    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]

    def schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    def validate_args(self, raw_args: dict) -> tuple[BaseModel | None, str | None]:
        try:
            return self.args_model(**raw_args), None
        except ValidationError as e:
            return None, f"ValidationError: {e.errors()}"

    def execute(self, raw_args: dict) -> Any:
        args, err = self.validate_args(raw_args)
        if err is not None:
            raise ValueError(f"[Tool:{self.name}] Error:\n----\n{err}\n----")
        if args is None:
            raise ValueError(f"[Tool:{self.name}] args is None after validation")

        return self.fn(**args.model_dump())

    def __repr__(self) -> str:
        """Return an LLM-readable description of the tool."""
        schema = self.schema()

        return (
            f"### Tool: {schema['name']}\n"
            f"Description: {schema['description']}\n"
            f"Input schema:\n{json.dumps(schema['input_schema'], indent=2, ensure_ascii=False)}"
        )
