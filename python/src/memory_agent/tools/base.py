from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Tool:
    """Simple callable tool abstraction."""

    name: str
    description: str
    fn: Callable[..., str]

    def execute(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)
