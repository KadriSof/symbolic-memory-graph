"""
Graph serialization and persistence.
"""

import json

from pathlib import Path
from typing import Union
from memory_graph import MemoryGraph


class GraphSerializer:
    """
    Save and load MemoryGraph to/from files.

    Supported formats:
    - JSON (.json)
    - Binary (.mgb) - future
    - Compressed (.mgz) - future
    """

    @staticmethod
    def save(
        graph: MemoryGraph, filepath: Union[str, Path], format: str = "json"
    ) -> None:
        """Save graph to file."""
        filepath = Path(filepath)

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(graph.to_json(), f, indent=2)

        else:
            raise ValueError(f"[GraphSerializer:save] Unsupported format: {format}")

    @staticmethod
    def load(filepath: Union[str, Path], format: str = "json") -> MemoryGraph:
        """Load graph from file."""
        filepath = Path(filepath)

        if format == "json":
            with open(filepath, "r") as f:
                data = json.load(f)
            return MemoryGraph.from_json(data)

        else:
            raise ValueError(f"[GraphSerializer:load] Unsupported format: {format}")

    @staticmethod
    def save_state(state: dict, filepath: Union[str, Path]) -> None:
        """Save agent state (conversation history, thoughts, tool calls)"""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @staticmethod
    def load_state(filepath: Union[str, Path]) -> dict:
        """Load agent state from file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            return json.load(f)
