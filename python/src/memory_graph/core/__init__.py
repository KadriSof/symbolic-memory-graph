"""
Core graph utilities for MemoryGraph.

Provides:
- Graph representation and serialization
- Traversal algorithms (re-exported from C++ core)
"""

from .representation import GraphRepresentation
from .serialization import GraphSerializer

# Re-export traversal functions from the C++ core module
# This provides a cleaner API: `from memory_graph.core import bfs`
from memory_graph.memory_graph_core import (
    bfs,
    dfs,
    shortest_path,
    find_all_paths,
    has_cycle,
    topological_sort,
    is_connected,
    subgraph,
    subgraph_by_predicate,
    find_nodes_by_label,
    find_nodes_by_metadata,
    get_context_window,
)

# Re-export core classes
from memory_graph.memory_graph_core import (
    Node,
    Edge,
    EdgeType,
    MemoryGraph,
)

__all__ = [
    # Python utilities
    "GraphRepresentation",
    "GraphSerializer",
    # Core classes (from C++)
    "Node",
    "Edge",
    "EdgeType",
    "MemoryGraph",
    # Traversal functions (from C++)
    "bfs",
    "dfs",
    "shortest_path",
    "find_all_paths",
    "has_cycle",
    "topological_sort",
    "is_connected",
    "subgraph",
    "subgraph_by_predicate",
    "find_nodes_by_label",
    "find_nodes_by_metadata",
    "get_context_window",
]
