"""
Graph Traversal Utilities for the Cogito Agent.

This module provides high-level, agent-friendly graph traversal operations
by wrapping the optimized C++ traversal functions.

Design decisions:
- Re-exports core C++ traversal functions for direct, zero-overhead access.
- Provides higher-level, agent-specific traversal helpers (e.g., confidence filtering).
- Ensures all returned data is easily consumable by Python/LLM pipelines.
"""

from typing import Any, Callable

from memory_graph.memory_graph_core import (
    MemoryGraph,
    Node,
    bfs as _cpp_bfs,
    dfs as _cpp_dfs,
    shortest_path as _cpp_shortest_path,
    find_all_paths as _cpp_find_all_paths,
    has_cycle as _cpp_has_cycle,
    topological_sort as _cpp_topological_sort,
    is_connected as _cpp_is_connected,
    subgraph as _cpp_subgraph,
    subgraph_by_predicate as _cpp_subgraph_by_predicate,
    find_nodes_by_label as _cpp_find_nodes_by_label,
    find_nodes_by_metadata as _cpp_find_nodes_by_metadata,
    get_context_window as _cpp_get_context_window,
)


# Direct C++ Function Re-exports (for zero-overhead)
def bfs(graph: MemoryGraph, start: str, max_depth: int = -1) -> list[str]:
    """Breadth-First Search traversal starting from a node."""
    return _cpp_bfs(graph, start, max_depth)


def dfs(graph: MemoryGraph, start: str, max_depth: int = -1) -> list[str]:
    """Depth-First Search traversal starting from a node."""
    return _cpp_dfs(graph, start, max_depth)


def shortest_path(graph: MemoryGraph, from_node: str, to_node: str) -> list[str]:
    """Find the shortest path between two nodes."""
    return _cpp_shortest_path(graph, from_node, to_node)


def find_all_paths(
    graph: MemoryGraph, from_node: str, to_node: str, max_depth: int = -1
) -> list[list[str]]:
    """Find all paths between two nodes."""
    return _cpp_find_all_paths(graph, from_node, to_node, max_depth)


def has_cycle(graph: MemoryGraph) -> bool:
    """Check if the graph contains a cycle."""
    return _cpp_has_cycle(graph)


def topological_sort(graph: MemoryGraph) -> list[str]:
    """Perform a topological sort on a directed acyclic graph."""
    return _cpp_topological_sort(graph)


def is_connected(graph: MemoryGraph, start: str) -> bool:
    """Check if graph is connected from a given start node."""
    return _cpp_is_connected(graph, start)


def subgraph(
    graph: MemoryGraph, center: str, radius: int, include_edges: bool = True
) -> MemoryGraph:
    """Extract a subgraph centered at a node within a radius."""
    return _cpp_subgraph(graph, center, radius, include_edges)


def subgraph_by_predicate(
    graph: MemoryGraph,
    predicate: Callable[[Node], bool],
    include_neighbors: bool = False,
) -> MemoryGraph:
    """Extract a subgraph containing all nodes matching a predicate."""
    return _cpp_subgraph_by_predicate(graph, predicate, include_neighbors)


def find_nodes_by_label(graph: MemoryGraph, label: str) -> list[str]:
    """Find node IDs by label."""
    return _cpp_find_nodes_by_label(graph, label)


def find_nodes_by_metadata(graph: MemoryGraph, key: str, value: Any) -> list[str]:
    """Find node IDs by metadata key-value pair."""
    return _cpp_find_nodes_by_metadata(graph, key, value)


def get_context_window(
    graph: MemoryGraph, center: str, max_tokens: int = 4096, min_relevance: float = 0.5
) -> dict[str, Any]:
    """Get a context window for an LLM."""
    return _cpp_get_context_window(graph, center, max_tokens, min_relevance)


# higher-Level Agent Helpers
def find_entities_by_type(graph: MemoryGraph, entity_type: str) -> list[Node]:
    """
    Find all nodes of a specific type.

    Args:
        graph: The MemoryGraph to search.
        entity_type: The type of entity to find (e.g., "witcher", "location").

    Returns:
        A list of Node objects matching the type.
    """
    node_ids = find_nodes_by_metadata(graph, "type", entity_type)
    return [graph.get_node(nid) for nid in node_ids]


def get_confidence_filtered_subgraph(
    graph: MemoryGraph,
    center: str,
    radius: int,
    min_confidence: float = 0.0,
    min_weight: float = 0.0,
) -> MemoryGraph:
    """
    Extract a sugbraph and filter it by node confidence and edge weight.

    This is highly useful for the Cogito Agent to ensure only high-quality,
    reliable knowledge is passed to the LLM.
    (for the eyes of FinOps/TokenOps only :p )

    Args:
        graph: The source MemoryGraph
        center: The center node ID for the subgraph
        radius: The maximum traversal depth
        min_confidence: Minimum confidence score for nodes (0.0 to 1.0)
        min_weight: Minimum weight score for edges (0.0 to 1.0)

    Returns:
        A new MemoryGraph containing only the filtered nodes and edges.
    """
    # 1. Get initial subgraph using optimized C++
    # (finally I am using the code that made me cry for weeks T-T)
    # (Why am I even adding those comments? Maybe I am just going crazy xD)
    sub = subgraph(graph, center, radius, include_edges=True)

    # 2. Filter nodes by confidence
    filtered_nodes = []
    valid_node_ids = set()
    for node in sub.get_nodes():
        metadata = node.get_metadata()
        confidence = metadata.get("confidence", 1.0) if metadata else 1.0

        if confidence >= min_confidence:
            filtered_nodes.append(node)
            valid_node_ids.add(node.get_id())

    # 3. Create new filtered graph
    filtered_graph = MemoryGraph(sub.get_metadata())
    for node in filtered_nodes:
        filtered_graph.add_node(node)

    # 4. Filter edges by weight and valid endpoints
    for edge in sub.get_edges():
        if edge.get_weight() < min_weight:
            continue

        conn = edge.get_connections()
        if edge.is_group_edge():
            # For group edges, all members must be valid!
            members = list(conn)
            if all(nid in valid_node_ids for nid in members):
                filtered_graph.add_group_edge(
                    edge.get_id(),
                    edge.get_label(),
                    set(members),
                    edge.get_weight(),
                    edge.get_metadata(),
                )
        else:
            source, target = conn
            if source in valid_node_ids and target in valid_node_ids:
                filtered_graph.add_edge(edge)

    return filtered_graph


def find_shortest_path_with_details(
    graph: MemoryGraph, from_node: str, to_node: str
) -> dict[str, Any] | None:
    """
    Find the shortest path and return detailed info about the nodes and edges.

    Args:
        graph: The MemoryGraph to search
        from_node: The starting node ID
        to_node: The target node ID

    Returns:
        A dictionary containing the path, nodes, and edges,
        or None if no path exists.
    """
    try:
        path_ids = shortest_path(graph, from_node, to_node)
    except RuntimeError:
        # C++ throws RuntimeError if no path is found
        return None

    if not path_ids or len(path_ids) < 2:
        return {"path": path_ids, "nodes": [], "edges": []}

    nodes = [graph.get_node(nid) for nid in path_ids]
    edges = []

    # Find edges between consecutive nodes in the path
    for i in range(len(path_ids) - 1):
        current_id = path_ids[i]
        next_id = path_ids[i + 1]

        for edge in graph.get_edges():
            conn = edge.get_connections()
            if edge.is_group_edge():
                if current_id in conn and next_id in conn:
                    edges.append(edge)
                    break

            else:
                source, target = conn
                if (source == current_id and target == next_id) or (
                    source == next_id and target == current_id
                ):
                    edges.append(edge)
                    break

    return {"path": path_ids, "nodes": nodes, "edges": edges}


def find_communities(
    graph: MemoryGraph, min_nodes: int = 3, max_communities: int = 5
) -> list[list[str]]:
    """
    Find communities/clusters in the graph using connected components.

    Simple community detection based on connected components
    with confidence-weighted clustering.

    Args:
        graph: The MemoryGraph to analyze
        min_nodes: Minimum nodes in a community
        max_communities: Maximum number of communities to return

    Returns:
        List of communities (node ID lists), sorted by size (largest first)

    Example:
        >>> communities = find_communities(graph)
        >>> # Returns: [["geralt", "yennefer", "ciri"], ["vesemir", "eskel"]] (cool! innit!)
    """
    nodes = graph.get_nodes()
    if not nodes:
        return []

    visited = set()
    communities = []

    for node in nodes:
        node_id = node.get_id()
        if node_id in visited:
            continue

        # BFS to find connected component
        component = bfs(graph=graph, start=node_id, max_depth=-1)

        # Add confidence scores
        scored_component = []
        for nid in component:
            if graph.has_node(nid):
                node = graph.get_node(nid)
                metadata = node.get_metadata()
                confidence = metadata.get("confidence", 1.0) if metadata else 1.0
                scored_component.append((nid, confidence))

        # Sort by confidence (highest first)
        scored_component.sort(key=lambda x: x[1], reverse=True)

        if len(scored_component) >= min_nodes:
            communities.append([nid for nid, _ in scored_component])
            visited.update([nid for nid, _ in scored_component])

    # Sort communities by size (largest first)
    communities.sort(key=len, reverse=True)

    # Limit number of communities
    return communities[:max_communities]
