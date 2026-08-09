"""
Graph representation utilities for LLM consumption.

Design decisions:
- All methods are static for easy to use without state.
- Returns strings, not files (caller handles I/O)
- Configurable depth limits to avoid token overflow
- Supports multiple output formats optimized for LLM consumption
- Confidence-based node selection
- Token-aware truncation
- Relevance scoring for query-aware context
- Uses C++ traversal methods for performance
"""

import json

from typing import Any

from memory_graph.memory_graph_core import (
    MemoryGraph,
    Node,
    Edge,
    EdgeType,
    subgraph,
    get_context_window,
)


class GraphRepresentation:
    """
    Convert MemoryGraph to LLM-readable formats.

    Supported formats:
    - "linearized": Compact, human-readable text (default)
    - "triples": Subject -> Predicate -> Object
    - "json": Full structured JSON
    - "cypher": Cypher-like query format
    - "narrative": Natural language description
    - "relevance": Query-relevance ranked context
    - "hierarchical": Tree/grouped structure
    """

    # Public API - Unified Entry Point
    @staticmethod
    def to_llm_context(
        graph: MemoryGraph,
        format: str = "linearized",
        max_nodes: int = 50,
        max_edges: int = 100,
        include_metadata: bool = True,
        min_confidence: float = 0.0,
        min_weight: float = 0.0,
        query: str | None = None,
        center_node: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Unified entry point for all formats.

        Args:
            graph: The MemoryGraph to represent
            format: "linearized", "triples", "json", "cypher", "narrative",
                    "relevance", "hierarchical", "subgraph"
            max_nodes: Maximum nodes to include
            max_edges: Maximum edges to include
            min_confidence: Minimum confidence score for nodes (0.0-1.0)
            min_weight: Minimum weight for edges (0.0-1.0)
            include_metadata: Include metadata in output
            query: Query string for relevance scoring
            center_node: Center node ID for subgraph extraction
            max_tokens: Maximum token limit (uses estimation)

        Returns:
            String representation of the graph
        """
        # Handle subgraph format first (uses C++ traversal)
        if format == "subgraph":
            if not center_node:
                raise ValueError("format='subgraph' requires a 'center_node' argument")
            return GraphRepresentation.to_subgraph_context(
                graph, center_node, radius=2, format="linearized"
            )

        # Handle relevance format
        if format == "relevance" and query:
            return GraphRepresentation.to_relevance_ranked_context(
                graph, query, max_tokens or 4096
            )

        # Handle hierarchical format
        if format == "hierarchical":
            return GraphRepresentation.to_hierarchical_text(graph, max_nodes)

        # Apply confidence and weight filtering
        filtered_graph = graph
        if min_confidence > 0.0 or min_weight > 0.0:
            filtered_graph = GraphRepresentation._filter_graph(
                graph, min_confidence, min_weight
            )

        # Generate base representation
        if format == "linearized":
            result = GraphRepresentation.to_linearized_text(
                filtered_graph, max_nodes, max_edges, include_metadata
            )
        elif format == "triples":
            result = GraphRepresentation.to_triples_text(
                filtered_graph, max_edges, min_weight
            )
        elif format == "json":
            result = GraphRepresentation.to_json_string(
                filtered_graph, include_metadata
            )
        elif format == "cypher":
            result = GraphRepresentation.to_cypher_like(filtered_graph)
        elif format == "narrative":
            result = GraphRepresentation.to_narrative(filtered_graph, max_nodes)
        else:
            raise ValueError(
                f"[GraphRepresentation:to_llm_context] Unsupported format: {format}"  # ✅ FIXED: Added closing brace
            )

        # Token-aware truncation
        if max_tokens:
            result = GraphRepresentation._truncate_to_tokens(result, max_tokens)

        return result

    # Format: Linearized Text (Compact HUD Mode)
    @staticmethod
    def to_linearized_text(
        graph: MemoryGraph,
        max_nodes: int = 50,
        max_edges: int = 100,
        include_metadata: bool = True,
    ) -> str:
        """
        Convert graph to compact, linearized text for LLM consumption.

        This format is optimized for:
        - Token efficiency (compact)
        - LLM comprehension (clear structure)
        - Context window (fits more information)

        Example:
            Nodes:
                [geralt] -> Geralt of Rivia (type: witcher) {confidence: 0.95}
                [vesemir] -> Vesemir (type: witcher) {confidence: 0.90}
            Relations:
                (geralt) -[fellow_witchers]- (vesemir) {weight: 0.8}
                (geralt) -[loves]-> (yennefer) {weight: 0.95}
        """
        lines = []
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        # Sort nodes by confidence (highest first) if available
        nodes = GraphRepresentation._sort_nodes_by_confidence(nodes)

        # Limit nodes
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            lines.append(
                f"[Note: Showing {max_nodes} of {len(graph.get_nodes())} nodes]"
            )

        # Node section
        lines.append("## Nodes")
        for node in nodes:
            label = node.get_label()
            node_id = node.get_id()
            meta = node.get_metadata() if include_metadata else {}

            # Build compact node description
            parts = [f"[{node_id} -> {label}]"]

            if include_metadata:
                # Show type
                if meta.get("type"):
                    parts.append(f"(type: {meta['type']})")

                # Show confidence
                if meta.get("confidence") is not None:
                    conf = meta["confidence"]
                    conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
                    parts.append(f"{{confidence: {conf_str}}}")

                # Show other important metadata
                other_meta = {
                    k: v for k, v in meta.items() if k not in ["type", "confidence"]
                }
                if other_meta:
                    meta_str = ", ".join(f"{k}: {v}" for k, v in other_meta.items())
                    if len(meta_str) > 100:
                        meta_str = meta_str[:97] + "..."
                    parts.append(f"{{{meta_str}}}")

            lines.append("  " + " ".join(parts))

        # Relations section
        if edges:
            lines.append("\n## Relations")
            # Sort edges by weight (highest first)
            sorted_edges = GraphRepresentation._sort_edges_by_weight(edges)

            for edge in sorted_edges[:max_edges]:
                conn_text = GraphRepresentation._connections_to_text(
                    edge.get_connections()
                )
                label = edge.get_label()
                weight = edge.get_weight()

                if edge.get_type() == EdgeType.SYMMETRIC:
                    lines.append(
                        f"  {conn_text} -[{label}]- (bidirectional) {{weight: {weight:.2f}}}"
                    )
                else:
                    lines.append(
                        f"  {conn_text} -> [{label}]-> {{weight: {weight:.2f}}}"
                    )

        return "\n".join(lines)

    # Format: Triples (Tactical Grid Overlay)
    @staticmethod
    def to_triples_text(
        graph: MemoryGraph, max_edges: int = 100, min_weight: float = 0.0
    ) -> str:
        """
        Convert graph to triple format: (subject) -[predicate]-> (object)

        This is the most structured and LLM-friendly format for reasoning.

        Example:
            (geralt) -[fellow_witchers]-> (vesemir) {weight: 0.8}
            (geralt) -[loves]-> (yennefer) {weight: 0.95}
        """
        triples = GraphRepresentation._to_triples_list(graph, min_weight)

        if len(triples) > max_edges:
            triples = triples[:max_edges]
            lines = [f"[Note: Showing {max_edges} of {len(triples)} relations]"]
        else:
            lines = []

        for t in triples:
            if t.get("weight", 1.0) < 1.0:
                lines.append(
                    f"({t['subject']}) -[{t['predicate']}]-> ({t['object']}) "
                    f"{{weight: {t['weight']:.2f}}}"
                )
            else:
                lines.append(f"({t['subject']}) -[{t['predicate']}]-> ({t['object']})")

        return "\n".join(lines)

    @staticmethod
    def _to_triples_list(
        graph: MemoryGraph, min_weight: float = 0.0
    ) -> list[dict[str, Any]]:
        """Convert graph to list of triples with weights."""
        triples = []

        for edge in graph.get_edges():
            if edge.get_weight() < min_weight:
                continue

            conn = edge.get_connections()
            label = edge.get_label()
            weight = edge.get_weight()

            # Handle group edges specially
            if edge.is_group_edge():
                nodes = list(conn)
                # Show up to 10 members, summarize the rest
                display_nodes = nodes[:10]
                suffix = f" (+{len(nodes) - 10} more)" if len(nodes) > 10 else ""
                members_str = ", ".join(display_nodes) + suffix

                triples.append(
                    {
                        "subject": "Group",
                        "predicate": f"{label} (members: {members_str})",
                        "object": "N/A",
                        "weight": weight,
                    }
                )
                continue  # ✅ FIXED: Skip to next edge after group

            # Handle symmetric edges (non-group)
            if edge.get_type() == EdgeType.SYMMETRIC:
                nodes = list(conn)
                max_pairs = 5
                pairs_generated = 0
                pair_limit_reached = False

                for i in range(len(nodes)):
                    if pair_limit_reached:
                        break
                    for j in range(i + 1, len(nodes)):
                        if pairs_generated >= max_pairs:
                            pair_limit_reached = True
                            break

                        triples.append(
                            {
                                "subject": nodes[i],
                                "predicate": label,
                                "object": nodes[j],
                                "weight": weight,
                            }
                        )
                        pairs_generated += 1

            else:
                # Asymmetric edge
                source, target = conn
                triples.append(
                    {
                        "subject": source,
                        "predicate": label,
                        "object": target,
                        "weight": weight,
                    }
                )

        # Sort by weight (highest first)
        triples.sort(key=lambda t: t.get("weight", 0.0), reverse=True)
        return triples

    # Format: JSON (Raw Sensor Data Feed)
    @staticmethod
    def to_json_string(graph: MemoryGraph, include_metadata: bool = True) -> str:
        """Convert graph to pretty JSON string."""
        data = GraphRepresentation.to_json_dict(graph, include_metadata)
        return json.dumps(data, indent=2)

    @staticmethod
    def to_json_dict(
        graph: MemoryGraph, include_metadata: bool = True
    ) -> dict[str, Any]:
        """Convert graph to JSON dict."""
        result = {
            "nodes": [],
            "edges": [],
            "metadata": graph.get_metadata() if include_metadata else {},
        }

        for node in graph.get_nodes():
            node_data: dict[str, Any] = {"id": node.get_id(), "label": node.get_label()}
            if include_metadata:
                node_data["metadata"] = node.get_metadata()
            result["nodes"].append(node_data)

        for edge in graph.get_edges():
            edge_data = {
                "id": edge.get_id(),
                "label": edge.get_label(),
                "type": (
                    "symmetric"
                    if edge.get_type() == EdgeType.SYMMETRIC
                    else "asymmetric"
                ),
                "connections": GraphRepresentation._serialize_connections(
                    edge.get_connections()
                ),
                "weight": edge.get_weight(),
            }

            if include_metadata:
                edge_data["metadata"] = edge.get_metadata()
            result["edges"].append(edge_data)

        return result

    # Format: Cypher-like (Database Query Mode)
    @staticmethod
    def to_cypher_like(graph: MemoryGraph, node_id: str | None = None) -> str:
        """Generate Cypher-like query format."""
        if node_id:
            return GraphRepresentation._subgraph_cypher(graph, node_id)
        return GraphRepresentation._full_graph_cypher(graph)

    @staticmethod
    def _subgraph_cypher(graph: MemoryGraph, node_id: str) -> str:
        """Generate Cypher-like representation for a subgraph using C++ traversal."""
        lines = [f"// Subgraph centered on '{node_id}'"]
        lines.append("")

        sub = subgraph(graph, node_id, radius=2)

        # Node Section
        lines.append("// Nodes")
        for node in sub.get_nodes():
            label = node.get_label().replace(" ", "_")
            # ✅ FIXED: Safe metadata truncation (copy first)
            meta = node.get_metadata()
            if meta:
                meta_keys = list(meta.keys())[:3]
                meta = {k: meta[k] for k in meta_keys}
            meta_str = json.dumps(meta)

            lines.append(
                f"CREATE (:{label} {{id: '{node.get_id()}', label: '{node.get_label()}', metadata: {meta_str}}})"
            )

        # Edge Section
        lines.append("")
        lines.append("// Relationships")
        for edge in sub.get_edges():
            label = edge.get_label().replace(" ", "_")
            weight = edge.get_weight()

            if edge.get_type() == EdgeType.SYMMETRIC:
                nodes = list(edge.get_connections())
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        lines.append(
                            f"MATCH (a {{id: '{nodes[i]}'}}), (b {{id: '{nodes[j]}'}}) "
                            f"CREATE (a)-[:{label} {{weight: {weight:.2f}}}]->(b);"
                        )
            else:
                source, target = edge.get_connections()
                lines.append(
                    f"MATCH (a {{id: '{source}'}}), (b {{id: '{target}'}}) "
                    f"CREATE (a)-[:{label} {{weight: {weight:.2f}}}]->(b);"
                )

        return "\n".join(lines)

    @staticmethod
    def _full_graph_cypher(graph: MemoryGraph) -> str:
        """Generate proper Cypher-like representation for full graph."""
        lines = ["// Full Graph Representation"]
        lines.append("")
        lines.append("// === NODES ===")

        for node in graph.get_nodes():
            node_id = node.get_id()
            label = node.get_label().replace(" ", "_")
            meta = node.get_metadata()
            if meta:
                meta_keys = list(meta.keys())[:3]
                meta = {k: meta[k] for k in meta_keys}
            meta_str = json.dumps(meta)

            lines.append(
                f"CREATE (:{label}) {{"
                f"id: '{node_id}', "
                f"label: '{label}', "
                f"metadata: {meta_str}"
                f"}};"
            )

        lines.append("")
        lines.append("// === RELATIONSHIPS ===")

        for edge in graph.get_edges():
            label = edge.get_label().replace(" ", "_")
            weight = edge.get_weight()

            if edge.is_group_edge():
                nodes = list(edge.get_connections())
                display_nodes = nodes[:10]
                suffix = f" (+{len(nodes) - 10} more)" if len(nodes) > 10 else ""
                members_str = ", ".join(display_nodes) + suffix
                lines.append(
                    f"// Group Edge: {label} connects: {members_str} (weight: {weight:.2f})"
                )

            elif edge.get_type() == EdgeType.SYMMETRIC:
                nodes = list(edge.get_connections())
                max_pairs = 5
                pairs_generated = 0
                pair_limit_reached = False

                for i in range(len(nodes)):
                    if pair_limit_reached:
                        break
                    for j in range(i + 1, len(nodes)):
                        if pairs_generated >= max_pairs:
                            pair_limit_reached = True
                            break
                        lines.append(
                            f"MATCH (a {{id: '{nodes[i]}'}}), (b {{id: '{nodes[j]}'}}) "
                            f"CREATE (a)-[:{label} {{weight: {weight:.2f}}}]->(b);"
                        )
                        pairs_generated += 1

            else:
                source, target = edge.get_connections()
                lines.append(
                    f"MATCH (a {{id: '{source}'}}), (b {{id: '{target}'}}) "
                    f"CREATE (a)-[:{label} {{weight: {weight:.2f}}}]->(b);"
                )

        return "\n".join(lines)

    # Format: Narrative (Natural Language Briefing)
    @staticmethod
    def to_narrative(graph: MemoryGraph, max_nodes: int = 20) -> str:
        """
        Convert graph to natural language narrative.

        This is useful for LLMs that perform better with narrative text.

        Example:
            "Geralt of Rivia is a witcher (confidence: 0.95).
            Geralt of Rivia is connected to Vesemir as fellow_witchers (weight: 0.8).
            Geralt of Rivia loves Yennefer (weight: 0.95)."
        """
        sentences = []

        # Get nodes with labels and metadata
        node_info = {}
        nodes = graph.get_nodes()[:max_nodes]
        for node in nodes:
            node_id = node.get_id()
            label = node.get_label()
            meta = node.get_metadata()

            # Build node description
            desc = label
            if meta.get("type"):
                desc += f" (a {meta['type']})"
            if meta.get("confidence") is not None:
                conf = meta["confidence"]
                desc += f" (confidence: {conf:.2f})"
            node_info[node_id] = desc

        for edge in graph.get_edges():
            conn = edge.get_connections()
            label = edge.get_label()
            weight = edge.get_weight()

            weight_text = f" (weight: {weight:.2f})" if weight < 1.0 else ""

            if edge.is_group_edge():
                nodes = list(conn)
                display_nodes = nodes[:5]
                suffix = f" (+{len(nodes) - 5} more)" if len(nodes) > 5 else ""
                members_str = ", ".join(display_nodes) + suffix
                sentences.append(
                    f"The group '{label}' includes: {members_str}{weight_text}."
                )

            elif edge.get_type() == EdgeType.SYMMETRIC:
                nodes = list(conn)
                max_pairs = 3
                pairs_generated = 0
                pair_limit_reached = False

                for i in range(len(nodes)):
                    if pair_limit_reached:
                        break
                    for j in range(i + 1, len(nodes)):
                        if pairs_generated >= max_pairs:
                            pair_limit_reached = True
                            break
                        s1 = node_info.get(nodes[i], nodes[i])
                        s2 = node_info.get(nodes[j], nodes[j])
                        sentences.append(
                            f"{s1} is connected to {s2} as {label}{weight_text}."
                        )
                        pairs_generated += 1

            else:
                source, target = conn
                s1 = node_info.get(source, source)
                s2 = node_info.get(target, target)
                sentences.append(f"{s1} {label} {s2}{weight_text}.")

        return "\n".join(sentences) if sentences else "The graph is empty."

    # Format: Relevance-Ranked Context (Precision Targeting HUD)
    @staticmethod
    def to_relevance_ranked_context(
        graph: MemoryGraph,
        query: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Convert graph to context ranked by relevance to the query.

        This is the premium HUD mode - shows only what matters.
        Uses the C++ get_context_window when available.
        """
        # Try using C++ context window first
        try:
            # Find most relevant node using simple keyword matching
            nodes = graph.get_nodes()
            best_node = None
            best_score = 0.0

            query_words = set(query.lower().split())
            for node in nodes:
                label = node.get_label().lower()
                label_words = set(label.split())
                # Simple overlap score
                overlap = len(query_words & label_words)
                if overlap > best_score:
                    best_score = overlap
                    best_node = node.get_id()

            if best_node and best_score > 0:
                # Use C++ context window
                context = get_context_window(
                    graph, best_node, max_tokens=max_tokens, min_relevance=0.5
                )
                return json.dumps(context, indent=2)
        except Exception:
            pass  # Fall back to Python implementation

        # Fallback: Python implementation
        return GraphRepresentation._relevance_ranked_fallback(graph, query, max_tokens)

    @staticmethod
    def _relevance_ranked_fallback(
        graph: MemoryGraph,
        query: str,
        max_tokens: int = 4096,
    ) -> str:
        """Fallback relevance ranking implementation with token awareness."""
        query_words = set(query.lower().split())

        # Score nodes by relevance
        scored_nodes = []
        for node in graph.get_nodes():
            label = node.get_label().lower()
            label_words = set(label.split())

            # Simple relevance score
            word_overlap = len(query_words & label_words)
            if word_overlap == 0:
                continue

            # Weight by confidence if available
            confidence = node.get_metadata().get("confidence", 0.5)
            relevance = word_overlap / max(1, len(query_words)) * confidence

            scored_nodes.append((node, relevance))

        # Sort by relevance
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        if not scored_nodes:
            return "No relevant nodes found in graph."

        # Extract subgraph around top nodes
        top_nodes = scored_nodes[:5]
        top_node_ids = [n[0].get_id() for n in top_nodes]

        # Use C++ subgraph extraction
        try:
            sub = subgraph(graph, top_node_ids[0], radius=2)
        except Exception:
            sub = graph

        # Format with relevance scores
        lines = ["# Relevance-Ranked Context"]
        lines.append(f"Query: {query}")
        lines.append("")
        lines.append("## Top Relevant Nodes")

        for node, score in scored_nodes[:10]:
            label = node.get_label()
            node_id = node.get_id()
            confidence = node.get_metadata().get("confidence", 1.0)
            lines.append(
                f"  [{node_id}] {label} (relevance: {score:.2f}, confidence: {confidence:.2f})"
            )

        lines.append("")
        lines.append("## Graph Context (Top 20 Nodes)")

        # Add the subgraph representation
        context = GraphRepresentation.to_linearized_text(
            sub, max_nodes=20, include_metadata=True
        )

        context = GraphRepresentation._truncate_to_tokens(context, max_tokens)
        lines.append(context)

        return "\n".join(lines)

    # Format: Hierarchical (Tree/Grouped Structure)
    @staticmethod
    def to_hierarchical_text(
        graph: MemoryGraph,
        max_nodes: int = 50,
    ) -> str:
        """
        Convert graph to hierarchical/tree structure.

        Groups nodes by type or metadata for better organization.
        """
        lines = ["# Hierarchical Graph View"]
        lines.append("")

        # Group nodes by type
        nodes_by_type = {}
        for node in graph.get_nodes():
            node_type = node.get_metadata().get("type", "unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)

        # Sort types by count (largest first)
        sorted_types = sorted(
            nodes_by_type.items(), key=lambda x: len(x[1]), reverse=True
        )

        total_displayed = 0
        for node_type, nodes in sorted_types[:10]:
            remaining = max_nodes - total_displayed
            if remaining <= 0:
                lines.append(
                    f"## ... and {sum(len(n) for _, n in sorted_types[10:])} more nodes"
                )
                break

            display_count = min(len(nodes), remaining)
            lines.append(f"## {node_type.upper()} ({len(nodes)} nodes)")

            for node in nodes[:display_count]:
                label = node.get_label()
                node_id = node.get_id()
                confidence = node.get_metadata().get("confidence", 1.0)
                lines.append(
                    f"  ├── [{node_id}] {label} (confidence: {confidence:.2f})"
                )
                total_displayed += 1

            if len(nodes) > display_count:
                lines.append(f"  └── ... and {len(nodes) - display_count} more")
            lines.append("")

        # Show relationships
        lines.append("## Relationships")
        edges = graph.get_edges()
        for edge in edges[:20]:
            conn = edge.get_connections()
            label = edge.get_label()
            if edge.is_group_edge():
                nodes = list(conn)
                display_nodes = nodes[:5]
                suffix = f" (+{len(nodes) - 5} more)" if len(nodes) > 5 else ""
                lines.append(
                    f"  Group '{label}' connects: {', '.join(display_nodes)}{suffix}"
                )

            elif edge.get_type() == EdgeType.SYMMETRIC:
                nodes = list(conn)
                max_pairs = 3
                pairs_generated = 0
                pair_limit_reached = False

                for i in range(len(nodes)):
                    if pair_limit_reached:
                        break
                    for j in range(i + 1, len(nodes)):
                        if pairs_generated >= max_pairs:
                            pair_limit_reached = True
                            break
                        lines.append(f"  {nodes[i]} ↔ {nodes[j]} ({label})")
                        pairs_generated += 1

            else:
                source, target = conn
                lines.append(f"  {source} → {target} ({label})")

        return "\n".join(lines)

    # Format: Subgraph Context (Precision Targeting)
    @staticmethod
    def to_subgraph_context(
        graph: MemoryGraph,
        center_node_id: str,
        radius: int = 2,
        format: str = "linearized",
    ) -> str:
        """
        Generate context around a specific node using C++ subgraph.

        This is the precision targeting system.
        """
        try:
            # Use C++ subgraph extraction
            sub = subgraph(graph, center_node_id, radius)

            # Add header
            header = [
                f"# Subgraph centered on '{center_node_id}' (radius: {radius})",
                f"Nodes: {len(sub.get_nodes())}, Edges: {len(sub.get_edges())}",
                "",
            ]

            context = GraphRepresentation.to_llm_context(
                sub,
                format=format,
                include_metadata=True,
            )

            return "\n".join(header) + context

        except Exception as e:
            return f"Error extracting subgraph: {e}"

    # Utility Methods
    @staticmethod
    def _filter_graph(
        graph: MemoryGraph,
        min_confidence: float = 0.0,
        min_weight: float = 0.0,
    ) -> MemoryGraph:
        """Filter graph by confidence and weight thresholds."""
        filtered = MemoryGraph(graph.get_metadata())

        # Add nodes that meet confidence threshold (Deep Copy)
        for node in graph.get_nodes():
            confidence = node.get_metadata().get("confidence", 1.0)
            if confidence >= min_confidence:
                filtered.add_node(Node.from_json(node.to_json()))

        # Add edges that meet weight threshold (Deep Copy)
        for edge in graph.get_edges():
            if edge.get_weight() >= min_weight:
                conn = edge.get_connections()
                if edge.get_type() == EdgeType.SYMMETRIC:
                    nodes = list(conn)
                    all_exist = all(filtered.has_node(n) for n in nodes)
                    if all_exist:
                        filtered.add_edge(Edge.from_json(edge.to_json()))
                else:
                    source, target = conn
                    if filtered.has_node(source) and filtered.has_node(target):
                        filtered.add_edge(Edge.from_json(edge.to_json()))

        return filtered

    @staticmethod
    def _sort_nodes_by_confidence(nodes: list[Node]) -> list[Node]:
        """Sort nodes by confidence score (highest first)."""

        def get_confidence(node: Node) -> float:
            return node.get_metadata().get("confidence", 0.0)

        return sorted(nodes, key=get_confidence, reverse=True)

    @staticmethod
    def _sort_edges_by_weight(edges: list[Edge]) -> list[Edge]:
        """Sort edges by weight (highest first)."""
        return sorted(edges, key=lambda e: e.get_weight(), reverse=True)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit."""
        # Simple token estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Find a good truncation point
        truncated = text[:max_chars]

        # Try to cut at a newline or period
        for cut_char in ["\n\n", "\n", ". ", "  "]:
            last_cut = truncated.rfind(cut_char)
            if last_cut > max_chars * 0.7:
                truncated = truncated[: last_cut + len(cut_char)]
                break

        return truncated + "\n\n[Truncated due to token limit]"

    @staticmethod
    def _serialize_connections(connections) -> dict[str, Any]:
        """Serialize connections to JSON-friendly format."""
        if isinstance(connections, set):
            return {"type": "symmetric", "nodes": list(connections)}
        else:
            source, target = connections
            return {"type": "asymmetric", "source": source, "target": target}

    @staticmethod
    def _connections_to_text(connections) -> str:
        """Convert connections to text."""
        if isinstance(connections, set):
            return " <-> ".join(list(connections))
        else:
            source, target = connections
            return f"{source} -> {target}"
