"""
Graph representation utilities for LLM consumption.

Design decisions:
- All methods are static for easy to use without state.
- Returns strings, not files (caller handles I/O)
- Configurable depth limits to avoid token overflow
- Supports multiple output formats optimized for LLM comsumption
"""

from typing import Any

from memory_graph import EdgeType, MemoryGraph


class GraphRepresentation:
    """
    Convert MemoryGraph to LLM-readable formats.

    Supported formats:
    - "linearized": Compact, human-readable text (default)
    - "triples": Subject -> Predicate -> Object
    - "json": Full structured JSON
    - "cypher": Cypher-like query format
    - "narrative": Natural language description
    """

    # Public API
    @staticmethod
    def to_llm_context(
        graph: MemoryGraph,
        format: str = "linearized",
        max_nodes: int = 50,
        max_edges: int = 100,
        include_metadata: bool = True,
    ) -> str:
        """
        Unified entry point for all formats.

        Args:
            graph: The MemoryGraph to represent
            format: "linearized", "triples", "json", "cypher", "narrative"
            max_nodes: Maximum nodes to include
            max_edges: Maximum edges to include
            include_metadata: Include metadata in output

        Returns:
            String representation of the graph
        """
        if format == "linearized":
            return GraphRepresentation.to_linearized_text(
                graph, max_nodes, include_metadata
            )
        elif format == "triples":
            return GraphRepresentation.to_triples_text(graph, max_edges)
        elif format == "json":
            return GraphRepresentation.to_json_string(graph, include_metadata)
        elif format == "cypher":
            return GraphRepresentation.to_cypher_like(graph)
        elif format == "narrative":
            return GraphRepresentation.to_narrative(graph, max_nodes)
        else:
            raise ValueError(
                f"[GraphRepresentation:to_llm_context] Unsupported format: {format}"
            )

    # TODO: [PERSPECTIVE] instead of using "max_nodes" we can select the nodes with highest confidence score.
    @staticmethod
    def to_linearized_text(
        graph: MemoryGraph,
        max_nodes: int = 50,
        max_edges: int = 100,
        include_metadata: bool = True,
    ) -> str:
        """
        Convert graph to compact, linearized text for LLM comsumption.

        This format is optimized for:
        - Token efficiency (compact)
        - LLM comprehension (clear structure)
        - Context window (fits more information)

        Example:
            Nodes:
                [geralt] -> label: Geralt of Rivia, type: witcher
                [vesemir] -> label: Vesemir, type: witcher
            Relations:
                (geralt) -[fellow_witchers]- (vesemir)
                (geralt) -[loves]- (yennefer)
        """
        lines = []
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        # limit nodes
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
            if include_metadata and meta:
                meta_str = ", ".join(
                    f"{k}: {v}" for k, v in meta.items() if k != "type"
                )
                if meta.get("type"):
                    parts.append(f"(type: {meta['type']})")
                if meta_str:
                    parts.append(f"{{{meta_str}}}")

            lines.append("  " + " ".join(parts))

        # Relations section
        if edges:
            lines.append("\n## Relations")
            for edge in edges[:max_edges]:
                conn_text = GraphRepresentation._connections_to_text(
                    edge.get_connections()
                )
                label = edge.get_label()

                if edge.get_type() == EdgeType.SYMMETRIC:
                    lines.append(f"  {conn_text} -[{label}]- (bidirectional)")
                else:
                    lines.append(f"  {conn_text} -> [{label}]->")

        return "\n".join(lines)

    # TODO: [PERSPECTIVE] we can also use edge weights to limit the number of edges here.
    @staticmethod
    def to_triples_text(graph: MemoryGraph, max_edges: int = 100) -> str:
        """
        Convert graph to triple format: (subject) -[predicate]-> (object)

        This is the most structured and LLM-friendly format for reasoning.
        Example:
            (geralt) -[fellow_witchers] -> (vesemir)
            (geralt) -[loves]-> (yennefer)
        """
        triples = GraphRepresentation._to_triples_list(graph)
        if len(triples) > max_edges:
            triples = triples[:max_edges]
            lines = [f"[Note: Showing {max_edges} of {len(triples)} relations"]
        else:
            lines = []

        for t in triples:
            lines.append(f"({t['subject']}) -[{t['predicate']}]-> ({t['object']})")

        return "\n".join(lines)

    @staticmethod
    def _to_triples_list(graph: MemoryGraph) -> list[dict[str, str]]:
        """Convert graph to list of triples."""
        triples = []

        for edge in graph.get_edges():
            conn = edge.get_connections()
            label = edge.get_label()

            if edge.get_type() == EdgeType.SYMMETRIC:
                nodes = list(conn)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        triples.append(
                            {
                                "subject": nodes[i],
                                "predicate": label,
                                "object": nodes[j],
                            }
                        )

            else:
                source, target = conn
                triples.append(
                    {"subject": source, "predicate": label, "object": target}
                )

        return triples

    @staticmethod
    def to_json_string(graph: MemoryGraph, include_metadata: bool = True) -> str:
        """Convert graph to pretty JSON string."""
        import json

        data = GraphRepresentation.to_json_string(graph, include_metadata)
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

    @staticmethod
    def to_cypher_like(graph: MemoryGraph, node_id: str | None = None) -> str:
        """Generate Cypher-like query format."""
        if node_id:
            return GraphRepresentation._subgraph_cypher(graph, node_id)
        return GraphRepresentation._full_graph_cypher(graph)

    @staticmethod
    def to_narrative(graph: MemoryGraph, max_nodes: int = 20) -> str:
        """
        Convert graph to natural language narrative.

        This is useful for LLMs that perform better with narrative text.

        Example:
            "Geralt of Rivia is connected to Vesemir as fellow_witchers.
            "Geralt loves Yennefer."
        """
        sentences = []

        # Get nodes with labels
        node_labels = {}
        for node in graph.get_nodes()[:max_nodes]:
            node_labels[node.get_id()] = node.get_label()

        # Build sentences from edges
        for edge in graph.get_edges():
            conn = edge.get_connections()
            label = edge.get_label()

            if edge.get_type() == EdgeType.SYMMETRIC:
                nodes = list(conn)
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        s1 = node_labels.get(nodes[i], nodes[i])
                        s2 = node_labels.get(nodes[j], nodes[j])
                        sentences.append(f"{s1} is connected to {s2} as {label}")

            else:
                source, target = conn
                s1 = node_labels.get(source, target)
                s2 = node_labels.get(target, source)
                sentences.append(f"{s1} {label} {s2}.")

        return "\n".join(sentences) if sentences else "The graph is empty."

    @staticmethod
    def _serialize_connections(connections):
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

    # TODO: We can use the C++ traversal graph/subgraph methods here.
    @staticmethod
    def _subgraph_cypher(graph: MemoryGraph, node_id: str) -> str:
        """Generate Cypher-like representation for a subgraph."""
        lines = [f"// Subgraph centered on '{node_id}'"]
        lines.append(f"MATCH (n {{id: '{node_id}'}})")

        neighbors = graph.get_neighbors(node_id)
        for neighbor in neighbors:
            neighbor_id = neighbor.get_id()
            for edge in graph.get_edges():
                conn = edge.get_connections()
                if node_id in conn and neighbor_id in conn:
                    lines.append(
                        f"  (n)-[r:{edge.get_label()}]->(m {{id: '{neighbor_id}'}})"
                    )

        return "\n".join(lines)

    # TODO: What the heck is this??
    @staticmethod
    def _full_graph_cypher(graph: MemoryGraph) -> str:
        """Generate Cypher-like representation for full graph."""
        lines = ["// Full graph representation"]
        lines.append("MATCH (n)-[r]-(m)")
        lines.append("RETURN n, r, m")
        return "\n".join(lines)
