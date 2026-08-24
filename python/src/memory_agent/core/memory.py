from datetime import datetime

from typing import Any

from memory_graph.core import MemoryGraph, Node, Edge, EdgeType


class SymbolicMemory:
    """
    Long-Term Memory (Knowledge Graph) for the Cogito Agent.

    Wraps the C++ MemoryGraph to provide semantic operations for:
    - Adding/updating concepts (nodes)
    - Creating relationships (edges)
    - Querying the knowledge graph
    - Managing confidence scores
    - Serialization/deserialization

    Example:
        >>> memory = SymbolicMemory()
        >>> memory.add_concept("geralt", "Geralt of Rivia", {"type": "witcher"})
        >>> memory.add_concept("yennefer", "Yennefer", {"type": "sorceress"})
        >>> memory.add_link("geralt", "yennefer", "loves", weight=0.95)
        >>> result = memory.query("geralt")
    """

    def __init__(self, graph: "MemoryGraph | None" = None) -> None:
        self.graph = graph or MemoryGraph({"type": "symbolic_memory"})

    # Concept operations
    def add_concept(
        self,
        concept_id: str,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a new concept to the knowledge graph.

        Args:
            concept_id: Unique identifier for the concept
            label: Human-readable label
            metadata: Additional metadata (type, confidence, etc.)

        Raises:
            ValueError: If concept_id already exists
        """
        if not concept_id or not label:
            raise ValueError("concept_id and label are required")

        if self.graph.has_node(concept_id):
            raise ValueError(
                f"Concept '{concept_id}' already exists. Use update_concept() to modify."
            )

        metadata = metadata or {}
        metadata.setdefault("created_at", datetime.now().isoformat())
        metadata.setdefault("last_updated", datetime.now().isoformat())
        metadata.setdefault("confidence", 1.0)

        node = Node(concept_id, label, metadata)
        self.graph.add_node(node)

    def get_concept(self, concept_id: str) -> Node | None:
        """
        Retrieve a concept by ID.

        Args:
            concept_id: Node ID to retrieve

        Returns:
            Node object or None if not found
        """
        if not self.graph.has_node(concept_id):
            return None
        return self.graph.get_node(concept_id)

    def update_concept(
        self,
        concept_id: str,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> None:
        """
        Update an existing concept.

        Args:
            concept_id: Node ID to update
            label: New label (None to keep existing)
            metadata: New metadata (merge_metadata controls behavior)
            merge_metadata: If True, merge with existing; if False, replace

        Raises:
            ValueError: If concept_id doesn't exist
        """
        if not self.graph.has_node(concept_id):
            raise ValueError(f"Concept '{concept_id}' does not exist")

        node = self.graph.get_node(concept_id)

        if label is not None:
            node.set_label(label)

        if metadata is not None:
            if merge_metadata:
                # Merge: update only provided keys
                existing = node.get_metadata()
                existing.update(metadata)
                existing["last_updated"] = datetime.now().isoformat()
                node.set_metadata(existing)
            else:
                # Replace: use new metadata entirely
                metadata["last_updated"] = datetime.now().isoformat()
                node.set_metadata(metadata)

    def delete_concept(self, concept_id: str) -> bool:
        """
        Delete a concept and all its relationships.

        Args:
            concept_id: Node ID to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.graph.has_node(concept_id):
            return False

        self.graph.remove_node(concept_id)
        return True

    def find_concepts_by_type(self, concept_type: str) -> list[Node]:
        """
        Find all concepts of a specific type.

        Args:
            concept_type: Type string (e.g., "witcher", "location")

        Returns:
            List of Node objects matching the type
        """
        from memory_graph.core.traversal import find_nodes_by_metadata

        node_ids = find_nodes_by_metadata(self.graph, "type", concept_type)
        return [self.graph.get_node(nid) for nid in node_ids]

    def find_concepts_by_confidence(self, min_confidence: float = 0.0) -> list[Node]:
        """
        Find concepts with confidence above threshold.

        Args:
            min_confidence: Minimum confidence score (0.0 to 1.0)

        Returns:
            List of Node objects meeting confidence threshold
        """
        nodes = self.graph.get_nodes()
        return [
            n
            for n in nodes
            if n.get_metadata().get("confidence", 0.0) >= min_confidence
        ]

    # Link Operations
    def add_link(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 0.5,
        metadata: dict[str, Any] | None = None,
        symmetric: bool = False,
    ) -> None:
        """
        Create a relationship between two concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relation: Relationship label
            weight: Relationship weight (0.0 to 1.0)
            metadata: Additional metadata
            symmetric: If True, creates bidirectional relationship

        Raises:
            ValueError: If source or target doesn't exist
        """
        # Validate nodes exist
        if not self.graph.has_node(source_id):
            raise ValueError(f"Source concept '{source_id}' does not exist")
        if not self.graph.has_node(target_id):
            raise ValueError(f"Target concept '{target_id}' does not exist")

        # Generate edge ID
        edge_id = f"{source_id}_{relation}_{target_id}"
        if symmetric:
            # For symmetric, use sorted IDs for consistent edge ID
            sorted_ids = sorted([source_id, target_id])
            edge_id = f"{sorted_ids[0]}_{relation}_{sorted_ids[1]}"

        # Check if edge already exists
        if self.graph.has_edge(edge_id):
            # Update existing edge
            edge = self.graph.get_edge(edge_id)
            edge.set_weight(weight)
            if metadata:
                for k, v in metadata.items():
                    edge.update_metadata(k, v)
            return

        # Create connection
        connections: set[str] | tuple[str, str]
        edge_type: int

        if symmetric:
            connections = {source_id, target_id}
            edge_type = EdgeType.SYMMETRIC
        else:
            connections = (source_id, target_id)
            edge_type = EdgeType.ASYMMETRIC

        metadata = metadata or {}
        metadata.setdefault("created_at", datetime.now().isoformat())
        metadata.setdefault("relation", relation)

        edge = Edge(edge_id, relation, edge_type, connections, weight, metadata)
        self.graph.add_edge(edge)

    def get_link(self, edge_id: str) -> Edge | None:
        """
        Retrieve a relationship by ID.

        Args:
            edge_id: Edge ID to retrieve

        Returns:
            Edge object or None if not found
        """
        if not self.graph.has_edge(edge_id):
            return None
        return self.graph.get_edge(edge_id)

    def has_link(self, edge_id: str) -> bool:
        """Check if a relationship exists."""
        return self.graph.has_edge(edge_id)

    def update_link(
        self,
        edge_id: str,
        weight: float | None = None,
        metadata: dict[str, Any] | None = None,
        merge_metadata: bool = True,
    ) -> None:
        """
        Update an existing relationship.

        Args:
            edge_id: Edge ID to update
            weight: New weight (None to keep existing)
            metadata: New metadata (merge_metadata controls behavior)
            merge_metadata: If True, merge with existing; if False, replace

        Raises:
            ValueError: If edge_id doesn't exist
        """
        if not self.graph.has_edge(edge_id):
            raise ValueError(f"Link '{edge_id}' does not exist")

        edge = self.graph.get_edge(edge_id)

        if weight is not None:
            edge.set_weight(weight)

        if metadata is not None:
            if merge_metadata:
                existing = edge.get_metadata()
                existing.update(metadata)
                existing["last_updated"] = datetime.now().isoformat()
                edge.set_metadata(existing)
            else:
                metadata["last_updated"] = datetime.now().isoformat()
                edge.set_metadata(metadata)

    def delete_link(self, edge_id: str) -> bool:
        """
        Delete a relationship.

        Args:
            edge_id: Edge ID to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.graph.has_edge(edge_id):
            return False

        self.graph.remove_edge(edge_id)
        return True

    def get_links_between(self, source_id: str, target_id: str) -> list[Edge]:
        """
        Get all links between two concepts.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID

        Returns:
            List of Edge objects between the two concepts
        """
        result = []
        for edge in self.graph.get_edges():
            conn = edge.get_connections()
            if edge.is_group_edge():
                # Group edge: check if both are in the group
                if source_id in conn and target_id in conn:
                    result.append(edge)
            else:
                # Regular edge: check source/target
                source, target = conn
                if (source == source_id and target == target_id) or (
                    source == target_id and target == source_id
                ):
                    result.append(edge)

        return result

    def get_neighbors(self, concept_id: str, max_depth: int = 1) -> list[Node]:
        """
        Get neighboring concepts within a depth.

        Args:
            concept_id: Center concept ID
            max_depth: Maximum traversal depth

        Returns:
            List of neighboring Node objects

        Raises:
            ValueError: If concept_id doesn't exist
        """
        if not self.graph.has_node(concept_id):
            raise ValueError(f"Concept '{concept_id}' does not exist")

        from memory_graph.core.traversal import bfs

        node_ids = bfs(self.graph, concept_id, max_depth)
        # Remove the center node itself (first in BFS result)
        if node_ids and node_ids[0] == concept_id:
            node_ids = node_ids[1:]

        return [self.graph.get_node(nid) for nid in node_ids]

    # Query Operations
    def query(
        self,
        concept_id: str,
        max_depth: int = 1,
        min_confidence: float = 0.0,
        min_weight: float = 0.0,
    ) -> dict[str, Any]:
        """
        Query the knowledge graph around a concept.

        Args:
            concept_id: Center concept ID
            max_depth: Maximum traversal depth
            min_confidence: Minimum node confidence
            min_weight: Minimum edge weight

        Returns:
            Dictionary with nodes, edges, and metadata

        Raises:
            ValueError: If concept_id doesn't exist
        """
        if not self.graph.has_node(concept_id):
            raise ValueError(f"Concept '{concept_id}' does not exist")

        # Use C++ query for performance
        result = self.graph.query(concept_id, max_depth, min_weight)

        # Filter nodes by confidence
        nodes = result.get("nodes", [])
        if nodes and min_confidence > 0.0:
            filtered_nodes = []
            for node_data in nodes:
                if (
                    node_data.get("metadata", {}).get("confidence", 1.0)
                    >= min_confidence
                ):
                    filtered_nodes.append(node_data)
            result["nodes"] = filtered_nodes

        return result

    def query_path(
        self,
        from_id: str,
        to_id: str,
    ) -> dict[str, Any] | None:
        """
        Find the shortest path between two concepts.

        Args:
            from_id: Starting concept ID
            to_id: Target concept ID

        Returns:
            Dictionary with path, nodes, and edges, or None if no path exists

        Raises:
            ValueError: If from_id or to_id doesn't exist
        """
        from memory_graph.core.traversal import (
            find_shortest_path_with_details,
        )

        if not self.graph.has_node(from_id):
            raise ValueError(f"Source concept '{from_id}' does not exist")
        if not self.graph.has_node(to_id):
            raise ValueError(f"Target concept '{to_id}' does not exist")

        return find_shortest_path_with_details(self.graph, from_id, to_id)

    def get_all_concepts(self) -> list[Node]:
        """Get all concepts in the knowledge graph."""
        return self.graph.get_nodes()

    def get_all_links(self) -> list[Edge]:
        """Get all relationships in the knowledge graph."""
        return self.graph.get_edges()

    # Confidence Management
    def set_confidence(
        self,
        concept_id: str,
        confidence: float,
    ) -> None:
        """
        Set confidence score for a concept.

        Args:
            concept_id: Concept ID
            confidence: Confidence score (0.0 to 1.0)

        Raises:
            ValueError: If concept doesn't exist or confidence out of range
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if not self.graph.has_node(concept_id):
            raise ValueError(f"Concept '{concept_id}' does not exist")

        node = self.graph.get_node(concept_id)
        node.update_metadata("confidence", confidence)
        node.update_metadata("last_updated", datetime.now().isoformat())

    def get_confidence(self, concept_id: str) -> float:
        """
        Get confidence score for a concept.

        Args:
            concept_id: Concept ID

        Returns:
            Confidence score (0.0 to 1.0), defaulting to 0.0 if not set

        Raises:
            ValueError: If concept doesn't exist
        """
        if not self.graph.has_node(concept_id):
            raise ValueError(f"Concept '{concept_id}' does not exist")

        node = self.graph.get_node(concept_id)
        return node.get_metadata().get("confidence", 0.0)

    # Serialization
    def to_json(self) -> dict[str, Any]:
        """Serialize the entire knowledge graph to JSON."""
        return self.graph.to_json()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SymbolicMemory":
        """Deserialize a knowledge graph from JSON."""
        graph = MemoryGraph.from_json(data)
        return cls(graph)

    def save(self, filepath: str) -> None:
        """Save knowledge graph to file."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SymbolicMemory":
        """Load knowledge graph from file."""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_json(data)

    # Utility Methods
    def clear(self) -> None:
        """Clear all concepts and links from the knowledge graph."""
        # Remove all edges first (needed to avoid orphaned connections)
        for edge in self.graph.get_edges():
            self.graph.remove_edge(edge.get_id())

        # Remove all nodes
        for node in self.graph.get_nodes():
            self.graph.remove_node(node.get_id())

    def __len__(self) -> int:
        """Number of concepts in the knowledge graph."""
        return len(self.graph)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SymbolicMemory(nodes={len(self.graph.get_nodes())}, "
            f"edges={len(self.graph.get_edges())})"
        )
