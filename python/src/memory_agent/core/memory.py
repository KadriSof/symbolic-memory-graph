import json
import uuid
import base64
import logging

from typing import Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

from memory_graph.core import MemoryGraph, Node, Edge, EdgeType
from memory_graph.core.traversal import (
    find_nodes_by_label,
    find_nodes_by_metadata,
    subgraph_by_predicate,
)
from memory_graph.core.serialization import GraphSerializer


# MEMORY SYSTEM
@dataclass
class WorkingMemory:
    """
    Short-term context.
    Contains current goal, plan, intermediate results, and errors.
    """

    goal: str = ""
    plan: list[str] = field(default_factory=list)
    current_step: int = 0

    extracted_entities: list[dict[str, Any]] = field(default_factory=list)
    extracted_relations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_knowledge: dict[str, Any] = field(default_factory=dict)
    consolidated_knowledge: dict[str, Any] = field(default_factory=dict)

    conflicts: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    reasoning: str = ""
    final_answer: str = ""

    reasoning_complete: bool = False
    needs_more_data: bool = False
    clarification_response: str | None = None

    start_time: float | None = None
    step_times: dict[str, float] = field(default_factory=dict)
    token_usage: int = 0

    def reset(self) -> None:
        """Reset working memory to initial defaults."""
        self.goal = ""
        self.plan.clear()
        self.current_step = 0
        self.extracted_entities.clear()
        self.extracted_relations.clear()
        self.retrieved_knowledge.clear()
        self.consolidated_knowledge.clear()
        self.conflicts.clear()
        self.gaps.clear()
        self.reasoning = ""
        self.final_answer = ""
        self.reasoning_complete = False
        self.needs_more_data = False
        self.clarification_response = None
        self.start_time = None
        self.step_times.clear()
        self.token_usage = 0

    def to_context_string(self, max_chars: int = 2000) -> str:
        """Convert working memory to a concise prompt context."""
        context = []
        if self.goal:
            context.append(f"## Goal: {self.goal}")
        if self.plan:
            step_str = f"{self.current_step + 1}/{len(self.plan)}"
            context.append(f"## Plan: {' -> '.join(self.plan)} (Step {step_str})")
        if self.extracted_entities:
            context.append(f"## Entities: {json.dumps(self.extracted_entities)}")
        if self.retrieved_knowledge:
            context.append(
                f"## Retrieved: {json.dumps(self.retrieved_knowledge)[:500]}"
            )
        if self.conflicts:
            context.append(f"## Conflicts: {json.dumps(self.conflicts)[:500]}")
        if self.gaps:
            context.append(f"## Gaps: {', '.join(self.gaps)}")
        if self.reasoning:
            context.append(f"## Reasoning: {self.reasoning[:500]}")

        return "\n".join(context)[:max_chars]

    def to_dict(self) -> dict[str, Any]:
        """Serialize working memory."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemory":
        """Deserialize working memory."""
        return cls(**data)


# SYMBOLIC MEMORY
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
        metadata.setdefault("confidence", 0.5)

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

    def has_concept(self, concept_id: str) -> bool:
        """Check if a concept exists in the knowledge graph."""
        return self.graph.has_node(concept_id)

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

        # Remove center node
        if node_ids and node_ids[0] == concept_id:
            node_ids = node_ids[1:]

        # For BFS, include incoming edges
        # The C++ bfs handles both directions for symmetric edges,
        # but for asymmetric edges, we need to manually add incoming neighbors
        result = []
        for nid in node_ids:
            result.append(self.graph.get_node(nid))

        # Add incoming neighbors (nodes that have this node as target)
        incoming_neighbors = []
        for edge in self.graph.get_edges():
            if not edge.is_group_edge():
                source, target = edge.get_connections()
                if target == concept_id and source not in node_ids:
                    incoming_neighbors.append(self.graph.get_node(source))

        # Also add neighbors from group edges
        for edge in self.graph.get_edges():
            if edge.is_group_edge():
                members = list(edge.get_connections())
                if concept_id in members:
                    for member in members:
                        if member != concept_id and member not in node_ids:
                            incoming_neighbors.append(self.graph.get_node(member))

        # Combine and deduplicate
        all_neighbors = result + incoming_neighbors
        seen = set()
        deduped = []
        for node in all_neighbors:
            if node.get_id() not in seen:
                seen.add(node.get_id())
                deduped.append(node)

        return deduped

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
        from memory_graph.core.traversal import find_shortest_path_with_details

        if not self.graph.has_node(from_id) or not self.graph.has_node(to_id):
            return None

        # Try original direction first
        result = find_shortest_path_with_details(self.graph, from_id, to_id)
        if result is not None:
            return result

        # If not found, try reverse direction (for undirected semantics)
        result = find_shortest_path_with_details(self.graph, to_id, from_id)
        if result is not None:
            # Reverse the path
            result["path"] = list(reversed(result["path"]))
            result["nodes"] = list(reversed(result["nodes"]))
            result["edges"] = list(reversed(result["edges"]))
            return result

        return None

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


# EPSISODIC MEMORY
class EpisodicMemory:
    """
    Episodic Memory as a DAG over C++ MemoryGraph.

    Stores conversation turns as nodes with PRECEDES edges for history
    and REFERENCES edges to symbolic concepts. Enables branching,
    cross-memory grounding, and efficient binary serialization.

    Example:
        >>> memory = EpisodicMemory()
        >>> turn1 = memory.add_turn("user", "Who is Geralt?")
        >>> turn2 = memory.add_turn("assistant", "A witcher.")
        >>> memory.link_to_concept(turn2, "geralt", "REFERENCES")
        >>> turns = memory.get_recent_turns(5)
        >>> referenced = memory.get_turns_referencing("geralt")
    """

    def __init__(self, graph: MemoryGraph | None = None) -> None:
        """Initialize episodic memory with optional existing graph."""
        self.graph = graph or MemoryGraph({"type": "episodic_memory"})
        self._current_turn_id: str | None = None
        self._turn_order: list[str] = []
        self._turn_count: int = 0

        self.logger = logging.getLogger(name="[EpisodicMemory]")

        if graph is not None:
            self._rebuild_order()

    # Core Operations
    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> str:
        """
        Add a new turn to the episodic DAG.

        Args:
            role: "user", "assistant", "system", or "tool"
            content: Message content
            metadata: Additional metadata (tool_name, confidence, etc.)
            turn_id: Optional custom ID (auto-generated)

        Returns:
            The ID of the created turn.

        Raises:
            ValueError: If role is invalid.
        """
        valid_roles = {"user", "assistant", "system", "tool"}

        if role not in valid_roles:
            raise ValueError(f"Invalid role '{role}'. Must be one of: {valid_roles}")

        turn_id = turn_id or f"turn_{uuid.uuid4().hex[:8]}"
        if self.graph.has_node(turn_id):
            raise ValueError(f"Turn '{turn_id}' already exists")

        timestamp = datetime.now().isoformat()

        node_metadata = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            # TODO: I already have a token count func in base.py
            # let's place it in 'utils' and use it here.
            "tokens": len(content) // 4,
            **(metadata or {}),
        }

        node = Node(turn_id, "EpisodicTurn", node_metadata)
        self.graph.add_node(node)

        # DAG seq linking to previous turn
        if self._current_turn_id:
            edge_id = f"edge_{self._current_turn_id}_{turn_id}"
            if not self.graph.has_edge(edge_id):
                edge = Edge(
                    edge_id,
                    "PRECEDES",
                    EdgeType.ASYMMETRIC,
                    (self._current_turn_id, turn_id),
                    1.0,
                )
                self.graph.add_edge(edge)

        self._current_turn_id = turn_id
        self._turn_order.append(turn_id)
        self._turn_count += 1

        self.logger.debug(f"Added turn {turn_id} ({role})")
        return turn_id

    def get_turn(self, turn_id: str) -> Node | None:
        """Retrieve a turn by its ID."""
        if self.graph.has_node(turn_id):
            return self.graph.get_node(turn_id)
        return None

    def get_recent_turns(self, count: int = 10) -> list[Node]:
        """Get the most recent N turns."""
        if count <= 0:
            return []

        recent_ids = self._turn_order[-count:] if self._turn_order else []
        return [
            self.graph.get_node(tid) for tid in recent_ids if self.graph.has_node(tid)
        ]

    def get_turns_by_role(self, role: str) -> list[Node]:
        """Get all turns with a specific role."""
        node_ids = find_nodes_by_metadata(self.graph, "role", role)
        return [self.graph.get_node(nid) for nid in node_ids]

    def get_turns_since(self, timestamp: str) -> list[Node]:
        """Get all turns after a given timestamp."""
        result = []
        for tid in self._turn_order:
            node = self.graph.get_node(tid)
            ts = node.get_metadata().get("timestamp", "")
            if ts > timestamp:
                result.append(node)

        result.sort(key=lambda n: n.get_metadata().get("timestamp", ""))
        return result

    def get_turns_between(self, start_time: str, end_time: str) -> list[Node]:
        """Get all turns between two timestamps."""
        result = []
        for tid in self._turn_order:
            node = self.graph.get_node(tid)
            ts = node.get_metadata().get("timestamp", "")
            if start_time < ts < end_time:
                result.append(node)
        result.sort(key=lambda n: n.get_metadata().get("timestamp", ""))
        return result

    # Cross-Memory Grounding
    def link_to_concept(
        self,
        turn_id: str,
        concept_id: str,
        relation: str = "REFERENCES",
        weight: float = 1.0,
    ) -> None:
        """
        Link a turn to a symbolic concept.

        Enables queries like "Show all turns about concept X".

        Args:
            turn_id: Turn node ID
            concept_id: Concept node ID (from SymbolicMemory)
            relation: Edge label (default: "REFERENCES")
            weight: Edge weight (0.0-1.0)

        Raises:
            ValueError: If turn or concept doesn't exist.
        """
        if not self.graph.has_node(turn_id):
            raise ValueError(f"Turn '{turn_id}' does not exist")
        if not self.graph.has_node(concept_id):
            raise ValueError(f"Concept '{concept_id}' does not exist")

        edge_id = f"edge_{turn_id}_{relation}_{concept_id}"
        if not self.graph.has_edge(edge_id):
            edge = Edge(
                id=edge_id,
                label=relation,
                type=EdgeType.ASYMMETRIC,
                connections=(turn_id, concept_id),
                weight=weight,
            )
            self.graph.add_edge(edge)

    def get_turns_referencing(self, concept_id: str) -> list[Node]:
        """Get all turns referencing a specific concept."""
        result = []
        for edge in self.graph.get_edges():
            if edge.get_label() == "REFERENCES":
                source, target = edge.get_connections()
                if target == concept_id and self.graph.has_node(source):
                    node = self.graph.get_node(source)
                    if node.get_label() == "EpisodicTurn":
                        result.append(node)

        return result

    # Query Operations
    def query(
        self,
        query: str,
        max_results: int = 5,
        min_confidence: float = 0.0,
    ) -> list[Node]:
        """
        Query turns by keyword matching in content.

        Args:
            query: Search query
            max_results: Maximum results to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching turns, sorted by relevance.
        """
        query_words = set(query.lower().split())
        if not query_words:
            return []

        def matches_query(node: Node) -> bool:
            content = node.get_metadata().get("content", "").lower()
            for word in query_words:
                if word not in content:
                    return False
            confidence = node.get_metadata().get("confidence", 1.0)
            return confidence >= min_confidence

        # C++ traversal for efficient filtering
        try:
            sub = subgraph_by_predicate(self.graph, matches_query)
            nodes = sub.get_nodes()
            nodes.sort(
                key=lambda n: n.get_metadata().get("timestamp", ""), reverse=True
            )
            return nodes[:max_results]
        except Exception as e:
            self.logger.warning(f"Query failed, falling back to manual: {e}")
            # Fallback: manual scoring
            turn_ids = find_nodes_by_label(self.graph, "EpisodicTurn")
            scored = []
            for tid in turn_ids:
                node = self.graph.get_node(tid)
                content = node.get_metadata().get("content", "").lower()
                score = sum(1 for w in query_words if w in content)
                if score > 0:
                    confidence = node.get_metadata().get("confidence", 1.0)
                    if confidence >= min_confidence:
                        scored.append((node, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [n for n, _ in scored[:max_results]]

    # Utility Operations
    def clear(self) -> None:
        """Clear all turns and edges."""
        turn_ids = set(find_nodes_by_label(self.graph, "EpisodicTurn"))
        if not turn_ids:
            return

        # Remove edges first to avoid orphaned references
        edges_to_remove = []
        for edge in self.graph.get_edges():
            conn = edge.get_connections()
            if not edge.is_group_edge():
                source, target = conn
                if source in turn_ids or target in turn_ids:
                    edges_to_remove.append(edge.get_id())
            else:
                if any(member in turn_ids for member in conn):
                    edges_to_remove.append(edge.get_id())

        for eid in edges_to_remove:
            self.graph.remove_edge(eid)

        for tid in turn_ids:
            self.graph.remove_node(tid)

        self._current_turn_id = None
        self._turn_order.clear()
        self._turn_count = 0

    def __len__(self) -> int:
        """Number of turns (O(1))."""
        return self._turn_count

    def __repr__(self) -> str:
        return f"EpisodicMemory(turns={len(self)})"

    # Serialization
    def to_binary_b64(self) -> str:
        """Serialize to base64-encoded binary."""
        binary_data = GraphSerializer.to_binary(self.graph)
        return base64.b64encode(binary_data).decode("ascii")

    @classmethod
    def from_binary_b64(cls, b64_data: str) -> "EpisodicMemory":
        """Deserialize from base64-encoded binary."""
        if not b64_data:
            return cls()
        try:
            binary_data = base64.b64decode(b64_data.encode("ascii"))
            graph = GraphSerializer.from_binary(binary_data)
            memory = cls(graph=graph)
            memory._rebuild_order()
            return memory
        except Exception as e:
            logging.getLogger("[EpisodicMemory]").error(f"Deserialization failed: {e}")
            return cls()

    def _rebuild_order(self) -> None:
        """Rebuild turn order from graph after deserialization."""
        turn_ids = find_nodes_by_label(self.graph, "EpisodicTurn")
        turns = []
        for tid in turn_ids:
            node = self.graph.get_node(tid)
            turns.append((tid, node.get_metadata().get("timestamp", "")))
        turns.sort(key=lambda x: x[1])
        self._turn_order = [tid for tid, _ in turns]
        self._turn_count = len(self._turn_order)
        self._current_turn_id = self._turn_order[-1] if self._turn_order else None
