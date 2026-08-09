"""
Unit tests for GraphRepresentation.

Uses mocking to isolate Python representation logic from C++ bindings,
ensuring fast, reliable, and deterministic test execution.
"""

import json
import re
import pytest

# Attempt to import real C++ types; fall back to mocks if not built
try:
    from memory_graph.memory_graph_core import EdgeType
    from memory_graph.core.representation import GraphRepresentation

    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False
    pytestmark = pytest.mark.skip(
        reason="C++ bindings not available. Build them first."
    )


# Mock Fixtures
class MockNode:
    def __init__(self, node_id: str, label: str, metadata: dict | None = None):
        self._id = node_id
        self._label = label
        self._metadata = metadata or {}

    def get_id(self):
        return self._id

    def get_label(self):
        return self._label

    def get_metadata(self):
        return self._metadata

    def to_json(self):
        return {"id": self._id, "label": self._label, "metadata": self._metadata or {}}

    @staticmethod
    def from_json(data: dict):
        return MockNode(data["id"], data["label"], data.get("metadata", {}))


class MockEdge:
    def __init__(
        self,
        edge_id: str,
        label: str,
        edge_type: int,
        connections,
        weight: float = 1.0,
        is_group: bool = False,
    ):
        self._id = edge_id
        self._label = label
        self._type = edge_type
        self._connections = connections
        self._weight = weight
        self._is_group = is_group

    def get_id(self):
        return self._id

    def get_label(self):
        return self._label

    def get_type(self):
        return self._type

    def get_connections(self):
        return self._connections

    def get_weight(self):
        return self._weight

    def is_group_edge(self):
        return self._is_group

    def to_json(self):
        # Serialize connections in the correct format
        if isinstance(self._connections, set):
            conn = {"type": "symmetric", "nodes": list(self._connections)}
        else:
            source, target = self._connections
            conn = {"type": "asymmetric", "source": source, "target": target}

        # Convert EdgeType enum to string representation
        type_str = "symmetric" if self._type == EdgeType.SYMMETRIC else "asymmetric"

        return {
            "id": self._id,
            "label": self._label,
            "type": type_str,
            "connections": conn,
            "weight": self._weight,
            "metadata": {},
        }

    @staticmethod
    def from_json(data: dict):
        return MockEdge(
            data["id"], data["label"], data["type"], data["connections"], data["weight"]
        )


class MockMemoryGraph:
    def __init__(self, nodes: list, edges: list, metadata: dict | None = None):
        self._nodes = nodes
        self._edges = edges
        self._metadata = metadata or {}
        self._node_ids = {n.get_id() for n in nodes}

    def get_nodes(self):
        return self._nodes

    def get_edges(self):
        return self._edges

    def get_metadata(self):
        return self._metadata

    def has_node(self, node_id: str):
        return node_id in self._node_ids

    def to_json(self):
        return {
            "nodes": [n.to_json() for n in self._nodes],
            "edges": [e.to_json() for e in self._edges],
            "metadata": self._metadata,
        }

    @staticmethod
    def from_json(data: dict):
        nodes = [MockNode.from_json(n) for n in data.get("nodes", [])]
        edges = [MockEdge.from_json(e) for e in data.get("edges", [])]
        return MockMemoryGraph(nodes, edges, data.get("metadata", {}))


@pytest.fixture
def sample_graph() -> MockMemoryGraph:
    """Creates a realistic sample graph for testing."""
    nodes = [
        MockNode(
            "geralt",
            "Geralt of Rivia",
            {"type": "witcher", "confidence": 0.95, "school": "Wolf"},
        ),
        MockNode("yennefer", "Yennefer", {"type": "sorceress", "confidence": 0.90}),
        MockNode("ciri", "Ciri", {"type": "witcher", "confidence": 0.98}),
    ]
    edges = [
        MockEdge(
            "e1", "loves", EdgeType.ASYMMETRIC, ("geralt", "yennefer"), weight=0.95
        ),
        MockEdge(
            "e2", "fellow_witchers", EdgeType.SYMMETRIC, {"geralt", "ciri"}, weight=0.85
        ),
    ]
    return MockMemoryGraph(nodes, edges, {"source": "test"})


@pytest.fixture
def large_group_graph() -> MockMemoryGraph:
    """Creates a graph with a massive group edge to test token-bomb prevention."""
    nodes = [MockNode(f"node_{i}", f"Node {i}", {"type": "member"}) for i in range(15)]
    # A group edge connecting all 15 nodes
    group_connections = {f"node_{i}" for i in range(15)}
    edges = [
        MockEdge(
            "group_1",
            "brotherhood",
            EdgeType.SYMMETRIC,
            group_connections,
            weight=1.0,
            is_group=True,
        )
    ]
    return MockMemoryGraph(nodes, edges)


# Test Cases
class TestGraphRepresentationLinearized:
    def test_linearized_formatting_and_sorting(self, sample_graph):
        result = GraphRepresentation.to_linearized_text(sample_graph)

        # Verify nodes are sorted by confidence (Ciri 0.98, Geralt 0.95, Yennefer 0.90)
        assert result.index("ciri") < result.index("geralt") < result.index("yennefer")

        # Verify metadata formatting
        assert "(type: witcher)" in result
        assert "{confidence: 0.95}" in result
        assert "{school: Wolf}" in result

        # Verify edge formatting (asymmetric)
        assert "geralt -> yennefer -> [loves]-> {weight: 0.95}" in result

        # Verify symmetric edge (order doesn't matter due to set iteration)
        assert "fellow_witchers" in result
        assert "bidirectional" in result
        assert "{weight: 0.85}" in result

    def test_linearized_truncation(self, sample_graph):
        result = GraphRepresentation.to_linearized_text(
            sample_graph, max_nodes=1, max_edges=0
        )
        assert "[Note: Showing 1 of 3 nodes]" in result
        # The header "## Relations" may still appear, but no edges should be listed
        lines = result.split("\n")
        relation_lines = [
            l
            for l in lines
            if l.strip().startswith("ciri") or l.strip().startswith("geralt")
        ]
        # Only the node should appear, not edges
        assert len([l for l in relation_lines if "->" in l or "<->" in l]) == 0


class TestGraphRepresentationTriples:
    def test_triples_formatting(self, sample_graph):
        result = GraphRepresentation.to_triples_text(sample_graph)
        assert "(geralt) -[loves]-> (yennefer) {weight: 0.95}" in result

        # Symmetric edge (order doesn't matter)
        assert "fellow_witchers" in result
        assert "{weight: 0.85}" in result

    def test_group_edge_token_bomb_prevention(self, large_group_graph):
        """Verifies that large group edges do not generate O(N^2) triples."""
        result = GraphRepresentation.to_triples_text(large_group_graph)

        # Should NOT contain 105 pairwise combinations
        assert result.count("->") < 5

        # Should contain the summarized group representation
        assert "Group" in result
        assert "brotherhood (members:" in result
        assert "(+5 more)" in result

        # Verify exactly 10 nodes are listed
        members_match = re.search(r"members: ([^(]+)\(\+5 more\)", result)
        assert members_match
        members = members_match.group(1).strip().split(", ")
        assert len(members) == 10


class TestGraphRepresentationCypher:
    def test_cypher_metadata_truncation(self, sample_graph):
        """Verifies that Cypher output safely truncates metadata to 3 keys, preventing invalid JSON."""
        result = GraphRepresentation.to_cypher_like(sample_graph)

        # Geralt has 3 metadata keys: type, confidence, school. All should be present.
        assert '"type": "witcher"' in result
        assert '"confidence": 0.95' in result
        assert '"school": "Wolf"' in result

        # Verify valid JSON structure (no truncated strings like "wit...)
        lines = result.split("\n")
        node_lines = [l for l in lines if l.startswith("CREATE (")]
        for line in node_lines:
            # Extract the JSON object from the Cypher line
            # Format: CREATE (:Label {id: '...', metadata: {...}})
            json_match = re.search(r"metadata: (\{.*\})\};$", line)
            assert json_match, f"Could not extract JSON from line: {line}"
            json_str = json_match.group(1)

            # This will raise JSONDecodeError if truncation broke the syntax
            parsed = json.loads(json_str)
            assert len(parsed) <= 3  # Max 3 keys


class TestGraphRepresentationNarrative:
    def test_narrative_generation(self, sample_graph):
        result = GraphRepresentation.to_narrative(sample_graph)
        assert "Geralt of Rivia (a witcher) (confidence: 0.95)" in result
        assert "loves" in result

    def test_narrative_group_edge_summary(self, large_group_graph):
        result = GraphRepresentation.to_narrative(large_group_graph)
        assert "The group 'brotherhood' includes:" in result
        assert "(+10 more)" in result

        # Verify exactly 5 nodes are listed
        members_match = re.search(r"includes: ([^(]+)\(\+10 more\)", result)
        assert members_match
        members = members_match.group(1).strip().split(", ")
        assert len(members) == 5


class TestGraphRepresentationHierarchical:
    def test_hierarchical_grouping(self, sample_graph):
        result = GraphRepresentation.to_hierarchical_text(sample_graph)
        assert "## WITCHER (2 nodes)" in result
        assert "## SORCERESS (1 nodes)" in result
        # Verify tree structure formatting
        assert "├── [geralt] Geralt of Rivia" in result


class TestGraphRepresentationFiltering:
    def test_filter_graph_deep_copy(self):
        """Verifies that _filter_graph uses to_json/from_json for safe deep copying."""
        # Create real mock objects with both nodes referenced by the edge
        mock_node1 = MockNode("n1", "Node 1", {"confidence": 0.9})
        mock_node2 = MockNode("n2", "Node 2", {"confidence": 0.8})
        mock_edge = MockEdge("e1", "rel", EdgeType.ASYMMETRIC, ("n1", "n2"), weight=0.8)
        mock_graph = MockMemoryGraph([mock_node1, mock_node2], [mock_edge], {})

        # Call the actual method
        result = GraphRepresentation._filter_graph(
            mock_graph, min_confidence=0.5, min_weight=0.5
        )

        # Verify the result is a new graph with copied nodes/edges
        assert len(result.get_nodes()) == 2
        assert len(result.get_edges()) == 1
        assert result.get_nodes()[0].get_id() in ["n1", "n2"]
        assert result.get_edges()[0].get_id() == "e1"

        # Verify they are different objects (deep copy)
        original_node_ids = {n.get_id() for n in [mock_node1, mock_node2]}
        result_node_ids = {n.get_id() for n in result.get_nodes()}
        assert original_node_ids == result_node_ids

        # Verify the objects themselves are different instances
        for result_node in result.get_nodes():
            assert result_node is not mock_node1
            assert result_node is not mock_node2

        assert result.get_edges()[0] is not mock_edge


class TestGraphRepresentationUtilities:
    def test_truncate_to_tokens_safe_boundaries(self):
        # Create a string longer than the limit to force truncation
        text = "A" * 88 + ". " + "B" * 20  # 110 chars total
        result = GraphRepresentation._truncate_to_tokens(
            text, max_tokens=25
        )  # 25 * 4 = 100 chars

        # Should truncate at the period
        assert result.endswith(". \n\n[Truncated due to token limit]")
        assert "B" not in result

    def test_truncate_no_truncation_needed(self):
        text = "Short text"
        result = GraphRepresentation._truncate_to_tokens(text, max_tokens=100)
        assert result == text

    def test_unsupported_format_raises_error(self, sample_graph):
        with pytest.raises(ValueError, match="Unsupported format: invalid_format"):
            GraphRepresentation.to_llm_context(sample_graph, format="invalid_format")

    def test_subgraph_requires_center_node(self, sample_graph):
        with pytest.raises(ValueError, match="requires a 'center_node' argument"):
            GraphRepresentation.to_llm_context(sample_graph, format="subgraph")
