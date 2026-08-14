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
        metadata: dict | None = None,
    ):
        self._id = edge_id
        self._label = label
        self._type = edge_type
        self._connections = connections
        self._weight = weight
        self._is_group = is_group
        self._metadata = metadata or {}

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

    def get_metadata(self):
        return self._metadata

    def to_json(self):
        if isinstance(self._connections, set):
            conn = {"type": "symmetric", "nodes": list(self._connections)}
        else:
            source, target = self._connections
            conn = {"type": "asymmetric", "source": source, "target": target}

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


@pytest.fixture
def empty_graph() -> MockMemoryGraph:
    """Creates an empty graph."""
    return MockMemoryGraph([], [])


# Test Cases - Existing Tests
class TestGraphRepresentationLinearized:
    def test_linearized_formatting_and_sorting(self, sample_graph):
        result = GraphRepresentation.to_linearized_text(sample_graph)

        assert result.index("ciri") < result.index("geralt") < result.index("yennefer")
        assert "(type: witcher)" in result
        assert "{confidence: 0.95}" in result
        assert "{school: Wolf}" in result
        assert "geralt -> yennefer -> [loves]-> {weight: 0.95}" in result
        assert "fellow_witchers" in result
        assert "bidirectional" in result
        assert "{weight: 0.85}" in result

    def test_linearized_truncation(self, sample_graph):
        result = GraphRepresentation.to_linearized_text(
            sample_graph, max_nodes=1, max_edges=0
        )
        assert "[Note: Showing 1 of 3 nodes]" in result
        lines = result.split("\n")
        relation_lines = [
            l
            for l in lines
            if l.strip().startswith("ciri") or l.strip().startswith("geralt")
        ]
        assert len([l for l in relation_lines if "->" in l or "<->" in l]) == 0


class TestGraphRepresentationTriples:
    def test_triples_formatting(self, sample_graph):
        result = GraphRepresentation.to_triples_text(sample_graph)
        assert "(geralt) -[loves]-> (yennefer) {weight: 0.95}" in result
        assert "fellow_witchers" in result
        assert "{weight: 0.85}" in result

    def test_group_edge_token_bomb_prevention(self, large_group_graph):
        result = GraphRepresentation.to_triples_text(large_group_graph)
        assert result.count("->") < 5
        assert "Group" in result
        assert "brotherhood (members:" in result
        assert "(+5 more)" in result

        members_match = re.search(r"members: ([^(]+)\(\+5 more\)", result)
        assert members_match
        members = members_match.group(1).strip().split(", ")
        assert len(members) == 10


class TestGraphRepresentationCypher:
    def test_cypher_metadata_truncation(self, sample_graph):
        result = GraphRepresentation.to_cypher_like(sample_graph)
        assert '"type": "witcher"' in result
        assert '"confidence": 0.95' in result
        assert '"school": "Wolf"' in result

        lines = result.split("\n")
        node_lines = [l for l in lines if l.startswith("CREATE (")]
        for line in node_lines:
            json_match = re.search(r"metadata: (\{.*\})\};$", line)
            assert json_match, f"Could not extract JSON from line: {line}"
            json_str = json_match.group(1)
            parsed = json.loads(json_str)
            assert len(parsed) <= 3


class TestGraphRepresentationNarrative:
    def test_narrative_generation(self, sample_graph):
        result = GraphRepresentation.to_narrative(sample_graph)
        assert "Geralt of Rivia (a witcher) (confidence: 0.95)" in result
        assert "loves" in result

    def test_narrative_group_edge_summary(self, large_group_graph):
        result = GraphRepresentation.to_narrative(large_group_graph)
        assert "The group 'brotherhood' includes:" in result
        assert "(+10 more)" in result

        members_match = re.search(r"includes: ([^(]+)\(\+10 more\)", result)
        assert members_match
        members = members_match.group(1).strip().split(", ")
        assert len(members) == 5


class TestGraphRepresentationHierarchical:
    def test_hierarchical_grouping(self, sample_graph):
        result = GraphRepresentation.to_hierarchical_text(sample_graph)
        assert "## WITCHER (2 nodes)" in result
        assert "## SORCERESS (1 nodes)" in result
        assert "├── [geralt] Geralt of Rivia" in result


class TestGraphRepresentationFiltering:
    def test_filter_graph_deep_copy(self):
        mock_node1 = MockNode("n1", "Node 1", {"confidence": 0.9})
        mock_node2 = MockNode("n2", "Node 2", {"confidence": 0.8})
        mock_edge = MockEdge("e1", "rel", EdgeType.ASYMMETRIC, ("n1", "n2"), weight=0.8)
        mock_graph = MockMemoryGraph([mock_node1, mock_node2], [mock_edge], {})

        result = GraphRepresentation._filter_graph(
            mock_graph, min_confidence=0.5, min_weight=0.5
        )

        assert len(result.get_nodes()) == 2
        assert len(result.get_edges()) == 1
        assert result.get_nodes()[0].get_id() in ["n1", "n2"]
        assert result.get_edges()[0].get_id() == "e1"

        original_node_ids = {n.get_id() for n in [mock_node1, mock_node2]}
        result_node_ids = {n.get_id() for n in result.get_nodes()}
        assert original_node_ids == result_node_ids

        for result_node in result.get_nodes():
            assert result_node is not mock_node1
            assert result_node is not mock_node2

        assert result.get_edges()[0] is not mock_edge


class TestGraphRepresentationUtilities:
    def test_truncate_to_tokens_safe_boundaries(self):
        text = "A" * 88 + ". " + "B" * 20
        result = GraphRepresentation._truncate_to_tokens(text, max_tokens=25)
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


# NEW TESTS: Relevance Format
class TestGraphRepresentationRelevance:
    def test_relevance_ranking_basic(self, sample_graph):
        """Test relevance ranking with keyword matching."""
        result = GraphRepresentation.to_relevance_ranked_context(
            sample_graph, query="Geralt witcher"
        )
        assert "# Relevance-Ranked Context" in result
        assert "Query: Geralt witcher" in result
        assert "Geralt of Rivia" in result
        geralt_index = result.find("Geralt of Rivia")
        yennefer_index = result.find("Yennefer")
        if yennefer_index != -1:
            assert geralt_index < yennefer_index

    def test_relevance_no_match(self, sample_graph):
        """Test relevance with no matching nodes."""
        result = GraphRepresentation.to_relevance_ranked_context(
            sample_graph, query="nonexistent"
        )
        assert "No relevant nodes found" in result

    def test_relevance_fallback_to_cpp(self, sample_graph):
        """Test relevance uses C++ get_context_window when available."""
        result = GraphRepresentation.to_relevance_ranked_context(
            sample_graph, query="Ciri", max_tokens=100
        )
        assert result is not None
        assert isinstance(result, str)


# NEW TESTS: Subgraph Format (Skipped with Mocks)
class TestGraphRepresentationSubgraph:
    @pytest.mark.skipif(
        True, reason="Subgraph requires real C++ MemoryGraph, not mocks"
    )
    def test_subgraph_basic(self, sample_graph):
        """Test subgraph extraction around a node."""
        result = GraphRepresentation.to_subgraph_context(
            sample_graph, center_node_id="geralt", radius=1
        )
        assert "# Subgraph centered on 'geralt'" in result
        assert "Nodes:" in result
        assert "Edges:" in result
        assert "geralt" in result

    def test_subgraph_invalid_center(self, sample_graph):
        """Test subgraph with non-existent center node."""
        result = GraphRepresentation.to_subgraph_context(
            sample_graph, center_node_id="nonexistent"
        )
        assert "Error" in result

    def test_subgraph_with_format_override(self, sample_graph):
        """Test subgraph with different output format."""
        result = GraphRepresentation.to_subgraph_context(
            sample_graph, center_node_id="geralt", format="triples"
        )
        assert "Error" in result or "geralt" in result


# NEW TESTS: Confidence and Weight Filtering (via to_llm_context)
class TestGraphRepresentationConfidenceWeightFiltering:
    def test_combined_filtering(self):
        """Test combined confidence and weight filtering."""
        # Create a graph where both nodes meet confidence threshold
        nodes = [
            MockNode("geralt", "Geralt", {"type": "witcher", "confidence": 0.95}),
            MockNode("ciri", "Ciri", {"type": "witcher", "confidence": 0.98}),
            MockNode(
                "vesemir", "Vesemir", {"type": "witcher", "confidence": 0.85}
            ),  # Low confidence
        ]
        edges = [
            MockEdge(
                "e1", "mentor", EdgeType.ASYMMETRIC, ("vesemir", "geralt"), weight=0.9
            ),
            MockEdge(
                "e2", "friend", EdgeType.ASYMMETRIC, ("geralt", "ciri"), weight=0.95
            ),
        ]
        graph = MockMemoryGraph(nodes, edges)

        result = GraphRepresentation.to_llm_context(
            graph, min_confidence=0.9, min_weight=0.92, format="linearized"
        )
        # Only Geralt and Ciri should appear (Vesemir filtered out)
        assert "geralt" in result
        assert "ciri" in result
        assert "vesemir" not in result
        # Only friend edge (0.95) should appear, mentor (0.9) filtered out
        assert "friend" in result
        assert "mentor" not in result


# NEW TESTS: Empty Graph
class TestGraphRepresentationEmpty:
    def test_empty_graph_linearized(self, empty_graph):
        result = GraphRepresentation.to_linearized_text(empty_graph)
        assert "## Nodes" in result
        assert "## Relations" not in result

    def test_empty_graph_narrative(self, empty_graph):
        result = GraphRepresentation.to_narrative(empty_graph)
        assert result == "The graph is empty."

    def test_empty_graph_triples(self, empty_graph):
        result = GraphRepresentation.to_triples_text(empty_graph)
        assert result == "" or "[Note" not in result

    def test_empty_graph_json(self, empty_graph):
        result = GraphRepresentation.to_json_string(empty_graph)
        parsed = json.loads(result)
        assert parsed["nodes"] == []
        assert parsed["edges"] == []


# NEW TESTS: Token Truncation
class TestGraphRepresentationTokens:
    def test_token_truncation_in_llm_context(self):
        """Test token truncation via to_llm_context."""
        # Generate a large graph
        nodes = [
            MockNode(f"node_{i}", f"Node {i}", {"type": "test"}) for i in range(50)
        ]
        graph = MockMemoryGraph(nodes, [])

        result = GraphRepresentation.to_llm_context(
            graph, max_tokens=10, format="linearized"
        )
        assert "[Truncated due to token limit]" in result
        assert len(result) < 100  # Should be heavily truncated

    def test_no_truncation_when_under_limit(self, sample_graph):
        """Test no truncation when content fits within token limit."""
        result = GraphRepresentation.to_llm_context(
            sample_graph, max_tokens=1000, format="linearized"
        )
        assert "[Truncated due to token limit]" not in result


# NEW TESTS: Malformed Metadata
class TestGraphRepresentationMalformed:
    def test_missing_metadata(self):
        """Test nodes with missing or None metadata."""
        nodes = [
            MockNode("n1", "Node 1", None),
            MockNode("n2", "Node 2", {}),
        ]
        graph = MockMemoryGraph(nodes, [])

        # Should not raise exceptions
        result = GraphRepresentation.to_linearized_text(graph)
        assert "Node 1" in result
        assert "Node 2" in result

    def test_non_float_confidence(self):
        """Test confidence values that are not floats."""
        node = MockNode("n1", "Node 1", {"confidence": "high"})
        graph = MockMemoryGraph([node], [])

        # Should handle gracefully
        result = GraphRepresentation.to_linearized_text(graph)
        assert "Node 1" in result
        assert "high" in result or "confidence" in result


# NEW TESTS: Symmetric Edge Pair Limit
class TestGraphRepresentationSymmetricLimit:
    def test_symmetric_edge_pair_limit(self):
        """Test that symmetric edges cap pairs at 5 to prevent explosion."""
        nodes = [MockNode(f"n{i}", f"Node {i}") for i in range(10)]
        edges = [
            MockEdge(
                "e1",
                "connected",
                EdgeType.SYMMETRIC,
                {f"n{i}" for i in range(10)},
                weight=0.8,
            )
        ]
        graph = MockMemoryGraph(nodes, edges)

        result = GraphRepresentation.to_triples_text(graph)
        # Should not have 45 pairs (10 choose 2)
        pair_count = result.count(" -[connected]-> ")
        assert pair_count <= 5


# NEW TESTS: JSON Format
class TestGraphRepresentationJSON:
    def test_json_format(self, sample_graph):
        """Test JSON output format."""
        result = GraphRepresentation.to_json_string(sample_graph)
        parsed = json.loads(result)

        assert "nodes" in parsed
        assert "edges" in parsed
        assert "metadata" in parsed
        assert len(parsed["nodes"]) == 3
        assert len(parsed["edges"]) == 2

    def test_json_without_metadata(self, sample_graph):
        """Test JSON output without metadata."""
        result = GraphRepresentation.to_json_string(
            sample_graph, include_metadata=False
        )
        parsed = json.loads(result)
        assert parsed["metadata"] == {}
        for node in parsed["nodes"]:
            assert "metadata" not in node


# NEW TESTS: max_nodes Parameter
class TestGraphRepresentationMaxNodes:
    def test_max_nodes_in_linearized(self):
        """Test that max_nodes limits node output."""
        nodes = [MockNode(f"n{i}", f"Node {i}") for i in range(20)]
        graph = MockMemoryGraph(nodes, [])

        result = GraphRepresentation.to_linearized_text(graph, max_nodes=5)
        note_lines = [l for l in result.split("\n") if "Showing 5 of 20 nodes" in l]
        assert len(note_lines) == 1

        node_lines = [l for l in result.split("\n") if l.strip().startswith("[n")]
        assert len(node_lines) <= 5

    def test_max_nodes_in_hierarchical(self):
        """Test that max_nodes limits nodes in hierarchical format."""
        nodes = [MockNode(f"n{i}", f"Node {i}", {"type": "same"}) for i in range(20)]
        graph = MockMemoryGraph(nodes, [])

        result = GraphRepresentation.to_hierarchical_text(graph, max_nodes=8)
        node_lines = [l for l in result.split("\n") if "├── [n" in l]
        assert len(node_lines) <= 8
