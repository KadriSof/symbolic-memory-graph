"""
Unit tests for the Graph Traversal component.

Uses a Dragon Ball-themed graph to validate C++ traversal functions
and Python helper utilities in a realistic, relatable context.
"""

import pytest

try:
    from memory_graph import MemoryGraph, Node, Edge, EdgeType
    from memory_graph.core.traversal import (
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
        find_entities_by_type,
        get_confidence_filtered_subgraph,
        find_shortest_path_with_details,
    )

    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False
    pytestmark = pytest.mark.skip(
        reason="C++ bindings not available. Build them first."
    )


@pytest.fixture
def db_graph() -> MemoryGraph:
    """Creates a Dragon Ball-themed knowledge graph with directed and undirected edges."""
    graph = MemoryGraph({"universe": "Dragon Ball", "saga": "Namek"})

    graph.add_node(
        Node(
            "goku",
            "Son Goku",
            {"type": "Saiyan", "power_level": 9000, "confidence": 0.99},
        )
    )
    graph.add_node(
        Node(
            "vegeta",
            "Vegeta",
            {"type": "Saiyan", "power_level": 8500, "confidence": 0.98},
        )
    )
    graph.add_node(
        Node(
            "gohan",
            "Son Gohan",
            {"type": "Saiyan", "power_level": 8000, "confidence": 0.95},
        )
    )
    graph.add_node(
        Node(
            "piccolo",
            "Piccolo",
            {"type": "Namekian", "power_level": 7000, "confidence": 0.90},
        )
    )
    graph.add_node(
        Node(
            "krillin",
            "Krillin",
            {"type": "Human", "power_level": 2000, "confidence": 0.95},
        )
    )
    graph.add_node(
        Node(
            "trunks",
            "Trunks",
            {"type": "Saiyan", "power_level": 7500, "confidence": 0.92},
        )
    )
    graph.add_node(
        Node(
            "frieza",
            "Frieza",
            {"type": "Frieza Race", "power_level": 10000, "confidence": 0.99},
        )
    )

    # Symmetric Edges (Allies/Friends)
    graph.add_edge(
        Edge(
            "goku_piccolo_allies",
            "allies",
            EdgeType.SYMMETRIC,
            {"goku", "piccolo"},
            weight=0.8,
        )
    )
    graph.add_edge(
        Edge(
            "goku_krillin_friends",
            "friends",
            EdgeType.SYMMETRIC,
            {"goku", "krillin"},
            weight=0.95,
        )
    )

    # Asymmetric Edges (Family/Enemies)
    graph.add_edge(
        Edge(
            "goku_vegeta_rivals",
            "rivals",
            EdgeType.ASYMMETRIC,
            ("goku", "vegeta"),
            weight=0.9,
        )
    )
    graph.add_edge(
        Edge(
            "goku_father_gohan",
            "father_of",
            EdgeType.ASYMMETRIC,
            ("goku", "gohan"),
            weight=1.0,
        )
    )
    graph.add_edge(
        Edge(
            "vegeta_father_trunks",
            "father_of",
            EdgeType.ASYMMETRIC,
            ("vegeta", "trunks"),
            weight=1.0,
        )
    )
    graph.add_edge(
        Edge(
            "frieza_enemy_goku",
            "enemy_of",
            EdgeType.ASYMMETRIC,
            ("frieza", "goku"),
            weight=1.0,
        )
    )
    graph.add_edge(
        Edge(
            "frieza_enemy_vegeta",
            "enemy_of",
            EdgeType.ASYMMETRIC,
            ("frieza", "vegeta"),
            weight=1.0,
        )
    )

    return graph


@pytest.fixture
def cycle_graph() -> MemoryGraph:
    graph = MemoryGraph({"test": "cycle"})
    graph.add_node(Node("a", "Node A", {}))
    graph.add_node(Node("b", "Node B", {}))
    graph.add_node(Node("c", "Node C", {}))
    graph.add_edge(Edge("e1", "points_to", EdgeType.ASYMMETRIC, ("a", "b"), weight=1.0))
    graph.add_edge(Edge("e2", "points_to", EdgeType.ASYMMETRIC, ("b", "c"), weight=1.0))
    graph.add_edge(Edge("e3", "points_to", EdgeType.ASYMMETRIC, ("c", "a"), weight=1.0))
    return graph


@pytest.fixture
def dag_graph() -> MemoryGraph:
    graph = MemoryGraph({"test": "dag"})
    graph.add_node(Node("base", "Base Form", {}))
    graph.add_node(Node("kaioken", "Kaioken", {}))
    graph.add_node(Node("ssj", "Super Saiyan", {}))
    graph.add_node(Node("ssj2", "Super Saiyan 2", {}))
    graph.add_edge(
        Edge("e1", "unlocks", EdgeType.ASYMMETRIC, ("base", "kaioken"), weight=1.0)
    )
    graph.add_edge(
        Edge("e2", "unlocks", EdgeType.ASYMMETRIC, ("base", "ssj"), weight=1.0)
    )
    graph.add_edge(
        Edge("e3", "unlocks", EdgeType.ASYMMETRIC, ("kaioken", "ssj"), weight=1.0)
    )
    graph.add_edge(
        Edge("e4", "unlocks", EdgeType.ASYMMETRIC, ("ssj", "ssj2"), weight=1.0)
    )
    return graph


class TestCoreTraversal:
    def test_bfs(self, db_graph: MemoryGraph):
        """Test BFS. Note: Frieza points TO Goku/Vegeta, so we start from Frieza to reach everyone."""
        result_depth_1 = bfs(db_graph, "frieza", max_depth=1)
        assert "frieza" in result_depth_1
        assert "goku" in result_depth_1
        assert "vegeta" in result_depth_1

        result_depth_2 = bfs(db_graph, "frieza", max_depth=2)
        assert "gohan" in result_depth_2
        assert "trunks" in result_depth_2
        assert "krillin" in result_depth_2

    def test_dfs(self, db_graph: MemoryGraph):
        result = dfs(db_graph, "frieza", max_depth=2)
        assert "frieza" in result
        assert "goku" in result

    def test_shortest_path(self, db_graph: MemoryGraph):
        # Krillin <-> Goku -> Vegeta
        path = shortest_path(db_graph, "krillin", "vegeta")
        assert path == ["krillin", "goku", "vegeta"]

    def test_find_all_paths(self, db_graph: MemoryGraph):
        paths = find_all_paths(db_graph, "frieza", "gohan", max_depth=3)
        assert len(paths) >= 1
        assert ["frieza", "goku", "gohan"] in paths

    def test_has_cycle(self, cycle_graph: MemoryGraph, db_graph: MemoryGraph):
        assert has_cycle(cycle_graph) is True
        assert has_cycle(db_graph) is False

    def test_topological_sort(self, dag_graph: MemoryGraph, cycle_graph: MemoryGraph):
        result = topological_sort(dag_graph)
        assert result.index("base") < result.index("ssj")
        with pytest.raises(RuntimeError, match="Graph has a cycle"):
            topological_sort(cycle_graph)

    def test_is_connected(self, db_graph: MemoryGraph):
        """Frieza can reach everyone via directed edges. Goku cannot reach Frieza."""
        assert is_connected(db_graph, "frieza") is True
        assert (
            is_connected(db_graph, "goku") is False
        )  # Directed edge is frieza -> goku

    def test_subgraph(self, db_graph: MemoryGraph):
        """Test subgraph extraction by radius from Frieza."""
        sub = subgraph(db_graph, center="frieza", radius=1, include_edges=True)
        node_ids = [n.get_id() for n in sub.get_nodes()]
        assert "frieza" in node_ids
        assert "goku" in node_ids
        assert "vegeta" in node_ids
        assert "gohan" not in node_ids  # Radius 2 from Frieza

    def test_subgraph_by_predicate(self, db_graph: MemoryGraph):
        def is_saiyan(node: Node) -> bool:
            return node.get_metadata().get("type") == "Saiyan"

        sub = subgraph_by_predicate(
            db_graph, predicate=is_saiyan, include_neighbors=False
        )
        node_ids = [n.get_id() for n in sub.get_nodes()]
        assert "goku" in node_ids and "vegeta" in node_ids
        assert "piccolo" not in node_ids

    def test_find_nodes_by_label(self, db_graph: MemoryGraph):
        assert find_nodes_by_label(db_graph, "Son Goku") == ["goku"]

    def test_find_nodes_by_metadata(self, db_graph: MemoryGraph):
        assert find_nodes_by_metadata(db_graph, "type", "Namekian") == ["piccolo"]

    def test_get_context_window(self, db_graph: MemoryGraph):
        """Test LLM context window generation."""
        context = get_context_window(
            db_graph, center="frieza", max_tokens=500, min_relevance=0.8
        )

        # If C++ binding returns None, it means the pybind11 binding is missing/misconfigured.
        assert (
            context is not None
        ), "C++ get_context_window returned None. Check pybind11 binding."
        assert context["center"] == "frieza"
        assert "nodes" in context and "edges" in context
        assert len(context["nodes"]) > 0


class TestTraversalHelpers:
    def test_find_entities_by_type(self, db_graph: MemoryGraph):
        saiyans = find_entities_by_type(db_graph, "Saiyan")
        assert len(saiyans) == 4
        assert "goku" in [n.get_id() for n in saiyans]

    def test_get_confidence_filtered_subgraph(self, db_graph: MemoryGraph):
        # Filter for high confidence (>= 0.96). Goku, Vegeta, Frieza pass.
        filtered = get_confidence_filtered_subgraph(
            db_graph, center="frieza", radius=2, min_confidence=0.96, min_weight=0.8
        )
        node_ids = [n.get_id() for n in filtered.get_nodes()]
        assert "frieza" in node_ids and "goku" in node_ids
        assert "piccolo" not in node_ids  # Confidence 0.90

    def test_find_shortest_path_with_details(self, db_graph: MemoryGraph):
        details = find_shortest_path_with_details(db_graph, "krillin", "vegeta")
        assert details is not None
        assert details["path"] == ["krillin", "goku", "vegeta"]
        assert len(details["nodes"]) == 3
        assert len(details["edges"]) == 2

    def test_find_shortest_path_with_details_no_path(self, db_graph: MemoryGraph):
        with pytest.raises(Exception):  # C++ throws NodeNotFoundError
            find_shortest_path_with_details(db_graph, "krillin", "nonexistent_node")
