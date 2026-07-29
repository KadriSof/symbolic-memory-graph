// bindings/traversal_binding.cpp
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "memory_graph/utils/traversal.hpp"

#include "types_caster.hpp"

namespace py = pybind11;

void init_traversal(py::module_ &m) {
  // Basic Traversals
  m.def("bfs", &memory_graph::utils::bfs, py::arg("graph"), py::arg("start"),
        py::arg("max_depth") = -1,
        "Breadth-First Search traversal starting from a node");

  m.def("dfs", &memory_graph::utils::dfs, py::arg("graph"), py::arg("start"),
        py::arg("max_depth") = -1,
        "Depth-First Search traversal starting from a node");

  // Path Finding
  m.def("shortest_path", &memory_graph::utils::shortestPath, py::arg("graph"),
        py::arg("from"), py::arg("to"),
        "Find the shortest path between two nodes (unweighted)");

  m.def("find_all_paths", &memory_graph::utils::findAllPaths, py::arg("graph"),
        py::arg("from"), py::arg("to"), py::arg("max_depth") = -1,
        "Find all paths between two nodes (up to maxDepth)");

  // Graph Properties
  m.def("has_cycle", &memory_graph::utils::hasCycle, py::arg("graph"),
        "Check if the graph contains a cycle");

  m.def("topological_sort", &memory_graph::utils::topologicalSort,
        py::arg("graph"),
        "Perform topological sort on a directed acyclic graph");

  m.def("is_connected", &memory_graph::utils::isConnected, py::arg("graph"),
        py::arg("start"),
        "Check if graph is connected from a given start node");

  // Subgraph Extraction
  m.def("subgraph", &memory_graph::utils::subgraph, py::arg("graph"),
        py::arg("center"), py::arg("radius"), py::arg("include_edges") = true,
        "Extract a subgraph centered at a node within a radius");

  m.def("subgraph_by_predicate", &memory_graph::utils::subgraphByPredicate,
        py::arg("graph"), py::arg("predicate"),
        py::arg("include_neighbors") = false,
        "Extract a subgraph containing all nodes matching a predicate");

  // Node Query Helpers
  m.def("find_nodes_by_label", &memory_graph::utils::findNodesByLabel,
        py::arg("graph"), py::arg("label"), "Find nodes by label");

  m.def("find_nodes_by_metadata", &memory_graph::utils::findNodesByMetadata,
        py::arg("graph"), py::arg("key"), py::arg("value"),
        "Find nodes by metadata key-value pair");

  // Context Window Utilities (for LLMs)
  m.def("get_context_window", &memory_graph::utils::getContextWindow,
        py::arg("graph"), py::arg("center"), py::arg("max_tokens") = 4096,
        py::arg("min_relevance") = 0.5f, "Get a context window for an LLM");
}
