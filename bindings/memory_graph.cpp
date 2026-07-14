// bindings/memory_graph.cpp
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/node.hpp"

#include "nlohmann/json.hpp"
#include <nlohmann/json_fwd.hpp>

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <unordered_set>
#include <utility>

namespace py = pybind11;

// Forward declaration of the binding function
void init_node(py::module_ &m);
void init_edge(py::module_ &m);
void init_memory_graph(py::module_ &m);

PYBIND11_MODULE(memory_graph_core, m) {
  m.doc() = "MemoryGraph - C++ bindings for Python";

  // Register exception translations
  py::register_exception<memory_graph::NodeNotFoundError>(m,
                                                          "NodeNotFoundError");
  py::register_exception<memory_graph::EdgeNotFoundError>(m,
                                                          "EdgeNotFoundError");
  py::register_exception<memory_graph::DuplicateIdError>(m, "DuplicateIdError");
  py::register_exception<memory_graph::InvalidConnectionError>(
      m, "InvalidConnectionError");

  // Initialize each class binding
  init_node(m);
  init_edge(m);
  init_memory_graph(m);
}

// Node binding
void init_node(py::module_ &m) {
  py::class_<memory_graph::Node>(m, "Node")
      .def(py::init<const std::string &, const std::string &,
                    const nlohmann::json &>(),
           py::arg("id"), py::arg("label"),
           py::arg("metadata") = nlohmann::json::object())

      // Getters
      .def("get_id", &memory_graph::Node::getId)
      .def("get_label", &memory_graph::Node::getLabel)
      .def("get_metadata", &memory_graph::Node::getMetadata)
      .def("get_connections", &memory_graph::Node::getConnections)

      // Setters
      .def("set_label", &memory_graph::Node::setLabel)
      .def("set_metadata", &memory_graph::Node::setMetadata)
      .def("update_metadata", &memory_graph::Node::updateMetadata)

      // Serialization
      .def("to_json", &memory_graph::Node::toJson)
      .def_static("from_json", &memory_graph::Node::fromJson)

      // Python special methods
      .def("__repr__",
           [](const memory_graph::Node &node) {
             return "<Node id='" + node.getId() + "' label='" +
                    node.getLabel() + "'>";
           })
      .def("__str__", [](const memory_graph::Node &node) {
        return "Node(id=" + node.getId() + ", label=" + node.getLabel() + ")";
      });
}

// Edge type enum binding
void init_edge_types(py::module_ &m) {
  py::enum_<memory_graph::EdgeType>(m, "EdgeType")
      .value("SYMMETRIC", memory_graph::EdgeType::SYMMETRIC)
      .value("ASYMMETRIC", memory_graph::EdgeType::ASYMMETRIC)
      .export_values()
      .def("__repr__", [](memory_graph::EdgeType type) {
        return type == memory_graph::EdgeType::SYMMETRIC
                   ? "EdgeType.SYMMETRIC"
                   : "EdgeType.ASYMMETRIC";
      });
}

// Connections types (std::varian handling)
void init_connections_types(py::module_ &m) {
  // SymmetricConnections in an unordered_set<string>
  py::class_<std::unordered_set<std::string>>(m, "SymmetricConnections")
      .def(py::init<>())
      .def(py::init<const std::unordered_set<std::string> &>())
      .def("__len__", &std::unordered_set<std::string>::size)
      .def("__contains__", [](const std::unordered_set<std::string> &set,
                              const std::string &value) {
        return set.find(value) != set.end();
      });

  py::class_<std::pair<std::string, std::string>>(m, "SymmetricConnections")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &>())
      .def_readwrite("first", &std::pair<std::string, std::string>::first)
      .def_readwrite("second", &std::pair<std::string, std::string>::second);
}

// Edge binding
void init_edge(py::module_ &m) {
  py::class_<memory_graph::Edge>(m, "Edge")
      .def(py::init<const std::string &, const std::string &,
                    memory_graph::EdgeType, const memory_graph::Connections &,
                    float, const nlohmann::json &>(),
           py::arg("id"), py::arg("label"), py::arg("type"),
           py::arg("connections"), py::arg("weight") = 1.0f,
           py::arg("metadata") = nlohmann::json::object())

      // Getters
      .def("get_id", &memory_graph::Edge::getId)
      .def("get_label", &memory_graph::Edge::getLabel)
      .def("get_type", &memory_graph::Edge::getType)
      .def("get_connections", &memory_graph::Edge::getConnections)
      .def("get_weight", &memory_graph::Edge::getWeight)
      .def("get_metadata", &memory_graph::Edge::getMetadata)
      .def("is_group_edge", &memory_graph::Edge::isGroupEdge)

      // Setters
      .def("set_label", &memory_graph::Edge::setLabel)
      .def("set_weight", &memory_graph::Edge::setWeight)
      .def("set_metadata", &memory_graph::Edge::setMetadata)
      .def("update_metadata", &memory_graph::Edge::updateMetadata)

      // Serialization
      .def("to_json", &memory_graph::Edge::toJson)
      .def_static("from_json", &memory_graph::Edge::fromJson)

      // Python special methods
      .def("__repr__", [](const memory_graph::Edge &edge) {
        return "<Edge id='" + edge.getId() + "' label='" + edge.getLabel() +
               "'>";
      });
}

// MemoryGraph binding
void init_memory_graph(py::module_ &m) {
  py::class_<memory_graph::MemoryGraph>(m, "MemoryGraph")
      .def(py::init<const nlohmann::json &>(),
           py::arg("metadata") = nlohmann::json::object())

      // Node operations
      .def("add_node", &memory_graph::MemoryGraph::addNode)
      .def("has_node", &memory_graph::MemoryGraph::hasNode)
      .def("get_node", &memory_graph::MemoryGraph::getNode,
           py::return_value_policy::reference_internal)
      .def("remove_node", &memory_graph::MemoryGraph::removeNode)
      .def("get_nodes", &memory_graph::MemoryGraph::getNodes)

      // Edge operations
      .def("add_edge", &memory_graph::MemoryGraph::addEdge)
      .def("add_group_edge", &memory_graph::MemoryGraph::addGroupEdge)
      .def("has_edge", &memory_graph::MemoryGraph::hasEdge)
      .def("get_edge", &memory_graph::MemoryGraph::getEdge,
           py::return_value_policy::reference_internal)
      .def("remove_edge", &memory_graph::MemoryGraph::removeEdge)
      .def("get_edges", &memory_graph::MemoryGraph::getEdges)
      .def("get_group_edges", &memory_graph::MemoryGraph::getGroupEdges)
      .def("get_group_members", &memory_graph::MemoryGraph::getGroupMembers)

      // Graph operations
      .def("get_neighbors", &memory_graph::MemoryGraph::getNeighbors)
      .def("query", &memory_graph::MemoryGraph::query)

      // Serialization
      .def("to_json", &memory_graph::MemoryGraph::toJson)
      .def_static("from_json", &memory_graph::MemoryGraph::fromJson)

      // Metadata
      .def("get_metadata", &memory_graph::MemoryGraph::getMetadata)
      .def("set_metadata", &memory_graph::MemoryGraph::setMetadata)
      .def("update_metadata", &memory_graph::MemoryGraph::updateMetadata)

      // Python special methods
      .def("__len__",
           [](const memory_graph::MemoryGraph &graph) {
             return graph.getNodes().size();
           })
      .def("__repr__", [](const memory_graph::MemoryGraph &graph) {
        return "<MemoryGraph nodes=" + std::to_string(graph.getNodes().size()) +
               " edges=" + std::to_string(graph.getEdges().size()) + ">";
      });
}
