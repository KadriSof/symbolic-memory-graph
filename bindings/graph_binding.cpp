#include "memory_graph/memory_graph.hpp"
#include "types_caster.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_memory_graph(py::module_ &m) {
  py::class_<memory_graph::MemoryGraph>(m, "MemoryGraph")
      .def(py::init<const nlohmann::json &>(),
           py::arg_v("metadata", nlohmann::json::object(), "{}"))

      .def("add_node", &memory_graph::MemoryGraph::addNode)
      .def("has_node", &memory_graph::MemoryGraph::hasNode)
      .def("get_node", &memory_graph::MemoryGraph::getNode,
           py::return_value_policy::reference_internal)
      .def("remove_node", &memory_graph::MemoryGraph::removeNode)
      .def("get_nodes", &memory_graph::MemoryGraph::getNodes)
      .def("add_edge", &memory_graph::MemoryGraph::addEdge)
      .def("add_group_edge", &memory_graph::MemoryGraph::addGroupEdge)
      .def("has_edge", &memory_graph::MemoryGraph::hasEdge)
      .def("get_edge", &memory_graph::MemoryGraph::getEdge,
           py::return_value_policy::reference_internal)
      .def("remove_edge", &memory_graph::MemoryGraph::removeEdge)
      .def("get_edges", &memory_graph::MemoryGraph::getEdges)
      .def("get_group_edges", &memory_graph::MemoryGraph::getGroupEdges)
      .def("get_group_members", &memory_graph::MemoryGraph::getGroupMembers)
      .def("get_neighbors", &memory_graph::MemoryGraph::getNeighbors)
      .def("query", &memory_graph::MemoryGraph::query)
      .def("to_json", &memory_graph::MemoryGraph::toJson)
      .def_static("from_json", &memory_graph::MemoryGraph::fromJson)
      .def("get_metadata", &memory_graph::MemoryGraph::getMetadata)
      .def("set_metadata", &memory_graph::MemoryGraph::setMetadata)
      .def("update_metadata", &memory_graph::MemoryGraph::updateMetadata)
      .def("__len__",
           [](const memory_graph::MemoryGraph &graph) {
             return graph.getNodes().size();
           })
      .def("__repr__", [](const memory_graph::MemoryGraph &graph) {
        return "<MemoryGraph nodes=" + std::to_string(graph.getNodes().size()) +
               " edges=" + std::to_string(graph.getEdges().size()) + ">";
      });
}
