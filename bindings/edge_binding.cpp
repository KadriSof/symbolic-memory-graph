#include "memory_graph/edge.hpp"
#include "types_caster.hpp"
#include <cstddef>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_set>

namespace py = pybind11;

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

void init_connections_types(py::module_ &m) {
  py::class_<std::unordered_set<std::string>>(m, "SymmetricConnections")
      .def(py::init<>())
      .def(py::init<const std::unordered_set<std::string> &>())
      .def("__len__", &std::unordered_set<std::string>::size)
      .def(
          "__contains__",
          [](const std::unordered_set<std::string> &set,
             const std::string &value) { return set.find(value) != set.end(); })
      .def("__repr__", [](const std::unordered_set<std::string> &set) {
        std::string result = "{";
        bool first = true;
        for (const auto &item : set) {
          if (!first)
            result += ", ";
          result += "'" + item + "'";
          first = false;
        }
        result += "}";
        return result;
      });

  py::class_<memory_graph::AsymmetricConnections>(m, "AsymmetricConnections")
      .def(py::init<const std::string &, const std::string &>())
      .def_property_readonly(
          "source",
          [](const memory_graph::AsymmetricConnections &conn) {
            return conn.first;
          })
      .def_property_readonly(
          "target",
          [](const memory_graph::AsymmetricConnections &conn) {
            return conn.second;
          })
      .def("__repr__",
           [](const memory_graph::AsymmetricConnections &conn) {
             return "AsymmetricConnections(source='" + conn.first +
                    "', target='" + conn.second + "')";
           })
      .def("__getitem__",
           [](const memory_graph::AsymmetricConnections &conn,
              size_t idx) -> std::string {
             if (idx == 0)
               return conn.first;
             if (idx == 1)
               return conn.second;
             throw py::index_error("Index out of range");
           })
      .def("__len__",
           [](const memory_graph::AsymmetricConnections &) { return 2; });
}

void init_edge(py::module_ &m) {
  py::class_<memory_graph::Edge>(m, "Edge")
      .def(py::init<const std::string &, const std::string &,
                    memory_graph::EdgeType, const memory_graph::Connections &,
                    float, const nlohmann::json &>(),
           py::arg("id"), py::arg("label"), py::arg("type"),
           py::arg("connections"), py::arg("weight") = 1.0f,
           py::arg_v("metadata", nlohmann::json::object(), "{}"))

      .def("get_id", &memory_graph::Edge::getId)
      .def("get_label", &memory_graph::Edge::getLabel)
      .def("get_type", &memory_graph::Edge::getType)
      .def("get_connections", &memory_graph::Edge::getConnections)
      .def("get_weight", &memory_graph::Edge::getWeight)
      .def("get_metadata", &memory_graph::Edge::getMetadata)
      .def("is_group_edge", &memory_graph::Edge::isGroupEdge)
      .def("set_label", &memory_graph::Edge::setLabel)
      .def("set_weight", &memory_graph::Edge::setWeight)
      .def("set_metadata", &memory_graph::Edge::setMetadata)
      .def("update_metadata", &memory_graph::Edge::updateMetadata)
      .def("to_json", &memory_graph::Edge::toJson)
      .def_static("from_json", &memory_graph::Edge::fromJson)
      .def("__repr__", [](const memory_graph::Edge &edge) {
        return "<Edge id='" + edge.getId() + "' label='" + edge.getLabel() +
               "'>";
      });
}
