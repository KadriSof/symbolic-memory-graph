#include "memory_graph/node.hpp"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "types_caster.hpp"

namespace py = pybind11;

void init_node(py::module_ &m) {
  py::class_<memory_graph::Node>(m, "Node")
      .def(py::init<const std::string &, const std::string &,
                    const nlohmann::json &>(),
           py::arg("id"), py::arg("label"),
           py::arg_v("metadata", nlohmann::json::object(), "{}"))

      .def("get_id", &memory_graph::Node::getId)
      .def("get_label", &memory_graph::Node::getLabel)
      .def("get_metadata", &memory_graph::Node::getMetadata)
      .def("get_connections", &memory_graph::Node::getConnections)
      .def("set_label", &memory_graph::Node::setLabel)
      .def("set_metadata", &memory_graph::Node::setMetadata)
      .def("update_metadata", &memory_graph::Node::updateMetadata)
      .def("to_json", &memory_graph::Node::toJson)
      .def_static("from_json", &memory_graph::Node::fromJson)
      .def("__repr__", [](const memory_graph::Node &node) {
        return "<Node id='" + node.getId() + "' label='" + node.getLabel() +
               "'>";
      });
}
