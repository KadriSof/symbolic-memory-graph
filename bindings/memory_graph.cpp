// bindings/memory_graph.cpp
#include "memory_graph/memory_graph.hpp"
#include "memory_graph/edge.hpp"
#include "memory_graph/exceptions.hpp"
#include "memory_graph/node.hpp"
#include "nlohmann/json.hpp"

#include <cstddef>
#include <exception>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/detail/descr.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <pyerrors.h>
#include <string>
#include <unordered_set>
#include <utility>

namespace py = pybind11;

// Custom type caster for nlohmann::json
namespace pybind11 {
namespace detail {

template <> struct type_caster<nlohmann::json> {
public:
  PYBIND11_TYPE_CASTER(nlohmann::json, _("json"));

  // Python -> C++ conversion
  bool load(handle src, bool convert) {
    try {
      py::object obj = py::reinterpret_borrow<py::object>(src);

      if (obj.is_none()) {
        value = nlohmann::json::object();
        return true;
      }

      if (py::isinstance<py::dict>(obj)) {
        py::dict dict = obj.cast<py::dict>();
        for (auto item : dict) {
          std::string key = py::str(item.first);
          nlohmann::json val = item.second.cast<nlohmann::json>();
          value[key] = val;
        }
        return true;
      }

      if (py::isinstance<py::list>(obj)) {
        py::list list = obj.cast<py::list>();
        for (auto item : list) {
          nlohmann::json val = item.cast<nlohmann::json>();
          value.push_back(val);
        }
        return true;
      }

      if (py::isinstance<py::str>(obj)) {
        value = py::str(obj).cast<std::string>();
        return true;
      }

      if (py::isinstance<py::int_>(obj)) {
        value = py::int_(obj).cast<long long>();
        return true;
      }

      if (py::isinstance<py::float_>(obj)) {
        value = py::float_(obj).cast<double>();
        return true;
      }

      if (py::isinstance<py::bool_>(obj)) {
        value = py::bool_(obj).cast<bool>();
        return true;
      }

      std::string json_str = py::str(obj).cast<std::string>();
      value = nlohmann::json::parse(json_str);
      return true;
    } catch (const std::exception &) {
      return false;
    }
  }

  // C++ -> Python conversion
  static handle cast(const nlohmann::json &src, return_value_policy policy,
                     handle parent) {
    try {
      if (src.is_null()) {
        return py::none().release();
      }
      if (src.is_boolean()) {
        return py::bool_(src.get<bool>()).release();
      }
      if (src.is_number_integer()) {
        return py::int_(src.get<long long>()).release();
      }
      if (src.is_number_unsigned()) {
        return py::int_(src.get<unsigned long long>()).release();
      }
      if (src.is_number_float()) {
        return py::float_(src.get<double>()).release();
      }
      if (src.is_string()) {
        return py::str(src.get<std::string>()).release();
      }
      if (src.is_array()) {
        py::list list;
        for (const auto &item : src) {
          list.append(cast(item, policy, parent));
        }
        return list.release();
      }
      if (src.is_object()) {
        py::dict dict;
        for (auto it = src.begin(); it != src.end(); ++it) {
          dict[py::str(it.key())] = cast(it.value(), policy, parent);
        }
        return dict.release();
      }
      return py::str(src.dump()).release();
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return handle();
    }
  }
};

} // namespace detail
} // namespace pybind11

// Forward declarations
void init_node(py::module_ &m);
void init_edge_types(py::module_ &m);
void init_connections_types(py::module_ &m);
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
  init_edge_types(m);
  init_connections_types(m);
  init_node(m);
  init_edge(m);
  init_memory_graph(m);
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

// Connections types
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

// Node binding
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

// Edge binding
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

// MemoryGraph binding
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
