#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "types_caster.hpp"

namespace py = pybind11;

// Forward declarations of all binding functions
void init_node(py::module_ &m);
void init_edge_types(py::module_ &m);
void init_connections_types(py::module_ &m);
void init_edge(py::module_ &m);
void init_memory_graph(py::module_ &m);
void init_exceptions(py::module_ &m);

PYBIND11_MODULE(memory_graph_core, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#else
  m.attr("__version__") = py::str("unknown");
#endif

  m.doc() = "MemoryGraph - C++ bindings for Python";

  init_exceptions(m);
  init_edge_types(m);
  init_connections_types(m);
  init_node(m);
  init_edge(m);
  init_memory_graph(m);
}
