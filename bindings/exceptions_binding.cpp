#include "memory_graph/exceptions.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_exceptions(py::module_ &m) {
  py::register_exception<memory_graph::NodeNotFoundError>(m,
                                                          "NodeNotFoundError");
  py::register_exception<memory_graph::EdgeNotFoundError>(m,
                                                          "EdgeNotFoundError");
  py::register_exception<memory_graph::DuplicateIdError>(m, "DuplicateIdError");
  py::register_exception<memory_graph::InvalidConnectionError>(
      m, "InvalidConnectionError");
}
