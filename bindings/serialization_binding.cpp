// bindings/serialization_binding.cpp
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "memory_graph/utils/serialization.hpp"

#include "types_caster.hpp"

namespace py = pybind11;

void init_serialization(py::module_ &m) {
  // Serialization Options
  py::enum_<memory_graph::utils::CompressionType>(m, "CompressionType")
      .value("NONE", memory_graph::utils::CompressionType::NONE)
      .value("ZLIB", memory_graph::utils::CompressionType::ZLIB)
      .value("LZ4", memory_graph::utils::CompressionType::LZ4)
      .export_values();

  py::class_<memory_graph::utils::SerializationOptions>(m,
                                                        "SerializationOptions")
      .def(py::init<>())
      .def_readwrite(
          "include_metadata",
          &memory_graph::utils::SerializationOptions::include_metadata)
      .def_readwrite("include_edges",
                     &memory_graph::utils::SerializationOptions::include_edges)
      .def_readwrite("compression",
                     &memory_graph::utils::SerializationOptions::compression)
      .def_readwrite("version",
                     &memory_graph::utils::SerializationOptions::version);

  // Core Serialization
  m.def("to_binary", &memory_graph::utils::toBinary, py::arg("graph"),
        py::arg("options") = memory_graph::utils::SerializationOptions{},
        "Serialize a MemoryGraph to binary format");

  m.def("from_binary", &memory_graph::utils::fromBinary, py::arg("data"),
        "Deserialize a MemoryGraph from binary format");

  // Incremental Serialization (Deltas)
  m.def("compute_delta", &memory_graph::utils::computeDelta, py::arg("before"),
        py::arg("after"), "Compute the difference between two graphs");

  m.def("apply_delta", &memory_graph::utils::applyDelta, py::arg("graph"),
        py::arg("delta"), "Apply a delta to a graph");

  m.def("apply_delta_binary", &memory_graph::utils::applyDeltaBinary,
        py::arg("graph"), py::arg("delta_data"),
        "Apply a delta from binary format");

  // Compression Helpers
  m.def("compress", &memory_graph::utils::compress, py::arg("data"),
        py::arg("type") = memory_graph::utils::CompressionType::ZLIB,
        "Compress binary data");

  m.def("decompress", &memory_graph::utils::decompress, py::arg("data"),
        "Decompress binary data");

  m.def("compression_ratio", &memory_graph::utils::compressionRatio,
        py::arg("original"), py::arg("compressed"), "Get compression ratio");

  // Version Management
  m.def("get_version", &memory_graph::utils::getVersion, py::arg("data"),
        "Get the serialization version from binary data");

  m.def("is_valid_format", &memory_graph::utils::isValidFormat, py::arg("data"),
        "Check if binary data is valid for this version");
}
