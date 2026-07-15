#pragma once

#include <exception>
#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/detail/descr.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pyerrors.h>
#include <string>

namespace py = pybind11;

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
