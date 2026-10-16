#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "ext_interface/nanobind_registry_helpers.hpp"
#include "nanobind_array_utils.hpp"
#include "nanobind_dense_array_utils.hpp"
#include "multi_parameter_rank_invariant/function_rips.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpfrn {

using tensor_dtype = int32_t;
using indices_type = int32_t;

using multipers::nanobind_dense_utils::cast_vector_from_array;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::visit_simplextree_wrapper;
using multipers::nanobind_helpers::visit_const_simplextree_wrapper;
using multipers::nanobind_utils::owned_array;

using degree_rips_interface = Gudhi::multiparameter::function_rips::interface_multi;

template <typename Func>
decltype(auto) visit_simplextree(const nb::handle& simplextree, Func&& func) {
  if (!is_simplextree_object(simplextree)) {
    throw nb::type_error("Expected a multipers SimplexTreeMulti.");
  }
  return visit_const_simplextree_wrapper(simplextree, std::forward<Func>(func));
}

inline indices_type grid_rows(const nb::handle& simplextree) {
  return visit_simplextree(simplextree, [&]<typename Desc>(const auto& wrapper) -> indices_type {
    nb::ndarray<> axis = nb::cast<nb::ndarray<>>(wrapper.get_filtration_grid()[0]);
    return static_cast<indices_type>(axis.shape(0));
  });
}

inline indices_type num_parameters(const nb::handle& simplextree) {
  return visit_simplextree(simplextree, [&]<typename Desc>(const auto& wrapper) -> indices_type {
    return static_cast<indices_type>(wrapper.num_parameters());
  });
}

inline std::vector<indices_type> flatten_points(const std::vector<std::vector<indices_type>>& points,
                                                size_t num_columns) {
  std::vector<indices_type> flat;
  flat.reserve(points.size() * num_columns);
  for (const auto& row : points) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return flat;
}

}  // namespace mpfrn

NB_MODULE(_function_rips_nanobind, m) {
  m.def(
      "get_degree_rips",
      [](nb::object target,
         nb::ndarray<nb::numpy, const int8_t, nb::ndim<1>, nb::c_contig> gudhi_state,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> degrees) -> nb::object {
        auto degree_vector = mpfrn::cast_vector_from_array<int>(degrees);
        mpfrn::visit_simplextree_wrapper(target, [&]<typename Desc>(auto& wrapper) {
          using Interface = typename Desc::interface_type;
          if constexpr (std::is_same_v<Interface, mpfrn::degree_rips_interface>) {
            Gudhi::multiparameter::function_rips::get_degree_rips_st_python(
                reinterpret_cast<const char*>(gudhi_state.data()), gudhi_state.shape(0), wrapper, degree_vector);
          } else {
            throw nb::type_error("get_degree_rips expects the canonical degree-Rips SimplexTreeMulti target.");
          }
        });
        return target;
      },
      "target"_a,
      "gudhi_state"_a,
      "degrees"_a);
}
