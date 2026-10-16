#pragma once

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "../gudhi/Multi_simplex_tree_interface.h"
#include "../gudhi/Persistence_slices_interface.h"

namespace Gudhi {
namespace multiparameter {
namespace function_rips {

using value_type = typename multi_persistence::Simplex_tree_std::Filtration_value;
using flat_multifiltration = multipers::tmp_interface::Degree_rips_bifiltration<value_type>;
using flat_multi_st = Gudhi::multi_persistence::Multi_simplex_tree_interface<flat_multifiltration>;
using mult_opt = Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<flat_multifiltration>;
using interface_multi = Gudhi::multi_persistence::Multi_simplex_tree_interface<flat_multifiltration>;

// Takes a standard simplextree and turn it into a simplextreemulti, whose first
// axis is the rips, and the others are the filtrations of the node at each
// degree in degrees Assumes that the degrees are sorted, and unique
// also return max_degree,filtration_values
inline flat_multi_st::Base get_degree_filtrations(multi_persistence::Simplex_tree_std &st, const std::vector<int> &degrees) {
  constexpr const bool verbose = false;
  using filtration_lists = std::vector<std::vector<value_type>>;

  if (st.dimension() != 1)
    throw std::invalid_argument("get_degree_filtrations only works for 1-dimensional simplex trees.");

  unsigned int num_degrees = degrees.size();
  // puts the st filtration in axis 0 + fitrations for each degrees afterward
  flat_multifiltration default_f(static_cast<int>(num_degrees), 0);
  flat_multi_st::Base st_multi = Gudhi::multi_persistence::make_multi_dimensional<mult_opt>(st, default_f, 0);

  // preprocess
  filtration_lists edge_filtration_of_nodes(st.num_vertices());
  std::vector<value_type> vertex_filtration(st.num_vertices(), 0);
  for (auto sh : st.complex_simplex_range()) {
    if (st.dimension(sh) == 0) {
      vertex_filtration[*st.simplex_vertex_range(sh).begin()] = st.filtration(sh);
      continue;
    }
    value_type filtration = st.filtration(sh);
    for (auto node : st.simplex_vertex_range(sh)) {
      edge_filtration_of_nodes[node].push_back(filtration);
    }
  }

  tbb::parallel_for(static_cast<size_t>(0u), st.num_vertices(), [&](size_t i) {
    auto &filtrations = edge_filtration_of_nodes[i];
    tbb::parallel_sort(filtrations.begin(), filtrations.end());
    unsigned int node_degree = filtrations.size();
    std::vector<value_type> requested_filtrations(num_degrees);
    for (unsigned int degree_index = 0; degree_index < num_degrees; degree_index++) {
      const int requested_degree = degrees[degree_index];
      // `degrees` is interpreted as the vertex-degree threshold: degree=0
      // keeps all vertices, degree=1 needs one incident edge, etc.
      if (requested_degree == 0) {
        requested_filtrations[degree_index] = vertex_filtration[i];
      } else if (requested_degree > 0 && requested_degree <= static_cast<int>(node_degree)) {
        requested_filtrations[degree_index] = filtrations[requested_degree - 1];
      } else {
        requested_filtrations[degree_index] = std::numeric_limits<value_type>::infinity();
      }
      if constexpr (verbose) std::cout << requested_filtrations[degree_index] << " ";
    }
    filtrations = std::move(requested_filtrations);
    std::reverse(filtrations.begin(),
                 filtrations.end());  // degree is in opposite direction
  });

  // fills the degree_rips simplextree with lower star
  auto sh_standard = st.complex_simplex_range().begin();  // waiting for c++23 & zip to remove this garbage
  auto _end = st.complex_simplex_range().end();
  auto sh_multi = st_multi.complex_simplex_range().begin();
  for (; sh_standard != _end; ++sh_multi, ++sh_standard) {
    // only fill using t
    // will be filled afterward
    if (st.dimension(*sh_standard) == 0) continue;
    // dimension is 1 by assumption. fill using the node + rips value
    value_type edge_filtration = st.filtration(*sh_standard);
    // the filtration vector to fill

    if constexpr (verbose) {
      std::cout << "\nSetting filtration of edge [";
      for (int node : st.simplex_vertex_range(*sh_standard)) {
        std::cout << std::to_string(node) << ",";
      }
      std::cout << "]  to : ";
    }
    auto &to_update = st_multi.get_filtration_value(*sh_multi);
    to_update.set_num_generators(num_degrees);
    for (auto degree_index = 0u; degree_index < num_degrees; degree_index++) {
      value_type edge_filtration_of_degree = edge_filtration;  // copy as we do the max with edges of degree index
      for (int node : st.simplex_vertex_range(*sh_standard)) {
        edge_filtration_of_degree = std::max(edge_filtration_of_degree, edge_filtration_of_nodes[node][degree_index]);
      }

      to_update(degree_index, 0) = edge_filtration_of_degree;
    }

    if constexpr (verbose) {
      for (auto k = 0u; k < num_degrees; k++) std::cout << " " << std::to_string(to_update(k, 0));
    }
    if constexpr (verbose) std::cout << "\n" << st_multi.get_filtration_value(*sh_multi) << std::endl;
  }

  // fills the dimension 0 simplices
  for (auto vertex : st_multi.complex_vertex_range()) {  // should be in increasing order
    auto &vertex_filtration = st_multi.get_filtration_value(st_multi.find({vertex}));
    if constexpr (verbose) {
      std::cout << "Setting filtration of node " << vertex << " to ";
      for (auto degree_index = 0u; degree_index < num_degrees; degree_index++) {
        std::cout << edge_filtration_of_nodes[vertex][degree_index] << " ";
      }
      std::cout << "\n";
    }

    if (edge_filtration_of_nodes[vertex].size() > num_degrees)
      throw std::overflow_error("Edge filtration of node of bad size");

    vertex_filtration.set_num_generators(num_degrees);
    for (auto k = 0u; k < num_degrees; k++) {
      vertex_filtration(k, 0) = edge_filtration_of_nodes[vertex][k];
    }
    if constexpr (verbose) {
      std::cout << "Filtration of node " << std::to_string(vertex) << " :\n" << vertex_filtration << "\n";
    }
  }
  if constexpr (verbose) std::cout << std::endl;

  return st_multi;
}


// python interface

template <typename simplex_tree_type>
inline void get_degree_rips_st_python(const char *buffer_start,
                                      const std::size_t buffer_size,
                                      simplex_tree_type &st_multi_python_container,
                                      const std::vector<int> &degrees) {
  flat_multi_st::Base st_multi;
  {
    nanobind::gil_scoped_release release;
    multi_persistence::Simplex_tree_std st_std;
    st_std.deserialize(buffer_start, buffer_size);
    st_multi = get_degree_filtrations(st_std, degrees);
  }
  st_multi_python_container = std::move(st_multi);
}





}  // namespace function_rips
}  // namespace multiparameter
}  // namespace Gudhi
