/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2026/02 Hannah Schreiber: reorganization + small optimizations + documentation
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file module_helpers.h
 * @author David Loiseaux
 * @brief Contains the helper methods @ref Gudhi::multi_persistence::.
 */

#ifndef MP_MODULE_HELPERS_H_
#define MP_MODULE_HELPERS_H_

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <array>

#ifdef GUDHI_USE_TBB
#include <oneapi/tbb/parallel_for.h>
#endif

#include <gudhi/simple_mdspan.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Module.h>
#include <gudhi/Multi_persistence/summand_helpers.h>

namespace Gudhi {
namespace multi_persistence {

/**
 * @private
 */
template <typename T>
std::vector<T> _get_module_landscape_values(const Module<T> &mod,
                                            const std::vector<T> &x,
                                            typename Module<T>::Dimension dimension) {
  std::vector<T> values;
  values.reserve(mod.size());
  for (std::size_t i = 0; i < mod.size(); i++) {
    const auto &summand = mod.get_summand(i);
    if (summand.get_dimension() == dimension) values.push_back(get_summand_landscape_value(summand, x));
  }
  std::sort(values.begin(), values.end(), [](T x, T y) { return x > y; });
  return values;
}

// TODO: extend in higher resolution dimension
template <typename T>
std::vector<std::vector<T>> compute_module_landscape(const Module<T> &mod,
                                                     typename Module<T>::Dimension dimension,
                                                     unsigned int k,
                                                     const Box<T> &box,
                                                     const std::array<unsigned int, 2> &resolution) {
  std::vector<std::vector<T>> image;
  image.resize(resolution[0], std::vector<T>(resolution[1]));
  T stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / resolution[0];
  T stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / resolution[1];

  auto get_image_values = [&](unsigned int i) {
    return [&](unsigned int j) {
      auto landscape = _get_module_landscape_values(
          mod, {box.get_lower_corner()[0] + (stepX * i), box.get_lower_corner()[1] + (stepY * j)}, dimension);
      image[i][j] = k < landscape.size() ? landscape[k] : 0;
    };
  };

#ifdef GUDHI_USE_TBB
  tbb::parallel_for(
      0U, resolution[0], [&](unsigned int i) { tbb::parallel_for(0U, resolution[1], get_image_values(i)); });
#else
  for (unsigned int i = 0; i < resolution[0]; ++i) {
    auto get_image_values_at = get_image_values(i);
    for (unsigned int j = 0; j < resolution[1]; ++j) {
      get_image_values_at(j);
    }
  }
#endif

  return image;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> compute_set_of_module_landscapes(const Module<T> &mod,
                                                                          typename Module<T>::Dimension dimension,
                                                                          const std::vector<unsigned int> &ks,
                                                                          const Box<T> &box,
                                                                          const std::array<unsigned int, 2> &resolution,
                                                                          [[maybe_unused]] int n_jobs = 0) {
  std::vector<std::vector<std::vector<T>>> images(ks.size());
  for (auto &image : images) image.resize(resolution[0], std::vector<T>(resolution[1]));
  T stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / resolution[0];
  T stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / resolution[1];

  auto get_image_values = [&](unsigned int i) {
    return [&](unsigned int j) {
      std::vector<T> landscapes = _get_module_landscape_values(
          mod, {box.get_lower_corner()[0] + (stepX * i), box.get_lower_corner()[1] + (stepY * j)}, dimension);
      for (std::size_t k_idx = 0; k_idx < ks.size(); ++k_idx) {
        unsigned int k = ks[k_idx];
        images[k_idx][i][j] = k < landscapes.size() ? landscapes[k] : 0;
      }
    };
  };

#ifdef GUDHI_USE_TBB
  oneapi::tbb::task_arena arena(n_jobs);
  arena.execute([&] {
    tbb::parallel_for(
        0U, resolution[0], [&](unsigned int i) { tbb::parallel_for(0U, resolution[1], get_image_values(i)); });
  });
#else
  for (unsigned int i = 0; i < resolution[0]; ++i) {
    auto get_image_values_at = get_image_values(i);
    for (unsigned int j = 0; j < resolution[1]; ++j) {
      get_image_values_at(j);
    }
  }
#endif

  return images;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> compute_set_of_module_landscapes(const Module<T> &mod,
                                                                          typename Module<T>::Dimension dimension,
                                                                          const std::vector<unsigned int> &ks,
                                                                          const std::vector<std::vector<T>> &grid,
                                                                          int n_jobs = 0) {
  if (grid.size() != 2) throw std::invalid_argument("Grid must be 2D.");

  std::vector<std::vector<std::vector<T>>> images(ks.size());
  for (auto &image : images) image.resize(grid[0].size(), std::vector<T>(grid[1].size()));

  auto get_image_values = [&](std::size_t i) {
    return [&](std::size_t j) {
      std::vector<T> landscapes = _get_module_landscape_values(mod, {grid[0][i], grid[1][j]}, dimension);
      for (std::size_t k_idx = 0; k_idx < ks.size(); ++k_idx) {
        unsigned int k = ks[k_idx];
        images[k_idx][i][j] = k < landscapes.size() ? landscapes[k] : 0;
      }
    };
  };

#ifdef GUDHI_USE_TBB
  oneapi::tbb::task_arena arena(n_jobs);
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0), grid[0].size(), [&](std::size_t i) {
      tbb::parallel_for(std::size_t(0), grid[1].size(), get_image_values(i));
    });
  });
#else
  for (std::size_t i = 0; i < grid[0].size(); ++i) {
    auto get_image_values_at = get_image_values(i);
    for (std::size_t j = 0; j < grid[1].size(); ++j) {
      get_image_values_at(j);
    }
  }
#endif

  return images;
}

template <typename T, class MultiFiltrationValue>
std::vector<int> compute_module_euler_curve(const Module<T> &mod, const std::vector<MultiFiltrationValue> &points) {
  std::vector<int> eulerCurve(points.size());

  auto get_curve = [&](std::size_t i) {
    for (const auto &sum : mod) {
      if (sum.contains(points[i])) {
        int sign = sum.get_dimension() % 2 ? -1 : 1;
        eulerCurve[i] += sign;
      }
    }
  };

#ifdef GUDHI_USE_TBB
  tbb::parallel_for(std::size_t(0), eulerCurve.size(), get_curve);
#else
  for (std::size_t i = 0; i < eulerCurve.size(); ++i) {
    get_curve(i);
  }
#endif

  return eulerCurve;
}

// TODO: change data_ptr to output when changing all the return types.
template <typename T>
void compute_module_distances_to(const Module<T> &mod,
                                 T *data_ptr,
                                 const std::vector<std::vector<T>> &pts,
                                 bool negative,
                                 int n_jobs) {
  Gudhi::Simple_mdspan data(data_ptr, pts.size(), mod.size());

  auto get_distances_of_point = [&](std::size_t i) {
    for (std::size_t j = 0; j < data.extent(1); ++j) {
      data(i, j) = compute_summand_distance_to(mod.get_summand(j), pts[i], negative);
    }
  };

#ifdef GUDHI_USE_TBB
  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] { tbb::parallel_for(std::size_t(0), pts.size(), get_distances_of_point); });
#else
  for (std::size_t i = 0; i < pts.size(); ++i) {
    get_distances_of_point(i);
  }
#endif
}

template <typename T>
std::vector<std::vector<std::vector<std::size_t>>> compute_module_lower_and_upper_generators_of(
    const Module<T> &mod,
    const std::vector<std::vector<T>> &pts,
    bool full,
    int n_jobs) {
  std::vector<std::vector<std::vector<std::size_t>>> out(pts.size(), std::vector<std::vector<std::size_t>>(mod.size()));

  auto get_generators = [&](std::size_t i) {
    return [&](std::size_t j) {
      out[i][j] = compute_summand_lower_and_upper_generator_of(mod.get_summand(j), pts[i], full);
    };
  };

#ifdef GUDHI_USE_TBB
  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0), pts.size(), [&](std::size_t i) {
      tbb::parallel_for(std::size_t(0), mod.size(), get_generators(i));
    });
  });
#else
  for (std::size_t i = 0; i < pts.size(); ++i) {
    auto get_generators_at = get_generators(i);
    for (std::size_t j = 0; j < mod.size(); ++j) {
      get_generators_at(j);
    }
  }
#endif

  return out;
}

/**
 * @private
 */
template <typename value_type>
inline std::vector<value_type> Module<value_type>::_compute_module_pixels_of_degree(
    const typename module_type::iterator start,
    const typename module_type::iterator end,
    const value_type delta,
    const value_type p,
    const bool normalize,
    const Box<value_type> &box,
    const std::vector<std::vector<value_type>> &coordinates,
    const int n_jobs) {
  unsigned int num_pixels = coordinates.size();
  std::vector<value_type> out(num_pixels);
  value_type moduleWeight = 0;
  {  // for Timer
    Debug::Timer timer("Computing module weight ...", verbose);
    for (auto it = start; it != end; it++)  //  precomputes interleaving restricted to box for all summands.
      it->get_interleaving(box);
    if (p == 0) {
      // #pragma omp parallel for reduction(+ : moduleWeight)
      for (auto it = start; it != end; it++) {
        moduleWeight += it->get_interleaving() > 0;
      }
    } else if (p != inf) {
      // #pragma omp parallel for reduction(+ : moduleWeight)
      for (auto it = start; it != end; it++) {
        // /!\ TODO deal with inf summands (for the moment,  depends on the box
        // ...)
        if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
          moduleWeight += std::pow(it->get_interleaving(), p);
      }
    } else {
      // #pragma omp parallel for reduction(std::max : moduleWeight)
      for (auto it = start; it != end; it++) {
        if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
          moduleWeight = std::max(moduleWeight, it->get_interleaving());
      }
    }
  }  // Timer death
  if (verbose) std::cout << "Module " << start->get_dimension() << " has weight : " << moduleWeight << "\n";
  if (!moduleWeight) return out;

  if constexpr (Debug::debug)
    if (moduleWeight < 0) {
      if constexpr (Debug::debug) std::cout << "!! Negative weight !!" << std::endl;
      // 		image.clear();
      return {};
    }

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    tbb::parallel_for(0u, num_pixels, [&](unsigned int i) {
      out[i] = _get_pixel_value(start, end, coordinates[i], delta, p, normalize, moduleWeight);
    });
  });
  return out;
}

template <typename T>
std::vector<std::vector<T>> compute_module_pixels(const Module<T> &mod,
                                                  const std::vector<std::vector<T>> &coordinates,
                                                  const std::vector<typename Module<T>::Dimension> &dimensions,
                                                  const Box<T> &box = {},
                                                  T delta = 0.1,
                                                  T p = 1,
                                                  bool normalize = true,
                                                  int n_jobs = 0) {
  auto num_degrees = dimensions.size();
  auto num_pts = coordinates.size();
  std::vector<std::vector<T>> out(num_degrees, std::vector<T>(num_pts));

  auto start = mod.begin();
  auto end = mod.begin();
  for (std::size_t degree_idx = 0; degree_idx < num_degrees; degree_idx++) {
    auto d = dimensions[degree_idx];
    start = end;
    while (start != mod.end() && start->get_dimension() != d) start++;
    if (start == mod.end()) break;
    end = start;
    while (end != mod.end() && end->get_dimension() == d) end++;
    out[degree_idx] = _compute_module_pixels_of_degree(start, end, delta, p, normalize, box, coordinates, n_jobs);
  }
  return out;
}

std::vector<std::vector<std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>>> to_idx(
    const Module &module,
    const std::vector<std::vector<T>> &grid);

std::vector<std::vector<std::vector<int>>> to_flat_idx(const Module &module, const std::vector<std::vector<T>> &grid);

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_MODULE_HELPERS_H_
