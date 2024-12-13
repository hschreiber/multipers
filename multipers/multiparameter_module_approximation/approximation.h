/*    This file is part of the MMA Library - https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT.
 *    See file LICENSE for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence class, renaming and cleanup.
 *      - 2022/05 Hannah Schreiber: Addition of Summand class and Module class.
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file approximation.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the functions related to the approximation of n-modules.
 */

#ifndef MMA_APPROXIMATION_H_INCLUDED
#define MMA_APPROXIMATION_H_INCLUDED

#include <cmath>     //std::max, std::min
#include <iostream>  //std::cout
#include <vector>

#include "debug.h"
#include "module.h"
#include "utilities.h"

#include <Persistence_slices_interface.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Line.h>

namespace Gudhi::multiparameter::mma {

using Debug::Timer;
using Gudhi::multi_persistence::Box;
using Gudhi::multi_persistence::Line;

template <typename value_type>
void threshold_filters_list(std::vector<value_type> &filtersList, const Box<value_type> &box);

template <typename filtration_type, typename value_type>
inline void threshold_filters_list(std::vector<filtration_type> &filtersList, const Box<value_type> &box) {
  return;
  for (unsigned int i = 0; i < filtersList.size(); i++) {
    for (value_type &value : filtersList[i]) {
      value = std::min(std::max(value, box.get_lower_corner()[i]), box.get_upper_corner()[i]);
    }
  }
}

template <class Filtration_value, int axis = 0, bool sign = true>
class LineIterator {
 public:
  using value_type = typename Filtration_value::value_type;
  LineIterator(const Filtration_value &basepoint,
               const Filtration_value &direction,
               value_type precision,
               int num_iterations)
      : precision(precision), remaining_iterations(num_iterations), current_line(std::move(basepoint), direction) {};

  inline LineIterator<Filtration_value, axis, sign> &operator++() {
    //
    auto &basepoint = current_line.base_point();
    if (this->is_finished()) return *this;
    // If we didn't reached the end, go to the next line
    basepoint[axis] += sign ? precision : -precision;
    --remaining_iterations;
    return *this;
  }

  inline const Line<value_type> &operator*() const { return current_line; }

  inline LineIterator<Filtration_value, axis, sign> &next(std::size_t i) {
    auto &basepoint = current_line.base_point();
    if (this->is_finished()) return *this;
    // If we didn't reached the end, go to the next line
    basepoint[i] += sign ? precision : -precision;
    --remaining_iterations;
    return *this;
  }

  inline bool is_finished() const { return remaining_iterations <= 0; }

 private:
  const value_type precision;
  int remaining_iterations;
  Line<value_type> current_line;
};

template <class Filtration_value, int axis_ = 0, bool sign = true, class Module, class Slicer>
inline void __add_vineyard_trajectory_to_module(Module &module,
                                                Slicer &&slicer,
                                                LineIterator<Filtration_value, axis_, sign> &line_iterator,
                                                const bool threshold,
                                                int axis = 0) {
  static_assert(
      std::is_same<typename Filtration_value::value_type, typename Slicer::Filtration_value::value_type>::value);
  using value_type = typename Filtration_value::value_type;
  // Line iterator should be on the biggest axis
  const bool verbose = false;
  const bool verbose2 = false;
  while (!line_iterator.is_finished()) {
    const Line<value_type> &new_line = (axis_ >= 0) ? *(++line_iterator) : *line_iterator.next(axis);
    // if constexpr (axis_ >= 0) {
    //   new_line = *(++line_iterator); // first line is always a persistence
    // } else {
    //   new_line = *line_iterator.next(axis);
    // }
    // copy, no need to add it
    if constexpr (verbose) std::cout << "----------------------------------------------" << std::endl;
    if constexpr (verbose) std::cout << "Line basepoint " << new_line.base_point() << std::endl;
    slicer.push_to(new_line);

    slicer.vineyard_update();
    if constexpr (verbose2) std::cout << slicer << std::endl;
    const auto &diagram = slicer.get_flat_nodim_barcode();
    module.add_barcode(new_line, std::move(diagram), threshold);
  };
  // std::cout << "out 9" << std::endl;
  // for (const auto& s : module)
  //   for (auto g : s.get_birth_list()) std::cout << g << std::endl;
  // std::cout << std::endl;
};

template <class Module, class Filtration_value, typename value_type, class Slicer = SimplicialVineMatrixTruc<>>
void _rec_mma(Module &module,
              Filtration_value &basepoint,
              const std::vector<int> &grid_size,
              int dim_to_iterate,
              Slicer &&current_persistence,
              const value_type precision,
              bool threshold) {
  if (dim_to_iterate <= 0) {
    LineIterator<Filtration_value, 0> line_iterator(std::move(basepoint), precision, grid_size[0]);
    __add_vineyard_trajectory_to_module<Filtration_value, 0, true, Module, Slicer>(
        module, std::move(current_persistence), line_iterator, threshold);
  //   std::cout << "out 7" << std::endl;
  // for (const auto& s : module)
  //   for (auto g : s.get_birth_list()) std::cout << g << std::endl;
  // std::cout << std::endl;
    return;
  }
  Slicer pers_copy;
  Filtration_value basepoint_copy;
  for (int i = 0; i < grid_size[dim_to_iterate]; ++i) {
    // TODO : multithread, but needs matrix to be thread safe + put mutex on
    // module
    pers_copy = current_persistence;
    basepoint_copy = basepoint;
    _rec_mma(module, basepoint_copy, grid_size, dim_to_iterate - 1, pers_copy, precision, threshold);
    basepoint[dim_to_iterate] += precision;
    // current_persistence.push_to(Line(basepoint));
    // current_persistence.vineyard_update();
  }
  // std::cout << "out 8" << std::endl;
  // for (const auto& s : module)
  //   for (auto g : s.get_birth_list()) std::cout << g << std::endl;
  // std::cout << std::endl;
}

template <int axis, class Module, class Filtration_value, typename value_type, class Slicer>
void _rec_mma2(Module &module,
               Filtration_value &&basepoint,
               const Filtration_value &direction,
               const std::vector<int> &grid_size,
               const std::vector<bool> &signs,
               int dim_to_iterate,
               Slicer &&current_persistence,
               const value_type precision,
               bool threshold) {
  static_assert(std::is_same<typename Filtration_value::value_type, typename Slicer::value_type>::value);

  if (dim_to_iterate <= axis) {
    if (signs[axis]) {
      LineIterator<Filtration_value, axis, true> line_iterator(
          std::move(basepoint), direction, precision, grid_size[axis]);
      __add_vineyard_trajectory_to_module<Filtration_value, axis, true, Module, Slicer>(
          module, std::move(current_persistence), line_iterator, threshold);
    } else {
      LineIterator<Filtration_value, axis, false> line_iterator(
          std::move(basepoint), direction, precision, grid_size[axis]);
      __add_vineyard_trajectory_to_module<Filtration_value, axis, false, Module, Slicer>(
          module, std::move(current_persistence), line_iterator, threshold);
    }
// std::cout << "out 4" << std::endl;
//   for (const auto& s : module)
//     for (auto g : s.get_birth_list()) std::cout << g << std::endl;
//   std::cout << std::endl;
    return;
  }
  if (grid_size[dim_to_iterate] == 0) {
    // no need to copy basepoint, we just skip the dim here
    _rec_mma2<axis, Module, Filtration_value, value_type, Slicer>(module,
                                                                  std::move(basepoint),
                                                                  direction,
                                                                  grid_size,
                                                                  signs,
                                                                  dim_to_iterate - 1,
                                                                  std::move(current_persistence),
                                                                  precision,
                                                                  threshold);
  //   std::cout << "out 5" << std::endl;
  // for (const auto& s : module)
  //   for (auto g : s.get_birth_list()) std::cout << g << std::endl;
  // std::cout << std::endl;
    return;
  }
  for (int i = 0; i < grid_size[dim_to_iterate]; ++i) {
    // TODO : multithread, but needs matrix to be thread safe + put mutex on
    // module
    _rec_mma2<axis, Module, Filtration_value, value_type, typename Slicer::ThreadSafe>(module,
                                                                                       Filtration_value(basepoint),
                                                                                       direction,
                                                                                       grid_size,
                                                                                       signs,
                                                                                       dim_to_iterate - 1,
                                                                                       current_persistence.weak_copy(),
                                                                                       precision,
                                                                                       threshold);
    basepoint[dim_to_iterate] += signs[dim_to_iterate] ? precision : -precision;
    // current_persistence.push_to(Line(basepoint));
    // current_persistence.vineyard_update();
  }
  // std::cout << "out 6" << std::endl;
  // for (const auto& s : module)
  //   for (auto g : s.get_birth_list()) std::cout << g << std::endl;
  // std::cout << std::endl;
}

template <class Module, class Slicer, typename value_type>
Module _multiparameter_module_approximation(
    Slicer &slicer,
    const Gudhi::multi_filtration::One_critical_filtration<value_type> &direction,
    const value_type precision,
    Box<value_type> &box,
    const bool threshold,
    const bool complete,
    const bool verbose) {
  static_assert(std::is_same<typename Slicer::Filtration_value::value_type,
                             value_type>::value);  // Value type can be exposed to python interface.
  if (verbose) std::cout << "Starting Module Approximation" << std::endl;
  /* using Filtration_value = Slicer::Filtration_value; */

  Gudhi::multi_filtration::One_critical_filtration<value_type> basepoint = box.get_lower_corner();
  const std::size_t num_parameters = box.dimension();
  std::vector<int> grid_size(num_parameters);
  std::vector<bool> signs(num_parameters);
  int signs_shifts = 0;
  int arg_max_signs_shifts = -1;
  for (std::size_t i = 0; i < num_parameters; i++) {
    auto &a = box.get_lower_corner()[i];
    auto &b = box.get_upper_corner()[i];
    grid_size[i] = static_cast<int>(std::ceil((std::fabs(b - a) / precision))) + 1;
    signs[i] = b > a;
    if (b < a) {
      std::swap(a, b);
      int local_shift;
      if (!direction.num_parameters())
        local_shift = grid_size[i];
      else {
        local_shift = direction[i] > 0 ? static_cast<int>(std::ceil(grid_size[i] / direction[i])) : 0;
      }
      if (local_shift > signs_shifts) {
        signs_shifts = std::max(signs_shifts, local_shift);
        arg_max_signs_shifts = i;
      }
    }

    // fix the box
  }
  if (signs_shifts > 0) {
    for (std::size_t i = 0; i < num_parameters; i++)
      grid_size[i] += signs_shifts;  // this may be too much for large num_parameters
    grid_size[arg_max_signs_shifts] = 1;
    if (verbose)
      std::cout << "Had to flatten/shift coordinate " << arg_max_signs_shifts << " by " << signs_shifts << std::endl;
  }
  Module out(box);
  box.inflate(2 * precision);  // for infinte summands

  if (verbose) std::cout << "Num parameters : " << num_parameters << std::endl;
  if (verbose) std::cout << "Box : " << box << std::endl;
  if (num_parameters < 1) return out;

  // first line to compute
  // TODO: change here
  // for (auto i = 0u; i < basepoint.size() - 1; i++)
  //   basepoint[i] -= box.get_upper_corner().back();
  // basepoint.back() = 0;
  Line<value_type> current_line(basepoint, direction);
  if (verbose) std::cout << "First line basepoint " << basepoint << std::endl;

  {
    Timer timer("Initializing mma...\n", verbose);
    // fills the first barcode
    slicer.push_to(current_line);
    slicer.compute_persistence();
    auto barcode = slicer.get_flat_barcode();
    auto num_bars = barcode.size();
    out.resize(num_bars);
    /* Filtration_value birthContainer(num_parameters), */
    /* deathContainer(num_parameters); */
    for (std::size_t i = 0; i < num_bars; i++) {
      const auto &[dim, bar] = barcode[i];
      /* const auto &[birth, death] = bar; */
      out[i].set_dimension(dim);
      /* out[i].add_bar(birth, death, basepoint, birthContainer, deathContainer,
       */
      /* threshold, box); */
    }

    out.add_barcode(current_line, barcode, threshold);
    // std::cout << "out 1" << std::endl;
    // for (const auto& s : out)
    //   for (auto g : s.get_death_list()) std::cout << g << std::endl;
    // std::cout << std::endl;
    if (verbose) std::cout << "Instantiated " << num_bars << " summands" << std::endl;
  }
  // TODO : change here
  // std::vector<int> grid_size(num_parameters - 1);
  // auto h = box.get_upper_corner().back() - box.get_lower_corner().back();
  // for (int i = 0; i < num_parameters - 1; i++) {
  //   auto a = box.get_lower_corner()[i];
  //   auto b = box.get_upper_corner()[i];
  //   grid_size[i] =
  //       static_cast<unsigned int>(std::ceil((std::abs(b - a + h) /
  //       precision)));
  // }
  // TODO : change here
  if (verbose) {
    std::cout << "Grid size " << Gudhi::multi_filtration::One_critical_filtration(grid_size) << " Signs ";
    if (signs.empty()) {
      std::cout << "[]";
    } else {
      std::cout << "[";
      for (std::size_t i = 0; i < signs.size() - 1; i++) {
        std::cout << signs[i] << ", ";
      }
      std::cout << signs.back() << "]";
    }
    std::cout << std::endl;
    std::cout << "Max error " << precision << std::endl;
  }

  {
    Timer timer("Computing mma...", verbose);
    // actual computation. -1 as line grid is n-1 dim, -1 as we start from 0
    // _rec_mma(out, basepoint, grid_size, num_parameters - 2, slicer,
    // precision,
    //          threshold);
    // TODO : change here

    for (std::size_t i = 1; i < num_parameters; i++) {
      // the loop is on the faces of the lower box
      // should be parallelizable, up to a mutex on out
      if (direction.num_parameters() && direction[i] == 0.0) continue;  // skip faces with codim d_i=0
      auto temp_grid_size = grid_size;
      temp_grid_size[i] = 0;
      if (verbose)
        std::cout << "Face " << i << "/" << num_parameters << " with grid size "
                  << Gudhi::multi_filtration::One_critical_filtration(temp_grid_size) << std::endl;
      // if (!direction.size() || direction[0] > 0)
      _rec_mma2<0>(out,
                   Gudhi::multi_filtration::One_critical_filtration<value_type>(basepoint),
                   direction,
                   temp_grid_size,
                   signs,
                   num_parameters - 1,
                   slicer.weak_copy(),
                   precision,
                   threshold);
    }
    // last one, we can destroy basepoint & cie
    if (!direction.num_parameters() || direction[0] > 0) {
      grid_size[0] = 0;
      if (verbose)
        std::cout << "Face " << num_parameters << "/" << num_parameters << " with grid size "
                  << Gudhi::multi_filtration::One_critical_filtration(grid_size) << std::endl;
      _rec_mma2<1>(out,
                   std::move(basepoint),
                   direction,
                   grid_size,
                   signs,
                   num_parameters - 1,
                   std::move(slicer),
                   precision,
                   threshold);
    }
  }
  // std::cout << "out 2" << std::endl;
  // for (const auto& s : out)
  //   for (auto g : s.get_death_list()) std::cout << g << std::endl;
  // std::cout << std::endl;
  {  // for Timer
    Timer timer("Cleaning output ... ", verbose);
    out.clean();
    if (complete) {
      if (verbose) std::cout << "Completing output ...";
      for (std::size_t i = 0; i < num_parameters; i++) out.fill(precision);
    }
  }  // Timer death
  // std::cout << "out 3" << std::endl;
  // for (const auto& s : out)
  //   for (auto g : s.get_death_list()) std::cout << g << std::endl;
  // std::cout << std::endl;
  return out;
};

template <class Slicer, typename value_type>
Module<value_type> multiparameter_module_approximation(
    Slicer &slicer,
    const Gudhi::multi_filtration::One_critical_filtration<value_type> &direction,
    const value_type precision,
    Box<value_type> &box,
    const bool threshold,
    const bool complete,
    const bool verbose) {
  return _multiparameter_module_approximation<Module<value_type> >(
      slicer, direction, precision, box, threshold, complete, verbose);
};

template <class Slicer, typename value_type>
Two_parameter_module<value_type> multiparameter_module_approximation_with_2_parameters(
    Slicer &slicer,
    const Gudhi::multi_filtration::One_critical_filtration<value_type> &direction,
    const value_type precision,
    Box<value_type> &box,
    const bool threshold,
    const bool complete,
    const bool verbose) {
  return _multiparameter_module_approximation<Two_parameter_module<value_type> >(
      slicer, direction, precision, box, threshold, complete, verbose);
};

}  // namespace Gudhi::multiparameter::mma

#endif  // MMA_APPROXIMATION_H_INCLUDED
