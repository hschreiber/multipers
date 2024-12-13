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
 * @file module.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the functions related to the approximation of n-modules.
 */

#ifndef MMA_MODULE_H_INCLUDED
#define MMA_MODULE_H_INCLUDED

#include <algorithm>   //std::remove_if
#include <cmath>       //std::max, std::min
#include <cstddef>     //std::size_t
#include <functional>  //std::function
#include <iostream>    //std::cout
#include <limits>      //std::numeric_limits
#include <string>      //std::to_string
#include <vector>

#include <oneapi/tbb/parallel_for.h>

#include "../tensor/tensor.h"
#include "utilities.h"
#include "debug.h"
#include "summand.h"

#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Line.h>

namespace Gudhi::multiparameter::mma {

using Gudhi::multi_persistence::Box;
using Gudhi::multi_persistence::Line;

template <class Summand>
class Module_base {
 public:
  using dimension_type = typename Summand::dimension_type;
  using filtration_type = typename Summand::filtration_type;
  using value_type = typename Summand::value_type;
  using module_type = std::vector<Summand>;
  using image_type = std::vector<std::vector<value_type>>;
  using get_2dpixel_value_function_type = std::function<value_type(const typename module_type::const_iterator,
                                                                   const typename module_type::const_iterator,
                                                                   value_type,
                                                                   value_type)>;
  using get_pixel_value_function_type = std::function<value_type(const typename module_type::const_iterator,
                                                                 const typename module_type::const_iterator,
                                                                 std::vector<value_type> &)>;
  using Point = typename Summand::Point;

  Module_base() {}

  Module_base(Box<value_type> &box) : box_(box) {}

  void resize(unsigned int size) { module_.resize(size); }

  Summand &at(unsigned int index) { return module_.at(index); }

  Summand &operator[](std::size_t index) { return this->module_[index]; }

  const Summand &operator[](const std::size_t index) const { return this->module_[index]; }

  template <class Barcode>
  void add_barcode(const Barcode &barcode) {
    const bool verbose = false;
    if (barcode.size() != module_.size()) {
      std::cerr << "Barcode sizes doesn't match. Module is " << std::to_string(module_.size()) << " and barcode is "
                << std::to_string(barcode.size()) << std::endl;
    }
    unsigned int count = 0;
    for (const auto &bar_ : barcode) {
      auto &summand = this->operator[](count++);
      auto &[dim, bar] = bar_;
      auto &[birth_filtration, death_filtration] = bar;
      if constexpr (verbose) std::cout << "Birth " << birth_filtration << " Death " << death_filtration << std::endl;
      summand.add_bar(birth_filtration, death_filtration);
    }
  }

  void add_barcode(const Line<value_type> &line,
                   const std::vector<std::pair<int, std::pair<value_type, value_type>>> &barcode,
                   const bool threshold_in) {
    assert(barcode.size() == module_.size() && "Barcode sizes doesn't match.");

    auto count = 0U;
    for (const auto &extBar : barcode) {
      auto &[dim, bar] = extBar;
      _add_bar_with_threshold(line, bar, threshold_in, this->operator[](count++));
    }
  }

  void add_barcode(const Line<value_type> &line,
                   const std::vector<std::pair<value_type, value_type>> &barcode,
                   const bool threshold_in) {
    assert(barcode.size() == module_.size() && "Barcode sizes doesn't match.");

    auto count = 0U;
    for (const auto &bar : barcode) {
      _add_bar_with_threshold(line, bar, threshold_in, this->operator[](count++));
    }
  }

  typename module_type::iterator begin() { return module_.begin(); }

  typename module_type::iterator end() { return module_.end(); }

  /**
   * @brief Remove the empty summands of the output
   *
   * @param output p_output:...
   * @param keep_order p_keep_order:... Defaults to false.
   */
  void clean() {
    module_type tmp;
    for (size_t i = 0u; i < module_.size(); i++) {
      module_[i].clean();
    }
    module_.erase(
        std::remove_if(module_.begin(), module_.end(), [](const Summand &s) { return s.get_upset().is_plus_inf(); }),
        module_.end());
  }

  void fill(const value_type precision) {
    if (module_.empty()) return;

    for (Summand &sum : module_) {
      sum.complete_birth(precision);
      sum.complete_death(precision);
    }
  }

  std::vector<image_type> get_vectorization(const value_type delta,
                                            const value_type p,
                                            const bool normalize,
                                            const Gudhi::multi_persistence::Box<value_type> &box,
                                            unsigned int horizontalResolution,
                                            unsigned int verticalResolution) {
    dimension_type maxDim = module_.back().get_dimension();
    std::vector<Module_base::image_type> image_vector(maxDim + 1);
    typename module_type::iterator start;
    typename module_type::iterator end = module_.begin();
    for (dimension_type d = 0; d <= maxDim; d++) {
      {  // for Timer
        Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
        start = end;
        while (end != module_.end() && end->get_dimension() == d) end++;
        _compute_2D_image(
            image_vector.at(d), start, end, delta, p, normalize, box, horizontalResolution, verticalResolution);
      }  // Timer death
    }
    return image_vector;
  }

  std::vector<image_type> get_vectorization(unsigned int horizontalResolution,
                                            unsigned int verticalResolution,
                                            get_2dpixel_value_function_type get_pixel_value) const {
    dimension_type maxDim = module_.back().get_dimension();
    std::vector<Module_base::image_type> image_vector(maxDim + 1);
    typename module_type::const_iterator start;
    typename module_type::const_iterator end = module_.begin();
    for (dimension_type d = 0; d <= maxDim; d++) {
      {  // for Timer
        Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
        start = end;
        while (end != module_.end() && end->get_dimension() == d) end++;
        _compute_2D_image(image_vector.at(d), start, end, horizontalResolution, verticalResolution, get_pixel_value);
      }  // Timer death
    }
    return image_vector;
  }

  image_type get_vectorization_in_dimension(const dimension_type dimension,
                                            const value_type delta,
                                            const value_type p,
                                            const bool normalize,
                                            const Gudhi::multi_persistence::Box<value_type> &box,
                                            unsigned int horizontalResolution,
                                            unsigned int verticalResolution) {
    Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

    Module_base::image_type image;
    typename module_type::iterator start = module_.begin();
    while (start != module_.end() && start->get_dimension() < dimension) start++;
    typename module_type::iterator end = start;
    while (end != module_.end() && end->get_dimension() == dimension) end++;
    _compute_2D_image(image, start, end, delta, p, normalize, box, horizontalResolution, verticalResolution);

    return image;
  }

  image_type get_vectorization_in_dimension(const dimension_type dimension,
                                            unsigned int horizontalResolution,
                                            unsigned int verticalResolution,
                                            get_2dpixel_value_function_type get_pixel_value) const {
    Debug::Timer timer("Computing image of dimension " + std::to_string(dimension) + " ...", verbose);

    typename Module_base::image_type image;
    typename module_type::const_iterator start = module_.begin();
    while (start != module_.end() && start->get_dimension() < dimension) start++;
    typename module_type::const_iterator end = start;
    while (end != module_.end() && end->get_dimension() == dimension) end++;
    _compute_2D_image(image, start, end, horizontalResolution, verticalResolution, get_pixel_value);

    return image;
  }

  std::vector<value_type> get_landscape_values(const std::vector<value_type> &x, const dimension_type dimension) const {
    std::vector<value_type> out;
    out.reserve(this->size());
    for (unsigned int i = 0; i < this->size(); i++) {
      const Summand &summand = this->module_[i];
      if (summand.get_dimension() == dimension) out.push_back(summand.get_landscape_value(x));
    }
    std::sort(out.begin(), out.end(), [](const value_type x, const value_type y) { return x > y; });
    return out;
  }

  image_type get_landscape(const dimension_type dimension,
                           const unsigned int k,
                           const Box<value_type> &box,
                           const std::vector<unsigned int> &resolution) const {
    // TODO extend in higher dimension (ie, change the image type to a template
    // class)
    Module_base::image_type image;
    image.resize(resolution[0], std::vector<value_type>(resolution[1]));
    value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / resolution[0];
    value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / resolution[1];
    tbb::parallel_for(0U, resolution[0], [&](unsigned int i) {
      tbb::parallel_for(0U, resolution[1], [&](unsigned int j) {
        auto landscape = this->get_landscape_values(
            {box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j}, dimension);
        image[i][j] = k < landscape.size() ? landscape[k] : 0;
      });
    });
    return image;
  }

  std::vector<image_type> get_landscapes(const dimension_type dimension,
                                         const std::vector<unsigned int> ks,
                                         const Box<value_type> &box,
                                         const std::vector<unsigned int> &resolution) const {
    std::vector<Module_base::image_type> images(ks.size());
    for (auto &image : images) image.resize(resolution[0], std::vector<value_type>(resolution[1]));
    value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / resolution[0];
    value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / resolution[1];

    tbb::parallel_for(0U, resolution[0], [&](unsigned int i) {
      tbb::parallel_for(0U, resolution[1], [&](unsigned int j) {
        std::vector<value_type> landscapes = this->get_landscape_values(
            {box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j}, dimension);
        for (const auto k : ks) {
          images[k][i][j] = k < landscapes.size() ? landscapes[k] : 0;
        }
      });
    });
    return images;
  }

  void add_summand(Summand summand, int degree = -1) {
    if (degree >= 0) summand.set_dimension(degree);
    module_.push_back(summand);
  }

  Box<value_type> get_box() const { return this->box_; }

  void set_box(Box<value_type> box) { this->box_ = box; }

  unsigned int size() const { return this->module_.size(); }

  void infer_box(std::vector<filtration_type> &f) { this->box_.infer_from_filters(f); }

  dimension_type get_dimension() const { return this->module_.empty() ? -1 : this->module_.back().get_dimension(); }

  module_type get_summands_of_dimension(const int dimension) const {
    std::vector<Summand> list;
    for (const Summand &summand : this->module_) {
      if (summand.get_dimension() == dimension) list.push_back(summand);
    }
    return list;
  }

  std::vector<std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>>>
  get_corners_of_dimension(const int dimension) const {
    std::vector<std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>>> list;
    list.reserve(module_.size());
    for (const Summand &summand : this->module_) {
      if (summand.get_dimension() == dimension) {
        auto bl = summand.get_birth_list();
        auto dl = summand.get_death_list();
        std::pair<std::vector<std::vector<value_type>>, std::vector<std::vector<value_type>>> sl;
        sl.first.reserve(bl.size());
        sl.second.reserve(dl.size());
        for (const auto& bf : bl){
          sl.first.emplace_back(bf.begin(), bf.end());
        }
        for (const auto& df : dl){
          sl.second.emplace_back(df.begin(), df.end());
        }

        list.push_back(std::move(sl));
      }
    }
    return list;
  }

  MultiDiagram<Point, value_type> get_barcode(const Line<value_type> &l,
                                                        const dimension_type dimension = -1,
                                                        const bool threshold = false) const {
    const bool verbose = false;
    if constexpr (verbose)
      std::cout << "Computing barcode of dimension " << dimension << " and threshold " << threshold << std::endl;
    std::vector<MultiDiagram_point<Point>> barcode(this->size());
    std::pair<value_type, value_type> threshold_bounds;
    if (threshold) threshold_bounds = l.get_bounds(this->box_);
    unsigned int summand_idx = 0;
    for (unsigned int i = 0; i < this->size(); i++) {
      const Summand &summand = this->module_[i];
      if constexpr (verbose) std::cout << "Summand of dimension " << summand.get_dimension() << std::endl;

      if (dimension != -1 && summand.get_dimension() != dimension) continue;
      /* if (dimension != -1 && summand.get_dimension() > dimension) */
      /* 	break; */
      auto pushed_summand = summand.get_bar(l);

      Point &pbirth = pushed_summand.first;
      Point &pdeath = pushed_summand.second;
      if constexpr (verbose) std::cout << "BAR : " << pbirth << " " << pdeath << std::endl;
      if (threshold) {
        auto min = l[threshold_bounds.first];
        auto max = l[threshold_bounds.second];
        if (!(pbirth < max) || !(pdeath > min)) {
          /* continue; */  // We still need summands to be aligned. The price to
          // pay is some memory.
          pbirth = std::numeric_limits<Point>::infinity();
          pdeath = pbirth;
        }
        pbirth.push_to_least_common_upper_bound(min);
        pdeath.pull_to_greatest_common_lower_bound(max);
      }
      barcode[summand_idx++] = MultiDiagram_point(summand.get_dimension(), pbirth, pdeath);
    }
    barcode.resize(summand_idx);
    return MultiDiagram<Point, value_type>(barcode);
  }

  std::vector<std::vector<std::pair<value_type, value_type>>> get_barcode2(const Line<value_type> &l,
                                                                           const dimension_type dimension) const {
    const bool verbose = false;
    std::vector<std::vector<std::pair<value_type, value_type>>> barcode(this->get_dimension() + 1);
    for (auto i = 0; i < this->get_dimension(); ++i) {
      barcode[i].reserve(this->size());
    }
    for (unsigned int i = 0; i < this->size(); i++) {
      const Summand &summand = this->module_[i];
      if constexpr (verbose) std::cout << "Summand of dimension " << summand.get_dimension() << std::endl;

      if (dimension != -1 && summand.get_dimension() != dimension) continue;
      /* if (dimension != -1 && summand.get_dimension() > dimension) */
      /* 	break; */
      const auto &pushed_summand = summand.get_bar2(l);

      barcode[summand.get_dimension()].push_back(pushed_summand);
    }
    return barcode;
  }

  std::vector<std::vector<std::vector<std::pair<value_type, value_type>>>> get_barcodes2(
      const std::vector<Line<value_type>> &lines,
      const dimension_type dimension = -1) const {
    unsigned int nlines = lines.size();
    std::vector<std::vector<std::vector<std::pair<value_type, value_type>>>> out(
        this->get_dimension() + 1, std::vector<std::vector<std::pair<value_type, value_type>>>(nlines));
    tbb::parallel_for(0U, nlines, [&](unsigned int i) {
      const Line<value_type> &l = lines[i];
      for (const auto &summand : module_) {
        if (dimension != -1 && summand.get_dimension() != dimension) continue;
        const auto &bar = summand.get_bar2(l);
        out[summand.get_dimension()][i].push_back(bar);
      }
    });
    return out;
  }

  MultiDiagrams<Point, value_type> get_barcodes(const std::vector<Line<value_type>> &lines,
                                                          const dimension_type dimension = -1,
                                                          const bool threshold = false) const {
    unsigned int nlines = lines.size();
    MultiDiagrams<Point, value_type> out(nlines);
    tbb::parallel_for(0U, nlines, [&](unsigned int i) {
      const Line<value_type> &l = lines[i];
      out[i] = this->get_barcode(l, dimension, threshold);
    });
    return out;
  }

  template<class CoordinateRange>
  MultiDiagrams<Point, value_type> get_barcodes(const std::vector<CoordinateRange> &basepoints,
                                                          const dimension_type dimension = -1,
                                                          const bool threshold = false) const {
    unsigned int nlines = basepoints.size();
    MultiDiagrams<Point, value_type> out(nlines);
    // for (unsigned int i = 0; i < nlines; i++){
    tbb::parallel_for(0U, nlines, [&](unsigned int i) {
      const Line<value_type> &l = Line<value_type>(basepoints[i]);
      out[i] = this->get_barcode(l, dimension, threshold);
    });
    return out;
  }

  template<class CoordinateRange>
  std::vector<int> euler_curve(const std::vector<CoordinateRange> &points) const {
    unsigned int npts = points.size();
    std::vector<int> out(npts);
    // #pragma omp parallel for
    tbb::parallel_for(0U, static_cast<unsigned int>(out.size()), [&](unsigned int i) {
      auto &euler_char = out[i];
      const Point &point = points[i];
      /* #pragma omp parallel for reduction(+ : euler_char) */
      for (const Summand &I : this->module_) {
        if (I.contains(point)) {
          int sign = I.get_dimension() % 2 ? -1 : 1;
          euler_char += sign;
        }
      }
    });
    return out;
  }

  inline Box<value_type> get_bounds() const {
    using P = typename Box<value_type>::Point;
    //TODO: verify if this does not make problem with the infinite case
    dimension_type num_parameters = box_.get_lower_corner().num_parameters();
    P lower_bound(num_parameters, std::numeric_limits<value_type>::infinity());
    P upper_bound(num_parameters, -std::numeric_limits<value_type>::infinity());
    for (const auto &summand : module_) {
      const auto &summand_bounds = summand.get_bounds();
      const auto &[m, M] = summand_bounds.get_bounding_corners();
      for (auto parameter = 0; parameter < num_parameters; parameter++) {
        lower_bound[parameter] = std::min(m[parameter], lower_bound[parameter]);
        upper_bound[parameter] = std::min(M[parameter], upper_bound[parameter]);
      }
    }
    return Box(lower_bound, upper_bound);
  }

  inline void rescale(const std::vector<value_type> &rescale_factors, int degree) {
    for (auto &summand : module_) {
      if (degree == -1 or summand.get_dimension() == degree) summand.rescale(rescale_factors);
    }
  }

  inline void translate(const std::vector<value_type> &translation, int degree) {
    for (auto &summand : module_) {
      if (degree == -1 or summand.get_dimension() == degree) summand.translate(translation);
    }
  }

  std::vector<std::vector<value_type>> compute_pixels(const std::vector<std::vector<value_type>> &coordinates,
                                                      const std::vector<int> &degrees,
                                                      const Box<value_type> &box = {},
                                                      const value_type delta = 0.1,
                                                      const value_type p = 1,
                                                      const bool normalize = true,
                                                      const int n_jobs = 0) {
    auto num_degrees = degrees.size();
    auto num_pts = coordinates.size();
    std::vector<std::vector<value_type>> out(num_degrees, std::vector<value_type>(num_pts));

    typename module_type::iterator start;
    typename module_type::iterator end = module_.begin();
    for (auto degree_idx = 0u; degree_idx < num_degrees; degree_idx++) {
      {  // for Timer
        auto d = degrees[degree_idx];
        Debug::Timer timer("Computing image of dimension " + std::to_string(d) + " ...", verbose);
        start = end;
        while (start != module_.end() && start->get_dimension() != d) start++;
        if (start == module_.end()) break;
        end = start;
        while (end != module_.end() && end->get_dimension() == d) end++;
        out[degree_idx] = compute_pixels_of_degree(start, end, delta, p, normalize, box, coordinates, n_jobs);
      }  // Timer death
    }
    return out;
  }

  std::vector<value_type> get_interleavings(const Box<value_type> &box) {
    std::vector<value_type> out(this->size());
    for (auto i = 0u; i < out.size(); ++i) {
      out[i] = module_[i].get_interleaving(box);
    }
    return out;
  }

  using distance_to_idx_type = std::vector<std::vector<int>>;

  distance_to_idx_type compute_distance_idx_to(const std::vector<value_type> &pt, bool full) const {
    distance_to_idx_type out(module_.size(), std::vector<int>(full ? 4 : 2));
    for (auto i = 0u; i < module_.size(); ++i) {
      out[i] = module_[i].distance_idx_to(pt, full);
    }
    return out;
  }

  std::vector<value_type> compute_distance_to(const std::vector<value_type> &pt, bool negative) const {
    std::vector<value_type> out(this->size());
    for (auto i = 0u; i < this->size(); ++i) {
      out[i] = module_[i].distance_to(pt, negative);
    }
    return out;
  }

  std::vector<std::vector<value_type>> compute_distances_to(const std::vector<std::vector<value_type>> &pts,
                                                            bool negative,
                                                            int n_jobs) const {
    std::vector<std::vector<value_type>> out(pts.size(), std::vector<value_type>(this->size()));
    oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
    arena.execute([&] {
      tbb::parallel_for(std::size_t(0u), pts.size(), [&](std::size_t i) {
        tbb::parallel_for(std::size_t(0u), std::size_t(this->size()), [&](std::size_t j) {
          out[i][j] = module_[j].distance_to(pts[i], negative);
        });
      });
    });
    return out;
  }

  template <typename dtype = value_type, typename indices_type = int32_t>
  void inline compute_distances_to(dtype *data_ptr,
                                   const std::vector<std::vector<value_type>> &pts,
                                   bool negative,
                                   int n_jobs) const {
    tensor::static_tensor_view<dtype, indices_type> container(
        data_ptr, {static_cast<int>(pts.size()), static_cast<int>(this->size())});
    oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
    arena.execute([&] {
      tbb::parallel_for(std::size_t(0u), pts.size(), [&](std::size_t i) {
        // tbb::parallel_for(std::size_t(0u), std::size_t(this->size()), [&](std::size_t j) {
        dtype *current_ptr = &container[{static_cast<int>(i), 0}];
        for (std::size_t j = 0u; j < this->size(); ++j) {
          *(current_ptr + j) = module_[j].distance_to(pts[i], negative);
        }
      });
      // });
    });
  }

  using distances_to_idx_type = std::vector<distance_to_idx_type>;

  distances_to_idx_type compute_distances_idx_to(const std::vector<std::vector<value_type>> &pts,
                                                 bool full,
                                                 int n_jobs) const {
    distances_to_idx_type out(pts.size(), distance_to_idx_type(module_.size(), std::vector<int>(full ? 4 : 2)));

    oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
    arena.execute([&] {
      tbb::parallel_for(std::size_t(0u), pts.size(), [&](std::size_t i) {
        tbb::parallel_for(std::size_t(0u), std::size_t(this->size()), [&](std::size_t j) {
          out[i][j] = module_[j].distance_idx_to(pts[i], full);
        });
      });
    });
    return out;
  }

  std::vector<value_type> compute_pixels_of_degree(const typename module_type::iterator start,
                                                   const typename module_type::iterator end,
                                                   const value_type delta,
                                                   const value_type p,
                                                   const bool normalize,
                                                   const Box<value_type> &box,
                                                   const std::vector<std::vector<value_type>> &coordinates,
                                                   const int n_jobs = 0) {
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

  using idx_dump_type =
      std::vector<std::vector<std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>>>;

  idx_dump_type to_idx(const std::vector<std::vector<value_type>> &grid) const {
    unsigned int num_parameters = grid.size();
    auto dimension = this->get_dimension();
    idx_dump_type out(dimension + 1);
    for (auto i = 0u; i < this->size(); ++i) {
      auto &interval = this->operator[](i);
      auto &out_of_dim = out[interval.get_dimension()];
      out_of_dim.reserve(this->size());
      std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> interval_idx;

      auto &birth_idx = interval_idx.first;
      birth_idx.reserve(interval.get_birth_list().size());
      auto &death_idx = interval_idx.second;
      death_idx.reserve(interval.get_death_list().size());

      for (const auto &pt : interval.get_birth_list()) {
        std::vector<int> pt_idx(pt.size());
        for (auto i = 0u; i < num_parameters; ++i) {
          pt_idx[i] = std::distance(grid[i].begin(), std::lower_bound(grid[i].begin(), grid[i].end(), pt[i]));
        }
        birth_idx.push_back(pt_idx);
      }
      for (const auto &pt : interval.get_death_list()) {
        std::vector<int> pt_idx(pt.size());
        for (auto i = 0u; i < num_parameters; ++i) {
          pt_idx[i] = std::distance(grid[i].begin(), std::lower_bound(grid[i].begin(), grid[i].end(), pt[i]));
        }
        death_idx.push_back(pt_idx);
      }
      out_of_dim.push_back(interval_idx);
    }
    return out;
  }

  template <class Out_summand>
  Module_base<Out_summand> grid_squeeze(const std::vector<std::vector<value_type>> &grid) const {
    auto dimension = this->get_dimension();
    Module_base<Out_summand> out(this->size());
    for (auto i = 0u; i < this->size(); ++i) {
      const auto &interval = this->operator[](i);
      out[i] = interval.template grid_squeeze<Out_summand>(grid);
    }
    return out;
  }

  std::vector<std::vector<std::vector<int>>> to_flat_idx(const std::vector<std::vector<value_type>> &grid) const {
    std::vector<std::vector<std::vector<int>>> out(3);
    auto &idx = out[0];
    auto &births = out[1];
    auto &deaths = out[2];

    idx.resize(2);
    idx[0].resize(this->size());
    idx[1].resize(this->size());

    // some heuristic: usually
    births.reserve(2 * this->size());
    deaths.reserve(2 * this->size());
    for (auto i = 0u; i < this->size(); ++i) {
      auto &interval = this->operator[](i);
      idx[0][i] = interval.get_birth_list().size();
      for (const auto &pt : interval.get_birth_list()) {
        births.push_back(_to_grid_coord(pt, grid));
      }
      idx[1][i] = interval.get_death_list().size();
      for (const auto &pt : interval.get_death_list()) {
        deaths.push_back(_to_grid_coord(pt, grid));
      }
    }
    return out;
  }

  std::vector<int> inline get_degree_splits() const {
    std::vector<int> splits = {};
    int current_degree = 0;
    for (auto i = 0u; i < this->size(); ++i) {
      const auto &summand = this->operator[](i);
      while (summand.get_dimension() > current_degree) [[unlikely]] {
        current_degree++;
        splits.push_back(i);
      }
    }
    return splits;
  }

 private:
  module_type module_;
  Box<value_type> box_;

  void _compute_2D_image(image_type &image,
                         const typename module_type::iterator start,
                         const typename module_type::iterator end,
                         const value_type delta = 0.1,
                         const value_type p = 1,
                         const bool normalize = true,
                         const Box<value_type> &box = Box<value_type>(),
                         const unsigned int horizontalResolution = 100,
                         const unsigned int verticalResolution = 100) {
    image.resize(horizontalResolution, std::vector<value_type>(verticalResolution));
    value_type moduleWeight = 0;
    {  // for Timer
      Debug::Timer timer("Computing module weight ...", verbose);
      for (auto it = start; it != end; it++)  //  precomputes interleaving restricted to box for all summands.
        it->get_interleaving(box);
      if (p == 0) {
        /* #pragma omp parallel for reduction(+ : moduleWeight) */
        for (auto it = start; it != end; it++) {
          moduleWeight += it->get_interleaving() > 0;
        }
      } else if (p != inf) {
        /* #pragma omp parallel for reduction(+ : moduleWeight) */
        for (auto it = start; it != end; it++) {
          // /!\ TODO deal with inf summands (for the moment,  depends on the box
          // ...)
          if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
            moduleWeight += std::pow(it->get_interleaving(), p);
        }
      } else {
        /* #pragma omp parallel for reduction(std::max : moduleWeight) */
        for (auto it = start; it != end; it++) {
          if (it->get_interleaving() > 0 && it->get_interleaving() != inf)
            moduleWeight = std::max(moduleWeight, it->get_interleaving());
        }
      }
    }  // Timer death
    if (verbose) std::cout << "Module " << start->get_dimension() << " has weight : " << moduleWeight << "\n";
    if (!moduleWeight) return;

    if constexpr (Debug::debug)
      if (moduleWeight < 0) {
        if constexpr (Debug::debug) std::cout << "!! Negative weight !!" << std::endl;
        // 		image.clear();
        return;
      }

    value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / horizontalResolution;
    value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / verticalResolution;

    {  // for Timer
      Debug::Timer timer("Computing pixel values ...", verbose);

      tbb::parallel_for(0U, horizontalResolution, [&](unsigned int i) {
        tbb::parallel_for(0U, verticalResolution, [&](unsigned int j) {
          image[i][j] = _get_pixel_value(start,
                                         end,
                                         {box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j},
                                         delta,
                                         p,
                                         normalize,
                                         moduleWeight);
        });
      });
    }  // Timer death
  }

  void _compute_2D_image(image_type &image,
                         const typename module_type::const_iterator start,
                         const typename module_type::const_iterator end,
                         unsigned int horizontalResolution,
                         unsigned int verticalResolution,
                         get_2dpixel_value_function_type get_pixel_value) const {
    image.resize(horizontalResolution, std::vector<value_type>(verticalResolution));
    const Box<value_type> &box = this->box_;
    value_type stepX = (box.get_upper_corner()[0] - box.get_lower_corner()[0]) / horizontalResolution;
    value_type stepY = (box.get_upper_corner()[1] - box.get_lower_corner()[1]) / verticalResolution;

    {  // for Timer
      Debug::Timer timer("Computing pixel values ...", verbose);

      // #pragma omp parallel for collapse(2)
      // 			for (unsigned int i = 0; i < horizontalResolution; i++)
      // 			{
      // 				for (unsigned int j = 0; j < verticalResolution;
      // j++)
      // 				{
      // 					image[i][j] = get_pixel_value(
      // 						start,
      // 						end,
      // 						box.get_lower_corner()[0] +
      // stepX * i,
      // box.get_lower_corner()[1] + stepY * j);
      // 				}
      // 			}
      tbb::parallel_for(0U, horizontalResolution, [&](unsigned int i) {
        tbb::parallel_for(0U, verticalResolution, [&](unsigned int j) {
          image[i][j] =
              get_pixel_value(start, end, box.get_lower_corner()[0] + stepX * i, box.get_lower_corner()[1] + stepY * j);
        });
      });

    }  // Timer death
  }

  template<class Filtration_value_range>
  value_type _get_pixel_value(const typename module_type::iterator start,
                              const typename module_type::iterator end,
                              const Filtration_value_range& x,
                              const value_type delta,
                              const value_type p,
                              const bool normalize,
                              const value_type moduleWeight) const {
    value_type value = 0;
    if (p == 0) {
      /* #pragma omp parallel for reduction(+ : value) */
      for (auto it = start; it != end; it++) {
        value += it->get_local_weight(x, delta);
      }
      if (normalize) value /= moduleWeight;
      return value;
    }
    if (p != inf) {
      /* #pragma omp parallel for reduction(+ : value) */
      for (auto it = start; it != end; it++) {
        value_type summandWeight = it->get_interleaving();
        value_type summandXWeight = it->get_local_weight(x, delta);
        value += std::pow(summandWeight, p) * summandXWeight;
      }
      if (normalize) value /= moduleWeight;
      return value;
    }

    /* #pragma omp parallel for reduction(std::max : value) */
    for (auto it = start; it != end; it++) {
      value = std::max(value, it->get_local_weight(x, delta));
    }
    return value;
  }

  void _add_bar_with_threshold(const Line<value_type> &line,
                               const std::pair<value_type, value_type> &bar,
                               const bool threshold_in,
                               Summand &summand) {
    const bool verbose = false;
    auto [birth_filtration, death_filtration] = bar;

    if (birth_filtration >= death_filtration) return;

    if constexpr (verbose) {
      std::cout << std::setprecision(std::numeric_limits<value_type>::digits) << "--BAR (" << birth_filtration << ", "
                << death_filtration << ") at basepoint " << line.base_point() << " direction " << line.direction()
                << std::endl;
    }

    auto birth_container = line[birth_filtration];
    if constexpr (verbose) std::cout << " B: " << birth_container << " B*d: " << birth_filtration * line.direction();
    if (birth_container.is_minus_inf()) {
      if (threshold_in) birth_container = box_.get_lower_corner();
    } else {
      bool allInf = true;
      for (std::size_t i = 0U; i < birth_container.num_parameters(); i++) {
        auto t = box_.get_lower_corner()[i];
        if (birth_container[i] < t - 1e-10) birth_container[i] = threshold_in ? t : -filtration_type::T_inf;
        if (birth_container[i] != -filtration_type::T_inf) allInf = false;
      }
      if (allInf) birth_container = decltype(birth_container)::minus_inf();
    }

    auto death_container = line[death_filtration];
    if constexpr (verbose) std::cout << " D: " << death_container;
    if (death_container.is_plus_inf()) {
      if (threshold_in) death_container = box_.get_upper_corner();
    } else {
      bool allInf = true;
      for (std::size_t i = 0U; i < death_container.num_parameters(); i++) {
        auto t = box_.get_upper_corner()[i];
        if (death_container[i] > t + 1e-10) death_container[i] = threshold_in ? t : filtration_type::T_inf;
        if (death_container[i] != filtration_type::T_inf) allInf = false;
      }
      if (allInf) death_container = decltype(death_container)::inf();
    }

    if constexpr (verbose) std::cout << " BT: " << birth_container << " DT: " << death_container << std::endl;
    summand.add_bar(birth_container, death_container);
  }

  template<class Coordinate_range>
  static std::vector<int> _to_grid_coord(const Coordinate_range &pt, const std::vector<std::vector<value_type>> &grid) {
    std::size_t num_parameters = grid.size();
    std::vector<int> out(num_parameters);
    //TODO: avoid going three times trough pt when an array in is_plus_inf/is_minus_inf/is_nan
    if (pt.is_plus_inf() || pt.is_nan()) [[unlikely]] {
      for (size_t i = 0; i < num_parameters; ++i) out[i] = grid[i].size() - 1;
      return out;
    }
    if (pt.is_minus_inf()) [[unlikely]] {
      for (size_t i = 0; i < num_parameters; ++i) out[i] = 0;
      return out;
    }
    // pt has to be of size num_parameters now
    for (size_t i = 0u; i < num_parameters; ++i) {
      if (pt[i] >= grid[i].back()) [[unlikely]]
        out[i] = grid[i].size() - 1;
      else if (pt[i] <= grid[i][0]) [[unlikely]] {
        out[i] = 0;
      } else
        out[i] = std::distance(grid[i].begin(), std::lower_bound(grid[i].begin(), grid[i].end(), pt[i]));
    }
    return out;
  }
};

template <typename value_type>
using Module = Module_base<Summand<value_type> >;
template <typename value_type, unsigned int N>
using Fixed_parameter_module = Module_base<Fixed_parameter_summand<value_type, N> >;
template <typename value_type>
using Two_parameter_module = Module_base<Two_parameter_summand<value_type> >;

}  // namespace Gudhi::multiparameter::mma

#endif  // MMA_MODULE_H_INCLUDED
