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
 * @file summand.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the functions related to the approximation of n-modules.
 */

#ifndef MMA_SUMMAND_H_INCLUDED
#define MMA_SUMMAND_H_INCLUDED

#include <algorithm>  //std::remove_if
#include <cmath>      //std::abs
#include <iostream>   //std::cout
#include <limits>     //std::numeric_limits
#include <cstdint>    //std::int64_t
#include <type_traits>
#include <utility>
#include <vector>

#include "debug.h"
#include "gudhi/One_critical_filtration.h"

#include <gudhi/Debug_utils.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Line.h>
#include <gudhi/Multi_critical_filtration_with_n_parameters.h>
#include <gudhi/One_critical_filtration_with_n_parameters.h>
#include <gudhi/One_critical_filtration_with_n_parameters_view.h>

namespace Gudhi::multiparameter::mma {

using Gudhi::multi_persistence::Box;
using Gudhi::multi_persistence::Line;

template <class Births, class Deaths, class Filtration_value>
class Summand_base {
 public:
  using births_type = Births;  // = Gudhi::multi_filtration::Multi_critical_filtration<value_type, false>;
  using deaths_type = Deaths;  // = Gudhi::multi_filtration::Multi_critical_filtration<value_type, true>;
  using filtration_type = Filtration_value;
  using dimension_type = int;
  using value_type = typename Filtration_value::value_type;
  using Point = typename Box<value_type>::Point;

  Summand_base()
      : birth_corners_(births_type::inf()),
        death_corners_(deaths_type::minus_inf()),
        distanceTo0_(-1),
        dimension_(-1) {}

  Summand_base(const births_type &birth_corners, const deaths_type &death_corners, dimension_type dimension)
      : birth_corners_(birth_corners), death_corners_(death_corners), distanceTo0_(-1), dimension_(dimension) {}

  Summand_base(const std::vector<filtration_type> &birth_corners,
               const std::vector<filtration_type> &death_corners,
               dimension_type dimension)
      : birth_corners_(birth_corners), death_corners_(death_corners), distanceTo0_(-1), dimension_(dimension) {}

  Summand_base(const Summand_base &other)
      : birth_corners_(other.birth_corners_),
        death_corners_(other.death_corners_),
        distanceTo0_(other.distanceTo0_),
        dimension_(other.dimension_) {}

  Summand_base(Summand_base &&other)
      : birth_corners_(std::move(other.birth_corners_)),
        death_corners_(std::move(other.death_corners_)),
        distanceTo0_(std::exchange(other.distanceTo0_, -1)),
        dimension_(std::exchange(other.dimension_, -1)) {}

  Summand_base& operator=(const Summand_base &other){
    birth_corners_ = other.birth_corners_;
    death_corners_ = other.death_corners_;
    distanceTo0_ = other.distanceTo0_;
    dimension_ = other.dimension_;

    return *this;
  }

  Summand_base& operator=(Summand_base &&other){
    birth_corners_ = std::move(other.birth_corners_);
    death_corners_ = std::move(other.death_corners_);
    distanceTo0_ = std::move(other.distanceTo0_);
    dimension_ = std::move(other.dimension_);

    return *this;
  }

  value_type get_interleaving() const { return distanceTo0_; }

  value_type get_interleaving(const Box<value_type> &box) {
    _compute_interleaving(box);
    return distanceTo0_;
  }

  template<class Filtration_value_range>
  value_type get_local_weight(const Filtration_value_range &x, const value_type delta) const {
    bool rectangle = delta <= 0;

    // TODO: add assert to verify that x.size == birth.size/death.size
    // if they are not infinite.

    filtration_type mini(x.size());
    filtration_type maxi(x.size());

    // box on which to compute the local weight
    for (unsigned int i = 0; i < x.size(); i++) {
      mini[i] = delta <= 0 ? x[i] + delta : x[i] - delta;
      maxi[i] = delta <= 0 ? x[i] - delta : x[i] + delta;
    }

    //TODO: aren't those just pull_tos/push_tos on x with the values of the corners?
    //TODO: replace vector of filtration_type with births_type/death_type

    // Pre-allocating
    std::vector<filtration_type> birthList(birth_corners_.num_generators());
    std::vector<filtration_type> deathList(death_corners_.num_generators());
    unsigned int lastEntry = 0;
    for (const auto &birth : birth_corners_) {
      if (birth <= maxi) {
        unsigned int dim = std::max(birth.num_parameters(), mini.num_parameters());
        filtration_type tmpBirth(dim);
        for (unsigned int i = 0; i < dim; i++) {
          auto birthi = birth.num_parameters() > i ? birth[i] : birth[0];
          auto minii = mini.num_parameters() > i ? mini[i] : mini[0];
          tmpBirth[i] = std::max(birthi, minii);
        }

        birthList[lastEntry].swap(tmpBirth);
        lastEntry++;
      }
    }
    birthList.resize(lastEntry);

    // Thresholds birthlist & deathlist to B_inf(x,delta)
    lastEntry = 0;
    for (const auto &death : death_corners_) {
      if (death >= mini) {
        unsigned int dim = std::max(death.num_parameters(), maxi.num_parameters());
        filtration_type tmpDeath(dim);
        for (unsigned int i = 0; i < dim; i++) {
          auto deathi = death.num_parameters() > i ? death[i] : death[0];
          auto maxii = maxi.num_parameters() > i ? maxi[i] : maxi[0];
          tmpDeath[i] = std::min(deathi, maxii);
        }

        deathList[lastEntry].swap(tmpDeath);
        lastEntry++;
      }
    }
    deathList.resize(lastEntry);
    value_type local_weight = 0;
    if (!rectangle) {
      // Local weight is inteleaving to 0 of module restricted to the square
      // #pragma omp parallel for reduction(std::max: local_weight)
      Box<value_type> trivial_box;
      for (const filtration_type &birth : birthList) {
        if (birth.num_parameters() == 0) continue;
        for (const filtration_type &death : deathList) {
          if (death.num_parameters() > 0)
            local_weight = std::max(local_weight,
                                    _get_max_diagonal(birth,
                                                      death,
                                                      trivial_box));  // if box is empty, does not thredhold
                                                                      // (already done before).
        }
      }
      return local_weight / (2 * std::abs(delta));
    } else {
      // local weight is the volume of the largest rectangle in the restricted
      // module #pragma omp parallel for reduction(std::max: local_weight)
      for (const filtration_type &birth : birthList) {
        if (birth.num_parameters() == 0) continue;
        for (const filtration_type &death : deathList) {
          if (death.num_parameters() > 0) local_weight = std::max(local_weight, _rectangle_volume(birth, death));
        }
      }
      return local_weight / std::pow(2 * std::abs(delta), x.size());
    }
  }

  template<class Filtration_value_range>
  value_type distance_to_upper(const Filtration_value_range &x, bool negative) const {
    value_type distance_to_upper = std::numeric_limits<value_type>::infinity();
    for (const auto &death : death_corners_) {
      value_type temp = negative ? -std::numeric_limits<value_type>::infinity() : 0;
      for (auto i = 0u; i < death.size(); ++i) {
        temp = std::max(temp, x[i] - death[i]);
      }
      distance_to_upper = std::min(distance_to_upper, temp);
    }
    return distance_to_upper;
  }

  template<class Filtration_value_range>
  value_type distance_to_lower(const Filtration_value_range &x, bool negative) const {
    value_type distance_to_lower = std::numeric_limits<value_type>::infinity();
    for (const auto &birth : birth_corners_) {
      value_type temp = negative ? -std::numeric_limits<value_type>::infinity() : 0;
      for (auto i = 0u; i < birth.size(); ++i) {
        temp = std::max(temp, birth[i] - x[i]);
      }
      distance_to_lower = std::min(distance_to_lower, temp);
    }
    return distance_to_lower;
  }

  template<class Filtration_value_range>
  value_type distance_to(const Filtration_value_range &x, bool negative) const {
    return std::max(Summand_base::distance_to_lower(x, negative), Summand_base::distance_to_upper(x, negative));
  }

  template<class Filtration_value_range>
  std::tuple<int, int> distance_idx_to_upper(const Filtration_value_range &x) const {
    value_type distance_to_upper = std::numeric_limits<value_type>::infinity();
    int d_idx = -1;  // argmin_d max_i (x-death)
    int param = 0;
    auto count = 0u;
    for (const auto &death : death_corners_) {
      value_type temp = -std::numeric_limits<value_type>::infinity();  // max_i(death-x)_+
      int temp_idx = 0;
      for (auto i = 0u; i < death.size(); ++i) {
        auto plus = x[i] - death[i];
        if (plus > temp) {
          temp_idx = i;
          temp = plus;
        }
      }
      if (temp < distance_to_upper) {
        distance_to_upper = temp;
        param = temp_idx;
        d_idx = count;
      }
      ++count;
    }
    return {d_idx, param};
  }

  template<class Filtration_value_range>
  std::tuple<int, int> distance_idx_to_lower(const Filtration_value_range &x) const {
    value_type distance_to_lower = std::numeric_limits<value_type>::infinity();
    int b_idx = -1;  // argmin_b max_i (b-x)_x
    int param = 0;
    auto count = 0u;
    for (const auto &birth : birth_corners_) {
      value_type temp = -std::numeric_limits<value_type>::infinity();  // max_i(birth - x)_+
      int temp_idx = 0;
      for (auto i = 0u; i < birth.size(); ++i) {
        auto plus = birth[i] - x[i];
        if (plus > temp) {
          temp_idx = i;
          temp = plus;
        }
      }
      if (temp < distance_to_lower) {
        distance_to_lower = temp;
        param = temp_idx;
        b_idx = count;
      }
      ++count;
    }
    return {b_idx, param};
  }

  template<class Filtration_value_range>
  std::vector<int> distance_idx_to(const Filtration_value_range &x, bool full) const {
    const auto &[a, b] = distance_idx_to_lower(x);
    const auto &[c, d] = distance_idx_to_upper(x);
    if (full) [[unlikely]]
      return {a, b, c, d};
    else {
      return {a, c};
    }
  }

  std::pair<Point, Point> get_bar(const Line<value_type> &l) const {
    const bool verbose = false;
    if constexpr (verbose)
      std::cout << "Computing bar of this summand of dimension " << this->get_dimension() << std::endl;
    Point pushed_birth = std::numeric_limits<Point>::infinity();
    Point pushed_death = std::numeric_limits<Point>::minus_infinity();
    for (const auto &birth : this->get_birth_list()) {
      Point pb = l[l.compute_forward_intersection(birth)];
      if constexpr (verbose)
        std::cout << "Updating birth " << pushed_birth << " with " << pb << " pushed at " << birth << " "
                  << pushed_birth.is_plus_inf();
      if ((pb <= pushed_birth) || pushed_birth.is_plus_inf()) {
        pushed_birth.swap(pb);
        if constexpr (verbose) std::cout << " swapped !";
      }
      if constexpr (verbose) std::cout << std::endl;
    }
    //
    for (const auto &death : this->get_death_list()) {
      Point pd = l[l.compute_backward_intersection(death)];
      if constexpr (verbose)
        std::cout << "Updating death " << pushed_death << " with " << pd << " pushed at " << death << " "
                  << pushed_death.is_minus_inf() << pushed_death[0];
      if ((pd >= pushed_death) || pushed_death.is_minus_inf()) {
        pushed_death.swap(pd);
        if constexpr (verbose) std::cout << " swapped !";
      }
      if constexpr (verbose) std::cout << std::endl;
    }

    if (!(pushed_birth <= pushed_death)) {
      if constexpr (verbose) std::cout << "Birth <!= Death ! Ignoring this value" << std::endl;
      return {std::numeric_limits<Point>::infinity(), std::numeric_limits<Point>::infinity()};
    }
    if constexpr (verbose) {
      std::cout << "Final values" << pushed_birth << " ----- " << pushed_death << std::endl;
    }
    return {pushed_birth, pushed_death};
  }

  std::pair<value_type, value_type> get_bar2(const Line<value_type> &l) const {
    const bool verbose = false;
    if constexpr (verbose)
      std::cout << "Computing bar of this summand of dimension " << this->get_dimension() << std::endl;
    value_type pushed_birth = std::numeric_limits<value_type>::infinity();
    value_type pushed_death = -pushed_birth;
    for (const auto &birth : this->get_birth_list()) {
      value_type pb = l.compute_forward_intersection(birth);
      pushed_birth = std::min(pb, pushed_birth);
    }
    //
    for (const auto &death : this->get_death_list()) {
      value_type pd = l.compute_backward_intersection(death);
      pushed_death = std::max(pd, pushed_death);
    }

    if (!(pushed_birth <= pushed_death)) {
      if constexpr (verbose) std::cout << "Birth <!= Death ! Ignoring this value" << std::endl;
      return {inf, inf};
    }
    if constexpr (verbose) {
      std::cout << "Final values" << pushed_birth << " ----- " << pushed_death << std::endl;
    }
    return {pushed_birth, pushed_death};
  }

  /**
   * @brief Adds the bar @p bar to the indicator module @p summand if @p bar
   * is non-trivial (ie. not reduced to a point or, if @p threshold is true,
   * its thresholded version should not be reduced to a point) .
   *
   * @param bar p_bar: to add to the support of the summand
   * @param summand p_summand: indicator module which is being completed
   * @param basepoint p_basepoint: basepoint of the line of the bar
   * @param birth p_birth: birth container (for memory optimization purposes).
   * Has to be of the size @p basepoint.size()+1.
   * @param death p_death: death container. Same purpose as @p birth but for
   * deathpoint.
   * @param threshold p_threshold: If true, will threshold the bar with @p box.
   * @param box p_box: Only useful if @p threshold is set to true.
   */
  template<class CoordinateRange, class Filtration_value_range>
  void add_bar(value_type baseBirth,
               value_type baseDeath,
               const CoordinateRange &basepoint,
               Filtration_value_range &birth,
               Filtration_value_range &death,
               const bool threshold,
               const Box<value_type> &box) {
    // bar is trivial in that case
    if (baseBirth >= baseDeath) return;
    // #pragma omp simd
    // 		for (unsigned int j = 0; j < birth.size() - 1; j++)
    // 		{
    // 			birth[j] = basepoint[j] + baseBirth;
    // 			death[j] = basepoint[j] + baseDeath;
    // 		}
    // 		birth.back() = baseBirth;
    // 		death.back() = baseDeath;

    /* #pragma omp simd */
    for (unsigned int j = 0; j < birth.size() - 1; j++) {
      value_type temp = basepoint[j] + baseBirth;
      // The box is assumed to contain all of the filtration values, if its
      // outside, its inf.
      birth[j] = temp < box.get_lower_corner()[j] ? negInf : temp;
      temp = basepoint[j] + baseDeath;
      death[j] = temp > box.get_upper_corner()[j] ? inf : temp;
    }
    birth.back() = baseBirth < box.get_lower_corner().back() ? negInf : baseBirth;
    death.back() = baseDeath > box.get_upper_corner().back() ? inf : baseDeath;

    if (threshold) {
      // std::cout << box;
      threshold_down(birth, box, basepoint);
      threshold_up(death, box, basepoint);
    }
    _add_birth(birth);
    _add_death(death);
  }

  template<class Filtration_value_range>
  void add_bar(const Filtration_value_range &birth, const Filtration_value_range &death) {
    _add_birth(birth);
    _add_death(death);
  }

  template<class CoordinateRange>
  void add_bar(const CoordinateRange &basepoint, value_type birth, value_type death, const Box<value_type> &box) {
    const bool verbose = false;
    if (birth >= death) return;
    if constexpr (verbose) {
      std::cout << "Bar : " << basepoint + birth << "--" << basepoint + death << std::endl;
    }
    auto inf = std::numeric_limits<value_type>::infinity();
    auto container = basepoint + birth;
    for (auto i = 0u; i < container.size(); i++) {
      if (container[i] < box.get_lower_corner()[i]) container[i] = -inf;
    }
    _add_birth(container);
    container = basepoint + death;
    for (auto i = 0u; i < container.size(); i++) {
      if (container[i] > box.get_upper_corner()[i]) container[i] = inf;
    }
    _add_death(container);
  }

  const typename births_type::Generators &get_birth_list() const { return birth_corners_.get_underlying_container(); }

  const typename deaths_type::Generators &get_death_list() const { return death_corners_.get_underlying_container(); }

  const births_type &get_upset() const { return birth_corners_; }

  const deaths_type &get_downset() const { return death_corners_; };

  void clean() {
    // birth_corners_.erase(
    //     std::remove_if(birth_corners_.begin(), birth_corners_.end(),
    //                    [](const std::vector<value_type> &bp) {
    //                      return std::any_of(
    //                          bp.begin(), bp.end(),
    //                          [](float value) { return !std::isfinite(value); });
    //                    }),
    //     birth_corners_.end());
    // TODO : clean
  }

  void complete_birth(const value_type precision) {
    if (!birth_corners_.is_finite()) return;

    for (std::size_t i = 0; i < birth_corners_.num_generators(); i++) {
      for (std::size_t j = i + 1; j < birth_corners_.num_generators(); j++) {
        value_type dinf = d_inf(birth_corners_[i], birth_corners_[j]);
        if (dinf < 1.1 * precision) {
          _factorize_min(birth_corners_[i], birth_corners_[j]);
          birth_corners_[j] = filtration_type::inf();
        }
      }
    }
    _clean(birth_corners_);
  }

  void complete_death(const value_type precision) {
    if (!death_corners_.is_finite()) return;

    for (std::size_t i = 0; i < death_corners_.num_generators(); i++) {
      for (std::size_t j = i + 1; j < death_corners_.num_generators(); j++) {
        value_type d = d_inf(death_corners_[i], death_corners_[j]);
        if (d < 1.1 * precision) {
          _factorize_max(death_corners_[i], death_corners_[j]);
          death_corners_[j] = filtration_type::minus_inf();
        }
      }
    }
    _clean(death_corners_);
  }

  dimension_type get_dimension() const { return dimension_; }

  void set_dimension(dimension_type dimension) { dimension_ = dimension; }

  value_type get_landscape_value(const std::vector<value_type> &x) const {
    value_type out = 0;
    Box<value_type> trivial_box;
    filtration_type filX(x.begin(), x.end());
    for (const auto &b : this->birth_corners_) {
      for (const auto &d : this->death_corners_) {
        value_type value =
            std::min(this->_get_max_diagonal(b, filX, trivial_box), this->_get_max_diagonal(filX, d, trivial_box));
        out = std::max(out, value);
      }
    }
    return out;
  }

  friend void swap(Summand_base &sum1, Summand_base &sum2) {
    swap(sum1.birth_corners_, sum2.birth_corners_);
    swap(sum1.death_corners_, sum2.death_corners_);
    std::swap(sum1.distanceTo0_, sum2.distanceTo0_);
    std::swap(sum1.dimension_, sum2.dimension_);
    // 	std::swap(sum1.updateDistance_, sum2.updateDistance_);
  };

  bool contains(const filtration_type &x) const {
    bool out = false;
    for (const auto &birth : this->birth_corners_) {  // checks if there exists a birth smaller than x
      if (birth <= x) {
        out = true;
        break;
      }
    }
    if (!out) return false;
    out = false;
    for (const auto &death : this->death_corners_) {
      if (x <= death) {
        out = true;
        break;
      }
    }
    return out;
  }

  template<class Filtration_value_range>
  bool contains(const Filtration_value_range &x) const {
    //TODO: temporary, this don't really work with any kind of Filtration_value_range
    return contains(_convert_filtration_value<Filtration_value_range, filtration_type>(x, 0));
  }
private:
  template<class F1, class F2>
  static auto _convert_filtration_value(const F1 &x, int) -> decltype(&F1::reserve, &F2::fill, F2())
  {
    if (x.empty()) return F2(0);
    if (x.size() == 1) return F2(x[0]);
    return F2(x.begin(), x.end());
  }

  template<class F1, class F2>
  static auto _convert_filtration_value(const F1 &x, long) -> decltype(&F1::fill, &F2::reserve, F2())
  {
    F2 res(x.size());
    bool isInf = true, isMinusInf = true, isNaN = true;
    for (unsigned int i = 0; i < x.size(); ++i){
      res[i] = x[i];
      if (x[i] != F2::T_inf) isInf = false;
      if (x[i] != -F2::T_inf) isMinusInf = false;
      if (!std::isnan(x[i])) isNaN = false;
    }
    if (isInf || isMinusInf || isNaN) res.resize(1);
    return res;
  }
public:
  inline Box<value_type> get_bounds() const {
    using P = typename Box<value_type>::Point;
    if (birth_corners_.num_generators() == 0) return Box<value_type>();
    P m = P::inf();
    P M = P::minus_inf();
    for (const auto &corner : birth_corners_) {
      m.pull_to_greatest_common_lower_bound(corner, true);
    }
    for (const auto &corner : death_corners_) {
      M.push_to_least_common_upper_bound(corner, true);
    }
    // auto dimension = birth_corners_.num_parameters();
    // filtration_type m(dimension, std::numeric_limits<value_type>::infinity());
    // filtration_type M(dimension, -std::numeric_limits<value_type>::infinity());
    // for (const auto &corner : birth_corners_) {
    //   for (auto parameter = 0u; parameter < dimension; parameter++) {
    //     m[parameter] = std::min(m[parameter], corner[parameter]);
    //   }
    // }
    // for (const auto &corner : death_corners_) {
    //   for (auto parameter = 0u; parameter < dimension; parameter++) {
    //     auto corner_i = corner[parameter];
    //     if (corner_i != std::numeric_limits<value_type>::infinity())
    //       M[parameter] = std::max(M[parameter], corner[parameter]);
    //   }
    // }
    return Box(m, M);
  }

  inline void rescale(const std::vector<value_type> &rescale_factors) {
    if (birth_corners_.num_generators() == 0) return;
    auto dimension = birth_corners_.num_parameters();
    for (unsigned int i = 0; i < birth_corners_.num_generators(); ++i) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        birth_corners_[i][parameter] *= rescale_factors[parameter];
      }
    }
    for (unsigned int i = 0; i < death_corners_.num_generators(); ++i) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        death_corners_[i][parameter] *= rescale_factors[parameter];
      }
    }
  }

  inline void translate(const std::vector<value_type> &translation) {
    if (birth_corners_.num_generators() == 0) return;
    auto dimension = birth_corners_.num_parameters();
    for (unsigned int i = 0; i < birth_corners_.num_generators(); ++i) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        birth_corners_[i][parameter] += translation[parameter];
      }
    }
    for (unsigned int i = 0; i < death_corners_.num_generators(); ++i) {
      for (auto parameter = 0u; parameter < dimension; parameter++) {
        death_corners_[i][parameter] += translation[parameter];
      }
    }
  }

  template <class Out_summand>
  inline Out_summand grid_squeeze(const std::vector<std::vector<value_type>> &grid) const {
    auto dimension = this->get_dimension();
    Out_summand out(compute_coordinates_in_grid(birth_corners_, grid),
                    compute_coordinates_in_grid(death_corners_, grid),
                    dimension_);
    return out;
  }

 private:
  births_type birth_corners_;  // TODO : use Multi_critical_filtration
  deaths_type death_corners_;
  value_type distanceTo0_;
  dimension_type dimension_;

  const value_type inf = std::numeric_limits<value_type>::infinity();
  const value_type negInf = -1 * inf;

  void _compute_interleaving(const Box<value_type> &box) {
    distanceTo0_ = 0;
    /* #pragma omp parallel for reduction(max : distanceTo0_) */
    for (const auto &birth : birth_corners_) {
      for (const auto &death : death_corners_) {
        distanceTo0_ = std::max(distanceTo0_, _get_max_diagonal(birth, death, box));
      }
    }
  }

  /**
   * @brief Adds @p birth to the summand's @p birth_list if it is not induced
   * from the @p birth_list (ie. not comparable or smaller than another birth),
   * and removes unnecessary birthpoints (ie. birthpoints that are induced
   * by @p birth).
   *
   * @param birth_list p_birth_list: birthpoint list of a summand
   * @param birth p_birth: birth to add to the summand
   */
  template<class Filtration_value_range>
  void _add_birth(const Filtration_value_range &birth) {
    birth_corners_.add_generator(birth);
    return;

    // // TODO : DEPRECATE THIS OLD CODE
    // if (birth_corners_.empty()) {
    //   birth_corners_.push_back(birth);
    //   return;
    // }

    // for (const auto &current_birth : birth_corners_) {
    //   if (birth >= current_birth) {
    //     return;
    //   }
    // }
    // // this birth value is useful, we can now remove useless other filtrations
    // for (auto &current_birth : birth_corners_) {
    //   if ((!current_birth.empty()) && (birth <= current_birth)) {
    //     current_birth.clear();
    //   }
    // }

    // _clean(birth_corners_);
    // birth_corners_.push_back(birth);
  }

  /**
   * @brief Adds @p death to the summand's @p death_list if it is not induced
   * from the @p death_list (ie. not comparable or greater than another death),
   * and removes unnecessary deathpoints (ie. deathpoints that are induced
   * by @p death)
   *
   * @param death_list p_death_list: List of deathpoints of a summand
   * @param death p_death: deathpoint to add to this list
   */
  template<class Filtration_value_range>
  void _add_death(const Filtration_value_range &death) {
    death_corners_.add_generator(death);
    return;
    // // TODO:  Deprecate this old code
    // if (death_corners_.empty()) {
    //   death_corners_.push_back(death);
    //   return;
    // }

    // for (const auto &current_death : death_corners_) {
    //   if (death <= current_death) {
    //     return;
    //   }
    // }
    // // this death value is useful, we can now remove useless other filtrations
    // for (auto &current_death : death_corners_) {
    //   if (!current_death.empty() && (death >= current_death)) {
    //     current_death.clear();
    //   }
    // }
    // _clean(death_corners_);
    // death_corners_.push_back(death);
  }

  value_type _rectangle_volume(const filtration_type &a, const filtration_type &b) const {
    if constexpr (Debug::debug) assert(a.size() == b.size() && "Inputs must be of the same size !");
    value_type s = b[0] - a[0];
    for (unsigned int i = 1; i < a.size(); i++) {
      s = s * (b[i] - a[i]);
    }
    return s;
  }

  template<class Filtration_value_range1, class Filtration_value_range2>
  value_type _get_max_diagonal(const Filtration_value_range1 &birth,
                               const Filtration_value_range2 &death,
                               const Box<value_type> &box) const {
    // assumes birth and death to be never NaN
    if constexpr (Debug::debug)
      assert(!birth.is_finite() || !death.is_finite() ||
             birth.size() == death.size() && "Inputs must be of the same size !");

    value_type s = inf;
    bool threshold_flag = !box.is_trivial();
    if (threshold_flag) {
      unsigned int dim = std::max(birth.size(), box.dimension());
      for (unsigned int i = 0; i < dim; ++i) {
        value_type max_i = box.get_upper_corner().size() > i ? box.get_upper_corner()[i] : inf;
        value_type min_i = box.get_lower_corner().size() > i ? box.get_lower_corner()[i] : negInf;
        value_type t_death = death.is_plus_inf() ? max_i : (death.is_minus_inf() ? -inf : std::min(death[i], max_i));
        value_type t_birth = birth.is_plus_inf() ? inf : (birth.is_minus_inf() ? min_i : std::max(birth[i], min_i));
        s = std::min(s, t_death - t_birth);
      }
    } else {
      unsigned int dim = std::max(birth.size(), death.size());
      for (unsigned int i = 0; i < dim; i++) {
        // if they don't have the same size, then one of them has to (+/-)infinite.
        value_type t_death = death.size() > i ? death[i] : death[0];  // assumes death is never empty
        value_type t_birth = birth.size() > i ? birth[i] : birth[0];  // assumes birth is never empty
        s = std::min(s, t_death - t_birth);
      }
    }

    return s;
  }

  template<class Filtration_value_range>
  value_type d_inf(const Filtration_value_range &a, const Filtration_value_range &b) const {
    if (a.empty() || b.empty() || a.size() != b.size()) return inf;

    value_type d = std::abs(a[0] - b[0]);
    for (unsigned int i = 1; i < a.size(); i++) d = std::max(d, std::abs(a[i] - b[i]));

    return d;
  }

  template<class Filtration_value_range>
  void _factorize_min(Filtration_value_range &a, const Filtration_value_range &b) {
    /* if (Debug::debug && (a.empty() || b.empty())) */
    /* { */
    /* 	std::cout << "Empty corners ??\n"; */
    /* 	return; */
    /* } */

    for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::min(a[i], b[i]);
  }
  //TODO: find a better solution for this.
  template <unsigned int N, class Filtration_value_range>
  void _factorize_min(multi_filtration::One_critical_filtration_with_n_parameters_view<N, std::vector<value_type>> a,
                      const Filtration_value_range &b) {
    /* if (Debug::debug && (a.empty() || b.empty())) */
    /* { */
    /* 	std::cout << "Empty corners ??\n"; */
    /* 	return; */
    /* } */

    for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::min(a[i], b[i]);
  }

  template<class Filtration_value_range>
  void _factorize_max(Filtration_value_range &a, const Filtration_value_range &b) {
    /* if (Debug::debug && (a.empty() || b.empty())) */
    /* { */
    /* 	std::cout << "Empty corners ??\n"; */
    /* 	return; */
    /* } */

    for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::max(a[i], b[i]);
  }
  //TODO: find a better solution for this.
  template <unsigned int N, class Filtration_value_range>
  void _factorize_max(multi_filtration::One_critical_filtration_with_n_parameters_view<N, std::vector<value_type>> a,
                      const Filtration_value_range &b) {
    /* if (Debug::debug && (a.empty() || b.empty())) */
    /* { */
    /* 	std::cout << "Empty corners ??\n"; */
    /* 	return; */
    /* } */

    for (unsigned int i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::max(a[i], b[i]);
  }

  // WARNING Does permute the output.
  /**
   * @brief Cleans empty entries of a corner list
   *
   * @param list corner list to clean
   * @param keep_sort If true, will keep the order of the corners,
   * with a computational overhead. Defaults to false.
   */
  static void _clean(std::vector<filtration_type> &list, bool keep_inf = true) {
    list.erase(std::remove_if(list.begin(),
                              list.end(),
                              [keep_inf](filtration_type &a) {
                                return a.empty() || ((!keep_inf) && (a.is_plus_inf() || a.is_minus_inf()));
                              }),
               list.end());
  }

  static inline void _clean(births_type &list, bool remove_inf = true) { list.remove_empty_generators(remove_inf); }

  static inline void _clean(deaths_type &list, bool remove_inf = true) { list.remove_empty_generators(remove_inf); }
};

template <typename value_type>
using Summand = Summand_base<Gudhi::multi_filtration::Multi_critical_filtration<value_type, false>,
                             Gudhi::multi_filtration::Multi_critical_filtration<value_type, true>,
                             Gudhi::multi_filtration::One_critical_filtration<value_type>>;
template <typename value_type, unsigned int N>
using Fixed_parameter_summand =
    Summand_base<Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<value_type, N, false>,
                 Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<value_type, N, true>,
                 Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<value_type, N> >;
template <typename value_type>
using Two_parameter_summand =
    Summand_base<Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<value_type, 2, false>,
                 Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<value_type, 2, true>,
                 Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<value_type, 2> >;

}  // namespace Gudhi::multiparameter::mma

#endif  // MMA_SUMMAND_H_INCLUDED
