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
 * @file Summand.h
 * @author David Loiseaux
 * @brief Contains the @ref Gudhi::multi_persistence::Summand class.
 */

#ifndef MP_SUMMAND_H_INCLUDED
#define MP_SUMMAND_H_INCLUDED

#include <cstddef>    // std::size_t
#include <ostream>    //std::ostream
#include <stdexcept>  //std::invalid_argument, std::runtime_error
#include <utility>

#ifdef GUDHI_USE_TBB
#include <oneapi/tbb/parallel_for.h>
#endif

#include <gudhi/Debug_utils.h>
#include <gudhi/Dynamic_multi_parameter_filtration.h>
#include <gudhi/Multi_filtration/multi_filtration_utils.h>
#include <gudhi/Multi_filtration/multi_filtration_conversions.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Line.h>

namespace Gudhi {
namespace multi_persistence {

/**
 * @class Summand Summand.h gudhi/Multi_persistence/Summand.h
 * @ingroup multi_persistence
 *
 * @brief
 *
 * @tparam T
 */
template <typename T, typename D = int>
class Summand {
 public:
  using value_type = T;
  using Births = Gudhi::multi_filtration::Dynamic_multi_parameter_filtration<value_type, false, false>;
  using Deaths = Gudhi::multi_filtration::Dynamic_multi_parameter_filtration<value_type, true, false>;
  using Dimension = D;
  using Index = Births::size_type;

  static constexpr T T_inf = Births::T_inf;
  static constexpr T T_m_inf = Births::T_m_inf;

  Summand(int numberOfParameters = 1)
      : birthCorners_(numberOfParameters, T_inf),
        deathCorners_(numberOfParameters, T_m_inf),
        interleaving_(get_null_value<value_type>()),
        dimension_(get_null_value<Dimension>()) {}

  Summand(const Births &birthCorners, const Deaths &deathCorners, Dimension dimension)
      : birthCorners_(birthCorners),
        deathCorners_(deathCorners),
        interleaving_(get_null_value<value_type>()),
        dimension_(dimension) {}

  // Builds filtration value with given number of parameters and values from the given range. Lets \f$ p \f$
  // be the number of parameters. The \f$ p \f$ first elements of the range have to correspond to the first generator,
  // the \f$ p \f$ next elements to the second generator and so on... So the length of the range has to be a multiple
  // of \f$ p \f$ and the number of generators will be \f$ length / p \f$. The range is represented by two iterators.
  template <class ValueRange = std::initializer_list<value_type>>
  Summand(const ValueRange &birthCorners, const ValueRange &deathCorners, int numberOfParameters, Dimension dimension)
      : birthCorners_(birthCorners.begin(), birthCorners.end(), numberOfParameters),
        deathCorners_(deathCorners.begin(), deathCorners.end(), numberOfParameters),
        interleaving_(get_null_value<value_type>()),
        dimension_(dimension) {}

  Dimension get_dimension() const { return dimension_; }

  void set_dimension(Dimension dimension) { dimension_ = dimension; }

  template <class MultiFiltrationValue>
  bool contains(const MultiFiltrationValue &x) const {
    auto xPos = Gudhi::multi_filtration::as_type<Births>(x);
    auto xNeg = Gudhi::multi_filtration::as_type<Deaths>(x);
    return birthCorners_ <= x && deathCorners_ <= x;
  }

  Box<value_type> compute_bounds() const {
    if (birthCorners_.num_generators() == 0) return {};

    auto numParam = birthCorners_.num_parameters();
    typename Box<value_type>::Point_t m(numParam, T_inf);
    typename Box<value_type>::Point_t M(numParam, T_m_inf);

    for (Index g = 0; g < birthCorners_.num_generators(); ++g) {
      for (Index p = 0; p < numParam; ++p) {
        m[p] = std::min(m[p], birthCorners_(g, p));
      }
    }
    for (Index g = 0; g < deathCorners_.num_generators(); ++g) {
      for (Index p = 0; p < numParam; ++p) {
        auto corner_i = deathCorners_(g, p);
        if (corner_i != T_inf) M[p] = std::max(M[p], corner_i);
      }
    }

    return Box<value_type>(m, M);
  }

  const Births &get_upset() const { return birthCorners_; }

  const Deaths &get_downset() const { return deathCorners_; }

  template <class MultiFiltrationValue>
  std::vector<MultiFiltrationValue> compute_birth_list() const {
    return _compute_list(birthCorners_);
  }

  template <class MultiFiltrationValue>
  std::vector<MultiFiltrationValue> compute_death_list() const {
    return _compute_list(deathCorners_);
  }

  void complete_birth(value_type precision) {
    if (!birthCorners_.is_finite()) return;

    const value_type error = 0.99;

    for (Index i = 0; i < birthCorners_.num_generators(); i++) {
      for (Index j = i + 1; j < birthCorners_.num_generators(); j++) {
        if (_d_inf(birthCorners_[i], birthCorners_[j]) < error * precision) {  // for machine error ?
          _factorize_min(birthCorners_[i], birthCorners_[j]);
          birthCorners_[j] = Births::Generator::inf();
          i++;
        }
      }
    }
    birthCorners_.simplify();
  }

  void complete_death(value_type precision) {
    if (!deathCorners_.is_finite()) return;

    const value_type error = 0.99;

    for (Index i = 0; i < deathCorners_.num_generators(); i++) {
      for (Index j = i + 1; j < deathCorners_.num_generators(); j++) {
        if (_d_inf(deathCorners_[i], deathCorners_[j]) < error * precision) {  // for machine error ?
          _factorize_max(deathCorners_[i], deathCorners_[j]);
          deathCorners_[j] = Deaths::Generator::minus_inf();
          i++;
        }
      }
    }
    deathCorners_.simplify();
  }

  std::pair<value_type, value_type> get_bar(const Line<value_type> &l) const {
    value_type pushedBirth = T_inf;
    value_type pushedDeath = T_m_inf;

    for (const auto &birth : birthCorners_) {
      value_type pb = l.compute_forward_intersection(birth.begin(), birth.end());
      pushedBirth = std::min(pb, pushedBirth);
    }
    for (const auto &death : deathCorners_) {
      value_type pd = l.compute_backward_intersection(death.begin(), death.end());
      pushedDeath = std::max(pd, pushedDeath);
    }

    // !(<=) in case there is a NaN ?
    if (!(pushedBirth <= pushedDeath)) {
      return {T_inf, T_inf};
    }
    return {pushedBirth, pushedDeath};
  }

  // TODO: generalize as a "GeneratorRange" like for `add_generator` ?
  // Not usefull for multipers right now, just for C++ interface purposes
  void add_bar(const Births::Generator &birth, const Deaths::Generator &death) {
    birthCorners_.add_generator(birth);
    deathCorners_.add_generator(death);
  }

  value_type update_and_get_interleaving(const Box<value_type> &box) {
    interleaving_ = _compute_interleaving(box);
    return interleaving_;
  }

  value_type get_interleaving() const { return interleaving_; }

  template <class RandomAccessValueRange>
  value_type compute_distance_to(const RandomAccessValueRange &x, bool negative) const {
    value_type lowerDist = std::get<0>(
        _compute_distance_to_front(x, birthCorners_, negative, [](value_type cornerVal, value_type xVal) -> value_type {
          return cornerVal - xVal;
        }));
    value_type upperDist = std::get<0>(
        _compute_distance_to_front(x, deathCorners_, negative, [](value_type cornerVal, value_type xVal) -> value_type {
          return xVal - cornerVal;
        }));
    return std::max(lowerDist, upperDist);
  }

  template <class RandomAccessValueRange>
  std::vector<Index> compute_lower_and_upper_generator_of(const RandomAccessValueRange &x, bool full) const {
    [[maybe_unused]] auto [lowerDist, lowerGen, lowerParam] = _compute_distance_to_front(
        x, birthCorners_, true, [](value_type cornerVal, value_type xVal) -> value_type { return cornerVal - xVal; });
    [[maybe_unused]] auto [upperDist, upperGen, upperParam] = _compute_distance_to_front(
        x, deathCorners_, true, [](value_type cornerVal, value_type xVal) -> value_type { return xVal - cornerVal; });

    if (full) return {lowerGen, lowerParam, upperGen, upperParam};
    return {lowerGen, upperGen};
  }

  template <class RandomAccessValueRange>
  value_type get_local_weight(const RandomAccessValueRange &x, value_type delta) const {
    using P = Box<value_type>::Point_t;

    GUDHI_CHECK(x.size() == birthCorners_.num_parameters(),
                std::invalid_argument("Input range does not have the right size."));

    bool rectangle = delta <= 0;

    // box on which to compute the local weight
    P mini(x.size());
    P maxi(x.size());
    for (Index i = 0; i < x.size(); i++) {
      mini[i] = rectangle ? x[i] + delta : x[i] - delta;
      maxi[i] = rectangle ? x[i] - delta : x[i] + delta;
    }
    Box<value_type> threshold(rectangle ? maxi : mini, rectangle ? mini : maxi);

    value_type localWeight = 0;

    if (rectangle) {
      // local weight is the volume of the largest rectangle in the restricted
      for (const auto &birth : birthCorners_) {
        for (const auto &death : deathCorners_) {
          localWeight = std::max(localWeight, _rectangle_volume(birth, death, threshold));
        }
      }
      return localWeight / std::pow(2 * std::abs(delta), x.size());
    }

    // local weight is interleaving to 0 of module restricted to the square
    localWeight = _compute_interleaving(threshold);
    return localWeight / (2 * std::abs(delta));
  }

  template <class RandomAccessValueRange>
  value_type get_landscape_value(const RandomAccessValueRange &x) const {
    value_type landscapeValue = 0;
    Box<value_type> trivialBox;
    for (const auto &birth : birthCorners_) {
      for (const auto &death : deathCorners_) {
        value_type value = std::min(_get_max_diagonal(birth, x, trivialBox), _get_max_diagonal(x, death, trivialBox));
        landscapeValue = std::max(landscapeValue, value);
      }
    }
    return landscapeValue;
  }

  void rescale(const std::vector<value_type> &rescaleFactors) {
    _transform(rescaleFactors, [](value_type &cornerVal, value_type fact) -> value_type { return cornerVal *= fact; });
  }

  void translate(const std::vector<value_type> &translation) {
    _transform(translation, [](value_type &cornerVal, value_type fact) -> value_type { return cornerVal += fact; });
  }

  void evaluate_in_grid(const std::vector<std::vector<value_type>> &grid) {
    if (birthCorners_.num_generators() == 0) return;

    auto snap = [](value_type x) -> std::size_t {
      value_type a = std::floor(x);
      value_type b = std::ceil(x);
      if (x - a < b - x) return a;
      return b;
    };
    auto evaluate_generator = [&](auto &corners) {
      return [&](std::size_t g) {
        for (Index p = 0; p < corners.num_parameters(); ++p) {
          GUDHI_CHECK(corners(g, p) >= 0, std::runtime_error("Values in the corners have to be positive."));
          if (corners(g, p) != T_inf) {
            std::size_t snapped = snap(corners(g, p));
            corners(g, p) = (snapped >= grid[p].size() ? T_inf : grid[p][snapped]);
          }
        }
      };
    };

#ifdef GUDHI_USE_TBB
    tbb::parallel_for(Index(0), birthCorners_.num_generators(), evaluate_generator(birthCorners_));
    if (deathCorners_.num_generators() == 0) return;
    tbb::parallel_for(Index(0), deathCorners_.num_generators(), evaluate_generator(deathCorners_));
#else
    for (Index g = 0; g < birthCorners_.num_generators(); ++g) {
      evaluate_generator(birthCorners_)(g);
    }
    for (Index g = 0; g < deathCorners_.num_generators(); ++g) {
      evaluate_generator(deathCorners_)(g);
    }
#endif
  }

  template <typename Y>
  static constexpr Y get_null_value() {
    return -1;
  }

  friend bool operator==(const Summand &a, const Summand &b);

  /**
   * @brief Outstream operator.
   */
  friend std::ostream &operator<<(std::ostream &stream, const Summand &s)
  {
    stream << "Summand:\n";
    stream << "Dimension: " << s.dimension_ << "\n";
    stream << "Birth corners:\n";
    stream << s.birthCorners_ << "\n";
    stream << "Death corners:\n";
    stream << s.deathCorners_ << "\n";

    return stream;
  }

  friend void swap(Summand &sum1, Summand &sum2) noexcept {
    std::swap(sum1.birthCorners_, sum2.birthCorners_);
    std::swap(sum1.deathCorners_, sum2.deathCorners_);
    std::swap(sum1.interleaving_, sum2.interleaving_);
    std::swap(sum1.dimension_, sum2.dimension_);
  }

 private:
  Births birthCorners_;
  Deaths deathCorners_;
  value_type interleaving_;
  Dimension dimension_;

  value_type _compute_interleaving(const Box<value_type> &box) {
    value_type interleaving = 0;
    for (const auto &birth : birthCorners_) {
      for (const auto &death : deathCorners_) {
        interleaving = std::max(interleaving, _get_max_diagonal(birth, death, box));
      }
    }
    return interleaving;
  }

  template <class F>
  void _transform(const std::vector<value_type> &factors, F &&operate) {
    if (birthCorners_.num_generators() == 0) return;

    auto dimension = birthCorners_.num_parameters();

    GUDHI_CHECK(dimension <= factors.size(), std::invalid_argument("Not enough factors in input."));

    // TODO: parallelize?
    for (unsigned int g = 0; g < birthCorners_.num_generators(); ++g) {
      for (unsigned int p = 0; p < dimension; ++p) {
        std::forward<F>(operate)(birthCorners_(g, p), factors[p]);
      }
    }
    for (unsigned int g = 0; g < deathCorners_.num_generators(); ++g) {
      for (unsigned int p = 0; p < dimension; ++p) {
        std::forward<F>(operate)(deathCorners_(g, p), factors[p]);
      }
    }
  }

  template <class MultiFiltrationValue, class Corners>
  static std::vector<MultiFiltrationValue> _compute_list(const Corners &corners) {
    std::vector<MultiFiltrationValue> res(corners.num_generators());

    for (Index g = 0; g < corners.num_generators(); ++g) {
      res[g] = as_type<MultiFiltrationValue>(corners[g], corners.num_parameters());
      // could be done in a more generic way, but this should be the fastest without changing the current interface
      if constexpr (Gudhi::multi_filtration::RangeTraits<MultiFiltrationValue>::is_dynamic_multi_filtration)
        res[g].force_generator_size_to_number_of_parameters(0);
    }

    return res;
  }

  // TODO: better name?
  template <class Generator>
  static value_type _d_inf(const Generator &a, const Generator &b) {
    if (a.size() == 0 || b.size() == 0 || a.size() != b.size()) return T_inf;

    value_type d = std::abs(a[0] - b[0]);
    for (Index i = 1; i < a.size(); i++) d = std::max(d, std::abs(a[i] - b[i]));

    return d;
  }

  template <class Generator>
  static void _factorize_min(Generator &a, const Generator &b) {
    for (Index i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::min(a[i], b[i]);
  }

  template <class Generator>
  static void _factorize_max(Generator &a, const Generator &b) {
    for (Index i = 0; i < std::min(b.size(), a.size()); i++) a[i] = std::max(a[i], b[i]);
  }

  // TODO: moving into Box class? But then, all the Generator management (get_val) will feel very artificial...
  template <class RandomAccessValueRange>
  static value_type _get_max_diagonal(const RandomAccessValueRange &birth,
                                      const RandomAccessValueRange &death,
                                      const Box<value_type> &box) {
    // assumes birth and death to be never NaN
    GUDHI_CHECK(birth.size() == 1 || death.size() == 1 || birth.size() == death.size(),
                std::invalid_argument("Inputs must be of the same size !"));

    bool useThreshold = !box.is_trivial();

    GUDHI_CHECK((birth.size() == 1 && death.size() == 1) || !useThreshold || birth.size() == box.get_dimension() ||
                    death.size() == box.get_dimension(),
                std::invalid_argument("Inputs must be of the same size !"));

    auto get_val = [](const RandomAccessValueRange &r, Index i) -> value_type {
      if (i < r.size()) return r[i];
      // never used if r.size() == 0
      return r[0];
    };

    value_type diag = T_inf;
    if (useThreshold) {
      for (Index i = 0; i < birth.size(); ++i) {
        value_type max_i = box.get_upper_corner()[i];
        value_type min_i = box.get_lower_corner()[i];
        value_type t_death = std::min(get_val(death, i), max_i);
        value_type t_birth = std::max(get_val(birth, i), min_i);
        diag = std::min(diag, t_death - t_birth);
      }
    } else {
      for (Index i = 0; i < birth.size(); i++) {
        diag = std::min(diag, get_val(death, i) - get_val(birth, i));
      }
    }

    return diag;
  }

  template <class RandomAccessValueRange, class Corners, typename F>
  static std::tuple<value_type, Index, Index> _compute_distance_to_front(const RandomAccessValueRange &x,
                                                                         const Corners &corners,
                                                                         bool negative,
                                                                         F &&diff) {
    value_type distance = T_inf;
    Index gen, param;

    for (Index g = 0; g < corners.num_generators(); ++g) {
      value_type tempDist = negative ? T_m_inf : 0;
      Index tempParam = 0;
      for (Index p = 0; p < corners.num_parameters(); ++p) {
        value_type d = std::forward<F>(diff)(corners(g, p), x[p]);
        if (tempDist < d) {
          tempDist = d;
          tempParam = p;
        }
      }
      if (distance > tempDist) {
        distance = tempDist;
        gen = g;
        param = tempParam;
      }
    }

    return {distance, gen, param};
  }

  static value_type _rectangle_volume(const Births::Generator &birth,
                                      const Deaths::Generator &death,
                                      const Box<value_type> &box) {
    // NaN?
    if (birth.size() == 0 || death.size() == 0) return 0;

    auto get_val = [](const auto &r, Index i) -> value_type {
      if (i < r.size()) return r[i];
      // never used if r.size() == 0
      return r[0];
    };

    value_type volume = std::min(death[0], box.get_upper_corner()[0]) - std::max(birth[0], box.get_lower_corner()[0]);
    for (Index i = 1; i < birth.size(); i++) {
      value_type t_death = std::min(get_val(death, i), box.get_upper_corner()[i]);
      value_type t_birth = std::max(get_val(birth, i), box.get_lower_corner()[i]);
      volume = volume * (t_death - t_birth);
    }
    return volume;
  }
};

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_SUMMAND_H_INCLUDED
