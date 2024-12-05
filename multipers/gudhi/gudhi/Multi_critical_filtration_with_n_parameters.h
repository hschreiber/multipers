/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2024 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file Multi_critical_filtration_with_n_parameters.h
 * @author Hannah Schreiber
 * @brief Contains the @ref Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters class.
 */

#ifndef MULTI_CRITICAL_FILTRATIONS_WITH_N_PARAMETERS_H_
#define MULTI_CRITICAL_FILTRATIONS_WITH_N_PARAMETERS_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gudhi/Debug_utils.h>
#include <gudhi/One_critical_filtration_with_n_parameters.h>
#include <gudhi/One_critical_filtration_with_n_parameters_view.h>
#include <gudhi/flat_2D_matrix.h>

namespace Gudhi::multi_filtration {

/**
 * @class Multi_critical_filtration_with_n_parameters multi_critical_filtration.h
 * gudhi/multi_critical_filtration.h
 * @ingroup multi_filtration
 *
 * @brief Class encoding the different generators, i.e., apparition times, of a \f$k\f$-critical
 * \f$\mathbb R^n\f$-filtration value, e.g., the filtration of a simplex, or of the algebraic generator of a module
 * presentation. The class can be used as a vector whose indices correspond each to a generator, i.e., a one-critical
 * filtration value. Then, the indices of each generator correspond to a particular parameter.
 * E.g., \f$ f[i][p] \f$ will be  \f$ p^{\textit{th}} \f$ parameter of the \f$ i^{\textit{th}} \f$ generator
 * of this filtration value with @ref Multi_critical_filtration_with_n_parameters \f$ f \f$.
 *
 * @details Overloads `std::numeric_limits` such that:
 * - `std::numeric_limits<Multi_critical_filtration_with_n_parameters<T,co> >::has_infinity` returns
 * `true`,
 * - `std::numeric_limits<Multi_critical_filtration_with_n_parameters<T,co> >::infinity()` returns
 * @ref Multi_critical_filtration_with_n_parameters<T,co>::inf() "",
 * - `std::numeric_limits<Multi_critical_filtration_with_n_parameters<T,co> >::minus_infinity()` returns
 *   @ref Multi_critical_filtration_with_n_parameters<T,co>::minus_inf() "",
 * - `std::numeric_limits<Multi_critical_filtration_with_n_parameters<T,co> >::max()` throws,
 * - `std::numeric_limits<Multi_critical_filtration_with_n_parameters<T,co> >::max(g,n)` returns a @ref
 * Multi_critical_filtration_with_n_parameters<T,co> with `g` generators of `n` parameters evaluated at
 * value `std::numeric_limits<T>::max()`,
 * - `std::numeric_limits<Multi_critical_filtration_with_n_parameters<T,co> >::quiet_NaN()` returns
 * @ref Multi_critical_filtration_with_n_parameters<T,co>::nan() "".
 *
 * Multi-critical filtrations are filtrations such that the lifetime of each object is union of positive cones in
 * \f$\mathbb R^n\f$, e.g.,
 *  - \f$ \{ x \in \mathbb R^2 : x \ge (1,2)\} \cap \{ x \in \mathbb R^2 : x \ge (2,1)\} \f$ is finitely critical,
 *    and more particularly 2-critical, while
 *  - \f$ \{ x \in \mathbb R^2 : x \ge \mathrm{epigraph}(y \mapsto e^{-y})\} \f$ is not.
 *
 * The particular case of 1-critical filtrations is handled by @ref One_critical_filtration "".
 *
 * @tparam T Arithmetic type of an entry for one parameter of a filtration value. Has to be **signed** and
 * to implement `std::isnan(T)`, `std::numeric_limits<T>::has_quiet_NaN`, `std::numeric_limits<T>::quiet_NaN()`,
 * `std::numeric_limits<T>::has_infinity`, `std::numeric_limits<T>::infinity()` and `std::numeric_limits<T>::max()`.
 * If `std::numeric_limits<T>::has_infinity` returns `false`, a call to `std::numeric_limits<T>::infinity()`
 * can simply throw. Examples are the native types `double`, `float` and `int`.
 * @tparam co If `true`, reverses the poset order, i.e., the order \f$ \le \f$  in \f$ \mathbb R^n \f$ becomes
 * \f$ \ge \f$.
 */
template <typename T, unsigned int N, bool co = false>
class Multi_critical_filtration_with_n_parameters {
 public:
  using Container = std::vector<T>;
  /**
   * @brief Type of the origin of a "lifetime cone". Common with @ref One_critical_filtration "".
   */
  using Generator = One_critical_filtration_with_n_parameters_view<N, Container>;
  using Generators =
      Dynamic_flat_2D_matrix<N, T, Container, Generator>;     /**< Container type for the filtration values. */
  using iterator = typename Generators::iterator;             /**< Iterator type for the generator container. */
  using const_iterator = typename Generators::const_iterator; /**< Const iterator type for the generator container. */

  // CONSTRUCTORS

  /**
   * @brief Default constructor. The constructed value will be either at infinity if `co` is true or at minus infinity
   * if `co` is false.
   */
  Multi_critical_filtration_with_n_parameters() : multi_filtration_(1u, _get_default_value()) {};
  /**
   * @brief Constructs a filtration value with one generator and @p n parameters.
   * All parameters will be initialized at -inf if `co` is false and at inf if `co` is true.
   *
   * @warning The generator `{-inf, -inf, ...}`/`{inf, inf, ...}` with \f$ n > 1 \f$ entries is not considered as
   * "(minus) infinity" (the resp. methods @ref is_minus_inf() and @ref is_plus_inf() "", as well as the ones of the
   * generator, will not return true). The `-inf/inf` are just meant as placeholders, at least one entry should be
   * modified by the user.
   * Otherwise, either use the static methods @ref minus_inf() or @ref inf(), or set @p n to 1 instead.
   *
   * @param n Number of parameters.
   */
  Multi_critical_filtration_with_n_parameters(T value) : multi_filtration_(1u, value) {};
  /**
   * @brief Constructs a filtration value with one generator which will be initialzed by the given vector.
   *
   * @param v Vector with values for each parameter.
   */
  template <class Value_range = std::initializer_list<T>,
            class = std::enable_if_t<!std::is_arithmetic_v<Value_range> && !std::is_same_v<Value_range, T> > >
  Multi_critical_filtration_with_n_parameters(const Value_range &v) : multi_filtration_(1u, v){};
  /**
   * @brief Constructs filtration value with as many generators than elements in the given vector and initialize
   * them with them.
   * If the vector is empty, then the filtration value is either initialized at infinity if `co` is true or at
   * minus infinity if `co` is false.
   * @pre All generators in the vector have to have the same number of parameters, i.e., size.
   * Furthermore, the generators have to be a minimal generating set.
   *
   * @warning If the set of generators is not minimal or not sorted, the behaviour of most methods is undefined.
   * It is possible to call @ref simplify() after construction if there is a doubt to ensure this property.
   *
   * @param v Vector of generators.
   */
  Multi_critical_filtration_with_n_parameters(const std::vector<One_critical_filtration_with_n_parameters<T, N> > &v)
      : multi_filtration_(v.size() == 0 ? 1 : v.size()) {
    if (v.size() == 0) {
      multi_filtration_[0].fill(_get_default_value());
      return;
    }
    for (unsigned int g = 0; g < v.size(); ++g) {
      multi_filtration_[g] = v[g];
    }
  };
  /**
   * @brief Constructs filtration value with as many generators than elements in the given vector and moves those
   * elements to initialize the generators.
   * If the vector is empty, then the filtration value is either initialized at infinity if `co` is true or at
   * minus infinity if `co` is false.
   * @pre All generators in the vector have to have the same number of parameters, i.e., size.
   * Furthermore, the generators have to be a minimal generating set.
   *
   * @warning If the set of generators is not minimal or not sorted, the behaviour of most methods is undefined.
   * It is possible to call @ref simplify() after construction if there is a doubt to ensure this property.
   *
   * @param v Vector of generators.
   */
  Multi_critical_filtration_with_n_parameters(Container &&v) : multi_filtration_(std::move(v)) {};
  Multi_critical_filtration_with_n_parameters(const Container &v) : multi_filtration_(v) {};
  Multi_critical_filtration_with_n_parameters(Generators &&v) : multi_filtration_(std::move(v)) {};
  Multi_critical_filtration_with_n_parameters(const Generators &v) : multi_filtration_(v) {};
  /**
   * @brief Constructs a filtration value with one generator initialzed by the range given by the begin and end
   * iterators.
   *
   * @param it_begin Start of the range.
   * @param it_end End of the range.
   */
  template <typename Iterator>
  Multi_critical_filtration_with_n_parameters(Iterator it_begin, Iterator it_end)
      : multi_filtration_(it_begin, it_end){};

  // VECTOR-LIKE

  using value_type = T; /**< Entry type. */

  T operator()(std::size_t g, std::size_t n) { return multi_filtration_(g, n); }

  /**
   * @brief Standard operator[].
   */
  Generator operator[](std::size_t i) { return multi_filtration_[i]; }
  /**
   * @brief Standard operator[] const.
   */
  const Generator operator[](std::size_t i) const { return multi_filtration_[i]; }

  /**
   * @brief Returns begin iterator of the generator range.
   *
   * @warning If the generator is modified and the new set of generators is not minimal or not sorted, the behaviour
   * of most methods is undefined. It is possible to call @ref simplify() after construction if there is a doubt to
   * ensure this property.
   */
  iterator begin() { return multi_filtration_.begin(); }

  /**
   * @brief Returns end iterator of the generator range.
   *
   * @warning If the generator is modified and the new set of generators is not minimal or not sorted, the behaviour
   * of most methods is undefined. It is possible to call @ref simplify() after construction if there is a doubt to
   * ensure this property.
   */
  iterator end() { return multi_filtration_.end(); }

  /**
   * @brief Returns begin const iterator of the generator range.
   */
  const_iterator begin() const { return multi_filtration_.begin(); }

  /**
   * @brief Returns end const iterator of the generator range.
   */
  const_iterator end() const { return multi_filtration_.end(); }

  /**
   * @brief Reserves space for the given number of generators in the underlying container.
   *
   * @param n Number of generators.
   */
  void reserve(std::size_t g) { multi_filtration_.reserve(g); }

  // CONVERTERS

  /**
   * @brief Casts the object into the type of a generator.
   * @pre The filtration value is 1-critical. If there are more than one generator, only the first will be preserved
   * and if there is no generator, the method will segfault.
   */
  operator Generator() {
    if constexpr (num_generators() != 1)
      throw std::logic_error("Casting a " + std::to_string(num_generators()) +
                             "-critical filtration value into an 1-critical filtration value.");
    else
      return multi_filtration_[0];
  }

  /**
   * @brief Casts the object into the type of a generator.
   * @pre The filtration value is 1-critical. If there are more than one generator, only the first will be preserved
   * and if there is no generator, the method will segfault.
   */
  operator const Generator() const {
    if constexpr (num_generators() != 1)
      throw std::logic_error("Casting a " + std::to_string(num_generators()) +
                             "-critical filtration value into an 1-critical filtration value.");
    else
      return multi_filtration_[0];
  }

  // like numpy
  /**
   * @brief Returns a copy with entries casted into the type given as template parameter.
   *
   * @tparam U New type for the entries.
   * @return Copy with new entry type.
   */
  template <typename U>
  Multi_critical_filtration_with_n_parameters<U, N, co> as_type() const {
    const auto &container = multi_filtration_.get_underlying_container();
    typename Multi_critical_filtration_with_n_parameters<U, N, co>::Container copy(container.begin(), container.end());
    Multi_critical_filtration_with_n_parameters<U, N, co> out(std::move(copy));
    return out;
  }

  // ACCESS

  /**
   * @brief Returns a reference to the underlying container storing the generators.
   *
   * @warning If a generator is modified and the new set of generators is not minimal or not sorted, the behaviour
   * of most methods is undefined. It is possible to call @ref simplify() after construction if there is a doubt to
   * ensure this property.
   */
  const Generators &get_underlying_container() const { return multi_filtration_; }

  /**
   * @brief Returns the number of parameters.
   */
  constexpr std::size_t num_parameters() const { return N; }

  /**
   * @brief Returns the number of generators.
   */
  std::size_t num_generators() const { return multi_filtration_.size(); }

  std::size_t num_entries() const { return N * multi_filtration_.size(); }

  /**
   * @brief Returns a filtration value for which @ref is_plus_inf() returns `true`.
   *
   * @return Infinity.
   */
  constexpr static Multi_critical_filtration_with_n_parameters inf() {
    return Multi_critical_filtration_with_n_parameters(Generator::T_inf);
  }

  /**
   * @brief Returns a filtration value for which @ref is_minus_inf() returns `true`.
   *
   * @return Minus infinity.
   */
  constexpr static Multi_critical_filtration_with_n_parameters minus_inf() {
    return Multi_critical_filtration_with_n_parameters(-Generator::T_inf);
  }

  /**
   * @brief Returns a filtration value for which @ref is_nan() returns `true`.
   *
   * @return NaN.
   */
  constexpr static Multi_critical_filtration_with_n_parameters nan() {
    if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
      return Multi_critical_filtration_with_n_parameters(std::numeric_limits<T>::quiet_NaN());
    } else {
      throw std::logic_error("No NaN value exists.");
    }
  }

  // DESCRIPTORS

  /**
   * @brief Returns `true` if and only if the filtration value is considered as infinity.
   */
  bool is_plus_inf() const {
    for (const auto &v : multi_filtration_.get_underlying_container()) {
      if (v != Generator::T_inf) return false;
    }
    return true;
  }

  /**
   * @brief Returns `true` if and only if the filtration value is considered as minus infinity.
   */
  bool is_minus_inf() const {
    for (const auto &v : multi_filtration_.get_underlying_container()) {
      if (v != -Generator::T_inf) return false;
    }
    return true;
  }

  /**
   * @brief Returns `true` if and only if the filtration value is considered as NaN.
   */
  bool is_nan() const {
    if constexpr (std::numeric_limits<value_type>::has_quiet_NaN) {
      for (const auto &v : multi_filtration_.get_underlying_container()) {
        if (!std::isnan(v)) return false;
      }
      return true;
    } else {
      return false;
    }
  }

  /**
   * @brief Returns `true` if and only if the filtration value is non-empty and is not considered as infinity,
   * minus infinity or NaN.
   */
  bool is_finite() const {
    bool isInf = true, isMinusInf = true, isNan = true;
    for (const auto &v : multi_filtration_.get_underlying_container()) {
      if (v != Generator::T_inf) isInf = false;
      if (v != -Generator::T_inf) isMinusInf = false;
      if (!_is_nan(v)) isNan = false;
      if (!isInf && !isMinusInf && !isNan) return true;
    }
    return false;
  }

  // COMPARAISON OPERATORS

  // TODO : this costs a lot... optimize / cheat in some way for python ?

  /**
   * @brief Returns `true` if and only if the positive cones generated by @p b are strictly contained in the
   * positive cones generated by @p a.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a > b \f$ and \f$ b > a \f$ returning both false
   * does **not** imply \f$ a == b \f$.
   */
  friend bool operator<(const Multi_critical_filtration_with_n_parameters &a,
                        const Multi_critical_filtration_with_n_parameters &b) {
    for (std::size_t i = 0u; i < b.num_generators(); ++i) {
      // for each generator in b, verify if it is strictly in the cone of at least one generator of a
      bool isContained = false;
      for (std::size_t j = 0u; j < a.num_generators() && !isContained; ++j) {
        // lexicographical order, so if a[j][0] dom b[j][0], than a[j'] can never strictly contain b[i] for all j' > j.
        if (_first_dominates(a.multi_filtration_[j], b.multi_filtration_[i])) return false;
        isContained = _strictly_contains(a.multi_filtration_[j], b.multi_filtration_[i]);
      }
      if (!isContained) return false;
    }
    return true;
  }

  /**
   * @brief Returns `true` if and only if the positive cones generated by @p a are strictly contained in the
   * positive cones generated by @p b.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a > b \f$ and \f$ b > a \f$ returning both false
   * does **not** imply \f$ a == b \f$.
   */
  friend bool operator>(const Multi_critical_filtration_with_n_parameters &a,
                        const Multi_critical_filtration_with_n_parameters &b) {
    return b < a;
  }

  /**
   * @brief Returns `true` if and only if the positive cones generated by @p b are contained in or are (partially)
   * equal to the positive cones generated by @p a.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a \le b \f$ and \f$ b \le a \f$ can both return
   * `false`.
   */
  friend bool operator<=(const Multi_critical_filtration_with_n_parameters &a,
                         const Multi_critical_filtration_with_n_parameters &b) {
    // check if this curves is below other's curve
    //  ie for each guy in this, check if there is a guy in other that dominates him
    for (std::size_t i = 0u; i < b.num_generators(); ++i) {
      // for each generator in b, verify if it is in the cone of at least one generator of a
      bool isContained = false;
      for (std::size_t j = 0u; j < a.num_generators() && !isContained; ++j) {
        // lexicographical order, so if a[j][0] strictly dom b[j][0], than a[j'] can never contain b[i] for all j' > j.
        if (_first_strictly_dominates(a.multi_filtration_[j], b.multi_filtration_[i])) return false;
        isContained = _contains(a.multi_filtration_[j], b.multi_filtration_[i]);
      }
      if (!isContained) return false;
    }
    return true;
  }

  /**
   * @brief Returns `true` if and only if the positive cones generated by @p a are contained in or are (partially)
   * equal to the positive cones generated by @p b.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a \ge b \f$ and \f$ b \ge a \f$ can both return
   * `false`.
   */
  friend bool operator>=(const Multi_critical_filtration_with_n_parameters &a,
                         const Multi_critical_filtration_with_n_parameters &b) {
    return b <= a;
  }

  /**
   * @brief Returns `true` if and only if for each \f$ i \f$, \f$ a[i] \f$ is equal to \f$ b[i] \f$.
   */
  friend bool operator==(const Multi_critical_filtration_with_n_parameters &a,
                         const Multi_critical_filtration_with_n_parameters &b) {
    // assumes lexicographical order for both
    return a.multi_filtration_ == b.multi_filtration_;
  }

  /**
   * @brief Returns `true` if and only if \f$ a == b \f$ returns `false`.
   */
  friend bool operator!=(const Multi_critical_filtration_with_n_parameters &a,
                         const Multi_critical_filtration_with_n_parameters &b) {
    return !(a == b);
  }

  // MODIFIERS

  /**
   * @brief Sets the number of generators. If there were less generators before, new empty generators are constructed.
   * If there were more generators before, the exceed of generators is destroyed (any generator with index higher or
   * equal than @p n to be more precise). If @p n is zero, the methods does nothing. A filtration value should never
   * be empty.
   *
   * @warning All empty generators have 0 parameters. This can be problematic for some methods if there are also
   * non empty generators in the container. Make sure to fill them with real generators or to remove them before
   * using those methods.
   *
   * @warning Be sure to call @ref simplify if necessary after setting all the generators. Most methods will have an
   * undefined behaviour if the set of generators is not minimal or sorted.
   *
   * @param n New number of generators.
   */
  void set_num_generators(std::size_t g) {
    if (g == 0) return;
    multi_filtration_.resize(g);
  }

  /**
   * @brief Sets all generators to the least common upper bound between the current generator value and the given value.
   *
   * More formally, it pushes the current filtration value to the cone \f$ \{ y \in \mathbb R^n : y \ge x \} \f$
   * originating in \f$ x \f$. The resulting values corresponds to the generators of the intersection of this cone
   * with the union of positive cones generated by the old generators.
   *
   * @param x The target filtration value towards which to push.
   */
  template <class GeneratorRange = std::initializer_list<T> >
  void push_to_least_common_upper_bound(const GeneratorRange &x) {
    GUDHI_CHECK(x.size() == N, "Wrong range size. Should correspond to the number of parameters.");

    bool xIsInf = true, xIsMinusInf = true, xIsNaN = true;
    bool thisIsInf = true, thisIsMinusInf = true, thisIsNaN = true;

    // if one is not finite, we can avoid the heavy simplification process
    _get_infinity_statuses(multi_filtration_[0], x, thisIsInf, thisIsMinusInf, thisIsNaN, xIsInf, xIsMinusInf, xIsNaN);

    if (thisIsInf || thisIsNaN || xIsNaN || xIsMinusInf) return;

    if (xIsInf || thisIsMinusInf) {
      multi_filtration_ = {x};
      return;
    }

    for (Generator g : multi_filtration_) {
      g.push_to_least_common_upper_bound(x);
    }

    simplify();
  }

  /**
   * @brief Sets all generators to the greatest common lower bound between the current generator value and the given
   * value.
   *
   * More formally, it pulls the current filtration value to the cone \f$ \{ y \in \mathbb R^n : y \le x \} \f$
   * originating in \f$ x \f$. The resulting values corresponds to the generators of the intersection of this cone
   * with the union of negative cones generated by the old generators.
   *
   * @param x The target filtration value towards which to pull.
   */
  template <class GeneratorRange = std::initializer_list<T> >
  void pull_to_greatest_common_lower_bound(const GeneratorRange &x) {
    GUDHI_CHECK(x.size() == N, "Wrong range size. Should correspond to the number of parameters.");

    bool xIsInf = true, xIsMinusInf = true, xIsNaN = true;
    bool thisIsInf = true, thisIsMinusInf = true, thisIsNaN = true;

    // if one is not finite, we can avoid the heavy simplification process
    _get_infinity_statuses(multi_filtration_[0], x, thisIsInf, thisIsMinusInf, thisIsNaN, xIsInf, xIsMinusInf, xIsNaN);

    if (xIsInf || thisIsNaN || xIsNaN || thisIsMinusInf) return;

    if (thisIsInf || xIsMinusInf) {
      multi_filtration_ = {x};
      return;
    }
    for (Generator g : multi_filtration_) {
      g.pull_to_greatest_common_lower_bound(x);
    }

    simplify();
  }

  /**
   * @brief Adds the given generator to the filtration value such that the sets remains minimal.
   * It is therefore possible that the generator is ignored if it does not generated any new lifetime or that
   * old generators disappear if they are overshadowed by the new one.
   * @pre If all are finite, the new generator has to have the same number of parameters than the others.
   *
   * @param x New generator to add.
   * @return true If and only if the generator is actually added to the set of generators.
   * @return false Otherwise.
   */
  template <class GeneratorRange = std::initializer_list<T> >
  bool add_generator(const GeneratorRange &x) {
    GUDHI_CHECK(x.size() == N, "Wrong range size. Should correspond to the number of parameters.");

    const int newIndex = -1;

    std::size_t end = multi_filtration_.size();
    std::vector<int> indices(end);
    std::iota(indices.begin(), indices.end(), 0);

    if (_generator_can_be_added(x, 0, end, indices)) {
      indices.resize(end);
      indices.push_back(newIndex);
      _build_from(indices, newIndex, x);
      return true;
    }

    return false;
  }

  /**
   * @brief Adds the given generator to the filtration value without any verifications or simplifications.
   *
   * @warning If the resulting set of generators is not minimal after modification, some methods will have an
   * undefined behaviour. Be sure to call @ref simplify() before using them.
   *
   * @param x
   */
  template <class GeneratorRange = std::initializer_list<T> >
  void add_guaranteed_generator(const GeneratorRange &x) {
    multi_filtration_.push_back(x);
  }

  /**
   * @brief Projects the filtration value into the given grid. If @p coordinate is false, the entries are set to
   * the nearest upper bound value with the same parameter in the grid and the new generators are simplified and
   * ordered. Otherwise, the entries are set to the indices of those nearest upper bound values. In this case,
   * no simplification or sort are done, such that the new coordinates have a one by one correspondence with the
   * positions of the old generators.
   * The grid has to be represented as a vector of ordered ranges of values convertible into `T`. An index
   * \f$ i \f$ of the vector corresponds to the same parameter as the index \f$ i \f$ in a generator.
   * The ranges correspond to the possible values of the parameters, ordered by increasing value, forming therefore
   * all together a 2D grid.
   *
   * @tparam one_d_array A range of values convertible into `T` ordered by increasing value. Has to implement
   * a begin, end and operator[] method.
   * @param grid Vector of @p one_d_array with size at least number of filtration parameters.
   * @param coordinate If true, the values are set to the coordinates of the projection in the grid. If false,
   * the values are set to the values at the coordinates of the projection.
   */
  template <typename one_d_array>
  void project_onto_grid(const std::vector<one_d_array> &grid, bool coordinate = true) {
    GUDHI_CHECK(grid.size() >= num_parameters(),
                "The grid should not be smaller than the number of parameters in the filtration value.");

    for (Generator x : multi_filtration_) {
      x.project_onto_grid(grid, coordinate);
    }

    if (!coordinate) simplify();
  }

  /**
   * @brief Removes all empty generators from the filtration value. If @p include_infinities is true, it also
   * removes the generators at infinity or minus infinity.
   * If the set of generators is empty after removals, it is set to minus infinity if `co` is false or to infinity
   * if `co` is true.
   *
   * @param include_infinities If true, removes also infinity values.
   */
  void remove_empty_generators(bool include_infinities = false) {
    std::vector<int> indices;
    indices.reserve(num_generators());
    for (unsigned int i = 0; i < num_generators(); ++i){
      if (!include_infinities || multi_filtration_[i].is_finite()) indices.push_back(i);
    }
    _build_from(indices); //sorts

    if (multi_filtration_.empty()) multi_filtration_.resize(N, _get_default_value());
  }

  /**
   * @brief Simplifies the current set of generators such that it becomes minimal. Also orders it in increasing
   * lexicographical order. Only necessary if generators were added "by hand" without verification either trough the
   * constructor or with @ref add_guaranteed_generator "", etc.
   */
  void simplify() {
    std::size_t end = 0;
    std::vector<int> indices(multi_filtration_.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (std::size_t curr = 0; curr < multi_filtration_.size(); ++curr) {
      if (_generator_can_be_added(multi_filtration_[indices[curr]], 0, end, indices)) {
        std::swap(indices[end], indices[curr]);
        ++end;
      }
    }

    indices.resize(end);
    _build_from(indices);
  }

  // FONCTIONNALITIES

  /**
   * @brief Returns a generator with the minimal values of all parameters in any generator of the given filtration
   * value. That is, the greatest lower bound of all generators.
   */
  friend One_critical_filtration_with_n_parameters<T, N> factorize_below(
      const Multi_critical_filtration_with_n_parameters &f) {
    if (f.num_generators() == 0) return One_critical_filtration_with_n_parameters<T, N>();
    One_critical_filtration_with_n_parameters<T, N> result(Generator::T_inf);
    for (const Generator &g : f) {
      for (std::size_t i = 0; i < N; ++i) {
        result[i] = std::min(result[i], g[i]);
      }
    }
    return result;
  }

  /**
   * @brief Returns a generator with the maximal values of all parameters in any generator of the given filtration
   * value. That is, the least upper bound of all generators.
   */
  friend One_critical_filtration_with_n_parameters<T, N> factorize_above(
      const Multi_critical_filtration_with_n_parameters &f) {
    if (f.num_generators() == 0) return One_critical_filtration_with_n_parameters<T, N>();
    One_critical_filtration_with_n_parameters<T, N> result(-Generator::T_inf);
    for (const Generator &g : f) {
      for (std::size_t i = 0; i < N; ++i) {
        result[i] = std::max(result[i], g[i]);
      }
    }
    return result;
  }

  /**
   * @brief Computes the smallest (resp. the greatest if `co` is true) scalar product of the all generators with the
   * given vector.
   *
   * @tparam U Arithmetic type of the result. Default value: `T`.
   * @param f Filtration value.
   * @param x Vector of coefficients.
   * @return Scalar product of @p f with @p x.
   */
  template <typename U = T>
  friend U compute_linear_projection(const Multi_critical_filtration_with_n_parameters &f, const std::vector<U> &x) {
    if constexpr (co) {
      U projection = std::numeric_limits<U>::lowest();
      for (const auto &y : f) {
        projection = std::max(projection, compute_linear_projection(y, x));
      }
      return projection;
    } else {
      U projection = std::numeric_limits<U>::max();
      for (const auto &y : f) {
        projection = std::min(projection, compute_linear_projection(y, x));
      }
      return projection;
    }
  }

  /**
   * @brief Computes the coordinates in the given grid, corresponding to the nearest upper bounds of the entries
   * in the given filtration value.
   * The grid has to be represented as a vector of vectors of ordered values convertible into `out_type`. An index
   * \f$ i \f$ of the vector corresponds to the same parameter as the index \f$ i \f$ in a generator.
   * The inner vectors correspond to the possible values of the parameters, ordered by increasing value,
   * forming therefore all together a 2D grid.
   *
   * @tparam out_type Signed arithmetic type. Default value: std::int32_t.
   * @tparam U Type which is convertible into `out_type`.
   * @param f Filtration value to project.
   * @param grid Vector of vectors to project into.
   * @return Filtration value \f$ out \f$ whose entry correspond to the indices of the projected values. That is,
   * the projection of \f$ f[i] \f$ is \f$ grid[i][out[i]] \f$ before simplification (if two generators were
   * projected to the same point, the doubles are removed in the output).
   */
  template <typename out_type = std::int32_t, typename U = T>
  friend Multi_critical_filtration_with_n_parameters<out_type, N, co> compute_coordinates_in_grid(
      Multi_critical_filtration_with_n_parameters f, const std::vector<std::vector<U> > &grid) {
    // TODO: by replicating the code of the 1-critical "project_onto_grid", this could be done with just one copy
    // instead of two. But it is not clear if it is really worth it, i.e., how much the change in type is really
    // necessary in the use cases. To see later.
    f.project_onto_grid(grid);
    if constexpr (std::is_same_v<out_type, T>) {
      return f;
    } else {
      return f.as_type<out_type>();
    }
  }

  /**
   * @brief Computes the values in the given grid corresponding to the coordinates given by the given filtration
   * value. That is, if \f$ out \f$ is the result, \f$ out[i] = grid[i][f[i]] \f$. Assumes therefore, that the
   * values stored in the filtration value corresponds to indices existing in the given grid.
   *
   * @tparam U Signed arithmetic type.
   * @param f Filtration value storing coordinates compatible with `grid`.
   * @param grid Vector of vector.
   * @return Filtration value \f$ out \f$ whose entry correspond to \f$ out[i] = grid[i][f[i]] \f$ before
   * simplification (the output is simplified).
   */
  template <typename U>
  friend Multi_critical_filtration_with_n_parameters<U, N, co> evaluate_coordinates_in_grid(
      const Multi_critical_filtration_with_n_parameters &f, const std::vector<std::vector<U> > &grid) {
    Multi_critical_filtration_with_n_parameters<U, N, co> out;
    out.set_num_generators(f.num_generators());
    for (std::size_t i = 0; i < f.num_generators(); ++i) {
      out[i] = evaluate_coordinates_in_grid(f[i], grid);
    }
    out.simplify();
    return out;
  }

  // UTILITIES

  /**
   * @brief Outstream operator.
   */
  friend std::ostream &operator<<(std::ostream &stream, const Multi_critical_filtration_with_n_parameters &f) {
    stream << "(k = " << f.multi_filtration_.size() << ")[";
    for (const auto &val : f) {
      stream << val << "; ";
    }
    if (f.multi_filtration_.size() > 0) {
      stream << "\b"
             << "\b";
    }
    stream << "]";
    return stream;
  }

 public:
  /**
   * @brief Indicates if the class manages multi-critical filtration values.
   */
  constexpr static const bool is_multi_critical = true;

 private:
  Generators multi_filtration_; /**< Container for generators. */

  static bool _is_nan(T val) {
    if constexpr (std::is_integral_v<T>) {
      // to avoid Windows issue which don't know how to cast integers for cmath methods
      return false;
    } else {
      return std::isnan(val);
    }
  }

  constexpr static T _get_default_value() { return co ? Generator::T_inf : -Generator::T_inf; }

  /**
   * @brief Verifies if @p b is strictly contained in the positive cone originating in `a`.
   */
  static bool _strictly_contains(const Generator &a, const Generator &b) {
    if constexpr (co)
      return a > b;
    else {
      return a < b;
    }
  }
  /**
   * @brief Verifies if @p b is contained in the positive cone originating in `a`.
   */
  static bool _contains(const Generator &a, const Generator &b) {
    if constexpr (co)
      return a >= b;
    else {
      return a <= b;
    }
  }

  static bool _first_strictly_dominates(const Generator &a, const Generator &b) {
    if constexpr (co) {
      return a[0] < b[0];
    } else {
      return a[0] > b[0];
    }
  }

  static bool _first_dominates(const Generator &a, const Generator &b) {
    if constexpr (co) {
      return a[0] <= b[0];
    } else {
      return a[0] >= b[0];
    }
  }

  enum class Rel { EQUAL, DOMINATES, IS_DOMINATED, NONE };

  template <class GeneratorRange = std::initializer_list<T> >
  static Rel _get_domination_relation(const Generator &a, const GeneratorRange &b) {
    bool equal = true;
    bool allGreater = true;
    bool allSmaller = true;
    bool allNaNA = true;
    bool allNaNB = true;
    auto itB = b.begin();
    for (unsigned int i = 0; i < N; ++i) {
      if (a[i] < *itB) {
        if (!allSmaller) return Rel::NONE;
        equal = false;
        allGreater = false;
      } else if (a[i] > *itB) {
        if (!allGreater) return Rel::NONE;
        equal = false;
        allSmaller = false;
      }
      if (!_is_nan(a[i])) allNaNA = false;
      if (!_is_nan(*itB)) allNaNB = false;
      ++itB;
    }
    if (allNaNA || allNaNB) return Rel::IS_DOMINATED;
    if (equal) return Rel::EQUAL;

    if constexpr (co) {
      if (allSmaller) return Rel::DOMINATES;
      return Rel::IS_DOMINATED;
    } else {
      if (allGreater) return Rel::DOMINATES;
      return Rel::IS_DOMINATED;
    }
  }

  // assumes between 'curr' and 'end' everything is simplified:
  // no nan values and if there is an inf/-inf, then 'end - curr == 1'
  // modifies multi_filtration_ only if true is returned.
  template <class GeneratorRange = std::initializer_list<T> >
  bool _generator_can_be_added(const GeneratorRange& x, std::size_t curr, std::size_t &end, std::vector<int>& indices) {
    while (curr != end) {
      Rel res = _get_domination_relation(multi_filtration_[indices[curr]], x);
      if (res == Rel::IS_DOMINATED || res == Rel::EQUAL) return false;  // x dominates or is equal
      if (res == Rel::DOMINATES) {                                      // x is dominated
        --end;
        std::swap(indices[curr], indices[end]);
      } else {  // no relation
        ++curr;
      }
    }
    return true;
  }

  template <class GeneratorRange = std::initializer_list<T> >
  void _build_from(std::vector<int> &indices, const int newIndex, const GeneratorRange &x) {
    auto comp = [&](int g1, int g2) -> bool {
      if (g1 == g2) {
        return false;
      }

      if (g1 == newIndex) {
        auto it = x.begin();
        for (T v : multi_filtration_[g2]) {
          if (*it != v) {
            if (_is_nan(v)) return true;
            return *it < v;
          }
          ++it;
        }
        return false;
      }

      if (g2 == newIndex) {
        auto it = x.begin();
        for (T v : multi_filtration_[g1]) {
          if (v != *it) {
            if (_is_nan(*it)) return true;
            return v < *it;
          }
          ++it;
        }
        return false;
      }

      for (std::size_t i = 0; i < N; ++i) {
        if (multi_filtration_[g1][i] != multi_filtration_[g2][i]) {
          if (_is_nan(multi_filtration_[g2][i])) return true;
          return multi_filtration_[g1][i] < multi_filtration_[g2][i];
        }
      }
      return false;
    };
    std::sort(indices.begin(), indices.end(), comp);

    Generators new_container;
    new_container.reserve(indices.size());
    for (int i : indices) {
      if (i == newIndex)
        new_container.push_back(x);
      else
        new_container.push_back(multi_filtration_[i]);
    }
    multi_filtration_.swap(new_container);
  }

  void _build_from(std::vector<int> &indices) {
    auto comp = [&](int g1, int g2) -> bool {
      for (std::size_t i = 0; i < N; ++i) {
        if (multi_filtration_[g1][i] != multi_filtration_[g2][i]) {
          if (_is_nan(multi_filtration_[g2][i])) return true;
          return multi_filtration_[g1][i] < multi_filtration_[g2][i];
        }
      }
      return false;
    };
    std::sort(indices.begin(), indices.end(), comp);

    Generators new_container;
    new_container.reserve(indices.size());
    for (int i : indices) {
      new_container.push_back(multi_filtration_[i]);
    }
    multi_filtration_.swap(new_container);
  }

  template <class GeneratorRange>
  static void _get_infinity_statuses(const Generator &a, const GeneratorRange &b, bool &aIsInf, bool &aIsMinusInf,
                                     bool &aIsNaN, bool &bIsInf, bool &bIsMinusInf, bool &bIsNaN) {
    auto itB = b.begin();
    for (std::size_t i = 0; i < N; ++i) {
      if (a[i] != Generator::T_inf) aIsInf = false;
      if (a[i] != -Generator::T_inf) aIsMinusInf = false;
      if (!_is_nan(a[i])) aIsNaN = false;
      if (*itB != Generator::T_inf) bIsInf = false;
      if (*itB != -Generator::T_inf) bIsMinusInf = false;
      if (!_is_nan(*itB)) bIsNaN = false;
      if (!aIsInf && !aIsMinusInf && !aIsNaN && !bIsInf && !bIsMinusInf && !bIsNaN) break;
      ++itB;
    }
  }
};

}  // namespace Gudhi::multi_filtration

namespace std {

template <typename T, unsigned int N, bool co>
class numeric_limits<Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co> > {
 public:
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = std::numeric_limits<T>::has_quiet_NaN;

  static constexpr Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co> infinity() noexcept {
    return Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co>::inf();
  };

  // non-standard
  static constexpr Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co>
  minus_infinity() noexcept {
    return Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co>::minus_inf();
  };

  static constexpr Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co> max() noexcept(
      false) {
    throw std::logic_error(
        "The maximal value cannot be represented with no finite numbers of generators."
        "Use `max(number_of_generators, number_of_parameters)` instead");
  };

  // non-standard, so I don't want to define default values.
  static constexpr Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co> max(
      unsigned int g) noexcept {
    return Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co>(
        std::vector<T>(N * g, std::numeric_limits<T>::max()));
  };

  static constexpr Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co> quiet_NaN() {
    if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
      return Gudhi::multi_filtration::Multi_critical_filtration_with_n_parameters<T, N, co>::nan();
    } else {
      throw std::logic_error("Does not have a NaN value.");
    }
  };
};

}  // namespace std

#endif  // MULTI_CRITICAL_FILTRATIONS_WITH_N_PARAMETERS_H_
