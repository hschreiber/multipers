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
 * @file One_critical_filtration_with_n_parameters.h
 * @author Hannah Schreiber
 * @brief Contains the @ref Gudhi::multi_filtration::One_critical_filtration_with_n_parameters class.
 */

#ifndef ONE_CRITICAL_FILTRATIONS_WITH_N_PARAMETERS_H_
#define ONE_CRITICAL_FILTRATIONS_WITH_N_PARAMETERS_H_

#include <algorithm>  //std::lower_bound
#include <array>
#include <cmath>    //std::isnan, std::min
#include <cstddef>  //std::size_t
#include <cstdint>  //std::int32_t
#include <cstring>  //memcpy
#include <ostream>  //std::ostream
#include <limits>   //std::numerical_limits
#include <vector>

#include <gudhi/Debug_utils.h>

namespace Gudhi::multi_filtration {

/**
 * @class One_critical_filtration_with_n_parameters one_critical_filtration.h gudhi/one_critical_filtration.h
 * @ingroup multi_filtration
 *
 * @brief Class encoding the apparition time, i.e., filtration value of an object
 * (e.g., simplex, cell, abstract algebraic generator) in the setting of 1-critical multiparameter filtrations.
 * The class can be used as a vector whose indices correspond to one parameter each.
 * It also follows numpy-like broadcast semantic.
 *
 * @details Inherits of `std::vector<T>`. Overloads `std::numeric_limits` such that:
 * - `std::numeric_limits<One_critical_filtration_with_n_parameters<T> >::has_infinity` returns `true`,
 * - `std::numeric_limits<One_critical_filtration_with_n_parameters<T> >::infinity()` returns @ref
 * One_critical_filtration_with_n_parameters<T>::inf() "",
 * - `std::numeric_limits<One_critical_filtration_with_n_parameters<T> >::minus_infinity()` returns
 *   @ref One_critical_filtration_with_n_parameters<T>::minus_inf() "",
 * - `std::numeric_limits<One_critical_filtration_with_n_parameters<T> >::max()` throws,
 * - `std::numeric_limits<One_critical_filtration_with_n_parameters<T> >::max(n)` returns a @ref
 * One_critical_filtration_with_n_parameters<T> with `n` parameters evaluated at value `std::numeric_limits<T>::max()`,
 * - `std::numeric_limits<One_critical_filtration_with_n_parameters<T> >::quiet_NaN()` returns @ref
 * One_critical_filtration_with_n_parameters<T>::nan() "".
 *
 * One critical simplicial filtrations are filtrations such that the lifetime of each object is a positive cone, e.g.
 *  - \f$ \{ x \in  \mathbb R^2 : x>=(1,2)\} \f$ is valid, while
 *  - \f$ \{ x \in  \mathbb R^2 : x>=(1,2)\} \cap \{x \in \mathbb R^2 : x>=(2,1)\} \f$ is not.
 *
 * If the lifetime corresponds to a union of such positive cones, the filtration is called a multi-critical filtration.
 * For those cases, use @ref Multi_critical_filtration instead.
 *
 * @tparam T Arithmetic type of an entry for one parameter of the filtration value. Has to be **signed** and
 * to implement `std::isnan(T)`, `std::numeric_limits<T>::has_quiet_NaN`, `std::numeric_limits<T>::quiet_NaN()`,
 * `std::numeric_limits<T>::has_infinity`, `std::numeric_limits<T>::infinity()` and `std::numeric_limits<T>::max()`.
 * If `std::numeric_limits<T>::has_infinity` returns `false`, a call to `std::numeric_limits<T>::infinity()`
 * can simply throw. Examples are the native types `double`, `float` and `int`.
 */
template <typename T, unsigned int N>
class One_critical_filtration_with_n_parameters : public std::array<T, N> {
 private:
  using Base = std::array<T, N>;

 public:
  /**
   * @brief Type of the origin of a "lifetime cone", i.e., of a one-critical filtration value.
   * Common with @ref Multi_critical_filtration "". In the 1-critical case, simply the class it-self.
   */
  using Generator = One_critical_filtration_with_n_parameters<T, N>;

  // CONSTRUCTORS

  /**
   * @brief Default constructor. Constructs a value at minus infinity.
   */
  One_critical_filtration_with_n_parameters() : Base() { Base::fill(-T_inf); }
  One_critical_filtration_with_n_parameters(T value) : Base() { Base::fill(value); }
  /**
   * @brief Construct a vector from the given vector.
   *
   * @param v Vector with values for each parameter.
   */
  One_critical_filtration_with_n_parameters(const std::array<T, N> &v) : Base(v) {}
  /**
   * @brief Construct a vector from the given vector by moving it to the new vector.
   *
   * @param v Vector with values for each parameter.
   */
  One_critical_filtration_with_n_parameters(std::array<T, N> &&v) : Base(std::move(v)) {}
  /**
   * @brief Construct a vector from the range given by the begin and end iterators.
   *
   * @param it_begin Start of the range.
   * @param it_end End of the range.
   */
  template <class Iterator>
  One_critical_filtration_with_n_parameters(Iterator it_begin, Iterator it_end) : Base() {
    for (unsigned int i = 0; i < N && it_begin != it_end; ++i) {
      Base::operator[](i) = *it_begin;
      ++it_begin;
    }
  }

  // HERITAGE

  using std::array<T, N>::operator[]; /**< Inheritance of entry access. */
  using std::array<T, N>::operator=;  /**< Inheritance of entry access. */
  using value_type = T;               /**< Entry type. */

  // CONVERTERS

  // like numpy
  /**
   * @brief Returns a copy with entries casted into the type given as template parameter.
   *
   * @tparam U New type for the entries.
   * @return Copy with new entry type.
   */
  template <typename U>
  One_critical_filtration_with_n_parameters<U, N> as_type() const {
    One_critical_filtration_with_n_parameters<U, N> out;
    for (unsigned int i = 0; i < N; ++i) out[i] = static_cast<U>(Base::operator[](i));
    return out;
  }

  // ACCESS

  /**
   * @brief Returns the number of parameters of the finite filtration value. If the value is "inf", "-inf" or "NaN",
   * returns 1.
   *
   * @return Number of parameters.
   */
  constexpr std::size_t num_parameters() const { return N; }

  /**
   * @brief Returns a filtration value for which @ref is_plus_inf() returns `true`.
   *
   * @return Infinity.
   */
  constexpr static One_critical_filtration_with_n_parameters inf() {
    One_critical_filtration_with_n_parameters res;
    res.fill(T_inf);
    return res;
  }

  /**
   * @brief Returns a filtration value for which @ref is_minus_inf() returns `true`.
   *
   * @return Minus infinity.
   */
  constexpr static One_critical_filtration_with_n_parameters minus_inf() {
    One_critical_filtration_with_n_parameters res;
    res.fill(-T_inf);
    return res;
  }

  /**
   * @brief Returns a filtration value for which @ref is_nan() returns `true`. If `T` does not support NaN values,
   * all values in the array will be 0 and will not be recognized as NaN.
   *
   * @return NaN.
   */
  constexpr static One_critical_filtration_with_n_parameters nan() {
    One_critical_filtration_with_n_parameters res;
    res.fill(std::numeric_limits<T>::quiet_NaN());
    return res;
  }

  // DESCRIPTORS

  constexpr static bool is_multicritical() { return false; }

  /**
   * @brief Returns `true` if and only if the filtration value is considered as infinity.
   */
  bool is_plus_inf() const {
    for (const auto &v : *this) {
      if (v != T_inf) return false;
    }
    return true;
  }

  /**
   * @brief Returns `true` if and only if the filtration value is considered as minus infinity.
   */
  bool is_minus_inf() const {
    for (const auto &v : *this) {
      if (v != -T_inf) return false;
    }
    return true;
  }

  /**
   * @brief Returns `true` if and only if the filtration value is considered as NaN.
   */
  bool is_nan() const {
    if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
      for (const auto &v : *this) {
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
    for (const auto &v : *this) {
      if (v != T_inf) isInf = false;
      if (v != -T_inf) isMinusInf = false;
      if (!is_nan_(v)) isNan = false;
    }
    return !isInf && !isMinusInf && !isNan;
  }

  // COMPARAISON OPERATORS

  /**
   * @brief Returns `true` if and only if for each \f$ i \f$, \f$ a[i] \f$ is strictly smaller than \f$ b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a < b \f$ and \f$ b < a \f$ returning both false
   * does **not** imply \f$ a == b \f$.
   */
  friend bool operator<(const One_critical_filtration_with_n_parameters &a,
                        const One_critical_filtration_with_n_parameters &b) {
    bool isSame = true;
    for (auto i = 0u; i < N; ++i) {
      if (a[i] > b[i] || is_nan_(a[i]) || is_nan_(b[i])) return false;
      if (isSame && a[i] != b[i]) isSame = false;
    }
    return !isSame;
  }

  /**
   * @brief Returns `true` if and only if for each \f$ i \f$, \f$ a[i] \f$ is smaller or equal than \f$ b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a \le b \f$ and \f$ b \le a \f$ can both return
   * `false`.
   */
  friend bool operator<=(const One_critical_filtration_with_n_parameters &a,
                         const One_critical_filtration_with_n_parameters &b) {
    for (std::size_t i = 0u; i < N; ++i) {
      if (a[i] > b[i] || (!is_nan_(a[i]) && is_nan_(b[i])) || (is_nan_(a[i]) && !is_nan_(b[i]))) return false;
    }
    return true;
  }

  /**
   * @brief Returns `true` if and only if for each \f$ i \f$, \f$ a[i] \f$ is strictly greater than \f$ b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a > b \f$ and \f$ b > a \f$ returning both false
   * does **not** imply \f$ a == b \f$.
   */
  friend bool operator>(const One_critical_filtration_with_n_parameters &a,
                        const One_critical_filtration_with_n_parameters &b) {
    return b < a;
  }

  /**
   * @brief Returns `true` if and only if for each \f$ i \f$, \f$ a[i] \f$ is greater or equal than \f$ b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Note that not all filtration values are comparable. That is, \f$ a \ge b \f$ and \f$ b \ge a \f$ can both return
   * `false`.
   */
  friend bool operator>=(const One_critical_filtration_with_n_parameters &a,
                         const One_critical_filtration_with_n_parameters &b) {
    return b <= a;
  }

  /**
   * @brief Returns `true` if and only if for each \f$ i \f$, \f$ a[i] \f$ is equal to \f$ b[i] \f$.
   */
  friend bool operator==(const One_critical_filtration_with_n_parameters &a,
                         const One_critical_filtration_with_n_parameters &b) {
    return static_cast<const Base &>(a) == static_cast<const Base &>(b);
  }

  /**
   * @brief Returns `true` if and only if \f$ a == b \f$ returns `false`.
   */
  friend bool operator!=(const One_critical_filtration_with_n_parameters &a,
                         const One_critical_filtration_with_n_parameters &b) {
    return !(a == b);
  }

  // ARITHMETIC OPERATORS

  // opposite
  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ -f[i] \f$.
   *
   * Used conventions:
   * - \f$ -NaN = NaN \f$.
   *
   * @param f Value to opposite.
   * @return The opposite of @p f.
   */
  friend One_critical_filtration_with_n_parameters operator-(const One_critical_filtration_with_n_parameters &f) {
    One_critical_filtration_with_n_parameters result;
    for (unsigned int i = 0; i < N; ++i) {
      result[i] = -f[i];
    }
    return result;
  }

  // subtraction
  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ a[i] - b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ inf - inf = NaN \f$,
   * - \f$ -inf - (-inf) = NaN \f$,
   * - \f$ NaN - b = NaN \f$,
   * - \f$ a - NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param a First element of the subtraction.
   * @param b Second element of the subtraction.
   * @return Entry-wise \f$ a - b \f$.
   */
  friend One_critical_filtration_with_n_parameters operator-(One_critical_filtration_with_n_parameters a,
                                                             const One_critical_filtration_with_n_parameters &b) {
    a -= b;
    return a;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ f[i] - val \f$.
   *
   * Used conventions:
   * - \f$ inf - inf = NaN \f$,
   * - \f$ -inf - (-inf) = NaN \f$,
   * - \f$ NaN - b = NaN \f$,
   * - \f$ a - NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param f First element of the subtraction.
   * @param val Second element of the subtraction.
   * @return Entry-wise \f$ f - val \f$.
   */
  friend One_critical_filtration_with_n_parameters operator-(One_critical_filtration_with_n_parameters f,
                                                             const T &val) {
    f -= val;
    return f;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ val - f[i] \f$.
   *
   * Used conventions:
   * - \f$ inf - inf = NaN \f$,
   * - \f$ -inf - (-inf) = NaN \f$,
   * - \f$ NaN - b = NaN \f$,
   * - \f$ a - NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param val First element of the subtraction.
   * @param f Second element of the subtraction.
   * @return Entry-wise \f$ val - f \f$.
   */
  friend One_critical_filtration_with_n_parameters operator-(const T &val,
                                                             One_critical_filtration_with_n_parameters f) {
    for (unsigned int i = 0; i < N; ++i) {
      f[i] = -f[i];
      add_(f[i], val);
    }
    return f;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] - to\_subtract[i] \f$.
   * If @p result and @p to_subtract are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ inf - inf = NaN \f$,
   * - \f$ -inf - (-inf) = NaN \f$,
   * - \f$ NaN - b = NaN \f$,
   * - \f$ a - NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the subtraction.
   * @param to_subtract Second element of the subtraction.
   * @return Entry-wise \f$ result - to\_subtract \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator-=(
      One_critical_filtration_with_n_parameters &result, const One_critical_filtration_with_n_parameters &to_subtract) {
    for (unsigned int i = 0; i < N; ++i) subtract_(result[i], to_subtract[i]);
    return result;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] - to\_subtract \f$.
   *
   * Used conventions:
   * - \f$ inf - inf = NaN \f$,
   * - \f$ -inf - (-inf) = NaN \f$,
   * - \f$ NaN - b = NaN \f$,
   * - \f$ a - NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the subtraction.
   * @param to_subtract Second element of the subtraction.
   * @return Entry-wise \f$ result - to\_subtract \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator-=(One_critical_filtration_with_n_parameters &result,
                                                               const T &to_subtract) {
    for (unsigned int i = 0; i < N; ++i) subtract_(result[i], to_subtract);
    return result;
  }

  // addition
  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ a[i] + b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ NaN + b = NaN \f$,
   * - \f$ a + NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param a First element of the addition.
   * @param b Second element of the addition.
   * @return Entry-wise \f$ a + b \f$.
   */
  friend One_critical_filtration_with_n_parameters operator+(One_critical_filtration_with_n_parameters a,
                                                             const One_critical_filtration_with_n_parameters &b) {
    a += b;
    return a;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ f[i] + val \f$.
   *
   * Used conventions:
   * - \f$ NaN + b = NaN \f$,
   * - \f$ a + NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param f First element of the addition.
   * @param val Second element of the addition.
   * @return Entry-wise \f$ f + val \f$.
   */
  friend One_critical_filtration_with_n_parameters operator+(One_critical_filtration_with_n_parameters f,
                                                             const T &val) {
    f += val;
    return f;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ val + f[i] \f$.
   *
   * Used conventions:
   * - \f$ NaN + b = NaN \f$,
   * - \f$ a + NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param val First element of the addition.
   * @param f Second element of the addition.
   * @return Entry-wise \f$ val + f \f$.
   */
  friend One_critical_filtration_with_n_parameters operator+(const T &val,
                                                             One_critical_filtration_with_n_parameters f) {
    f += val;
    return f;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] + to\_add[i] \f$.
   * If @p result and @p to_add are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ NaN + b = NaN \f$,
   * - \f$ a + NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the addition.
   * @param to_add Second element of the addition.
   * @return Entry-wise \f$ result + to\_add \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator+=(
      One_critical_filtration_with_n_parameters &result, const One_critical_filtration_with_n_parameters &to_add) {
    for (unsigned int i = 0; i < N; ++i) add_(result[i], to_add[i]);
    return result;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] + to\_add \f$.
   *
   * Used conventions:
   * - \f$ NaN + b = NaN \f$,
   * - \f$ a + NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the addition.
   * @param to_add Second element of the addition.
   * @return Entry-wise \f$ result + to\_add \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator+=(One_critical_filtration_with_n_parameters &result,
                                                               const T &to_add) {
    for (unsigned int i = 0; i < N; ++i) add_(result[i], to_add);
    return result;
  }

  // multiplication
  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ a[i] * b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ inf * 0 = NaN \f$,
   * - \f$ 0 * inf = NaN \f$,
   * - \f$ -inf * 0 = NaN \f$,
   * - \f$ 0 * -inf = NaN \f$,
   * - \f$ NaN * b = NaN \f$,
   * - \f$ a * NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param a First element of the multiplication.
   * @param b Second element of the multiplication.
   * @return Entry-wise \f$ a * b \f$.
   */
  friend One_critical_filtration_with_n_parameters operator*(One_critical_filtration_with_n_parameters a,
                                                             const One_critical_filtration_with_n_parameters &b) {
    a *= b;
    return a;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ f[i] * val \f$.
   *
   * Used conventions:
   * - \f$ inf * 0 = NaN \f$,
   * - \f$ 0 * inf = NaN \f$,
   * - \f$ -inf * 0 = NaN \f$,
   * - \f$ 0 * -inf = NaN \f$,
   * - \f$ NaN * b = NaN \f$,
   * - \f$ a * NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param f First element of the multiplication.
   * @param val Second element of the multiplication.
   * @return Entry-wise \f$ f * val \f$.
   */
  friend One_critical_filtration_with_n_parameters operator*(One_critical_filtration_with_n_parameters f,
                                                             const T &val) {
    f *= val;
    return f;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ val * f[i] \f$.
   *
   * Used conventions:
   * - \f$ inf * 0 = NaN \f$,
   * - \f$ 0 * inf = NaN \f$,
   * - \f$ -inf * 0 = NaN \f$,
   * - \f$ 0 * -inf = NaN \f$,
   * - \f$ NaN * b = NaN \f$,
   * - \f$ a * NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param val First element of the multiplication.
   * @param f Second element of the multiplication.
   * @return Entry-wise \f$ val * f \f$.
   */
  friend One_critical_filtration_with_n_parameters operator*(const T &val,
                                                             One_critical_filtration_with_n_parameters f) {
    f *= val;
    return f;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] * to\_mul[i] \f$.
   * If @p result and @p to_mul are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ inf * 0 = NaN \f$,
   * - \f$ 0 * inf = NaN \f$,
   * - \f$ -inf * 0 = NaN \f$,
   * - \f$ 0 * -inf = NaN \f$,
   * - \f$ NaN * b = NaN \f$,
   * - \f$ a * NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the multiplication.
   * @param to_mul Second element of the multiplication.
   * @return Entry-wise \f$ result * to\_mul \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator*=(
      One_critical_filtration_with_n_parameters &result, const One_critical_filtration_with_n_parameters &to_mul) {
    for (unsigned int i = 0; i < N; ++i) multiply_(result[i], to_mul[i]);
    return result;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] * to\_mul \f$.
   *
   * Used conventions:
   * - \f$ inf * 0 = NaN \f$,
   * - \f$ 0 * inf = NaN \f$,
   * - \f$ -inf * 0 = NaN \f$,
   * - \f$ 0 * -inf = NaN \f$,
   * - \f$ NaN * b = NaN \f$,
   * - \f$ a * NaN = NaN \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the multiplication.
   * @param to_mul Second element of the multiplication.
   * @return Entry-wise \f$ result * to\_mul \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator*=(One_critical_filtration_with_n_parameters &result,
                                                               const T &to_mul) {
    for (unsigned int i = 0; i < N; ++i) multiply_(result[i], to_mul);
    return result;
  }

  // division
  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ a[i] / b[i] \f$.
   * If @p a and @p b are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ a / 0 = NaN \f$,
   * - \f$ inf / inf = NaN \f$,
   * - \f$ -inf / inf = NaN \f$,
   * - \f$ inf / -inf = NaN \f$,
   * - \f$ -inf / -inf = NaN \f$,
   * - \f$ NaN / b = NaN \f$,
   * - \f$ a / NaN = NaN \f$,
   * - \f$ a / inf = 0 \f$,
   * - \f$ a / -inf = 0 \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param a First element of the division.
   * @param b Second element of the division.
   * @return Entry-wise \f$ a / b \f$.
   */
  friend One_critical_filtration_with_n_parameters operator/(One_critical_filtration_with_n_parameters a,
                                                             const One_critical_filtration_with_n_parameters &b) {
    a /= b;
    return a;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ f[i] / val \f$.
   *
   * Used conventions:
   * - \f$ a / 0 = NaN \f$,
   * - \f$ inf / inf = NaN \f$,
   * - \f$ -inf / inf = NaN \f$,
   * - \f$ inf / -inf = NaN \f$,
   * - \f$ -inf / -inf = NaN \f$,
   * - \f$ NaN / b = NaN \f$,
   * - \f$ a / NaN = NaN \f$,
   * - \f$ a / inf = 0 \f$,
   * - \f$ a / -inf = 0 \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param f First element of the division.
   * @param val Second element of the division.
   * @return Entry-wise \f$ f / val \f$.
   */
  friend One_critical_filtration_with_n_parameters operator/(One_critical_filtration_with_n_parameters f,
                                                             const T &val) {
    f /= val;
    return f;
  }

  /**
   * @brief Returns a filtration value such that an entry at index \f$ i \f$ is equal to \f$ val / f[i] \f$.
   *
   * Used conventions:
   * - \f$ a / 0 = NaN \f$,
   * - \f$ inf / inf = NaN \f$,
   * - \f$ -inf / inf = NaN \f$,
   * - \f$ inf / -inf = NaN \f$,
   * - \f$ -inf / -inf = NaN \f$,
   * - \f$ NaN / b = NaN \f$,
   * - \f$ a / NaN = NaN \f$,
   * - \f$ a / inf = 0 \f$,
   * - \f$ a / -inf = 0 \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param val First element of the division.
   * @param f Second element of the division.
   * @return Entry-wise \f$ val / f \f$.
   */
  friend One_critical_filtration_with_n_parameters operator/(const T &val,
                                                             const One_critical_filtration_with_n_parameters &f) {
    One_critical_filtration_with_n_parameters result;
    result.fill(val);
    for (unsigned int i = 0; i < N; ++i) divide_(result[i], f[i]);
    return result;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] / to\_div[i] \f$.
   * If @p result and @p to_div are both not infinite or NaN, they have to have the same number of parameters.
   *
   * Used conventions:
   * - \f$ a / 0 = NaN \f$,
   * - \f$ inf / inf = NaN \f$,
   * - \f$ -inf / inf = NaN \f$,
   * - \f$ inf / -inf = NaN \f$,
   * - \f$ -inf / -inf = NaN \f$,
   * - \f$ NaN / b = NaN \f$,
   * - \f$ a / NaN = NaN \f$,
   * - \f$ a / inf = 0 \f$,
   * - \f$ a / -inf = 0 \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the division.
   * @param to_div Second element of the division.
   * @return Entry-wise \f$ result / to\_div \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator/=(
      One_critical_filtration_with_n_parameters &result, const One_critical_filtration_with_n_parameters &to_div) {
    for (unsigned int i = 0; i < N; ++i) divide_(result[i], to_div[i]);
    return result;
  }

  /**
   * @brief Modifies the first parameters such that an entry at index \f$ i \f$ is equal to
   * \f$ result[i] / to\_div \f$.
   *
   * Used conventions:
   * - \f$ a / 0 = NaN \f$,
   * - \f$ inf / inf = NaN \f$,
   * - \f$ -inf / inf = NaN \f$,
   * - \f$ inf / -inf = NaN \f$,
   * - \f$ -inf / -inf = NaN \f$,
   * - \f$ NaN / b = NaN \f$,
   * - \f$ a / NaN = NaN \f$,
   * - \f$ a / inf = 0 \f$,
   * - \f$ a / -inf = 0 \f$.
   *
   * If `std::numeric_limits<T>::has_quiet_NaN` is false, then the returned filtration value will be @ref nan()
   * if any operation results in NaN, not only if all operations result in NaN.
   *
   * @param result First element of the division.
   * @param to_div Second element of the division.
   * @return Entry-wise \f$ result / to\_div \f$.
   */
  friend One_critical_filtration_with_n_parameters &operator/=(One_critical_filtration_with_n_parameters &result,
                                                               const T &to_div) {
    for (unsigned int i = 0; i < N; ++i) divide_(result[i], to_div);
    return result;
  }

  // MODIFIERS

  /**
   * @brief Sets the filtration value to the least common upper bound between the current value and the given value.
   *
   * More formally, it pushes the current filtration value to the cone \f$ \{ y \in \mathbb R^n : y \ge x \} \f$
   * originating in the given filtration value \f$ x \f$. The resulting value corresponds to the intersection of both
   * cones: \f$ \mathrm{this} = \min \{ y \in \mathbb R^n : y \ge this \} \cap \{ y \in \mathbb R^n : y \ge x \} \f$.
   *
   * @param x The target filtration value towards which to push.
   * @return True if and only if the value of this actually changed.
   */
  template <class GeneratorRange = std::initializer_list<value_type> >
  bool push_to_least_common_upper_bound(const GeneratorRange &x) {
    GUDHI_CHECK(x.size() == N, "Wrong range size. Should correspond to the number of parameters.");
    bool modified = false;
    for (unsigned int i = 0; i < N; ++i) {
      modified |= Base::operator[](i) < x[i];
      Base::operator[](i) = x[i] > Base::operator[](i) ? x[i] : Base::operator[](i);
    }
    return modified;
  }

  /**
   * @brief Sets the filtration value to the greatest common lower bound between the current value and the given value.
   *
   * More formally, it pulls the current filtration value to the cone \f$ \{ y \in \mathbb R^n : y \le x \} \f$
   * originating in the given filtration value \f$ x \f$. The resulting value corresponds to the intersection of both
   * cones: \f$ \mathrm{this} = \min \{ y \in \mathbb R^n : y \le this \} \cap \{ y \in \mathbb R^n : y \le x \} \f$.
   *
   * @param x The target filtration value towards which to pull.
   * @return True if and only if the value of this actually changed.
   */
  template <class GeneratorRange = std::initializer_list<value_type> >
  bool pull_to_greatest_common_lower_bound(const GeneratorRange &x) {
    GUDHI_CHECK(x.size() == N, "Wrong range size. Should correspond to the number of parameters.");
    bool modified = false;
    for (unsigned int i = 0; i < N; ++i) {
      modified |= Base::operator[](i) > x[i];
      Base::operator[](i) = Base::operator[](i) > x[i] ? x[i] : Base::operator[](i);
    }
    return modified;
  }

  /**
   * @brief Projects the filtration value into the given grid. If @p coordinate is false, the entries are set to
   * the nearest upper bound value with the same parameter in the grid. Otherwise, the entries are set to the indices
   * of those nearest upper bound values.
   * The grid has to be represented as a vector of ordered ranges of values convertible into `T`. An index
   * \f$ i \f$ of the vector corresponds to the same parameter as the index \f$ i \f$ in the filtration value.
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
    GUDHI_CHECK(grid.size() >= N,
                "The grid should not be smaller than the number of parameters in the filtration value.");
    for (unsigned int parameter = 0u; parameter < N; ++parameter) {
      const auto &filtration = grid[parameter];
      auto d =
          std::distance(filtration.begin(),
                        std::lower_bound(filtration.begin(), filtration.end(),
                                         static_cast<typename one_d_array::value_type>(Base::operator[](parameter))));
      Base::operator[](parameter) = coordinate ? static_cast<T>(d) : static_cast<T>(filtration[d]);
    }
  }

  // FONCTIONNALITIES

  /**
   * @brief Computes the scalar product of the given filtration value with the given vector.
   *
   * @tparam U Arithmetic type of the result. Default value: `T`.
   * @param f Filtration value.
   * @param x Vector of coefficients.
   * @return Scalar product of @p f with @p x.
   */
  template <typename U = T>
  friend U compute_linear_projection(const One_critical_filtration_with_n_parameters &f, const std::vector<U> &x) {
    U projection = 0;
    std::size_t size = std::min(x.size(), static_cast<std::size_t>(N));
    for (std::size_t i = 0; i < size; i++) projection += x[i] * static_cast<U>(f[i]);
    return projection;
  }

  /**
   * @brief Computes the norm of the given filtration value.
   *
   * @param f Filtration value.
   * @return The norm of @p f.
   */
  friend T compute_norm(const One_critical_filtration_with_n_parameters &f) {
    T out = 0;
    for (auto &val : f) out += (val * val);
    if constexpr (std::is_integral_v<T>) {
      // to avoid Windows issue which don't know how to cast integers for cmath methods
      return std::sqrt(static_cast<double>(out));
    } else {
      return std::sqrt(out);
    }
  }

  /**
   * @brief Computes the euclidean distance from the first parameter to the second parameter.
   *
   * @param f Start filtration value.
   * @param other End filtration value.
   * @return Euclidean distance between @p f and @p other.
   */
  friend T compute_euclidean_distance_to(const One_critical_filtration_with_n_parameters &f,
                                         const One_critical_filtration_with_n_parameters &other) {
    T out = 0;
    for (unsigned int i = 0; i < N; i++) {
      out += (f[i] - other[i]) * (f[i] - other[i]);
    }
    if constexpr (std::is_integral_v<T>) {
      // to avoid Windows issue which don't know how to cast integers for cmath methods
      return std::sqrt(static_cast<double>(out));
    } else {
      return std::sqrt(out);
    }
  }

  /**
   * @brief Computes the coordinates in the given grid, corresponding to the nearest upper bounds of the entries
   * in the given filtration value.
   * The grid has to be represented as a vector of vectors of ordered values convertible into `out_type`. An index
   * \f$ i \f$ of the vector corresponds to the same parameter as the index \f$ i \f$ in the filtration value.
   * The inner vectors correspond to the possible values of the parameters, ordered by increasing value,
   * forming therefore all together a 2D grid.
   *
   * @tparam out_type Signed arithmetic type. Default value: std::int32_t.
   * @tparam U Type which is convertible into `out_type`.
   * @param f Filtration value to project.
   * @param grid Vector of vectors to project into.
   * @return Filtration value \f$ out \f$ whose entry correspond to the indices of the projected values. That is,
   * the projection of \f$ f[i] \f$ is \f$ grid[i][out[i]] \f$.
   */
  template <typename out_type = std::int32_t, typename U = T>
  friend One_critical_filtration_with_n_parameters<out_type, N> compute_coordinates_in_grid(
      One_critical_filtration_with_n_parameters f, const std::vector<std::vector<U> > &grid) {
    // TODO: by replicating the code of "project_onto_grid", this could be done with just one copy
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
   * @return Filtration value \f$ out \f$ whose entry correspond to \f$ out[i] = grid[i][f[i]] \f$.
   */
  template <typename U>
  friend One_critical_filtration_with_n_parameters<U, N> evaluate_coordinates_in_grid(
      const One_critical_filtration_with_n_parameters &f, const std::vector<std::vector<U> > &grid) {
    One_critical_filtration_with_n_parameters<U, N> pushed_value;

    GUDHI_CHECK(grid.size() == N,
                "The size of the grid should correspond to the number of parameters in the filtration value.");

    U grid_inf = One_critical_filtration_with_n_parameters<U, N>::T_inf;

    for (std::size_t parameter = 0u; parameter < grid.size(); ++parameter) {
      const auto &filtration = grid[parameter];
      const auto &c = f[parameter];
      pushed_value[parameter] = c == f.T_inf ? grid_inf : filtration[c];
    }
    return pushed_value;
  }

  // UTILITIES

  /**
   * @brief Outstream operator.
   */
  friend std::ostream &operator<<(std::ostream &stream, const One_critical_filtration_with_n_parameters &f) {
    if (N == 0) {
      stream << "[]";
      return stream;
    }
    stream << "[";
    for (std::size_t i = 0; i < N - 1; ++i) {
      stream << f[i] << ", ";
    }
    stream << f.back();
    stream << "]";
    return stream;
  }

  friend bool unify_lifetimes(One_critical_filtration_with_n_parameters &f1,
                              const One_critical_filtration_with_n_parameters &f2) {
    // WARNING: costly check
    GUDHI_CHECK(f1 <= f2 || f2 <= f1, "When 1-critical, two non-comparable filtration values cannot be unified.");

    return f1.pull_to_greatest_common_lower_bound(f2);
  }

  friend bool intersect_lifetimes(One_critical_filtration_with_n_parameters &f1,
                                  const One_critical_filtration_with_n_parameters &f2) {
    return f1.push_to_least_common_upper_bound(f2);
  }

  friend char *serialize_trivial(const One_critical_filtration_with_n_parameters &value, char *start) {
    const unsigned int length = N;
    const std::size_t arg_size = sizeof(T) * length;
    const std::size_t type_size = sizeof(unsigned int);
    // to enable deserialization from One_critical_filtration_with_n_parameters with another N template parameter
    memcpy(start, &length, type_size);
    memcpy(start + type_size, value.data(), arg_size);
    return start + arg_size + type_size;
  }

  friend const char *deserialize_trivial(One_critical_filtration_with_n_parameters &value, const char *start) {
    const std::size_t type_size = sizeof(unsigned int);
    unsigned int length;
    memcpy(&length, start, type_size);
    std::size_t arg_size = sizeof(T) * length;
    // to enable deserialization from One_critical_filtration_with_n_parameters with another N template parameter
    std::size_t actual_arg_size = sizeof(T) * std::min(length, N);
    memcpy(value.data(), start + type_size, actual_arg_size);
    return start + arg_size + type_size;
  }

  friend std::size_t get_serialization_size_of(const One_critical_filtration_with_n_parameters &value) {
    return sizeof(unsigned int) + sizeof(T) * N;
  }

  /**
   * @brief Infinity value of an entry of the filtration value.
   */
  constexpr static const T T_inf =
      std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();

  /**
   * @brief Indicates if the class manages multi-critical filtration values.
   */
  constexpr static bool is_multi_critical = false;

 private:
  static bool is_nan_(T val) {
    if constexpr (std::is_integral_v<T>) {
      // to avoid Windows issue which don't know how to cast integers for cmath methods
      return false;
    } else {
      return std::isnan(val);
    }
  }

  constexpr static bool subtract_(T &v1, T v2) { return add_(v1, -v2); }

  constexpr static bool add_(T &v1, T v2) {
    if (is_nan_(v1) || is_nan_(v2) || (v1 == T_inf && v2 == -T_inf) || (v1 == -T_inf && v2 == T_inf)) {
      v1 = std::numeric_limits<T>::quiet_NaN();
      return false;
    }
    if (v1 == T_inf || v1 == -T_inf) {
      return true;
    }
    if (v2 == T_inf || v2 == -T_inf) {
      v1 = v2;
      return true;
    }

    v1 += v2;
    return true;
  }

  constexpr static bool multiply_(T &v1, T v2) {
    bool v1_is_infinite = v1 == T_inf || v1 == -T_inf;
    bool v2_is_infinite = v2 == T_inf || v2 == -T_inf;

    if (is_nan_(v1) || is_nan_(v2) || (v1_is_infinite && v2 == 0) || (v1 == 0 && v2_is_infinite)) {
      v1 = std::numeric_limits<T>::quiet_NaN();
      return false;
    }

    if ((v1 == T_inf && v2 > 0) || (v1 == -T_inf && v2 < 0) || (v1 < 0 && v2 == -T_inf) || (v1 > 0 && v2 == T_inf)) {
      v1 = T_inf;
      return true;
    }

    if ((v1 == T_inf && v2 < 0) || (v1 == -T_inf && v2 > 0) || (v1 > 0 && v2 == -T_inf) || (v1 < 0 && v2 == T_inf)) {
      v1 = -T_inf;
      return true;
    }

    v1 *= v2;
    return true;
  }

  constexpr static bool divide_(T &v1, T v2) {
    bool v1_is_infinite = v1 == T_inf || v1 == -T_inf;
    bool v2_is_infinite = v2 == T_inf || v2 == -T_inf;

    if (is_nan_(v1) || is_nan_(v2) || v2 == 0 || (v1_is_infinite && v2_is_infinite)) {
      v1 = std::numeric_limits<T>::quiet_NaN();
      return false;
    }

    if (v1 == 0 || (v1_is_infinite && v2 > 0)) return true;

    if (v1_is_infinite && v2 < 0) {
      v1 = -v1;
      return true;
    }

    if (v2_is_infinite) {
      v1 = 0;
      return true;
    }

    v1 /= v2;
    return true;
  }
};

}  // namespace Gudhi::multi_filtration

namespace std {

template <typename T, unsigned int N>
class numeric_limits<Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N> > {
 public:
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = std::numeric_limits<T>::has_quiet_NaN;

  static constexpr Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N> infinity() noexcept {
    return Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N>::inf();
  };

  // non-standard
  static constexpr Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N> minus_infinity() noexcept {
    return Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N>::minus_inf();
  };

  static constexpr Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N> max() noexcept {
    Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N> res;
    res.fill(std::numeric_limits<T>::max());
    return res;
  };

  static constexpr Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N> quiet_NaN() noexcept(
      false) {
    if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
      return Gudhi::multi_filtration::One_critical_filtration_with_n_parameters<T, N>::nan();
    } else {
      throw std::logic_error("Does not have a NaN value.");
    }
  };
};

}  // namespace std

#endif  // ONE_CRITICAL_FILTRATIONS_WITH_N_PARAMETERS_H_
