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
 * @file flat_2D_matrix.h
 * @author Hannah Schreiber
 * @brief Contains the @ref Gudhi::Flat_2D_matrix "", the @ref Gudhi::Dynamic_flat_2D_matrix "",
 * the @ref Gudhi::Row_view class and the @ref Gudhi::Row_views_iterator class.
 */

#ifndef GUDHI_FLAT_2D_MATRIX_H_
#define GUDHI_FLAT_2D_MATRIX_H_

#include <array>
#include <boost/iterator/iterator_categories.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>  //std::size_t
#include <initializer_list>
#include <iterator>
#include <stdexcept>  //std::out_of_range
#include <algorithm>  //std::lexicographical_compare
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/utility/enable_if.hpp>

#include <gudhi/Debug_utils.h>

namespace Gudhi {

template <std::size_t C, class Container>
class Row_view {
 public:
  using Underlying_container = Container;  // outside access
  using value_type = typename Container::value_type;
  using size_type = typename Container::size_type;
  using difference_type = typename Container::difference_type;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = typename Container::pointer;
  using const_pointer = typename Container::const_pointer;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;
  using reverse_iterator = typename Container::reverse_iterator;
  using const_reverse_iterator = typename Container::const_reverse_iterator;

  Row_view(Container& matrix, size_type rowIndex) : matrix_(&matrix), startIndex_(rowIndex * C) {}

  Row_view(const Row_view& other) : matrix_(other.matrix_), startIndex_(other.startIndex_) {}
  Row_view(Row_view&& other)
      : matrix_(std::exchange(other.matrix_, nullptr)), startIndex_(other.startIndex_) {}

  Row_view& operator=(const Row_view& other) {
    for (size_type i = 0; i < C; ++i) {
      this->operator[](i) = other[i];
    }
    return *this;
  }

  // Row_view& operator=(Row_view&& other) {
  //   for (size_type i = 0; i < C; ++i) {
  //     this->operator[](i) = std::move(other[i]);
  //   }
  //   return *this;
  // }

  template <class Initializing_range = std::initializer_list<value_type>,
            class = std::enable_if_t<!std::is_same_v<Initializing_range, Row_view> > >
  Row_view& operator=(const Initializing_range& other) {
    auto it = other.begin();
    for (size_type i = 0; i < size() && it != other.end(); ++i) {
      this->operator[](i) = *it;
      ++it;
    }
    return *this;
  }

  constexpr size_type size() const noexcept { return C; }

  constexpr size_type max_size() const noexcept { return C; }

  constexpr bool empty() const noexcept { return size() == 0; }

  reference at(size_type pos) {
    if (pos >= size()) throw std::out_of_range("Index is out of bound.");
    return (*matrix_)[index_(pos)];
  }
  const_reference at(size_type pos) const {
    if (pos >= size()) throw std::out_of_range("Index is out of bound.");
    return (*matrix_)[index_(pos)];
  }

  reference operator[](size_type pos) { return (*matrix_)[index_(pos)]; }
  const_reference operator[](size_type pos) const { return (*matrix_)[index_(pos)]; }

  reference front() { return (*matrix_)[index_(0)]; }
  const_reference front() const { return (*matrix_)[index_(0)]; }

  reference back() { return (*matrix_)[index_(size() - 1)]; }
  const_reference back() const { return (*matrix_)[index_(size() - 1)]; }

  value_type* data() noexcept { return matrix_->data() + startIndex_; }
  const value_type* data() const noexcept { return matrix_->data() + startIndex_; }

  iterator begin() noexcept { return matrix_->begin() + startIndex_; }
  const_iterator begin() const noexcept { return matrix_->begin() + startIndex_; }
  const_iterator cbegin() const noexcept { return matrix_->begin() + startIndex_; }

  iterator end() noexcept { return begin() + size(); }
  const_iterator end() const noexcept { return begin() + size(); }
  const_iterator cend() const noexcept { return begin() + size(); }

  reverse_iterator rbegin() noexcept { return rend() - size(); }
  const_reverse_iterator rbegin() const noexcept { return rend() - size(); }
  const_reverse_iterator crbegin() const noexcept { return rend() - size(); }

  reverse_iterator rend() noexcept { return matrix_->rend() - startIndex_; }
  const_reverse_iterator rend() const noexcept { return matrix_->rend() - startIndex_; }
  const_reverse_iterator crend() const noexcept { return matrix_->rend() - startIndex_; }

  void fill(const value_type& value) {
    for (auto& v : *this) v = value;
  }

  void swap(Row_view& other) {
    if (matrix_ == other.matrix_ && startIndex_ == other.startIndex_) return;

    for (size_type i = 0; i < C; ++i) {
      std::swap((*this)[i], other[i]);
    }
  }

  template <std::size_t C2, class Other_container>
  friend bool operator==(const Row_view& lhs, const Row_view<C2, Other_container>& rhs) {
    if constexpr (C != C2) {
      return false;
    } else {
      for (size_type i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
      }
      return true;
    }
  }
  template <std::size_t C2, class Other_container>
  friend bool operator!=(const Row_view& lhs, const Row_view<C2, Other_container>& rhs) {
    return !(lhs == rhs);
  }
  template <std::size_t C2, class Other_container>
  friend bool operator<(const Row_view& lhs, const Row_view<C2, Other_container>& rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }
  template <std::size_t C2, class Other_container>
  friend bool operator<=(const Row_view& lhs, const Row_view<C2, Other_container>& rhs) {
    for (size_type i = 0; i < std::min(lhs.size(), rhs.size()); ++i) {
      if (lhs[i] < rhs[i]) return true;
      if (lhs[i] > rhs[i]) return false;
    }
    return lhs.size() <= rhs.size();
  }
  template <std::size_t C2, class Other_container>
  friend bool operator>(const Row_view& lhs, const Row_view<C2, Other_container>& rhs) {
    return rhs < lhs;
  }
  template <std::size_t C2, class Other_container>
  friend bool operator>=(const Row_view& lhs, const Row_view<C2, Other_container>& rhs) {
    return rhs <= lhs;
  }

  template <std::size_t I>
  friend constexpr value_type& get(Row_view& a) noexcept {
    static_assert(I < C,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and the row size (excluded).");
    return a[I];
  }
  template <std::size_t I>
  friend constexpr value_type&& get(Row_view&& a) noexcept {
    static_assert(I < C,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and the row size (excluded).");
    return std::move(a[I]);
  }
  template <std::size_t I>
  friend constexpr const value_type& get(const Row_view& a) noexcept {
    static_assert(I < C,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and the row size (excluded).");
    return a[I];
  }
  template <std::size_t I>
  friend constexpr const value_type&& get(const Row_view&& a) noexcept {
    static_assert(I < C,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and the row size (excluded).");
    return std::move(a[I]);
  }

  friend void swap(Row_view lhs, Row_view rhs) {
    lhs.swap(rhs);
  }

 private:
  Container* matrix_;
  const size_type startIndex_;

  size_type index_(size_type pos) const { return startIndex_ + pos; }
};

template <class Row_view>
class Row_views_iterator : public boost::iterator_facade<Row_views_iterator<Row_view>, Row_view,
                                                         boost::random_access_traversal_tag, Row_view> {
 public:
  using Container = typename Row_view::Underlying_container;
  using size_type = typename Container::size_type;
  using difference_type = typename Container::difference_type;

  Row_views_iterator(Container* matrix, size_type pos, size_type numberOfRows)
      : matrix_(numberOfRows == 0 ? nullptr : matrix), currPos_(pos), endPos_(numberOfRows) {
    GUDHI_CHECK(matrix_ == nullptr || matrix->size() % numberOfRows == 0, "Matrix should contain R rows.");
  }

  Row_views_iterator() : matrix_(nullptr), currPos_(0), endPos_(0) {}

  template <class Other_row_view>
  Row_views_iterator(Row_views_iterator<Other_row_view> const& other,
                     typename boost::enable_if<boost::is_convertible<Other_row_view*, Row_view*>, int>::type = 0)
      : matrix_(other.matrix_), currPos_(other.currPos_), endPos_(other.endPos_) {}

  template <class Other_row_view,
            class = typename boost::enable_if<boost::is_convertible<Other_row_view*, Row_view*> >::type>
  Row_views_iterator& operator=(Row_views_iterator<Other_row_view> other) {
    std::swap(matrix_, other.matrix_);
    std::swap(currPos_, other.currPos_);
    std::swap(endPos_, other.endPos_);
    return *this;
  };

  size_type get_current_row_index() const { return currPos_; }

 private:
  // mandatory for the boost::iterator_facade inheritance.
  friend class boost::iterator_core_access;
  template <class>
  friend class Row_views_iterator;

  Container* matrix_;
  size_type currPos_;
  size_type endPos_;

  template <class Other_row_view>
  bool equal(Row_views_iterator<Other_row_view> const& other) const {
    if (_is_end() && other._is_end()) {
      if (matrix_ == nullptr || other.matrix_ == nullptr) return true;
      return matrix_ == other.matrix_;
    }
    return matrix_ == other.matrix_ && currPos_ == other.currPos_ && endPos_ == other.endPos_;
  }

  Row_view dereference() const { 
    return Row_view(*matrix_, currPos_); }

  void increment() { ++currPos_; }

  void decrement() { --currPos_; }

  void advance(int n) { currPos_ += n; }

  template <class Other_row_view>
  difference_type distance_to(const Row_views_iterator<Other_row_view>& other) const {
    return other.currPos_ - currPos_;
  }

  bool _is_end() const { return currPos_ == endPos_; }
};

template <std::size_t R, std::size_t C, typename T, class Container = std::array<T, R * C>,
          class Row = Row_view<C, Container> >
class Flat_2D_matrix {
 public:
  using value_type = Row;
  using size_type = typename Container::size_type;
  using difference_type = typename Container::difference_type;
  using reference = value_type;
  using const_reference = const value_type;
  // using pointer = typename Container::pointer;
  // using const_pointer = typename Container::const_pointer;
  using iterator = Row_views_iterator<Row>;
  using const_iterator = Row_views_iterator<Row const>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  Flat_2D_matrix() {}

  Flat_2D_matrix(T value) { container_.fill(value); }

  template <class ColumnRange>
  Flat_2D_matrix(const ColumnRange& col) {
    assert(col.size() == C);
    for (size_type r = 0; r < R; ++r) {
      auto it = col.begin();
      for (size_type c = 0; c < C; ++c) {
        container_[_index(r, c)] = *it;
        ++it;
      }
    }
  }

  Flat_2D_matrix(const Container& values) : container_(values) {}

  Flat_2D_matrix(Container&& values) : container_(std::move(values)) {}

  template <typename Iterator>
  Flat_2D_matrix(Iterator start, Iterator end) {
    for (size_type i = 0; i < R * C && start != end; ++i) {
      container_[i] = *start;
      ++start;
    }
  }

  Flat_2D_matrix(const Flat_2D_matrix& other) : container_(other.container_) {}

  Flat_2D_matrix(Flat_2D_matrix&& other) : container_(std::move(other.container_)) {}

  Flat_2D_matrix& operator=(const Flat_2D_matrix& other) {
    container_ = other.container_;
    return *this;
  }

  template <class Initializing_range = std::initializer_list<value_type> >
  Flat_2D_matrix& operator=(const Initializing_range& other) {
    container_ = other;
    return *this;
  }

  T& operator()(size_type r, size_type c) { return container_[_index(r, c)]; }

  T operator()(size_type r, size_type c) const { return container_[_index(r, c)]; }

  reference at(size_type r) {
    if (r >= R) throw std::out_of_range("Index is out of bound.");
    return Row(container_, r);
  }
  const_reference at(size_type r) const {
    if (r >= R) throw std::out_of_range("Index is out of bound.");
    // a bit unsafe, but Row should never modify container_ when const
    return Row(const_cast<Container&>(container_), r);
  }

  reference operator[](size_type r) { return Row(container_, r); }
  // a bit unsafe, but Row should never modify container_ when const
  const_reference operator[](size_type r) const { return Row(const_cast<Container&>(container_), r); }

  reference front() { return Row(container_, 0); }
  const_reference front() const { return Row(container_, 0); }

  reference back() { return Row(container_, R - 1); }
  const_reference back() const { return Row(container_, R - 1); }

  T* data() noexcept { return container_.data(); }
  const T* data() const noexcept { return container_.data(); }

  iterator begin() noexcept { return iterator(&container_, 0, R); }
  const_iterator begin() const noexcept { return const_iterator(&container_, 0, R); }
  const_iterator cbegin() const noexcept { return const_iterator(&container_, 0, R); }

  iterator end() noexcept { return iterator(&container_, R, R); }
  const_iterator end() const noexcept { return const_iterator(&container_, R, R); }
  const_iterator cend() const noexcept { return const_iterator(&container_, R, R); }

  reverse_iterator rbegin() noexcept { return reverse_iterator(&container_, 0, R); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(&container_, 0, R); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(&container_, 0, R); }

  reverse_iterator rend() noexcept { return reverse_iterator(&container_, R, R); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(&container_, R, R); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(&container_, R, R); }

  constexpr size_type size() const noexcept { return R; }

  constexpr size_type max_size() const noexcept { return R; }

  constexpr bool empty() const noexcept { return R == 0; }

  void fill(const value_type& value) { container_.fill(value); }

  void swap(Flat_2D_matrix& other) { container_.swap(other.container_); }

  template <std::size_t R2, std::size_t C2, class Other_row>
  friend bool operator==(const Flat_2D_matrix& lhs, const Flat_2D_matrix<R2, C2, T, Container, Other_row>& rhs) {
    if constexpr (R != R2 || C != C2)
      return false;
    else
      return lhs.container_ == rhs.container_;
  }
  template <std::size_t R2, std::size_t C2, class Other_row>
  friend bool operator!=(const Flat_2D_matrix& lhs, const Flat_2D_matrix<R2, C2, T, Container, Other_row>& rhs) {
    return !(lhs == rhs);
  }

  template <std::size_t I>
  friend constexpr value_type& get(Flat_2D_matrix& a) noexcept {
    static_assert(I < R,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and number of rows (excluded).");
    return a[I];
  }
  template <std::size_t I>
  friend constexpr value_type&& get(Flat_2D_matrix&& a) noexcept {
    static_assert(I < R,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and number of rows (excluded).");
    return std::move(a[I]);
  }
  template <std::size_t I>
  friend constexpr const value_type& get(const Flat_2D_matrix& a) noexcept {
    static_assert(I < R,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and number of rows (excluded).");
    return a[I];
  }
  template <std::size_t I>
  friend constexpr const value_type&& get(const Flat_2D_matrix&& a) noexcept {
    static_assert(I < R,
                  "Value mismatch at argument 1 in template parameter list. Possible value are between 0 (included) "
                  "and number of rows (excluded).");
    return std::move(a[I]);
  }

  friend void swap(Flat_2D_matrix& lhs, Flat_2D_matrix& rhs) { lhs.swap(rhs); }

  constexpr static size_type get_number_of_rows() { return R; }
  constexpr static size_type get_number_of_columns() { return C; }
  constexpr static size_type get_number_of_entries() { return R * C; }

  Container& get_underlying_container() { return container_; }
  const Container& get_underlying_container() const { return container_; }

 private:
  Container container_;

  constexpr static size_type _index(unsigned int r, unsigned int c) {
    assert(r < R && c < C && "Out of bounds.");
    return r * C + c;
  }
};

template <std::size_t C, typename T, class Container = std::vector<T>, class Row = Row_view<C, Container> >
class Dynamic_flat_2D_matrix {
 public:
  using value_type = Row;
  using size_type = typename Container::size_type;
  using allocator_type = typename Container::allocator_type;
  using difference_type = typename Container::difference_type;
  using reference = value_type;
  using const_reference = const value_type;
  // using pointer = typename Container::pointer;
  // using const_pointer = typename Container::const_pointer;
  using iterator = Row_views_iterator<Row>;
  using const_iterator = Row_views_iterator<Row const>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  Dynamic_flat_2D_matrix() noexcept(noexcept(allocator_type())) : container_(), R_(0) {}

  explicit Dynamic_flat_2D_matrix(const allocator_type& alloc) noexcept : container_(alloc), R_(0) {};

  explicit Dynamic_flat_2D_matrix(size_type r, const allocator_type& alloc = allocator_type())
      : container_(r * C, alloc), R_(r) {}

  Dynamic_flat_2D_matrix(size_type r, const T& value, const allocator_type& alloc = allocator_type())
      : container_(r * C, value, alloc), R_(r) {}

  template <class RowRange = std::initializer_list<T>,
            class = std::enable_if_t<!std::is_arithmetic_v<RowRange> && !std::is_same_v<RowRange, T> > >
  Dynamic_flat_2D_matrix(size_type r, const RowRange& row, const allocator_type& alloc = allocator_type())
      : container_(r * C, alloc), R_(r) {
    GUDHI_CHECK(row.size() == C, "Initial row does not have a compatible length.");
    for (size_type i = 0; i < r; ++i) {
      auto it = row.begin();
      for (size_type c = 0; c < C; ++c) {
        container_[_index(i, c)] = *it;
        ++it;
      }
    }
  }

  Dynamic_flat_2D_matrix(const Container& values) : container_(values), R_(values.size() / C) {
    GUDHI_CHECK(values.size() % C == 0, "Initial container does not have a compatible length.");
  }

  Dynamic_flat_2D_matrix(Container&& values) : container_(std::move(values)), R_(container_.size() / C) {
    GUDHI_CHECK(values.size() % C == 0, "Initial container does not have a compatible length.");
  }

  template <typename Iterator>
  Dynamic_flat_2D_matrix(Iterator start, Iterator end, const allocator_type& alloc = allocator_type())
      : container_(alloc) {
    container_.reserve(std::distance(start, end));
    while (start != end) {
      container_.push_back(*start);
      ++start;
    }
    GUDHI_CHECK(container_.size() % C == 0, "Initial range does not have a compatible length.");
    R_ = container_.size() / C;
  }

  Dynamic_flat_2D_matrix(const Dynamic_flat_2D_matrix& other) : container_(other.container_), R_(other.R_) {}
  Dynamic_flat_2D_matrix(Dynamic_flat_2D_matrix&& other) noexcept
      : container_(std::move(other.container_)), R_(std::exchange(other.R_, 0)) {}
  Dynamic_flat_2D_matrix(const Dynamic_flat_2D_matrix& other, const allocator_type& alloc)
      : container_(other.container_, alloc), R_(other.R_) {}
  Dynamic_flat_2D_matrix(Dynamic_flat_2D_matrix&& other, const allocator_type& alloc)
      : container_(std::move(other.container_), alloc), R_(std::exchange(other.R_, 0)) {}
  Dynamic_flat_2D_matrix(std::initializer_list<T> init, const allocator_type& alloc = allocator_type())
      : container_(init, alloc), R_(container_.size() / C) {
    GUDHI_CHECK(container_.size() % C == 0, "Initializer list does not have a compatible length.");
  }
  template <class RowRanges = std::initializer_list<std::initializer_list<T> > >
  Dynamic_flat_2D_matrix(const RowRanges& init, const allocator_type& alloc = allocator_type())
      : container_(init.size() * C, alloc), R_(init.size()) {
    size_type i = 0;
    for (const auto& row : init) {
      GUDHI_CHECK(row.size() == C, "Initializer lists do not have a compatible length.");
      for (const T& v : row) {
        container_[i] = v;
        ++i;
      }
    }
  }

  Dynamic_flat_2D_matrix& operator=(const Dynamic_flat_2D_matrix& other) {
    container_ = other.container_;
    R_ = other.R_;
    return *this;
  }
  Dynamic_flat_2D_matrix& operator=(Dynamic_flat_2D_matrix&& other) {
    container_ = std::move(other.container_);
    R_ = std::exchange(other.R_, 0);
    return *this;
  }
  Dynamic_flat_2D_matrix& operator=(std::initializer_list<T> ilist) {
    GUDHI_CHECK(ilist.size() % C == 0, "Initializer list does not have a compatible length.");
    container_ = ilist;
    R_ = container_.size() / C;
    return *this;
  }
  template <class RowRange = std::initializer_list<T> >
  Dynamic_flat_2D_matrix& operator=(std::initializer_list<RowRange> ilist) {
    container_.resize(ilist.size() * C);
    size_type i = 0;
    for (const RowRange& row : ilist) {
      GUDHI_CHECK(row.size() == C, "Initializer lists do not have a compatible length.");
      for (const T& v : row) {
        container_[i] = v;
        ++i;
      }
    }
    R_ = ilist.size();
    return *this;
  }

  void assign(size_type r, const T& value) {
    container_ = Container(r * C, value);
    R_ = r;
  }
  template <class RowRange = std::initializer_list<T> >
  void assign(size_type r, const RowRange& row) {
    GUDHI_CHECK(row.size() == C, "Initial row does not have a compatible length.");
    container_.resize(r * C);
    for (size_type i = 0; i < r; ++i) {
      auto it = row.begin();
      for (size_type c = 0; c < C; ++c) {
        container_[_index(i, c)] = *it;
        ++it;
      }
    }
    R_ = r;
  }
  template <class InputIt>
  void assign(InputIt first, InputIt last) {
    container_.clear();
    while (first != last) {
      container_.push_back(*first);
      ++first;
    }
    GUDHI_CHECK(container_.size() % C == 0, "Initial range does not have a compatible length.");
    R_ = container_.size() / C;
  }
  void assign(std::initializer_list<T> ilist) {
    GUDHI_CHECK(ilist.size() % C == 0, "Initializer list does not have a compatible length.");
    container_ = ilist;
    R_ = container_.size() / C;
  }
  template <class RowRange = std::initializer_list<T> >
  void assign(std::initializer_list<RowRange> ilist) {
    container_.resize(ilist.size() * C);
    size_type i = 0;
    for (const RowRange& row : ilist) {
      GUDHI_CHECK(row.size() == C, "Initializer lists do not have a compatible length.");
      for (const T& v : row) {
        container_[i] = v;
        ++i;
      }
    }
    R_ = ilist.size();
  }

  allocator_type get_allocator() const noexcept { return container_.get_allocator(); }

  T& operator()(size_type r, size_type c) {
    GUDHI_CHECK(r < get_number_of_rows() && c < get_number_of_columns(), "Indices are out of bound.");
    return container_[_index(r, c)];
  }

  T operator()(size_type r, size_type c) const {
    GUDHI_CHECK(r < get_number_of_rows() && c < get_number_of_columns(), "Indices are out of bound.");
    return container_[_index(r, c)];
  }

  reference at(size_type r) {
    if (r >= get_number_of_rows()) throw std::out_of_range("Index is out of bound.");
    return Row(container_, r);
  }

  const_reference at(size_type r) const {
    if (r >= get_number_of_rows()) throw std::out_of_range("Index is out of bound.");
    // a bit unsafe, but Row should never modify container_ when const
    return Row(const_cast<Container&>(container_), r);
  }

  reference operator[](size_type r) { return Row(container_, r); }

  // a bit unsafe, but Row should never modify container_ when const
  const_reference operator[](size_type r) const { return Row(const_cast<Container&>(container_), r); }

  reference front() { return (*this)[0]; }

  const_reference front() const { return (*this)[0]; }

  reference back() { return (*this)[get_number_of_rows() - 1]; }

  const_reference back() const { return (*this)[get_number_of_rows() - 1]; }

  T* data() noexcept { return container_.data(); }

  const T* data() const noexcept { return container_.data(); }

  iterator begin() noexcept { return iterator(&container_, 0, get_number_of_rows()); }

  const_iterator begin() const noexcept {
    return const_iterator(&const_cast<Container&>(container_), 0, get_number_of_rows());
  }

  const_iterator cbegin() const noexcept {
    return const_iterator(&const_cast<Container&>(container_), 0, get_number_of_rows());
  }

  iterator end() noexcept { return iterator(&container_, get_number_of_rows(), get_number_of_rows()); }

  const_iterator end() const noexcept {
    return const_iterator(&const_cast<Container&>(container_), get_number_of_rows(), get_number_of_rows());
  }

  const_iterator cend() const noexcept {
    return const_iterator(&const_cast<Container&>(container_), get_number_of_rows(), get_number_of_rows());
  }

  // TODO: not so sure about the initialization of the reverse iterators...
  reverse_iterator rbegin() noexcept { return reverse_iterator(&container_, 0, get_number_of_rows()); }

  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(&const_cast<Container&>(container_), 0, get_number_of_rows());
  }

  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(&const_cast<Container&>(container_), 0, get_number_of_rows());
  }

  reverse_iterator rend() noexcept { return reverse_iterator(&container_, get_number_of_rows(), get_number_of_rows()); }

  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(&const_cast<Container&>(container_), get_number_of_rows(), get_number_of_rows());
  }

  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(&const_cast<Container&>(container_), get_number_of_rows(), get_number_of_rows());
  }

  bool empty() const noexcept { return get_number_of_rows() == 0; }

  size_type size() const { return get_number_of_rows(); }

  size_type max_size() const { return std::floor(container_.max_size() / C); }

  void reserve(size_type new_cap) { container_.reserve(new_cap * C); }

  size_type capacity() const noexcept { return std::floor(container_.capacity() / C); }

  void shrink_to_fit() { container_.shrink_to_fit(); }

  void clear() noexcept {
    container_.clear();
    R_ = 0;
  }

  template <class RowRange = std::initializer_list<T> >
  iterator insert(const_iterator pos, const RowRange& range) {
    GUDHI_CHECK(range.size() == C, "Range size is not compatible.");
    ++R_;
    return _to_row_iterator(container_.insert(_to_container_iterator(pos), range.begin(), range.end()));
  }
  template <class RowRange = std::initializer_list<T> >
  iterator insert(const_iterator pos, size_type count, const RowRange& range) {
    GUDHI_CHECK(range.size() == C, "Range size is not compatible.");
    size_type startIndex = pos.get_current_row_index() * C;
    auto itPos = container_.begin() + startIndex;
    auto res = container_.insert(itPos, count * C, 0);
    for (size_type i = 0; i < count; ++i) {
      for (const T& v : range) {
        container_[startIndex] = v;
        ++startIndex;
      }
    }
    R_ += count;
    return _to_row_iterator(res);
  }
  template <class InputIt>
  iterator insert(const_iterator pos, InputIt first, InputIt last) {
    GUDHI_CHECK(std::distance(first, last) == C, "Range size is not compatible.");
    auto res = _to_row_iterator(container_.insert(_to_container_iterator(pos), first, last));
    R_ = container_.size() / C;
    return res;
  }
  template <class RowRange = std::initializer_list<T> >
  iterator insert(const_iterator pos, std::initializer_list<RowRange> ilist) {
    size_type startIndex = pos.get_current_row_index() * C;
    auto itPos = container_.begin() + startIndex;
    auto res = container_.insert(itPos, ilist.size() * C, 0);
    for (const RowRange& row : ilist) {
      GUDHI_CHECK(row.size() == C, "Row size is not compatible.");
      for (const T& v : row) {
        container_[startIndex] = v;
        ++startIndex;
      }
    }
    R_ += ilist.size();
    return _to_row_iterator(res);
  }

  template <class... Args>
  iterator emplace(const_iterator pos, Args&&... args) {
    std::array<T, C> element(args...);
    ++R_;
    return _to_row_iterator(container_.insert(_to_container_iterator(pos), element.begin(), element.end()));
  }

  iterator erase(const_iterator pos) {
    auto it = _to_container_iterator(pos);
    --R_;
    return _to_row_iterator(container_.erase(it, it + C));
  }

  iterator erase(const_iterator first, const_iterator last) {
    auto it = _to_container_iterator(first);
    size_type diff = last.get_current_row_index() - first.get_current_row_index();
    R_ -= diff;
    return _to_row_iterator(container_.erase(it, it + diff * C));
  }

  template <class RowRange = std::initializer_list<T> >
  void push_back(const RowRange& range) {
    GUDHI_CHECK(range.size() == C, "Range size is not compatible.");
    container_.insert(container_.end(), range.begin(), range.end());
    ++R_;
  }

  template <class... Args>
  reference emplace_back(Args&&... args) {
    std::array<T, C> element(args...);
    ++R_;
    auto res = container_.insert(container_.end(), element.begin(), element.end());
    return Row(container_, std::distance(container_.begin(), res) / C);
  }

  void pop_back() {
    auto it = container_.begin() + (get_number_of_rows() - 1) * C;
    container_.erase(it, container_.end());
    --R_;
  }

  void resize(size_type count) {
    container_.resize(count * C);
    R_ = count;
  }
  void resize(size_type count, const T& value) {
    container_.resize(count * C, value);
    R_ = count;
  }
  template <class RowRange = std::initializer_list<T> >
  void resize(size_type count, const RowRange& range) {
    GUDHI_CHECK(range.size() == C, "Range size is not compatible.");
    size_type start = container_.size();
    container_.resize(count * C);
    size_type newCounts = (container_.size() - start) / C;
    for (size_type i = 0; i < newCounts; ++i) {
      for (const T& v : range) {
        container_[start] = v;
        ++start;
      }
    }
    R_ = count;
  }

  void swap(Dynamic_flat_2D_matrix& other) {
    container_.swap(other.container_);
    std::swap(R_, other.R_);
  }

  friend void swap(Dynamic_flat_2D_matrix& lhs, Dynamic_flat_2D_matrix& rhs) { lhs.swap(rhs); }

  template <std::size_t C2, class Other_row>
  friend bool operator==(const Dynamic_flat_2D_matrix& lhs,
                         const Dynamic_flat_2D_matrix<C2, T, Container, Other_row>& rhs) {
    if constexpr (C != C2)
      return false;
    else
      return lhs.container_ == rhs.container_;
  }
  template <std::size_t C2, class Other_row>
  friend bool operator!=(const Dynamic_flat_2D_matrix& lhs,
                         const Dynamic_flat_2D_matrix<C2, T, Container, Other_row>& rhs) {
    return !(lhs == rhs);
  }

  size_type get_number_of_rows() const { return R_; }
  constexpr static size_type get_number_of_columns() { return C; }
  size_type get_number_of_entries() const { return get_number_of_rows() * C; }

  Container& get_underlying_container() { return container_; }
  const Container& get_underlying_container() const { return container_; }

 private:
  Container container_;
  size_type R_;

  constexpr static size_type _index(unsigned int r, unsigned int c) { return r * C + c; }

  template <class Iterator>
  auto _to_container_iterator(Iterator it) {
    return container_.begin() + it.get_current_row_index() * C;
  }
  template <class Iterator>
  iterator _to_row_iterator(Iterator it) {
    size_type pos = std::distance(container_.begin(), it) / C;
    return iterator(&container_, pos, get_number_of_rows());
  }
};

}  // namespace Gudhi

namespace std {

template <size_t C, class Container>
struct tuple_size<Gudhi::Row_view<C, Container> > : integral_constant<size_t, C> {};

template <size_t I, size_t C, class Container>
struct tuple_element<I, Gudhi::Row_view<C, Container> > {
  static_assert(I < C, "Value mismatch at argument 1 in template parameter list.");

  using type = typename Gudhi::Row_view<C, Container>::value_type;
};

template <size_t R, size_t C, typename T, class Container, class Row>
struct tuple_size<Gudhi::Flat_2D_matrix<R, C, T, Container, Row> > : integral_constant<size_t, R> {};

template <size_t I, size_t R, size_t C, typename T, class Container, class Row>
struct tuple_element<I, Gudhi::Flat_2D_matrix<R, C, T, Container, Row> > {
  static_assert(I < R, "Value mismatch at argument 1 in template parameter list.");

  using type = Row;
};

}  // namespace std

#endif  // GUDHI_FLAT_2D_MATRIX_H_