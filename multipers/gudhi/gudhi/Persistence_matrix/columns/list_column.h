/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Hannah Schreiber
 *
 *    Copyright (C) 2022-24 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file list_column.h
 * @author Hannah Schreiber
 * @brief Contains the @ref List_column class.
 * Also defines the std::hash method for @ref List_column.
 */

#ifndef PM_LIST_COLUMN_H
#define PM_LIST_COLUMN_H

#include <vector>
#include <stdexcept>
#include <type_traits>
#include <list>
#include <utility>  //std::swap, std::move & std::exchange

#include <boost/iterator/indirect_iterator.hpp>

#include <gudhi/Persistence_matrix/allocators/cell_constructors.h>
#include <gudhi/Persistence_matrix/columns/column_utilities.h>

namespace Gudhi {
namespace persistence_matrix {

/**
 * @class List_column list_column.h gudhi/Persistence_matrix/columns/list_column.h
 * @ingroup persistence_matrix
 *
 * @brief Column class following the @ref PersistenceMatrixColumn concept.
 *
 * Column based on a list structure. The cells are always ordered by row index and only non-zero values
 * are stored uniquely in the underlying container.
 * 
 * @tparam Master_matrix An instanciation of @ref Matrix from which all types and options are deduced.
 * @tparam Cell_constructor Factory of @ref Cell classes.
 */
template <class Master_matrix>
class List_column : public Master_matrix::Row_access_option,
                    public Master_matrix::Column_dimension_option,
                    public Master_matrix::Chain_column_option 
{
 public:
  using Master = Master_matrix;
  using index = typename Master_matrix::index;
  using id_index = typename Master_matrix::id_index;
  using dimension_type = typename Master_matrix::dimension_type;
  using Field_element_type = typename Master_matrix::element_type;
  using Cell = typename Master_matrix::Cell_type;
  using Column_settings = typename Master_matrix::Column_settings;

 private:
  using Field_operators = typename Master_matrix::Field_operators;
  using Column_type = std::list<Cell*>;
  using Cell_constructor = typename Master_matrix::Cell_constructor;

 public:
  using iterator = boost::indirect_iterator<typename Column_type::iterator>;
  using const_iterator = boost::indirect_iterator<typename Column_type::const_iterator>;
  using reverse_iterator = boost::indirect_iterator<typename Column_type::reverse_iterator>;
  using const_reverse_iterator = boost::indirect_iterator<typename Column_type::const_reverse_iterator>;

  List_column(Column_settings* colSettings = nullptr);
  template <class Container_type = typename Master_matrix::boundary_type>
  List_column(const Container_type& nonZeroRowIndices, Column_settings* colSettings);
  template <class Container_type = typename Master_matrix::boundary_type, class Row_container_type>
  List_column(index columnIndex, 
              const Container_type& nonZeroRowIndices, 
              Row_container_type* rowContainer,
              Column_settings* colSettings);
  template <class Container_type = typename Master_matrix::boundary_type>
  List_column(const Container_type& nonZeroChainRowIndices, 
              dimension_type dimension, 
              Column_settings* colSettings);
  template <class Container_type = typename Master_matrix::boundary_type, class Row_container_type>
  List_column(index columnIndex, 
              const Container_type& nonZeroChainRowIndices, 
              dimension_type dimension,
              Row_container_type* rowContainer, 
              Column_settings* colSettings);
  List_column(const List_column& column, 
              Column_settings* colSettings = nullptr);
  template <class Row_container_type>
  List_column(const List_column& column, 
              index columnIndex, 
              Row_container_type* rowContainer,
              Column_settings* colSettings = nullptr);
  List_column(List_column&& column) noexcept;
  ~List_column();

  std::vector<Field_element_type> get_content(int columnLength = -1) const;
  bool is_non_zero(id_index rowIndex) const;
  bool is_empty() const;
  std::size_t size() const;

  template <class Map_type>
  void reorder(const Map_type& valueMap, [[maybe_unused]] index columnIndex = -1);
  void clear();
  void clear(id_index rowIndex);

  id_index get_pivot() const;
  Field_element_type get_pivot_value() const;

  iterator begin() noexcept;
  const_iterator begin() const noexcept;
  iterator end() noexcept;
  const_iterator end() const noexcept;
  reverse_iterator rbegin() noexcept;
  const_reverse_iterator rbegin() const noexcept;
  reverse_iterator rend() noexcept;
  const_reverse_iterator rend() const noexcept;

  template <class Cell_range>
  List_column& operator+=(const Cell_range& column);
  List_column& operator+=(List_column& column);

  List_column& operator*=(unsigned int v);

  // this = v * this + column
  template <class Cell_range>
  List_column& multiply_target_and_add(const Field_element_type& val, const Cell_range& column);
  List_column& multiply_target_and_add(const Field_element_type& val, List_column& column);
  // this = this + column * v
  template <class Cell_range>
  List_column& multiply_source_and_add(const Cell_range& column, const Field_element_type& val);
  List_column& multiply_source_and_add(List_column& column, const Field_element_type& val);

  friend bool operator==(const List_column& c1, const List_column& c2) {
    if (&c1 == &c2) return true;

    auto it1 = c1.column_.begin();
    auto it2 = c2.column_.begin();
    if (c1.column_.size() != c2.column_.size()) return false;
    while (it1 != c1.column_.end() && it2 != c2.column_.end()) {
      if constexpr (Master_matrix::Option_list::is_z2) {
        if ((*it1)->get_row_index() != (*it2)->get_row_index()) return false;
      } else {
        if ((*it1)->get_row_index() != (*it2)->get_row_index() || (*it1)->get_element() != (*it2)->get_element())
          return false;
      }
      ++it1;
      ++it2;
    }
    return true;
  }
  friend bool operator<(const List_column& c1, const List_column& c2) {
    if (&c1 == &c2) return false;

    auto it1 = c1.column_.begin();
    auto it2 = c2.column_.begin();
    while (it1 != c1.column_.end() && it2 != c2.column_.end()) {
      if ((*it1)->get_row_index() != (*it2)->get_row_index()) return (*it1)->get_row_index() < (*it2)->get_row_index();
      if constexpr (!Master_matrix::Option_list::is_z2) {
        if ((*it1)->get_element() != (*it2)->get_element()) return (*it1)->get_element() < (*it2)->get_element();
      }
      ++it1;
      ++it2;
    }
    return it2 != c2.column_.end();
  }

  // Disabled with row access.
  List_column& operator=(const List_column& other);

  friend void swap(List_column& col1, List_column& col2) {
    swap(static_cast<typename Master_matrix::Row_access_option&>(col1),
         static_cast<typename Master_matrix::Row_access_option&>(col2));
    swap(static_cast<typename Master_matrix::Column_dimension_option&>(col1),
         static_cast<typename Master_matrix::Column_dimension_option&>(col2));
    swap(static_cast<typename Master_matrix::Chain_column_option&>(col1),
         static_cast<typename Master_matrix::Chain_column_option&>(col2));
    col1.column_.swap(col2.column_);
    std::swap(col1.operators_, col2.operators_);
    std::swap(col1.cellPool_, col2.cellPool_);
  }

 private:
  using ra_opt = typename Master_matrix::Row_access_option;
  using dim_opt = typename Master_matrix::Column_dimension_option;
  using chain_opt = typename Master_matrix::Chain_column_option;

  Column_type column_;
  Field_operators* operators_;
  Cell_constructor* cellPool_;

  template <class Column_type, class Cell_iterator, typename F1, typename F2, typename F3, typename F4>
  friend void _generic_merge_cell_to_column(Column_type& targetColumn,
                                            Cell_iterator& itSource,
                                            typename Column_type::Column_type::iterator& itTarget,
                                            F1&& process_target,
                                            F2&& process_source,
                                            F3&& update_target1,
                                            F4&& update_target2,
                                            bool& pivotIsZeroed);
  template <class Column_type, class Cell_range, typename F1, typename F2, typename F3, typename F4, typename F5>
  friend bool _generic_add_to_column(const Cell_range& source,
                                     Column_type& targetColumn,
                                     F1&& process_target,
                                     F2&& process_source,
                                     F3&& update_target1,
                                     F4&& update_target2,
                                     F5&& finish_target);
  template <class Column_type, class Cell_range>
  friend bool _add_to_column(const Cell_range& source, Column_type& targetColumn);
  template <class Column_type, class Cell_range>
  friend bool _multiply_target_and_add_to_column(const typename Column_type::Field_element_type& val,
                                                 const Cell_range& source,
                                                 Column_type& targetColumn);
  template <class Column_type, class Cell_range>
  friend bool _multiply_source_and_add_to_column(const typename Column_type::Field_element_type& val,
                                                 const Cell_range& source,
                                                 Column_type& targetColumn);

  void _delete_cell(typename Column_type::iterator& it);
  Cell* _insert_cell(const Field_element_type& value,
                     id_index rowIndex,
                     const typename Column_type::iterator& position);
  void _insert_cell(id_index rowIndex, const typename Column_type::iterator& position);
  void _update_cell(const Field_element_type& value, id_index rowIndex, const typename Column_type::iterator& position);
  void _update_cell(id_index rowIndex, const typename Column_type::iterator& position);
  template <class Cell_range>
  bool _add(const Cell_range& column);
  template <class Cell_range>
  bool _multiply_target_and_add(const Field_element_type& val, const Cell_range& column);
  template <class Cell_range>
  bool _multiply_source_and_add(const Cell_range& column, const Field_element_type& val);
};

template <class Master_matrix>
inline List_column<Master_matrix>::List_column(Column_settings* colSettings)
    : ra_opt(),
      dim_opt(),
      chain_opt(),
      operators_(nullptr),
      cellPool_(colSettings == nullptr ? nullptr : &(colSettings->cellConstructor))
{
  if (operators_ == nullptr && cellPool_ == nullptr) return; // to allow default constructor which gives a dummy column
  if constexpr (!Master_matrix::Option_list::is_z2) {
    operators_ = &(colSettings->operators);
  }
}

template <class Master_matrix>
template <class Container_type>
inline List_column<Master_matrix>::List_column(const Container_type& nonZeroRowIndices,
                                                                 Column_settings* colSettings)
    : ra_opt(),
      dim_opt(nonZeroRowIndices.size() == 0 ? 0 : nonZeroRowIndices.size() - 1),
      chain_opt(),
      column_(nonZeroRowIndices.size()),
      operators_(nullptr),
      cellPool_(&(colSettings->cellConstructor))
{
  static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type,
                "Constructor not available for chain columns, please specify the dimension of the chain.");

  auto it = column_.begin();
  if constexpr (Master_matrix::Option_list::is_z2) {
    for (id_index id : nonZeroRowIndices) {
      _update_cell(id, it++);
    }
  } else {
    operators_ = &(colSettings->operators);
    for (const auto& p : nonZeroRowIndices) {
      _update_cell(operators_->get_value(p.second), p.first, it++);
    }
  }
}

template <class Master_matrix>
template <class Container_type, class Row_container_type>
inline List_column<Master_matrix>::List_column(index columnIndex,
                                                                 const Container_type& nonZeroRowIndices,
                                                                 Row_container_type* rowContainer,
                                                                 Column_settings* colSettings)
    : ra_opt(columnIndex, rowContainer),
      dim_opt(nonZeroRowIndices.size() == 0 ? 0 : nonZeroRowIndices.size() - 1),
      chain_opt([&] {
        if constexpr (Master_matrix::Option_list::is_z2) {
          return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : *std::prev(nonZeroRowIndices.end());
        } else {
          return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : std::prev(nonZeroRowIndices.end())->first;
        }
      }()),
      column_(nonZeroRowIndices.size()),
      operators_(nullptr),
      cellPool_(&(colSettings->cellConstructor))
{
  static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type,
                "Constructor not available for chain columns, please specify the dimension of the chain.");

  auto it = column_.begin();
  if constexpr (Master_matrix::Option_list::is_z2) {
    for (id_index id : nonZeroRowIndices) {
      _update_cell(id, it++);
    }
  } else {
    operators_ = &(colSettings->operators);
    for (const auto& p : nonZeroRowIndices) {
      _update_cell(operators_->get_value(p.second), p.first, it++);
    }
  }
}

template <class Master_matrix>
template <class Container_type>
inline List_column<Master_matrix>::List_column(const Container_type& nonZeroRowIndices,
                                                                 dimension_type dimension, 
                                                                 Column_settings* colSettings)
    : ra_opt(),
      dim_opt(dimension),
      chain_opt([&] {
        if constexpr (Master_matrix::Option_list::is_z2) {
          return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : *std::prev(nonZeroRowIndices.end());
        } else {
          return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : std::prev(nonZeroRowIndices.end())->first;
        }
      }()),
      column_(nonZeroRowIndices.size()),
      operators_(nullptr),
      cellPool_(&(colSettings->cellConstructor))
{
  auto it = column_.begin();
  if constexpr (Master_matrix::Option_list::is_z2) {
    for (id_index id : nonZeroRowIndices) {
      _update_cell(id, it++);
    }
  } else {
    operators_ = &(colSettings->operators);
    for (const auto& p : nonZeroRowIndices) {
      _update_cell(operators_->get_value(p.second), p.first, it++);
    }
  }
}

template <class Master_matrix>
template <class Container_type, class Row_container_type>
inline List_column<Master_matrix>::List_column(
    index columnIndex, 
    const Container_type& nonZeroRowIndices, 
    dimension_type dimension,
    Row_container_type* rowContainer, 
    Column_settings* colSettings)
    : ra_opt(columnIndex, rowContainer),
      dim_opt(dimension),
      chain_opt([&] {
        if constexpr (Master_matrix::Option_list::is_z2) {
          return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : *std::prev(nonZeroRowIndices.end());
        } else {
          return nonZeroRowIndices.begin() == nonZeroRowIndices.end() ? -1 : std::prev(nonZeroRowIndices.end())->first;
        }
      }()),
      column_(nonZeroRowIndices.size()),
      operators_(nullptr),
      cellPool_(&(colSettings->cellConstructor))
{
  auto it = column_.begin();
  if constexpr (Master_matrix::Option_list::is_z2) {
    for (id_index id : nonZeroRowIndices) {
      _update_cell(id, it++);
    }
  } else {
    operators_ = &(colSettings->operators);
    for (const auto& p : nonZeroRowIndices) {
      _update_cell(operators_->get_value(p.second), p.first, it++);
    }
  }
}

template <class Master_matrix>
inline List_column<Master_matrix>::List_column(const List_column& column, 
                                                                 Column_settings* colSettings)
    : ra_opt(),
      dim_opt(static_cast<const dim_opt&>(column)),
      chain_opt(static_cast<const chain_opt&>(column)),
      column_(column.column_.size()),
      operators_(colSettings == nullptr ? column.operators_ : nullptr),
      cellPool_(colSettings == nullptr ? column.cellPool_ : &(colSettings->cellConstructor))
{
  static_assert(!Master_matrix::Option_list::has_row_access,
                "Simple copy constructor not available when row access option enabled. Please specify the new column "
                "index and the row container.");

  if constexpr (!Master_matrix::Option_list::is_z2){
    if (colSettings != nullptr) operators_ = &(colSettings->operators);
  }

  auto it = column_.begin();
  for (const Cell* cell : column.column_) {
    if constexpr (Master_matrix::Option_list::is_z2) {
      _update_cell(cell->get_row_index(), it++);
    } else {
      _update_cell(cell->get_element(), cell->get_row_index(), it++);
    }
  }
}

template <class Master_matrix>
template <class Row_container_type>
inline List_column<Master_matrix>::List_column(const List_column& column, 
                                                                 index columnIndex,
                                                                 Row_container_type* rowContainer,
                                                                 Column_settings* colSettings)
    : ra_opt(columnIndex, rowContainer),
      dim_opt(static_cast<const dim_opt&>(column)),
      chain_opt(static_cast<const chain_opt&>(column)),
      column_(column.column_.size()),
      operators_(colSettings == nullptr ? column.operators_ : nullptr),
      cellPool_(colSettings == nullptr ? column.cellPool_ : &(colSettings->cellConstructor))
{
  if constexpr (!Master_matrix::Option_list::is_z2){
    if (colSettings != nullptr) operators_ = &(colSettings->operators);
  }

  auto it = column_.begin();
  for (const Cell* cell : column.column_) {
    if constexpr (Master_matrix::Option_list::is_z2) {
      _update_cell(cell->get_row_index(), it++);
    } else {
      _update_cell(cell->get_element(), cell->get_row_index(), it++);
    }
  }
}

template <class Master_matrix>
inline List_column<Master_matrix>::List_column(List_column&& column) noexcept
    : ra_opt(std::move(static_cast<ra_opt&>(column))),
      dim_opt(std::move(static_cast<dim_opt&>(column))),
      chain_opt(std::move(static_cast<chain_opt&>(column))),
      column_(std::move(column.column_)),
      operators_(std::exchange(column.operators_, nullptr)),
      cellPool_(std::exchange(column.cellPool_, nullptr)) 
{}

template <class Master_matrix>
inline List_column<Master_matrix>::~List_column() 
{
  for (auto* cell : column_) {
    if constexpr (Master_matrix::Option_list::has_row_access) ra_opt::unlink(cell);
    cellPool_->destroy(cell);
  }
}

template <class Master_matrix>
inline std::vector<typename List_column<Master_matrix>::Field_element_type>
List_column<Master_matrix>::get_content(int columnLength) const 
{
  if (columnLength < 0 && column_.size() > 0)
    columnLength = column_.back()->get_row_index() + 1;
  else if (columnLength < 0)
    return std::vector<Field_element_type>();

  std::vector<Field_element_type> container(columnLength, 0);
  for (auto it = column_.begin(); it != column_.end() && (*it)->get_row_index() < static_cast<id_index>(columnLength);
       ++it) {
    if constexpr (Master_matrix::Option_list::is_z2) {
      container[(*it)->get_row_index()] = 1;
    } else {
      container[(*it)->get_row_index()] = (*it)->get_element();
    }
  }
  return container;
}

template <class Master_matrix>
inline bool List_column<Master_matrix>::is_non_zero(id_index rowIndex) const 
{
  // could be changed to dichotomic search as column is ordered by row index,
  // but I am not sure if it is really worth it as there is no random access
  // and the columns should not be that long anyway.
  for (const Cell* cell : column_)
    if (cell->get_row_index() == rowIndex) return true;

  return false;
}

template <class Master_matrix>
inline bool List_column<Master_matrix>::is_empty() const 
{
  return column_.empty();
}

template <class Master_matrix>
inline std::size_t List_column<Master_matrix>::size() const 
{
  return column_.size();
}

template <class Master_matrix>
template <class Map_type>
inline void List_column<Master_matrix>::reorder(const Map_type& valueMap, [[maybe_unused]] index columnIndex) 
{
  static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type,
                "Method not available for chain columns.");

  for (auto it = column_.begin(); it != column_.end(); ++it) {
    Cell* cell = *it;
    if constexpr (Master_matrix::Option_list::has_row_access) {
      ra_opt::unlink(cell);
      if (columnIndex != static_cast<index>(-1)) cell->set_column_index(columnIndex);
    }
    cell->set_row_index(valueMap.at(cell->get_row_index()));
    if constexpr (Master_matrix::Option_list::has_intrusive_rows && Master_matrix::Option_list::has_row_access)
      ra_opt::insert_cell(cell->get_row_index(), cell);
  }

  // all cells have to be deleted first, to avoid problem with insertion when row is a set
  if constexpr (!Master_matrix::Option_list::has_intrusive_rows && Master_matrix::Option_list::has_row_access) {
    for (auto it = column_.begin(); it != column_.end(); ++it) {
      Cell* cell = *it;
      ra_opt::insert_cell(cell->get_row_index(), cell);
    }
  }

  column_.sort([](const Cell* c1, const Cell* c2) { return *c1 < *c2; });
}

template <class Master_matrix>
inline void List_column<Master_matrix>::clear() 
{
  static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type,
                "Method not available for chain columns as a base element should not be empty.");

  for (auto* cell : column_) {
    if constexpr (Master_matrix::Option_list::has_row_access) ra_opt::unlink(cell);
    cellPool_->destroy(cell);
  }

  column_.clear();
}

template <class Master_matrix>
inline void List_column<Master_matrix>::clear(id_index rowIndex) 
{
  static_assert(!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type,
                "Method not available for chain columns.");

  auto it = column_.begin();
  while (it != column_.end() && (*it)->get_row_index() != rowIndex) it++;
  if (it != column_.end()) _delete_cell(it);
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::id_index
List_column<Master_matrix>::get_pivot() const 
{
  static_assert(Master_matrix::isNonBasic,
                "Method not available for base columns.");  // could technically be, but is the notion usefull then?

  if constexpr (Master_matrix::Option_list::is_of_boundary_type) {
    if (column_.empty()) return -1;
    return column_.back()->get_row_index();
  } else {
    return chain_opt::get_pivot();
  }
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::Field_element_type
List_column<Master_matrix>::get_pivot_value() const 
{
  static_assert(Master_matrix::isNonBasic,
                "Method not available for base columns.");  // could technically be, but is the notion usefull then?

  if constexpr (Master_matrix::Option_list::is_z2) {
    return 1;
  } else {
    if constexpr (Master_matrix::Option_list::is_of_boundary_type) {
      if (column_.empty()) return 0;
      return column_.back()->get_element();
    } else {
      if (chain_opt::get_pivot() == static_cast<id_index>(-1)) return Field_element_type();
      for (const Cell* cell : column_) {
        if (cell->get_row_index() == chain_opt::get_pivot()) return cell->get_element();
      }
      return Field_element_type();  // should never happen if chain column is used properly
    }
  }
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::iterator
List_column<Master_matrix>::begin() noexcept 
{
  return column_.begin();
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::const_iterator
List_column<Master_matrix>::begin() const noexcept 
{
  return column_.begin();
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::iterator
List_column<Master_matrix>::end() noexcept 
{
  return column_.end();
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::const_iterator
List_column<Master_matrix>::end() const noexcept 
{
  return column_.end();
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::reverse_iterator
List_column<Master_matrix>::rbegin() noexcept 
{
  return column_.rbegin();
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::const_reverse_iterator
List_column<Master_matrix>::rbegin() const noexcept 
{
  return column_.rbegin();
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::reverse_iterator
List_column<Master_matrix>::rend() noexcept 
{
  return column_.rend();
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::const_reverse_iterator
List_column<Master_matrix>::rend() const noexcept 
{
  return column_.rend();
}

template <class Master_matrix>
template <class Cell_range>
inline List_column<Master_matrix>& List_column<Master_matrix>::operator+=(
    const Cell_range& column) 
{
  static_assert((!Master_matrix::isNonBasic || std::is_same_v<Cell_range, List_column>),
                "For boundary columns, the range has to be a column of same type to help ensure the validity of the "
                "base element.");  // could be removed, if we give the responsability to the user.
  static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type),
                "For chain columns, the given column cannot be constant.");

  _add(column);

  return *this;
}

template <class Master_matrix>
inline List_column<Master_matrix>& List_column<Master_matrix>::operator+=(
    List_column& column) 
{
  if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
    // assumes that the addition never zeros out this column.
    if (_add(column)) {
      chain_opt::swap_pivots(column);
      dim_opt::swap_dimension(column);
    }
  } else {
    _add(column);
  }

  return *this;
}

template <class Master_matrix>
inline List_column<Master_matrix>& List_column<Master_matrix>::operator*=(
    unsigned int v) 
{
  if constexpr (Master_matrix::Option_list::is_z2) {
    if (v % 2 == 0) {
      if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
        throw std::invalid_argument("A chain column should not be multiplied by 0.");
      } else {
        clear();
      }
    }
  } else {
    Field_element_type val = operators_->get_value(v);

    if (val == 0u) {
      if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
        throw std::invalid_argument("A chain column should not be multiplied by 0.");
      } else {
        clear();
      }
      return *this;
    }

    if (val == 1u) return *this;

    for (Cell* cell : column_) {
      operators_->multiply_inplace(cell->get_element(), val);
      if constexpr (Master_matrix::Option_list::has_row_access) ra_opt::update_cell(*cell);
    }
  }

  return *this;
}

template <class Master_matrix>
template <class Cell_range>
inline List_column<Master_matrix>& List_column<Master_matrix>::multiply_target_and_add(
    const Field_element_type& val, const Cell_range& column) 
{
  static_assert((!Master_matrix::isNonBasic || std::is_same_v<Cell_range, List_column>),
                "For boundary columns, the range has to be a column of same type to help ensure the validity of the "
                "base element.");  // could be removed, if we give the responsability to the user.
  static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type),
                "For chain columns, the given column cannot be constant.");

  if constexpr (Master_matrix::Option_list::is_z2) {
    if (val) {
      _add(column);
    } else {
      clear();
      _add(column);
    }
  } else {
    _multiply_target_and_add(val, column);
  }

  return *this;
}

template <class Master_matrix>
inline List_column<Master_matrix>& List_column<Master_matrix>::multiply_target_and_add(
    const Field_element_type& val, List_column& column) 
{
  if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
    // assumes that the addition never zeros out this column.
    if constexpr (Master_matrix::Option_list::is_z2) {
      if (val) {
        if (_add(column)) {
          chain_opt::swap_pivots(column);
          dim_opt::swap_dimension(column);
        }
      } else {
        throw std::invalid_argument("A chain column should not be multiplied by 0.");
      }
    } else {
      if (_multiply_target_and_add(val, column)) {
        chain_opt::swap_pivots(column);
        dim_opt::swap_dimension(column);
      }
    }
  } else {
    if constexpr (Master_matrix::Option_list::is_z2) {
      if (val) {
        _add(column);
      } else {
        clear();
        _add(column);
      }
    } else {
      _multiply_target_and_add(val, column);
    }
  }

  return *this;
}

template <class Master_matrix>
template <class Cell_range>
inline List_column<Master_matrix>& List_column<Master_matrix>::multiply_source_and_add(
    const Cell_range& column, const Field_element_type& val) 
{
  static_assert((!Master_matrix::isNonBasic || std::is_same_v<Cell_range, List_column>),
                "For boundary columns, the range has to be a column of same type to help ensure the validity of the "
                "base element.");  // could be removed, if we give the responsability to the user.
  static_assert((!Master_matrix::isNonBasic || Master_matrix::Option_list::is_of_boundary_type),
                "For chain columns, the given column cannot be constant.");

  if constexpr (Master_matrix::Option_list::is_z2) {
    if (val) {
      _add(column);
    }
  } else {
    _multiply_source_and_add(column, val);
  }

  return *this;
}

template <class Master_matrix>
inline List_column<Master_matrix>& List_column<Master_matrix>::multiply_source_and_add(
    List_column& column, const Field_element_type& val) 
{
  if constexpr (Master_matrix::isNonBasic && !Master_matrix::Option_list::is_of_boundary_type) {
    // assumes that the addition never zeros out this column.
    if constexpr (Master_matrix::Option_list::is_z2) {
      if (val) {
        if (_add(column)) {
          chain_opt::swap_pivots(column);
          dim_opt::swap_dimension(column);
        }
      }
    } else {
      if (_multiply_source_and_add(column, val)) {
        chain_opt::swap_pivots(column);
        dim_opt::swap_dimension(column);
      }
    }
  } else {
    if constexpr (Master_matrix::Option_list::is_z2) {
      if (val) {
        _add(column);
      }
    } else {
      _multiply_source_and_add(column, val);
    }
  }

  return *this;
}

template <class Master_matrix>
inline List_column<Master_matrix>& List_column<Master_matrix>::operator=(const List_column& other) 
{
  static_assert(!Master_matrix::Option_list::has_row_access, "= assignement not enabled with row access option.");

  dim_opt::operator=(other);
  chain_opt::operator=(other);

  auto tmpPool = cellPool_;
  cellPool_ = other.cellPool_;

  while (column_.size() > other.column_.size()) {
    if (column_.back() != nullptr) {
      if constexpr (Master_matrix::Option_list::has_row_access) ra_opt::unlink(column_.back());
      tmpPool->destroy(column_.back());
    }
    column_.pop_back();
  }

  column_.resize(other.column_.size(), nullptr);
  auto it = column_.begin();
  for (const Cell* cell : other.column_) {
    if (*it != nullptr) {
      if constexpr (Master_matrix::Option_list::has_row_access) ra_opt::unlink(*it);
      tmpPool->destroy(*it);
    }
    if constexpr (Master_matrix::Option_list::is_z2) {
      _update_cell(cell->get_row_index(), it++);
    } else {
      _update_cell(cell->get_element(), cell->get_row_index(), it++);
    }
  }

  operators_ = other.operators_;

  return *this;
}

template <class Master_matrix>
inline void List_column<Master_matrix>::_delete_cell(typename Column_type::iterator& it) 
{
  if constexpr (Master_matrix::Option_list::has_row_access) ra_opt::unlink(*it);
  cellPool_->destroy(*it);
  it = column_.erase(it);
}

template <class Master_matrix>
inline typename List_column<Master_matrix>::Cell* List_column<Master_matrix>::_insert_cell(
    const Field_element_type& value, id_index rowIndex, const typename Column_type::iterator& position)
{
  if constexpr (Master_matrix::Option_list::has_row_access) {
    Cell* newCell = cellPool_->construct(ra_opt::columnIndex_, rowIndex);
    newCell->set_element(value);
    column_.insert(position, newCell);
    ra_opt::insert_cell(rowIndex, newCell);
    return newCell;
  } else {
    Cell* newCell = cellPool_->construct(rowIndex);
    newCell->set_element(value);
    column_.insert(position, newCell);
    return newCell;
  }
}

template <class Master_matrix>
inline void List_column<Master_matrix>::_insert_cell(id_index rowIndex,
                                                                       const typename Column_type::iterator& position) 
{
  if constexpr (Master_matrix::Option_list::has_row_access) {
    Cell* newCell = cellPool_->construct(ra_opt::columnIndex_, rowIndex);
    column_.insert(position, newCell);
    ra_opt::insert_cell(rowIndex, newCell);
  } else {
    Cell* newCell = cellPool_->construct(rowIndex);
    column_.insert(position, newCell);
  }
}

template <class Master_matrix>
inline void List_column<Master_matrix>::_update_cell(const Field_element_type& value,
                                                                       id_index rowIndex,
                                                                       const typename Column_type::iterator& position) 
{
  if constexpr (Master_matrix::Option_list::has_row_access) {
    *position = cellPool_->construct(ra_opt::columnIndex_, rowIndex);
    (*position)->set_element(value);
    ra_opt::insert_cell(rowIndex, *position);
  } else {
    *position = cellPool_->construct(rowIndex);
    (*position)->set_element(value);
  }
}

template <class Master_matrix>
inline void List_column<Master_matrix>::_update_cell(id_index rowIndex,
                                                                       const typename Column_type::iterator& position) 
{
  if constexpr (Master_matrix::Option_list::has_row_access) {
    *position = cellPool_->construct(ra_opt::columnIndex_, rowIndex);
    ra_opt::insert_cell(rowIndex, *position);
  } else {
    *position = cellPool_->construct(rowIndex);
  }
}

template <class Master_matrix>
template <class Cell_range>
inline bool List_column<Master_matrix>::_add(const Cell_range& column) 
{
  if (column.begin() == column.end()) return false;
  if (column_.empty()) {  // chain should never enter here.
    column_.resize(column.size());
    auto it = column_.begin();
    for (const Cell& cell : column) {
      if constexpr (Master_matrix::Option_list::is_z2) {
        _update_cell(cell.get_row_index(), it++);
      } else {
        _update_cell(cell.get_element(), cell.get_row_index(), it++);
      }
    }
    return true;
  }

  return _add_to_column(column, *this);
}

template <class Master_matrix>
template <class Cell_range>
inline bool List_column<Master_matrix>::_multiply_target_and_add(const Field_element_type& val,
                                                                            const Cell_range& column) 
{
  return _multiply_target_and_add_to_column(val, column, *this);
}

template <class Master_matrix>
template <class Cell_range>
inline bool List_column<Master_matrix>::_multiply_source_and_add(const Cell_range& column,
                                                                            const Field_element_type& val) 
{
  return _multiply_source_and_add_to_column(val, column, *this);
}

}  // namespace persistence_matrix
}  // namespace Gudhi

/**
 * @ingroup persistence_matrix
 *
 * @brief Hash method for @ref Gudhi::persistence_matrix::List_column.
 * 
 * @tparam Master_matrix Template parameter of @ref Gudhi::persistence_matrix::List_column.
 * @tparam Cell_constructor Template parameter of @ref Gudhi::persistence_matrix::List_column.
 */
template <class Master_matrix>
struct std::hash<Gudhi::persistence_matrix::List_column<Master_matrix> > 
{
  std::size_t operator()(const Gudhi::persistence_matrix::List_column<Master_matrix>& column) const {
    return Gudhi::persistence_matrix::hash_column(column);
  }
};

#endif  // PM_LIST_COLUMN_H
