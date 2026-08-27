/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Loiseaux, Hannah Schreiber
 *
 *    Copyright (C) 2026 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file Multi_simplex_tree_interface.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains the @ref Gudhi::multi_persistence::Multi_simplex_tree_interface class for python bindings.
 */

#ifndef MP_PY_MULTI_SIMPLEX_TREE_H_INCLUDED
#define MP_PY_MULTI_SIMPLEX_TREE_H_INCLUDED

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>

#include <gudhi/Simplex_tree.h>
#include <gudhi/multi_simplex_tree_helpers.h>
#include <python_interfaces/numpy_utils.h>

#include "interface_helpers.h"

namespace Gudhi {
namespace multi_persistence {

/**
 * @private
 */
template <class MultiFiltrationValue>
class Multi_simplex_tree_interface
    : public Simplex_tree<
          Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<MultiFiltrationValue>> {
 public:
  using Options = Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<MultiFiltrationValue>;
  using Base = Simplex_tree<Options>;
  using Filtration_value = MultiFiltrationValue;
  using value_type = typename Filtration_value::value_type;
  using Vertex_handle = typename Base::Vertex_handle;
  using Simplex_handle = typename Base::Simplex_handle;
  using Simplex = std::vector<Vertex_handle>;
  using Complex_simplex_iterator = typename Base::Complex_simplex_iterator;
  using Skeleton_simplex_iterator = typename Base::Skeleton_simplex_iterator;
  using Boundary_simplex_iterator = typename Base::Boundary_simplex_iterator;
  template <typename U>
  using Tensor1D = nanobind::ndarray<const U, nanobind::ndim<1>, nanobind::any_contig>;
  template <typename U>
  using Tensor2D = nanobind::ndarray<const U, nanobind::ndim<2>>;

  Multi_simplex_tree_interface() = default;
  Multi_simplex_tree_interface(const Base& st) : Base(st) {};
  Multi_simplex_tree_interface(Base&& st) : Base(std::move(st)) {};

  Multi_simplex_tree_interface& operator=(const Base& st) {
    Base::operator=(st);
    return *this;
  }

  Multi_simplex_tree_interface& operator=(Base&& st) {
    Base::operator=(std::move(st));
    return *this;
  }

  // makes the method public
  template <typename OtherSimplexTreeOptions, typename F>
  void copy_from(const Simplex_tree<OtherSimplexTreeOptions> &complex_source, F &&translate_filtration_value) {
    Base::copy_from(complex_source, std::forward<F>(translate_filtration_value));
  }

  bool find_simplex(const Simplex &simplex) { return (Base::find(simplex) != Base::null_simplex()); }

  bool insert(const Simplex& simplex, const Filtration_value& filtration) {
    auto result = Base::insert_simplex_and_subfaces(Base::Filtration_maintenance::INCREASE_NEW, simplex, filtration);
    if (result.first != Base::null_simplex()) Base::clear_filtration();
    return result.second;
  }

  bool insert_force(const Simplex& simplex, const Filtration_value& filtration) {
    auto result = Base::insert_simplex_and_subfaces(Base::Filtration_maintenance::IGNORE_VALIDITY, simplex, filtration);
    Base::clear_filtration();
    return result.second;
  }

  void remove_maximal_simplex(const Simplex& simplex) {
    Base::remove_maximal_simplex(Base::find(simplex));
    Base::clear_filtration();
  }

  Filtration_value& simplex_filtration(const Simplex& simplex) {
    return Base::get_filtration_value(Base::find(simplex));
  }

  const Filtration_value& simplex_filtration(const Simplex& simplex) const {
    return Base::get_filtration_value(Base::find(simplex));
  }

  void assign_simplex_filtration(const Simplex &simplex, const Filtration_value &filtration) {
    Simplex_handle sh = Base::find(simplex);
    if (sh == Base::null_simplex())
      throw std::invalid_argument("Cannot assign a filtration to a simplex that is not in the complex");
    Base::assign_filtration(sh, filtration);
    Base::clear_filtration();
  }

  // TODO: remove
  auto get_simplex_and_filtration(Simplex_handle f_simplex) {
    auto it = Base::simplex_vertex_range(f_simplex);
    Simplex simplex(it.begin(), it.end());
    std::reverse(simplex.begin(), simplex.end());
    return std::make_pair(std::move(simplex), &Base::get_filtration_value(f_simplex));
  }

  auto get_simplices_of_dimension(int dimension) {
    if (dimension < 0) throw std::invalid_argument("Dimension cannot be negative.");

    std::size_t numSimplices = 0;
    for ([[maybe_unused]] auto sh : Base::dimension_simplex_range(dimension)) ++numSimplices;

    std::vector<Vertex_handle> simplices(numSimplices * (dimension + 1));
    std::size_t i = 0;
    for (auto sh : Base::dimension_simplex_range(dimension)) {
      for (auto vertex : Base::simplex_vertex_range(sh)) {
        simplices[i] = vertex;
        ++i;
      }
    }
    return _wrap_as_numpy_array(std::move(simplices), numSimplices, static_cast<std::size_t>(dimension) + 1);
  }

  template <typename T = value_type>
  auto get_edge_list() {
    // TODO: generalize for more parameters? As edges is already a std::vector, it should not be too difficult.
    if (Base::num_parameters() != 2) throw std::logic_error("Method only implemented for 2-parameter filtrations.");

    std::size_t numEdges = 0;
    for ([[maybe_unused]] auto sh : Base::dimension_simplex_range(1)) ++numEdges;

    std::vector<T> edges(numEdges * 4);
    std::size_t i = 0;
    for (auto sh : Base::dimension_simplex_range(1)) {
      for (auto vertex : Base::simplex_vertex_range(sh)) {
        edges[i] = static_cast<T>(vertex);
        ++i;
      }
      const auto& f = Base::get_filtration_value(sh);
      if (f.num_parameters() != 2)
        throw std::runtime_error(
            "Inconsistency between number of parameters of the simplex tree (=2) and the number of parameters of its "
            "filtration values (!=2).");
      edges[i] = static_cast<T>(f(0, 0));
      edges[i + 1] = static_cast<T>(f(0, 1));
      i += 2;
    }

    return _wrap_as_numpy_array(std::move(edges), numEdges, 4);
  }

  auto get_simplex_python_iterator() {
    return _make_iterator("simplex_iterator", Complex_simplex_iterator(this), Complex_simplex_iterator());
  }

  auto get_skeleton_python_iterator(int dimension) {
    return _make_iterator("skeleton_iterator", Skeleton_simplex_iterator(this, dimension), Skeleton_simplex_iterator());
  }

  auto get_boundary_python_iterator(const Simplex& simplex) {
    auto bd_sh = Base::find(simplex);
    if (bd_sh == Base::null_simplex()) throw std::runtime_error("simplex not found - cannot find boundaries");
    return _make_iterator("boundary_iterator", Boundary_simplex_iterator(this, bd_sh), Boundary_simplex_iterator(this));
  }

  void fill_lowerstar(nanobind::object filtration, int axis) {
    if (filtration.is_none()) throw std::invalid_argument("Filtration values cannot be None.");
    // assuming Base::num_parameters() was properly set
    if (axis < 0) axis += Base::num_parameters();
    if (axis < 0 || axis >= Base::num_parameters()) throw std::invalid_argument("Axis is not a valid parameter index.");

    auto cast_as_vector = [&]() -> void {
      std::vector<value_type> val;
      if (!nanobind::try_cast<std::vector<value_type>>(filtration, val))
        throw std::invalid_argument("Filtration values must be either iterable[U] or ndarray[U, ndim=1].");
      // if vertex indexing is continuous, this catches all filtration size problems
      // but if not, the condition is not sufficient, but at least not wrong
      if (val.size() < Base::num_vertices())
        throw std::invalid_argument("Vertex filtration values does not have a value for every vertex.");
      {
        nanobind::gil_scoped_release release;
        Gudhi::multi_persistence::fill_axis_with_lowerstar(*this, val, static_cast<std::size_t>(axis));
      }
    };
    auto cast_first_as_tensor_then_as_vector = [&]<typename U>() -> void {
      if (Tensor1D<U> val; nanobind::try_cast<Tensor1D<U>>(filtration, val, false)) {
        if (val.shape(0) < Base::num_vertices())
          throw std::invalid_argument("Vertex filtration values does not have a value for every vertex.");
        {
          nanobind::gil_scoped_release release;
          Gudhi::multi_persistence::fill_axis_with_lowerstar(*this, Numpy_span(val), static_cast<std::size_t>(axis));
        }
        return;
      }
      cast_as_vector();
    };
    detail::_dispatch_dtype(
        filtration,
        cast_first_as_tensor_then_as_vector,
        []() -> void {
          if (Base::num_vertices() != 0)
            throw std::invalid_argument("Vertex filtration values is empty, but not the simplex tree.");
        },
        cast_as_vector);
  }

 private:
  template <class Iterator>
  class Simplex_filtration_iterator : public boost::iterator_facade<Simplex_filtration_iterator<Iterator>,
                                                                    nanobind::tuple,
                                                                    boost::forward_traversal_tag,
                                                                    nanobind::tuple> {
   public:
    Simplex_filtration_iterator(const Iterator& start, Multi_simplex_tree_interface* tree = nullptr)
        : curr_(start), tree_(tree) {}

   private:
    friend class boost::iterator_core_access;

    bool equal(Simplex_filtration_iterator const& other) const { return curr_ == other.curr_; }

    nanobind::tuple dereference() const {
      // here just in case, but should never happen as never directly used in python
      if (tree_ == nullptr) throw std::runtime_error("Iterator is at the end of the range.");
      return tree_->_get_simplex_and_filtration(*curr_);
    }

    void increment() { ++curr_; }

    Iterator curr_;
    Multi_simplex_tree_interface* tree_;
  };

  nanobind::tuple _get_simplex_and_filtration(Simplex_handle sh) const {
    Simplex simplex;
    for (auto vertex : Base::simplex_vertex_range(sh)) {
      simplex.push_back(vertex);
    }
    std::reverse(simplex.begin(), simplex.end());
    const auto& fil = Base::get_filtration_value(sh);
    return nanobind::make_tuple(_wrap_as_numpy_array(std::move(simplex), simplex.size()),
                                detail::_get_filtration_array(fil));
  }

  template <class Iterator>
  auto _make_iterator(const char* name, Iterator start, Iterator end) const {
    return nanobind::make_iterator(nanobind::type<Multi_simplex_tree_interface>(),
                                   name,
                                   Simplex_filtration_iterator(start, this),
                                   Simplex_filtration_iterator(end));
  }
};

template <class MultiSimplexTreeInterface>
inline MultiSimplexTreeInterface deserialize_multi_simplex_tree_from_python(nanobind::tuple state) {
  // if (nanobind::len(state) != 3)
  //   throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");
  // std::uint8_t version;
  // if (!nanobind::try_cast<std::uint8_t>(state[0], version, false))
  //   throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");
  // if (version < SlicerInterface::SERIALIZATION_VERSION)
  //   throw std::invalid_argument(
  //       "Given state to deserialize is not compatible with current multipers version: try an older release");
  // if (version > SlicerInterface::SERIALIZATION_VERSION)
  //   throw std::invalid_argument(
  //       "Given state to deserialize is not compatible with current multipers version: try an newer release");

  // nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy> data;
  // if (!nanobind::try_cast<nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy>>(state[2], data, false))
  //   throw std::invalid_argument("Given state to deserialize is not compatible with current multipers version.");
  // SlicerInterface slicer;
  // {
  //   nanobind::gil_scoped_release release;
  //   deserialize_value_from_char_buffer(slicer, data.data());
  // }
  // slicer.set_filtration_grid(state[1]);
  // return slicer;
}

}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_MULTI_SIMPLEX_TREE_H_INCLUDED
