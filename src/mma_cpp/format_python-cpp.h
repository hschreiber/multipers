/*    This file is part of the MMA Library - https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT.
 *    See file LICENSE for full license details.
 *    Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence class, renaming and cleanup.
 */
/**
 * @file format_python-cpp.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Functions that change the format of data to communicate between C++ and python.
 */

#ifndef FORMAT_PYTHON_CPP_H_INCLUDED
#define FORMAT_PYTHON_CPP_H_INCLUDED

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

#include "combinatory.h"
#include "utilities.h"

#include "../gudhi/Simplex_tree.h"
#include "../gudhi/Simplex_tree_multi.h"



using Vineyard::boundary_type;
using Vineyard::boundary_matrix;
using Vineyard::filtration_type;
using Vineyard::permutation_type;
using Vineyard::multifiltration_type;

using Vineyard::negInf;

boundary_matrix build_sparse_boundary_matrix_from_simplex_list(
        std::vector<std::vector<unsigned int> >& simplexList);
std::pair<boundary_matrix, std::vector<filtration_type> >
build_boundary_matrix_from_simplex_list(
        std::vector<boundary_type>& simplexList,
        const std::vector<filtration_type>& filtrations,
        std::vector<unsigned int>& indices_of_filtrations_to_order);
bool is_strictly_smaller_simplex(const boundary_type& s1,
                                 const boundary_type& s2);
unsigned int hash_simplex_into_unsigned_int(
        boundary_type& simplex, unsigned int scale);
unsigned int hash_simplex_face_into_unsigned_int(
        boundary_type& simplex, unsigned int j, unsigned int scale);

boundary_matrix build_sparse_boundary_matrix_from_simplex_list(
        std::vector<std::vector<unsigned int> >& simplexList)
{
    unsigned int numberOfSimplices = simplexList.size();
    unsigned int scale = std::pow(10, std::ceil(std::log10(numberOfSimplices)));

    for (unsigned int i = 0; i < numberOfSimplices; i++){
        std::sort(simplexList[i].begin(), simplexList[i].end());
    }
    std::stable_sort(simplexList.begin(),
                     simplexList.end(),
                     is_strictly_smaller_simplex);
    boundary_matrix output(numberOfSimplices);

    // Dictionary to store simplex ids. simplex [0,2,4] number is simplex_id[024];
    // that's why we needed to sort first
    std::unordered_map<unsigned int, unsigned int> simplexIDs;
    for (unsigned int i = 0; i < numberOfSimplices; i++){
        // populate the dictionary with this simplex
        simplexIDs.emplace(hash_simplex_into_unsigned_int(simplexList[i],
                                                          scale),
                           i);

        // If simplex is of dimension 0, there is no boundary
        if (simplexList[i].size() <= 1) continue;

        // Fills the output matrix with the boundary of simplex cursor
        for (unsigned int j = 0; j < simplexList[i].size(); j++){
            unsigned int childID =
                    simplexIDs[hash_simplex_face_into_unsigned_int(simplexList[i],
                                                                   j,
                                                                   scale)];
            output[i].push_back(childID);
        }
    }

    for (unsigned int i = 0; i < numberOfSimplices; i++){
        std::sort(output[i].begin(), output[i].end());
    }
    std::stable_sort(output.begin(), output.end(), is_strictly_smaller_simplex);

    return output;
}

std::pair<boundary_matrix, std::vector<filtration_type> >
build_boundary_matrix_from_simplex_list(
        std::vector<boundary_type> &simplexList,
        const std::vector<filtration_type> &filtrations,
        std::vector<unsigned int>& indices_of_filtrations_to_order)
{
	return std::make_pair(boundary_matrix(),std::vector<filtration_type>());
    unsigned int numberOfSimplices = simplexList.size();
    // for dictionary hashmap
    unsigned int scale = std::pow(10, std::ceil(std::log10(numberOfSimplices)));
    unsigned int filtrationDimension = filtrations.size();

    for (unsigned int i = 0; i < numberOfSimplices; i++){
        std::sort(simplexList[i].begin(), simplexList[i].end());
    }

    //sort list_simplices with filtration
    //of size num_simplices
    permutation_type p = Combinatorics::sort_and_return_permutation<boundary_type>(
                simplexList, &is_strictly_smaller_simplex);

    boundary_matrix boundaries(numberOfSimplices);
    // WARNING We assume here that point filtration has the same order as
    // the ordered list of simplices.
    // This fills the filtration of the 0-skeleton by points_filtration
    std::vector<filtration_type> filtersList(filtrationDimension, filtration_type(numberOfSimplices, negInf));
    for (unsigned int i = 0; i < filtrationDimension; i++)
		for (unsigned int j = 0; j < filtrations[i].size(); j++)
			filtersList[i][j] = filtrations[i][j];

    // permute filters the same as simplices
    for(const unsigned int index : indices_of_filtrations_to_order){
        Combinatorics::compose(filtersList[index], p);
    }

    // Dictionary to store simplex ids. simplex [0,2,4] number is
    // simplex_id[024]; that's why we needed to sort first
    std::unordered_map<unsigned int, unsigned int> simplexID;
    for (unsigned int i = 0; i < numberOfSimplices; i++){
        // populate the dictionary with this simplex
        // stores the id of the simplex
        simplexID.emplace(
                    hash_simplex_into_unsigned_int(simplexList[i], scale),
                    i);

        // If simplex is of dimension 0, there is no boundary
        if (simplexList[i].size() <= 1) continue;

        // Fills the output matrix with the boundary of simplex cursor,
        // and computes filtration of the simplex
        for (unsigned int j = 0; j < simplexList[i].size(); j++){
            // computes the id of the child
            unsigned int childID =
                    simplexID[hash_simplex_face_into_unsigned_int(
                        simplexList[i], j, scale
                        )];

            // add this child to the boundary
            boundaries[i].push_back(childID);

            // this simplex filtration is greater than the childs filtration in the ls case
            for (unsigned int k = 0; k < filtrationDimension; k++)
                filtersList[k][i] = std::max(filtersList[k][i],
                                             filtersList[k][childID]);
        }
    }

    for (unsigned int i = 0; i < numberOfSimplices; i++){
        std::sort(boundaries[i].begin(), boundaries[i].end());
    }

    return std::make_pair(boundaries, filtersList);
}


std::pair<boundary_matrix, filtration_type>
__old__simplextree_to_boundary_filtration(
        std::vector<boundary_type> &simplexList,
        filtration_type &filtration)
{
    unsigned int numberOfSimplices = simplexList.size();
    // for dictionary hashmap
    unsigned int scale = std::pow(10, std::ceil(std::log10(numberOfSimplices)));

    for (unsigned int i = 0; i < numberOfSimplices; i++){
        std::sort(simplexList[i].begin(), simplexList[i].end());
    }

    //sort list_simplices with filtration
    //of size num_simplices
    permutation_type p = Combinatorics::sort_and_return_permutation<boundary_type>(
                simplexList, &is_strictly_smaller_simplex);
	
    boundary_matrix boundaries(numberOfSimplices);
	// permute filters the same as simplices
	Combinatorics::compose(filtration, p);

    // Dictionary to store simplex ids. simplex [0,2,4] number is
    // simplex_id[024]; that's why we needed to sort first
    std::unordered_map<unsigned int, unsigned int> simplexID;
    for (unsigned int i = 0; i < numberOfSimplices; i++){
        // populate the dictionary with this simplex
        // stores the id of the simplex
        simplexID.emplace(
                    hash_simplex_into_unsigned_int(simplexList[i], scale),
                    i);

        // If simplex is of dimension 0, there is no boundary
        if (simplexList[i].size() <= 1) continue;

        // Fills the output matrix with the boundary of simplex cursor,
        // and computes filtration of the simplex
        for (unsigned int j = 0; j < simplexList[i].size(); j++){
            // computes the id of the child
            unsigned int childID =
                    simplexID[hash_simplex_face_into_unsigned_int(
                        simplexList[i], j, scale
                        )];

            // add this child to the boundary
            boundaries[i].push_back(childID);
        }
    }

    for (unsigned int i = 0; i < numberOfSimplices; i++){
        std::sort(boundaries[i].begin(), boundaries[i].end());
    }

    return std::make_pair(boundaries, filtration);
}


std::pair<boundary_matrix, multifiltration_type> simplextree_to_boundary_filtration(const uintptr_t splxptr)
{
	using option = Gudhi::Simplex_tree_options_multidimensional_filtration;
	Gudhi::Simplex_tree<option> &simplexTree = *(Gudhi::Simplex_tree<option>*)(splxptr);

	unsigned int numberOfSimplices = simplexTree.num_simplices();
	boundary_matrix boundaries(numberOfSimplices);
	boundary_matrix simplices(numberOfSimplices);
	if (simplexTree.num_simplices() <= 0)
		return {{}, {{}}};
	unsigned int filtration_number = simplexTree.filtration(*(simplexTree.complex_simplex_range().begin())).size();
	std::vector<filtration_type> filtration(filtration_number, filtration_type(numberOfSimplices));

	unsigned int count = 0;
	for (auto sh : simplexTree.filtration_simplex_range())
		simplexTree.assign_key(sh, count++);

	unsigned int i = 0;
	for (auto &simplex : simplexTree.filtration_simplex_range()){
		for (const auto &simplex_id : simplexTree.boundary_simplex_range(simplex)){
			boundaries[i].push_back(simplexTree.key(simplex_id));
		}
		for (const auto &vertex : simplexTree.simplex_vertex_range(simplex)){
			simplices[i].push_back(vertex);
		}
		const auto &temp = simplexTree.filtration(simplex);
		for (unsigned int j = 0; j< temp.size(); j++)
			filtration[j][i] = temp[j];
		i++;
	}
	for (boundary_type &simplex : simplices){
		std::sort(simplex.begin(), simplex.end());
	}
	permutation_type p = Combinatorics::sort_and_return_permutation<boundary_type>(
				simplices, &is_strictly_smaller_simplex);

	for (auto &F : filtration){
		Combinatorics::compose(F, p);
	}

	Combinatorics::compose(boundaries, p);

	auto inv = Combinatorics::inverse(p);

	for (boundary_type &simplex : boundaries){
		for (unsigned int &b : simplex)
			b = inv[b];
		std::sort(simplex.begin(), simplex.end());
	}


	return std::make_pair(boundaries, filtration);
}







// Lexical order + dimension
bool is_strictly_smaller_simplex(const boundary_type& s1, const boundary_type& s2)
{
    if (s1.size() < s2.size()) return true;
    if (s1.size() > s2.size()) return false;

    for (unsigned int i = 0; i < s1.size(); i++){
        if (s1[i] < s2[i]) return true;
        if (s1[i] > s2[i]) return false;
    }
    return false;
}

// Converts a simplex into an unsigned int for dictionary
unsigned int hash_simplex_into_unsigned_int(
        boundary_type& simplex, unsigned int scale)
{
    std::sort(simplex.begin(), simplex.end());
    unsigned int output = 0;
    for (unsigned int i = 0; i < simplex.size(); i++){
        output += simplex[i] * std::pow(scale,i);
    }
    return output;
}

// converts the simplex j in boundary of simplex to an unsigned int for dictionnary
unsigned int hash_simplex_face_into_unsigned_int(
        boundary_type& simplex, unsigned int j, unsigned int scale)
{
    std::sort(simplex.begin(), simplex.end());

    unsigned int output = 0;
    bool passedThroughJ = 0;
    for (unsigned int i = 0; i < simplex.size(); i++){
        if (i == j){
            passedThroughJ = 1;
            continue;
        }
        output += simplex[i] * std::pow(scale, i - passedThroughJ);
    }

    return output;
}




// void complete_filtration(Gudhi::Simplex_tree<> &simplexTree, std::vector<filtration_type> &filtersList){
//
//     for (filtration_type &fi : filtersList){
//         std::vector<bool> has_filtration(simplexTree.num_simplices());
//         for (unsigned int i = 0; i < fi.size(); i++)
//             has_filtration[i] = true;
//         fi.resize(simplexTree.num_simplices());
//
//     }
// }

// std::vector<unsigned int> splxKey2splxVertices(const boundary_matrix &boundary, unsigned int splx_key){
//     if (boundary[splx_key].empty())
//         return {splx_key};
//     else{
//         std::vector<unsigned int> out(boundary[splx_key].size());
//         for (unsigned int i = 0; i < boundary[splx_key].size(); i++){
//             unsigned int child_key =
//         }
//     }
// }


bool is_face(boundary_type &face, boundary_type &splx){
    if (splx.empty() || splx.size() != face.size()+1) return false;
    if (splx.size() <=2 )
        return splx[0] == face[0] || splx[1] == face[0];
    bool flag = false;
    for (unsigned int i = 0; i < splx.size(); i++){
        if (splx[i] != face[i-flag]){
            if (flag) return false;
            flag = true;
        }
    }
    return true;
}


std::string __old__simplextree2rivet(const uintptr_t splxptr, std::vector<filtration_type> &F) {
    Gudhi::Simplex_tree<> &simplexTree = *(Gudhi::Simplex_tree<>*)(splxptr);

    int numberOfSimplices = simplexTree.num_simplices();
    boundary_matrix simplices(numberOfSimplices);
    unsigned int i = 0;
    for (auto &simplex : simplexTree.complex_simplex_range()){
        for (const auto &vertex : simplexTree.simplex_vertex_range(simplex)){
            simplices[i].push_back(vertex);
        }
        i++;
    }
    for (boundary_type &simplex : simplices){
        std::sort(simplex.begin(), simplex.end());
    }
    std::sort(simplices, &is_strictly_smaller_simplex);
    std::string output;
    output.reserve(numberOfSimplices * 10);
    for (auto &Fi : F)
        Fi.resize(numberOfSimplices, Vineyard::negInf);
    for (int i = 0; i < numberOfSimplices; i++){
        for (const auto &vertex : simplices[i]) {output.append(std::to_string(vertex) + " ");}
        output.append("; ");
        for (filtration_type  &Fi : F){
            Vineyard::filtration_value_type f = Fi[i];
            for (int j=0; j<i; j++){
                if (is_face(simplices[j],simplices[i])){
                    f = std::max(f, Fi[j]);
                }
            }
            Fi[i] = f;
            output.append(std::to_string(Fi[i]) + " ");
        }
        output.append("\n");
    }
    return output;
}




void simplextree2rivet(const std::string& path, const uintptr_t splxptr, std::vector<filtration_type> &F) {
    Gudhi::Simplex_tree<> &simplexTree = *(Gudhi::Simplex_tree<>*)(splxptr);

    int numberOfSimplices = simplexTree.num_simplices();
    boundary_matrix simplices(numberOfSimplices);
    unsigned int i = 0;
    for (auto &simplex : simplexTree.complex_simplex_range()){
        for (const auto &vertex : simplexTree.simplex_vertex_range(simplex)){
            simplices[i].push_back(vertex);
        }
        i++;
    }
    for (boundary_type &simplex : simplices){
        std::sort(simplex.begin(), simplex.end());
    }
    std::sort(simplices, &is_strictly_smaller_simplex);
    std::ofstream output;
    output.open(path, std::ios_base::app);
    for (auto &Fi : F)
        Fi.resize(numberOfSimplices, Vineyard::negInf);
    for (int i = 0; i < numberOfSimplices; i++){
        for (const auto &vertex : simplices[i]) {output << std::to_string(vertex) + " ";}
        output << "; ";
        for (filtration_type  &Fi : F){
            Vineyard::filtration_value_type f = Fi[i];
            for (int j=0; j<i; j++){
                if (is_face(simplices[j],simplices[i])){
                    f = std::max(f, Fi[j]);
                }
            }
            Fi[i] = f;
            output << std::to_string(Fi[i]) + " ";
        }
        output << "\n";
    }
}






#endif // FORMAT_PYTHON_CPP_H_INCLUDED