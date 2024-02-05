from multipers.simplex_tree_multi import SimplexTreeMulti
from gudhi import SimplexTree
import gudhi as gd
import numpy as np
import os

from typing import Optional

mpfree_path = None
function_delaunay_path = None

mpfree_in_path = "multipers_mpfree_input.scc"
mpfree_out_path = "multipers_mpfree_output.scc"
function_delaunay_out_path = "function_delaunay_output.scc"
function_delaunay_in_path = "function_delaunay_input.txt" # point cloud

def scc_parser(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    # Find scc2020
    while lines[0].strip() != "scc2020":
        lines = lines[1:]
    lines = lines[1:]
    # stripped scc2020 we can start

    def pass_line(line):
        return len(line) == 0 or line[0] == "#"

    for i, line in enumerate(lines):
        line = line.strip()
        if pass_line(line):
            continue
        num_parameters = int(line)
        lines = lines[i + 1 :]
        break

    block_sizes = []

    for i, line in enumerate(lines):
        line = line.strip()
        if pass_line(line):
            continue
        block_sizes = [int(i) for i in line.split(" ")]
        lines = lines[i + 1 :]
        break
    blocks = []
    for block_size in block_sizes:
        counter = block_size
        block_filtrations = []
        block_boundaries = []
        for i, line in enumerate(lines):
            if counter == 0:
                lines = lines[i:]
                break
            line = line.strip()
            if pass_line(line):
                continue
            filtration, boundary = line.split(";")
            block_filtrations.append(
                    [float(x) for x in filtration.split(" ") if len(x) > 0]
                    )
            block_boundaries.append([int(x) for x in boundary.split(" ") if len(x) > 0])
            counter -= 1
        blocks.append((block_filtrations, block_boundaries))

    return blocks


def _init_external_softwares(requires=[]):

    global function_delaunay_path, mpfree_path, mpfree_in_path, mpfree_out_path, function_delaunay_out_path, function_delaunay_in_path
    if function_delaunay_path is not None and mpfree_path is not None:
        return

    from shutil import which
    if mpfree_path is None:
        a = which("./mpfree")
        b = which("mpfree")
        if a:
            mpfree_path = a
        elif b:
            mpfree_path = b

    if function_delaunay_path is None:
        a = which("./function_delaunay")
        b = which("function_delaunay")
        if a:
            function_delaunay_path = a
        elif b:
            function_delaunay_path = b

    if mpfree_path is None and "mpfree" in requires:
        raise Exception(
"""mpfree not found. Install it from https://bitbucket.org/mkerber/mpfree/,
and put it in the current directory or in your $PATH.
                """
                )
    if function_delaunay_path is None  and "function_delaunay" in requires:
        raise Exception(
"""Function_Delaunay not found. Install it from https://bitbucket.org/mkerber/function_delaunay/,
and put it in the current directory or in your $PATH.
                """
                )

    shm_memory = "/tmp/"  # on unix, we can write in RAM instead of disk.
    if os.access(shm_memory, os.W_OK):
        mpfree_in_path = shm_memory + mpfree_in_path
        mpfree_out_path = shm_memory + mpfree_out_path
        function_delaunay_in_path = shm_memory + function_delaunay_in_path
        function_delaunay_out_path = shm_memory + function_delaunay_out_path

def minimal_presentation_from_str_mpfree(
        path:str,
        full_resolution=True,
        dimension: int | np.int64 = 1,
        clear: bool = True,
        id: str = "",  # For parallel stuff
        verbose:bool=False,
    ):
    global mpfree_path, mpfree_in_path, mpfree_out_path
    if not mpfree_path:
        _init_external_softwares(requires=["mpfree"])

    resolution_str = "--resolution" if full_resolution else ""
    # print(mpfree_in_path + id, mpfree_out_path + id)
    if os.path.exists(mpfree_out_path + id):
        os.remove(mpfree_out_path + id)
    verbose_arg = "> /dev/null 2>&1" if not verbose else ""
    os.system(
        f"{mpfree_path} {resolution_str} --dim={dimension} {path} {mpfree_out_path+id} {verbose_arg}"
             )
    blocks = scc_parser(mpfree_out_path + id)
    if clear:
        clear_io(path, mpfree_out_path + id)
    return blocks

def minimal_presentation_from_mpfree(
        simplextree: SimplexTreeMulti,
        full_resolution: bool = True,
        dimension: int | np.int64 = 1,
        clear: bool = True,
        id: str = "",  # For parallel stuff
        verbose:bool=False,
    ):
    global mpfree_path, mpfree_in_path, mpfree_out_path
    if mpfree_path is None:
        _init_external_softwares(requires=["mpfree"])

    simplextree.to_scc(
            path=mpfree_in_path + id,
            rivet_compatible=False,
            strip_comments=False,
            ignore_last_generators=False,
            overwrite=True,
            reverse_block=True,
            )
    return minimal_presentation_from_str_mpfree(mpfree_in_path+id,full_resolution,dimension,clear,id,verbose)




def function_delaunay_presentation(
        point_cloud:np.ndarray,
        function_values:np.ndarray,
        id:str = "",
        clear:bool = True,
        verbose:bool=False,
        degree: Optional[int] = None,
        ):
    global function_delaunay_path, function_delaunay_in_path, function_delaunay_out_path
    if  function_delaunay_path is None :
        _init_external_softwares(requires=["function_delaunay"])

    to_write = np.concatenate([point_cloud, function_values.reshape(-1,1)], axis=1)
    np.savetxt(function_delaunay_in_path+id,to_write,delimiter=' ')
    verbose_arg = "> /dev/null 2>&1" if not verbose else ""
    degree_arg = f"--minpres {degree}" if degree is not None else ""
    if os.path.exists(function_delaunay_out_path + id):
        os.remove(function_delaunay_out_path+ id)
    os.system(
        f"{function_delaunay_path} {degree_arg} {function_delaunay_in_path+id} {function_delaunay_out_path+id} {verbose_arg}"
             )

    blocks = scc_parser(function_delaunay_out_path + id)
    if clear:
        clear_io(function_delaunay_out_path + id, function_delaunay_in_path + id)
    return blocks



def clear_io(*args):
    global mpfree_in_path, mpfree_out_path, function_delaunay_out_path
    for x in [mpfree_in_path, mpfree_out_path, function_delaunay_out_path] + list(args):
        if os.path.exists(x):
            os.remove(x)





from multipers.mma_structures cimport Finitely_critical_multi_filtration,uintptr_t,boundary_matrix,float,pair,vector,intptr_t
cdef extern from "multiparameter_module_approximation/format_python-cpp.h" namespace "Gudhi::multiparameter::mma":
    pair[boundary_matrix, vector[Finitely_critical_multi_filtration]] simplextree_to_boundary_filtration(uintptr_t)
    vector[pair[boundary_matrix, vector[vector[float]]]] simplextree_to_scc(uintptr_t)

def simplex_tree2boundary_filtrations(simplextree:SimplexTreeMulti | SimplexTree):
    """Computes a (sparse) boundary matrix, with associated filtration. Can be used as an input of approx afterwards.

    Parameters
    ----------
    simplextree: Gudhi or mma simplextree
        The simplextree defining the filtration to convert to boundary-filtration.

    Returns
    -------
    B:List of lists of ints
        The boundary matrix.
    F: List of 1D filtration
        The filtrations aligned with B; the i-th simplex of this simplextree has boundary B[i] and filtration(s) F[i].

    """
    cdef intptr_t cptr
    if isinstance(simplextree, SimplexTreeMulti):
        cptr = simplextree.thisptr
    elif isinstance(simplextree, SimplexTree):
        temp_st = gd.SimplexTreeMulti(simplextree, parameters=1)
        cptr = temp_st.thisptr
    else:
        raise TypeError("Has to be a simplextree")
    cdef pair[boundary_matrix, vector[Finitely_critical_multi_filtration]] cboundary_filtration = simplextree_to_boundary_filtration(cptr)
    boundary = cboundary_filtration.first
    multi_filtrations = np.array(Finitely_critical_multi_filtration.to_python(cboundary_filtration.second))
    return boundary, multi_filtrations

def simplextree2scc(simplextree:SimplexTreeMulti | SimplexTree):
    cdef intptr_t cptr
    if isinstance(simplextree, SimplexTreeMulti):
        cptr = simplextree.thisptr
    elif isinstance(simplextree, SimplexTree):
        temp_st = gd.SimplexTreeMulti(simplextree, parameters=1)
        cptr = temp_st.thisptr
    else:
        raise TypeError("Has to be a simplextree")

    return simplextree_to_scc(cptr)


