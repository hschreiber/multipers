import pickle
from itertools import product

### OPTIONS

## Columns of the matrix backend.
# with ordered by their performance on some synthetic benchmarks.
columns_name = [  # only one column is necessary
    "Available_columns::" + stuff
    for stuff in (
        "INTRUSIVE_SET",
        # "SET",
        # "HEAP",
        # "UNORDERED_SET",
        # "NAIVE_VECTOR",
        # "VECTOR",
        # "INTRUSIVE_LIST",
        # "LIST",
        # "SMALL_VECTOR",
    )
]

## Value types : CTYPE, PYTHON_TYPE, short
value_types = [
    ("int32_t", "np.int32", "i32"),  # necessary
    # ("int64_t", "np.int64", "i64"),
    # ("float", "np.float32", "f32"),  # necessary for mma (TODO: fixme)
    ("double", "np.float64", "f64"),  # necessary
]

## True, False necessary
vineyards_values = [
    #
    True,
    False,
]

## Kcritical Filtrations
kcritical_options = [
    #
    True,
    False,
]

##
matrix_types = [
    #
    "Matrix",
    # "Graph",
    # "Clement",
    "GudhiCohomology",
]

number_of_parameters = [
    (2, "p002"),
    # (3, "p003"),
    (-1, "pAny"),
]

flat_options = [
    #
    True,
    False,
]


def check_combination(backend_type, is_vine, is_kcritical, value_type, column_type, num_parameter, is_flat):
    if backend_type in ["Clement", "Graph"]:
        if not is_vine:
            return False
    if backend_type in ["GudhiCohomology"]:
        if is_vine:
            return False
    if backend_type in ["Graph", "GudhiCohomology"]:
        if column_type[0] != 0:
            return False
    if is_kcritical and num_parameter[0] != -1:
        return False
    if is_flat and num_parameter[0] == -1:
        return False
    return True


def get_slicer(backend_type, is_vine, is_kcritical, value_type, column_type, num_parameter, is_flat):
    stuff = locals()
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    num_p, short_p = num_parameter
    PYTHON_TYPE = f"_{'K' if is_kcritical else ''}Slicer_{backend_type}{col_idx}{'_vine' if is_vine else ''}_{short_p}{'_fil_flat' if is_flat else ''}_{short_type}"
    CTYPE = f"TrucPythonInterface<BackendsEnum::{backend_type},{'true' if is_vine else 'false'},{'true' if is_kcritical else 'false'},{ctype},{col},{num_p},{'true' if is_flat else 'false'}>"
    IS_SIMPLICIAL = False
    IS_VINE = is_vine
    IS_KCRITICAL = is_kcritical
    multi_fil_type = (
        "Multi_critical_filtration[" + ctype + "]"
    )
    one_fil_type1 = (
        "One_critical_filtration[" + ctype + "]"
    )
    one_fil_type2 = (
        "One_critical_filtration_with_" + f"{num_p}" + "_parameters[" + ctype + "]"
    )
    ONE_FIL_TYPE = one_fil_type1 if num_p == -1 else one_fil_type2
    FILTRATION_TYPE = multi_fil_type if is_kcritical else ONE_FIL_TYPE
    return {
        "TRUC_TYPE": CTYPE,
        "C_TEMPLATE_TYPE": "C" + PYTHON_TYPE,
        "PYTHON_TYPE": PYTHON_TYPE,
        "IS_SIMPLICIAL": IS_SIMPLICIAL,
        "IS_VINE": IS_VINE,
        "IS_KCRITICAL": IS_KCRITICAL,
        "C_VALUE_TYPE": ctype,
        "PY_VALUE_TYPE": pytype,
        "COLUMN_TYPE": col.split("::")[1],
        "SHORT_VALUE_TYPE": short_type,
        "FILTRATION_TYPE": FILTRATION_TYPE,
        "ONE_CRITICAL_TYPE": ONE_FIL_TYPE,
        "PERS_BACKEND_TYPE": backend_type,
        "IS_FLOAT": short_type[0] == "f",
        "NUM_PARAMETERS": num_p,
        "PARAM_SHORT": short_p,
        "IS_FLAT": is_flat
    }


# {{for CTYPE_H, CTYPE,PYTHON_TYPE, IS_SIMPLICIAL, IS_VINE, IS_KCRITICAL, C_VALUE_TYPE, PY_VALUE_TYPE, COL, SHORT,FILTRATION_TYPE  in slicers}}
slicers = [
    get_slicer(**kwargs)
    for backend_type, is_vine, is_kcritical, value_type, column_type, num_parameter, is_flat,  in product(
        matrix_types,
        vineyards_values,
        kcritical_options,
        value_types,
        enumerate(columns_name),
        number_of_parameters,
        flat_options
    )
    if check_combination(
        **(
            kwargs := {
                "backend_type": backend_type,
                "is_vine": is_vine,
                "is_kcritical": is_kcritical,
                "value_type": value_type,
                "column_type": column_type,
                "num_parameter": num_parameter,
                "is_flat": is_flat,
            }
        )
    )
]

for D in slicers:
    print(D)

with open("build/tmp/_slicer_names.pkl", "wb") as f:
    pickle.dump(slicers, f)

## Simplextree

Filtrations_types = [
    (
        ("Multi_critical_filtration", True)
        if kcritical
        else ("One_critical_filtration", False)
    )
    for kcritical in kcritical_options
]


## CTYPE, PYTYPE, SHORT, FILTRATION
to_iter = [
    (
        CTYPE,
        PYTYPE,
        SHORT,
        Filtration + "[" + CTYPE + "]",
        is_kcritical,
        ("K" if is_kcritical else "") + "F" + SHORT,
    )
    for (CTYPE, PYTYPE, SHORT), (Filtration, is_kcritical) in product(
        value_types, Filtrations_types
    )
]


with open("build/tmp/_simplextrees_.pkl", "wb") as f:
    pickle.dump(to_iter, f)
