##  Slicers : CPP NAME, CTYPE, PYTHON_TYPE, IS_SIMPLICIAL, IS_VINE, IS_KCRITICAL, CVALUE_TYPE, PYVALUE_TYPE, COLUMN_TYPE, SHORT_DTYPE
slicers = [ # this is temporarily necessary
  ("GeneralVineClementTruc<>", "GeneralVineClementTruc","_SlicerClement", False, True,False, "float", "np.float32", None, "f32", "Finitely_critical_multi_filtration[float]"),
  ("SimplicialVineGraphTruc", "SimplicialVineGraphTruc","_SlicerVineGraph", True,True,False, "float", "np.float32",None, "f32", "Finitely_critical_multi_filtration[float]"),
  ("SimplicialVineMatrixTruc<>", "SimplicialVineMatrixTruc","_SlicerVineSimplicial", True, True,False, "float", "np.float32",None, "f32", "Finitely_critical_multi_filtration[float]"),
  ("SimplicialNoVineMatrixTruc<>", "SimplicialNoVineMatrixTruc", "_SlicerNoVineSimplicial",True,False,False, "float", "np.float32",None, "f32", "Finitely_critical_multi_filtration[float]"),
]
columns_name=[ # only one column is necessary
  "Available_columns::"+stuff
  for stuff in (
  "INTRUSIVE_SET",
  # "LIST",
  # "SET",
  # "HEAP",
  # "VECTOR",
  # "NAIVE_VECTOR",
  # "UNORDERED_SET",
  # "INTRUSIVE_LIST",
  )
]

## Value types : CTYPE, PYTHON_TYPE, short
value_types = [
  ("int32_t", "np.int32",   "i32"), # necessary
  # ("int64_t", "np.int64",   "i64"),
  ("float",   "np.float32", "f32"), # necessary for mma (TODO: fixme)
  ("double",  "np.float64", "f64"), # necessary
]
from itertools import product
def get_slicer(is_vine, is_kcritical, value_type, column_type):
  ctype, pytype, short_type = value_type
  col_idx, col = column_type
  PYTHON_TYPE = f"_{'K' if is_kcritical else ''}Slicer{col_idx}{'_vine' if is_vine else ''}_{short_type}"
  CTYPE = f"MatrixTrucPythonInterface<{'true' if is_vine else 'false'},{'true' if is_kcritical else 'false'},{ctype},{col}>"
  IS_SIMPLICIAL = False
  IS_VINE = is_vine
  IS_KCRITICAL = is_kcritical
  FILTRATION_TYPE = ("KCriticalFiltration" if is_kcritical else "Finitely_critical_multi_filtration") + "[" + ctype + "]"
  return (CTYPE,"C"+PYTHON_TYPE, PYTHON_TYPE, IS_SIMPLICIAL, IS_VINE, IS_KCRITICAL, ctype, pytype, col.split("::")[1], short_type, FILTRATION_TYPE)

matrix_slicers = [
  get_slicer(is_vine, is_kcritical, value_type, column_type)
  for is_vine, is_kcritical, value_type, column_type in product([True,False], [True,False], value_types, enumerate(columns_name))
]

slicers += matrix_slicers
import pickle

with open("_slicer_names.pkl", "wb") as f:
  pickle.dump(slicers, f)
