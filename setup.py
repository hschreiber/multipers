
from setuptools import Extension, setup, find_packages
import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options


Options.docstrings = True
Options.embed_pos_in_docstring = True
Options.fast_fail = True
# Options.warning_errors = True

cython_modules = [
	"simplex_tree_multi", 
	"rank_invariant",
	"function_rips",
	"multiparameter_module_approximation", 
	'hilbert_function',
	'euler_characteristic',
	# 'cubical_multi_complex',
	'point_measure_integration',
]

cythonize_flags = {
	# "depfile":True,
	# "nthreads": len(cython_modules), # Broken on mac
	# "show_all_warnings":True,
}

cython_compiler_directives = {
	"language_level": 3,
	"embedsignature": True,
	"binding": True,
	"infer_types": True,
	"boundscheck":False,
	"wraparound":True,
	# "profile":True,
	# "unraisable_tracebacks":True,
	"annotation_typing":True,
	"emit_code_comments":True,
}

## When venv is not properly set, we have to add the current python path
import site
PYTHON_ENV_PATH = "/".join((site.getsitepackages()[0]).split("/")[:-3]) # removes lib / python3.x / site-packages
INCLUDE_PATH = PYTHON_ENV_PATH + "/include/"
LIBRARY_PATH = PYTHON_ENV_PATH + "/lib/"

cpp_dirs = [
	"multipers/gudhi",
	"multipers", 
	np.get_include(),
	INCLUDE_PATH,
]
library_dirs = [
	LIBRARY_PATH,
]

python_dependencies=[
#	"gudhi",
	"numpy",
	"Cython",
#	"scikit-learn",
	"tbb",
	"tbb-devel",
#	"tqdm",
	"setuptools",
]





extensions = [Extension(f"multipers.{module}",
		sources=[f"multipers/{module}.pyx",],
		language='c++',
		extra_compile_args=[
			"-Ofast",
			#"-march=native",
			"-std=c++17", #Stuck here bc of Windows...
			# "-fno-aligned-new", # Uncomment this if you have trouble compiling on macos.
			"-Wall",
		],
		include_dirs=cpp_dirs,
		define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
		libraries=["tbb", "tbbmalloc"],
		library_dirs=library_dirs,
	) for module in cython_modules
]
setup(
	name='multipers',
	author="David Loiseaux",
	author_email="david.loiseaux@inria.fr",
	description="Multiparameter persistence toolkit",
	version="1.0",
	ext_modules=cythonize(
		extensions, 
		compiler_directives=cython_compiler_directives, 
		**cythonize_flags),
	packages=find_packages(),
	package_data={
		"multipers":["*.pyi", "*.pyx", "*.pxd"],
		},
	python_requires=">=3.10",

)
