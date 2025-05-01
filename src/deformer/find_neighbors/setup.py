from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Fix the compiler path by explicitly setting it with the leading slash
os.environ["CC"] = "/usr/bin/clang"
os.environ["CXX"] = "/usr/bin/clang++"

# Match the extension name to the actual filename (mesh_utils.pyx)
extensions = [
    Extension(
        "deformer_cython",  # Output module name
        ["find_neighbors.pyx"],  # Input source file (your actual file)
        include_dirs=[np.get_include()],
        language="c++",  # Use C++ compiler
        extra_compile_args=["-O3"],
    )
]

setup(
    name="deformer_cython",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=True
    )
)
