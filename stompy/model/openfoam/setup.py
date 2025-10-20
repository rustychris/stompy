from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("mesh_ops_cy.py",annotate=True)
)
