from Cython.Build import cythonize
from setuptools import setup

setup(ext_modules=cythonize("prediction_errors.pyx", compiler_directives={"language_level": "3"}))
