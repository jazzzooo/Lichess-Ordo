from Cython.Build import cythonize
from distutils import sysconfig
from setuptools import setup

sysconfig.get_config_vars()["CC"] = "clang"
sysconfig.get_config_vars()["CFLAGS"] = "-DNDEBUG -fopenmp -Ofast -march=native -mtune=native -flto=full"
sysconfig.get_config_vars()["LDSHARED"] = "clang -shared -fopenmp -Ofast -march=native -mtune=native -flto=full"

setup(ext_modules=cythonize("prediction_errors.pyx", compiler_directives={"language_level": "3"}))
