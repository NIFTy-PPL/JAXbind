import sys
import pybind11
from setuptools import find_packages, setup, Extension

pkgname = 'jax_linop'
version = '0.1.0'

include_dirs = [pybind11.get_include(True), pybind11.get_include(False)]

extra_compile_args = ['-std=c++17', '-fvisibility=hidden']

python_module_link_args = []

if sys.platform == 'darwin':
    extra_compile_args += ['-mmacosx-version-min=10.14']
    python_module_link_args += ['-mmacosx-version-min=10.14']
elif sys.platform == 'win32':
    extra_compile_args = ['/EHsc', '/std:c++17']

extensions = [Extension("_jax_linop",
                        language='c++',
                        sources=['src/_jax_linop.cc'],
                        depends=['src/_jax_linop.cc', 'setup.py'],
                        include_dirs=include_dirs,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=python_module_link_args)]

setup(name=pkgname,
      version=version,
      packages=find_packages(include=["jax_linop"]),
      ext_modules=extensions
      )
