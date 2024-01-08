import sys
import os.path
import itertools
from glob import iglob
import os

from setuptools import find_packages, setup, Extension
import pybind11

pkgname = 'jax_linop'
version = '0.1.0'

def _get_files_by_suffix(directory, suffix):
    path = directory
    iterable_sources = (iglob(os.path.join(root, '*.'+suffix))
                        for root, dirs, files in os.walk(path))
    return list(itertools.chain.from_iterable(iterable_sources))


include_dirs = [pybind11.get_include(True),
                pybind11.get_include(False)]

extra_compile_args = ['-std=c++17', '-fvisibility=hidden']

python_module_link_args = []

define_macros = [("PKGNAME", pkgname),
                 ("PKGVERSION", version)]

if sys.platform == 'darwin':
    extra_compile_args += ['-mmacosx-version-min=10.14', '-pthread']
    python_module_link_args += ['-mmacosx-version-min=10.14', '-pthread']
elif sys.platform == 'win32':
    extra_compile_args = ['/EHsc', '/std:c++17']
    if do_optimize:
        extra_compile_args += ['/Ox']
else:
    extra_compile_args += ['-Wfatal-errors',
                           '-Wfloat-conversion',
                           '-W',
                           '-Wall',
                           '-Wstrict-aliasing',
                           '-Wwrite-strings',
                           '-Wredundant-decls',
                           '-Woverloaded-virtual',
                           '-Wcast-qual',
                           '-Wcast-align',
                           '-Wpointer-arith',
                           '-Wnon-virtual-dtor',
                           '-Wzero-as-null-pointer-constant']

depfiles = (_get_files_by_suffix('.', 'h') +
            _get_files_by_suffix('.', 'cc') +
            ['setup.py'])

extensions = [Extension("_jax_linop",
                        language='c++',
                        sources=['src/_jax_linop.cc'],
                        depends=depfiles,
                        include_dirs=include_dirs,
                        define_macros=define_macros,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=python_module_link_args)]

setup(name=pkgname,
      version=version,
      packages=find_packages(include=["jax_linop"]),
      ext_modules = extensions
      )
