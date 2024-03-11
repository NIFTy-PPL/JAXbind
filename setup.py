import sys
import os
import codecs
import pybind11
from setuptools import find_packages, setup, Extension

pkgname = 'jaxbind'


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


include_dirs = [pybind11.get_include(True), pybind11.get_include(False)]

extra_compile_args = ['-std=c++17', '-fvisibility=hidden']

python_module_link_args = []

if sys.platform == 'darwin':
    extra_compile_args += ['-mmacosx-version-min=10.14']
    python_module_link_args += ['-mmacosx-version-min=10.14']
elif sys.platform == 'win32':
    extra_compile_args = ['/EHsc', '/std:c++17']

extensions = [Extension("_jaxbind",
                        language='c++',
                        sources=['src/_jaxbind.cc'],
                        depends=['src/_jaxbind.cc', 'setup.py'],
                        include_dirs=include_dirs,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=python_module_link_args)]

setup(name=pkgname,
      version=get_version(f"{pkgname}/__init__.py"),
      packages=find_packages(include=["jaxbind"]),
      ext_modules=extensions
      )
