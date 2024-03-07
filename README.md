# Simple linear operators for Jax

This package provides convenience functionality which allows easy integration
of arbitrary linear operations into Jax.

## Requirements

- [Python >= 3.8](https://www.python.org/)
- only when compiling from source: [pybind11](https://github.com/pybind/pybind11)
- only when compiling from source: a C++17-capable compiler, e.g.
  - `g++` 7 or later
  - `clang++`
  - MSVC 2019 or later
  - Intel `icpx` (oneAPI compiler series). (Note that the older `icpc` compilers
    are not supported.)

## Sources

The latest version of jax_linop can be obtained by cloning the repository via

    git clone https://gitlab.mpcdf.mpg.de/mtr/jax_linop.git

## TODOs

* Tests:
    - test mlin
* Check if reference to function can get lost
* Paper
  * final editing
* PiPy release
* CI Pipline
* README

## Licensing terms

All source code in this package is released under the 3-clause BSD license.
