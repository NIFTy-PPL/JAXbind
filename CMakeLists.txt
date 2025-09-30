cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_jaxbind NOSTRIP NB_SUPPRESS_WARNINGS NOMINSIZE src/_jaxbind.cc)
target_include_directories(_jaxbind PUBLIC ${XLA_DIR})

install(TARGETS _jaxbind LIBRARY DESTINATION .)
