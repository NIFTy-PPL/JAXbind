cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_jaxbind NOSTRIP NB_SUPPRESS_WARNINGS NOMINSIZE src/_jaxbind.cc)

install(TARGETS _jaxbind LIBRARY DESTINATION .)
