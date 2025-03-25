cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PKGNAME ${SKBUILD_PROJECT_NAME})
set(PKGVERSION ${SKBUILD_PROJECT_VERSION})

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

set(CXX_SOURCES src/_jaxbind.cc)

find_package(pybind11 REQUIRED)
pybind11_add_module(_jaxbind NO_EXTRAS ${PYMOD_SOURCES} ${CXX_SOURCES})

target_compile_definitions(_jaxbind PRIVATE PKGNAME=${PKGNAME} PKGVERSION=${PKGVERSION})

include_directories(${CMAKE_SOURCE_DIR}/src)

target_compile_features(_jaxbind PRIVATE cxx_std_17)

install(TARGETS _jaxbind LIBRARY DESTINATION .)
