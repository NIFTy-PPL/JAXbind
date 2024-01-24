/* SPDX-License-Identifier: BSD-3-Clause */

/*
 *  Jax_linop is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2023, 2024 Max-Planck-Society
 *  Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <map>

namespace detail_pymodule_jax {

namespace py = pybind11;
using namespace std;

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept
  {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
  }

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn)
  { return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET"); }

void pycall(void *out, void **in)
  {
  py::gil_scoped_acquire get_GIL;

  static const map<uint8_t, py::object> tcdict = {
    { 3, py::dtype::of<float>()},
    { 7, py::dtype::of<double>()},
    {67, py::dtype::of<complex<float>>()},
    {71, py::dtype::of<complex<double>>()}};

  py::str dummy;

  size_t nargs = *reinterpret_cast<uint64_t *>(in[0]);
  size_t idx = 1;
  py::list args;
  for (size_t i=0; i<nargs; i++) {
    // Getting type, rank, and shape of the input
    // TODO: encode list/tuple/array type
    auto dtp_a = tcdict.at(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
    size_t ndim_a = *reinterpret_cast<uint64_t *>(in[idx++]);
    vector<size_t> shape_a;
    for (size_t j=0; j<ndim_a; ++j) {
      shape_a.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));
    }
    // Building "pseudo" numpy.ndarays on top of the provided memory regions.
    // This should be completely fine, as long as the called function does not
    // keep any references to them.
    py::array py_a (dtp_a, shape_a, in[idx++], dummy);
    py_a.attr("flags").attr("writeable") = false;
    args.append(py_a);
  }
  // Getting type, rank, and shape of the output
  auto dtp_y = tcdict.at(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
  size_t ndim_y = *reinterpret_cast<uint64_t *>(in[idx++]);
  vector<size_t> shape_y;
  for (size_t i=0; i<ndim_y; ++i) {
    shape_y.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));
  }
  py::array py_y (dtp_y, shape_y, out, dummy);

  // Getting the "state" dictionary from the passed ID
  py::handle hnd(*reinterpret_cast<PyObject **>(in[idx++]));
  auto obj = py::reinterpret_borrow<py::object>(hnd);
  const py::dict state(obj);

  // the "opid" in in[idx++] is not used; it is only passed to guarantee
  // uniqueness of the passed parameters for every distinct operator, so that
  // JAX knows when and when not to recompile.

  // Execute the Python function implementing the linear operation
  state["_func"](args, py_y, state);
  }

pybind11::dict Registrations()
  {
  pybind11::dict dict;
  dict["cpu_pycall"] = EncapsulateFunction(pycall);
  return dict;
  }

}

PYBIND11_MODULE(_jax_linop, m) {
  m.def("registrations", detail_pymodule_jax::Registrations);
}

