/* SPDX-License-Identifier: BSD-2-Clause */

/*
 *  JAXbind is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2023-2025 Max-Planck-Society
 *  Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <iostream>

#include <vector>
#include <map>
#include <exception>

namespace detail_pymodule_jax {

namespace nb=nanobind;
using namespace std;

using shape_t = vector<size_t>;

using NpArr = nb::ndarray<nb::numpy, nb::device::cpu>;
using CNpArr = nb::ndarray<nb::numpy, nb::ro, nb::device::cpu>;

CNpArr make_CArr_wrapper(const nb::dlpack::dtype &dtype, const void *ptr, const shape_t &dims)
  {
  nb::capsule owner(ptr, [](void *p) noexcept {});
  CNpArr res_(ptr, dims.size(), dims.data(), owner, nullptr, dtype);
  return res_;
  }
NpArr make_Arr_wrapper(const nb::dlpack::dtype &dtype, void *ptr, const shape_t &dims)
  {
  nb::capsule owner(ptr, [](void *p) noexcept {});
  NpArr res_(ptr, dims.size(), dims.data(), owner, nullptr, dtype);
  return res_;
  }

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
nb::capsule EncapsulateFunction(T* fn)
  { return nb::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET"); }

void pycall(void *out_raw, void **in)
  {
  nb::gil_scoped_acquire get_GIL;

  static const map<uint8_t, nb::dlpack::dtype> tcdict = {
    { 3, nb::dtype<float>()},
    { 7, nb::dtype<double>()},
    {32, nb::dtype<uint8_t>()},
    {39, nb::dtype<uint64_t>()},
    {67, nb::dtype<complex<float>>()},
    {71, nb::dtype<complex<double>>()}};

  nb::str dummy;

  nb::handle hnd(*reinterpret_cast<PyObject **>(in[0]));
  auto func = nb::borrow<nb::object>(hnd);

  size_t idx = 1;
  size_t nargs = *reinterpret_cast<uint64_t *>(in[idx++]);
  nb::list py_in;
  for (size_t i=0; i<nargs; i++) {
    // Getting type, rank, and shape of the input
    auto dtp_a = tcdict.at(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
    size_t ndim_a = *reinterpret_cast<uint64_t *>(in[idx++]);
    shape_t shape_a;
    for (size_t j=0; j<ndim_a; ++j) {
      shape_a.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));
    }
    // Building "pseudo" numpy arrays on top of the provided memory regions.
    // This should be completely fine, as long as the called function does not
    // keep any references to them.
    CNpArr py_a = make_CArr_wrapper(dtp_a, in[idx++], shape_a);
    py_in.append(py_a);
  }

  // if we have only one output, out_raw is a void * pointing to the data of this output
  // otherwise, out_raw is a void ** pointing to an array of void * pointing to the individual data
  void **out = reinterpret_cast<void **>(out_raw);
  void *out_single = reinterpret_cast<void *>(out_raw);
  size_t nout = *reinterpret_cast<uint64_t *>(in[idx++]);
  nb::list py_out;
  for (size_t i=0; i<nout; i++) {
    // Getting type, rank, and shape of the output
    auto dtp_out = tcdict.at(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
    size_t ndim_out = *reinterpret_cast<uint64_t *>(in[idx++]);
    shape_t shape_out;
    for (size_t j=0; j<ndim_out; ++j) {
      shape_out.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));
    }
    NpArr py_o = make_Arr_wrapper(dtp_out, (nout==1) ? out_single : out[i], shape_out);
    py_out.append(py_o);
  }

  auto dtp_kwargs = tcdict.at(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
  size_t size_kwargs = *reinterpret_cast<uint64_t *>(in[idx++]);
  CNpArr py_kwargs = make_CArr_wrapper(dtp_kwargs, in[idx++], {size_kwargs});

  // Execute the Python function implementing the desired operation
  func(py_out, py_in, py_kwargs);
  }

nb::dict Registrations()
  {
  nb::dict dict;
  dict["cpu_pycall"] = EncapsulateFunction(pycall);
  return dict;
  }

}

NB_MODULE(_jaxbind, m) {
  m.def("registrations", detail_pymodule_jax::Registrations);
}

