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

inline nb::object normalizeDtype(const nb::object &dtype)
  {
  static nb::object converter = nb::module_::import_("numpy").attr("dtype");
  return converter(dtype);
  }
template<typename T> inline nb::object Dtype();
template<> inline nb::object Dtype<uint8_t>()
  { static auto res = normalizeDtype(nb::cast("u1")); return res; }
template<> inline nb::object Dtype<uint64_t>()
  { static auto res = normalizeDtype(nb::cast("u8")); return res; }
template<> inline nb::object Dtype<float>()
  { static auto res = normalizeDtype(nb::cast("f4")); return res; }
template<> inline nb::object Dtype<double>()
  { static auto res = normalizeDtype(nb::cast("f8")); return res; }
template<> inline nb::object Dtype<complex<float>>()
  { static auto res = normalizeDtype(nb::cast("c8")); return res; }
template<> inline nb::object Dtype<complex<double>>()
  { static auto res = normalizeDtype(nb::cast("c16")); return res; }
template<typename T> bool isDtype(const nb::object &dtype)
  { return Dtype<T>().equal(dtype); }

using NpArr = nb::ndarray<nb::numpy, nb::device::cpu>;
using CNpArr = nb::ndarray<nb::numpy, nb::ro, nb::device::cpu>;
template<typename T> using NpArrT = nb::ndarray<nb::numpy, nb::device::cpu, T>;
template<typename T> using CNpArrT = nb::ndarray<nb::numpy, nb::ro, nb::device::cpu, T>;

template<typename T> CNpArr make_CArr_wrapper(const T *ptr, const shape_t &dims)
  {
  nb::capsule owner(ptr, [](void *p) noexcept {});
  CNpArr res_(CNpArrT<T>(ptr, dims.size(), dims.data(), owner));
  return res_;
  }
CNpArr make_CArr_wrapper(const nb::object &dtype, const void *ptr, const shape_t &dims)
  {
  if (isDtype<uint8_t>(dtype))
    return make_CArr_wrapper<uint8_t>(reinterpret_cast<const uint8_t *>(ptr), dims);
  if (isDtype<uint64_t>(dtype))
    return make_CArr_wrapper<uint64_t>(reinterpret_cast<const uint64_t *>(ptr), dims);
  if (isDtype<float>(dtype))
    return make_CArr_wrapper<float>(reinterpret_cast<const float *>(ptr), dims);
  if (isDtype<double>(dtype))
    return make_CArr_wrapper<double>(reinterpret_cast<const double *>(ptr), dims);
  if (isDtype<complex<float>>(dtype))
    return make_CArr_wrapper<complex<float>>(reinterpret_cast<const complex<float> *>(ptr), dims);
  if (isDtype<complex<double>>(dtype))
    return make_CArr_wrapper<complex<double>>(reinterpret_cast<const complex<double> *>(ptr), dims);
  throw runtime_error("unsupported data type");
  }
template<typename T> NpArr make_Arr_wrapper(T *ptr, const shape_t &dims)
  {
  nb::capsule owner(ptr, [](void *p) noexcept {});
  NpArr res_(NpArrT<T>(ptr, dims.size(), dims.data(), owner));
  return res_;
  }
NpArr make_Arr_wrapper(const nb::object &dtype, void *ptr, const shape_t &dims)
  {
  if (isDtype<uint8_t>(dtype))
    return make_Arr_wrapper<uint8_t>(reinterpret_cast<uint8_t *>(ptr), dims);
  if (isDtype<uint64_t>(dtype))
    return make_Arr_wrapper<uint64_t>(reinterpret_cast<uint64_t *>(ptr), dims);
  if (isDtype<float>(dtype))
    return make_Arr_wrapper<float>(reinterpret_cast<float *>(ptr), dims);
  if (isDtype<double>(dtype))
    return make_Arr_wrapper<double>(reinterpret_cast<double *>(ptr), dims);
  if (isDtype<complex<float>>(dtype))
    return make_Arr_wrapper<complex<float>>(reinterpret_cast<complex<float> *>(ptr), dims);
  if (isDtype<complex<double>>(dtype))
    return make_Arr_wrapper<complex<double>>(reinterpret_cast<complex<double> *>(ptr), dims);
  throw runtime_error("unsupported data type");
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

  static const map<uint8_t, nb::object> tcdict = {
    { 3, Dtype<float>()},
    { 7, Dtype<double>()},
    {32, Dtype<uint8_t>()},
    {39, Dtype<uint64_t>()},
    {67, Dtype<complex<float>>()},
    {71, Dtype<complex<double>>()}};

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
    vector<size_t> shape_a;
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
    vector<size_t> shape_out;
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

