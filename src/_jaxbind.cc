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
#include <xla/ffi/api/ffi.h>

#include <cstring>  // for memcpy()
#include <vector>
#include <map>

namespace detail_pymodule_jax {

namespace nb=nanobind;
namespace ffi = xla::ffi;
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

ffi::Error pycallImpl(int64_t func_id,
                      ffi::AnyBuffer kwargs,
                      ffi::RemainingArgs args,
                      ffi::RemainingRets results)
  {
  nb::gil_scoped_acquire get_GIL;
// MR: disabling the try-catch for the time being, since the diagnostics
// appear to be better when leaving the exception handling to JAX.
//  try {
    static const map<ffi::DataType, nb::dlpack::dtype> tcdict = {
      {ffi::DataType::F32 , nb::dtype<float>()},
      {ffi::DataType::F64 , nb::dtype<double>()},
      {ffi::DataType::U8  , nb::dtype<uint8_t>()},
      {ffi::DataType::U64 , nb::dtype<uint64_t>()},
      {ffi::DataType::C64 , nb::dtype<complex<float>>()},
      {ffi::DataType::C128, nb::dtype<complex<double>>()}
    };
  
    auto dtp_kwargs = tcdict.at(kwargs.element_type());
    auto dims = kwargs.dimensions();
    shape_t shape_kwargs;
    for (auto x : dims) shape_kwargs.push_back(x);
    CNpArr py_kwargs = make_CArr_wrapper(dtp_kwargs, kwargs.untyped_data(), shape_kwargs);
  
    nb::list py_in;
    for (size_t i=0; i<args.size(); i++)
      {
      auto arg = args.get<ffi::AnyBuffer>(i).value();
      auto dtp_a = tcdict.at(arg.element_type());
      auto dims = arg.dimensions();
      shape_t shape_a;
      for (auto x : dims) shape_a.push_back(x);
      // Building "pseudo" numpy arrays on top of the provided memory regions.
      // This should be completely fine, as long as the called function does not
      // keep any references to them.
      CNpArr py_a = make_CArr_wrapper(dtp_a, arg.untyped_data(), shape_a);
      py_in.append(py_a);
      }
  
    nb::list py_out;
    for (size_t i=0; i<results.size(); i++) {
      auto out = results.get<ffi::AnyBuffer>(i).value();
      auto dtp_out = tcdict.at(out->element_type());
      auto dims = out->dimensions();
      shape_t shape_out;
      for (auto x : dims) shape_out.push_back(x);
      NpArr py_o = make_Arr_wrapper(dtp_out, out->untyped_data(), shape_out);
      py_out.append(py_o);
    }
  
    PyObject* raw_ptr = reinterpret_cast<PyObject*>(func_id);
    nb::handle hnd(raw_ptr);
    nb::object func = nb::borrow<nb::object>(hnd);
    func(py_out, py_in, py_kwargs);
//    }
//  catch (...)
//    {
//    return ffi::Error::Internal("Something happened; no idea what");
//    }
  return ffi::Error::Success();
  }

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pycall, pycallImpl,
    ffi::Ffi::Bind()
                  .Attr<int64_t>("id_func")
                  .Arg<ffi::AnyBuffer>()
                  .RemainingArgs()
                  .RemainingRets());

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

