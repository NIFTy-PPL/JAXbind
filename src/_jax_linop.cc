/* SPDX-License-Identifier: BSD-3-Clause */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  Jax_linop is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2023 Max-Planck-Society
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

void linop(void *out, void **in)
  {
  py::gil_scoped_acquire get_GIL;

  static const map<uint8_t, py::object> tcdict = {
    { 3, py::dtype::of<float>()},
    { 7, py::dtype::of<double>()},
    {67, py::dtype::of<complex<float>>()},
    {71, py::dtype::of<complex<double>>()}};

  // the "opid" in in[1] is not used; it is only passed to guarantee uniqueness
  // of the passed parameters for every distinvt operator, so that JAX knows
  // when and when not to recompile.

  // Getting the "state" dictionary from the passed ID
  py::handle hnd(*reinterpret_cast<PyObject **>(in[2]));
  auto obj = py::reinterpret_borrow<py::object>(hnd);
  const py::dict state(obj);
  // Are we doing the forward operation or the adjoint?
  auto adjoint = bool(*reinterpret_cast<int64_t *>(in[3]));
  
  size_t idx = 4;
  // Getting type, rank, and shape of the input
  auto dtin = tcdict.at(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
  size_t ndim_in = *reinterpret_cast<uint64_t *>(in[idx++]);
  vector<size_t> shape_in;
  for (size_t i=0; i<ndim_in; ++i)
    shape_in.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));
  // Getting type, rank, and shape of the output
  auto dtout = tcdict.at(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
  size_t ndim_out = *reinterpret_cast<uint64_t *>(in[idx++]);
  vector<size_t> shape_out;
  for (size_t i=0; i<ndim_out; ++i)
    shape_out.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));

  // Building "pseudo" numpy.ndarays on top of the provided memory regions.
  // This should be completely fine, as long as the called function does not
  // keep any references to them.
  py::str dummy;
  py::array pyin (dtin, shape_in, in[0], dummy);
//  MR_assert(!pyin.owndata(), "owndata should be false");
  pyin.attr("flags").attr("writeable")=false;
//  MR_assert(!pyin.writeable(), "input array should not be writeable");
  py::array pyout (dtout, shape_out, out, dummy);
//  MR_assert(!pyout.owndata(), "owndata should be false");
//  MR_assert(pyout.writeable(), "output data must be writable");

  // Execute the Python function implementing the linear operation
  state["_func"](pyin, pyout, adjoint, state);
  }

pybind11::dict Registrations()
  {
  pybind11::dict dict;
  dict["cpu_linop"] = EncapsulateFunction(linop);
  return dict;
  }

}

PYBIND11_MODULE(_jax_linop, m) {
    m.def("registrations", detail_pymodule_jax::Registrations);
}

