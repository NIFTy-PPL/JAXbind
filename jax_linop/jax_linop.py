# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society

from functools import partial

import _jax_linop
import jax
import numpy as np
from jax.interpreters import ad, mlir, batching
from jaxlib.hlo_helpers import custom_call

# incremented for every registered operator, strictly for uniqueness purposes
_global_opcounter = 0

for _name, _value in _jax_linop.registrations().items():
    jax.lib.xla_client.register_custom_call_target(
        _name, _value, platform="cpu"
    )


def _from_id(objectid):
    import ctypes

    return ctypes.cast(objectid, ctypes.py_object).value


def _exec_abstract(x, stateid):
    state = _from_id(stateid)
    shp, tp = state["_func_abstract"](x.shape, x.dtype, state)
    return (jax.core.ShapedArray(shp, tp), )


# the values are explained in src/duc0/bindings/typecode.h
_dtype_dict = {
    np.dtype(np.float32): 3,
    np.dtype(np.float64): 7,
    np.dtype(np.complex64): 67,
    np.dtype(np.complex128): 71,
}


def _lowering(ctx, x, *, platform="cpu", stateid):
    state = _from_id(stateid)
    if len(ctx.avals_in) != 1:
        raise RuntimeError("need exactly one input object")
    shape_in = ctx.avals_in[0].shape
    dtype_in = ctx.avals_in[0].dtype
    if len(ctx.avals_out) != 1:
        raise RuntimeError("need exactly one output object")
    shape_out, dtype_out = state["_func_abstract"](shape_in, dtype_in, state)

    dtype_out_mlir = mlir.dtype_to_ir_type(dtype_out)
    jaxtype_out = mlir.ir.RankedTensorType.get(shape_out, dtype_out_mlir)
    layout_in = tuple(range(len(shape_in) - 1, -1, -1))
    layout_out = tuple(range(len(shape_out) - 1, -1, -1))

    # add array
    operands = [x]
    # add opid and stateid
    operands.append(mlir.ir_constant(state["_opid"]))
    operands.append(mlir.ir_constant(stateid))
    # add input dtype, rank, and shape
    operands.append(mlir.ir_constant(_dtype_dict[dtype_in]))
    operands.append(mlir.ir_constant(len(shape_in)))
    operands += [mlir.ir_constant(i) for i in shape_in]
    # add output dtype, rank, and shape
    operands.append(mlir.ir_constant(_dtype_dict[dtype_out]))
    operands.append(mlir.ir_constant(len(shape_out)))
    operands += [mlir.ir_constant(i) for i in shape_out]

    operand_layouts = [layout_in] + [()] * (6 + len(shape_in) + len(shape_out))

    if platform == "cpu":
        return custom_call(
            platform + "_pycall",
            result_types=[jaxtype_out],
            result_layouts=[layout_out],
            operands=operands,
            operand_layouts=operand_layouts,
        ).results
    elif platform == "gpu":
        raise ValueError("No GPU support")
    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


def _jvp(args, tangents, *, stateid):
    res = _prim.bind(args[0], stateid=stateid)
    return (
        res,
        jax.lax.zeros_like_array(res) if type(tangents[0]) is ad.Zero else
        _prim.bind(tangents[0], stateid=stateid),
    )


def _transpose(cotangents, args, *, stateid):
    state = _from_id(stateid)
    state["_func_T"], state["_func"] = state["_func"], state["_func_T"]
    state["_func_abstract_T"], state["_func_abstract"] = state[
        "_func_abstract"], state["_func_abstract_T"]
    tmp = _prim.bind(cotangents[0], stateid=stateid)
    return tmp


def _batch(args, axes, *, stateid):
    raise NotImplementedError("FIXME")


_prim = jax.core.Primitive("jax_linop_prim")
_prim.multiple_results = True
_prim.def_impl(partial(jax.interpreters.xla.apply_primitive, _prim))
_prim.def_abstract_eval(_exec_abstract)

for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _prim, partial(_lowering, platform=platform), platform=platform
    )
    ad.primitive_jvps[_prim] = _jvp
    ad.primitive_transposes[_prim] = _transpose
    batching.primitive_batchers[_prim] = _batch


def _call(x, state):
    return _prim.bind(x, stateid=id(state))


def get_linear_call(func, func_T, /, func_abstract, func_abstract_T, **kwargs):
    """Create Jax functions for the provided linear operator

    Parameters
    ----------
    func, func_T : linear function respectively its transpose
        The function signature must be (inp, out, state), where
        inp and out are numpy.ndarrays of float[32/64] or complex[64/128] type,
        and state is a dictionary containing additional information that the
        operator might need.
    func_abstract, func_abstract_T : function respectively its tranpose
        computing the shape and dtype of the operator's output from shape and dtype of its input
        Its signature must be (shape, dtype, state), where `state` is analogous
        to the one from `func`, `shape` is a tuple of integers, and dtype is a
        numpy data type (float[32/64] or complex[64/128]). The function must
        return the tuple (shape_out, dtype_out).
    **kwargs : optional arguments that will be provided in `state` when calling
        `func` and `func_abstract`.

    Returns
    -------
    op : function of the operator for use in Jax computations

    Notes
    -----
    - `func` must not return anything; the result of the computation must be
      written into `out`.
    - the contents of `inp` must not be modified.
    - no references to `inp` or `out` may be stored beyond the execution
      time of `func`
    - `state` must not be modified
    - when calling `func`* or `func_abstract`*, `state` may contain some entries
      that the user did not supply in `**kwargs`; these will have names starting
      with an underscore.
    """
    import copy

    # somehow make sure that kwargs_clean only contains deep copies of
    # everything in kwargs that are not accessible from anywhere else.
    kwargs_clean = copy.deepcopy(kwargs)  # FIXME TODO
    global _global_opcounter
    kwargs_clean["_opid"] = _global_opcounter
    _global_opcounter += 1
    kwargs_clean["_func"] = func
    kwargs_clean["_func_T"] = func_T
    kwargs_clean["_func_abstract"] = func_abstract
    kwargs_clean["_func_abstract_T"] = func_abstract_T
    return partial(_call, state=kwargs_clean)
