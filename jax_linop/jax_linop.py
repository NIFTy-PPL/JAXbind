# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society

from functools import partial

import _jax_linop
import jax
import numpy as np
from jax.interpreters import ad, mlir
from jaxlib.hlo_helpers import custom_call

__all__ = ["make_linop"]

# incremented for every registered operator, strictly for uniqueness purposes
_global_opcounter = 0

for _name, _value in _jax_linop.registrations().items():
    jax.lib.xla_client.register_custom_call_target(
        _name, _value, platform="cpu"
    )


def _from_id(objectid):
    import ctypes

    return ctypes.cast(objectid, ctypes.py_object).value


def _exec_abstract(x, stateid, adjoint):
    state = _from_id(stateid)
    shp, tp = state["_func_abstract"](x.shape, x.dtype, adjoint, state)
    return (jax.core.ShapedArray(shp, tp), )


# the values are explained in src/duc0/bindings/typecode.h
_dtype_dict = {
    np.dtype(np.float32): 3,
    np.dtype(np.float64): 7,
    np.dtype(np.complex64): 67,
    np.dtype(np.complex128): 71,
}


def _lowering(ctx, x, *, platform="cpu", stateid, adjoint):
    state = _from_id(stateid)
    if len(ctx.avals_in) != 1:
        raise RuntimeError("need exactly one input object")
    shape_in = ctx.avals_in[0].shape
    dtype_in = ctx.avals_in[0].dtype
    if len(ctx.avals_out) != 1:
        raise RuntimeError("need exactly one output object")
    shape_out, dtype_out = state["_func_abstract"](
        shape_in, dtype_in, adjoint, state
    )

    dtype_out_mlir = mlir.dtype_to_ir_type(dtype_out)
    jaxtype_out = mlir.ir.RankedTensorType.get(shape_out, dtype_out_mlir)
    layout_in = tuple(range(len(shape_in) - 1, -1, -1))
    layout_out = tuple(range(len(shape_out) - 1, -1, -1))

    # add array
    operands = [x]
    # add opid and stateid
    operands.append(mlir.ir_constant(state["_opid"]))
    operands.append(mlir.ir_constant(stateid))
    # add forward/adjoint mode
    operands.append(mlir.ir_constant(int(adjoint)))
    # add input dtype, rank, and shape
    operands.append(mlir.ir_constant(_dtype_dict[dtype_in]))
    operands.append(mlir.ir_constant(len(shape_in)))
    operands += [mlir.ir_constant(i) for i in shape_in]
    # add output dtype, rank, and shape
    operands.append(mlir.ir_constant(_dtype_dict[dtype_out]))
    operands.append(mlir.ir_constant(len(shape_out)))
    operands += [mlir.ir_constant(i) for i in shape_out]

    operand_layouts = [layout_in] + [()] * (7 + len(shape_in) + len(shape_out))

    if platform == "cpu":
        return custom_call(
            platform + "_linop",
            result_types=[jaxtype_out],
            result_layouts=[layout_out],
            operands=operands,
            operand_layouts=operand_layouts,
        ).results
    elif platform == "gpu":
        raise ValueError("No GPU support")
    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


def _jvp(args, tangents, *, stateid, adjoint):
    res = _prim.bind(args[0], stateid=stateid, adjoint=adjoint)
    return (
        res,
        jax.lax.zeros_like_array(res) if type(tangents[0]) is ad.Zero else
        _prim.bind(tangents[0], stateid=stateid, adjoint=adjoint),
    )


def _transpose(cotangents, args, *, stateid, adjoint):
    tmp = _prim.bind(cotangents[0].conj(), stateid=stateid, adjoint=not adjoint)
    tmp[0] = tmp[0].conj()
    return tmp


def _batch(args, axes, *, stateid, adjoint):
    raise NotImplementedError("FIXME")


_prim = jax.core.Primitive("ducc_linop_prim")
_prim.multiple_results = True
_prim.def_impl(partial(jax.interpreters.xla.apply_primitive, _prim))
_prim.def_abstract_eval(_exec_abstract)

for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _prim, partial(_lowering, platform=platform), platform=platform
    )
    ad.primitive_jvps[_prim] = _jvp
    ad.primitive_transposes[_prim] = _transpose
    jax.interpreters.batching.primitive_batchers[_prim] = _batch


def _call(x, state, adjoint):
    return _prim.bind(x, stateid=id(state), adjoint=adjoint)


def make_linop(func, func_abstract, **kwargs):
    """Create Jax functions for the provided linear operator

    Parameters
    ----------
    func : the function which performs the linear operation
        The function signature must be (inp, out, adjoint, state), where
        inp and out are numpy.ndarrays of float[32/64] or complex[64/128] type,
        adjoint is a bool indicating whether the forward or adjoint operation
        should be carried out, and state is a dictionary containing additional
        information that the operator might need.
    func_abstract : a function which computes the shape and dtype of the
        operator's output from shape and dtype of its input.
        Its signature must be (shape, dtype, adjoint, state), where `adjoint`
        and `state` are analogous to those from `func`, `shape` is a
        tuple of integers, and dtype is a numpy data type (float[32/64]
        or complex[64/128]).
        The function must return the tuple (shape_out, dtype_out).
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
    - when calling `func` or `func_abstract`, `state` may contain some entries
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
    kwargs_clean["_func_abstract"] = func_abstract
    return partial(_call, state=kwargs_clean, adjoint=False)
