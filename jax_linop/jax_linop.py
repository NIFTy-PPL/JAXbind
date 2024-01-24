# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society

from functools import partial

import jax
import numpy as np
from jax.interpreters import ad, batching, mlir
from jaxlib.hlo_helpers import custom_call

import _jax_linop

# incremented for every registered operator, strictly for uniqueness purposes
_global_opcounter = 0

for _name, _value in _jax_linop.registrations().items():
    jax.lib.xla_client.register_custom_call_target(
        _name, _value, platform="cpu"
    )


def _from_id(objectid):
    import ctypes

    return ctypes.cast(objectid, ctypes.py_object).value


def _exec_abstract(x, *, stateid, stateTid):
    state = _from_id(stateid)
    shp, tp, _ = state["_func_abstract"](x.shape, x.dtype, state)
    return (jax.core.ShapedArray(shp, tp), )


# the values are explained in src/duc0/bindings/typecode.h
_dtype_dict = {
    np.dtype(np.float32): 3,
    np.dtype(np.float64): 7,
    np.dtype(np.complex64): 67,
    np.dtype(np.complex128): 71,
}


def _lowering(ctx, x, *, platform="cpu", stateid, stateTid):
    state = _from_id(stateid)
    if len(ctx.avals_in) != 1:
        raise RuntimeError("need exactly one input object")
    if len(ctx.avals_out) != 1:
        raise RuntimeError("need exactly one output object")
    shape_x, dtype_x = ctx.avals_in[0].shape, ctx.avals_in[0].dtype
    shape_y, dtype_y = ctx.avals_out[0].shape, ctx.avals_out[0].dtype

    dtype_mlir_y = mlir.dtype_to_ir_type(dtype_y)
    jaxtype_y = mlir.ir.RankedTensorType.get(shape_y, dtype_mlir_y)
    layout_x = tuple(range(len(shape_x) - 1, -1, -1))
    layout_y = tuple(range(len(shape_y) - 1, -1, -1))

    irc = mlir.ir_constant
    operands = [
        x,
        irc(state["_opid"]),
        irc(stateid),
        irc(_dtype_dict[dtype_x]),
        irc(len(shape_x))
    ] + [irc(i) for i in shape_x]
    operands += [irc(_dtype_dict[dtype_y]),
                 irc(len(shape_y))] + [irc(i) for i in shape_y]
    operand_layouts = [layout_x] + [()] * (6 + len(shape_x) + len(shape_y))

    if platform == "cpu":
        return custom_call(
            platform + "_pycall",
            result_types=[jaxtype_y],
            result_layouts=[layout_y],
            operands=operands,
            operand_layouts=operand_layouts,
        ).results
    elif platform == "gpu":
        raise ValueError("No GPU support")
    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


def _jvp(args, tangents, *, stateid, stateTid):
    res = _prim.bind(args[0], stateid=stateid, stateTid=stateTid)
    return (
        res,
        jax.lax.zeros_like_array(res) if type(tangents[0]) is ad.Zero else
        _prim.bind(tangents[0], stateid=stateid, stateTid=stateTid),
    )


def _transpose(cotangents, args, *, stateid, stateTid):
    return _prim.bind(cotangents[0], stateid=stateTid, stateTid=stateid)


def _batch(args, in_axes, *, stateid, stateTid):
    from .custom_map import smap

    ia, _ = in_axes
    state = _from_id(stateid)
    if state["_func_can_batch"] is False:
        y = smap(
            partial(_prim.bind, stateid=stateid, stateTid=stateTid),
            in_axes=(ia, ),
            out_axes=(ia, )
        )(*args)
        oa = ia
    else:
        internal_fields = (
            "_opid", "_func_can_batch", "_batch_axes", "_func", "_func_T",
            "_func_abstract", "_func_abstract_T"
        )
        kw_batch = {
            k.removeprefix("_") if k in internal_fields else k: v
            for k, v in state.items()
        }
        # TODO: transposed batched state
        batch_axes = kw_batch.pop("batch_axes", ())
        for b in batch_axes:
            if ia >= b:
                ia += 1
        kw_batch["batch_axes"] = sorted(batch_axes + (ia, ))

        call = get_linear_call(**kw_batch, deepcopy_kwargs=False)
        x, = args
        y = call(x)

        _, _, ba_wob = state["_func_abstract"](
            x.shape[:ia] + x.shape[ia + 1:], x.dtype, state
        )
        _, _, ba_wb = state["_func_abstract"](
            x.shape, x.dtype, call.keywords["state"]
        )
        out_axes = set(ba_wob) - set(ba_wb)
        oa, = out_axes
        for b in ba_wb[::-1]:
            if oa >= b:
                oa -= 1
    return y, (oa, )


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


def _call(x, *, state, stateT):
    return _prim.bind(x, stateid=id(state), stateTid=id(stateT))


def get_linear_call(
    func,
    func_T,
    /,
    func_abstract,
    func_abstract_T,
    batch_axes=(),
    func_can_batch=False,
    deepcopy_kwargs=True,
    **kwargs
) -> partial:
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

    safe_copy = copy.deepcopy if deepcopy_kwargs is True else copy.copy
    # somehow make sure that kwargs_clean only contains deep copies of
    # everything in kwargs that are not accessible from anywhere else.
    state = safe_copy(kwargs)  # FIXME TODO
    stateT = copy.copy(state)
    global _global_opcounter
    state["_opid"] = stateT["_opid"] = _global_opcounter
    _global_opcounter += 1
    state["_func"] = stateT["_func_T"] = func
    state["_func_T"] = stateT["_func"] = func_T
    state["_func_abstract"] = stateT["_func_abstract_T"] = func_abstract
    state["_func_abstract_T"] = stateT["_func_abstract"] = func_abstract_T
    state["_batch_axes"] = stateT["_batch_axes"] = batch_axes
    state["_func_can_batch"] = stateT["_func_can_batch"] = func_can_batch

    @partial(partial, state=state, stateT=stateT)
    def call(*args, state, stateT):
        assert len(args) == 1
        out, = _call(*args, state=state, stateT=stateT)
        return out

    return call
