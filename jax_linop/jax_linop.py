# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society

import pickle
from functools import partial

import jax
import jaxlib.mlir.dialects.stablehlo as hlo
import jaxlib.mlir.ir as ir
import numpy as np
from jax.interpreters import ad, batching, mlir
from jax.interpreters.mlir import ir_constant as irc
from jaxlib.hlo_helpers import custom_call

import _jax_linop

# incremented for every registered operator, strictly for uniqueness purposes
_global_opcounter = 0
_global_states = []

for _name, _value in _jax_linop.registrations().items():
    jax.lib.xla_client.register_custom_call_target(
        _name, _value, platform="cpu"
    )


def _from_id(objectid):
    import ctypes

    return ctypes.cast(objectid, ctypes.py_object).value


def _exec_abstract(*args, _func, _func_T, _func_abstract, _func_abstract_T, **kwargs):
    shp, dtp, _ = _func_abstract(*args, **kwargs)
    return (jax.core.ShapedArray(shp, dtp), )


# the values are explained in src/duc0/bindings/typecode.h
_dtype_dict = {
    np.dtype(np.float32): 3,
    np.dtype(np.float64): 7,
    np.dtype(np.uint8): 32,
    np.dtype(np.complex64): 67,
    np.dtype(np.complex128): 71,
}


def _lowering(
    ctx, *args, _func, _func_T, _func_abstract, _func_abstract_T, _platform="cpu", **kwargs
):
    assert len(ctx.avals_out) == 1
    shape_y, dtype_y = ctx.avals_out[0].shape, ctx.avals_out[0].dtype
    jaxtype_y = mlir.ir.RankedTensorType.get(
        shape_y, mlir.dtype_to_ir_type(dtype_y)
    )
    layout_y = tuple(range(len(shape_y) - 1, -1, -1))

    operands = [irc(id(_func))]  # Pass the ID of the callable to C++
    operands_layout = [()]
    operands += [irc(len(args))]
    operands_layout += [()]
    assert len(args) == len(ctx.avals_in)
    # All `args` are assumed to be JAX arrays
    for a, ca in zip(args, ctx.avals_in):
        operands += [irc(_dtype_dict[ca.dtype]),
                     irc(ca.ndim)] + [irc(i) for i in ca.shape] + [a]
        lyt_a = tuple(range(ca.ndim - 1, -1, -1))
        operands_layout += [()] * (2 + ca.ndim) + [lyt_a]

    operands += [irc(_dtype_dict[dtype_y]),
                 irc(len(shape_y))] + [irc(i) for i in shape_y]
    operands_layout += [()] * (2 + len(shape_y))

    kwargs = np.frombuffer(pickle.dumps(kwargs), dtype=np.uint8)
    kwargs_ir = hlo.constant(
        ir.DenseElementsAttr.get(kwargs, type=ir.IntegerType.get_unsigned(8))
    )
    operands += [irc(_dtype_dict[kwargs.dtype]), irc(kwargs.size), kwargs_ir]
    operands_layout += [(), (), [0]]

    if _platform == "cpu":
        return custom_call(
            _platform + "_pycall",
            result_types=[jaxtype_y],
            result_layouts=[layout_y],
            operands=operands,
            operand_layouts=operands_layout,
        ).results
    elif _platform == "gpu":
        raise ValueError("No GPU support")
    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


def _jvp(args, tangents, *, _func, _func_T, _func_abstract, _func_abstract_T, **kwargs):
    res = _prim.bind(
        *args,
        **kwargs,
        _func=_func,
        _func_T=_func_T,
        _func_abstract=_func_abstract,
        _func_abstract_T=_func_abstract_T
    )

    def make_zeros(tan):
        return jax.lax.zeros_like_array(res) if type(tan) is ad.Zero else tan

    if all(type(t) is ad.Zero for t in tangents):
        tans = (jax.lax.zeros_like_array(res), )
    else:
        tans = None
        for i, t in enumerate(tangents):
            t = make_zeros(t)
            tn = _prim.bind(
                *args[:i],
                t,
                *args[i + 1:],
                **kwargs,
                _func=_func,
                _func_T=_func_T,
                _func_abstract=_func_abstract,
                _func_abstract_T=_func_abstract_T
            )
            tans = tn if tans is None else tuple(
                t + tn_i for t, tn_i in zip(tans, tn)
            )
    assert tans is not None
    return (res, tans)


def _transpose(cotangents, args, *, _func, _func_T, _func_abstract, _func_abstract_T, **kwargs):
    return _prim.bind(
        *cotangents,
        **kwargs,
        _func=_func_T,
        func_T=_func,
        _func_abstract=_func_abstract_T,
        _func_abstract_T=_func_abstract
    )


def _batch(args, in_axes, *, stateid, stateTid):
    from .custom_map import smap

    state = _from_id(stateid)
    if state["_func_can_batch"] is False:
        out_axes = in_axes
        y = smap(
            partial(_prim.bind, stateid=stateid, stateTid=stateTid),
            in_axes=in_axes,
            out_axes=out_axes
        )(*args)
    else:
        ia, = in_axes
        internal_fields = (
            "_opid", "_func_can_batch", "_batch_axes", "_func", "_func_T",
            "_func_abstract", "_func_abstract_T"
        )
        kw_batch = {
            k.removeprefix("_") if k in internal_fields else k: v
            for k, v in state.items()
        }
        del kw_batch["opid"]
        func, func_T = kw_batch.pop("func"), kw_batch.pop("func_T")
        batch_axes = kw_batch.pop("batch_axes", ())
        for b in batch_axes:
            if ia >= b:
                ia += 1
        batch_axes = tuple(sorted(batch_axes + (ia, )))

        call = get_linear_call(
            func,
            func_T,
            **kw_batch,
            batch_axes=batch_axes,
            deepcopy_kwargs=False
        )
        x, = args
        y = (call(x), )  # Consistent signature with `_prim.bind`

        global _global_states  # HACK AND FIXME
        _global_states.append((call.keywords["state"], call.keywords["stateT"]))

        _, _, ba_wob = state["_func_abstract"](
            x.shape[:ia] + x.shape[ia + 1:], x.dtype, state
        )
        _, _, ba_wb = state["_func_abstract"](
            x.shape, x.dtype, call.keywords["state"]
        )
        oa, = set.difference(set(ba_wb), set(ba_wob))
        for b in ba_wob[::-1]:
            if oa >= b:
                oa -= 1
        assert oa >= 0
        out_axes = (oa)
    return y, out_axes


_prim = jax.core.Primitive("jax_linop_prim")
_prim.multiple_results = True
_prim.def_impl(partial(jax.interpreters.xla.apply_primitive, _prim))
_prim.def_abstract_eval(_exec_abstract)

for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _prim, partial(_lowering, _platform=platform), platform=platform
    )
    ad.primitive_jvps[_prim] = _jvp
    ad.primitive_transposes[_prim] = _transpose
    batching.primitive_batchers[_prim] = _batch


def _call(*args, _func, _func_T, _func_abstract, _func_abstract_T, **kwargs):
    out, = _prim.bind(
        *args,
        **kwargs,
        _func=_func,
        _func_T=_func_T,
        _func_abstract=_func_abstract,
        _func_abstract_T=_func_abstract_T
    )
    return out


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
    # TODO: register all func* in global scope such that the user does not need
    # keep a reference. Ideally this reference is cheap but just to be sure,
    # also implemenet a clear cache function
    return partial(
        _call, _func=func, _func_T=func_T, _func_abstract=func_abstract, _func_abstract_T=func_abstract_T
    )
