# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society

import pickle
from collections import namedtuple
from functools import partial
from typing import Union

import jax
import jaxlib.mlir.dialects.stablehlo as hlo
import jaxlib.mlir.ir as ir
import numpy as np
from jax.interpreters import ad, batching, mlir
from jax.interpreters.mlir import ir_constant as irc
from jaxlib.hlo_helpers import custom_call


__all__ = ['get_linear_call', 'get_nonlinear_call']

import _jaxbind

for _name, _value in _jaxbind.registrations().items():
    jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")


# Hack to avoid classes and having to register a PyTree
_shared_args_names = (
    "abstract",
    "abstract_T",
    "first_n_args_fixed",
    "can_batch",
    "batch_axes",
)
LinearFunction = namedtuple("LinearFunction", ("f", "T") + _shared_args_names)
MultiLinearFunction = namedtuple("MultiLinearFunction", ("f", "T") + _shared_args_names)
NonLinearFunction = namedtuple(
    "NonLinearFunction", ("f", "derivatives") + _shared_args_names
)

FunctionType = Union[LinearFunction, MultiLinearFunction, NonLinearFunction]


def _exec_abstract(*args, _func: FunctionType, **kwargs):
    if _func.can_batch:
        assert "batch_axes" not in kwargs
        kwargs["batch_axes"] = _func.batch_axes
    ae = _func.abstract(*args, **kwargs)
    # NOTE, do not attempt to unpack the batch axes b/c it might not be there
    return tuple(jax.core.ShapedArray(*sdb[:2]) for sdb in ae)


# the values are explained in src/ducc0/bindings/typecode.h
_dtype_dict = {
    np.dtype(np.float32): 3,
    np.dtype(np.float64): 7,
    np.dtype(np.uint8): 32,
    np.dtype(np.uint64): 39,
    np.dtype(np.complex64): 67,
    np.dtype(np.complex128): 71,
}


def _lowering(ctx, *args, _func: FunctionType, _platform="cpu", **kwargs):
    operands = [irc(id(_func.f))]  # Pass the ID of the callable to C++
    operand_layouts = [()]

    operands += [irc(len(args))]
    operand_layouts += [()]
    assert len(args) == len(ctx.avals_in)
    # All `args` are assumed to be JAX arrays
    for a, ca in zip(args, ctx.avals_in):
        operands += (
            [irc(_dtype_dict[ca.dtype]), irc(ca.ndim)]
            + [irc(i) for i in ca.shape]
            + [a]
        )
        lyt_a = tuple(range(ca.ndim - 1, -1, -1))
        operand_layouts += [()] * (2 + ca.ndim) + [lyt_a]

    operands += [irc(len(ctx.avals_out))]
    operand_layouts += [()]
    result_layouts = []
    result_types = []
    for co in ctx.avals_out:
        operands += [irc(_dtype_dict[co.dtype]), irc(co.ndim)] + [
            irc(i) for i in co.shape
        ]
        operand_layouts += [()] * (2 + co.ndim)
        result_layouts += [tuple(range(co.ndim - 1, -1, -1))]
        rs_typ = mlir.ir.RankedTensorType.get(co.shape, mlir.dtype_to_ir_type(co.dtype))
        result_types += [rs_typ]

    if _func.can_batch:
        assert "batch_axes" not in kwargs
        kwargs["batch_axes"] = _func.batch_axes
    kwargs = np.frombuffer(pickle.dumps(kwargs), dtype=np.uint8)
    kwargs_ir = hlo.constant(
        ir.DenseElementsAttr.get(kwargs, type=ir.IntegerType.get_unsigned(8))
    )
    operands += [irc(_dtype_dict[kwargs.dtype]), irc(kwargs.size), kwargs_ir]
    operand_layouts += [(), (), [0]]

    assert len(operand_layouts) == len(operands)
    if _platform == "cpu":
        return custom_call(
            _platform + "_pycall",
            result_types=result_types,
            result_layouts=result_layouts,
            operands=operands,
            operand_layouts=operand_layouts,
        ).results
    elif _platform == "gpu":
        raise ValueError("No GPU support")
    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


def _explicify_zeros(x):
    if isinstance(x, (tuple, list)):
        return [ad.instantiate_zeros(t) if isinstance(t, ad.Zero) else t for t in x]
    return ad.instantiate_zeros(x) if isinstance(x, ad.Zero) else x


def _jvp(args, tangents, *, _func: FunctionType, **kwargs):
    res = _prim.bind(*args, **kwargs, _func=_func)

    def zero_tans(tans):
        if isinstance(tans, (tuple, list)):
            return [isinstance(t, ad.Zero) for t in tans]
        return [isinstance(tans, ad.Zero)]

    n_args = len(args)
    n_f_args = _func.first_n_args_fixed
    assert n_args > n_f_args
    args_fixed = n_f_args * (True,) + (n_args - n_f_args) * (False,)
    assert len(args) == len(tangents) == len(args_fixed)

    tan_is_zero = zero_tans(tangents)
    assert len(args) == len(tan_is_zero)

    for i, (a, t) in enumerate(zip(args_fixed, tan_is_zero)):
        if a and not t:
            raise RuntimeError(f"{i}th positional argument not differentiable")

    if all(type(t) is ad.Zero for t in tangents):
        tans = list((jax.lax.zeros_like_array(a) for a in res))
        return (res, tans)

    tans = None
    if isinstance(_func, MultiLinearFunction):
        for i, t in enumerate(tangents):
            if not args_fixed[i]:
                t = _explicify_zeros(t)
                tn = _prim.bind(*args[:i], t, *args[i + 1 :], **kwargs, _func=_func)
                tans = (
                    tn if tans is None else tuple(t + tn_i for t, tn_i in zip(tans, tn))
                )
    elif isinstance(_func, LinearFunction):
        inp = []
        for a, f, t in zip(args, args_fixed, tangents):
            inp.append(a if f else _explicify_zeros(t))
        tans = _prim.bind(*inp, **kwargs, _func=_func)
    elif isinstance(_func, NonLinearFunction):
        f, f_T = _func.derivatives
        tan_in = [t for f, t in zip(args_fixed, tangents) if not f]
        tan_in = _explicify_zeros(tan_in)
        _func = LinearFunction(
            f=f,
            T=f_T,
            abstract=_func.abstract,
            abstract_T=_func.abstract_T,
            first_n_args_fixed=len(args),
            can_batch=_func.can_batch,
            batch_axes=_func.batch_axes,
        )
        tans = _prim.bind(*args, *tan_in, **kwargs, _func=_func)
    else:
        raise TypeError(f"JVP for {type(_func)} not implemented")

    assert tans is not None
    return (res, tans)


# NOTE: for whatever reason JAX will pass each arg separately to _transpose
# and not as a tuple as for _jvp. Thus we need *args since we don't know
# the number of arguments.
def _transpose(cotangents, *args, _func: FunctionType, **kwargs):
    assert isinstance(_func, (LinearFunction, MultiLinearFunction))
    if _func.T is None:
        raise NotImplementedError(f"transpose of {_func} not implemented")
    n_args = len(args)
    n_f_args = _func.first_n_args_fixed
    assert n_args > n_f_args
    args_fixed = n_f_args * (True,) + (n_args - n_f_args) * (False,)
    arg_is_lin = [ad.is_undefined_primal(a) for a in args]
    assert len(args_fixed) >= len(arg_is_lin)

    for i, (a, is_lin) in enumerate(zip(args_fixed, arg_is_lin)):
        if a and is_lin:
            raise RuntimeError(
                f"Cannot transpose with respect to positional argument number {i}"
            )

    if isinstance(_func, MultiLinearFunction):
        assert sum(arg_is_lin) == 1
        lin_arg = arg_is_lin.index(True)
        c_in = cotangents
        a_in = args[:lin_arg] + args[lin_arg + 1 :]

        assert isinstance(_func.T, tuple)
        assert isinstance(_func.abstract_T, tuple)
        # TODO(edh): I think this should be a LinearFunction
        _func = _func._replace(
            f=_func.T[lin_arg],
            T=None,
            abstract=_func.abstract_T[lin_arg],
            abstract_T=None,
        )
        res = _prim.bind(*a_in, *c_in, **kwargs, _func=_func)
        res = [None] * lin_arg + res + [None] * (len(arg_is_lin) - (lin_arg + 1))
    elif isinstance(_func, LinearFunction):
        inp = []
        for a, f in zip(args, args_fixed):
            if f:
                assert not ad.is_undefined_primal(a)
                inp.append(a)
        cot = _explicify_zeros(cotangents)
        _func = _func._replace(
            f=_func.T,
            T=_func.f,
            abstract=_func.abstract_T,
            abstract_T=_func.abstract,
            first_n_args_fixed=len(inp),
        )
        res = _prim.bind(*inp, *cot, **kwargs, _func=_func)
        res = n_f_args * [None] + res
    else:
        raise TypeError(f"transpose for {type(_func)} not implemented")
    return res


def _batch(args, in_axes, *, _func: FunctionType, **kwargs):
    from .custom_map import smap

    if not _func.can_batch:
        y = smap(partial(_prim.bind, _func=_func), in_axes=in_axes)(*args)
        out_axes = [0] * len(y)
    else:
        batch_axes = _func.batch_axes
        batch_axes = ((),) * len(in_axes) if batch_axes is None else batch_axes
        new_batch_axes = []
        assert len(in_axes) == len(batch_axes)
        for ia, baxes in zip(in_axes, batch_axes):
            baxes_new = []
            if ia is not None:
                assert isinstance(ia, int)
                for b in baxes:
                    if b >= ia:
                        b += 1
                    baxes_new.append(b)
                baxes = tuple(tuple(baxes_new) + (ia,))
            new_batch_axes.append(baxes)
        new_batch_axes = tuple(new_batch_axes)
        _func = _func._replace(batch_axes=new_batch_axes)

        args_w = [jax.ShapeDtypeStruct(el.shape, el.dtype) for el in args]
        out_w = _func.abstract(*args_w, batch_axes=new_batch_axes, **kwargs)
        out_axes = [ba_wb for _, _, ba_wb in out_w]
        y = _call(*args, _func=_func, **kwargs)
    return y, out_axes


_prim = jax.core.Primitive("jaxbind_prim")
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


def _call(*args, _func: FunctionType, **kwargs):
    return _prim.bind(*args, **kwargs, _func=_func)


def get_linear_call(
    f,
    f_T,
    /,
    abstract,
    abstract_T,
    *,
    first_n_args_fixed=0,
    func_can_batch=False,
) -> partial:
    """Create a JAX primitive for the provided linear function

    Parameters
    ----------
    f, f_T : linear function respectively its transpose
        The function signature must be (out, args, kwargs_dump), where
        out and args are tuples. The results of the functions should be written as
        numpy.ndarrays of float[32/64] or complex[64/128] type into the out tuple.
        The args tuple contains the input for the function. In kwargs_dump,
        potential keyword arguments are contained in serialized form. The
        keyword arguments can be deserialized via
        `jaxbind.load_kwargs(kwargs_dump)`.
    abstract, abstract_T : function respectively its transpose
        Computing the shape and dtype of the operator's output from shape and
        dtype of its input. Its signature must be `(*args, **kwargs)`. `args`
        will be a tuple containing abstract tracer arrays with shape and dtype
        for each input argument of `f` respectively `f_T`. Via `**kwargs`,
        potential keyword arguments are passed to the function. The function
        must return a tuple containing tuples of (shape_out, dtype_out) for each
        output argument of `f` respectively `f_T`.
    first_n_args_fixed : int
        If the function cannot be differentiated with respect to some of the
        arguments, these can be passed as the first arguments to the function.
        fist_n_args_fixed indicates the number of non-differential arguments.
        Note: The function does not need to be linear with respect to these
        arguments. Default 0 (all arguments are differentiable).
    func_can_batch : bool
        Indicator whether the function natively supports batching. If true, the
        function will receive one additional argument called `batch_axes`. The
        parameter will be a tuple of tuples, or None if no batching is currently
        performed. The tuple will be of length of the input and for each input
        will contain a tuple of integer indices along which the computation
        shall be batched.

    Returns
    -------
    op : Jax primitive corresponding to the function f.

    Notes
    -----
    - `f` and 'f_T' must not return anything; the result of the computation must be
      written into the member arrays of `out`.
    - the contents of `args` must not be modified.
    - no reference to the contents of `args` or `out` may be stored beyond
      the execution time of `f` or `f_T`.
    """
    kw = dict(
        f=f,
        T=f_T,
        abstract=abstract,
        abstract_T=abstract_T,
        first_n_args_fixed=first_n_args_fixed,
        batch_axes=None,
        can_batch=func_can_batch,
    )
    if isinstance(f_T, (tuple, list)):
        _func = MultiLinearFunction(**kw)
    else:
        _func = LinearFunction(**kw)
    return partial(_call, _func=_func)


def get_nonlinear_call(
    f,
    f_derivative,
    /,
    abstract,
    abstract_reverse,
    *,
    first_n_args_fixed=0,
    func_can_batch=False,
) -> partial:
    """Create a JAX primitive for the provided (nonlinear) function

    Parameters
    ----------
    f : function
        The function signature must be (out, args, kwargs_dump), where
        out and args are tuples. The results of the functions should be written as
        numpy.ndarrays of float[32/64] or complex[64/128] type into the out tuple.
        The args tuple contains the input for the function. In kwargs_dump,
        potential keyword arguments are contained in serialized form. The
        keyword arguments can be deserialized via
        `jaxbind.load_kwargs(kwargs_dump)`.
    f_derivative: tuple of functions
        Tuple containing functions for evaluating jvp and vjp of f. The fist entry
        in the function should evaluate jvp, the second vjp. The signature of the
        jvp and vjp functions should be (out, args, kwargs_dump) analogous to f.
    abstract, abstract_reverse : functions
        Computing the shape and dtype of the operator's output from shape and
        dtype of its input. Its signature must be (*args, **kwargs). *args will
        be a tuple containing abstract tracer arrays with shape and dtype for
        each input argument of f. Via **kwargs, potential keyword arguments are
        passed to the function. The function must return the tuple containing a
        tuples of (shape_out, dtype_out). abstract should compute the output
        shapes of f and jvp. abstract_reverse should compute the output shape of
        vjp.
    first_n_args_fixed : int
        If the function cannot be differentiated with respect to some of the
        arguments, these can be passed as the first arguments to the function.
        fist_n_args_fixed indicates the number of non-differential arguments.
        Default 0 (all arguments are differentiable).
    func_can_batch : bool
        Indicator whether the function natively supports batching. If true, the
        function will receive one additional argument called `batch_axes`. The
        parameter will be a tuple of tuples, or None if no batching is currently
        performed. The tuple will be of length of the input and for each input
        will contain a tuple of integer indices along which the computation
        shall be batched.

    Returns
    -------
    op : Jax primitive corresponding to the function f.

    Notes
    -----
    - `f` and members of 'f_derivative' must not return anything; the result
      of the computation must be written into the member arrays `out`.
    - the contents of `args` must not be modified.
    - no references to the contents of `args` or `out` may be stored beyond
      the execution time of `f` or `f_derivative`.
    """
    _func = NonLinearFunction(
        f=f,
        abstract=abstract,
        abstract_T=abstract_reverse,
        first_n_args_fixed=first_n_args_fixed,
        batch_axes=None,
        can_batch=func_can_batch,
        derivatives=f_derivative,
    )
    return partial(_call, _func=_func)
