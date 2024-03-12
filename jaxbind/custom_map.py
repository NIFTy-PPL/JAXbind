# Copyright(C) 2022-2024 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial

import jax
from jax import lax
from jax import numpy as jnp


def _int_or_none(x):
    return isinstance(x, int) or x is None


def _fun_reord(_, mapped, *, fun, unmapped, unflatten, in_axes):
    un, mapped = list(unmapped), list(mapped)
    assert len(un) + len(mapped) == len(in_axes)
    args = tuple(un.pop(0) if a is None else mapped.pop(0) for a in in_axes)
    y = fun(*unflatten(args))
    return None, y


def _moveaxis(a, source, destination):
    # Ensure that arrays are never completely unnecessarily copied
    if source == destination:
        return a
    return jnp.moveaxis(a, source, destination)


def _generic_smap(fun, in_axes, out_axes, unroll, *x, _scan=lax.scan, **k):
    from jax.tree_util import tree_flatten, tree_map, tree_unflatten

    if k:
        raise TypeError("keyword arguments are not allowed in map")

    if isinstance(in_axes, int):
        in_axes = tree_map(lambda _: in_axes, x)
    elif isinstance(in_axes, tuple):
        if len(in_axes) != len(x):
            ve = f"`in_axes` {in_axes!r} and input {x!r} must be of same length"
            raise ValueError(ve)
        new_inax = []
        for el, i in zip(x, in_axes):
            new_inax.append(tree_map(lambda _: i, el) if _int_or_none(i) else i)
        in_axes = tuple(new_inax)
    else:
        te = f"`in_axes` must be an int or tuple of pytrees/int; got {in_axes!r}"
        raise TypeError(te)
    x, x_td = tree_flatten(x)
    in_axes, in_axes_td = tree_flatten(in_axes, is_leaf=_int_or_none)
    if in_axes_td != x_td:
        ve = f"`in_axes` {in_axes_td!r} incompatible with input `*x` {x_td!r}"
        raise ValueError(ve)

    unmapped = []
    mapped = []
    for i, el in zip(in_axes, x):
        if i is None:
            unmapped.append(el)
        elif isinstance(i, int):
            mapped.append(_moveaxis(el, i, 0))
        else:
            raise TypeError(f"expected `in_axes` index of type int; got {i!r}")

    fun_reord = partial(
        _fun_reord,
        fun=fun,
        unmapped=unmapped,
        unflatten=partial(tree_unflatten, x_td),
        in_axes=in_axes,
    )
    _, y = _scan(fun_reord, None, mapped, unroll=unroll)

    if out_axes is None:
        out_axes, out_axes_td = tree_flatten(out_axes)
    if isinstance(out_axes, int):
        out_axes = tree_map(lambda el: out_axes if el is not None else el, y)
        out_axes, out_axes_td = tree_flatten(out_axes)
    else:
        out_axes, out_axes_td = tree_flatten(out_axes, is_leaf=_int_or_none)
    y, y_td = tree_flatten(y)
    if y is not None and out_axes_td != y_td:
        ve = f"`out_axes` {out_axes_td!r} incompatible with output {y_td!r}"
        raise ValueError(ve)
    out = []
    for i, el in zip(out_axes, y):
        if i is None:
            out.append(unmapped.pop(0))
        elif isinstance(i, int):
            out.append(_moveaxis(el, 0, i))
        else:
            raise TypeError(f"expected `out_axes` index of type int; got {i!r}")

    return tree_unflatten(y_td, out)


# The function over which to `scan` depends on the data. This leads to
# unnecessary recompiles. Ensure scan is compiled only once by compiling the
# whole data dependence.
_smap = jax.jit(
    _generic_smap, static_argnames=("fun", "in_axes", "out_axes", "unroll", "_scan")
)


def smap(fun, in_axes=0, out_axes=0, *, unroll=1):
    """Stupid/sequential map.

    Many of JAX's control flow logic reduces to a simple `jax.lax.scan`. This
    function is one of these. In contrast to `jax.lax.map` or
    `jax.lax.fori_loop`, it behaves much like `jax.vmap`. In fact, it
    re-implements `in_axes` and `out_axes` and can be used in much the same way
    as `jax.vmap`. However, instead of batching the input, it works through it
    sequentially.

    This implementation makes no claim on being efficient. It explicitly swaps
    around axis in the input and output, potentially allocating more memory
    than strictly necessary and worsening the memory layout.

    For the semantics of `in_axes` and `out_axes` see `jax.vmap`. For the
    semantics of `unroll` see `jax.lax.scan`.
    """
    return partial(_smap, fun, in_axes, out_axes, unroll)
