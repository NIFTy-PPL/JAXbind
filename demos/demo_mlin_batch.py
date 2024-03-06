# %%
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)


def mlin_call(x, y):
    return x * y, x * y


def mlin(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    batch_axes = kwargs.pop("batch_axes", ())
    call = mlin_call
    if batch_axes != () and batch_axes != ((),) * len(args):
        assert all(
            len(ba) in (0, 1) for ba in batch_axes
        )  # Allow vmapping exactly once
        call = jax.vmap(
            mlin_call,
            in_axes=tuple((ba[0] if len(ba) == 1 else None) for ba in batch_axes),
        )
    o = call(*args)
    out[0][()], out[1][()] = o


def mlin_T1(out, args, kwargs_dump):
    out[0][()] = args[0] * args[1] + args[0] * args[2]


def mlin_T2(out, args, kwargs_dump):
    out[0][()] = args[0] * args[1] + args[0] * args[2]


def mlin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    batch_axes = kwargs.pop("batch_axes", ())
    call = mlin_call
    if batch_axes != () and batch_axes != ((),) * len(args):
        assert all(
            len(ba) in (0, 1) for ba in batch_axes
        )  # Allow vmapping exactly once
        call = jax.vmap(
            mlin_call,
            in_axes=tuple((ba[0] if len(ba) == 1 else None) for ba in batch_axes),
        )
    out = jax.eval_shape(call, *args)
    return tuple((o.shape, o.dtype, 0) for o in out)


def mlin_abstract_T1(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, None),)


def mlin_abstract_T2(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, None),)


func_T = (mlin_T1, mlin_T2)
func_abstract_T = (mlin_abstract_T1, mlin_abstract_T2)
mlin_jax = jax_linop.get_linear_call(
    mlin,
    func_T,
    mlin_abstract,
    func_abstract_T,
    func_can_batch=True,
)


inp = (4 + jnp.zeros((2,)), 1 + jnp.zeros((2,)))\

vm = jax.vmap(mlin_jax, in_axes=(0, 0))
vmj = jax.vmap(mlin_call, in_axes=(0, 0))

r1 = vm(*inp)
r2 = vmj(*inp)

assert(r1[0].shape == r2[0].shape)