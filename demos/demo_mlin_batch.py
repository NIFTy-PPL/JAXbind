# %%
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.test_util import check_grads

import jaxbind

jax.config.update("jax_enable_x64", True)


def mlin_call(x, y):
    return x * y, x * y


def mlin(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump)
    batch_axes = kwargs.pop("batch_axes", None)
    call = mlin_call
    if batch_axes is not None and batch_axes != ((),) * len(args):
        # Allow vmapping exactly once
        assert all(len(ba) in (0, 1) for ba in batch_axes)
        in_axes = tuple((ba[0] if len(ba) == 1 else None) for ba in batch_axes)
        call = jax.vmap(mlin_call, in_axes=in_axes)
    o = call(*args)
    out[0][()] = o[0]
    out[1][()] = o[1]


def mlin_T1(out, args, kwargs_dump):
    out[0][()] = args[0] * args[1] + args[0] * args[2]


def mlin_T2(out, args, kwargs_dump):
    out[0][()] = args[0] * args[1] + args[0] * args[2]


def mlin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    batch_axes = kwargs.pop("batch_axes", None)
    call = mlin_call
    if batch_axes is not None and batch_axes != ((),) * len(args):
        # Allow vmapping exactly once
        assert all(len(ba) in (0, 1) for ba in batch_axes)
        in_axes = tuple((ba[0] if len(ba) == 1 else None) for ba in batch_axes)
        call = jax.vmap(mlin_call, in_axes=in_axes)
    out_axes = [0 for ba in batch_axes]
    out = jax.eval_shape(call, *args)
    return tuple((o.shape, o.dtype, oa) for o, oa in zip(out, out_axes))


def mlin_abstract_T1(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, None),)


def mlin_abstract_T2(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, None),)


func_T = (mlin_T1, mlin_T2)
func_abstract_T = (mlin_abstract_T1, mlin_abstract_T2)
mlin_jax = jaxbind.get_linear_call(
    mlin,
    func_T,
    mlin_abstract,
    func_abstract_T,
    func_can_batch=True,
)

inp = (4 + jnp.zeros((2,)), 1 + jnp.zeros((2,)))


in_ax = []
in_ax += ((None, 0),)
in_ax += ((0, 0),)
in_ax += ((0, None),)

for ia in in_ax:
    vm = jax.vmap(mlin_jax, in_axes=ia)
    vmj = jax.vmap(mlin_call, in_axes=ia)
    r = vm(*inp)
    rj = vmj(*inp)

    assert r[0].shape == rj[0].shape
    assert r[1].shape == rj[1].shape
    np.testing.assert_allclose(r, rj)
