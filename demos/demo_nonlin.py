# %%
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)


# (x,y) -> (xy, y**2)
def nonlin(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y = args
    out[0][()] = x * y
    out[1][()] = y * y


# (x,y,dx,dy) -> (ydx + xdy, 2 * y * dy)
def nonlin_deriv(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, dx, dy = args
    out[0][()] = y * dx + x * dy
    out[1][()] = 2 * y * dy


# (x, y, da, db) -> (yda, xda + 2ydb)
def nonlin_deriv_T(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, da, db = args
    out[0][()] = y * da
    out[1][()] = x * da + 2 * y * db


def nonlin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype, out_axes),
        (args[0].shape, args[0].dtype, out_axes),
    )


funcs_deriv = (nonlin_deriv, nonlin_deriv_T)
nonlin_jax = jax_linop.get_nonlinear_call(
    nonlin,
    funcs_deriv,
    nonlin_abstract,
    nonlin_abstract,  # FIXME
    first_n_args_fixed=0,
    func_can_batch=True,
)


# inp = (4 + jnp.zeros((2, 2)), 1 + jnp.zeros((2, 2)))
inp = (4 + jnp.zeros((1,)), 1 + jnp.zeros((1,)))

check_grads(
    partial(nonlin_jax, axes=(3, 4)), inp, order=1, modes=["fwd", "rev"], eps=1e-3
)


################################################## test non diff args
# (x,y) -> (xy, y**2)
def nonlin(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y = args
    out[0][()] = x * y
    out[1][()] = y * y


# (x,y,dy) -> (ydx + xdy, 2 * y * dy)
def nonlin_deriv(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, dy = args
    out[0][()] = x * dy
    out[1][()] = 2 * y * dy


# (x, y, da, db) -> (yda, xda + 2ydb)
def nonlin_deriv_T(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, da, db = args
    out[0][()] = x * da + 2 * y * db


def nonlin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype, out_axes),
        (args[0].shape, args[0].dtype, out_axes),
    )


def nonlin_abstract_T(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, out_axes),)


funcs_deriv = (nonlin_deriv, nonlin_deriv_T)
nonlin_jax = jax_linop.get_nonlinear_call(
    nonlin,
    funcs_deriv,
    nonlin_abstract,
    nonlin_abstract_T,  # FIXME
    first_n_args_fixed=1,
    func_can_batch=True,
)

inp1 = 4 + jnp.zeros((1,))
inp2 = 6 + jnp.zeros((1,))

nonlin_jax_pt = partial(nonlin_jax, inp1, axes=(3, 4))


check_grads(
    partial(nonlin_jax_pt, axes=(3, 4)),
    (inp2,),
    order=1,
    modes=["fwd", "rev"],
    eps=1e-3,
)
