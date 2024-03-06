from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.test_util import check_grads
import pytest

pmp = pytest.mark.parametrize


import jax_linop

jax.config.update("jax_enable_x64", True)


def f(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y = args
    out[0][()] = x * y
    out[1][()] = y * y


def f_jvp(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, dx, dy = args
    out[0][()] = y * dx + x * dy
    out[1][()] = 2 * y * dy


def f_vjp(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, da, db = args
    out[0][()] = y * da
    out[1][()] = x * da + 2 * y * db

def f_jvp_fix_x(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, dy = args
    out[0][()] = x * dy
    out[1][()] = 2 * y * dy


def f_vjp_fix_x(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    x, y, da, db = args
    out[0][()] = x * da + 2 * y * db


def f_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype, out_axes),
        (args[0].shape, args[0].dtype, out_axes),
    )

def f_abstract_vjp_fix_x(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, out_axes),)


@pmp("first_n_args_fixed", (0, 1))
@pmp("dtype", (np.complex64, np.complex128))
@pmp("shape", ((2,), (3,4)))
def test_derivatives(first_n_args_fixed, dtype, shape):
    if first_n_args_fixed == 0:
        funcs_deriv = (f_jvp, f_vjp)
        absr_T = f_abstract
    else:
        funcs_deriv = (f_jvp_fix_x, f_vjp_fix_x)
        absr_T = f_abstract_vjp_fix_x

    f_jax = jax_linop.get_nonlinear_call(
        f,
        funcs_deriv,
        f_abstract,
        absr_T,
        first_n_args_fixed=first_n_args_fixed,
    )

    inp1 = jnp.full(shape, 3., dtype=dtype)
    inp2 = jnp.full(shape, 5., dtype=dtype)

    if first_n_args_fixed == 0:
        check_grads(
            f_jax, (inp1,inp2), order=1, modes=["fwd", "rev"], eps=1e-3
        )
    else:
        f_jax_pt = partial(f_jax, inp1)
        check_grads(
            f_jax_pt, (inp2,), order=1, modes=["fwd", "rev"], eps=1e-3
        )


