# %%
import jax
from functools import partial
import numpy as np
import scipy.fft
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import pickle


# (x,y) -> (xy, y**2)
def nonlin(out, args, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    x, y = args
    out[0][()] = x * y
    out[1][()] = y * y


# (x,y,dx,dy) -> (ydx + xdy, 2 * y * dy)
def nonlin_deriv(out, args, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    x, y, dx, dy = args
    out[0][()] = y * dx + x * dy
    out[1][()] = 2 * y * dy



# (x, y, da, db) -> (yda, xda + 2ydb)
def nonlin_deriv_T(out, args, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    print(f"{args=}")
    x, y, da, db = args
    out[0][()] = y * da
    out[1][()] = x * da + 2 * y * db


def nonlin_dif2(out, args, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    pass


def nonlin_T1(out, args, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    out[0][()] = args[0] * args[1] + args[0] * args[2]


def nonlin_T2(out, args, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    out[0][()] = args[0] * args[1] + args[0] * args[2]


def nonlin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype, out_axes),
        (args[0].shape, args[0].dtype, out_axes),
    )

# TODO: maybe give the user the possibility to provide more functions, such that
# more transforms can be computed
funcs = None

funcs_deriv = (nonlin_deriv, nonlin_deriv_T)

nonlin_jax = jax_linop.get_linear_call(
    nonlin,
    None,
    nonlin_abstract,
    nonlin_abstract, # FIXME
    funcs,
    funcs_deriv,
    "nonlin",
    arg_fixed=(False, False),
    func_can_batch=True,
)


# inp = (4 + jnp.zeros((2, 2)), 1 + jnp.zeros((2, 2)))
inp = (4 + jnp.zeros((1,)), 1 + jnp.zeros((1,)))

check_grads(
    partial(nonlin_jax, axes=(3, 4)), inp, order=1, modes=["fwd", "rev"], eps=1e-3
)
