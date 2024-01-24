import jax
from functools import partial
import numpy as np
import scipy.fft
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import pickle


def fft(args, out, kwargs_dump):
    print(kwargs_dump)
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    print(kwargs)
    out[()] = args[0] * args[1]


def fft_T(args, out, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    out[()] = args[0] + args[1]


def fft_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    return args[0].shape, args[0].dtype, out_axes


fft_jl = jax_linop.get_linear_call(
    fft, fft_T, fft_abstract, fft_abstract, func_can_batch=True
)
inp = (4 + jnp.zeros((2, 2, 14)), 1 + jnp.zeros((2, 2, 14)))
fft_jl(*inp, axes=(3, 4))

check_grads(partial(fft_jl, axes=(3, 4)), inp, order=2, modes=["fwd"], eps=1.)
# fft_jl_j = jax.jit(jax.vmap(jax.vmap(fft_jl, in_axes=0), in_axes=0))
# fft_jl_j = jax.vmap(jax.vmap(fft_jl, in_axes=0), in_axes=0)
# fft_jl_j(jnp.zeros((2, 2, 14)).astype(jnp.complex128))
