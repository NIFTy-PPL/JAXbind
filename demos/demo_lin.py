#%%
import jax
from functools import partial
import numpy as np
import scipy.fft
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import pickle


def lin(out, args, kwargs_dump):

    # NOTE, this might look inefficient but for most practical problems it
    # really is not:
    # ```
    # kwargs_dump = np.frombuffer(pickle.dumps({"asdas": (1, 2, 3,)}), dtype=np.uint8)
    # %timeit pickle.loads(np.ndarray.tobytes(kwargs_dump))
    # # 582 ns ± 1.12 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    # ```
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    # print(kwargs)
    out[0][()] = args[0] + args[1]
    out[1][()] = args[0] + args[1]


def lin_T(out, args, kwargs_dump):
    kwargs = pickle.loads(np.ndarray.tobytes(kwargs_dump))
    out[0][()] = args[0] + args[1]
    out[1][()] = args[0] + args[1]

def lin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, out_axes)
    , (args[0].shape, args[0].dtype, out_axes))

def lin_abstract_T(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    out_axes = kwargs.pop("batch_axes", ())
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, out_axes), (args[0].shape, args[0].dtype, out_axes))


lin_jax = jax_linop.get_linear_call(
    lin, lin_T, lin_abstract, lin_abstract_T, None, None, 'lin', arg_fixed=(False, False), func_can_batch=True
)
inp = (4 + jnp.zeros((2, 2)), 1 + jnp.zeros((2, 2)))
lin_jax(*inp, axes=(3, 4))

check_grads(partial(lin_jax, axes=(3, 4)), inp, order=2, modes=["fwd"], eps=1.)
check_grads(partial(lin_jax, axes=(3, 4)), inp, order=2, modes=["rev"], eps=1.)


# %%
