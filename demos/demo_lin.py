# %%
from functools import partial

import jax
from jax import numpy as jnp
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)


def lin(out, args, kwargs_dump):
    # NOTE, this might look inefficient but for most practical problems it
    # really is not:
    # ```
    # kwargs_dump = np.frombuffer(pickle.dumps({"asdas": (1, 2, 3,)}), dtype=np.uint8)
    # %timeit pickle.loads(np.ndarray.tobytes(kwargs_dump))
    # # 582 ns ± 1.12 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    # ```
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    out[0][()] = args[0] + args[1]
    out[1][()] = args[0] + args[1]


def lin_T(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    out[0][()] = args[0] + args[1]
    out[1][()] = args[0] + args[1]


def lin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype, None),
        (args[0].shape, args[0].dtype, None),
    )


def lin_abstract_T(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype, None),
        (args[0].shape, args[0].dtype, None),
    )


lin_jax = jax_linop.get_linear_call(
    lin, lin_T, lin_abstract, lin_abstract_T, args_fixed=(False, False)
)
inp = (4 + jnp.zeros((2, 2)), 1 + jnp.zeros((2, 2)))
lin_jax(*inp, axes=(3, 4))

check_grads(partial(lin_jax, axes=(3, 4)), inp, order=2, modes=["fwd"], eps=1.0)
check_grads(partial(lin_jax, axes=(3, 4)), inp, order=2, modes=["rev"], eps=1.0)


##################################### check fixing args #######################
def lin(out, args, kwargs_dump):
    out[0][()] = args[0] * args[1] * args[0]
    out[1][()] = args[0] * args[1]


def lin_T(out, args, kwargs_dump):
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    out[0][()] = args[0] * args[0] * args[1] + args[0] * args[2]


def lin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype, None),
        (args[0].shape, args[0].dtype, None),
    )


def lin_abstract_T(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype, None),)


lin_jax = jax_linop.get_linear_call(
    lin,
    lin_T,
    lin_abstract,
    lin_abstract_T,
    args_fixed=(True, False),
    func_can_batch=True,
)

inp1 = 4 + jnp.zeros((2, 2))
inp2 = 1 + jnp.zeros((2, 2))
lin_jax_pt = partial(lin_jax, inp1, axes=(3, 4))

check_grads(lin_jax_pt, (inp2,), order=2, modes=["fwd", "rev"], eps=1.0)
