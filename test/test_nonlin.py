from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.test_util import check_grads
from jax import random
import pytest

pmp = pytest.mark.parametrize


import jax_linop

jax.config.update("jax_enable_x64", True)


precision_dict = {
    np.dtype(np.complex64): np.dtype(np.float32),
    np.dtype(np.complex128): np.dtype(np.float64),
    np.dtype(np.float32): np.dtype(np.float32),
    np.dtype(np.float64): np.dtype(np.float64),
}


def real_type_same_precision(dtype):
    return precision_dict[np.dtype(dtype)]


def is_complex_type(dtype):
    if dtype is np.float64 or dtype is np.float32:
        return False
    else:
        return True


def get_batched_func(call, kwargs_dump):
    call_new = call
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    batch_axes = kwargs.pop("batch_axes", ())
    if batch_axes != () and batch_axes != ((),) * len(args):
        assert all(
            len(ba) in (0, 1) for ba in batch_axes
        )  # Allow vmapping exactly once
        call_new = jax.vmap(
            call,
            in_axes=tuple((ba[0] if len(ba) == 1 else None) for ba in batch_axes),
        )
    return call_new


def f_call(x, y):
    return x * y, y * y


def f(out, args, kwargs_dump, batch_version=False):
    if batch_version:
        call = get_batched_func(f_call, kwargs_dump)
    else:
        call = f_call
    out[0][()], out[1][()] = call(*args)


def f_jvp_call(x, y, dx, dy):
    return y * dx + x * dy, 2 * y * dy


def f_jvp(out, args, kwargs_dump, batch_version=False):
    if batch_version:
        call = get_batched_func(f_jvp_call, kwargs_dump)
    else:
        call = f_jvp_call
    out[0][()], out[1][()] = call(*args)


def f_vjp_call(x, y, da, db):
    return y * da, x * da + 2 * y * db


def f_vjp(out, args, kwargs_dump, batch_version=False):
    if batch_version:
        call = get_batched_func(f_vjp_call, kwargs_dump)
    else:
        call = f_vjp_call
    out[0][()], out[1][()] = call(*args)


def f_jvp_fix_x_call(x, y, dy):
    return x * dy, 2 * y * dy


def f_jvp_fix_x(out, args, kwargs_dump, batch_version=False):
    if batch_version:
        call = get_batched_func(f_jvp_fix_x_call, kwargs_dump)
    else:
        call = f_jvp_fix_x_call
    out[0][()], out[1][()] = call(*args)


def f_vjp_fix_x_call(x, y, da, db):
    return x * da + 2 * y * db


def f_vjp_fix_x(out, args, kwargs_dump, batch_version=False):
    if batch_version:
        call = get_batched_func(f_vjp_fix_x_call, kwargs_dump)
    else:
        call = f_vjp_fix_x_call
    out[0][()] = call(*args)


def f_abstract(*args, **kwargs):
    out = jax.eval_shape(f_call, *args[0:2])
    return tuple((o.shape, o.dtype, 0) for o in out)


def f_abstract_vjp_fix_x(*args, **kwargs):
    out = jax.eval_shape(f_vjp_fix_x_call, *args[0:4])
    return ((out.shape, out.dtype, ()),)


@pmp("first_n_args_fixed", (0, 1))
@pmp("can_batch", (True, False))
@pmp("dtype", (np.float32, np.float64, np.complex64, np.complex128))
@pmp("shape", ((2,), (3, 4)))
@pmp("seed", (42,))
@pmp("jit", (False, True))
def test_derivatives(first_n_args_fixed, can_batch, dtype, shape, seed, jit):
    if first_n_args_fixed == 0:
        funcs_deriv = (
            partial(f_jvp, batch_version=can_batch),
            partial(f_vjp, batch_version=can_batch),
        )
        absr_T = partial(f_abstract, batch_version=can_batch)
    else:
        funcs_deriv = (
            partial(f_jvp_fix_x, batch_version=can_batch),
            partial(f_vjp_fix_x, batch_version=can_batch),
        )
        absr_T = partial(f_abstract_vjp_fix_x, batch_version=can_batch)

    f_jax = jax_linop.get_nonlinear_call(
        partial(f, batch_version=False),
        funcs_deriv,
        partial(f_abstract, batch_version=can_batch),
        absr_T,
        first_n_args_fixed=first_n_args_fixed,
    )
    if jit:
        f_jax = jax.jit(f_jax)
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    inp1 = jax.random.uniform(
        subkey, shape=shape, dtype=real_type_same_precision(dtype)
    )
    if is_complex_type(dtype):
        key, subkey = random.split(key)
        inp1 = inp1 + jax.random.uniform(
            subkey, shape=shape, dtype=real_type_same_precision(dtype)
        )

    key, subkey = random.split(key)
    inp2 = jax.random.uniform(
        subkey, shape=shape, dtype=real_type_same_precision(dtype)
    )
    if is_complex_type(dtype):
        key, subkey = random.split(key)
        inp2 = inp2 + jax.random.uniform(
            subkey, shape=shape, dtype=real_type_same_precision(dtype)
        )

    if first_n_args_fixed == 0:
        inp = (inp1, inp2)
    else:
        f_jax = partial(f_jax, inp1)
        inp = (inp2,)

    check_grads(f_jax, inp, order=1, modes=["fwd", "rev"], eps=1e-3)


@pmp("in_axes", ((0, 0), (0, None), (None, 0)))
@pmp("can_batch", (True, False))
@pmp("dtype", (np.float32, np.float64, np.complex64, np.complex128))
@pmp("shape", ((2,), (3, 4)))
@pmp("seed", (42,))
@pmp("jit", (False, True))
def test_vmap(in_axes, can_batch, dtype, shape, seed, jit):
    funcs_deriv = (
        partial(f_jvp, batch_version=can_batch),
        partial(f_vjp, batch_version=can_batch),
    )
    absr_T = partial(f_abstract, batch_version=can_batch)

    f_jax = jax_linop.get_nonlinear_call(
        partial(f, batch_version=False),
        funcs_deriv,
        partial(f_abstract, batch_version=can_batch),
        absr_T,
    )
    if jit:
        f_jax = jax.jit(f_jax)
    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    inp1 = jax.random.uniform(
        subkey, shape=shape, dtype=real_type_same_precision(dtype)
    )
    if is_complex_type(dtype):
        key, subkey = random.split(key)
        inp1 = inp1 + jax.random.uniform(
            subkey, shape=shape, dtype=real_type_same_precision(dtype)
        )

    key, subkey = random.split(key)
    inp2 = jax.random.uniform(
        subkey, shape=shape, dtype=real_type_same_precision(dtype)
    )
    if is_complex_type(dtype):
        key, subkey = random.split(key)
        inp2 = inp2 + jax.random.uniform(
            subkey, shape=shape, dtype=real_type_same_precision(dtype)
        )

    inp = (inp1, inp2)

    # test consistency against JAX native function
    res1 = f_jax(*inp)
    res2 = f_call(*inp)
    np.testing.assert_allclose(res1, res2)

    if in_axes[0] == 0:
        i1_shape = (10,) + shape
    else:
        i1_shape = shape
    if in_axes[1] == 0:
        i2_shape = (10,) + shape
    else:
        i2_shape = shape

    key, subkey = random.split(key)
    inp1 = jax.random.uniform(
        subkey, shape=i1_shape, dtype=real_type_same_precision(dtype)
    )
    if is_complex_type(dtype):
        key, subkey = random.split(key)
        inp1 = inp1 + jax.random.uniform(
            subkey, shape=i1_shape, dtype=real_type_same_precision(dtype)
        )

    key, subkey = random.split(key)
    inp2 = jax.random.uniform(
        subkey, shape=i2_shape, dtype=real_type_same_precision(dtype)
    )
    if is_complex_type(dtype):
        key, subkey = random.split(key)
        inp2 = inp2 + jax.random.uniform(
            subkey, shape=i2_shape, dtype=real_type_same_precision(dtype)
        )

    inp = (inp1, inp2)

    f_jax_vmap = jax.vmap(f_jax, in_axes=in_axes)
    f_call_vmap = jax.vmap(f_call, in_axes=in_axes)

    if jit:
        f_jax_vmap = jax.jit(f_jax_vmap)
        f_call_vmap = jax.jit(f_call_vmap)

    # test consistency against JAX native function with vmap
    res1 = f_jax_vmap(*inp)
    res2 = f_call_vmap(*inp)
    np.testing.assert_allclose(res1, res2)

    # test derivatives of vmap function
    check_grads(f_jax_vmap, inp, order=1, modes=["fwd", "rev"], eps=1e-3)
