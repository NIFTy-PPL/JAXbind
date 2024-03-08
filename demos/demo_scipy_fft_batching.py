import jax
from jax import numpy as jnp
from jax.test_util import check_grads

import scipy
from functools import partial

import jax_linop

jax.config.update("jax_enable_x64", True)



def f(out, args, kwargs_dump):
    x, = args
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    workers = kwargs.pop("workers", None)
    batch_axes = kwargs.pop("batch_axes", None)
    axes = list(range(len(x.shape)))
    if batch_axes:
        axes = [i for i in range(len(x.shape)) if not i in batch_axes[0]]
    out[0][()] = scipy.fft.fft2(x, axes=axes, norm="forward", workers=workers)


def f_T(out, args, kwargs_dump):
    x, = args
    kwargs = jax_linop.load_kwargs(kwargs_dump)
    workers = kwargs.pop("workers", None)
    batch_axes = kwargs.pop("batch_axes", None)
    axes = list(range(len(x.shape)))
    if batch_axes:
        axes = [i for i in range(len(x.shape)) if not i in batch_axes[0]]
    out[0][()] = scipy.fft.ifft2(x.conj(), axes=axes, norm="backward", workers=workers).conj()


def f_a(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    batch_axes = kwargs.pop("batch_axes", None)
    x, = args
    out_ax = ()
    if batch_axes:
        if len(batch_axes[0]) > 0:
            out_ax = batch_axes[0][-1]
    return ((x.shape, x.dtype, out_ax),)


def f_a_T(*args, **kwargs):
    # Returns `shape` and `dtype` of output as well as the added batch_axes of the `output``
    batch_axes = kwargs.pop("batch_axes", None)
    a, = args
    out_ax = ()
    if batch_axes:
        if len(batch_axes[0]) > 0:
            out_ax = batch_axes[0][-1]
    return ((a.shape, a.dtype, out_ax),)

f_jax = jax_linop.get_linear_call(
    f, f_T, f_a, f_a_T, func_can_batch=False
)


f_jax_can_batch = jax_linop.get_linear_call(
    f, f_T, f_a, f_a_T, func_can_batch=True
)

import numpy as np
rng = np.random.default_rng(42)

a = rng.random((10, 20,30))-0.5 + 1j*(rng.random((10 ,20,30))-0.5)
av = rng.random((2, 10, 20,30))-0.5 + 1j*(rng.random((2, 10 ,20,30))-0.5)


for bt in range(len(a.shape)):
    vj = jax.vmap(f_jax, in_axes=bt)
    vb = jax.vmap(f_jax_can_batch, in_axes=bt)
    rj = vj(a)
    rb = vb(a)
    np.testing.assert_allclose(rj, rb)

    check_grads(vj, (a,), order=2, modes=["fwd", "rev"], eps=1.0)
    check_grads(vb, (a,), order=2, modes=["fwd", "rev"], eps=1.0)

    for bt2 in range(len(a.shape)):
        vvj = jax.vmap(vj, in_axes=bt2)
        vvb = jax.vmap(vb, in_axes=bt2)
        rb = vvb(av)
        rj = vvj(av)
        np.testing.assert_allclose(rj, rb)

        check_grads(vvj, (av,), order=2, modes=["fwd", "rev"], eps=1.0)
        check_grads(vvb, (av,), order=2, modes=["fwd", "rev"], eps=1.0)


