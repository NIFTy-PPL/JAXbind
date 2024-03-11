# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2024 Max-Planck-Society

import jax
import numpy as np
from jax import numpy as jnp
from jax.test_util import check_grads

import jaxbind

from functools import partial

jax.config.update("jax_enable_x64", True)


def test_mlin():
    def mlin(out, args, kwargs_dump):
        x, y = args[0], args[1]
        out[0][()] = x * y
        out[1][()] = x * y

    def mlin_T1(out, args, kwargs_dump):
        y, da, db = args[0], args[1], args[2]
        out[0][()] = y * da + y * db

    def mlin_T2(out, args, kwargs_dump):
        x, da, db = args[0], args[1], args[2]
        out[0][()] = x * da + x * db

    def mlin_abstract(*args, **kwargs):
        assert args[0].shape == args[1].shape
        return (
            (args[0].shape, args[0].dtype, None),
            (args[0].shape, args[0].dtype, None),
        )

    def mlin_abstract_T1(*args, **kwargs):
        assert args[0].shape == args[1].shape
        return ((args[0].shape, args[0].dtype, None),)

    def mlin_abstract_T2(*args, **kwargs):
        out_axes = kwargs.pop("batch_axes", ())
        assert args[0].shape == args[1].shape
        return ((args[0].shape, args[0].dtype, out_axes),)

    func_T = (mlin_T1, mlin_T2)
    func_abstract_T = (mlin_abstract_T1, mlin_abstract_T2)
    mlin_jax = jaxbind.get_linear_call(
        mlin,
        func_T,
        mlin_abstract,
        func_abstract_T,
    )

    inp = (4 + jnp.zeros((2, 2)), 1 + jnp.zeros((2, 2)))
    check_grads(partial(mlin_jax, axes=(3, 4)), inp, order=2, modes=["fwd"], eps=1.0)
    check_grads(partial(mlin_jax, axes=(3, 4)), inp, order=1, modes=["rev"], eps=1.0)

    inp2 = (7 + jnp.zeros((2, 2)), -3 + jnp.zeros((2, 2)))
    inp3 = (10 + jnp.zeros((2, 2)), 5 + jnp.zeros((2, 2)))

    def mlin_purejax(x, y):
        return [x * y, x * y]

    primals_jax, njf_vjp = jax.vjp(mlin_purejax, *inp)
    res_vjp_jax = njf_vjp(mlin_purejax(*inp))
    res_jvp_jax = jax.jvp(mlin_purejax, inp2, inp3)

    primals, f_vjp = jax.vjp(mlin_jax, *inp)
    res_vjp = f_vjp(mlin_jax(*inp))
    res_jvp = jax.jvp(mlin_jax, inp2, inp3)

    np.testing.assert_allclose(primals, primals_jax)
    np.testing.assert_allclose(res_vjp, res_vjp_jax)
    np.testing.assert_allclose(res_jvp, res_jvp_jax)


def test_mlin_fixed_arg():
    def mlin(out, args, kwargs_dump):
        x, y = args[0], args[1]
        out[0][()] = x * x * y
        out[1][()] = x * y

    def mlin_T2(out, args, kwargs_dump):
        x, da, db = args[0], args[1], args[2]
        out[0][()] = x * x * da + x * db

    def mlin_abstract(*args, **kwargs):
        assert args[0].shape == args[1].shape
        return (
            (args[0].shape, args[0].dtype, None),
            (args[0].shape, args[0].dtype, None),
        )

    def mlin_abstract_T2(*args, **kwargs):
        assert args[0].shape == args[1].shape
        return ((args[0].shape, args[0].dtype, None),)

    func_T = (None, mlin_T2)
    func_abstract_T = (None, mlin_abstract_T2)
    mlin_jax = jaxbind.get_linear_call(
        mlin,
        func_T,
        mlin_abstract,
        func_abstract_T,
        first_n_args_fixed=1,
    )

    inp1 = 4 + jnp.zeros((2, 2))
    inp2 = 1 + jnp.zeros((2, 2))

    def mlin_purejax(x, y):
        return [x * y * x, x * y]

    mlin_jax_pt = partial(mlin_jax, inp1, axes=(3, 4))
    mlin_purejax_pt = partial(mlin_purejax, inp1)

    check_grads(mlin_jax_pt, (inp2,), order=2, modes=["fwd"], eps=1.0)
    check_grads(mlin_jax_pt, (inp2,), order=1, modes=["rev"], eps=1.0)

    primals, f_vjp = jax.vjp(mlin_jax_pt, inp2)
    res_vjp = f_vjp(mlin_jax_pt(inp2))
    res_jvp = jax.jvp(mlin_jax_pt, (inp1,), (inp2,))

    primals_jax, njf_vjp = jax.vjp(mlin_purejax_pt, inp2)
    res_vjp_jax = njf_vjp(mlin_purejax_pt(inp2))
    res_jvp_jax = jax.jvp(mlin_purejax_pt, (inp1,), (inp2,))

    np.testing.assert_allclose(primals, primals_jax)
    np.testing.assert_allclose(res_vjp, res_vjp_jax)
    np.testing.assert_allclose(res_jvp, res_jvp_jax)
