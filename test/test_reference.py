# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2024 Max-Planck-Society

# %%
import jax
from jax import numpy as jnp

import jaxbind

jax.config.update("jax_enable_x64", True)

def test_reference():
    def lin(out, args, kwargs_dump):
        x, y = args
        out[0][()] = x + y
        out[1][()] = x + y


    def lin_T(out, args, kwargs_dump):
        a, b = args
        out[0][()] = a + b
        out[1][()] = a + b


    def lin_abstract(*args, **kwargs):
        x, y = args
        assert x.shape == y.shape
        return ((x.shape, x.dtype), (x.shape, x.dtype))


    lin_jax = jaxbind.get_linear_call(lin, lin_T, lin_abstract, lin_abstract)
    inp = (4 + jnp.zeros((2, 2)), 1 + jnp.zeros((2, 2)))
    lin_jax = jax.jit(lin_jax, static_argnames=("axes",))

    del lin, lin_abstract, lin_T


    def lin(*a):
        raise ValueError()


    _ = {"a": jnp.arange(10)}

    lin_jax(*inp, axes=(3, 4))
    jax.linear_transpose(lin_jax, *inp)(list(inp))
