# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2024 Max-Planck-Society

# %%
from functools import partial

import jax
from jax import numpy as jnp

import jaxbind

jax.config.update("jax_enable_x64", True)


# %% [markdown]

# # Binding non-linear functions to JAX with fixed arguments

# Bind a nonlinear function to JAX using JAXbind. Some input parts are fixed
# (nondifferentiable) to showcase JAXbind's handling of fixed arguments.

# The script begins by defining the nonlinear function using the JAXbind
# compatible signature of out, args, and kwargs_dump. The derivative
# function, which is registered with JAX, has the same signature as the
# nonlinear function but includes an additional variable for the tangent of
# the variable being differentiated. Additionally, we define the transpose of
# the derivative function, as JAX may transpose it for operations like
# retrieving the VJP.

# %%


def nonlin(out, args, kwargs_dump):
    # (x,y) -> (xy, y**2)
    x, y = args
    out[0][()] = x * y
    out[1][()] = y * y


def nonlin_deriv(out, args, kwargs_dump):
    # (x,y,dy) -> (ydx + xdy, 2 * y * dy)
    x, y, dy = args
    out[0][()] = x * dy
    out[1][()] = 2 * y * dy


def nonlin_deriv_T(out, args, kwargs_dump):
    # (x, y, da, db) -> (yda, xda + 2ydb)
    x, y, da, db = args
    out[0][()] = x * da + 2 * y * db


# %% [markdown]

# Define the abstract evaluation functions for JAX that translate input
# shape and dtypes to output shape and dtypes.

# %%


def nonlin_abstract(*args, **kwargs):
    # Returns `shape` and `dtype` of output
    x, y, *_ = args
    assert x.shape == x.shape and x.dtype is y.dtype
    return ((x.shape, x.dtype), (x.shape, x.dtype))


def nonlin_abstract_T(*args, **kwargs):
    # Returns `shape` and `dtype` of output
    a, b, da, db = args
    assert a.shape == b.shape == da.shape == db.shape
    assert a.dtype is b.dtype and a.dtype is da.dtype and a.dtype is db.dtype
    return ((a.shape, a.dtype),)


# %%
nonlin_jax = jaxbind.get_nonlinear_call(
    nonlin,
    (nonlin_deriv, nonlin_deriv_T),
    nonlin_abstract,
    nonlin_abstract_T,
    # Tell JAXbind that the first parameter to `nonlin` (and derived functions)
    # is not to be differentiated.
    first_n_args_fixed=1,
    func_can_batch=True,
)

inp_f = 4 + jnp.zeros((1,))
inp = 6 + jnp.zeros((1,))
nonlin_jax_pt = partial(nonlin_jax, inp_f)

_ = jax.jvp(nonlin_jax_pt, (inp,), (inp,))
_, nonlin_jax_pt_vjp = jax.vjp(nonlin_jax_pt, inp)
nonlin_jax_pt_vjp = jax.jit(nonlin_jax_pt_vjp)
_ = nonlin_jax_pt_vjp([inp, inp])
