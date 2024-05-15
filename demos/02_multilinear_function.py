# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2024 Max-Planck-Society

# %%
import jax
import jax.numpy as jnp
from jax import random


import jaxbind

jax.config.update("jax_enable_x64", True)

# %% [markdown]

# # Binding a multi-linear function to JAX

# This demo showcases the use of JAXbind for binding multi-linear functions to
# JAX. As an example we bind the Python function mlin computing
# (x,y) -> (x*y, x*y) to a JAX primitive. Note: multi-linear functions can also
# be regarded als general non-linear functions. For the JAXbind interface for
# non-linear functions see the 'demo_nonlin.py' and the docstring of the
# jaxbind.get_nonlinear_call'. Additional information for linear functions can
# also be found in 'demo_scipy_fft.py'.


# %%


def mlin(out, args, kwargs_dump):
    # extract the input from the input tuple
    x, y = args[0], args[1]

    # do the computation and write result in the out tuple
    out[0][()] = x * y
    out[1][()] = x * y


# %% [markdown]

# Besides the application of the function ('mlin') itself, JAXbind requires the
# linear transpose of the partial derivatives of 'mlin'.
# %%


# linear transpose of the partial derivative of 'mlin' with respect to the fist
# variable x.
def mlin_T1(out, args, kwargs_dump):
    y, da, db = args[0], args[1], args[2]
    out[0][()] = y * da + y * db


# linear transpose of the partial derivative of 'mlin' with respect to the second
# variable y.
def mlin_T2(out, args, kwargs_dump):
    x, da, db = args[0], args[1], args[2]
    out[0][()] = x * da + x * db


# %% [markdown]

# JAX needs to abstractly evaluate the code, thus needs to be able to evaluate
# the shape and dtype of the output of a function given the shape and dtype of
# the input. For this we have to provide the abstract eval functions for mlin,
# mlin_T1, and mlin_T2. The abstract evaluations functions return for each
# output argument a tuple containing the shape and dtype of this output. More
# details are in the 'demo_scipy_fft.py'


# %%
def mlin_abstract(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return (
        (args[0].shape, args[0].dtype),
        (args[0].shape, args[0].dtype),
    )


def mlin_abstract_T1(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype),)


def mlin_abstract_T2(*args, **kwargs):
    assert args[0].shape == args[1].shape
    return ((args[0].shape, args[0].dtype),)


# %% [markdown]

# Now we can register the JAX primitive corresponding to the Python function
# mlin.

# %%

func_T = (mlin_T1, mlin_T2)
func_abstract_T = (mlin_abstract_T1, mlin_abstract_T2)
mlin_jax = jaxbind.get_linear_call(
    mlin,
    func_T,
    mlin_abstract,
    func_abstract_T,
)

# generate some random input to showcase the use of the newly register JAX primitive
key = random.PRNGKey(42)
key, subkey = random.split(key)
inp0 = jax.random.uniform(subkey, shape=(10, 10), dtype=jnp.float64)
key, subkey = random.split(key)
inp1 = jax.random.uniform(subkey, shape=(10, 10), dtype=jnp.float64)
inp = (inp0, inp1)

# apply the new primitive
res = mlin_jax(*inp)

# jit compile the new primitive
mlin_jit = jax.jit(mlin_jax)
res_jit = mlin_jit(*inp)

# compute the jvp
res_jvp = jax.jvp(mlin_jit, inp, inp)
