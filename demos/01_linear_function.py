# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2024 Max-Planck-Society

# %%
import jax
import jax.numpy as jnp
import scipy
from jax import random

import jaxbind

jax.config.update("jax_enable_x64", True)

# %% [markdown]

# # Binding Scipy's FFT to JAX

# This demo showcases the use of the interface of JAXbind for binding linear
# functions to jax. Specifically, it wraps scipy's FFT as a JAX primitive.

# The code wraps the scipy.fft.fftn function as a JAX primitive using JAXbind.
# It defines the fftn function with the required signature for JAXbind. The
# fftn function takes three arguments: out, args, and kwargs_dump. It extracts
# the input array from args, deserializes the keyword arguments from
# kwargs_dump, and computes the FFT using scipy.fft.fftn. Similarly, it defines
# the fftn_transposed function for the linear transposed operation. The code
# also includes abstract evaluation functions for fftn and fftn_transposed.
# Finally, it explains the purpose of wrapping scipy.fft.fftn as a JAX
# primitive.

# All native JAX primitives can be batched via 'jax.vmap'. Primitives
# registered via JAXbind are no exception here. By default, JAXbind performs
# the batching by sequential applying the function along the batching axis.
# However, it also exposes an interface to allow the user to perform the
# batching themself. As the FFT natively support mapping over input axis and
# yields significant speedups compared to a sequential computation, we
# demonstrate in the following how JAXbind can be used to register a custom
# batching.

# %%


def fftn(out, args, kwargs_dump):
    # extract the input for the fft form the input tuple
    (x,) = args

    # deserialize keyword arguments
    kwargs = jaxbind.load_kwargs(kwargs_dump)
    # extract keyword argument which can be given to the JAX primitive
    workers = kwargs.pop("workers", None)
    # extract the axes over which the FFT should be batched. This is only
    # necessary when supporting custom batching.
    batch_axes = kwargs.pop("batch_axes", None)

    # translate the batch_axes into the axes argument for scipy.fft.fftn
    axes = list(range(len(x.shape)))
    if batch_axes:
        axes = [i for i in range(len(x.shape)) if not i in batch_axes[0]]

    # compute the fft and write the result in the out tuple
    out[0][()] = scipy.fft.fftn(x, axes=axes, norm="forward", workers=workers)


# %% [markdown]

# In addition to applying the FFT, JAXbind also requires the implementation
# of the linear transposed function. The syntax for computing the linear
# transposed is identical to fftn.

# %%


def fftn_transposed(out, args, kwargs_dump):
    (x,) = args
    kwargs = jaxbind.load_kwargs(kwargs_dump)

    workers = kwargs.pop("workers", None)
    batch_axes = kwargs.pop("batch_axes", None)

    axes = list(range(len(x.shape)))
    if batch_axes:
        axes = [i for i in range(len(x.shape)) if not i in batch_axes[0]]

    # JAX need the transposed function and not he adjoint (transposed+complex
    # conjugated). Since scipy.fft.ifftn is the adjoint of scipy.fft.fftn we
    # have to undo the complex conjugation.
    out[0][()] = scipy.fft.ifftn(
        x.conj(), axes=axes, norm="backward", workers=workers
    ).conj()


# %% [markdown]

# JAX needs to abstractly evaluate the code, thus needs to be able to
# evaluate the shape and dtype of the output of a function given the shape
# and dtype of the input. For this we have to provide the abstract eval
# functions for fftn and fftn_transposed.

# The abstract eval functions take normal arguments and keyword arguments
# and return a tuple containing the output information for each output
# argument of the function. Since fftn has only one output argument the
# output tuple of the abstract eval function has length one.

# The output description of each output argument is also a tuple. The
# first entry in the tuple contains the shape of the output array. The
# second entry is the dtype of this array. The third entry in the tuple
# is only required for functions supporting custom batching and indicates
# the batching axis of the output of the function (thus fftn). In our case
# the batching axis of the output is identical to the batching axis of the
# input.


# %%


def fftn_abstract_eval(*args, **kwargs):
    # extract the input
    (x,) = args

    # extract potential batching axis
    batch_axes = kwargs.pop("batch_axes", None)

    # indicate the output batching axis if fftn is batched
    out_ax = ()
    if batch_axes:
        if len(batch_axes[0]) > 0:  # check if function is batched
            # check along which axis the fftn is batched. If fftn was batched
            # multiple times take the last batching axis
            out_ax = batch_axes[0][-1]

    # return shape dtype and potential batching axis of output
    return ((x.shape, x.dtype, out_ax),)


# %% [markdown]

# JAX also needs to abstractly evaluate the transposed function. For that we
# have to provide the same information as for fftn. Since a fft is not changing
# the shape or dtype this function is identical to the fftn_abstract_eval. For
# general linear functions this might be different.

# %%


def fftn_transposed_abstract_eval(*args, **kwargs):
    (a,) = args
    batch_axes = kwargs.pop("batch_axes", None)
    out_ax = ()
    if batch_axes:
        if len(batch_axes[0]) > 0:
            out_ax = batch_axes[0][-1]
    return ((a.shape, a.dtype, out_ax),)


# %% [markdown]

# Now we register our function as a custom JAX primitive using the interface for
# linear functions of JAXbind. JAXbind returns the resulting JAX primitive.

# %%
fftn_jax = jaxbind.get_linear_call(
    fftn,
    fftn_transposed,
    fftn_abstract_eval,
    fftn_transposed_abstract_eval,
    func_can_batch=True,  # indicate that our function supports custom batching
)


# generate some random input to showcase the use of the newly register JAX primitive
key = random.PRNGKey(42)
key, subkey = random.split(key)
inp = jax.random.uniform(subkey, shape=(10, 10), dtype=jnp.float64)
inp = inp + 1j * jax.random.uniform(subkey, shape=(10, 10), dtype=jnp.float64)


# apply the new primitive
res = fftn_jax(inp)

# apply the new primitive and pass the keyword argument "workers=2" to the scipy fft
res2 = fftn_jax(inp, workers=2)

# jit compile the new primitive
fftn_jax_jit = jax.jit(fftn_jax)
res_jit = fftn_jax_jit(inp)

# vmap fft_jax over the fist axis of the input
res_vmap = jax.vmap(fftn_jax, in_axes=0)

# compute the jvp of fftn
res_jvp = jax.jvp(fftn_jax, (inp,), (inp,))
