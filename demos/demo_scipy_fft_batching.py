import jax
import jax.numpy as jnp
from jax import random

import scipy

import jax_linop

jax.config.update("jax_enable_x64", True)

# This demo showcases the use of the interface of jax_op for  binding linear
# functions to jax. Specifically it wraps the scipy fft as a JAX primitive.

# All JAX primitives can batched over input axes with 'jax.vmap'. JAX primitives
# registered via jax_op are no exception from that. By default jax_op internally
# translates the mapping operation into a sequential application of the function
# along the batching axis. Nevertheless, some custom function such as the scipy
# FFT natively support mapping over input axis giving significant speedups
# compared to a sequential computation. The jax_op interface allows to makes of
# custom batch, which will be demonstrated in this demo.


# scipy.fft.fftn is the function we want to wrap as a JAX primitive. Therefore,
# we wrap the scipy fft into a function 'fftn' having the signature required by
# jax_op. jax_op requires the function to take 3 arguments.
# The fist argument 'out' is a tuple into which the result is written, thus in
# this case array containing output of scipy.fft.fftn.
# The second argument is also a tuple and contains the input for the function,
# thus in our case this will be a tuple of length 1 containing the input array
# for the scipy fft.
# The third argument are potential keyword arguments given later to the JAX
# primitive. Due to the internals of JAX and jax_linop these keyword arguments
# are passed to the function in serialized from and need to be deserialized.
# For functions supporting custom batching the keyword arguments also contain as
# a lis the axis over which the function should be batched.


def fftn(out, args, kwargs_dump):
    # extract the input for the fft form the input tuple
    (x,) = args

    # deserialize keyword arguments
    kwargs = jax_linop.load_kwargs(kwargs_dump)

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


# Besides the application of the FFT jax_linop also requires the application of
# the linear transposed function. The syntax is identical to fftn just computing
# the linear transposed.


def fftn_transposed(out, args, kwargs_dump):
    (x,) = args
    kwargs = jax_linop.load_kwargs(kwargs_dump)

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


# JAX needs to abstractly evaluate the code, thus needs to be abel to evaluate
# the shape and dtype of the output of a function given the shape and dtype of
# the input. For this we have to provide the abstract eval functions for fftn
# and fftn_transposed.
# The abstract eval functions take normal arguments and keyword arguments and
# return a tuple containg the output information for each output argument of the
# function. Since fftn has only one output argument the output tuple of the
# abstract eval function has length 1.
# The output description of each output argument is also a tuple. The first
# entry in the tuple contains the shape of the output array. The second entry is
# the dtype of this array. The third entry in the tuple is only required for
# functions supporting custom batching and indicated the batching axis of the
# output of the function (thus fftn).


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


# JAX also needs to abstractly evaluate the transposed function. For that we
# have to provide the same information as for fftn. Since a fft is not changing
# the shape or dtype this function is identical to the fftn_abstract_eval. For
# general linear functions this might be different


def fftn_transposed_abstract_eval(*args, **kwargs):
    batch_axes = kwargs.pop("batch_axes", None)
    (a,) = args
    out_ax = ()
    if batch_axes:
        if len(batch_axes[0]) > 0:
            out_ax = batch_axes[0][-1]
    return ((a.shape, a.dtype, out_ax),)


# now we register our function as a custom JAX primitive using the interface for
# linear functions of jax_lino. jax_linop returns the resulting JAX primitive.
fftn_jax = jax_linop.get_linear_call(
    fftn,
    fftn_transposed,
    fftn_abstract_eval,
    fftn_transposed_abstract_eval,
    func_can_batch=True,  # indicate that our function supports custom batching
)


# generate some random input to showcase the use of the newly register JAX primitive
key = random.PRNGKey(42)
key, subkey = random.split(key)
inp = jax.random.uniform(subkey, shape=(10, 10), dtype=jnp.complex64)
inp = inp + 1j * jax.random.uniform(subkey, shape=(10, 10), dtype=jnp.complex64)


# apply the new primitive
res = fftn_jax(inp)

# apply the new primitive and pass the keyword argument "workers=2" to the scipy fft
res2 = fftn_jax(inp, workers=2)

# jit compile the new primitive
fftn_jax_jit = jax.jit(fftn_jax)
res_jit = fftn_jax_jit(res)

# vmap fft_jax over the fist axis of the input
res_vmap = jax.vmap(fftn_jax, in_axes=0)

# compute the jvp of fftn
res_jvp = jax.vjp(fftn, (inp,), (inp,))
