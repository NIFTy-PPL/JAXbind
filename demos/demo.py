import scipy.fft
import jax_linop
import numpy as np

def fft_operator(axes):
    def fftfunc(inp, out, adjoint, state):
        # we don't want normalization, just adjointness, hence the strange norms
        if adjoint:
            out[()] = scipy.fft.ifftn(inp, axes=state["axes"], norm="forward")
        else:
            out[()] = scipy.fft.fftn(inp, axes=state["axes"], norm="backward")

    def fftfunc_abstract(shape, dtype, adjoint, state):
        return shape, dtype

    return jax_linop.make_linop(fftfunc, fftfunc_abstract, axes=tuple(axes))

# create an FFT operator that transforms the first axis of its input array,
# as well as its adjoint operator
fft_op, fft_op_adjoint = fft_operator(axes=(0,))

from jax import config
config.update("jax_enable_x64", True)

from jax.test_util import check_grads

rng = np.random.default_rng(42)
a = rng.random((100,20,30))-0.5 + 1j*(rng.random((100,20,30))-0.5)

check_grads(fft_op, (a,), order=2, modes=["fwd", "rev"], eps=1.)
check_grads(fft_op_adjoint, (a,), order=2, modes=["fwd", "rev"], eps=1.)
