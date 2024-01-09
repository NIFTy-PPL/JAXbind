import jax
import numpy as np
import scipy.fft
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)


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
fft_op = fft_operator(axes=(0, ))

rng = np.random.default_rng(42)
a = rng.random((100, 20, 30)) - 0.5 + 1j * (rng.random((100, 20, 30)) - 0.5)

fft_op_T = jax.linear_transpose(fft_op, a)

check_grads(fft_op, (a, ), order=2, modes=["fwd", "rev"], eps=1.)
check_grads(fft_op_T, ([a], ), order=2, modes=["fwd", "rev"], eps=1.)
