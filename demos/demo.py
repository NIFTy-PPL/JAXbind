import jax
import numpy as np
import scipy.fft
from jax.test_util import check_grads

import jax_linop

jax.config.update("jax_enable_x64", True)


def fft_operator(axes):
    def fft(inp, out, state):
        # we don't want normalization, just transpose, hence the strange norms
        out[()] = scipy.fft.fftn(inp, axes=state["axes"], norm="backward")

    def fft_T(inp, out, state):
        out[()
           ] = scipy.fft.ifftn(inp.conj(), axes=state["axes"],
                               norm="forward").conj()

    def fft_abstract(shape, dtype, state):
        return shape, dtype

    return jax_linop.get_linear_call(
        fft, fft_T, fft_abstract, fft_abstract, axes=tuple(axes)
    )


# create an FFT operator that transforms the first axis of its input array,
# as well as its adjoint operator
fft_op = fft_operator(axes=(0, ))

rng = np.random.default_rng(42)
a = rng.random((100, 20, 30)) - 0.5 + 1j * (rng.random((100, 20, 30)) - 0.5)

fft_op_T = jax.linear_transpose(fft_op, a)

check_grads(fft_op, (a, ), order=2, modes=["fwd", "rev"], eps=1.)
check_grads(fft_op_T, ([a], ), order=2, modes=["fwd", "rev"], eps=1.)
