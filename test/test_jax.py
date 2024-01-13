import ducc0
import jax
import numpy as np
import pytest
from jax.test_util import check_grads
from numpy.testing import assert_allclose

import jax_linop

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize

r2cdict = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128)
}

c2rdict = {
    np.dtype(np.complex64): np.dtype(np.float32),
    np.dtype(np.complex128): np.dtype(np.float64)
}


def realtype(dtype):
    return c2rdict[np.dtype(dtype)]


def complextype(dtype):
    return r2cdict[np.dtype(dtype)]


def fht_operator(axes, nthreads):
    def fhtfunc(inp, out, state):
        # This function must _not_ keep any reference to 'inp' or 'out'!
        # Also, it must not change 'inp' or 'state'.
        ducc0.fft.genuine_fht(
            inp, out=out, axes=state["axes"], nthreads=state["nthreads"]
        )

    def fhtfunc_abstract(shape, dtype, state):
        return shape, dtype

    return jax_linop.get_linear_call(
        fhtfunc,
        fhtfunc,
        fhtfunc_abstract,
        fhtfunc_abstract,
        axes=tuple(axes),
        nthreads=int(nthreads)
    )


def c2c_operator(axes, forward, nthreads):
    def c2cfunc(inp, out, state):
        ducc0.fft.c2c(
            inp,
            out=out,
            axes=state["axes"],
            nthreads=state["nthreads"],
            forward=state["forward"]
        )

    def c2cfunc_abstract(shape, dtype, state):
        return shape, dtype

    return jax_linop.get_linear_call(
        c2cfunc,
        c2cfunc,  # The C2C FFT matrix is symmetric!
        c2cfunc_abstract,
        c2cfunc_abstract,
        axes=tuple(axes),
        forward=bool(forward),
        nthreads=int(nthreads)
    )


def alm2realalm(alm, lmax, dtype):
    res = np.zeros((alm.shape[0], alm.shape[1] * 2 - lmax - 1), dtype=dtype)
    res[:, 0:lmax + 1] = alm[:, 0:lmax + 1].real
    res[:, lmax + 1:] = alm[:, lmax + 1:].view(dtype) * np.sqrt(2.)
    return res


def realalm2alm(alm, lmax, dtype):
    res = np.zeros((alm.shape[0], (alm.shape[1] + lmax + 1) // 2), dtype=dtype)
    res[:, 0:lmax + 1] = alm[:, 0:lmax + 1]
    res[:, lmax + 1:] = alm[:, lmax + 1:].view(dtype) * (np.sqrt(2.) / 2)
    return res


def sht2d_operator(lmax, mmax, ntheta, nphi, geometry, spin, nthreads):
    def sht2dfunc(inp, out, state):
        tmp = realalm2alm(inp, state["lmax"], complextype(inp.dtype))
        ducc0.sht.synthesis_2d(
            lmax=state["lmax"],
            mmax=state["mmax"],
            spin=state["spin"],
            map=out,
            alm=tmp,
            nthreads=state["nthreads"],
            geometry=state["geometry"]
        )

    def sht2dfunc_T(inp, out, state):
        tmp = ducc0.sht.adjoint_synthesis_2d(
            lmax=state["lmax"],
            mmax=state["mmax"],
            spin=state["spin"],
            map=inp,
            nthreads=state["nthreads"],
            geometry=state["geometry"]
        )
        out[()] = alm2realalm(tmp, state["lmax"], inp.dtype)

    def sht2dfunc_abstract(shape_in, dtype_in, state):
        spin = state["spin"]
        ncomp = 1 if spin == 0 else 2
        shape_out = (ncomp, state["ntheta"], state["nphi"])
        return shape_out, dtype_in

    def sht2dfunc_abstract_T(shape_in, dtype_in, state):
        spin = state["spin"]
        ncomp = 1 if spin == 0 else 2
        lmax, mmax = state["lmax"], state["mmax"]
        nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
        nalm = nalm * 2 - lmax - 1
        shape_out = (ncomp, nalm)
        return shape_out, dtype_in

    return jax_linop.get_linear_call(
        sht2dfunc,
        sht2dfunc_T,
        sht2dfunc_abstract,
        sht2dfunc_abstract_T,
        lmax=int(lmax),
        mmax=int(mmax),
        spin=int(spin),
        ntheta=int(ntheta),
        nphi=int(nphi),
        geometry=str(geometry),
        nthreads=int(nthreads)
    )


def _assert_close(a, b, epsilon):
    assert_allclose(ducc0.misc.l2error(a, b), 0, atol=epsilon)


@pmp("shape,axes", (((100, ), (0, )), ((10, 17), (0, 1)), ((10, 17, 3), (1, ))))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_fht(shape, axes, dtype, nthreads):
    rng = np.random.default_rng(42)

    op = fht_operator(axes=axes, nthreads=nthreads)
    a = (rng.random(shape) - 0.5).astype(dtype)
    b1 = np.array(op(a)[0])
    b2 = ducc0.fft.genuine_fht(a, axes=axes, nthreads=nthreads)
    _assert_close(b1, b2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    max_order = 2
    check_grads(op, (a, ), order=max_order, modes=("fwd", "rev"), eps=1.)


@pmp("shape,axes", (((100, ), (0, )), ((10, 17), (0, 1)), ((10, 17, 3), (1, ))))
@pmp("forward", (False, True))
@pmp("dtype", (np.complex64, np.complex128))
@pmp("nthreads", (1, 2))
def test_c2c(shape, axes, forward, dtype, nthreads):
    rng = np.random.default_rng(42)

    op = c2c_operator(axes=axes, forward=forward, nthreads=nthreads)
    a = (rng.random(shape) -
         0.5).astype(dtype) + (1j * (rng.random(shape) - 0.5)).astype(dtype)
    b1 = np.array(op(a)[0])
    b2 = ducc0.fft.c2c(a, axes=axes, forward=forward, nthreads=nthreads)
    _assert_close(b1, b2, epsilon=1e-6 if dtype == np.complex64 else 1e-14)

    max_order = 2
    check_grads(op, (a, ), order=max_order, modes=("fwd", "rev"), eps=1.)


def nalm(lmax, mmax):
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax + 1].imag = 0.
    ofs = 0
    for s in range(spin):
        res[:, ofs:ofs + spin - s] = 0.
        ofs += lmax + 1 - s
    return res


@pmp("lmmax", ((10, 10), (20, 5)))
@pmp("geometry", ("GL", "F1", "F2", "CC", "DH", "MW", "MWflip"))
@pmp("ntheta", (20, ))
@pmp("nphi", (30, ))
@pmp("spin", (0, 2))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_sht2d(lmmax, geometry, ntheta, nphi, spin, dtype, nthreads):
    rng = np.random.default_rng(42)

    lmax, mmax = lmmax
    ncomp = 1 if spin == 0 else 2
    alm0 = random_alm(lmax, mmax, spin, ncomp, rng).astype(complextype(dtype))
    alm0r = alm2realalm(alm0, lmax, dtype)

    op = sht2d_operator(
        lmax=lmax,
        mmax=mmax,
        ntheta=ntheta,
        nphi=nphi,
        geometry=geometry,
        spin=spin,
        nthreads=nthreads
    )
    op_adj = lambda x: jax.linear_transpose(lambda y: op(y)[0], alm0r
                                           )(x.conj())[0].conj()

    map1 = np.array(op(alm0r)[0])
    map2 = ducc0.sht.synthesis_2d(
        alm=alm0,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        geometry=geometry,
        ntheta=ntheta,
        nphi=nphi,
        nthreads=nthreads
    )
    _assert_close(map1, map2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    map0 = (rng.random((ncomp, ntheta, nphi)) - 0.5).astype(dtype)
    alm1r = np.array(op_adj(map0))
    alm1 = realalm2alm(alm1r, lmax, complextype(dtype))
    alm2 = ducc0.sht.adjoint_synthesis_2d(
        map=map0,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        geometry=geometry,
        nthreads=nthreads
    )
    _assert_close(alm1, alm2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    max_order = 2
    check_grads(op, (alm0r, ), order=max_order, modes=("fwd", "rev"), eps=1.)
    check_grads(op_adj, (map0, ), order=max_order, modes=("fwd", "rev"), eps=1.)
