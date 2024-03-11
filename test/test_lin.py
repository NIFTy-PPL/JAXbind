# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2024 Max-Planck-Society

from functools import partial

have_ducc0 = True
try:
    import ducc0
except ImportError:
    have_ducc0 = False

import jax
import scipy
import numpy as np
import pytest
from jax.test_util import check_grads
from numpy.testing import assert_allclose

import jaxbind

pmp = pytest.mark.parametrize

jax.config.update("jax_enable_x64", True)

r2cdict = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128),
}

c2rdict = {
    np.dtype(np.complex64): np.dtype(np.float32),
    np.dtype(np.complex128): np.dtype(np.float64),
}


def realtype(dtype):
    return c2rdict[np.dtype(dtype)]


def complextype(dtype):
    return r2cdict[np.dtype(dtype)]


def fhtfunc(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump)
    # This function must _not_ keep any reference to 'inp' or 'args'!
    # Also, it must not change 'args'.
    if have_ducc0:
        ducc0.fft.genuine_fht(args[0], out=out[0], **kwargs)
    else:
        tmp = scipy.fft.fftn(args[0], axes=kwargs["axes"])
        out[0][()] = tmp.real - tmp.imag


def fhtfunc_abstract(*args, **kwargs):
    (x,) = args
    return ((x.shape, x.dtype),)


def c2cfunc(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump)
    (x,) = args
    if have_ducc0:
        ducc0.fft.c2c(x, out=out[0], **kwargs)
    else:
        func = scipy.fft.fftn if kwargs["forward"] else scipy.fft.ifftn
        norm = "backward" if kwargs["forward"] else "forward"
        out[0][()] = func(x, norm=norm, axes=kwargs["axes"])


def c2cfunc_abstract(*args, **kwargs):
    (x,) = args
    return ((x.shape, x.dtype),)


def alm2realalm(alm, lmax, dtype, out=None):
    if out is None:
        out = np.empty((alm.shape[0], alm.shape[1] * 2 - lmax - 1), dtype=dtype)
    out[:, 0 : lmax + 1] = alm[:, 0 : lmax + 1].real
    out[:, lmax + 1 :] = alm[:, lmax + 1 :].view(dtype)
    out[:, lmax + 1 :] *= np.sqrt(2.0)
    return out


def realalm2alm(alm, lmax, dtype, out=None):
    if out is None:
        out = np.empty((alm.shape[0], (alm.shape[1] + lmax + 1) // 2), dtype=dtype)
    out[:, 0 : lmax + 1] = alm[:, 0 : lmax + 1]
    out[:, lmax + 1 :] = alm[:, lmax + 1 :].view(dtype)
    out[:, lmax + 1 :] *= np.sqrt(2.0) / 2
    return out


def sht2d_operator(lmax, mmax, ntheta, nphi, geometry, spin, nthreads):
    def sht2dfunc(out, args, kwargs_dump):
        inp = args[0]
        state = jaxbind.load_kwargs(kwargs_dump)
        tmp = realalm2alm(inp, state["lmax"], complextype(inp.dtype))
        ducc0.sht.synthesis_2d(
            lmax=state["lmax"],
            mmax=state["mmax"],
            spin=state["spin"],
            map=out[0],
            alm=tmp,
            nthreads=state["nthreads"],
            geometry=state["geometry"],
            ntheta=state["ntheta"],
            nphi=state["nphi"],
        )

    def sht2dfunc_T(out, args, kwargs_dump):
        inp = args[0]
        state = jaxbind.load_kwargs(kwargs_dump)
        tmp = ducc0.sht.adjoint_synthesis_2d(
            lmax=state["lmax"],
            mmax=state["mmax"],
            spin=state["spin"],
            map=inp,
            nthreads=state["nthreads"],
            geometry=state["geometry"],
        )
        alm2realalm(tmp, state["lmax"], inp.dtype, out[0])

    def sht2dfunc_abstract(*args, **kwargs):
        spin = kwargs["spin"]
        ncomp = 1 if spin == 0 else 2
        shape_out = (ncomp, kwargs["ntheta"], kwargs["nphi"])
        return ((shape_out, args[0].dtype),)

    def sht2dfunc_abstract_T(*args, **kwargs):
        spin = kwargs["spin"]
        ncomp = 1 if spin == 0 else 2
        lmax, mmax = kwargs["lmax"], kwargs["mmax"]
        nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
        nalm = nalm * 2 - lmax - 1
        shape_out = (ncomp, nalm)
        return ((shape_out, args[0].dtype),)

    func = jaxbind.get_linear_call(
        sht2dfunc,
        sht2dfunc_T,
        sht2dfunc_abstract,
        sht2dfunc_abstract_T,
    )

    return partial(
        func,
        geometry=geometry,
        ntheta=ntheta,
        nphi=nphi,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        nthreads=nthreads,
    )


def healpixfunc(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump).copy()
    theta, phi0, nphi, ringstart, x = args
    tmp = realalm2alm(x, kwargs["lmax"], complextype(x.dtype))
    ducc0.sht.synthesis(
        map=out[0],
        alm=tmp,
        theta=theta,
        phi0=phi0,
        nphi=nphi,
        ringstart=ringstart,
        spin=kwargs["spin"],
        lmax=kwargs["lmax"],
        mmax=kwargs["mmax"],
        nthreads=kwargs["nthreads"],
    )


def healpixfunc_T(out, args, kwargs_dump):
    kwargs = jaxbind.load_kwargs(kwargs_dump).copy()
    theta, phi0, nphi, ringstart, x = args
    tmp = ducc0.sht.adjoint_synthesis(
        map=x,
        theta=theta,
        phi0=phi0,
        nphi=nphi,
        ringstart=ringstart,
        spin=kwargs["spin"],
        lmax=kwargs["lmax"],
        mmax=kwargs["mmax"],
        nthreads=kwargs["nthreads"],
    )
    alm2realalm(tmp, kwargs["lmax"], x.dtype, out[0])


def healpixfunc_abstract(*args, **kwargs):
    _, _, _, _, x = args
    spin = kwargs["spin"]
    ncomp = 1 if spin == 0 else 2
    shape_out = (ncomp, 12 * kwargs["nside"] ** 2)
    return ((shape_out, x.dtype),)


def healpixfunc_abstract_T(*args, **kwargs):
    _, _, _, _, x = args
    spin = kwargs["spin"]
    ncomp = 1 if spin == 0 else 2
    lmax, mmax = kwargs["lmax"], kwargs["mmax"]
    nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
    nalm = nalm * 2 - lmax - 1
    shape_out = (ncomp, nalm)
    return ((shape_out, x.dtype),)


def _assert_close(a, b, epsilon):
    if have_ducc0:
        assert_allclose(ducc0.misc.l2error(a, b), 0, atol=epsilon)
    else:
        assert_allclose(
            scipy.linalg.norm(a - b) / scipy.linalg.norm(a), 0, atol=epsilon
        )


@pmp("shape,axes", (((100,), (0,)), ((10, 17), (0, 1)), ((10, 17, 3), (1,))))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_fht(shape, axes, dtype, nthreads):
    rng = np.random.default_rng(42)

    fht = jaxbind.get_linear_call(
        fhtfunc, fhtfunc, fhtfunc_abstract, fhtfunc_abstract
    )
    kw = dict(axes=axes, nthreads=nthreads)

    a = (rng.random(shape) - 0.5).astype(dtype)
    b1 = np.array(fht(a, **kw)[0])
    if have_ducc0:
        b2 = ducc0.fft.genuine_fht(a, **kw)
    else:
        b2 = scipy.fft.fftn(a, axes=kw["axes"])
        b2 = b2.real - b2.imag

    _assert_close(b1, b2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    max_order = 2
    check_grads(
        partial(fht, **kw), (a,), order=max_order, modes=("fwd", "rev"), eps=1.0
    )


@pmp("shape,axes", (((100,), (0,)), ((10, 17), (0, 1)), ((10, 17, 3), (1,))))
@pmp("forward", (False, True))
@pmp("dtype", (np.complex64, np.complex128))
@pmp("nthreads", (1, 2))
def test_c2c(shape, axes, forward, dtype, nthreads):
    rng = np.random.default_rng(42)

    # The C2C FFT matrix is symmetric!
    c2c = jaxbind.get_linear_call(
        c2cfunc, c2cfunc, c2cfunc_abstract, c2cfunc_abstract
    )
    kw = dict(axes=axes, forward=forward, nthreads=nthreads)

    a = (rng.random(shape) - 0.5).astype(dtype) + (
        1j * (rng.random(shape) - 0.5)
    ).astype(dtype)
    b1 = np.array(c2c(a, **kw)[0])
    if have_ducc0:
        b2 = ducc0.fft.c2c(a, **kw)
    else:
        if forward:
            b2 = scipy.fft.fftn(a, axes=kw["axes"])
        else:
            b2 = scipy.fft.ifftn(a, norm="forward", axes=kw["axes"])
    _assert_close(b1, b2, epsilon=1e-6 if dtype == np.complex64 else 1e-14)

    max_order = 2
    check_grads(
        partial(c2c, **kw), (a,), order=max_order, modes=("fwd", "rev"), eps=1.0
    )


def nalm(lmax, mmax):
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1.0, 1.0, (ncomp, nalm(lmax, mmax))) + 1j * rng.uniform(
        -1.0, 1.0, (ncomp, nalm(lmax, mmax))
    )
    # make a_lm with m==0 real-valued
    res[:, 0 : lmax + 1].imag = 0.0
    ofs = 0
    for s in range(spin):
        res[:, ofs : ofs + spin - s] = 0.0
        ofs += lmax + 1 - s
    return res


@pmp("lmmax", ((10, 10), (20, 5)))
@pmp("geometry", ("GL", "F1", "F2", "CC", "DH", "MW", "MWflip"))
@pmp("ntheta", (20,))
@pmp("nphi", (30,))
@pmp("spin", (0, 2))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_sht2d(lmmax, geometry, ntheta, nphi, spin, dtype, nthreads):
    if not have_ducc0:
        pytest.skip()
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
        nthreads=nthreads,
    )
    # The conjugations are only necessary if the input or output data types
    # are complex. Leaving them out makes things (surprisingly) faster.
    #    op_adj = lambda x: jax.linear_transpose(lambda y: op(y)[0], alm0r)(x.conj())[
    #        0
    #    ].conj()
    op_adj = lambda x: jax.linear_transpose(lambda y: op(y)[0], alm0r)(x)[0]

    map1 = np.array(op(alm0r)[0])
    map2 = ducc0.sht.synthesis_2d(
        alm=alm0,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        geometry=geometry,
        ntheta=ntheta,
        nphi=nphi,
        nthreads=nthreads,
    )
    _assert_close(map1, map2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    map0 = (rng.random((ncomp, ntheta, nphi)) - 0.5).astype(dtype)
    alm1r = np.array(op_adj(map0))
    alm1 = realalm2alm(alm1r, lmax, complextype(dtype))
    alm2 = ducc0.sht.adjoint_synthesis_2d(
        map=map0, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, nthreads=nthreads
    )
    _assert_close(alm1, alm2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    max_order = 2
    check_grads(op, (alm0r,), order=max_order, modes=("fwd", "rev"), eps=1.0)
    check_grads(op_adj, (map0,), order=max_order, modes=("fwd", "rev"), eps=1.0)


@pmp("lmmax", ((10, 10), (20, 5)))
@pmp("nside", (16, 2))
@pmp("spin", (0, 2))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_healpix(lmmax, nside, spin, dtype, nthreads):
    if not have_ducc0:
        pytest.skip()
    rng = np.random.default_rng(42)

    lmax, mmax = lmmax
    ncomp = 1 if spin == 0 else 2
    alm0 = random_alm(lmax, mmax, spin, ncomp, rng).astype(complextype(dtype))
    alm0r = alm2realalm(alm0, lmax, dtype)
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    hpxparam = base.sht_info()

    hp = jaxbind.get_linear_call(
        healpixfunc,
        healpixfunc_T,
        healpixfunc_abstract,
        healpixfunc_abstract_T,
        first_n_args_fixed=4,
    )

    def hpp(x):
        # Partial insert where the first parameter is not inserted
        return hp(
            hpxparam["theta"],
            hpxparam["phi0"],
            hpxparam["nphi"],
            hpxparam["ringstart"],
            x,
            lmax=lmax,
            mmax=mmax,
            spin=spin,
            nthreads=nthreads,
            nside=nside,
        )

    # The conjugations are only necessary if the input or output data types
    # are complex. Leaving them out makes things (surprisingly) faster.
    #    hp_adj = lambda x: jax.linear_transpose(lambda y: hp(y)[0], alm0r)(x.conj())[
    #        0
    #    ].conj()
    hpp_adj = lambda x: jax.linear_transpose(lambda y: hpp(y)[0], alm0r)(x)[0]

    map1 = np.array(hpp(alm0r)[0])
    map2 = ducc0.sht.synthesis(
        alm=alm0, lmax=lmax, mmax=mmax, spin=spin, nthreads=nthreads, **hpxparam
    )
    _assert_close(map1, map2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    map0 = (rng.random((ncomp, 12 * nside**2)) - 0.5).astype(dtype)
    alm1r = np.array(hpp_adj(map0))
    alm1 = realalm2alm(alm1r, lmax, complextype(dtype))
    alm2 = ducc0.sht.adjoint_synthesis(
        map=map0, lmax=lmax, mmax=mmax, spin=spin, nthreads=nthreads, **hpxparam
    )
    _assert_close(alm1, alm2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    max_order = 2
    check_grads(hpp, (alm0r,), order=max_order, modes=("fwd", "rev"), eps=1.0)
    check_grads(hpp_adj, (map0,), order=max_order, modes=("fwd", "rev"), eps=1.0)
