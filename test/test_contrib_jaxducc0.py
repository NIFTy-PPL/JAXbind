# SPDX-License-Identifier: BSD-2-Clause

# Copyright(C) 2024 Max-Planck-Society

from functools import partial

import jax
import numpy as np
import pytest
from jax.test_util import check_grads
from numpy.testing import assert_allclose

ducc0 = pytest.importorskip("ducc0")

from jaxbind.contrib import jaxducc0

pmp = pytest.mark.parametrize

jax.config.update("jax_enable_x64", True)


def _assert_close(a, b, epsilon):
    assert_allclose(ducc0.misc.l2error(a, b), 0, atol=epsilon)


@pmp("shape,axes", (((100,), (0,)), ((10, 17), (0, 1)), ((10, 17, 3), (1,))))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_fht(shape, axes, dtype, nthreads):
    rng = np.random.default_rng(42)

    kw = dict(axes=axes, nthreads=nthreads)
    fht = partial(jaxducc0.genuine_fht, **kw)
    fht_alt = partial(ducc0.fft.genuine_fht, **kw)

    a = (rng.random(shape) - 0.5).astype(dtype)
    b1 = np.array(fht(a)[0])
    b2 = fht_alt(a)

    _assert_close(b1, b2, epsilon=1e-6 if dtype == np.float32 else 1e-14)

    max_order = 2
    check_grads(fht, (a,), order=max_order, modes=("fwd", "rev"), eps=1.0)


@pmp("shape,axes", (((100,), (0,)), ((10, 17), (0, 1)), ((10, 17, 3), (1,))))
@pmp("forward", (False, True))
@pmp("dtype", (np.complex64, np.complex128))
@pmp("nthreads", (1, 2))
def test_c2c(shape, axes, forward, dtype, nthreads):
    rng = np.random.default_rng(42)

    # The C2C FFT matrix is symmetric!
    kw = dict(axes=axes, forward=forward, nthreads=nthreads)
    c2c = partial(jaxducc0.c2c, **kw)
    c2c_alt = partial(ducc0.fft.c2c, **kw)

    a = (rng.random(shape) - 0.5).astype(dtype)
    a += (1j * (rng.random(shape) - 0.5)).astype(dtype)

    b1 = np.array(c2c(a)[0])
    b2 = c2c_alt(a)
    _assert_close(b1, b2, epsilon=1e-6 if dtype == np.complex64 else 1e-14)

    max_order = 2
    check_grads(c2c, (a,), order=max_order, modes=("fwd", "rev"), eps=1.0)


rng = np.random.default_rng(42)
a1 = rng.random((5, 5, 5)) - 0.5 + 1j * (rng.random((5, 5, 5)) - 0.5)
a = (a1,)
av1 = rng.random((5, 5, 5, 5)) - 0.5 + 1j * (rng.random((5, 5, 5, 5)) - 0.5)
av = (av1,)

# HACK to disable native batching again
fmap = {}
for f_native in (jaxducc0.c2c, jaxducc0.genuine_fht):
    assert isinstance(f_native, partial) and "_func" in f_native.keywords
    kw = f_native.keywords
    kw["_func"] = kw["_func"]._replace(can_batch=False)
    fmap[f_native] = partial(f_native.func, *f_native.args, **kw)


@pmp(
    "f_native,f_map",
    (
        (jaxducc0.c2c, fmap[jaxducc0.c2c]),
        (partial(jaxducc0.c2c, axes=0), partial(fmap[jaxducc0.c2c], axes=0)),
        (
            lambda x: jaxducc0.genuine_fht(x.real),
            lambda x: fmap[jaxducc0.genuine_fht](x.real),
        ),
        (
            lambda x: jaxducc0.genuine_fht(x.real, axes=0),
            lambda x: fmap[jaxducc0.genuine_fht](x.real, axes=0),
        ),
    ),
)
@pmp("bt_a1", (0, 1))
@pmp("bt2_a1", (3, 0))
@pmp("o_a1", (0, 1))
def test_ht_vmap(f_native, f_map, bt_a1, bt2_a1, o_a1):
    vj = jax.vmap(f_map, in_axes=(bt_a1,), out_axes=[o_a1])
    vb = jax.vmap(f_native, in_axes=(bt_a1,), out_axes=[o_a1])
    rj = vj(*a)
    rb = vb(*a)
    np.testing.assert_allclose(rj, rb)
    assert rj[0].shape == rb[0].shape
    check_grads(vb, a, order=1, modes=["fwd", "rev"], eps=1.0)

    vvj = jax.vmap(vj, in_axes=(bt2_a1,), out_axes=[o_a1])
    vvb = jax.vmap(vb, in_axes=(bt2_a1,), out_axes=[o_a1])
    rb = vvb(*av)
    rj = vvj(*av)
    assert rj[0].shape == rb[0].shape
    np.testing.assert_allclose(rj[0], rb[0])


@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_wgridder(dtype, nthreads):
    speedoflight = 299792458.0
    rng = np.random.default_rng(42)

    fov = 0.1 * np.pi
    npix_x = 128
    npix_y = 64
    nrow, nchan = 1000, 2
    pixsize_x = fov / npix_x
    pixsize_y = fov / npix_y
    f0 = 1e9
    freq = f0 + np.arange(nchan) * (f0 / nchan)
    uvw = (rng.random((nrow, 3)) - 0.5) / (pixsize_x * f0 / speedoflight)
    uvw[:, 2] /= 20
    epsilon = 1e-5
    do_wgridding = True

    dirty = rng.random((npix_x, npix_y), dtype)

    wgridder = jaxducc0.get_wgridder(
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        npix_x=npix_x,
        npix_y=npix_y,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
    )
    vis_jaxducc = wgridder(uvw, freq, dirty)[0]
    vis_ducc = ducc0.wgridder.experimental.dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=dirty,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
    )

    np.testing.assert_allclose(vis_jaxducc, vis_ducc, atol=epsilon, rtol=epsilon)
    check_grads(
        partial(wgridder, uvw, freq), (dirty,), order=2, modes=("fwd", "rev"), eps=1.0
    )


def _random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1.0, 1.0, (ncomp, jaxducc0.nalm(lmax, mmax))) + 1j * rng.uniform(
        -1.0, 1.0, (ncomp, jaxducc0.nalm(lmax, mmax))
    )
    # make a_lm with m==0 real-valued
    res[:, 0 : lmax + 1].imag = 0.0
    ofs = 0
    for s in range(spin):
        res[:, ofs : ofs + spin - s] = 0.0
        ofs += lmax + 1 - s
    return res


@pmp("lmmax", ((10, 10), (20, 5)))
@pmp("nside", (16, 2))
@pmp("spin", (0, 2))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_healpix(lmmax, nside, spin, dtype, nthreads):
    lmax, mmax = lmmax
    ncomp = 1 if spin == 0 else 2
    rng = np.random.default_rng(42)
    alm0 = _random_alm(lmax, mmax, spin, ncomp, rng).astype(
        jaxducc0._complextype(dtype)
    )
    alm0r = jaxducc0._alm2realalm(alm0, lmax, dtype)
    hpp = jaxducc0.get_healpix_sht(nside, lmax, mmax, spin, nthreads)

    map0 = np.array(hpp(alm0r)[0])
    map1 = ducc0.sht.synthesis(
        alm=alm0,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        nthreads=nthreads,
        **ducc0.healpix.Healpix_Base(nside, "RING").sht_info(),
    )
    np.testing.assert_allclose(map0, map1, atol=1e-5, rtol=1e-5)

    max_order = 2
    check_grads(hpp, (alm0r,), order=max_order, modes=("fwd", "rev"), eps=1.0)


#    map0 = (rng.random((ncomp, 12 * nside**2)) - 0.5).astype(dtype)
#    hpp_t = jax.linear_transpose(hpp)
#    check_grads(hpp_t, (map0,), order=max_order, modes=("fwd", "rev"), eps=1.0)
