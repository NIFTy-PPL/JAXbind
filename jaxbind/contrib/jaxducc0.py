# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2024 Max-Planck-Society

from functools import partial

import ducc0
import numpy as np

from .. import get_linear_call, load_kwargs

_r2cdict = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128),
}


def _complextype(dtype):
    return _r2cdict[np.dtype(dtype)]


def _fht(out, args, kwargs_dump):
    (x,) = args
    kwargs = load_kwargs(kwargs_dump)
    ducc0.fft.genuine_fht(x, out=out[0], **kwargs)


def _fht_abstract(*args, **kwargs):
    (x,) = args
    return ((x.shape, x.dtype),)


genuine_fht = get_linear_call(_fht, _fht, _fht_abstract, _fht_abstract)


def _c2c(out, args, kwargs_dump):
    (x,) = args
    kwargs = load_kwargs(kwargs_dump)
    ducc0.fft.c2c(x, out=out[0], **kwargs)


def _c2c_abstract(*args, **kwargs):
    (x,) = args
    return ((x.shape, x.dtype),)


c2c = get_linear_call(_c2c, _c2c, _c2c_abstract, _c2c_abstract)


def _alm2realalm(alm, lmax, dtype, out=None):
    if out is None:
        out = np.empty((alm.shape[0], alm.shape[1] * 2 - lmax - 1), dtype=dtype)
    out[:, 0 : lmax + 1] = alm[:, 0 : lmax + 1].real
    out[:, lmax + 1 :] = alm[:, lmax + 1 :].view(dtype)
    out[:, lmax + 1 :] *= np.sqrt(2.0)
    return out


def _realalm2alm(alm, lmax, dtype, out=None):
    if out is None:
        out = np.empty((alm.shape[0], (alm.shape[1] + lmax + 1) // 2), dtype=dtype)
    out[:, 0 : lmax + 1] = alm[:, 0 : lmax + 1]
    out[:, lmax + 1 :] = alm[:, lmax + 1 :].view(dtype)
    out[:, lmax + 1 :] *= np.sqrt(2.0) / 2
    return out


def _healpix_sht(out, args, kwargs_dump):
    theta, phi0, nphi, ringstart, x = args
    kwargs = load_kwargs(kwargs_dump).copy()
    tmp = _realalm2alm(x, kwargs["lmax"], _complextype(x.dtype))
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


def _healpix_sht_T(out, args, kwargs_dump):
    theta, phi0, nphi, ringstart, x = args
    kwargs = load_kwargs(kwargs_dump).copy()
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
    _alm2realalm(tmp, kwargs["lmax"], x.dtype, out[0])


def _healpix_sht_abstract(*args, **kwargs):
    _, _, _, _, x = args
    spin = kwargs["spin"]
    ncomp = 1 if spin == 0 else 2
    shape_out = (ncomp, 12 * kwargs["nside"] ** 2)
    return ((shape_out, x.dtype),)


def _healpix_sht_abstract_T(*args, **kwargs):
    _, _, _, _, x = args
    spin = kwargs["spin"]
    ncomp = 1 if spin == 0 else 2
    lmax, mmax = kwargs["lmax"], kwargs["mmax"]
    nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
    nalm = nalm * 2 - lmax - 1
    shape_out = (ncomp, nalm)
    return ((shape_out, x.dtype),)


_hp_sht = get_linear_call(
    _healpix_sht,
    _healpix_sht_T,
    _healpix_sht_abstract,
    _healpix_sht_abstract_T,
    first_n_args_fixed=4,
)


def healpix_sht(x, *, spin, nthreads):
    # TODO: is there a sane default for spin?
    # TODO: infer lmmax and nside from shape of x?
    lmax, mmax = lmmax
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    hpxparam = base.sht_info()

    hpp = partial(
        _hp_sht,
        hpxparam["theta"],
        hpxparam["phi0"],
        hpxparam["nphi"],
        hpxparam["ringstart"],
        lmax=lmax,
        mmax=mmax,
        spin=spin,  # TODO:
        nthreads=nthreads,
        nside=nside,
    )
    return hpp(x)


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
