# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2024 Max-Planck-Society

from functools import partial

import ducc0
import numpy as np

from .. import get_linear_call, load_kwargs

__all__ = ["c2c", "genuine_fht", "get_healpix_sht", "nalm", "get_wgridder"]


_r2cdict = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128),
}

_c2rdict = {
    np.dtype(np.complex64): np.dtype(np.float32),
    np.dtype(np.complex128): np.dtype(np.float64),
}


def _complextype(dtype):
    return _r2cdict[np.dtype(dtype)]


def _realtype(dtype):
    return _c2rdict[np.dtype(dtype)]


def _fht(out, args, kwargs_dump):
    (x,) = args
    kwargs = load_kwargs(kwargs_dump)
    batch_axes = kwargs.pop("batch_axes", None)
    axes = list(range(x.ndim))
    if batch_axes is not None:
        axes = [i for i in range(x.ndim) if i not in batch_axes[0]]
    orig_axis = kwargs.pop("axes", None)
    if orig_axis is not None:
        axes = [i for idx, i in enumerate(axes) if idx in orig_axis]
    ducc0.fft.genuine_fht(x, out=out[0], axes=axes, **kwargs)


def _fht_abstract(*args, **kwargs):
    (x,) = args
    batch_axes = kwargs.pop("batch_axes", None)
    out_ax = ()
    if batch_axes is not None and len(batch_axes[0]) > 0:
        out_ax = batch_axes[0][-1]
    return ((x.shape, x.dtype, out_ax),)


genuine_fht = get_linear_call(
    _fht, _fht, _fht_abstract, _fht_abstract, func_can_batch=True
)
genuine_fht.__doc__ = ducc0.fft.genuine_fht.__doc__


def _c2c(out, args, kwargs_dump):
    (x,) = args
    kwargs = load_kwargs(kwargs_dump)
    batch_axes = kwargs.pop("batch_axes", None)
    axes = list(range(x.ndim))
    if batch_axes is not None:
        axes = [i for i in range(x.ndim) if i not in batch_axes[0]]
    orig_axis = kwargs.pop("axes", None)
    if orig_axis is not None:
        axes = [i for idx, i in enumerate(axes) if idx in orig_axis]
    ducc0.fft.c2c(x, out=out[0], axes=axes, **kwargs)


def _c2c_abstract(*args, **kwargs):
    (x,) = args
    batch_axes = kwargs.pop("batch_axes", None)
    out_ax = ()
    if batch_axes is not None and len(batch_axes[0]) > 0:
        out_ax = batch_axes[0][-1]
    return ((x.shape, x.dtype, out_ax),)


c2c = get_linear_call(_c2c, _c2c, _c2c_abstract, _c2c_abstract, func_can_batch=True)
c2c.__doc__ = ducc0.fft.c2c.__doc__


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


def get_healpix_sht(nside, lmax, mmax, spin, nthreads=1):
    """Create a JAX primitive for the ducc0 SHT synthesis for HEALPix

    Parameters
    ----------
    nside : int
        Parameter of the HEALPix sphere.
    lmax, mmax : int
        Maximum l respectively m moment of the transformation (inclusive).
    spin : int
        Spin to use for the transfomration.
    nthreads : int
        Number of threads to use for the computation. If 0, use as many threads
        as there are hardware threads available on the system.

    Returns
    -------
    op : JAX primitive
        The Jax primitive of the SHT synthesis for HEALPix.
    """
    hpxparam = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()

    hpp = partial(
        _hp_sht,
        hpxparam["theta"],
        hpxparam["phi0"],
        hpxparam["nphi"],
        hpxparam["ringstart"],
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        nthreads=nthreads,
        nside=nside,
    )
    return hpp


def nalm(lmax, mmax):
    return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)


def _dirty2vis(out, args, kwargs_dump):
    uvw, freq, dirty = args
    kwargs = load_kwargs(kwargs_dump)
    kwargs.pop("npix_x")
    kwargs.pop("npix_y")
    ducc0.wgridder.experimental.dirty2vis(
        uvw=uvw, freq=freq, dirty=dirty, vis=out[0], **kwargs
    )


def _dirty2vis_abstract(*args, **kwargs):
    uvw, freq, dirty = args
    shape_out = (uvw.shape[0], freq.shape[0])
    dtype_out = _complextype(dirty.dtype)
    return ((shape_out, dtype_out),)


def _vis2dirty(out, args, kwargs_dump):
    uvw, freq, vis = args
    kwargs = load_kwargs(kwargs_dump)
    ducc0.wgridder.experimental.vis2dirty(
        uvw=uvw, freq=freq, vis=vis.conj(), dirty=out[0], **kwargs
    )


def _vis2dirty_abstract(*args, **kwargs):
    _, _, vis = args
    shape_out = (kwargs["npix_x"], kwargs["npix_y"])
    dtype_out = _realtype(vis.dtype)
    return ((shape_out, dtype_out),)


_wgridder = get_linear_call(
    _dirty2vis,
    _vis2dirty,
    _dirty2vis_abstract,
    _vis2dirty_abstract,
    first_n_args_fixed=2,
)


def get_wgridder(
    *,
    pixsize_x,
    pixsize_y,
    npix_x,
    npix_y,
    epsilon,
    do_wgridding,
    nthreads=1,
    flip_v=False,
    verbosity=0,
    **kwargs,
):
    """Create a JAX primitive for the ducc0 wgridder

    Parameters
    ----------
    pixsize_x, pixsize_y : float
        Size of the pixels in radian.
    npix_x, npix_y : int
        Number of pixels.
    epsilon : float
        Sets the required accuracy of the wgridder evaluation.
    nthreads : int
        Sets the number of threads used for evaluation. Default 1.
    flip_v : bool
        Whether or not to flip the v coordinate of the visibilities. Default
        `False`.
    verbosity : int
        Sets the verbosity of the wgridder. For 0 no print out, for >0 verbose
        output. Default 0.
    **kwargs : dict
        Additional forwarded to ducc wgridder.

    Returns
    -------
    op : JAX primitive evaluating the ducc wgridder.
        The Jax primitive has the
        signature `(uvw, freq, image)` with `uvw` being an (N, 3) array the uvw
        coordinates of the visibilities in meter, `freq`  a 1D array with the
        frequencies in Herz, and `image` a 2D arrays of shape `(npix_x, npix_y)`
        with the sky brightness in Jansky per Steradian.
    """
    wgridder = partial(
        _wgridder,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        npix_x=npix_x,
        npix_y=npix_y,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
        flip_v=flip_v,
        verbosity=verbosity,
        **kwargs,
    )
    return wgridder
