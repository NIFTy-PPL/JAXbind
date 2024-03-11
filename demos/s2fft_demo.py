# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2024 Max-Planck-Society

import jax

from jax import config
config.update("jax_enable_x64", True)

import ducc0
import numpy as np
import s2fft
import jaxbind
from time import time
from functools import partial

r2cdict = {np.dtype(np.float32) : np.dtype(np.complex64),
           np.dtype(np.float64) : np.dtype(np.complex128)}

c2rdict = {np.dtype(np.complex64) : np.dtype(np.float32),
           np.dtype(np.complex128) : np.dtype(np.float64)}

def isrealtype(dtype):
    return np.dtype(dtype) in r2cdict

def iscomplextype(dtype):
    return np.dtype(dtype) in c2rdict

def isvalidtype(dtype):
    return isrealtype(dtype) or iscomplextype(dtype)

def realtype(dtype):
    return c2rdict[np.dtype(dtype)]

def complextype(dtype):
    return r2cdict[np.dtype(dtype)]

def ssht_operator(L, sampling, spin, reality, nthreads, nside=0):
    def get_config(alm, map, state):
        if not iscomplextype(alm.dtype):
            raise RuntimeError("invalid alm data type")
        if not isvalidtype(map.dtype):
            raise RuntimeError("invalid map data type")
        reality = state["reality"]
        L = state["L"]
        spin = state["spin"]
        nthreads = state["nthreads"]
        sampling = state["sampling"]
        hpxparam={}
        if sampling == "mw":
            geometry = "MW"
            mapshape = (L, 2*L-1)
        elif sampling == "mwss":
            geometry = "CC"
            mapshape = (L+1, 2*L)
        elif sampling == "dh":
            geometry = "F1"
            mapshape = (2*L, 2*L-1)
        elif sampling == "healpix":
            geometry = "healpix"
            mapshape = (12*state["nside"]**2,)
            base = ducc0.healpix.Healpix_Base(state["nside"], "RING")
            hpxparam = base.sht_info()
        if map.shape!=mapshape:
            raise RuntimeError("bad map shape")
        if alm.shape!=(L,2*L-1):
            raise RuntimeError("bad alm shape")
        return L, spin, nthreads, reality, geometry, mapshape, hpxparam

    def ssht_T(out_, args, kwargs_dump):
        out, = out_
        inp, = args
        state = jaxbind.load_kwargs(kwargs_dump)
        L, spin, nthreads, reality, geometry, mapshape, hpxparam = get_config(out, inp, state)

        if spin==0:
            inp2 = inp.reshape((1, *mapshape))
            inp2 = inp2.conj()
            tmp = np.empty((1 if reality else 2, (L*(L+1))//2), out.dtype)
            if sampling == "healpix":
                ducc0.sht.adjoint_synthesis(lmax=L-1, spin=np.abs(spin), map=inp2.real, alm=tmp[0:1], nthreads=nthreads, **hpxparam)
                if not reality:
                    ducc0.sht.adjoint_synthesis(lmax=L-1, spin=np.abs(spin), map=inp2.imag, alm=tmp[1:2], nthreads=nthreads, **hpxparam)
            else:
                ducc0.sht.adjoint_synthesis_2d(
                    lmax=L-1, spin=np.abs(spin),
                    map=inp2.real, alm=tmp[0:1], nthreads=nthreads,
                    geometry=geometry)
                if not reality:
                    ducc0.sht.adjoint_synthesis_2d(
                        lmax=L-1, spin=np.abs(spin),
                        map=inp2.imag, alm=tmp[1:2], nthreads=nthreads,
                        geometry=geometry)
        else:
            if spin > 0:
                tmap = inp.view(realtype(out.dtype)).reshape((*mapshape,2))
                tmap = np.moveaxis(tmap,-1,0)
            else:
                tmap = np.empty((2, inp.shape[0], inp.shape[1]), dtype=realtype(inp.dtype))
                tmap[0] = -inp.real
                tmap[1] = inp.imag
            tmap = tmap.copy()
            tmap[1] *= -1
            if sampling == "healpix":
                tmp = ducc0.sht.adjoint_synthesis(lmax=L-1, spin=np.abs(spin), map=tmap, nthreads=nthreads, **hpxparam)
            else:
                tmp = ducc0.sht.adjoint_synthesis_2d(
                    lmax=L-1, spin=np.abs(spin),
                    map=tmap, nthreads=nthreads,
                    geometry=geometry)
            if spin > 0:
                tmp *= -1
        ducc0.sht.experimental.alm2flm(tmp, spin, out)
        out.imag *= -1

    def ssht(out_, args, kwargs_dump):
        out, = out_
        inp, = args
        state = jaxbind.load_kwargs(kwargs_dump)
        L, spin, nthreads, reality, geometry, mapshape, hpxparam = get_config(inp, out, state)

        tmp = ducc0.sht.experimental.flm2alm(inp, spin, real=reality)
        out2 = out.view(realtype(inp.dtype))
        out2 = out2.reshape((*mapshape, 1 if reality else 2))
        out2 = np.moveaxis(out2, -1, 0)
        if spin==0:
            if sampling == "healpix":
                ducc0.sht.synthesis(lmax=L-1, spin=np.abs(spin), alm=tmp[0:1], map=out2[0:1], nthreads=nthreads, **hpxparam)
                if not reality:
                    ducc0.sht.synthesis(lmax=L-1, spin=np.abs(spin), alm=tmp[1:2], map=out2[1:2], nthreads=nthreads, **hpxparam)
            else:
                ducc0.sht.synthesis_2d(
                    lmax=L-1, spin=np.abs(spin),
                    alm=tmp[0:1], map=out2[0:1], nthreads=nthreads, geometry=geometry)
                if not reality:
                    ducc0.sht.synthesis_2d(
                        lmax=L-1, spin=np.abs(spin),
                        alm=tmp[1:2], map = out2[1:2], nthreads=nthreads, geometry=geometry)
        else:
            if sampling == "healpix":
                ducc0.sht.synthesis(lmax=L-1, spin=np.abs(spin), alm=tmp, map=out2, nthreads=nthreads, **hpxparam)
            else:
                ducc0.sht.synthesis_2d(
                    lmax=L-1, spin=np.abs(spin),
                    alm=tmp, map=out2, nthreads=nthreads,
                    geometry=geometry)
            if spin > 0:
                out *= -1
            else:
                out.real *= -1

    def ssht_abstract(*args, **state):
        shape_in, dtype_in = args[0].shape, args[0].dtype
        if not isvalidtype(dtype_in):
            raise RuntimeError("invalid data type")
        spin = state["spin"]
        L = state["L"]
        reality = state["reality"]
        sampling = state["sampling"]
        if sampling == "mw":
            mapshape = (L, 2*L-1)
        elif sampling == "mwss":
            mapshape = (L+1, 2*L)
        elif sampling == "dh":
            mapshape = (2*L, 2*L-1)
        elif sampling == "healpix":
            mapshape = (12*state["nside"]**2,)
        
        if shape_in != (L, 2*L-1):
            raise RuntimeError("bad flm shape")
        shape_out = mapshape
        dtype_out = dtype_in if not reality else realtype(dtype_in)
        return ((shape_out, dtype_out),)

    def ssht_abstract_T(*args, **state):
        shape_in, dtype_in = args[0].shape, args[0].dtype
        if not isvalidtype(dtype_in):
            raise RuntimeError("invalid data type")
        spin = state["spin"]
        L = state["L"]
        reality = state["reality"]
        sampling = state["sampling"]
        if sampling == "mw":
            mapshape = (L, 2*L-1)
        elif sampling == "mwss":
            mapshape = (L+1, 2*L)
        elif sampling == "dh":
            mapshape = (2*L, 2*L-1)
        elif sampling == "healpix":
            mapshape = (12*state["nside"]**2,)
        
        if shape_in != mapshape:
            raise RuntimeError("bad map shape")
        if isrealtype(dtype_in):
            if (not reality):
                raise RuntimeError("real map provided, but complex map expected")
            dtype_out = complextype(dtype_in)
        else:
            dtype_out = dtype_in
        shape_out = (L, 2*L-1)
        return ((shape_out, dtype_out),)

    L = int(L)
    if (L<=0):
        raise ValueError("L must be positive")
    spin = int(spin)
    reality = bool(reality)
    if spin != 0 and reality:
        raise ValueError("if spin is nonzero, reality must be False")
    sampling = str(sampling)
    if sampling!="mw" and sampling!="dh" and sampling!="mwss" and sampling!="healpix":
        raise ValueError("unsupported sampling type")
    op = jaxbind.get_linear_call(
        ssht, ssht_T, ssht_abstract, ssht_abstract_T,
        func_can_batch=False)
    return partial(op,
        L=L, spin=spin,
        sampling=sampling,
        reality=reality,
        nthreads=int(nthreads),
        nside=int(nside))


from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.test_util import check_grads


L=32
lmax = L-1
spin=0
nthreads=1
reality=True
rng = np.random.default_rng(42)
sampling="healpix"
nside=L//2

flm = s2fft.utils.signal_generator.generate_flm(rng, L, 0, spin, reality)

t0 = time()
op = ssht_operator(L=L, sampling=sampling, spin=spin, nthreads=nthreads, reality=reality, nside=nside)

map0 = np.array(op(flm)[0])
print("ducc time:", time()-t0)

def func2(flm):
    f = op(flm)[0]
    return jnp.sum(jnp.abs(f)**2)
t0 = time()
check_grads(func2, (flm,), order=3, modes=('fwd','rev'))
print("check_grad ducc time (fwd+rev):", time()-t0)

t0 = time()
precomps = s2fft.generate_precomputes_jax(L, forward=False, spin=spin, sampling=sampling, nside=nside)
print("precomp time:", time()-t0)

t0 = time()
map1 = np.array(s2fft.inverse(flm, L, spin=spin, reality=reality, sampling=sampling, precomps=precomps, method="jax", nside=nside))
print("s2fft time:", time()-t0)
print(ducc0.misc.l2error(map0, map1))

def func1(flm):
    f = s2fft.inverse(flm, L, reality=reality, precomps=precomps, spin=spin, sampling=sampling,nside=nside, method="jax")
    return jnp.sum(jnp.abs(f)**2)
t0 = time()
check_grads(func1, (flm,), order=3, modes=('rev'))
print("check_grad s2fft time (rev):", time()-t0)
