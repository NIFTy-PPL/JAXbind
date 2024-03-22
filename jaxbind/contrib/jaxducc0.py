# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2024 Max-Planck-Society

import ducc0

from .. import get_linear_call, load_kwargs


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
