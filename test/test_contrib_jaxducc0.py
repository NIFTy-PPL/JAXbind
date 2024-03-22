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
