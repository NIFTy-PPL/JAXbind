# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society


def load_kwargs(kwargs_dump, /):
    import pickle
    import numpy as np

    return pickle.loads(np.ndarray.tobytes(kwargs_dump))
