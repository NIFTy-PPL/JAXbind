# SPDX-License-Identifier: BSD-2-Clause
# Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer

# Copyright(C) 2023, 2024 Max-Planck-Society


def load_kwargs(kwargs_dump, /):
    """Deserialize keyword arguments

    Parameters
    ----------
    kwargs_dump : numpy.ndarray
        Keyword arguments serialized by JAXbind.

    Returns
    -------
    dict : Dictionary containing the keyword arguments.

    Notes
    -----
    - The usage of `load_kwargs` is easiest to understand by looking at the
      demos.
    """
    import pickle
    import numpy as np

    return pickle.loads(np.ndarray.tobytes(kwargs_dump))
