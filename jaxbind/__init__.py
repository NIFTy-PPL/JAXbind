import importlib.metadata

__version__ = importlib.metadata.version("jaxbind")

from .jaxbind import get_linear_call, get_nonlinear_call
from .misc import load_kwargs
