from importlib.metadata import version

from . import dark
from .mesolve import mesolve
from .sesolve import sesolve
from .smesolve import smesolve
from .result import Result
from .plots import *
from .time_tensor import TimeTensor, totime
from .utils import *
from ._utils import *  # todo: remove, dev purpose only
from . import solvers

# get version from pyproject.toml
__version__ = version(__package__)
