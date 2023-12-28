from importlib.metadata import version

from . import dark
from .mesolve import mesolve
from .plots import *
from .sesolve import sesolve
from .smesolve import smesolve
from .solvers.result import Result
from .time_tensor import TimeTensor, totime
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)
