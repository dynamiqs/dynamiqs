from importlib.metadata import version

from . import dark
from ._utils import *  # TODO: remove, dev purpose only
from .mesolve import mesolve
from .options import Options
from .plots import *
from .result import Result
from .sesolve import sesolve
from .time_array import *
from .time_array import TimeArray, TimeArrayLike
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)
