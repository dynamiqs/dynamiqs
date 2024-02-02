from importlib.metadata import version

from . import dark, rand
from ._utils import *  # todo: remove, dev purpose only
from .mesolve import mesolve
from .plots import *
from .result import Result
from .sesolve import sesolve
from .smesolve import smesolve
from .time_array import TimeArray, totime
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)
