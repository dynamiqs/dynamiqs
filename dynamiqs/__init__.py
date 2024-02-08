from importlib.metadata import version

from . import dark, rand
from ._utils import *  # todo: remove, dev purpose only
from .mesolve import mesolve
from .options import Options
from .plots import *
from .result import Result
from .sesolve import sesolve
from .time_array import TimeArray, TimeArrayLike, totime
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)
