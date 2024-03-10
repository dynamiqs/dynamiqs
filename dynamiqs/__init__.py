from importlib.metadata import version

from . import dark
from .mesolve import *
from .options import *
from .plots import *
from .result import *
from .sesolve import *
from .smesolve import *
from .time_array import *
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)
