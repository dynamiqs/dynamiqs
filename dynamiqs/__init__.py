from importlib.metadata import version

from . import dark
from .mesolve import mesolve
from .plots import *
from .sesolve import sesolve
from .smesolve import smesolve
from dynamiqs.result import Result
from .time_tensor import TimeTensor, totime
from .utils import *
from ._utils import *  # todo: remove, dev purpose only

# get version from pyproject.toml
__version__ = version(__package__)
