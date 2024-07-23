from importlib.metadata import version

from . import dark
from .integrators import mesolve, sesolve, smesolve
from .options import *
from .plots import *
from .result import *
from .time_array import *
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)

# set default matmul precision to 'highest'
set_matmul_precision('highest')
