from importlib.metadata import version

from . import dark, plot, random
from .integrators import *
from .options import *
from .qarrays import *
from .qarrays.layout import dense, dia
from .result import *
from .time_array import *
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)

# set default matmul precision to 'highest'
set_matmul_precision('highest')
