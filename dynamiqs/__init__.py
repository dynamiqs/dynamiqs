from . import dark, plot, random
from .integrators import *
from .options import *
from .result import *
from .time_array import *
from .utils import *

__version__ = '0.2.0'

# set default matmul precision to 'highest'
set_matmul_precision('highest')
