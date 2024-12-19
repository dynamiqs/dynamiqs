from . import dark, plot, random
from .hermitian_conjugate import hc
from .integrators import *
from .options import *
from .qarrays import *
from .qarrays.layout import dense, dia
from .result import *
from .time_qarray import *
from .utils import *

__version__ = '0.2.3'

# set default matmul precision to 'highest'
set_matmul_precision('highest')
