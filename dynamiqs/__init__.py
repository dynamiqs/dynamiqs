from . import plot, random
from .hermitian_conjugate import *
from .integrators import *
from .options import *
from .qarrays import *
from .qarrays.layout import dense, dia
from .result import *
from .time_qarray import *
from .utils import *

__version__ = '0.3.3'

# set default matmul precision to 'highest'
set_matmul_precision('highest')
