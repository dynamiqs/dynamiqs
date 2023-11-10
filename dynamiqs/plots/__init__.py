import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from .misc import *
from .utils import *

cmap_colors = [
    (0.0, '#05527B'),
    (0.225, '#639DC1'),
    (0.5, '#FFFFFF'),
    (0.775, '#E27777'),
    (1.0, '#BF0C0C'),
]
dq_cmap = LinearSegmentedColormap.from_list('dq', cmap_colors)
cm.register_cmap(name='dq', cmap=dq_cmap)
