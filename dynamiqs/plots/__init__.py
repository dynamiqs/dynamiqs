import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from .namespace import *
from .utils import *

cmap_colors = [
    (0.0, '#05527B'),
    (0.225, '#639DC1'),
    (0.5, '#FFFFFF'),
    (0.775, '#E27777'),
    (1.0, '#BF0C0C'),
]
dq_cmap = LinearSegmentedColormap.from_list('dq', cmap_colors)
matplotlib.colormaps.register(dq_cmap)
