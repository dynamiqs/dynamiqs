import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from .namespace import *
from .utils import *

# diverging
cmap_colors = [
    (0.0, '#05527B'),
    (0.225, '#639DC1'),
    (0.5, '#FFFFFF'),
    (0.775, '#E27777'),
    (1.0, '#BF0C0C'),
]
cmap = LinearSegmentedColormap.from_list('dq', cmap_colors)
matplotlib.colormaps.register(cmap)

# cyclic colormap
cmap_colors = [
    (0.0, '#07689D'),
    (0.25, '#AC98AB'),
    (0.5, '#C62525'),
    (0.75, '#5E1A5B'),
    (1.0, '#07689D'),
]
cmap = LinearSegmentedColormap.from_list('dq_cyclic', cmap_colors)
matplotlib.colormaps.register(cmap)
