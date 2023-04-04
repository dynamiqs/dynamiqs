from functools import partial

from tqdm import tqdm as std_tqdm

# define a default progress bar format
PBAR_FORMAT = '|{bar}| {percentage:4.1f}% - time {elapsed}/{remaining}'

# redefine tqdm with some default arguments
tqdm = partial(std_tqdm, bar_format=PBAR_FORMAT)
