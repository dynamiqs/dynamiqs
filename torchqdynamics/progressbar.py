from functools import partial

from tqdm.auto import tqdm as std_tqdm

# Define a default progress bar format
PBAR_FORMAT = '{desc}: {percentage:3.1f}% - Time {elapsed}/{remaining}'

# Redefine tqdm with some default arguments
tqdm = partial(std_tqdm, bar_format=PBAR_FORMAT)
