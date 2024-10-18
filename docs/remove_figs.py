import sys
from itertools import chain
from pathlib import Path

if __name__ == '__main__':
    path = Path(sys.argv[1])  # get path from first command-line argument
    figs_path = chain(path.glob('*.png'), path.glob('*.gif'))
    for fig_path in figs_path:
        fig_path.unlink()
