import sys
from pathlib import Path

from jax.typing import ArrayLike


def remove_files(pattern: ArrayLike):
    files = Path.glob.glob(pattern)
    for file in files:
        Path.unlink(file)


if __name__ == '__main__':
    if sys.platform.startswith('win'):
        # Windows
        remove_files('docs/figs_code/*.*')
    else:
        # Linux or macOS
        remove_files('docs/figs_code/*.{png,gif}')
