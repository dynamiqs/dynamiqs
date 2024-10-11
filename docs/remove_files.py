import sys
from pathlib import Path


def remove_files(pattern: str):
    files = Path().glob(pattern)
    for file in files:
        Path.unlink(file)


if __name__ == '__main__':
    if sys.platform.startswith('win'):
        # Windows
        remove_files('figs_code/*.*')
    else:
        # Linux or macOS
        remove_files('figs_code/*.{png,gif}')
