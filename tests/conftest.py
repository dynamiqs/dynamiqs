import matplotlib
import pytest


@pytest.fixture(scope='session', autouse=True)
def _mpl_backend():
    # use a non-interactive backend for matplotlib, to avoid opening a display window
    matplotlib.use('Agg')
