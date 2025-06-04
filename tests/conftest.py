import matplotlib
import pytest

from .order import TEST_INSTANT


@pytest.fixture(scope='session', autouse=True)
def mpl_backend():
    # use a non-interactive backend for matplotlib, to avoid opening a display window
    matplotlib.use('Agg')


def pytest_collection_modifyitems(config, items):
    # Assign a default priority of INSTANT to unmarked tests
    for item in items:
        marker = item.get_closest_marker('run')
        if not marker:
            continue

        order = marker.kwargs.get('order', None)
        item.priority = order or TEST_INSTANT

    # Sort items based on their priority
    items.sort(key=lambda x: getattr(x, 'priority', TEST_INSTANT))
