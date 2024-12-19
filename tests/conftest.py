import matplotlib
import pytest

from tests.order import TEST_INSTANT


@pytest.fixture(scope='session', autouse=True)
def _mpl_backend():
    # use a non-interactive backend for matplotlib, to avoid opening a display window
    matplotlib.use('Agg')


def pytest_collection_modifyitems(config, items):
    # Assign a default priority of INSTANT to unmarked tests
    for item in items:
        marker = item.get_closest_marker('run')
        item.priority = marker.args[0] if marker else TEST_INSTANT

    # Sort items based on their priority
    items.sort(key=lambda x: getattr(x, 'priority', TEST_INSTANT))
