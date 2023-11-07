import pytest
import torch
from matplotlib import pyplot as plt

import dynamiqs


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def add_dq(doctest_namespace):
    doctest_namespace['dq'] = dynamiqs
    doctest_namespace['plt'] = plt


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def torch_set_printoptions():
    torch.set_printoptions(precision=3, sci_mode=False)


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def mplstyle():
    dynamiqs.plots.utils.mplstyle()
