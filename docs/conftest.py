import pytest
import torch

import dynamiqs


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def add_dq(doctest_namespace):
    doctest_namespace['dq'] = dynamiqs


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def set_torch_print_options():
    torch.set_printoptions(precision=3, sci_mode=False)
