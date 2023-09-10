import os
import sys

import pytest
import torch
from matplotlib import pyplot as plt

import dynamiqs


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def add_default_imports(doctest_namespace):
    doctest_namespace['dq'] = dynamiqs
    doctest_namespace['plt'] = plt


def capture_stdout():
    sys.stdout = open(os.devnull, 'w')


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def add_utils_function(doctest_namespace):
    doctest_namespace['capture_stdout'] = capture_stdout


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def set_torch_print_options():
    torch.set_printoptions(precision=3, sci_mode=False)
