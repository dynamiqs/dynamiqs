from doctest import ELLIPSIS

import pytest
import torch
from matplotlib import pyplot as plt
from sybil import Sybil
from sybil.parsers.myst import DocTestDirectiveParser, PythonCodeBlockParser

import dynamiqs


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def add_dq(doctest_namespace):
    doctest_namespace['dq'] = dynamiqs
    doctest_namespace['plt'] = plt


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def set_torch_print_options():
    torch.set_printoptions(precision=3, sci_mode=False)


# sybil configuration (better doctest for the documentation)
pytest_collect_file = Sybil(
    parsers=[
        DocTestDirectiveParser(optionflags=ELLIPSIS),
        PythonCodeBlockParser(),
    ],
    patterns=['*.md'],
).pytest()
