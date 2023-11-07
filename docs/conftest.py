from doctest import ELLIPSIS
from sybil import Sybil
from sybil.parsers.myst import DocTestDirectiveParser, PythonCodeBlockParser
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


# pytest configuration
pytest_collect_file = Sybil(
    parsers=[
        DocTestDirectiveParser(optionflags=ELLIPSIS),
        PythonCodeBlockParser(),
    ],
    patterns=['*.md'],
).pytest()
