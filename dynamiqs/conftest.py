from doctest import ELLIPSIS

import pytest
import torch
from matplotlib import pyplot as plt
from sybil import Sybil
from sybil.parsers.doctest import DocTestParser

import dynamiqs


def sybil_setup(namespace):
    namespace['dq'] = dynamiqs
    namespace['plt'] = plt


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def torch_set_printoptions():
    torch.set_printoptions(precision=3, sci_mode=False)


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def mplstyle():
    dynamiqs.plots.utils.mplstyle(latex=False)


# doctest fixture
@pytest.fixture()
def render():
    def savefig_code(figname):
        filename = f'docs/figs-code/{figname}.png'
        plt.gcf().savefig(filename, bbox_inches='tight', dpi=300)

    return savefig_code


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS),
    ],
    patterns=['*.py'],
    setup=sybil_setup,
    fixtures=['render'],
).pytest()
