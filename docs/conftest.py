from doctest import ELLIPSIS

import pytest
import torch
from matplotlib import pyplot as plt
from sybil import Sybil
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.myst import (
    CodeBlockParser,
    DocTestDirectiveParser,
    PythonCodeBlockParser,
    SkipParser,
)

import dynamiqs


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
def renderfig():
    def savefig_docs(figname):
        filename = f'docs/figs-docs/{figname}.png'
        plt.gcf().savefig(filename, bbox_inches='tight', dpi=300)

    return savefig_docs


# pycon code blocks parser
class PyconCodeBlockParser(PythonCodeBlockParser):
    def __init__(self):
        super().__init__()
        self.codeblock_parser = CodeBlockParser('pycon', PythonEvaluator())


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[
        DocTestDirectiveParser(optionflags=ELLIPSIS),
        PythonCodeBlockParser(),
        PyconCodeBlockParser(),
        SkipParser(),
    ],
    patterns=['*.md'],
    fixtures=['renderfig'],
).pytest()
