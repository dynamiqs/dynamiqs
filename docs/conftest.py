from typing import Sequence

import jax.numpy as jnp
import pytest
from matplotlib import pyplot as plt
from sybil import Sybil
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

import dynamiqs


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def jax_set_printoptions():
    jnp.set_printoptions(precision=3, suppress=True)


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def mplstyle():
    dynamiqs.plots.utils.mplstyle()


# doctest fixture
@pytest.fixture()
def renderfig():
    def savefig_docs(figname):
        filename = f'docs/figs-docs/{figname}.png'
        plt.gcf().savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    return savefig_docs


# pycon code blocks parser
class PyconCodeBlockParser(PythonCodeBlockParser):
    def __init__(
        self, future_imports: Sequence[str] = (), doctest_optionflags: int = 0
    ) -> None:
        super().__init__(
            future_imports=future_imports, doctest_optionflags=doctest_optionflags
        )

        # override self.codeblock_parser
        self.codeblock_parser = self.codeblock_parser_class(
            'pycon', PythonEvaluator(future_imports)
        )


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        PyconCodeBlockParser(),
        SkipParser(),
    ],
    patterns=['*.md'],
    fixtures=['renderfig'],
).pytest()
