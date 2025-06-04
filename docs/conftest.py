from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import pytest
from matplotlib import pyplot as plt
from sybil import Sybil
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

import dynamiqs


@pytest.fixture(scope='session', autouse=True)
def jax_set_printoptions():
    jnp.set_printoptions(precision=3, suppress=True)


@pytest.fixture(scope='session', autouse=True)
def mpl_params():
    dynamiqs.plot.mplstyle(dpi=150)
    # use a non-interactive backend for matplotlib, to avoid opening a display window
    matplotlib.use('Agg')


@pytest.fixture
def renderfig():
    def savefig_docs(figname):
        filename = f'docs/figs_docs/{figname}.png'
        plt.gcf().savefig(filename, bbox_inches='tight')
        plt.close()

    return savefig_docs


@pytest.fixture
def rendergif():
    def savegif_docs(gif, figname):
        filename = f'docs/figs_docs/{figname}.gif'
        with Path(filename).open('wb') as f:
            f.write(gif.data)

    return savegif_docs


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
    parsers=[PythonCodeBlockParser(), PyconCodeBlockParser(), SkipParser()],
    patterns=['*.md'],
    fixtures=['jax_set_printoptions', 'mpl_params', 'renderfig', 'rendergif'],
).pytest()
