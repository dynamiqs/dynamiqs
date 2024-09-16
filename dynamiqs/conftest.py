from doctest import ELLIPSIS

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from sybil import Sybil
from sybil.parsers.doctest import DocTestParser
from sybil.parsers.markdown import PythonCodeBlockParser

import dynamiqs


def sybil_setup(namespace):
    namespace['dq'] = dynamiqs
    namespace['np'] = np
    namespace['plt'] = plt
    namespace['jax'] = jax
    namespace['jnp'] = jnp


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def _jax_set_printoptions():
    jnp.set_printoptions(precision=3, suppress=True)


# doctest fixture
@pytest.fixture(scope='session', autouse=True)
def _mplstyle():
    dynamiqs.plot.utils.mplstyle()


# doctest fixture
@pytest.fixture
def default_mpl_style():
    def set_default_mpl_style():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    return set_default_mpl_style


@pytest.fixture(scope='session', autouse=True)
def _mpl_backend():
    # use a non-interactive backend for matplotlib, to avoid opening a display window
    matplotlib.use('Agg')


# doctest fixture
@pytest.fixture
def renderfig():
    def savefig_code(figname):
        filename = f'docs/figs_code/{figname}.png'
        plt.gcf().savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    return savefig_code


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[DocTestParser(optionflags=ELLIPSIS), PythonCodeBlockParser()],
    patterns=['*.py'],
    setup=sybil_setup,
    fixtures=[
        '_jax_set_printoptions',
        '_mplstyle',
        'default_mpl_style',
        '_mpl_backend',
        'renderfig',
    ],
).pytest()
