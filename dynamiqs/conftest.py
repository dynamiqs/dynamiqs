from doctest import ELLIPSIS
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
import qutip
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
    namespace['qt'] = qutip


@pytest.fixture(scope='session', autouse=True)
def jax_set_printoptions():
    jnp.set_printoptions(precision=3, suppress=True)


@pytest.fixture(scope='session', autouse=True)
def mpl_params():
    dynamiqs.plot.mplstyle(dpi=150)
    # use a non-interactive backend for matplotlib, to avoid opening a display window
    matplotlib.use('Agg')


@pytest.fixture
def default_mpl_style():
    def set_default_mpl_style():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    return set_default_mpl_style


@pytest.fixture
def renderfig():
    def savefig_code(figname):
        filename = f'docs/figs_code/{figname}.png'
        plt.gcf().savefig(filename, bbox_inches='tight')
        plt.close()

    return savefig_code


@pytest.fixture
def rendergif():
    def savegif_code(gif, figname):
        filename = f'docs/figs_code/{figname}.gif'
        with Path(filename).open('wb') as f:
            f.write(gif.data)

    return savegif_code


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[DocTestParser(optionflags=ELLIPSIS), PythonCodeBlockParser()],
    patterns=['*.py'],
    excludes=['options.py'],
    setup=sybil_setup,
    fixtures=[
        'jax_set_printoptions',
        'mpl_params',
        'default_mpl_style',
        'renderfig',
        'rendergif',
    ],
).pytest()
