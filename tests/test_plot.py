import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import pyplot as plt

from dynamiqs import coherent, plot, todm
from dynamiqs.utils.quantum_utils.wigner import _diag_element, wigner

# TODO : add comparison with analytical wigner for coherent states and cat states


class TestPlots:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.psis = [coherent(10, x) for x in np.linspace(0, 1, 10)]
        self.rhos = list(map(todm, self.psis))

        self.psis = jnp.asarray(self.psis)
        self.rhos = jnp.asarray(self.rhos)

    @pytest.fixture(autouse=True)
    def _teardown(self):
        # once the test is finished, pytest will go back here and run the code after
        # the yield statement
        yield
        plt.close('all')

    def test_plot_wigner_psi(self):
        plot.wigner(self.psis[0])

    def test_plot_wigner_psis(self):
        plot.wigner_mosaic(self.psis)

    def test_plot_wigner_rho(self):
        plot.wigner(self.rhos[0])

    def test_plot_wigner_rhos(self):
        plot.wigner_mosaic(self.rhos)

    def test_diag_element(self):
        mat = jnp.arange(25).reshape(5, 5)
        for diag in range(-4, 5):
            diag_len = 5 - abs(diag)
            for element in range(-diag_len + 1, diag_len):
                x = _diag_element(mat, diag, element)
                y = np.diag(mat, diag)[element]
                err_msg = (
                    f'Failed for diag = {diag}, element = {element}, expected "{y}",'
                    f' got "{x}"'
                )
                assert x == y, err_msg

    def test_wigner_psi_xvec_yvec(self):
        vec = np.linspace(-5, 5, 10)
        wigner(self.psis[0], xvec=vec)
        wigner(self.psis[0], yvec=vec)
        wigner(self.psis[0], xvec=vec, yvec=vec)
