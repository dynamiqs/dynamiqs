import pytest
import numpy as np
import jax.numpy as jnp

from dynamiqs import coherent, plot_wigner, todm, plot_wigner_mosaic

# todo : add comparison with analytical wigner for coherent states and cat states


class TestPlots:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.psis = [coherent(10, x) for x in np.linspace(0, 1, 10)]
        self.rhos = list(map(todm, self.psis))

        self.psis = jnp.asarray(self.psis)
        self.rhos = jnp.asarray(self.rhos)

    def test_plot_wigner_psi(self):
        plot_wigner(self.psis[0])

    def test_plot_wigner_psis(self):
        plot_wigner_mosaic(self.psis)

    def test_plot_wigner_rho(self):
        plot_wigner(self.rhos[0])

    def test_plot_wigner_rhos(self):
        plot_wigner_mosaic(self.rhos)
