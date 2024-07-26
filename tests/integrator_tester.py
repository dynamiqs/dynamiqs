from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from dynamiqs.gradient import Gradient
from dynamiqs.options import Options
from dynamiqs.solver import Solver

from .system import System


class IntegratorTester:
    def _test_correctness(
        self,
        system: System,
        solver: Solver,
        *,
        options: Options = Options(),  # noqa: B008
        ysave_atol: float = 1e-3,
        esave_rtol: float = 1e-3,
        esave_atol: float = 1e-4,
    ):
        result = system.run(solver, options=options)

        # === test ysave
        true_ysave = system.states(system.tsave)
        errs = jnp.linalg.norm(true_ysave - result.states, axis=(-2, -1))
        logging.warning(f'true_ysave = {true_ysave}')
        logging.warning(f'ysave      = {result.states}')
        assert jnp.all(errs <= ysave_atol)

        # === test Esave
        true_Esave = system.expects(system.tsave)
        logging.warning(f'true_Esave = {true_Esave}')
        logging.warning(f'Esave      = {result.expects}')
        assert jnp.allclose(
            true_Esave, result.expects, rtol=esave_rtol, atol=esave_atol
        )

    def test_correctness(self):
        pass

    def _test_gradient(
        self,
        system: System,
        solver: Solver,
        gradient: Gradient,
        *,
        options: Options = Options(),  # noqa: B008
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        def assert_allclose(pytree1, pytree2):
            # assert two pytrees are equal
            f = partial(jnp.allclose, rtol=rtol, atol=atol)
            allclose_tree = jtu.tree_map(f, pytree1, pytree2)
            # reduce the tree to a single boolean value
            all_true = jtu.tree_reduce(lambda x, y: x and y, allclose_tree, True)
            assert all_true, f'Pytrees are not close enough: \n{pytree1}\n{pytree2}'

        # === test gradients depending on final ysave
        def loss_ysave(params):
            res = system.run(solver, gradient=gradient, options=options, params=params)
            return system.loss_state(res.states[-1])

        true_grads_ysave = system.grads_state(system.tsave[-1])
        grads_ysave = jax.grad(loss_ysave)(system.params_default)

        logging.warning(f'true_grads_ysave = {true_grads_ysave}')
        logging.warning(f'grads_ysave      = {grads_ysave}')

        assert_allclose(true_grads_ysave, grads_ysave)

        # === test gradients depending on final Esave
        def loss_Esave(params):
            res = system.run(solver, gradient=gradient, options=options, params=params)
            return system.loss_expect(res.expects[:, -1])

        true_grads_Esave = system.grads_expect(system.tsave[-1])
        grads_Esave = jax.jacrev(loss_Esave)(system.params_default)

        logging.warning(f'true_grads_Esave = {true_grads_Esave}')
        logging.warning(f'grads_Esave      = {grads_Esave}')

        assert_allclose(true_grads_ysave, grads_ysave)
