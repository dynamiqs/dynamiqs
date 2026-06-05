from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from dynamiqs.gradient import Forward, Gradient
from dynamiqs.method import Method

from .systems import System


class IntegratorTester:
    def _test_correctness(
        self,
        system: System,
        method: Method,
        *,
        ysave_atol: float = 1e-3,
        esave_rtol: float = 1e-3,
        esave_atol: float = 1e-4,
        **solver_kwargs,
    ):
        result = system.run(method, **solver_kwargs)

        # === test ysave
        true_ysave = system.states(system.tsave)
        errs = jnp.linalg.norm(
            true_ysave.to_jax() - result.states.to_jax(), axis=(-2, -1)
        )
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
        method: Method,
        gradient: Gradient,
        *,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        **solver_kwargs,
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
            res = system.run(method, gradient=gradient, params=params, **solver_kwargs)
            return system.loss_state(res.states[-1])

        # jax.grad uses reverse mode by default
        if isinstance(gradient, Forward):
            jax_grad, jax_jac = jax.jacfwd, jax.jacfwd
        else:
            jax_grad, jax_jac = jax.grad, jax.jacrev

        true_grads_ysave = system.grads_state(system.tsave[-1])
        grads_ysave = jax_grad(loss_ysave)(system.params_default)

        logging.warning(f'true_grads_ysave = {true_grads_ysave}')
        logging.warning(f'grads_ysave      = {grads_ysave}')

        assert_allclose(true_grads_ysave, grads_ysave)

        # === test gradients depending on final Esave
        def loss_Esave(params):
            res = system.run(method, gradient=gradient, params=params, **solver_kwargs)
            return system.loss_expect(res.expects[:, -1])

        true_grads_Esave = system.grads_expect(system.tsave[-1])
        grads_Esave = jax_jac(loss_Esave)(system.params_default)

        logging.warning(f'true_grads_Esave = {true_grads_Esave}')
        logging.warning(f'grads_Esave      = {grads_Esave}')

        assert_allclose(true_grads_ysave, grads_ysave)

    def _test_hessian(
        self, system, method, gradient, *, rtol: float = 1e-2, atol: float = 1e-2
    ):
        # index of the observable to test (HQubit has a single observable)
        i_expect = 0
        t_final = system.tsave[-1]

        # scalar loss: final-time loss of the chosen expectation value, as a
        # function of the system parameters
        def loss(params):
            result = system.run(method, gradient=gradient, params=params)
            # result.expects has shape (n_expect, ntsave)
            expect = result.expects[i_expect, -1]
            return system.loss_expect(expect)

        # computed Hessian wrt the parameters PyTree
        computed = jax.hessian(loss)(system.params_default)

        # analytical Hessian for this observable at the final time
        expected = system.hess_expect(t_final)

        # compare leaf-by-leaf over the Params PyTree
        for c_leaf, e_leaf in zip(
            jax.tree_util.tree_leaves(computed),
            jax.tree_util.tree_leaves(expected),
            strict=False,
        ):
            assert jnp.allclose(c_leaf, jnp.asarray(e_leaf), rtol=rtol, atol=atol)
