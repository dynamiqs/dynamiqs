import random

import jax
import jax.numpy as jnp

import dynamiqs as dq
from dynamiqs.sparse import *


class TestSparseDIA:
    # @pytest.fixture(autouse=True)
    # def _setup(self):
    #     # pass
    #     N = 4
    #     delta = 0.2
    #     eta = 1.5
    #     alpha0 = 0.3 - 0.5j
    #     time = 10.0
    #     num_tsave = 10

    #     self.tsave = jnp.linspace(0.0, time, num_tsave)
    #     self.a = dq.destroy(N)
    #     self.n = dq.number(N)

    def test_sesolve_constant_H(self, rtol=1e-05, atol=1e-08):
        # parameters
        N = 4
        delta = 0.2
        eta = 1.5
        alpha0 = 0.3 - 0.5j
        time = 10.0
        num_tsave = 10

        # === time evolution
        tsave = jnp.linspace(0.0, time, num_tsave)

        # === operators
        a = dq.destroy(N)
        n = dq.number(N)
        a_sparse = jax.jit(dq.to_sparse(a))
        n_sparse = dq.to_sparse(n)
        H0 = delta * n + eta * (a + dq.dag(a))
        H0_sparse = delta * n_sparse + eta * (a_sparse + dq.dag(a_sparse))

        psi0 = dq.coherent(N, alpha0)
        exp_ops = [dq.dag(a) @ a]

        result = dq.sesolve(H0, psi0, tsave, exp_ops=exp_ops)
        result_sparse = dq.sesolve(H0_sparse, psi0, tsave, exp_ops=exp_ops)

        assert jnp.allclose(result.states, result_sparse.states, rtol=rtol, atol=atol)

    def test_sesolve_modulated_H(self, rtol=1e-05, atol=1e-08):
        # parameters
        N = 4
        alpha0 = 0.3 - 0.5j
        time = 10.0
        num_tsave = 10

        # === time evolution
        tsave = jnp.linspace(0.0, time, num_tsave)

        # === operators
        a = dq.destroy(N)
        a_sparse = dq.to_sparse(a)

        psi0 = dq.coherent(N, alpha0)
        exp_ops = [dq.dag(a) @ a]

        f = lambda t: jnp.sin(t)

        H1 = dq.modulated(f, a + dq.dag(a))
        H1_sparse = dq.modulated(f, a_sparse + dq.dag(a_sparse))

        result = dq.sesolve(H1, psi0, tsave, exp_ops=exp_ops)
        result_sparse = dq.sesolve(H1_sparse, psi0, tsave, exp_ops=exp_ops)

        assert jnp.allclose(result.states, result_sparse.states, rtol=rtol, atol=atol)

    def test_sesolve_pwc_H(self, rtol=1e-05, atol=1e-08):
        # parameters
        N = 4
        alpha0 = 0.3 - 0.5j
        time = 10.0
        num_tsave = 10

        # === time evolution
        tsave = jnp.linspace(0.0, time, num_tsave)

        # === operators
        a = dq.destroy(N)
        a_sparse = dq.to_sparse(a)

        psi0 = dq.coherent(N, alpha0)
        exp_ops = [dq.dag(a) @ a]

        times = jnp.linspace(0.0, time, 4)
        values = jnp.arange(3)

        H2 = dq.pwc(times, values, a + dq.dag(a))
        H2_sparse = dq.pwc(times, values, a_sparse + dq.dag(a_sparse))

        result = dq.sesolve(H2, psi0, tsave, exp_ops=exp_ops)
        result_sparse = dq.sesolve(H2_sparse, psi0, tsave, exp_ops=exp_ops)

        assert jnp.allclose(result.states, result_sparse.states, rtol=rtol, atol=atol)

    def test_mesolve_constant_jump(self, rtol=1e-05, atol=1e-06):
        # parameters
        N = 4
        delta = 0.2
        eta = 1.5
        alpha0 = 0.3 - 0.5j
        time = 10.0
        num_tsave = 10
        kappa = 0.1

        # === time evolution
        tsave = jnp.linspace(0.0, time, num_tsave)

        # === operators
        a = dq.destroy(N)
        n = dq.number(N)
        a_sparse = dq.to_sparse(a)
        n_sparse = dq.to_sparse(n)
        H0 = delta * n + eta * (a + dq.dag(a))
        H0_sparse = delta * n_sparse + eta * (a_sparse + dq.dag(a_sparse))
        L0 = kappa * a
        L0_sparse = kappa * a_sparse

        psi0 = dq.coherent(N, alpha0)
        exp_ops = [dq.dag(a) @ a]

        result = dq.mesolve(H0, [L0], psi0, tsave, exp_ops=exp_ops)
        result_sparse = dq.mesolve(H0_sparse, [L0_sparse], psi0, tsave, exp_ops=exp_ops)

        assert jnp.allclose(result.states, result_sparse.states, rtol=rtol, atol=atol)

    def test_mesolve_modulated_jump(self, rtol=1e-05, atol=1e-06):
        # parameters
        N = 4
        delta = 0.2
        eta = 1.5
        alpha0 = 0.3 - 0.5j
        time = 10.0
        num_tsave = 10
        kappa = 0.1

        # === time evolution
        tsave = jnp.linspace(0.0, time, num_tsave)

        f = lambda t: jnp.sin(t)

        # === operators
        a = dq.destroy(N)
        n = dq.number(N)
        a_sparse = dq.to_sparse(a)
        n_sparse = dq.to_sparse(n)
        H0 = delta * n + eta * (a + dq.dag(a))
        H0_sparse = delta * n_sparse + eta * (a_sparse + dq.dag(a_sparse))
        L1 = dq.modulated(f, kappa * a)
        L1_sparse = dq.modulated(f, kappa * a_sparse)

        psi0 = dq.coherent(N, alpha0)
        exp_ops = [dq.dag(a) @ a]

        result = dq.mesolve(H0, [L1], psi0, tsave, exp_ops=exp_ops)
        result_sparse = dq.mesolve(H0_sparse, [L1_sparse], psi0, tsave, exp_ops=exp_ops)

        assert jnp.allclose(result.states, result_sparse.states, rtol=rtol, atol=atol)

    def test_mesolve_jump_list(self, rtol=1e-05, atol=1e-06):
        # parameters
        N = 4
        delta = 0.2
        eta = 1.5
        alpha0 = 0.3 - 0.5j
        time = 10.0
        num_tsave = 10
        kappa = 0.1

        # === time evolution
        tsave = jnp.linspace(0.0, time, num_tsave)

        f = lambda t: jnp.sin(t)

        # === operators
        a = dq.destroy(N)
        n = dq.number(N)
        a_sparse = dq.to_sparse(a)
        n_sparse = dq.to_sparse(n)
        H0 = delta * n + eta * (a + dq.dag(a))
        H0_sparse = delta * n_sparse + eta * (a_sparse + dq.dag(a_sparse))
        L0 = kappa * a
        L0_sparse = kappa * a_sparse
        L1 = dq.modulated(f, kappa * a)
        L1_sparse = dq.modulated(f, kappa * a_sparse)
        L2 = [L0, L1]
        L2_sparse = [L0_sparse, L1_sparse]

        psi0 = dq.coherent(N, alpha0)
        exp_ops = [dq.dag(a) @ a]

        result = dq.mesolve(H0, L2, psi0, tsave, exp_ops=exp_ops)
        result_sparse = dq.mesolve(H0_sparse, L2_sparse, psi0, tsave, exp_ops=exp_ops)

        assert jnp.allclose(result.states, result_sparse.states, rtol=rtol, atol=atol)

    def test_matmul(self, rtol=1e-05, atol=1e-08):
        N = 4
        matrixA = dq.destroy(N)
        matrixB = dq.number(N)

        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = (sparseA @ sparseB).to_dense()
        out_dia_dense = sparseA @ matrixB
        out_dense_dia = matrixA @ sparseB
        out_dense_dense = matrixA @ matrixB

        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_add(self, rtol=1e-05, atol=1e-08):
        N = 4
        matrixA = dq.destroy(N)
        matrixB = dq.number(N)

        sparseA = to_sparse(matrixA)
        sparseB = to_sparse(matrixB)

        out_dia_dia = (sparseA + sparseB).to_dense()
        out_dia_dense = sparseA + matrixB
        out_dense_dia = matrixA + sparseB
        out_dense_dense = matrixA + matrixB

        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_transform(self, rtol=1e-05, atol=1e-08):
        N = 4
        key = jax.random.PRNGKey(1)
        matrix = jax.random.normal(key, (N, N))
        assert jnp.allclose(matrix, to_sparse(matrix).to_dense(), rtol=rtol, atol=atol)

    def test_mul(self, rtol=1e-05, atol=1e-08):
        N = 4
        key = jax.random.PRNGKey(1)
        matrix = jax.random.normal(key, (N, N))
        random_float = random.uniform(1.0, 10.0)

        sparse = to_sparse(matrix)

        out_dense_left = random_float * matrix
        out_dense_right = matrix * random_float
        out_dia_left = (random_float * sparse).to_dense()
        out_dia_right = (sparse * random_float).to_dense()

        assert jnp.allclose(out_dense_left, out_dense_right, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_left, out_dia_left, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_left, out_dia_right, rtol=rtol, atol=atol)
