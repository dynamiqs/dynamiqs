# Composite qarrays

The operators of a system made of several subsystems, such as coupled cavities or a qubit chain, are sums of tensor products of small operators. Dynamiqs can store such an operator as its factors, in a **composite qarray**, instead of its full matrix. This tutorial shows how to create composite qarrays, which operations keep them composite, and how solvers use them.

```python
import jax.numpy as jnp
import dynamiqs as dq
```

## Storing an operator as its factors

A composite qarray stores an operator on $N$ subsystems as a sum of **terms**. Each term is a coefficient $c_j$ times a tensor product of one factor $A_{j,k}$ per subsystem:
$$
    H = \sum_j c_j\, A_{j,1} \otimes A_{j,2} \otimes \dots \otimes A_{j,N}.
$$
Each factor is a matrix of size $n_k\times n_k$, where $n_k$ is the dimension of subsystem $k$, and the full $n\times n$ matrix is never built. Solvers apply each term to the state one subsystem at a time. For two subsystems and a density matrix, this costs $O(n^2(n_1+n_2))$ instead of $O(n^3)$ with the full dense matrix.

!!! Example "Example for two coupled cavities"
    Two cavities coupled by a beam-splitter interaction have the Hamiltonian
    $$
        H = \omega_a\, a^\dag a\otimes I + \omega_b\, I\otimes b^\dag b + g\,(a^\dag\otimes b + a\otimes b^\dag).
    $$
    As a composite qarray, its four terms hold eight small matrices instead of one matrix of size $n_1n_2\times n_1n_2$.

## Creating composite qarrays

The `composite` argument of [`dq.tensor()`][dynamiqs.tensor] returns a tensor product of operators as a single composite term:

```pycon
>>> dq.tensor(dq.sigmax(), dq.sigmaz(), composite=True)
CompositeQArray: shape=(4, 4), dims=(2, 2), dtype=complex64, layout=dia, n_terms=1
  term[0]: coeff=1.0, op_dims=((2,), (2,))
```

The printout lists each term with its coefficient and the dimensions of its factors (`op_dims`). The operator functions that take several dimensions also accept `composite`: [`dq.destroy()`][dynamiqs.destroy], [`dq.create()`][dynamiqs.create], [`dq.number()`][dynamiqs.number], [`dq.fdestroy()`][dynamiqs.fdestroy], [`dq.fcreate()`][dynamiqs.fcreate] and [`dq.hadamard()`][dynamiqs.hadamard]. Sums and products of their results stay composite:

```pycon
>>> a, b = dq.destroy(3, 4, layout=dq.dense, composite=True)
>>> H = a.dag() @ a + 2.0 * b.dag() @ b + 0.5 * (a.dag() @ b + a @ b.dag())
>>> H
CompositeQArray: shape=(12, 12), dims=(3, 4), dtype=complex64, layout=dense, n_terms=4
  term[0]: coeff=1.0, op_dims=((3,), (4,))
  term[1]: coeff=2.0, op_dims=((3,), (4,))
  term[2]: coeff=0.5, op_dims=((3,), (4,))
  term[3]: coeff=0.5, op_dims=((3,), (4,))
```

[`dq.set_composite()`][dynamiqs.set_composite] changes the default of all these functions for the rest of the program:

```pycon
>>> dq.set_composite(True)
>>> dq.number(3, 4)[0]
CompositeQArray: shape=(12, 12), dims=(3, 4), dtype=complex64, layout=dia, n_terms=1
  term[0]: coeff=1.0, op_dims=((3,), (4,))
>>> dq.set_composite(False)  # back to the default
```

The `&` operator ignores this setting, and returns the full matrix of two regular qarrays. With a composite operand, it adds a factor to each term:

```pycon
>>> (H & dq.eye(2, layout=dq.dense)).dims
(3, 4, 2)
```

States are never composite, because a simulated state fills its full matrix after one step. [`dq.tensor()`][dynamiqs.tensor] always returns full kets and bras, and solvers convert a composite initial density matrix to its full matrix.

## Operations on composite qarrays

A composite qarray behaves exactly like its full matrix: operations return the same values, shape, `dims`, data type and layout, and raise the same errors. Most operations work term by term and return a composite qarray. The others return a regular qarray:

| Operation | Result |
| --- | --- |
| `x + y`, `x - y`, with `y` a composite qarray or an operator | composite |
| `x @ y`, `x & y`, `x.elmul(y)`, with `y` a composite qarray | composite |
| `x * c`, `x / c` with a scalar `c`, `-x`, `x.dag()`, `x.conj()`, `x.mT` | composite |
| `x @ y`, `y @ x`, with `y` a regular qarray or array, e.g. a state | regular, without building `x` |
| `x.elmul(y)` with a regular `y`, `x.expm()`, `x.powm(n)`, `x.ptrace(*keep)` | regular |
| `x.to_jax()`, `x.to_numpy()`, `x.to_qutip()` | full matrix |

The product of two composite qarrays pairs every term of one with every term of the other, so the number of terms multiplies:

```pycon
>>> Hab = a.dag() @ a + b.dag() @ b
>>> Hab @ Hab
CompositeQArray: shape=(12, 12), dims=(3, 4), dtype=complex64, layout=dense, n_terms=4
  term[0]: coeff=1.0, op_dims=((3,), (4,))
  term[1]: coeff=1.0, op_dims=((3,), (4,))
  term[2]: coeff=1.0, op_dims=((3,), (4,))
  term[3]: coeff=1.0, op_dims=((3,), (4,))
```

To keep the number of terms low, write a Hamiltonian as a sum of simple terms rather than as products of sums.

## Layouts of composite qarrays

The layout of a composite qarray is that of its factors, `dense` or `dia` (see [Quantum arrays and layouts](qarrays-and-layouts.md#dense-and-sparse-diagonal-layouts)). Mixing the two converts all factors to `dense`. The `.asdense()` and `.assparsedia()` methods convert the factors and keep the qarray composite:

```pycon
>>> H.layout, H.assparsedia().layout
(dense, dia)
```

Composite qarrays pay off with `dense` factors, see the [Performance](performance.md#store-the-operators-of-several-subsystems-as-composite-qarrays) tutorial.

## Simulating with composite qarrays

Any solver accepts composite qarrays, for the Hamiltonian, the jump operators and the expectation operators, constant or time-dependent. Let us simulate the two coupled cavities above, with losses:

```pycon
>>> jump_ops = [0.5 * a, 0.5 * b]
>>> psi0 = dq.tensor(dq.coherent(3, 1.0), dq.fock(4, 0))
>>> tsave = jnp.linspace(0.0, 1.0, 3)
>>> exp_ops = [a.dag() @ a, b.dag() @ b]
>>> result = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, progress_meter=False)
>>> result.expects.real
Array([[0.923, 0.766, 0.567],
       [0.   , 0.049, 0.152]], dtype=float32)
```

The following solvers apply composite operators to the state one subsystem at a time, without building the full matrices:

- [`dq.sesolve()`][dynamiqs.sesolve] and [`dq.mesolve()`][dynamiqs.mesolve], with the ODE methods, the Rouchon methods and [`LowRank`][dynamiqs.method.LowRank], and their gradients,
- [`dq.dssesolve()`][dynamiqs.dssesolve] and [`dq.dsmesolve()`][dynamiqs.dsmesolve],
- [`dq.mesteadystate()`][dynamiqs.mesteadystate] with [`DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState].

The other cases build the full matrices once, and gain nothing: `vectorized=True`, the [`Expm`][dynamiqs.method.Expm] method (so the propagator solvers for constant operators), the jump solvers [`dq.jssesolve()`][dynamiqs.jssesolve] and [`dq.jsmesolve()`][dynamiqs.jsmesolve], and [`LyapunovGMRES`][dynamiqs.method.LyapunovGMRES]. Expectation values also use the full matrix of composite expectation operators, which is faster than applying them term by term.
