# Open systems

This tutorial introduces the quantum state for an open quantum system, describes its evolution with the Lindblad master equation, and explains three common numerical methods to simulate the evolution: computing the propagator, solving the ODE iteratively or sampling trajectories.

## The quantum state

The quantum state that describes an open quantum system is a **density matrix** $\rho$. It is a positive semi-definite Hermitian matrix with unit trace, of size $n\times n$(1).
{ .annotate }

1. Where $n$ is the dimension of the finite-dimensional complex Hilbert space of the system.

!!! Example "Example for a two-level system"
    For a two-level system, $\rho=\begin{pmatrix}a & b\\ c & d\end{pmatrix}$ with $a,d\in\R^+$ and $b,c\in\mathbb{C}$ with $b^*=c$ (Hermitian matrix), $a+d=1$ (unit trace) and all its eigenvalues are positive (positive semi-definite).

Numerically, each coefficient of the state is stored as a complex number represented by two real numbers (the real and the imaginary parts), stored either

- in single precision: the `complex64` type which uses two `float32`,
- in double precision: the `complex128` type which uses two `float64`.

A greater precision will give a more accurate result, but will also take longer to calculate.

## The Lindblad master equation

The state evolution is described by the **Lindblad master equation**:
$$
    \frac{\dd\rho(t)}{\dt} = -\frac{i}{\hbar}[H, \rho(t)] + \sum_{k=1}^N \left(L_k \rho(t) L_k^\dag - \frac{1}{2} L_k^\dag L_k \rho(t) - \frac{1}{2} \rho(t) L_k^\dag L_k\right),
$$
where $H$ is a linear operator called the **Hamiltonian**, a matrix of size $n\times n$, and $\{L_k\}$ is a collection of arbitrary operators called **jump operators** which are also matrices of size $n\times n$. This equation is a *first-order (linear and homogeneous) ordinary differential equation* (ODE). To simplify notations, we set $\hbar=1$. In this tutorial we consider a constant Hamiltonian and jump operators, but note that they can also be time-dependent $H(t)$ and $L_k(t)$.

!!! Example "Example for a two-level system"
    For example, $H=\frac{\omega}{2}\sigma_z=\begin{pmatrix}\omega/2&0\\0&-\omega/2\end{pmatrix}$ and a single jump operator $L=\sqrt\gamma\sigma_-=\begin{pmatrix}0&0\\\sqrt\gamma&0\end{pmatrix}$.

We can also write
$$
    \frac{\dd\rho(t)}{\dt} = \mathcal{L}(\rho(t)),
$$
where $\mathcal{L}$ is a superoperator(1) called the **Liouvillian** (sometimes referred as Lindbladian). We can write the state and Liouvillian in vectorized form, where we see the state $\rho(t)$ as a column vector of size $n^2$, and the Liouvillian as a matrix of size $n^2\times n^2$.
{ .annotate }

1. A superoperator is a linear map that takes an operator and returns an operator.

!!! Example "Example for a two-level system"
    For example, for $H=\frac{\omega}{2}\sigma_z$ and a single jump operator $L=\sqrt\gamma\sigma_-$, the Liouvillian in vectorized form is a $4\times4$ matrix:
    $$
        \mathcal{L} = \begin{pmatrix}
        -\gamma & 0 & 0 & 0\\\\
        0 & -\gamma/2+\omega i & 0 & 0\\\\
        0 & 0 & -\gamma/2-\omega i & 0\\\\
        \gamma & 0 & 0 & 0\\\\
        \end{pmatrix}
    $$

## Solving the Lindblad master equation numerically

There are three common ideas for solving the Lindblad master equation.

### Computing the propagator

The state at time $t$ is given by $\rho(t)=e^{t\mathcal{L}}(\rho(0))$, where $\rho(0)$ is the state at time $t=0$. The superoperator $e^{t\mathcal{L}}$ is called the **propagator**, in vectorized form it is a matrix of size $n^2\times n^2$.

??? Note "Solution for a time-dependent Liouvillian"
    For a time-dependent Liouvillian $\mathcal{L}(t)$, the solution at time $t$ is
    $$
        \rho(t) = \mathscr{T}\exp\left(\int_0^t\mathcal{L}(t')\dt'\right)(\rho(0)),
    $$
    where $\mathscr{T}$ is the time-ordering symbol, which indicates the time-ordering of the Liouvillians upon expansion of the matrix exponential (Liouvillians at different times do not commute).

The first idea is to explicitly compute the propagator to evolve the state up to time $t$. There are various ways to compute the matrix exponential, such as exact diagonalization of the Liouvillian or approximate methods such as truncated Taylos series expansions.

^^Space complexity^^: $O(n^4)$ (storing the Liouvillian).

^^Time complexity^^: $O(n^6)$ (complexity of computing the $n^2\times n^2$ Liouvillian matrix exponential(1)).
{ .annotate }

1. Computing a matrix exponential requires a few matrix multiplications, and the time complexity of multiplying two dense matrices of size $n\times n$ is $\mathcal{O(n^3)}$.

For large Hilbert space sizes, the time complexity of computing the matrix exponential is often prohibitive, hence the need for other methods such as the ones we now describe below.

### Integrating the ODE

The Lindblad master equation is an ODE, for which a wide variety of solvers have been developed. The simplest approach is the Euler method, a first-order ODE solver with a fixed step size which we describe shortly. Let us write the Taylor series expansion of the state at time $t+\dt$ up to first order:
$$
    \begin{aligned}
        \rho(t+\dt) &= \rho(t)+\dt\frac{\dd\rho(t)}{\dt}+\mathcal{O}(\dt^2) \\\\
        &\approx \rho(t)+\dt\mathcal{L}(\rho(t)) \\\\
        &\approx \rho(t)+\dt\left(-i[H(t), \rho(t)] + \sum_{k=1}^N \left(L_k \rho(t) L_k^\dag - \frac{1}{2} L_k^\dag L_k \rho(t) - \frac{1}{2} \rho(t) L_k^\dag L_k\right)\right),
    \end{aligned}
$$
where we used the Lindblad master equation to replace the time derivative of the state. By choosing a sufficiently small step size $\dt$ and starting from $\rho(0)$, the state is then iteratively evolved to a final time using the previous equation.

There are two main types of ODE solvers:

- **Fixed step size**: as with the Euler method, the step size $\dt$ is fixed during the simulation. The best known higher order methods are the *Runge-Kutta methods*. It is important for all these methods that the time step is sufficiently small to ensure the accuracy of the solution.
- **Adaptive step size**: the step size is automatically adjusted during the simulation, at each time step. A well-known method is the *Dormand-Prince method*.

^^Space complexity^^: $O(n^2)$ (storing the Hamiltonian and jump operators).

^^Time complexity^^: $O(n^3\times\text{number of time steps})$ (complexity of the matrix-matrix product at each time step).

### Sampling trajectories

Also called the **quantum-jump** approach.

!!! Warning "Work in progress."

## Using Dynamiqs

You can create the state, Hamiltonian and jump operators using any array-like object. Let's take the example of a two-level system with a simple Hamiltonian and a single jump operator:

```python
import jax.numpy as jnp
import dynamiqs as dq

psi0 = dq.excited()                         # initial state
H = dq.sigmaz()                             # Hamiltonian
jump_ops = [dq.sigmam()]                    # list of jump operators
tsave = jnp.linspace(0, 1.0, 11)            # saving times
res = dq.mesolve(H, jump_ops, psi0, tsave)  # run the simulation
print(res.states[-1])                       # print the final state
```

```text title="Output"
|██████████| 100.0% ◆ elapsed 4.47ms ◆ remaining 0.00ms
Array([[0.368+0.j, 0.   +0.j],
       [0.   +0.j, 0.632+0.j]], dtype=complex64)
```

If you want to know more about the available solvers or the different options, head to the [`dq.mesolve()`][dynamiqs.mesolve] API documentation.

You can also directly compute the propagator with the [`dq.mepropagator()`][dynamiqs.mepropagator] solver. Continuing the last example:

```python
res = dq.mepropagator(H, jump_ops, tsave)
print(res.propagators[-1])  # print the final propagator
```

```text title="Output"
|██████████| 100.0% ◆ elapsed 2.56ms ◆ remaining 0.00ms
Array([[ 0.368+0.j     0.   +0.j     0.   +0.j     0.   +0.j   ]
       [ 0.   +0.j    -0.252+0.552j  0.   +0.j     0.   +0.j   ]
       [ 0.   +0.j     0.   +0.j    -0.252-0.552j  0.   +0.j   ]
       [ 0.632+0.j     0.   +0.j     0.   +0.j     1.   +0.j   ]], dtype=complex64)
```
