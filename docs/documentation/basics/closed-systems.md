# Closed systems

This tutorial introduces the quantum state for a closed quantum system, describes its evolution with the Schrödinger equation, and explains two common numerical methods to simulate the evolution: computing the propagator or solving the ODE iteratively.

## The quantum state

The quantum state that describes a closed quantum system is a **state vector** $\ket\psi$, i.e. a column vector of size $n$(1):
$$
    \ket\psi=
    \begin{pmatrix}
    \alpha_0\\\\
    \vdots\\\\
    \alpha_{n-1}
    \end{pmatrix},
$$
with $\alpha_0,\dots,\alpha_{n-1}\in\mathbb{C}$ and such that $\sum |\alpha_i|^2=1$ (the state is a unit vector).
{ .annotate }

1. Where $n$ is the dimension of the finite-dimensional complex Hilbert space of the system.

!!! Example "Example for a two-level system"
    For a two-level system, $\ket\psi=\begin{pmatrix}\alpha_0\\\alpha_1\end{pmatrix}$ with $|\alpha_0|^2+|\alpha_1|^2=1$.

Numerically, each coefficient of the state is stored as a complex number represented by two real numbers (the real and the imaginary parts), stored either

- in single precision: the `complex64` type which uses two `float32`,
- in double precision: the `complex128` type which uses two `float64`.

A greater precision will give a more accurate result, but will also take longer to calculate.

## The Schrödinger equation

The state evolution is described by the **Schrödinger equation**:
$$
    i\hbar\frac{\dd\ket{\psi(t)}}{\dt}=H\ket{\psi(t)},
$$
where $H$ is a linear operator called the **Hamiltonian**, a matrix of size $n\times n$. This equation is a *first-order (linear and homogeneous) ordinary differential equation* (ODE). To simplify notations, we set $\hbar=1$. In this tutorial we consider a constant Hamiltonian, but note that it can also be time-dependent $H(t)$.

!!! Example "Example for a two-level system"
    The Hamiltonian of a two-level system with energy difference $\omega$ is $H=\frac{\omega}{2}\sigma_z=\begin{pmatrix}\omega/2&0\\0&-\omega/2\end{pmatrix}$.

## Solving the Schrödinger equation numerically

There are two common ideas for solving the Schrödinger equation.

### Computing the propagator

The state at time $t$ is given by $\ket{\psi(t)}=e^{-iHt}\ket{\psi(0)}$, where $\psi(0)$ is the state at time $t=0$. The operator $U(t)=e^{-iHt}$ is called the **propagator**, a matrix of size $n\times n$.

??? Note "Solution for a time-dependent Hamiltonian"
    For a time-dependent Hamiltonian $H(t)$, the solution at time $t$ is
    $$
        \ket{\psi(t)} = \mathscr{T}\exp\left(-i\int_0^tH(t')\dt'\right)\ket{\psi(0)},
    $$
    where $\mathscr{T}$ is the time-ordering symbol, which indicates the time-ordering of the Hamiltonians upon expansion of the matrix exponential (Hamiltonians at different times do not commute).

The first idea is to explicitly compute the propagator to evolve the state up to time $t$. There are various ways to compute the matrix exponential, such as exact diagonalization of the Hamiltonian or approximate methods such as truncated Taylor series expansions.

^^Space complexity^^: $O(n^2)$ (storing the Hamiltonian).

^^Time complexity^^: $O(n^3)$ (complexity of computing the matrix exponential(1)).
{ .annotate }

1. Computing a matrix exponential requires a few matrix multiplications, and the time complexity of multiplying two dense matrices of size $n\times n$ is $\mathcal{O(n^3)}$.

!!! Example "Example for a two-level system"
    For $H=\frac{\omega}{2}\sigma_z$, the propagator is straighforward to compute:
    $$
        U(t) = e^{-iHt} = \begin{pmatrix}e^{-i\omega t/2} & 0 \\\\ 0 & e^{i\omega t/2}\end{pmatrix}.
    $$

### Integrating the ODE

The Schrödinger equation is an ODE, for which a wide variety of solvers have been developed. The simplest approach is the Euler method, a first-order ODE solver with a fixed step size which we describe shortly. Let us write the Taylor series expansion of the state at time $t+\dt$ up to first order:
$$
    \begin{aligned}
        \ket{\psi(t+\dt)} &= \ket{\psi(t)}+\dt\frac{\dd\ket{\psi(t)}}{\dt}+\mathcal{O}(\dt^2) \\\\
        &\approx \ket{\psi(t)}-iH\dt\ket{\psi(t)},
    \end{aligned}
$$
where we used the Schrödinger equation to replace the time derivative of the state. By choosing a sufficiently small step size $\dt$ and starting from $\ket{\psi(0)}$, the state is then iteratively evolved to a final time using the previous equation.

There are two main types of ODE solvers:

- **Fixed step size**: as with the Euler method, the step size $\dt$ is fixed during the simulation. The best known higher order methods are the *Runge-Kutta methods*. It is important for all these methods that the time step is sufficiently small to ensure the accuracy of the solution.
- **Adaptive step size**: the step size is automatically adjusted during the simulation, at each time step. A well-known method is the *Dormand-Prince method*.

^^Space complexity^^: $O(n^2)$ (storing the Hamiltonian).

^^Time complexity^^: $O(n^2\times\text{number of time steps})$ (complexity of the matrix-vector product at each time step).

## Using Dynamiqs

You can create the state and Hamiltonian using any array-like object. Let's take the example of a two-level system with a simple Hamiltonian:

```python
import jax.numpy as jnp
import dynamiqs as dq

psi0 = dq.ground()                # initial state
H = dq.sigmaz()                   # Hamiltonian
tsave = jnp.linspace(0, 1.0, 11)  # saving times
res = dq.sesolve(H, psi0, tsave)  # run the simulation
print(res.states[-1])             # print the final state
```

```text title="Output"
|██████████| 100.0% ◆ elapsed 2.52ms ◆ remaining 0.00ms
Array([[0.  +0.j   ],
       [0.54+0.841j]], dtype=complex64)
```

If you want to know more about the available solvers or the different options, head to the [`dq.sesolve()`][dynamiqs.sesolve] API documentation.

You can also directly compute the propagator with the [`dq.sepropagator()`][dynamiqs.sepropagator] solver. Continuing the last example:

```python
res = dq.sepropagator(H, tsave)
print(res.propagators[-1])  # print the final propagator
```

```text title="Output"
Array([[0.54-0.841j 0.  +0.j   ]
       [0.  +0.j    0.54+0.841j]], dtype=complex64)
```
