# Floquet Integration

This tutorial introduces Floquet integration, why it is useful and how it is implemented numerically.

## Floquet's Theorem

Consider a Hamiltonian that is periodic with period $T$: $H(t+T)=H(t)$. Floquet's theorem teaches us that the
solutions to the Schrödinger equation can be written as
$$
    |\Psi_{m}(t)\rangle = e^{-i\epsilon_{m}t}|\phi_{m}(t)\rangle
$$
where $m$ indexes the states, $\epsilon_m$ is the so-called quasienergy, the $|\phi_{m}(t)\rangle$ are
called Floquet modes and $|\Psi_{m}(t)\rangle$ are called Floquet states. The Floquet modes $|\phi_{m}(t)\rangle$
are time periodic: $|\phi_{m}(t+T)\rangle=|\phi_{m}(t)\rangle$. Floquet's theorem is in
complete analogy to Bloch's theorem in 3 dimensions for a particle in a periodic potential. 

A numerical recipe for computing the Floquet modes and quasienergies can be obtained by using the definition
of the Floquet states. On the one hand, we have
$$
    \begin{aligned}
    |\Psi_{m}(t+T)\rangle &= e^{-i\epsilon_{m}(t+T)}|\phi_{m}(t+T)\rangle
                          &= e^{-i\epsilon_{m}(t+T)}|\phi_{m}(t)\rangle.
    \end{aligned}
$$
On the other hand, the time-propagated Floquet state $|\Psi_{m}(t+T)\rangle$ can be obtained from the 
propagator
$$
    |\Psi_{m}(t+T)\rangle=U(0,T)|\Psi_{m}(t)\rangle.
$$
Thus we find that the Floquet modes satisfy an eigenvalue equation
$$
    U(0,T)|\phi_{m}(0)\rangle=e^{-i\epsilon_{m}T}|\phi_{m}(0)\rangle
$$
where the eigenvalues are exponentials of the quasienergies. Thus to obtain the Floquet modes and associated
quasienergies, one only has to diagonalize the propagator. The Floquet modes at times that are not integer
multiples of the drive period can then be immediately obtained from the Floquet modes at $t=0$
$$
    |\Psi_{m}(t)\rangle = e^{-i\epsilon_{m}t}|\phi_{m}(t)\rangle=U(0,t)|\Psi_{m}(0)\rangle=U(0,t)|\phi_{m}(0)\rangle,
$$
therefore
$$
    |\phi_{m}(t)\rangle=e^{i\epsilon_{m}t}U(0,t)|\phi_{m}(0)\rangle.
$$
In many cases, the diagonalization of the propagator is best done numerically. However, there are some examples 
that are amenable to an analytic treatment that help to build intuition.

## Driven qubit

Let us consider a driven qubit within the rotating-wave approximation (RWA)
$$
    H(t) = -\frac{\omega}{2}\sigma_{z}+\frac{A}{2}(\sigma_{+}e^{-i\omega_{d}t} + {\rm H.~c.~}),
$$
where $\sigma_{+}=|1\rangle\langle0|$. We can straightforwardly solve for the propagator associated with this
Hamiltonian by first moving into a rotating frame (also called the interaction frame) defined by the unitary 
transformation $U_{r}(t)=\exp(i\omega_{d}t\sigma_{z}/2)$. The Hamiltonian in the rotating frame is
$$
    H'(t)=U_{r}^{\dagger} H(t)U_{r}-iU_{r}^{\dagger}\dot{U_{r}} =-\frac{\delta\omega}{2}\sigma_{z}+\frac{A}{2}(\sigma_{+} + {\rm H.~c.~}),
$$
where $\delta\omega=\omega-\omega_{d}$. This is a time-independent Hamiltonian, for which we can obtain the propagator immediately by exponentiation. The
propagator in this frame after one period of the drive is
$$
    U'(0,T) = \cos(\Omega_{R}t)I-i\frac{\sin(\Omega_{R}t)}{\Omega_{R}}\left(-\frac{\delta\omega}{2}\sigma_{z} + \frac{A}{2}\sigma_{x}\right),
$$
where $\Omega_{r}=\sqrt{}$
CONTINUE HERE XXXXXXXXXX
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

## Using dynamiqs

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
