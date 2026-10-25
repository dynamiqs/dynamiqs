# What is Dynamiqs?

## In a nutshell

**Dynamiqs** is a Python library for **GPU-accelerated** and **differentiable** quantum simulations. Solvers are available for the Schrödinger and Lindblad master equations, their propagators and steady states, Floquet modes and quasienergies, and quantum trajectories of the jump and diffusive stochastic Schrödinger and master equations. The library is built with [JAX](https://docs.jax.dev/) and the main solvers are based on [Diffrax](https://github.com/patrick-kidger/diffrax).

See the [Python API](../../python_api/index.md) for a list of available functions and classes.

The main features of **Dynamiqs** are:

- Running simulations on **CPUs** and **GPUs** with high-performance.
- Executing many simulations **concurrently** by batching over Hamiltonians, initial states or jump operators.
- Computing **gradients** of arbitrary functions with respect to arbitrary parameters of the system.
- Full **compatibility** with the [JAX](https://docs.jax.dev/) ecosystem with a [QuTiP](https://qutip.org/)-like API.

We hope that this library will prove useful to the community for e.g. simulation of large quantum systems, gradient-based parameter estimation or quantum optimal control. The library is designed for large-scale problems, but also runs efficiently on CPUs for smaller problems.

Starting from version 1.0, the public API follows [semantic versioning](https://semver.org/): breaking changes are only introduced in major releases.

## The Dynamiqs project

### Philosophy

There is a noticeable gap in the availability of an open-source library that simplifies gradient-based parameter estimation and quantum optimal control. In addition, faster simulations of large systems are essential to accelerate the development of quantum technologies. The **Dynamiqs** library addresses both of these needs. It aims to be a fast and reliable building block for **GPU-accelerated** and **differentiable** solvers. We also work to make the library compatible with the existing Python ecosystem (i.e. JAX and QuTiP) to allow easy interfacing with other libraries.

### Team and sponsoring

The library is being developed by a **team of physicists and developers**. We are working with theorists, experimentalists, machine learning practitioners, optimization and numerical methods experts to make the library as useful and as powerful as possible. The library is sponsored by the startup [Alice & Bob](https://alice-bob.com/), where it is being used to simulate, calibrate and control chips made of superconducting-based dissipative cat qubits.

### History

Development started in early 2023, the library was originally based on PyTorch with homemade solvers and gradient methods. It was completely rewritten in JAX in early 2024 for performance.

## More features!

Below are some cool features of **Dynamiqs**.

### Solvers

- Choose between a variety of methods for each solver, from **modern** explicit and implicit ODE methods (e.g. Tsit5 and PID controllers for adaptive step-sizing) to **quantum-tailored** methods that preserve the physicality of the evolution (the state trace and positivity are preserved).
- Simulate **time-varying problems** (both Hamiltonian and jump operators) with support for various formats (piecewise constant operator, constant operator modulated by a time-dependent factor, etc.).
- Define a **custom save function** during the evolution (e.g. to register only the state purity, to track a subsystem by taking the partial trace of the full system, or to compute the population in the last Fock states to regularize your QOC problem).
- Compute the **steady state** of the Lindblad master equation directly with [`dq.mesteadystate()`][dynamiqs.mesteadystate], with a preconditioned iterative solver for large systems and implicit differentiation of the result.
- Simulate **quantum trajectories** of the jump and diffusive stochastic Schrödinger and master equations ([`dq.jssesolve()`][dynamiqs.jssesolve], [`dq.dssesolve()`][dynamiqs.dssesolve], [`dq.jsmesolve()`][dynamiqs.jsmesolve], [`dq.dsmesolve()`][dynamiqs.dsmesolve]) **orders of magnitude faster** by batching the simulation over the trajectories.

### Gradients

- Choose between **various methods** to compute the gradient, to trade off speed and memory (e.g. use the optimal online checkpointing scheme of [Diffrax](https://github.com/patrick-kidger/diffrax) to compute gradients for large systems).
- Compute gradients with **machine-precision accuracy**.
- Evaluate **derivatives with respect to evolution time** (e.g. for time-optimal quantum control).
- Compute **higher order derivatives** (e.g. the Hessian).

### Utilities

- Balance **accuracy and speed** by choosing between single precision (`float32` and `complex64`) or double precision (`float64` and `complex128`).
- Discover a custom **sparse data format** designed for matrices with only a few dense diagonals, offering substantial speedups for large systems.
- Store the operators of systems made of several subsystems as their **tensor product factors**, without building the full matrix.
- Plot beautiful figures by using our **handcrafted plotting function**.
- Apply any functions to **batched arrays** (e.g. `dq.wigner(states)` to compute the Wigner functions of many states at once).
- Use **QuTiP objects as arguments** to any functions (e.g. if you have existing code to define your Hamiltonian in QuTiP, or if you want to use our nice plotting functions on a list of QuTiP states).

### Library development

- Enjoy **modern software development practices and tools**.
- Build confidence from the **analytical tests** that verify state correctness and gradient accuracy for every solver, at each commit.
