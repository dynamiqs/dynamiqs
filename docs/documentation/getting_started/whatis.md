# What is Dynamiqs ?

## In a nutshell

**Dynamiqs** is a Python library for **GPU-accelerated** and **differentiable** quantum simulations. Solvers are available for the Schrödinger equation, the Lindblad master equation, and the stochastic master equation. The library is built with [JAX](https://jax.readthedocs.io/en/latest/index.html) and the main solvers are based on [Diffrax](https://github.com/patrick-kidger/diffrax).

See the [Python API](../../python_api/index.md) for a list of available functions and classes.

The main features of **Dynamiqs** are:

- Running simulations on **CPUs** and **GPUs** with high-performance.
- Executing many simulations **concurrently** by batching over Hamiltonians, initial states or jump operators.
- Computing **gradients** of arbitrary functions with respect to arbitrary parameters of the system.
- Full **compatibility** with the [JAX](https://jax.readthedocs.io/en/latest/index.html) ecosystem with a [QuTiP](https://qutip.org/)-like API.

We hope that this library will prove useful to the community for e.g. simulation of large quantum systems, gradient-based parameter estimation or quantum optimal control. The library is designed for large-scale problems, but also runs efficiently on CPUs for smaller problems.

!!! Warning
    This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

## The Dynamiqs project

### Philosophy

There is a noticeable gap in the availability of an open-source library that simplifies gradient-based parameter estimation and quantum optimal control. In addition, faster simulations of large systems are essential to accelerate the development of quantum technologies. The **Dynamiqs** library addresses both of these needs. It aims to be a fast and reliable building block for **GPU-accelerated** and **differentiable** solvers. We also work to make the library compatible with the existing Python ecosystem (i.e. JAX and QuTiP) to allow easy interfacing with other libraries.

### Team and sponsoring

The library is being developed by a **team of physicists and developers**. We are working with theorists, experimentalists, machine learning practitioners, optimisation and numerical methods experts to make the library as useful and as powerful as possible. The library is sponsored by the startup [Alice & Bob](https://alice-bob.com/), where it is being used to simulate, calibrate and control chips made of superconducting-based dissipative cat qubits.

### History

Development started in early 2023, the library was originally based on PyTorch with homemade solvers and gradient methods. It was completely rewritten in JAX in early 2024 for performance.

## More features!

Below are some cool features of **Dynamiqs** that are either already available or planned for the near future.

### Solvers

- Choose between a variety of solvers, from **modern** explicit and implicit ODE solvers (e.g. Tsit5 and PID controllers for adaptive step-sizing) to **quantum-tailored** solvers that preserve the physicality of the evolution (the state trace and positivity are preserved).
- Simulate **time-varying problems** (both Hamiltonian and jump operators) with support for various formats (piecewise constant operator, constant operator modulated by a time-dependent factor, etc.).
- Define a **custom save function** during the evolution (e.g. to register only the state purity, to track a subsystem by taking the partial trace of the full system, or to compute the population in the last Fock states to regularise your QOC problem).
- Easily implement **your own solvers** by subclassing our base solver class and focusing directly on the solver logic.
- Simulate SME trajectories **orders of magnitude faster** by batching the simulation over the stochastic trajectories.
- Use **adaptive step-size solvers** to solve the SME (based on Brownian bridges to generate the correct statistics).
- **Parallelise** large simulations across multiple CPUs/GPUs.

### Gradients

- Choose between **various methods** to compute the gradient, to tradeoff speed and memory (e.g. use the optimal online checkpointing scheme of [Diffrax](https://github.com/patrick-kidger/diffrax) to compute gradients for large systems).
- Compute gradients with **machine-precision accuracy**.
- Evaluate **derivatives with respect to evolution time** (e.g. for time-optimal quantum control).
- Compute **higher order derivatives** (e.g. the Hessian).

### Utilities

- Balance **accuracy and speed** by choosing between single precision (`float32` and `complex64`) or double precision (`float64` and `complex128`).
- Plot beautiful figures by using our **handcrafted plotting function**.
- Apply any functions to **batched arrays** (e.g. `dq.wigner(states)` to compute the wigners of many states at once).
- Use **QuTiP objects as arguments** to any functions (e.g. if you have existing code to define your Hamiltonian in QuTiP, or if you want to use our nice plotting functions on a list of QuTiP states).

### Library development

- Enjoy **modern software development practices and tools**.
- Build confidence from the **analytical tests** that verify state correctness and gradient accuracy for every solver, at each commit.

### Coming soon

- Discover a custom **sparse format**, with substantial speedups for large systems.
- Simulate using propagators solvers based on **Krylov subspace methods**.
- **Benchmark code** to compare solvers and performance for different systems.
