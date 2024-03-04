# What is dynamiqs ?

**dynamiqs** is a Python library for **GPU-accelerated** and **differentiable** quantum simulations. Solvers are available for the Schrödinger equation, the Lindblad master equation, and the stochastic master equation. The library is built with [JAX](https://jax.readthedocs.io/en/latest/index.html) and the main solvers are based on [Diffrax](https://github.com/patrick-kidger/diffrax).

See the [Python API](../python_api/index.html) for a list of all implemented functions.

The main features of **dynamiqs** are:

- Running simulations on **GPUs** with high-performance
- Executing many simulations **concurrently** by batching over Hamiltonians, initial states or jump operators
- Computing **gradients** of arbitrary functions with respect to arbitrary parameters of the system
- Full **compatibility** with the [JAX](https://jax.readthedocs.io/en/latest/index.html) ecosystem with a [QuTiP](https://qutip.org/)-like API

We hope that this library will prove useful to the community for e.g. simulation of large quantum systems, gradient-based parameter estimation or quantum optimal control. The library is designed for large-scale problems, but also runs efficiently on CPUs for smaller problems.

!!! Warning
    This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!