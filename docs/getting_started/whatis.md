# What is dynamiqs ?

dynamiqs is a high-performance quantum systems simulation library based on JAX.

The **dynamiqs** library enables GPU simulation of large quantum systems, and computation of gradients based on the evolved quantum state. Differentiable solvers are available for the Schrödinger equation, the Lindblad master equation, and the stochastic master equation. The library is fully built on JAX and can efficiently run on CPUs and GPUs.

!!! Warning
    This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

Some exciting features of dynamiqs include:

- Running simulations on **GPUs** and **TPUs**, with a significant speedup for large Hilbert space dimensions.
- **Batching** many simulations of different Hamiltonians, jump operators or initial states to run them concurrently.
- Exploring solvers **tailored to quantum** simulations that preserve the properties of the state, such as trace and positivity.
- Computing **gradients** of any function of the evolved quantum state with respect to any parameter of the Hamiltonian, jump operators, or initial state.
- Implementing **your own solvers** with ease by subclassing our base solver class and focusing directly on the solver logic.

We hope that this library will prove beneficial to the community for e.g. simulations of large quantum systems, batched simulations of time-varying problems, gradient-based parameter estimation, or large-scale quantum optimal control.
