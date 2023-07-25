<h1 align="center">
    <img src="./media/dynamiqs-logo.png" width="520" alt="dynamiqs library logo">
</h1>

[P. Guilmin](https://github.com/pierreguilmin), [R. Gautier](https://github.com/gautierronan), [A. Bocquet](https://github.com/abocquet), [E. Genois](https://github.com/eliegenois)

[![ci](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml)  ![python version](https://img.shields.io/badge/python-3.8%2B-blue) [![chat](https://badgen.net/badge/icon/on%20slack?icon=slack&label=chat&color=orange)](https://join.slack.com/t/dynamiqs-org/shared_invite/zt-1z4mw08mo-qDLoNx19JBRtKzXlmlFYLA) [![license: GPLv3](https://img.shields.io/badge/license-GPLv3-yellow)](https://github.com/dynamiqs/dynamiqs/blob/main/LICENSE) [![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

High-performance quantum systems simulation with PyTorch.

The **dynamiqs** library enables GPU simulation of large quantum systems, and computation of gradients based on the evolved quantum state. Differentiable solvers are available for the Schrödinger equation, the Lindblad master equation, and the Stochastic master equation. The library is fully built on PyTorch and can efficiently run on CPUs and GPUs.

:hammer_and_wrench: This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

Some exciting features of dynamiqs include:

- Running simulations on GPUs, with a significant speedup for large Hilbert space dimensions.
- Batching many simulations of different Hamiltonians or initial states to run them concurrently.
- Exploring quantum-specific solvers that preserve the properties of the state, such as trace and positivity.
- Computing gradients of any function of the evolved quantum state with respect to any parameter of the Hamiltonian, jump operators, or initial state.
- Using the library as a drop-in replacement for [QuTiP](https://qutip.org/) by directly passing QuTiP-defined quantum objects to our solvers.
- Implementing your own solvers with ease by subclassing our base solver class and focusing directly on the solver logic.
- Enjoy reading our carefully crafted documentation.

We hope that this library will prove beneficial to the community for e.g. simulations of large quantum systems, gradient-based parameter estimation, or large-scale quantum optimal control.
