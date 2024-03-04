# More features!

Below are some cool features of **dynamiqs** that are either already available or planned for the near future.

## Solvers

- Choose between a variety of solvers, from **modern** ODE solvers (e.g. Tsit5 and PID controllers for adaptive step-sizing) to **quantum-tailored** solvers that preserve the physicality of the evolution (the state trace and positivity are preserved).
- Simulate **time-varying problems** (both Hamiltonian and jump operators) with support for various formats (piecewise constant operator, constant operator modulated by a time-dependent factor, etc.).
- Define a **custom save function** during the evolution (e.g. to register only the state purity, to track a subsystem by taking the partial trace of the full system, or to compute the population in the last Fock states to regularise your QOC problem).
- Easily implement **your own solvers** by subclassing our base solver class and focusing directly on the solver logic.
- Simulate SME trajectories **orders of magnitude faster** by batching the simulation over the stochastic trajectories.
- Use **adaptive step-size solvers** to solve the SME (based on Brownian bridges to generate the correct statistics).
- **Parallelise** large simulations across multiple CPUs/GPUs.

## Gradients

- Choose between **various methods** to compute the gradient, to tradeoff speed and memory (e.g. use the optimal online checkpointing scheme of [Diffrax](https://github.com/patrick-kidger/diffrax) to compute gradients for large systems).
- Compute gradients with **machine-precision accuracy**.
- Evaluate **derivatives with respect to evolution time** (e.g. for time-optimal quantum control).
- Compute **higher order derivatives** (e.g. the Hessian).

## Utilities

- Balance **accuracy and speed** by choosing between single precision (`float32` and `complex64`) or double precision (`float64` and `complex128`).
- Plot beautiful figures by using our **handcrafted plotting function**.
- Apply any functions to **batched arrays** (e.g. `dq.wigner(states)` to compute the wigners of many states at once).
- Use **QuTiP objects as arguments** to any functions (e.g. if you have existing code to define your Hamiltonian in QuTiP, or if you want to use our nice plotting functions on a list of QuTiP states).

## Library development

- Enjoy **modern software development practices and tools**.
- Build confidence from the **analytical tests** that verify state correctness and gradient accuracy for every solver, at each commit.

## Coming soon

- Discover a custom **sparse format**, with substantial speedups for large systems.
- Use **implicit** ODE solvers.
- Simulate using propagators solvers based on **Krylov subspace methods**.
- **Benchmark code** to compare solvers and performance for different systems.