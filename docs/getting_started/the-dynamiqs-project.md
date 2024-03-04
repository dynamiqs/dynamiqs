# The dynamiqs project

## Philosophy

There is a noticeable gap in the availability of an open-source library that simplifies gradient-based parameter estimation and quantum optimal control. In addition, faster simulations of large systems are essential to accelerate the development of quantum technologies. The **dynamiqs** library addresses both of these needs. It aims to be a fast and reliable building block for **GPU-accelerated** and **differentiable** solvers. We also work to make the library compatible with the existing Python ecosystem (i.e. JAX and QuTiP) to allow easy interfacing with other libraries.

## Team and sponsoring

The library is being developed by a **team of physicists and developers**. We are working with theorists, experimentalists, machine learning practitioners, optimisation and numerical methods experts to make the library as useful and as powerful as possible. The library is sponsored by the startup [Alice & Bob](https://alice-bob.com/), where it is being used to simulate, calibrate and control chips made of superconducting-based dissipative cat qubits.

## History

Development started in early 2023, the library was originally based on PyTorch with homemade solvers and gradient methods. It was completely rewritten in JAX in early 2024 for performance.
