# Python API

The **dynamiqs** Python API features two main types of functions: solvers of differential equations describing quantum systems, and various utility functions to ease the creation and manipulation of quantum states and operators.

## Quantum solvers

::: dynamiqs.solvers
    options:
        table: true
        members:
        - sesolve
        - mesolve
        - smesolve

## Core

### Time-dependent arrays

::: dynamiqs.time_array
    options:
        table: true
        members:
        - TimeArray
        - constant
        - pwc
        - modulated
        - timecallable

### Solvers (dq.solver)

::: dynamiqs.solver
    options:
        table: true
        members:
        - Tsit5
        - Dopri5
        - Dopri8
        - Euler
        - Rouchon1
        - Rouchon2
        - Propagator

### Gradients (dq.gradient)

::: dynamiqs.gradient
    options:
        table: true

### Options

::: dynamiqs.options
    options:
        table: true

### Results

::: dynamiqs.result
    options:
        table: true

## Utilities

### Operators

::: dynamiqs.utils.operators
    options:
        table: true

### States

::: dynamiqs.utils.states
    options:
        table: true

### Quantum utilities

::: dynamiqs.utils.utils
    options:
        table: true

### JAX-related utilities

::: dynamiqs.utils.jax_utils
    options:
        table: true

### Vectorization

::: dynamiqs.utils.vectorization
    options:
        table: true

### Quantum optimal control

::: dynamiqs.utils.optimal_control
    options:
        table: true

### Random arrays

::: dynamiqs.utils.random
    options:
        table: true

### Plotting

::: dynamiqs.plots
    options:
        table: true
        members:
        - plot_wigner
        - plot_wigner_mosaic
        - plot_wigner_gif
        - plot_pwc_pulse
        - plot_fock
        - plot_fock_evolution
        - plot_hinton
        - gridplot
        - mplstyle
