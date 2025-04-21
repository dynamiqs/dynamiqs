# Python API

The **Dynamiqs** Python API features two main types of functions: solvers of differential equations describing quantum systems, and various utility functions to ease the creation and manipulation of quantum states and operators.

## Solvers

### General

::: dynamiqs.integrators
    options:
        table: true
        members:
        - sesolve
        - mesolve
        - sepropagator
        - mepropagator
        - floquet

### Stochastic

::: dynamiqs.integrators
    options:
        table: true
        members:
        - jssesolve
        - dssesolve
        - jsmesolve
        - dsmesolve

## Core

### Quantum arrays

::: dynamiqs.qarrays.qarray
    options:
        table: true
        members:
        - QArray

### Time-dependent qarrays

::: dynamiqs.time_qarray
    options:
        table: true
        members:
        - TimeQArray
        - constant
        - pwc
        - modulated
        - timecallable

### Methods (dq.method)

::: dynamiqs.method
    options:
        table: true
        members:
        - Dopri5
        - Dopri8
        - Euler
        - EulerMaruyama
        - Event
        - Expm
        - Kvaerno3
        - Kvaerno5
        - Rouchon1
        - Tsit5

### Gradients (dq.gradient)

::: dynamiqs.gradient
    options:
        table: true
        members:
        - Autograd
        - CheckpointAutograd
        - ForwardAutograd

## Utilities

### Operators

::: dynamiqs.utils.operators
    options:
        table: true
        members:
        - cnot
        - create
        - destroy
        - displace
        - eye
        - eye_like
        - hadamard
        - momentum
        - number
        - parity
        - position
        - quadrature
        - rx
        - ry
        - rz
        - sgate
        - sigmam
        - sigmap
        - sigmax
        - sigmay
        - sigmaz
        - squeeze
        - tgate
        - toffoli
        - zeros
        - zeros_like

### States

::: dynamiqs.utils.states
    options:
        table: true
        members:
        - basis
        - basis_dm
        - coherent
        - coherent_dm
        - excited
        - fock
        - fock_dm
        - ground

### Quantum utilities

::: dynamiqs.utils
    options:
        table: true
        members:
        - bloch_coordinates
        - braket
        - cosm
        - dag
        - dissipator
        - entropy_vn
        - expect
        - expm
        - fidelity
        - isbra
        - isdm
        - isherm
        - isket
        - isop
        - lindbladian
        - norm
        - overlap
        - powm
        - proj
        - ptrace
        - purity
        - signm
        - sinm
        - tensor
        - tobra
        - todm
        - toket
        - trace
        - tracemm
        - unit
        - wigner
        namespace: utils/general/

### QArray utilities

::: dynamiqs.qarrays.utils
    options:
        table: true
        members:
        - asqarray
        - isqarraylike
        - sparsedia_from_dict
        - stack
        - to_jax
        - to_numpy
        - to_qutip

### Global settings

::: dynamiqs.utils.global_settings
    options:
        table: true
        members:
        - set_device
        - set_layout
        - set_matmul_precision
        - set_precision
        - set_progress_meter

### Vectorization

::: dynamiqs.utils.vectorization
    options:
        table: true
        members:
        - operator_to_vector
        - sdissipator
        - slindbladian
        - spre
        - sprepost
        - spost
        - vector_to_operator

### Quantum optimal control

::: dynamiqs.utils.optimal_control
    options:
        table: true
        members:
        - cd_gate
        - snap_gate

### Random (dq.random)

::: dynamiqs.random
    options:
        table: true
        members:
        - complex
        - dm
        - herm
        - ket
        - psd
        - real

### Plotting (dq.plot)

::: dynamiqs.plot
    options:
        table: true
        members:
        - fock
        - fock_evolution
        - gifit
        - grid
        - hinton
        - mplstyle
        - pwc_pulse
        - wigner
        - wigner_gif
        - wigner_mosaic
        - xyz

### Magic helpers

::: dynamiqs.hermitian_conjugate
    options:
        table: true
        members:
        - hc
