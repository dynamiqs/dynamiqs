# Python API

The **Dynamiqs** Python API features two main types of functions: solvers of differential equations describing quantum systems, and various utility functions to ease the creation and manipulation of quantum states and operators.

## Solvers

### General

::: dynamiqs.integrators
    options:
        extra:
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
        extra:
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
        extra:
            table: true
        members:
        - QArray

### Time-dependent qarrays

::: dynamiqs.time_qarray
    options:
        extra:
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
        extra:
            table: true
        members:
        - Tsit5
        - Dopri5
        - Dopri8
        - Kvaerno3
        - Kvaerno5
        - Euler
        - EulerJump
        - EulerMaruyama
        - Rouchon1
        - Rouchon2
        - Rouchon3
        - Expm
        - Event
        - JumpMonteCarlo
        - DiffusiveMonteCarlo

### Gradients (dq.gradient)

::: dynamiqs.gradient
    options:
        extra:
            table: true
        members:
        - Direct
        - BackwardCheckpointed
        - Forward

## Utilities

### Operators

::: dynamiqs.utils.operators
    options:
        extra:
            table: true
        members:
        - eye
        - eye_like
        - zeros
        - zeros_like
        - destroy
        - create
        - number
        - parity
        - displace
        - squeeze
        - quadrature
        - position
        - momentum
        - sigmax
        - sigmay
        - sigmaz
        - xyz
        - sigmap
        - sigmam
        - hadamard
        - rx
        - ry
        - rz
        - sgate
        - tgate
        - cnot
        - toffoli


### States

::: dynamiqs.utils.states
    options:
        extra:
            table: true
        members:
        - fock
        - fock_dm
        - basis
        - basis_dm
        - coherent
        - coherent_dm
        - ground
        - ground_dm
        - excited
        - excited_dm
        - thermal_dm


### Quantum utilities

::: dynamiqs.utils
    options:
        extra:
            table: true
            namespace: utils/general/
        members:
        - dag
        - powm
        - expm
        - cosm
        - sinm
        - signm
        - trace
        - tracemm
        - ptrace
        - tensor
        - expect
        - norm
        - unit
        - dissipator
        - lindbladian
        - isket
        - isbra
        - isdm
        - isop
        - isherm
        - toket
        - tobra
        - todm
        - proj
        - braket
        - overlap
        - fidelity
        - purity
        - entropy_vn
        - entropy_relative
        - bloch_coordinates
        - wigner


### QArray utilities

::: dynamiqs.qarrays.utils
    options:
        extra:
            table: true
        members:
        - asqarray
        - isqarraylike
        - stack
        - to_jax
        - to_numpy
        - to_qutip
        - sparsedia_from_dict

### Global settings

::: dynamiqs.utils.global_settings
    options:
        extra:
            table: true
        members:
        - set_device
        - set_precision
        - set_matmul_precision
        - set_layout
        - set_progress_meter


### Vectorization

::: dynamiqs.utils.vectorization
    options:
        extra:
            table: true
        members:
        - vectorize
        - unvectorize
        - spre
        - spost
        - sprepost
        - sdissipator
        - slindbladian


### Quantum optimal control

::: dynamiqs.utils.optimal_control
    options:
        extra:
            table: true
        members:
        - snap_gate
        - cd_gate


### Random (dq.random)

::: dynamiqs.random
    options:
        extra:
            table: true
        members:
        - real
        - complex
        - herm
        - psd
        - dm
        - operator
        - ket


### Plotting (dq.plot)

::: dynamiqs.plot
    options:
        extra:
            table: true
        members:
        - wigner
        - wigner_mosaic
        - wigner_gif
        - wigner_data
        - pwc_pulse
        - fock
        - fock_evolution
        - hinton
        - xyz
        - gifit
        - grid
        - mplstyle

### Helpers

::: dynamiqs.helpers
    options:
        extra:
            table: true
        members:
        - hc
        - clicktimes_sse_to_sme
        - measurements_sse_to_sme
