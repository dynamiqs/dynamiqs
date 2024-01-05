# Python API

The **dynamiqs** Python API features two main types of functions: solvers of differential equations describing quantum systems, and various utility functions to ease the creation and manipulation of quantum states and operators.

## Solvers

<div class="doc doc-object doc-module">
    <div class="doc doc-contents first">
        <div class="md-typeset__scrollwrap">
            <div class="md-typeset__table">
                <table>
                    <colgroup>
                        <col span="1" style="width: 30%;">
                        <col span="1" style="width: 70%;">
                    </colgroup>
                    <tbody>
                        <tr>
                            <td class="fixed_height">
                                <a href="/python_api/solvers/sesolve.html">
                                <code>
                                    sesolve
                                </code>
                                </a>
                            </td>
                            <td class="fixed_height">
                                <p>
                                    Solve the Schrödinger equation.
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="fixed_height">
                                <a href="/python_api/solvers/mesolve.html">
                                <code>
                                    mesolve
                                </code>
                                </a>
                            </td>
                            <td class="fixed_height">
                                <p>
                                    Solve the Lindblad master equation.
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td class="fixed_height">
                                <a href="/python_api/solvers/smesolve.html">
                                <code>
                                    smesolve
                                </code>
                                </a>
                            </td>
                            <td class="fixed_height">
                                <p>
                                    Solve the diffusive stochastic master equation (SME).
                                </p>
                            </td>
                        </tr>
                </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

## Utilities

### Operators

::: dynamiqs.utils.operators
    options:
        table: true
        members:
        - eye
        - zero
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
        - sigmap
        - sigmam
        - hadamard

### States

::: dynamiqs.utils.states
    options:
        table: true
        members:
        - fock
        - fock_dm
        - basis
        - basis_dm
        - coherent
        - coherent_dm

### Quantum utilities

::: dynamiqs.utils.utils
    options:
        table: true
        members:
        - dag
        - mpow
        - trace
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
        - toket
        - tobra
        - todm
        - braket
        - overlap
        - fidelity

### Tensor conversion

::: dynamiqs.utils.tensor_types
    options:
        table: true
        members:
        - to_qutip

### Wigner distribution

::: dynamiqs.utils.wigners
    options:
        table: true
        members:
        - wigner

### Vectorization

::: dynamiqs.utils.vectorization
    options:
        table: true
        members:
        - operator_to_vector
        - vector_to_operator
        - spre
        - spost
        - sprepost
        - sdissipator
        - slindbladian

### Quantum optimal control

::: dynamiqs.utils.optimal_control
    options:
        table: true
        members:
        - rand_real
        - rand_complex
        - snap_gate
        - cd_gate

### Plotting

::: dynamiqs.plots.namespace
    options:
        table: true
        members:
        - plot_wigner
        - plot_wigner_mosaic
        - plot_pwc_pulse
        - plot_fock
        - plot_fock_evolution
        - plot_hinton
