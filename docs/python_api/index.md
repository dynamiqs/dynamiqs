# Python API

The **dynamiqs** Python API features two main types of functions: solvers of differential equations describing quantum systems, and various utility functions to ease the creation and manipulation of quantum states and operators.

## Quantum solvers

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

## Core

### Time-dependent arrays

::: dynamiqs.time_array
    options:
        table: true

### Solvers (dq.solver)

::: dynamiqs.solver
    options:
        table: true

### Gradients (dq.gradient)

::: dynamiqs.gradient
    options:
        table: true

### Options

::: dynamiqs.options
    options:
        table: true

### Result

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
