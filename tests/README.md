# Test systems

We test our solvers on systems with an analytical solution.

## `Cavity` and `OCavity`

**Problem**

- Hamiltonian: $H = \Delta a^\dagger a$
- Jump operators: $L = \sqrt{\kappa} a$
- Expectation operators: $O^{(1)} = X = (a+a^\dagger)/2$ and $O^{(2)} = P = i(a^\dagger-a)/2$
- Initial state: $\ket{\psi_0} = \ket{\alpha_0}$ with $\alpha_0\in\mathbb{R}$
- Loss: $l(t) = \braket{a^\dagger a}(t)$
- Gradient parameters: $\theta=(\Delta, \alpha_0, \kappa)$

**Solution at time $t$**

- State:

$$
\ket{\psi(t)} = \ket{\alpha(t)}\ \text{with}\ \alpha(t) = \alpha_0 e^{-i\Delta t - \kappa t/2}
$$

- Expectation values:

$$
\begin{aligned}
    \braket{O^{(1)}}(t) &= \mathrm{Re}[\alpha(t)] = \alpha_0 \cos(\Delta t) e^{-\kappa t/2}\\
    \braket{O^{(2)}}(t) &= \mathrm{Im}[\alpha(t)] = -\alpha_0 \sin(\Delta t) e^{-\kappa t/2}
\end{aligned}
$$

- Loss:

$$
l(t) = |\alpha(t)|^2 = \alpha_0^2 e^{-\kappa t}
$$

- Gradient of the loss:

$$
\nabla_\theta l(t) =
\begin{pmatrix}
  0.0 \\
  2 \alpha_0 e^{-\kappa t} \\
  -\alpha_0^2 t e^{-\kappa t}
\end{pmatrix}
$$

- Gradients of the expectation values:

$$
\begin{aligned}
    \nabla_\theta \braket{O^{(1)}}(t) &=
    \begin{pmatrix}
        -\alpha_0 t \sin(\Delta t) e^{-\kappa t/2} \\
        \cos(\Delta t) e^{-\kappa t/2} \\
        - \frac12 \alpha_0 t \cos(\Delta t) e^{-\kappa t/2}
    \end{pmatrix} \\
    \nabla_\theta \braket{O^{(2)}}(t) &=
    \begin{pmatrix}
        -\alpha_0 t \cos(\Delta t) e^{-\kappa t/2} \\
        -\sin(\Delta t) e^{-\kappa t/2} \\
        \frac12 \alpha_0 t \sin(\Delta t) e^{-\kappa t/2}
    \end{pmatrix}
\end{aligned}
$$
