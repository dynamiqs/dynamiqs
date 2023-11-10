# Test systems

We test our solvers on examples with analytical solutions.

## `Cavity` and `OCavity`

**Problem**

- Hamiltonian: $H = \Delta a^\dag a$
- Jump operators: $L = \sqrt{\kappa} a$
- Expectation operators: $O^{(1)} = X = (a+a^\dag)/2$ and $O^{(2)} = P = i(a^\dag-a)/2$
- Initial state: $\ket{\psi_0} = \ket{\alpha_0}$ with $\alpha_0\in\R$
- Loss: $l_t = \braket{a^\dag a}_t$
- Gradient parameters: $\theta=(\Delta, \alpha_0, \kappa)$

**Solution at time $t$**

- State at time $t$:
  $$
      \ket{\psi_t} = \ket{\alpha_t}\ \text{with}\ \alpha_t = \alpha_0 e^{-i\Delta t - \kappa t/2}
  $$
- Expectation values at time $t$:
  $$
  \begin{aligned}
      \braket{O^{(1)}}_t &= \mathrm{Re}[\alpha_t] = \alpha_0 \cos(\Delta t) e^{-\kappa t/2}\\
      \braket{O^{(2)}}_t &= \mathrm{Im}[\alpha_t] = -\alpha_0 \sin(\Delta t) e^{-\kappa t/2}\\
  \end{aligned}
  $$
- Loss at time $t$:
  $$
      l_t = |\alpha_t|^2 = \alpha_0^2 e^{-\kappa t}
  $$
- Gradient of the loss at time $t$:
  $$
      \nabla_\theta l_t =
      \begin{pmatrix}
        0.0 \\
        2 \alpha_0 e^{-\kappa t} \\
        -\alpha_0^2 t e^{-\kappa t}
      \end{pmatrix}
  $$
- Gradients of the expectation values at time $t$:
  $$
  \begin{aligned}
      \nabla_\theta \braket{O^{(1)}}_t &=
      \begin{pmatrix}
            -\alpha_0 t \sin(\Delta t) e^{-\kappa t/2} \\
            \cos(\Delta t) e^{-\kappa t/2} \\
            - \frac12 \alpha_0 t \cos(\Delta t) e^{-\kappa t/2}\\
      \end{pmatrix} \\
      \nabla_\theta \braket{O^{(2)}}_t &=
      \begin{pmatrix}
            -\alpha_0 t \cos(\Delta t) e^{-\kappa t/2} \\
            -\sin(\Delta t) e^{-\kappa t/2} \\
            \frac12 \alpha_0 t \sin(\Delta t) e^{-\kappa t/2}
      \end{pmatrix}
  \end{aligned}
  $$
