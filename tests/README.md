# Test systems

We test our solvers on systems with an analytical solution.

## `Cavity` and `OCavity`

**Problem**

- Hamiltonian: $H = \Delta a^\dagger a$
- Jump operators: $L = \sqrt{\kappa} a$
- Expectation operators:
  - $O_1 = (a+a^\dagger)/2$
  - $O_2 = i(a^\dagger-a)/2$
- Initial state: $\ket{\psi_0} = \ket{\alpha_0}$ with $\alpha_0\in\mathbb{R}$
- Cost function: $C(\rho) = \mathrm{Tr}[a^\dagger a \rho]$
- Gradient parameters: $\theta=(\Delta, \alpha_0, \kappa)$

**Solution at time $t$**

- State:

$$
\rho_t = \ket{\alpha(t)}\bra{\alpha(t)} \ \text{with} \ \alpha(t) = \alpha_0 e^{-i\Delta t - \kappa t/2}
$$

- Expectation values:

$$
\begin{aligned}
    \braket{O_1}(t) &= \mathrm{Re}[\alpha(t)] = \alpha_0 \cos(\Delta t) e^{-\kappa t/2}\\
    \braket{O_2}(t) &= \mathrm{Im}[\alpha(t)] = -\alpha_0 \sin(\Delta t) e^{-\kappa t/2}
\end{aligned}
$$

- Loss:

$$
C(\rho_t) = |\alpha(t)|^2 = \alpha_0^2 e^{-\kappa t}
$$

- Gradient of the loss:

$$
\nabla_\theta C(\rho_t) =
\begin{pmatrix}
    \frac{\partial C(\rho_t)}{\partial \Delta} \\
    \frac{\partial C(\rho_t)}{\partial \alpha_0} \\
    \frac{\partial C(\rho_t)}{\partial \kappa}
\end{pmatrix} =
\begin{pmatrix}
  0.0 \\
  2 \alpha_0 e^{-\kappa t} \\
  -\alpha_0^2 t e^{-\kappa t}
\end{pmatrix}
$$

- Gradients of the expectation values:

$$
\begin{aligned}
    \nabla_\theta \braket{O_1}(t) &=
    \begin{pmatrix}
        \frac{\partial \braket{O_1}(t)}{\partial \Delta} \\
        \frac{\partial \braket{O_1}(t)}{\partial \alpha_0} \\
        \frac{\partial \braket{O_1}(t)}{\partial \kappa}
    \end{pmatrix} =
    \begin{pmatrix}
        -\alpha_0 t \sin(\Delta t) e^{-\kappa t/2} \\
        \cos(\Delta t) e^{-\kappa t/2} \\
        - \frac12 \alpha_0 t \cos(\Delta t) e^{-\kappa t/2}
    \end{pmatrix} \\
    \nabla_\theta \braket{O_2}(t) &=
    \begin{pmatrix}
        \frac{\partial \braket{O_2}(t)}{\partial \Delta} \\
        \frac{\partial \braket{O_2}(t)}{\partial \alpha_0} \\
        \frac{\partial \braket{O_2}(t)}{\partial \kappa}
    \end{pmatrix} =
    \begin{pmatrix}
        -\alpha_0 t \cos(\Delta t) e^{-\kappa t/2} \\
        -\sin(\Delta t) e^{-\kappa t/2} \\
        \frac12 \alpha_0 t \sin(\Delta t) e^{-\kappa t/2}
    \end{pmatrix}
\end{aligned}
$$

## `TDQubit` and `OTDQubit`

**Problem**

- Hamiltonian: $H(t) = \varepsilon \cos(\omega t) \sigma_x$
- Jump operators: $L = \sqrt{\gamma} \sigma_x$
- Expectation operators:
  - $O_x = \sigma_x$
  - $O_y = \sigma_y$
  - $O_z = \sigma_z$
- Initial state: $\ket{\psi_0} = \ket{g}$
- Cost function: $C(\rho) = \mathrm{Tr}[\sigma_z \rho]$
- Gradient parameters: $\theta=(\varepsilon, \omega, \gamma)$

**Solution at time $t$**

- State:

$$
\rho_t = \tfrac{1}{2} \left[I + \mathrm{e}^{-2\gamma t}\left( \cos(\Omega(t)) \sigma_z - \sin(\Omega(t)) \sigma_y \right)\right] \\ \text{with} \ \Omega(t) = \tfrac{2\varepsilon}{\omega} \sin(\omega t)
$$

- Expectation values:

$$
\begin{aligned}
    \braket{O_x}(t) &= 0\\
    \braket{O_y}(t) &= -\mathrm{e}^{-2\gamma t}\sin(\Omega(t))\\
    \braket{O_z}(t) &= \mathrm{e}^{-2\gamma t}\cos(\Omega(t))
\end{aligned}
$$

- Loss:

$$
C(\rho_t) = \braket{O_z}(t) = \mathrm{e}^{-2\gamma t}\cos(\Omega(t))
$$

- Gradient of the loss:

$$
\nabla_\theta C(\rho_t) =
\begin{pmatrix}
    \frac{\partial C(\rho_t)}{\partial \varepsilon} \\
    \frac{\partial C(\rho_t)}{\partial \omega} \\
    \frac{\partial C(\rho_t)}{\partial \gamma}
\end{pmatrix} =
\begin{pmatrix}
    -\tfrac{2}{\omega} \sin(\omega t) \mathrm{e}^{-2\gamma t} \sin(\Omega(t)) \\
    -\left( \tfrac{2\varepsilon t}{\omega} \cos(\omega t) - \tfrac{2\varepsilon}{\omega^2} \sin(\omega t)\right)\mathrm{e}^{-2\gamma t} \sin(\Omega(t)) \\
    -2 t \mathrm{e}^{-2\gamma t} \cos(\Omega(t))
\end{pmatrix}
$$

- Gradients of the expectation values:

$$
\begin{aligned}
    \nabla_\theta \braket{O_x}(t) &=
    \begin{pmatrix}
        \frac{\partial \braket{O_x}(t)}{\partial \varepsilon} \\
        \frac{\partial \braket{O_x}(t)}{\partial \omega} \\
        \frac{\partial \braket{O_x}(t)}{\partial \gamma}
    \end{pmatrix} =
    \begin{pmatrix}
        0 \\
        0 \\
        0
    \end{pmatrix} \\
    \nabla_\theta \braket{O_y}(t) &=
    \begin{pmatrix}
        \frac{\partial \braket{O_y}(t)}{\partial \varepsilon} \\
        \frac{\partial \braket{O_y}(t)}{\partial \omega} \\
        \frac{\partial \braket{O_y}(t)}{\partial \gamma}
    \end{pmatrix} =
    \begin{pmatrix}
        -\tfrac{2}{\omega} \sin(\omega t) \mathrm{e}^{-2\gamma t} \cos(\Omega(t)) \\
        -\left( \tfrac{2\varepsilon t}{\omega} \cos(\omega t) - \tfrac{2\varepsilon}{\omega^2} \sin(\omega t)\right)\mathrm{e}^{-2\gamma t} \cos(\Omega(t)) \\
        2 t \mathrm{e}^{-2\gamma t} \sin(\Omega(t))
    \end{pmatrix} \\
    \nabla_\theta \braket{O_x}(t) &=
    \begin{pmatrix}
        \frac{\partial \braket{O_x}(t)}{\partial \varepsilon} \\
        \frac{\partial \braket{O_x}(t)}{\partial \omega} \\
        \frac{\partial \braket{O_x}(t)}{\partial \gamma}
    \end{pmatrix} =
    \begin{pmatrix}
        -\tfrac{2}{\omega} \sin(\omega t) \mathrm{e}^{-2\gamma t} \sin(\Omega(t)) \\
        -\left( \tfrac{2\varepsilon t}{\omega} \cos(\omega t) - \tfrac{2\varepsilon}{\omega^2} \sin(\omega t)\right)\mathrm{e}^{-2\gamma t} \sin(\Omega(t)) \\
        -2 t \mathrm{e}^{-2\gamma t} \cos(\Omega(t))
    \end{pmatrix}
\end{aligned}
$$
