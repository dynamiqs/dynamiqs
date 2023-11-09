# Summary of benchmark systems

## Transmon qubit pi-pulse

$$
    H = 4 E_C (n - n_g)^2 - E_J \cos(\varphi) + \varepsilon(t) \cos(\omega_d t + \phi_d) n
$$
where $\omega_d = \omega_{eg}$ is the transmon frequency, $\varepsilon(t)$ is the drive amplitude, and $\phi_d$ is the drive phase.

For a gaussian drive pulse, the drive amplitude is given by
$$
    \varepsilon(t) = \varepsilon_0 (\exp\left(-\frac{(t - \frac{T}{2})^2}{2 T_0^2}\right) - \exp\left(-\frac{T^2}{8 T_0^2}\right))
$$
where $T$ is the gate time, $T_0$ is the gaussian width, and $\varepsilon_0$ is the maximum amplitude. $\varepsilon_0$ is such that
$$
    \pi = \int_0^T \varepsilon(t) dt
$$
