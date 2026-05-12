## Visco-Plasticity (Power-Law Flow Rule)

### Power-Law Flow Rule
The evolution of plastic slip on system $\alpha$ is governed by a power-law (Perzyna-type) relationship:
$$\dot{\gamma}^\alpha = \dot{\gamma}_0 \left\langle \frac{\tau^\alpha}{g^\alpha} \right\rangle^p$$
where:
* $\dot{\gamma}_0$ is the reference slip rate.
* $p$ is the rate sensitivity exponent.
* $\tau^\alpha = \boldsymbol{Z}^\alpha : \boldsymbol{\sigma}$ is the resolved shear stress on the slip system.
* $g^\alpha = \tau_0 + \tau_\text{h}^\alpha$ is the current slip resistance, combining the initial yield stress $\tau_0$ and the hardening stress $\tau_\text{h}^\alpha$.
* $\langle \bullet \rangle = \max(\bullet, 0)$ denotes the Macaulay bracket, ensuring plastic flow only occurs under positive resolved shear stress.

### Hardening
Currently, perfect plasticity is assumed ($\tau_\text{h}^\alpha = 0$), so $g^\alpha = \tau_0$.

For future extension, an exponential (Voce-type) isotropic hardening is defined via the plastic potential based on the total accumulated slip $A = \sum_\alpha \gamma^\alpha$:
$$\Psi^\text{p}(\gamma^\alpha) = \tau_\infty \left[ A + \frac{1}{\xi} \exp(- \xi A) \right]$$

The corresponding thermodynamic driving force (hardening stress):
$$\tau_\text{h}^\alpha(A) = \frac{\partial \Psi^\text{p}}{\partial \gamma^\alpha} = \tau_\infty \left[ 1 - \exp(-\xi A) \right]$$
where $\tau_\infty$ is the saturation stress and $\xi$ the hardening shape parameter. Since $\tau_\text{h}^\alpha$ depends only on $A$ (shared by all systems), this is a Taylor-type hardening.

### Time Discretization
Using a backward Euler integration scheme over a time step $\Delta t$, the plastic slip increment is approximated as:
$$\Delta \gamma^\alpha \approx \dot{\gamma}^\alpha \Delta t$$

We introduce a time-discrete reference slip parameter:
$$\Delta\bar{\gamma}_0 = \dot{\gamma}_0 \, \Delta t$$

### Time-Discrete Residual
Substituting the time discretization into the flow rule yields the non-linear residual equation for each slip system $\alpha$:
$$R^\alpha(\Delta \gamma) = \frac{\Delta \gamma^\alpha}{\Delta\bar{\gamma}_0} - \left\langle \frac{\tau^\alpha(\Delta \gamma)}{g^\alpha} \right\rangle^p = 0$$

### Newton-Raphson System
The Jacobian $\boldsymbol{J}$ is computed via forward-mode automatic differentiation (JAX). The iterative update solves:
$$\sum_\beta \frac{\partial R^\alpha}{\partial \Delta \gamma^\beta} \delta (\Delta \gamma^\beta) = - R^\alpha$$
or in matrix form: $\boldsymbol{J} \delta(\Delta \boldsymbol{\gamma}) = -\boldsymbol{R}$.

The plastic slip increments are updated using a backtracking line search to enforce the physical constraint $\Delta \gamma^\alpha \geq 0$:
$$\Delta \gamma_{k+1}^\alpha = \max\left( \Delta \gamma_k^\alpha + \alpha_{\text{step}} \, \delta(\Delta \gamma^\alpha), 0 \right)$$
where $\alpha_{\text{step}} \in (0, 1]$ is halved each iteration until the residual norm decreases.
