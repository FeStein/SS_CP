## Constitutive Model for Single Crystal Plasticity at Small Strains

### Strain Decomposition
Additive decomposition of the strain field into elastic and plastic parts:
$$\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}^\text{e} + \boldsymbol{\varepsilon}^\text{p}$$

### Helmholtz Free Energy
Decomposition of the free energy into elastic and plastic contributions:
$$\Psi(\boldsymbol{\varepsilon}^\text{e}, \gamma^\alpha) = \Psi^\text{e}(\boldsymbol{\varepsilon}^\text{e}) + \Psi^\text{p}(\gamma^\alpha)$$
where $\gamma^\alpha$ are the internal variables describing plastic slip on system $\alpha$.

### Elasticity
The elastic potential with the fourth-order linear elastic stiffness tensor $\mathbb{C}^\text{e}$:
$$\Psi^\text{e}(\boldsymbol{\varepsilon}^\text{e}) = \frac{1}{2} \boldsymbol{\varepsilon}^\text{e} : \mathbb{C}^\text{e} : \boldsymbol{\varepsilon}^\text{e}$$

Yielding the constitutive law for stress:
$$\boldsymbol{\sigma} = \frac{\partial \Psi^\text{e}}{\partial \boldsymbol{\varepsilon}^\text{e}} = \mathbb{C}^\text{e} : \boldsymbol{\varepsilon}^\text{e}$$

### Hardening
Exponential saturation hardening based on the total accumulated plastic slip $A = \sum_\alpha \gamma^\alpha$:
$$\Psi^\text{p}(\gamma^\alpha) = \tau_\infty \left[ A + \frac{1}{\xi} \exp(-\xi A) \right]$$

The thermodynamic hardening driving force:
$$\tau_\text{h}^\alpha = \frac{\partial \Psi^\text{p}}{\partial \gamma^\alpha} = \tau_\infty \left[ 1 - \exp(-\xi A) \right]$$
where $\tau_\infty$ is the saturation stress and $\xi$ the hardening shape parameter. Note that $\tau_\text{h}^\alpha$ is equal for all slip systems (Taylor-type hardening).

### Yield Criterion
Admissible stress states bounded by yield criteria for each slip system $\alpha$:
$$\Phi^\alpha = \tau^\alpha - (\tau_0 + \tau_\text{h}^\alpha) \leq 0 \quad \text{with} \quad \tau^\alpha = \boldsymbol{Z}^\alpha : \boldsymbol{\sigma}$$
where $\tau^\alpha$ is the resolved shear stress and $\boldsymbol{Z}^\alpha = \text{sym}(\boldsymbol{s}^\alpha \otimes \boldsymbol{n}^\alpha)$ is the Schmid tensor based on the slip direction $\boldsymbol{s}^\alpha$ and slip plane normal $\boldsymbol{n}^\alpha$.

### Flow Rule and Dissipation
Evolution of plastic strain based on the principle of maximum plastic dissipation:
$$\dot{\boldsymbol{\varepsilon}}^\text{p} = \sum_\alpha \dot{\gamma}^\alpha \, \boldsymbol{Z}^\alpha$$

Reduced dissipation inequality as a constrained optimization problem:
$$\max_{\boldsymbol{\sigma}, \tau_\text{h}^\alpha} \mathfrak{D}_\text{red} = \boldsymbol{\sigma} : \dot{\boldsymbol{\varepsilon}}^\text{p} - \sum_\alpha \tau_\text{h}^\alpha \dot{\gamma}^\alpha \quad \text{s.t.} \quad \Phi^\alpha(\boldsymbol{\sigma}, \tau_\text{h}^\alpha) \leq 0$$

### Karush-Kuhn-Tucker (KKT) Conditions
Stationarity of the Lagrangian yields the associated flow rule $\dot{\gamma}^\alpha = \dot{\lambda}^\alpha$. Admissible states are determined by the time-continuous KKT conditions:
$$\Phi^\alpha \leq 0, \quad \dot{\gamma}^\alpha \geq 0, \quad \Phi^\alpha \dot{\gamma}^\alpha = 0$$
