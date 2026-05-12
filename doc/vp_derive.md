## Derivation of the Visco-Plastic Flow Rule via Penalty Regularization

### Starting Point: Principle of Maximum Plastic Dissipation

The reduced dissipation inequality (see `definitions.md`) is cast as a constrained minimization over the stress $\boldsymbol{\sigma}$ and hardening stress $\tau_\text{h}^\alpha$, with the slip rates $\dot{\gamma}^\alpha$ treated as fixed primal data:

$$\min_{\boldsymbol{\sigma},\, \tau_\text{h}^\alpha} \; -\mathfrak{D}_\text{red} = -\sum_\alpha \bigl(\tau^\alpha - \tau_\text{h}^\alpha\bigr) \dot{\gamma}^\alpha \quad \text{s.t.} \quad \Phi^\alpha(\boldsymbol{\sigma}, \tau_\text{h}^\alpha) \leq 0 \;\; \forall\, \alpha$$

where $\tau^\alpha = \boldsymbol{Z}^\alpha : \boldsymbol{\sigma}$ and $\Phi^\alpha = \tau^\alpha - g^\alpha$ with $g^\alpha = \tau_0 + \tau_\text{h}^\alpha$.

The KKT conditions of this problem yield the rate-independent flow rule $\dot{\gamma}^\alpha = \lambda^\alpha$. The IPM solves this system by introducing a logarithmic barrier on the constraint (see `ipm.md`).

---

### Penalty Regularization

Instead of the log-barrier, the constraint $\Phi^\alpha \leq 0$ is relaxed by adding a **quadratic penalty** on constraint violations, turning the constrained problem into an unconstrained one:

$$\min_{\boldsymbol{\sigma},\, \tau_\text{h}^\alpha} \; -\mathfrak{D}_\text{red} + \frac{1}{2\eta} \sum_\alpha \langle \Phi^\alpha \rangle^2$$

where $\langle \bullet \rangle = \max(\bullet, 0)$ is the Macaulay bracket (penalty is active only when $\Phi^\alpha > 0$, i.e. when the yield surface is violated) and $\eta > 0$ is the viscosity parameter.

---

### Stationarity and the Flow Rule

Taking the stationarity condition with respect to $\tau^\alpha$ (via $\boldsymbol{\sigma}$), using $\partial\Phi^\alpha / \partial\tau^\alpha = 1$:

$$\frac{\partial}{\partial \tau^\alpha}\left[ -\tau^\alpha \dot{\gamma}^\alpha + \frac{1}{2\eta}\langle\Phi^\alpha\rangle^2 \right] = 0$$

$$-\dot{\gamma}^\alpha + \frac{1}{\eta}\langle\Phi^\alpha\rangle = 0$$

$$\boxed{\dot{\gamma}^\alpha = \frac{1}{\eta} \langle \Phi^\alpha \rangle = \frac{1}{\eta} \langle \tau^\alpha - g^\alpha \rangle}$$

This is a **linear overstress (Bingham-type) viscoplastic flow rule**: slip is driven by the amount by which the resolved shear stress exceeds the slip resistance $g^\alpha$.

---

### Analogy with the IPM

Both models regularize the identical rate-independent constrained problem, differing only in the choice of regularization function:

| | IPM (log-barrier) | VP (quadratic penalty) |
|---|---|---|
| Regularized objective | $-\mathfrak{D}_\text{red} - \mu \sum_\alpha \ln(-\Phi^\alpha)$ | $-\mathfrak{D}_\text{red} + \dfrac{1}{2\eta}\sum_\alpha\langle\Phi^\alpha\rangle^2$ |
| Flow rule | $\dot{\gamma}^\alpha = \dfrac{\mu}{g^\alpha - \tau^\alpha}$ | $\dot{\gamma}^\alpha = \dfrac{1}{\eta}\langle\Phi^\alpha\rangle$ |
| Regularization domain | interior: $\Phi^\alpha < 0$ | exterior: $\Phi^\alpha > 0$ |
| Rate-independent limit | $\mu \to 0$ | $\eta \to 0$ |

The key structural difference is that the log-barrier is defined strictly **inside** the feasible set and prevents crossing the yield surface, while the quadratic penalty acts **outside** and pulls violated states back toward the yield surface. Both recover the rate-independent KKT conditions in their respective limits.

---

### Perzyna-Type Generalization (Power-Law Penalty)

For use as a pseudo-viscous regularization of rate-independent plasticity, the quadratic penalty is generalized to an arbitrary exponent $p \geq 1$. To keep the argument of the Macaulay bracket dimensionless, the yield function is normalized by a reference stress $\tau_\text{ref}$ (a natural choice is the initial slip resistance $\tau_\text{ref} = \tau_0$):

$$\min_{\boldsymbol{\sigma},\, \tau_\text{h}^\alpha} \; -\mathfrak{D}_\text{red} + \frac{\tau_\text{ref}}{\eta\,(p+1)} \sum_\alpha \left\langle \frac{\Phi^\alpha}{\tau_\text{ref}} \right\rangle^{p+1}$$

Stationarity with respect to $\tau^\alpha$, using $\partial\Phi^\alpha/\partial\tau^\alpha = 1$:

$$-\dot{\gamma}^\alpha + \frac{1}{\eta} \left\langle \frac{\Phi^\alpha}{\tau_\text{ref}} \right\rangle^{p} = 0$$

$$\boxed{\dot{\gamma}^\alpha = \frac{1}{\eta} \left\langle \frac{\Phi^\alpha}{\tau_\text{ref}} \right\rangle^p = \frac{1}{\eta} \left\langle \frac{\tau^\alpha - g^\alpha}{\tau_\text{ref}} \right\rangle^p}$$

The linear case $p = 1$ recovers the Bingham model above (up to normalization). For $p > 1$ the model is a **Perzyna-type power-law overstress model**: the flow rate grows nonlinearly with the normalized overstress $\Phi^\alpha / \tau_\text{ref}$.

#### Rate-Independent Limit

Two equivalent routes recover the rate-independent KKT conditions:

- **$\eta \to 0$** (fixed $p$): the penalty becomes infinitely stiff, forcing $\Phi^\alpha \leq 0$.
- **$p \to \infty$** (fixed $\eta$): the power-law sharpens into a step function at $\Phi^\alpha = 0$.

For pseudo-viscous regularization of rate-independent crystal plasticity, $p$ is chosen large (e.g. $p = 20$–$50$) and $\eta$ small but finite, so that the flow rule closely approximates the KKT conditions while keeping the system smooth and well-conditioned for Newton's method.
