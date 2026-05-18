## Modified-Barrier Interior Point Method for Rate-Independent Single Crystal Plasticity

The classical IPM (see `ipm.md`) couples the barrier parameter $\mu$ to the constitutive response: a finite $\mu_\text{end}$ acts as an effective viscosity that biases the flow stress below $\tau_0$ (see `ipm_derive.md`). The Modified-Barrier IPM (MB-IPM), following Polyak's *nonlinear rescaling* (Polyak 1992; Polyak & Teboulle 1997), replaces the logarithmic barrier with a *shifted* variant that admits **explicit Lagrange-multiplier updates**. The shift parameter $\mu$ then plays a purely numerical role — it is held *fixed* during the inner solve and need not be driven to zero. Convergence to the exact rate-independent KKT solution is achieved via outer multiplier sweeps. This is the IPM counterpart of the augmented-Lagrange treatment of the visco-plastic model (see `vp_derive.md`).

---

### Time-Continuous Formulation

#### Modified Barrier Function

The constrained dissipation maximization (see `definitions.md`) is recast as in `ipm.md` by introducing slack variables $s^\alpha$ with $\Phi^\alpha + s^\alpha = 0$. The classical log-barrier $-\mu \sum_\alpha \ln(s^\alpha)$ is replaced by the **modified barrier** (Polyak):

$$B_\mu(\boldsymbol{s};\boldsymbol{\lambda}_k) = -\mu \sum_\alpha \lambda^\alpha_k\, \ln\!\left(1 + \frac{s^\alpha}{\mu}\right)$$

where $\lambda^\alpha_k \geq 0$ are the current Lagrange-multiplier estimates, **frozen during the inner solve** and updated only between outer sweeps. Key properties:

- **Finite at the original boundary**: $B_\mu(s^\alpha=0; \lambda_k) = 0$. The barrier no longer blows up at $s^\alpha = 0$.
- **Singularity shifted to $s^\alpha = -\mu$**: the implicit feasible region of the inner problem is *enlarged* to $s^\alpha > -\mu$, i.e. transient yield violations of magnitude up to $\mu$ are admissible during iteration.
- **Reproduces classical multiplier at the boundary**: $\partial B_\mu/\partial s^\alpha \big|_{s^\alpha=0} = -\lambda^\alpha_k$.

The barrier subproblem reads:
$$\arg\min_{\boldsymbol{\sigma},\,\tau_\text{h}^\alpha}\; -\mathfrak{D}_\text{red} - \mu\sum_\alpha \lambda^\alpha_k \ln(1 + s^\alpha/\mu) \quad \text{s.t.}\quad \Phi^\alpha + s^\alpha = 0$$

#### Lagrangian and Shifted Optimality

Introducing Lagrange multipliers $\lambda^\alpha \geq 0$ for the equality constraints:
$$\mathfrak{L}_\mu = -\boldsymbol{\sigma}:\!\left(\sum_\alpha \dot{\gamma}^\alpha \boldsymbol{Z}^\alpha\right) + \sum_\alpha \lambda^\alpha (\Phi^\alpha + s^\alpha) - \mu \sum_\alpha \lambda^\alpha_k \ln(1 + s^\alpha/\mu)$$

Stationarity with respect to $s^\alpha$:
$$\frac{\partial \mathfrak{L}_\mu}{\partial s^\alpha} = \lambda^\alpha - \frac{\lambda^\alpha_k}{1 + s^\alpha/\mu} = 0$$

Clearing the denominator yields the **shifted complementarity condition**:
$$\boxed{\lambda^\alpha\,(s^\alpha + \mu) = \mu\,\lambda^\alpha_k}$$

Compared with the classical perturbed condition $\lambda^\alpha s^\alpha = \mu$:
- The slack is shifted by $\mu$ inside the product.
- The right-hand side carries the previous multiplier estimate.
- **At the fixed point** $\lambda^\alpha_{k+1} = \lambda^\alpha_k = \lambda^\alpha_*$, dividing the converged identity by $\lambda^\alpha_*$ (when $\lambda^\alpha_* > 0$) gives $s^\alpha_* = 0$; if $\lambda^\alpha_* = 0$, the relation is trivially satisfied. Either branch is the **exact** KKT complementarity $\lambda^\alpha_* s^\alpha_* = 0$ — *with no $\mu$ residual*. This is the structural reason why a finite $\mu$ does not bias the converged solution.

Stationarity with respect to $\boldsymbol{\sigma}$ is unchanged from the classical IPM and yields the flow rule with $\dot{\gamma}^\alpha = \lambda^\alpha$.

#### Time-Continuous Shifted KKT and Flow Rule

Collecting:
$$\lambda^\alpha \geq 0, \qquad \Phi^\alpha + s^\alpha = 0, \qquad s^\alpha > -\mu, \qquad \lambda^\alpha(s^\alpha+\mu) = \mu\,\lambda^\alpha_k$$

Substituting $s^\alpha = -\Phi^\alpha$:
$$\boxed{\dot{\gamma}^\alpha = \frac{\mu\,\lambda^\alpha_k}{\mu - \Phi^\alpha} = \frac{\mu\,\lambda^\alpha_k}{\mu + g^\alpha - \tau^\alpha}}$$

Contrast with the classical IPM flow rule $\dot{\gamma}^\alpha = \mu/(-\Phi^\alpha)$:

- **Bounded as $-\Phi^\alpha \to 0$**: the rate saturates at $\lambda^\alpha_k$ rather than diverging. The shift acts as a regularization in the denominator.
- **Suppressed on inactive systems**: for $-\Phi^\alpha \gg \mu$, $\dot{\gamma}^\alpha \to \mu\lambda^\alpha_k/(-\Phi^\alpha)$, which is small both because of the $\mu$ factor *and* because $\lambda^\alpha_k$ decays toward zero on inactive systems under repeated multiplier updates.
- **Exact at the fixed point**: $\dot{\gamma}^\alpha_* = \lambda^\alpha_*$ is the true rate-independent slip rate, independent of $\mu$.

This is the structural analogue of the augmented-Lagrange visco-plasticity treatment: a single-parameter regularization (where the parameter must go to zero) is replaced by a multiplier-augmented form where the parameter stays finite and explicit multiplier updates drive convergence.

---

### Time Discretization

Backward Euler over $[t_n, t_{n+1}]$ with the identifications $\Delta\gamma^\alpha = \lambda^\alpha$, $A = \sum_\alpha (\gamma^\alpha_n + \Delta\gamma^\alpha)$, and $\Phi^\alpha(\Delta\boldsymbol{\gamma}) = \boldsymbol{Z}^\alpha:\boldsymbol{\sigma}(\Delta\boldsymbol{\gamma}) - (\tau_0 + \tau_\text{h}(A))$. The **time-discrete shifted KKT system** is:

$$\Delta\gamma^\alpha \geq 0, \qquad \Phi^\alpha + s^\alpha = 0, \qquad s^\alpha > -\mu, \qquad \Delta\gamma^\alpha (s^\alpha + \mu) = \mu\,\lambda^\alpha_k$$

#### Primal-Dual Residual System

For fixed $\mu$ and fixed multiplier estimates $\boldsymbol{\lambda}_k$, the inner problem reads:
$$\mathbf{R}^\alpha = \begin{bmatrix} R_g^\alpha \\[0.5em] R_s^\alpha \end{bmatrix} = \begin{bmatrix} \Phi^\alpha(\Delta\boldsymbol{\gamma}) + s^\alpha \\[0.5em] \Delta\gamma^\alpha(s^\alpha + \mu) - \mu\,\lambda^\alpha_k \end{bmatrix} = \mathbf{0}$$

The Newton system for the correction $(\delta\!\Delta\boldsymbol{\gamma}, \delta\mathbf{s})$:
$$\begin{bmatrix} \dfrac{\partial \boldsymbol{\Phi}}{\partial \Delta\boldsymbol{\gamma}} & \mathbf{I} \\[0.8em] \operatorname{diag}(\mathbf{s}+\mu) & \operatorname{diag}(\Delta\boldsymbol{\gamma}) \end{bmatrix}\!\begin{bmatrix} \delta\!\Delta\boldsymbol{\gamma} \\[0.5em] \delta\mathbf{s} \end{bmatrix} = -\begin{bmatrix} \mathbf{R}_g \\[0.5em] \mathbf{R}_s \end{bmatrix}$$

Structurally identical to the classical IPM system (`ipm.md`) except the (2,1) block carries the shift $\mathbf{s} + \mu$ and the (2) residual uses $\mu\,\boldsymbol{\lambda}_k$ instead of $\mu$. The yield-Jacobian $\partial\boldsymbol{\Phi}/\partial\Delta\boldsymbol{\gamma}$ is unchanged and is still obtained by automatic differentiation (JAX).

---

### Nested-Loop Algorithm

#### Initialization

Elastic predictor at $\Delta\boldsymbol{\gamma} = \mathbf{0}$. If $\Phi^\alpha \leq 0$ for all $\alpha$, the step is purely elastic.

Otherwise:
$$\mu_0 = \mu_\text{init}, \qquad \lambda^\alpha_{k=0} = \lambda_\text{init} > 0, \qquad s^\alpha_0 = \max(-\Phi^\alpha_\text{trial},\,\sqrt{\mu_0}), \qquad \Delta\gamma^\alpha_0 = \sqrt{\mu_0}$$

The multiplier estimates are initialized to a uniform *positive* value (e.g. $\lambda_\text{init} = 1$). Initializing at zero is forbidden: the update rule below preserves zero as a fixed point and would lock such a system out permanently.

#### Outer Lagrange-Multiplier Loop (index $k$)

**1. Inner Newton solve** at fixed $(\mu_k, \boldsymbol{\lambda}_k)$, iterating the Newton system above with the fraction-to-boundary safeguard
$$\alpha_\text{max} = \min\!\left(1,\; \min_{\delta\!\Delta\gamma^\alpha < 0}\frac{-\tau_\text{min}\,\Delta\gamma^\alpha}{\delta\!\Delta\gamma^\alpha},\; \min_{\delta s^\alpha < 0}\frac{-\tau_\text{min}\,(s^\alpha + \mu)}{\delta s^\alpha}\right)$$
until $\|\mathbf{R}\|/r_0 < \texttt{tol}_\text{inner}$. Note the slack constraint is on $s^\alpha + \mu > 0$, not $s^\alpha > 0$ — temporary mild yield violation is allowed during iteration, which makes long Newton steps acceptable.

**2. Multiplier update** from the converged inner state:
$$\boxed{\lambda^\alpha_{k+1} = \frac{\mu_k\,\lambda^\alpha_k}{s^\alpha_* + \mu_k}}$$
This is the stationarity condition rearranged. It preserves positivity, decays on inactive systems ($s^\alpha_* \gg \mu_k \Rightarrow \lambda^\alpha_{k+1} \ll \lambda^\alpha_k$), and remains essentially unchanged on active ones ($s^\alpha_* \approx 0 \Rightarrow \lambda^\alpha_{k+1} \approx \lambda^\alpha_k$).

**3. (Optional) Shift update**. Unlike the classical IPM, driving $\mu_k \to 0$ is **not** required: Polyak's theorem guarantees that for any $\mu < \mu^*$ (a problem-dependent threshold) the multiplier sequence converges linearly to the true KKT multipliers. The classical choice is therefore a fixed $\mu$:
$$\mu_{k+1} = \max(\rho\,\mu_k,\,\mu_\text{floor}), \qquad 0 < \rho \leq 1$$
A safe default is $\rho = 1$ (fixed $\mu$); for stiffer problems, $\rho \in [0.5, 1]$ accelerates convergence at the cost of slightly harder inner Newton steps.

**4. Outer convergence**. Terminate when the multipliers have stabilized,
$$\max_\alpha \frac{|\lambda^\alpha_{k+1} - \lambda^\alpha_k|}{\max(1,\lambda^\alpha_k)} < \texttt{tol}_\text{mult},$$
or equivalently when the *unshifted* complementarity gap measured at the converged inner state is below threshold:
$$\max_\alpha |\lambda^\alpha_{k+1}\,s^\alpha_*| < \texttt{tol}_\text{compl}$$

The latter is the direct diagnostic that the *exact* KKT (not the perturbed one) is satisfied.

---

### Consistent Tangent

At outer convergence both the primal and the multiplier iterates are fixed, and the residual that defines the converged map $\Delta\boldsymbol{\gamma}(\boldsymbol{\varepsilon})$ is the **exact** KKT residual: $\Phi^\alpha = 0$ on active systems with $\lambda^\alpha_* > 0$, and $\Delta\gamma^\alpha = 0$ with $\Phi^\alpha < 0$ on inactive systems. The shift $\mu$ has dropped out of every relation that matters for sensitivity.

Therefore the consistent tangent reduces to the classical rate-independent expression with no $\mu$-dependent bias:
$$\frac{\partial \Delta\boldsymbol{\gamma}}{\partial \boldsymbol{\varepsilon}} = -\left(\frac{\partial \mathbf{R}}{\partial \Delta\boldsymbol{\gamma}}\right)^{\!-1}\!\frac{\partial \mathbf{R}}{\partial \boldsymbol{\varepsilon}}, \qquad \mathbb{C}^\text{ep}_{ijmn} = C_{ijmn} - C_{ijkl}\sum_\alpha Z^\alpha_{kl}\,\frac{\partial \Delta\gamma^\alpha}{\partial \varepsilon_{mn}}$$

Both Jacobians are computed via forward-mode AD. This is the central practical payoff: the converged stress and tangent are **not polluted by a finite $\mu$**, in contrast to the classical IPM (cf. the experiments in `ipm_derive.md`).

---

### Dimensional Note and Normalization

The shifted complementarity in the discrete case carries the same units as in the classical IPM: $[\Delta\gamma^\alpha (s^\alpha+\mu)] = [\mu\,\lambda^\alpha_k]$ implies $[\mu] = \text{Pa}$ (since $[\Delta\gamma]$ and $[\lambda_k]$ are dimensionless and $[s] = \text{Pa}$).

A fully dimensionless variant is obtained by using $\tau_0$ as a reference, exactly as in `ipm_derive.md`:
$$\tilde{\mu} = \mu/\tau_0, \qquad B_{\tilde\mu} = -\tilde{\mu}\,\tau_0\sum_\alpha \lambda^\alpha_k \ln\!\left(1 + \frac{s^\alpha}{\tilde{\mu}\,\tau_0}\right)$$

with the corresponding shifted condition $\Delta\gamma^\alpha(s^\alpha + \tilde\mu\,\tau_0) = \tilde\mu\,\tau_0\,\lambda^\alpha_k$. Because $\mu$ is no longer a convergence parameter (it stays fixed), the choice $\tilde\mu = O(10^{-2})\text{–}O(10^{-1})$ relative to $\tau_0$ is typically adequate.

---

### Comparison: Classical IPM vs. MB-IPM vs. Aug-Lagrange VP

| | Classical IPM | MB-IPM | Aug-Lagrange VP |
|---|---|---|---|
| Side of yield surface | interior ($\Phi \leq 0$) | shifted-interior ($\Phi < \mu$) | exterior penalized |
| Regularizer | $-\mu \ln s^\alpha$ | $-\mu\lambda^\alpha_k\ln(1+s^\alpha/\mu)$ | $\tfrac{1}{2\eta}\langle\Phi^\alpha+\eta\lambda^\alpha_k\rangle^2$ |
| Perturbed complementarity | $\lambda^\alpha s^\alpha = \mu$ | $\lambda^\alpha(s^\alpha+\mu)=\mu\lambda^\alpha_k$ | $\lambda^\alpha_{k+1}=\max(0,\lambda^\alpha_k+\Phi^\alpha/\eta)$ |
| Multiplier update | implicit (via $\mu \to 0$) | explicit (Polyak update) | explicit (projection update) |
| Parameter at convergence | must $\to 0$ | may stay finite | may stay finite |
| Bias on $\tau_\text{ss}$ at convergence | $\delta\tau = \mu_\text{end}/\Delta\gamma_\text{ss}$ | none | none |
| Flow stress during iteration | $\tau < \tau_0$ (undershoot) | undershoot $\to 0$ as $k\to\infty$ | overshoot $\to 0$ as $k\to\infty$ |

The MB-IPM is the structural analogue of the augmented-Lagrange treatment of the VP model — both replace a single-parameter regularization (parameter $\to 0$ required) with a multiplier-augmented form (parameter finite, explicit updates drive convergence).

---

### Solver Parameters

| Parameter | Description |
|---|---|
| `mu_init` | Initial shift parameter $\mu_0$ (typically $\sim 10^{-2}\tau_0$) |
| `mu_floor` | Lower bound for $\mu_k$ (often $= \mu_\text{init}$, i.e. fixed shift) |
| `rho` | Outer reduction factor for $\mu_k$ (default 1.0 = fixed) |
| `lambda_init` | Initial multiplier estimate $\lambda^\alpha_0$ (default 1) |
| `tol_inner` | Inner Newton tolerance |
| `tol_mult` | Outer multiplier-stagnation tolerance |
| `tol_compl` | Complementarity-gap tolerance $\max_\alpha |\lambda^\alpha_*\,s^\alpha_*|$ |
| `tau_min` | Fraction-to-boundary safety factor (applied to $s^\alpha+\mu$, not $s^\alpha$) |
| `k_max` | Maximum outer multiplier iterations |
| `max_inner` | Maximum Newton steps per outer iteration |

---

### References

- B. T. Polyak, *Modified barrier functions: theory and methods*, Mathematical Programming **54** (1992), 177–222.
- R. Polyak & M. Teboulle, *Nonlinear rescaling and proximal-like methods in convex optimization*, Mathematical Programming **76** (1997), 265–284.
- F. Niehüser & J. Mosler, *An interior point method for rate-independent single crystal plasticity*, 2023 (basis of `ipm.md`).
