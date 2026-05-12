## Interior Point Method for Rate-Independent Single Crystal Plasticity

The classical IPM (nested-loop, following Niehüser & Mosler 2023) solves the time-discrete KKT system using a log-barrier approach. The derivation proceeds in two stages: first the time-continuous IPM formulation, then time discretization.

---

### Time-Continuous Formulation

#### Slack Variables and the Barrier Problem

The constrained dissipation maximization (see `definitions.md`) is recast as a minimization problem with equality constraints by introducing slack variables $s^\alpha \geq 0$:
$$\arg \min_{\boldsymbol{\sigma},\, \tau_\text{h}^\alpha} \; -\mathfrak{D}_\text{red}(\boldsymbol{\sigma}, \tau_\text{h}^\alpha) \quad \text{s.t.} \quad \Phi^\alpha(\boldsymbol{\sigma}, \tau_\text{h}^\alpha) + s^\alpha = 0 \quad \forall\, \alpha, \quad s^\alpha \geq 0$$

Applying a logarithmic barrier to the slack variables yields the barrier problem:
$$\arg \min_{\boldsymbol{\sigma},\, \tau_\text{h}^\alpha} \; \mathfrak{D}_\mu(\boldsymbol{\sigma}, \tau_\text{h}^\alpha) = -\mathfrak{D}_\text{red} - \mu \sum_\alpha \ln(s^\alpha) \quad \text{s.t.} \quad \Phi^\alpha + s^\alpha = 0 \quad \forall\, \alpha$$
where $\mu > 0$ is a fixed barrier parameter. The logarithmic term enforces $s^\alpha > 0$ implicitly.

#### Lagrangian and Optimality

Incorporating the equality constraints via Lagrange multipliers $\lambda^\alpha \geq 0$:
$$\mathfrak{L}_\mu(\boldsymbol{\sigma}, \tau_\text{h}^\alpha, \lambda^\alpha, s^\alpha) = -\boldsymbol{\sigma} : \!\left(\sum_\alpha \dot{\gamma}^\alpha \boldsymbol{Z}^\alpha\right) + \sum_\alpha \lambda^\alpha \left(\Phi^\alpha(\boldsymbol{\sigma}, \tau_\text{h}^\alpha) + s^\alpha\right) - \mu \sum_\alpha \ln(s^\alpha)$$

The first-order optimality conditions of $\mathfrak{L}_\mu$ with respect to $s^\alpha$ yield:
$$\frac{\partial \mathfrak{L}_\mu}{\partial s^\alpha} = \lambda^\alpha - \frac{\mu}{s^\alpha} = 0 \quad \Longrightarrow \quad \lambda^\alpha s^\alpha = \mu$$

Stationarity with respect to $\boldsymbol{\sigma}$ further yields the flow rule with $\dot{\gamma}^\alpha = \lambda^\alpha$.

#### Time-Continuous Perturbed KKT Conditions

Collecting the optimality conditions gives the time-continuous perturbed KKT system:
$$\lambda^\alpha \geq 0, \qquad \Phi^\alpha(\boldsymbol{\sigma}, \tau_\text{h}^\alpha) + s^\alpha = 0, \qquad s^\alpha \geq 0, \qquad \lambda^\alpha s^\alpha = \mu$$

As $\mu \to 0$ the exact KKT conditions ($\Phi^\alpha \leq 0$, $\dot{\gamma}^\alpha \geq 0$, $\Phi^\alpha \dot{\gamma}^\alpha = 0$) are recovered. The solution strategy is to solve the perturbed system for a fixed $\mu$ and then successively decrease $\mu$ until sufficiently small.

---

### Time Discretization

The time-continuous equations are discretized over the load increment $[t_n, t_{n+1}]$ using a Backward Euler scheme. Subscripts $(\cdot)_{n+1}$ are omitted for conciseness; $(\cdot)_n$ denotes the previous converged step.

The slip rates are replaced by incremental slips $\Delta\gamma^\alpha = \gamma^\alpha - \gamma^\alpha_n \geq 0$, and the identification $\dot{\gamma}^\alpha = \lambda^\alpha$ becomes $\Delta\gamma^\alpha = \lambda^\alpha$. The plastic strain and accumulated slip update are:
$$\boldsymbol{\varepsilon}^\text{p} = \boldsymbol{\varepsilon}^\text{p}_n + \sum_\alpha \Delta\gamma^\alpha \, \boldsymbol{Z}^\alpha, \qquad A = \sum_\alpha \left(\gamma^\alpha_n + \Delta\gamma^\alpha\right)$$

The yield function evaluated at the current iterate $\Delta\boldsymbol{\gamma}$ becomes:
$$\Phi^\alpha(\Delta\boldsymbol{\gamma}) = \boldsymbol{Z}^\alpha : \boldsymbol{\sigma}(\Delta\boldsymbol{\gamma}) - \left(\tau_0 + \tau_\text{h}(A)\right)$$

Substituting $\Delta\gamma^\alpha = \lambda^\alpha$ into the perturbed KKT conditions gives the **time-discrete perturbed KKT system**:
$$\Delta\gamma^\alpha \geq 0, \qquad \Phi^\alpha(\Delta\boldsymbol{\gamma}) + s^\alpha = 0, \qquad s^\alpha \geq 0, \qquad \Delta\gamma^\alpha \cdot s^\alpha = \mu$$

#### Primal-Dual Residual System

For each slip system $\alpha$, the discrete perturbed KKT conditions are cast as a root-finding problem:
$$
\mathbf{R}^\alpha = \begin{bmatrix} R_g^\alpha \\[0.5em] R_s^\alpha \end{bmatrix} = \begin{bmatrix} \Phi^\alpha(\Delta\boldsymbol{\gamma}) + s^\alpha \\[0.5em] \Delta\gamma^\alpha \, s^\alpha - \mu \end{bmatrix} = \begin{bmatrix} 0 \\[0.5em] 0 \end{bmatrix}
$$
This is a nonlinear system of $2N$ equations for $N$ slip systems, solved by the nested-loop algorithm below.

---

### Nested-Loop Algorithm

#### Initialization

An elastic predictor is evaluated with $\Delta\boldsymbol{\gamma} = \mathbf{0}$. If $\Phi^\alpha \leq 0$ for all $\alpha$, the step is purely elastic and no IPM solve is needed.

Otherwise, the iterates are initialized as:
$$\mu_0 = \mu_\text{init}, \qquad s^\alpha_0 = \max\!\left(-\Phi^\alpha_\text{trial},\; \sqrt{\mu_0}\right), \qquad \Delta\gamma^\alpha_0 = \sqrt{\mu_0}$$

The initial residual norm $r_0 = \|\mathbf{R}_0\|$ is stored for relative convergence assessment.

#### Outer Barrier Loop (index $k$)

For each outer iteration $k$, an adaptive inner Newton tolerance is set:
$$\texttt{tol}_\text{inner} = \max(\texttt{tol}_\text{end},\; \theta \cdot \mu_k)$$

**Inner Newton loop:** while $\|\mathbf{R}\| / r_0 > \texttt{tol}_\text{inner}$, perform a Newton step on $\mathbf{R}$.

The Newton system for the correction $(\delta\!\Delta\boldsymbol{\gamma},\, \delta\mathbf{s})$ is:
$$
\begin{bmatrix} \dfrac{\partial \boldsymbol{\Phi}}{\partial \Delta\boldsymbol{\gamma}} & \mathbf{I} \\[0.8em] \operatorname{diag}(\mathbf{s}) & \operatorname{diag}(\Delta\boldsymbol{\gamma}) \end{bmatrix} \begin{bmatrix} \delta\!\Delta\boldsymbol{\gamma} \\[0.5em] \delta\mathbf{s} \end{bmatrix} = -\begin{bmatrix} \mathbf{R}_g \\[0.5em] \mathbf{R}_s \end{bmatrix}
$$
The Jacobian $\partial\boldsymbol{\Phi}/\partial\Delta\boldsymbol{\gamma}$ is computed via automatic differentiation (JAX).

**Fraction-to-boundary:** to maintain $\Delta\gamma^\alpha > 0$ and $s^\alpha > 0$, the step length is limited by:
$$\alpha_{\max} = \min\!\left(1,\; \min_{\delta\!\Delta\gamma^\alpha < 0} \frac{-\tau_\text{min}\,\Delta\gamma^\alpha}{\delta\!\Delta\gamma^\alpha},\; \min_{\delta s^\alpha < 0} \frac{-\tau_\text{min}\, s^\alpha}{\delta s^\alpha}\right)$$

Iterates are updated: $\Delta\boldsymbol{\gamma} \leftarrow \Delta\boldsymbol{\gamma} + \alpha_{\max}\,\delta\!\Delta\boldsymbol{\gamma}$, $\;\mathbf{s} \leftarrow \mathbf{s} + \alpha_{\max}\,\delta\mathbf{s}$.

**Barrier update** (after inner loop convergence, Niehüser 2023):
$$\mu_{k+1} = \max\!\left(\left(\max_\alpha \frac{\Delta\gamma^\alpha s^\alpha}{r_0}\right)^{1+\delta},\; \mu_\text{end}\right)$$

#### Convergence

The outer loop terminates when $\mu_k \leq \mu_\text{end}$ and $\|\mathbf{R}\| / r_0 < \texttt{tol}_\text{end}$.

---

### Consistent Tangent

At convergence, the consistent elastoplastic tangent $\mathbb{C}^\text{ep}$ is obtained via the implicit function theorem applied to $\mathbf{R}(\Delta\boldsymbol{\gamma}, \boldsymbol{\varepsilon}) = \mathbf{0}$:
$$\frac{\partial \Delta\boldsymbol{\gamma}}{\partial \boldsymbol{\varepsilon}} = -\left(\frac{\partial \mathbf{R}}{\partial \Delta\boldsymbol{\gamma}}\right)^{-1} \frac{\partial \mathbf{R}}{\partial \boldsymbol{\varepsilon}}$$

The fourth-order consistent tangent follows as:
$$\mathbb{C}^\text{ep}_{ijmn} = C_{ijmn} - C_{ijkl}\sum_\alpha Z^\alpha_{kl}\,\frac{\partial \Delta\gamma^\alpha}{\partial \varepsilon_{mn}}$$

Both Jacobians are computed via forward-mode automatic differentiation (JAX).

---

### Solver Parameters

| Parameter | Description |
|---|---|
| `mu_init` | Initial barrier parameter $\mu_0$ |
| `mu_end` | Final barrier parameter (convergence threshold) |
| `tol_end` | Final Newton tolerance $\texttt{tol}_\text{end}$ |
| `theta` | Adaptive inner tolerance factor |
| `delta` | Barrier reduction exponent |
| `tau_min` | Fraction-to-boundary safety factor |
| `k_max` | Maximum number of outer iterations |
| `max_inner` | Maximum Newton steps per outer iteration |
