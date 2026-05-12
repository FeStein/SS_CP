## Interior Point Method for Rate-Independent Single Crystal Plasticity

The classical IPM (nested-loop, following Niehüser & Mosler 2023) solves the time-discrete KKT system using a log-barrier approach. All quantities are per load step; the subscript $n$ denotes the previous converged step.

### Time Discretization

Over a load step from $t_n$ to $t_{n+1}$, incremental plastic slips $\Delta\gamma^\alpha \geq 0$ replace the time-continuous slip rates. The plastic strain update is:
$$\boldsymbol{\varepsilon}^\text{p}_{n+1} = \boldsymbol{\varepsilon}^\text{p}_n + \sum_\alpha \Delta\gamma^\alpha \, \boldsymbol{Z}^\alpha$$

The accumulated slip sum entering the hardening law becomes:
$$A = \sum_\alpha (\gamma^\alpha_n + \Delta\gamma^\alpha)$$

The yield function for each slip system $\alpha$ evaluated at the current iterate:
$$\Phi^\alpha(\Delta\boldsymbol{\gamma}) = \boldsymbol{Z}^\alpha : \boldsymbol{\sigma}(\Delta\boldsymbol{\gamma}) - (\tau_0 + \tau_\text{h}(A)) \leq 0$$

### Slack Variables

A slack variable $s^\alpha \geq 0$ converts the inequality $\Phi^\alpha \leq 0$ into an equality:
$$\Phi^\alpha + s^\alpha = 0$$

### Perturbed KKT Conditions

The complementary slackness condition $\Phi^\alpha \Delta\gamma^\alpha = 0$ is perturbed with a strictly positive barrier parameter $\mu > 0$:
$$\Delta\gamma^\alpha \cdot s^\alpha = \mu$$
where $\Delta\gamma^\alpha > 0$ and $s^\alpha > 0$. As $\mu \to 0$, the exact KKT conditions are recovered.

### Primal-Dual Residual System

For each slip system $\alpha$ the perturbed KKT conditions yield two equations:
$$
\mathbf{R}^\alpha = \begin{bmatrix} R_g^\alpha \\[0.5em] R_s^\alpha \end{bmatrix} = \begin{bmatrix} \Phi^\alpha(\Delta\boldsymbol{\gamma}) + s^\alpha \\[0.5em] \Delta\gamma^\alpha \, s^\alpha - \mu \end{bmatrix} = \begin{bmatrix} 0 \\[0.5em] 0 \end{bmatrix}
$$
This is a nonlinear system of $2N$ equations for $N$ slip systems.

### Nested-Loop Algorithm

#### Initialization

An elastic predictor is evaluated with $\Delta\boldsymbol{\gamma} = \mathbf{0}$. If $\Phi^\alpha \leq 0$ for all $\alpha$, the step is purely elastic and no IPM solve is needed.

Otherwise, the barrier parameter and iterates are initialized as:
$$\mu_0 = \mu_\text{init}, \qquad s^\alpha_0 = \max(-\Phi^\alpha_\text{trial},\; \sqrt{\mu_0}), \qquad \Delta\gamma^\alpha_0 = \sqrt{\mu_0}$$

The initial residual norm $r_0 = \|\mathbf{R}_0\|$ is stored for relative convergence assessment.

#### Outer Barrier Loop (index $k$)

For each outer iteration $k$, an adaptive inner Newton tolerance is set:
$$\texttt{tol}_\text{inner} = \max(\texttt{tol}_\text{end},\; \theta \cdot \mu_k)$$

**Inner Newton loop:** while $\|\mathbf{R}\| / r_0 > \texttt{tol}_\text{inner}$, perform a Newton step.

The Newton system for the correction $(\delta\!\Delta\boldsymbol{\gamma},\, \delta\mathbf{s})$ is:
$$
\begin{bmatrix} \dfrac{\partial \Phi}{\partial \Delta\boldsymbol{\gamma}} & \mathbf{I} \\[0.8em] \operatorname{diag}(\mathbf{s}) & \operatorname{diag}(\Delta\boldsymbol{\gamma}) \end{bmatrix} \begin{bmatrix} \delta\!\Delta\boldsymbol{\gamma} \\[0.5em] \delta\mathbf{s} \end{bmatrix} = -\begin{bmatrix} \mathbf{R}_g \\[0.5em] \mathbf{R}_s \end{bmatrix}
$$
The Jacobian $\partial\Phi/\partial\Delta\boldsymbol{\gamma}$ is computed via automatic differentiation (JAX).

**Fraction-to-boundary:** to maintain $\Delta\gamma^\alpha > 0$ and $s^\alpha > 0$, the step length is limited:
$$\alpha_{\max} = \min\!\left(1,\; \min_{\delta\!\Delta\gamma^\alpha < 0} \frac{-\tau_\text{min}\,\Delta\gamma^\alpha}{\delta\!\Delta\gamma^\alpha},\; \min_{\delta s^\alpha < 0} \frac{-\tau_\text{min}\, s^\alpha}{\delta s^\alpha}\right)$$

Iterates are updated as $\Delta\boldsymbol{\gamma} \leftarrow \Delta\boldsymbol{\gamma} + \alpha_{\max}\,\delta\!\Delta\boldsymbol{\gamma}$, $\mathbf{s} \leftarrow \mathbf{s} + \alpha_{\max}\,\delta\mathbf{s}$.

**Barrier update** (after inner loop convergence, Niehüser 2023):
$$\mu_{k+1} = \max\!\left(\left(\max_\alpha \frac{\Delta\gamma^\alpha s^\alpha}{r_0}\right)^{1+\delta},\; \mu_\text{end}\right)$$

#### Convergence

The outer loop terminates when $\mu_k \leq \mu_\text{end}$ and $\|\mathbf{R}\| / r_0 < \texttt{tol}_\text{end}$.

### Consistent Tangent

At convergence, the consistent elastoplastic tangent $\mathbb{C}^\text{ep}$ is obtained via the implicit function theorem applied to $\mathbf{R}(\Delta\boldsymbol{\gamma}, \boldsymbol{\varepsilon}) = \mathbf{0}$:
$$\frac{\partial \Delta\boldsymbol{\gamma}}{\partial \boldsymbol{\varepsilon}} = -\left(\frac{\partial \mathbf{R}}{\partial \Delta\boldsymbol{\gamma}}\right)^{-1} \frac{\partial \mathbf{R}}{\partial \boldsymbol{\varepsilon}}$$

The fourth-order consistent tangent follows as:
$$\mathbb{C}^\text{ep}_{ijmn} = C_{ijmn} - C_{ijkl}\sum_\alpha Z^\alpha_{kl}\,\frac{\partial \Delta\gamma^\alpha}{\partial \varepsilon_{mn}}$$

Both Jacobians are computed via forward-mode automatic differentiation (JAX).

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
