# Regularized Solvers for Rate-Independent Crystal Plasticity: Primal–Dual Systems and Conditioning

Concise backbone comparing the classical interior-point (IPM), modified-barrier
IPM (MB-IPM), augmented-Lagrangian (AL) and viscoplastic-penalty (VP) solvers for
the **perfectly plastic** small-strain single crystal. See
[definitions.md](definitions.md) for the full constitutive model.

## 1. Discrete problem (perfect plasticity, small strain)

Backward-Euler increment over one step; perfect plasticity sets $\tau_\text{h}^\alpha = 0$.
The unknowns are the slip increments $\Delta\gamma^\alpha \ge 0$, $\alpha = 1,\dots,m$:

$$
\boldsymbol{\sigma} = \mathbb{C} : \big(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^\text{p}_n - \textstyle\sum_\beta \Delta\gamma^\beta \boldsymbol{Z}^\beta\big),
\qquad
\Phi^\alpha(\Delta\boldsymbol{\gamma}) = \boldsymbol{Z}^\alpha : \boldsymbol{\sigma} - \tau_0 .
$$

Because $\boldsymbol{\sigma}$ is affine in $\Delta\boldsymbol{\gamma}$, the yield function is **affine**,

$$
\Phi^\alpha(\Delta\boldsymbol{\gamma}) = \Phi^\alpha_\text{tr} - \sum_\beta G^{\alpha\beta}\,\Delta\gamma^\beta,
\qquad
G^{\alpha\beta} := \boldsymbol{Z}^\alpha : \mathbb{C} : \boldsymbol{Z}^\beta,
\qquad
\frac{\partial \Phi^\alpha}{\partial \Delta\gamma^\beta} = -\,G^{\alpha\beta},
$$

with the trial value $\Phi^\alpha_\text{tr} = \boldsymbol{Z}^\alpha : \mathbb{C} : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^\text{p}_n) - \tau_0$.
The discrete KKT system is then a **linear complementarity problem (LCP)**:

$$
\Phi^\alpha \le 0, \qquad \Delta\gamma^\alpha \ge 0, \qquad \Delta\gamma^\alpha\,\Phi^\alpha = 0 .
$$

- $\boldsymbol{G} = [G^{\alpha\beta}]$ is symmetric positive **semi**-definite — the slip-space stiffness.
- For FCC the $m=24$ Schmid tensors are deviatoric and span only **5 dimensions**
  ($\boldsymbol{Z}^{\alpha+12} = -\boldsymbol{Z}^\alpha$), so $\operatorname{rank}\boldsymbol{G} \le 5$. Any active
  set with $>5$ systems makes $\boldsymbol{G}$ on that set **singular** (Taylor slip ambiguity).
- All four solvers replace the non-smooth complementarity by a parametrized
  regularization; each ends in a Newton matrix of the generic slip-space form
  $\boxed{\;\boldsymbol{D}(\text{param}) + \boldsymbol{B}\,\boldsymbol{G}\;}$ with a diagonal/identity
  regularizer $\boldsymbol{D}$. **It is $\boldsymbol{D}$ that regularizes $\operatorname{null}(\boldsymbol{G})$, and hence sets the conditioning.**

## 2. Primal–dual systems

### 2.1 Classical IPM — barrier $\mu \to 0$

Slacks $s^\alpha = -\Phi^\alpha \ge 0$, perturbed complementarity $\Delta\gamma^\alpha s^\alpha = \mu$:

$$
\boldsymbol{R} =
\begin{bmatrix} \boldsymbol{\Phi} + \boldsymbol{s} \\[2pt] \operatorname{diag}(\Delta\boldsymbol{\gamma})\,\boldsymbol{s} - \mu\,\boldsymbol{1} \end{bmatrix},
\qquad
\underbrace{\begin{bmatrix} -\boldsymbol{G} & \boldsymbol{I} \\[2pt] \operatorname{diag}(\boldsymbol{s}) & \operatorname{diag}(\Delta\boldsymbol{\gamma}) \end{bmatrix}}_{\boldsymbol{J}\;(2m\times 2m)}
\begin{bmatrix} \mathrm{d}\Delta\boldsymbol{\gamma} \\ \mathrm{d}\boldsymbol{s} \end{bmatrix} = -\boldsymbol{R}.
$$

Schur complement onto $\Delta\boldsymbol{\gamma}$:
$\;\boldsymbol{M}_\text{IPM} = \operatorname{diag}(\boldsymbol{s}) + \operatorname{diag}(\Delta\boldsymbol{\gamma})\,\boldsymbol{G}.$

### 2.2 Modified-barrier IPM — shift $\mu$, multiplier $\lambda_k$

Shifted complementarity $\Delta\gamma^\alpha (s^\alpha + \mu) = \mu\,\lambda_k^\alpha$ with $s = -\Phi$, frozen $\lambda_k$:

$$
\boldsymbol{R} =
\begin{bmatrix} \boldsymbol{\Phi} + \boldsymbol{s} \\[2pt] \operatorname{diag}(\Delta\boldsymbol{\gamma})\,(\boldsymbol{s}+\mu) - \mu\,\boldsymbol{\lambda}_k \end{bmatrix},
\qquad
\boldsymbol{M}_\text{MB} = \operatorname{diag}(\boldsymbol{s}+\mu) + \operatorname{diag}(\Delta\boldsymbol{\gamma})\,\boldsymbol{G}.
$$

Outer Polyak update $\;\lambda_{k+1}^\alpha = \mu\,\lambda_k^\alpha / (s^\alpha + \mu)$. KKT is recovered exactly for any fixed $\mu$ as $\lambda_k$ converges.

### 2.3 Augmented Lagrangian — penalty $\eta \to \infty$

Semi-smooth residual and active set $\mathcal{A} = \{\alpha : \Delta\gamma^\alpha + \eta\,\Phi^\alpha > 0\}$:

$$
R^\alpha = \Delta\gamma^\alpha - \max\!\big(0,\; \Delta\gamma^\alpha + \eta\,\Phi^\alpha\big),
\qquad
\boldsymbol{F} = \boldsymbol{I} + \eta\,\boldsymbol{G}\big|_{\mathcal{A}} \;\;\text{(identity on inactive rows)} .
$$

Newton system $\boldsymbol{F}\,\mathrm{d}\Delta\boldsymbol{\gamma} = -\boldsymbol{R}$; outer loop raises $\eta$ until complementarity holds.

### 2.4 Viscoplastic penalty — viscosity $\eta \to 0$ (Perzyna, $p=1$)

Rate regularization $\Delta\gamma^\alpha / \Delta\gamma_0 = \langle \Phi^\alpha/\tau_0\rangle^p$, $\Delta\gamma_0 = \Delta t/\eta$:

$$
r^\alpha = \frac{\Delta\gamma^\alpha}{\Delta\gamma_0} - \Big\langle \frac{\Phi^\alpha}{\tau_0}\Big\rangle^{p},
\qquad
\frac{\partial \boldsymbol{r}}{\partial \Delta\boldsymbol{\gamma}}\Big|_{p=1,\,\mathcal{A}} = \frac{\eta}{\Delta t}\,\boldsymbol{I} + \frac{1}{\tau_0}\,\boldsymbol{G}\big|_{\mathcal{A}} .
$$

The KKT limit is recovered only as $\eta \to 0$ (vanishing overstress).

## 3. Conditioning vs. the regularization parameter

All slip-space matrices share the form $\boldsymbol{D} + \boldsymbol{B}\boldsymbol{G}$; the smallest eigenvalue
lives on $\operatorname{null}(\boldsymbol{G})$ (the Taylor null space) and is fixed by $\boldsymbol{D}$:

- **Classical IPM** — $\boldsymbol{D} = \operatorname{diag}(\boldsymbol{s})$, and active slacks $s^\alpha \sim \mu \to 0$, so
  $\kappa(\boldsymbol{M}_\text{IPM}) \sim \mathcal{O}(\mu^{-1})$. The full $2m\times2m$ $\boldsymbol{J}$ carries an
  **additional** $\mathcal{O}(\mu^{-1})$ from block scaling (active $s\sim\mu$, inactive $\Delta\gamma\sim\sqrt\mu$).
  This is *benign, structured* ill-conditioning (the Newton step stays accurate), but it grows without bound as $\mu\to0$.

- **MB-IPM** — $\boldsymbol{D} = \operatorname{diag}(\boldsymbol{s}+\mu) \ge \mu > 0$: the shift floors the null
  space, $\kappa(\boldsymbol{M}_\text{MB}) \sim \mathcal{O}(\mu^{-1})$ **but $\mu$ is held at a moderate value**
  (never driven to 0 — exactness comes from the $\lambda_k$ update, not $\mu\to0$). Smaller $\mu$ ⇒ faster outer rate but higher $\kappa$; APV self-tunes $\mu$ to the largest stable value.

- **Augmented Lagrangian** — $\boldsymbol{D} = \boldsymbol{I}$ **independent of $\eta$**: on
  $\operatorname{null}(\boldsymbol{G})$ the matrix equals $\boldsymbol{I}$, so $\boldsymbol{F}$ is *never singular*.
  $\kappa(\boldsymbol{F}) \approx 1 + \eta\,\lambda_\text{max}(\boldsymbol{G}) \sim \mathcal{O}(\eta)$ — grows only
  **linearly** with the penalty, with a perfectly conditioned null space ⇒ lowest, most robust $\kappa$.

- **Viscoplastic penalty** — $\boldsymbol{D} = (\eta/\Delta t)\,\boldsymbol{I}$ is the *only* term
  regularizing $\operatorname{null}(\boldsymbol{G})$, so $\lambda_\text{min} = \eta/\Delta t$ and
  $\kappa \approx \dfrac{\lambda_\text{max}(\boldsymbol{G})/\tau_0}{\eta/\Delta t} \sim \mathcal{O}(\eta^{-1})$:
  conditioning **blows up as $\eta \to 0$**, precisely the rate-independent limit. (With $p=1$ the
  problem is piecewise-linear, so Newton still converges in one step per active set regardless of $\kappa$ — fast iterations, ill-conditioned matrix.)

### Summary

| Method | Regularizer $\boldsymbol{D}$ | Limit | $\kappa$ scaling | Null space |
|---|---|---|---|---|
| Classical IPM | $\operatorname{diag}(\boldsymbol{s})$ | $\mu \to 0$ | $\mathcal{O}(\mu^{-1})$ (+ $\mu^{-1}$ in $2m$ system) | collapses as $\mu\to0$ |
| MB-IPM | $\operatorname{diag}(\boldsymbol{s}+\mu)$ | $\mu$ fixed | $\mathcal{O}(\mu^{-1})$, $\mu$ moderate | floored at $\mu$ |
| Augmented Lagrangian | $\boldsymbol{I}$ | $\eta \to \infty$ | $\mathcal{O}(\eta)$ | conditioned by $\boldsymbol{I}$ |
| Viscoplastic penalty | $(\eta/\Delta t)\,\boldsymbol{I}$ | $\eta \to 0$ | $\mathcal{O}(\eta^{-1})$ | vanishes as $\eta\to0$ |

**Take-away.** The conditioning trade-off is set by *what regularizes the Taylor null space of $\boldsymbol{G}$*:
IPM/VP tie it to the parameter that must go to the limit ($\mu, \eta \to 0$) ⇒ $\kappa$ diverges there; MB
keeps that parameter moderate; AL regularizes with the identity ⇒ conditioning decoupled from the KKT limit.
The numerical comparison is generated by [examples/ex_single_step.py](../examples/ex_single_step.py).
