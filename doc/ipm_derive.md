## IPM Flow Rule, Dimensions of $\mu$, and Analogy with the VP Model

### The IPM Flow Rule from the Perturbed KKT Conditions

The time-continuous perturbed KKT conditions (see `ipm.md`) are:

$$\dot{\gamma}^\alpha \geq 0, \qquad \Phi^\alpha + s^\alpha = 0, \qquad s^\alpha \geq 0, \qquad \dot{\gamma}^\alpha s^\alpha = \mu$$

Substituting the equality constraint $s^\alpha = -\Phi^\alpha$ into the complementarity condition gives:

$$\dot{\gamma}^\alpha \cdot (-\Phi^\alpha) = \mu$$

Since inside the feasible set $\Phi^\alpha < 0$, the denominator $-\Phi^\alpha = g^\alpha - \tau^\alpha > 0$ and the flow rate is positive:

$$\boxed{\dot{\gamma}^\alpha = \frac{\mu}{g^\alpha - \tau^\alpha} = \frac{\mu}{-\Phi^\alpha}}$$

This is the IPM viscous flow rule: slip is driven by the inverse of the distance to the yield surface. As $\mu \to 0$ the flow rate vanishes except when $g^\alpha - \tau^\alpha \to 0$ simultaneously, recovering the rate-independent KKT condition $\Phi^\alpha \dot{\gamma}^\alpha = 0$.

---

### Dimensions of $\mu$ in the Time-Continuous Case

The dimensional check follows directly from the perturbed complementarity condition:

$$[\dot{\gamma}^\alpha \cdot s^\alpha] = [\mu]$$

$$\left[\frac{1}{\text{s}}\right] \cdot [\text{Pa}] = [\mu] \quad \Longrightarrow \quad [\mu] = \frac{\text{Pa}}{\text{s}}$$

The barrier parameter $\mu$ carries **units of stress per time** in the time-continuous formulation. It is not dimensionless, and not a pure stress — it is a stress rate.

---

### Dimensions in the Time-Discrete Case and the Role of $\Delta t$

In the time-discrete formulation (see `ipm.md`), the slip rate $\dot{\gamma}^\alpha$ is replaced by the incremental slip $\Delta\gamma^\alpha = \gamma^\alpha - \gamma^\alpha_n$, which is a dimensionless strain increment. The perturbed KKT condition becomes:

$$\Delta\gamma^\alpha \cdot s^\alpha = \mu_\text{disc}$$

$$[\Delta\gamma^\alpha \cdot s^\alpha] = [-] \cdot [\text{Pa}] = [\text{Pa}] \quad \Longrightarrow \quad [\mu_\text{disc}] = \text{Pa}$$

The discrete barrier parameter carries **units of stress** only. The factor of time has been absorbed: writing $\Delta\gamma^\alpha \approx \dot{\gamma}^\alpha \Delta t$ and inserting the continuous relation $\dot{\gamma}^\alpha(-\Phi^\alpha) = \mu_\text{cont}$ gives:

$$\mu_\text{disc} = \mu_\text{cont} \cdot \Delta t$$

**Implication for numerical practice.** The IPM is used purely as a numerical tool that drives $\mu_\text{disc} \to 0$ to enforce the KKT conditions. In this role $\mu_\text{disc}$ need not be related to $\Delta t$: it is simply reduced until the residual is small enough, regardless of the load step size. However, if one wants to interpret the discrete IPM as a consistent time-discretization of the continuous viscous flow rule $\dot{\gamma}^\alpha = \mu_\text{cont}/(g^\alpha - \tau^\alpha)$, the initial barrier must be scaled as $\mu_\text{disc,0} = \mu_\text{cont,0} \cdot \Delta t$.

---

### Dimensionless Normalization via a Reference Stress

In analogy with the Perzyna normalization in `vp_derive.md`, a reference stress $\tau_\text{ref}$ (natural choice: the initial slip resistance $\tau_0$) can be used to render the log-barrier argument dimensionless. The normalized barrier objective reads:

$$\mathfrak{D}_{\tilde{\mu}} = -\mathfrak{D}_\text{red} - \tilde{\mu}\,\tau_\text{ref} \sum_\alpha \ln\!\left(\frac{-\Phi^\alpha}{\tau_\text{ref}}\right)$$

where $\tilde{\mu} = \mu/\tau_\text{ref}$. The constant shift $\ln(\tau_\text{ref})$ per system does not affect the optimality conditions, so the flow rule is unchanged in form but now expressed in terms of $\tilde{\mu}$:

$$\dot{\gamma}^\alpha = \frac{\tilde{\mu}}{(-\Phi^\alpha)/\tau_\text{ref}}$$

The units become:
- Continuous: $[\tilde{\mu}] = [\mu]/[\tau_\text{ref}] = \text{Pa/s} / \text{Pa} = \text{s}^{-1}$
- Discrete: $[\tilde{\mu}_\text{disc}] = \text{Pa}/\text{Pa} = \text{dimensionless}$

In the discrete case $\tilde{\mu}_\text{disc}$ is fully dimensionless, and the perturbed complementarity condition becomes:

$$\Delta\gamma^\alpha \cdot \frac{-\Phi^\alpha}{\tau_\text{ref}} = \tilde{\mu}_\text{disc}$$

Both sides are now dimensionless. The normalized barrier parameter $\tilde{\mu}_\text{disc}$ has a direct physical interpretation: it is the geometric mean of the incremental slip and the normalized yield-function distance, and convergence to $\tilde{\mu}_\text{disc} \to 0$ is equivalent to convergence in strain units.

---

### Effect of a Finite $\mu_\text{end}$ and Its Relation to $\Delta t$

In practice the outer loop terminates at $\mu_k \leq \mu_\text{end} > 0$ rather than at $\mu = 0$. Understanding what a finite $\mu_\text{end}$ means is important for choosing it correctly.

#### What finite $\mu_\text{end}$ implies

At termination the converged discrete state satisfies the perturbed condition:

$$\Delta\gamma^\alpha \cdot (-\Phi^\alpha) = \mu_\text{end} \quad \forall\, \alpha$$

For a **plastically inactive** system where the true solution has $\Delta\gamma^\alpha = 0$ and $-\Phi^\alpha \approx \tau_\text{ref}$, this produces **spurious incremental slip**:

$$\Delta\gamma^\alpha_\text{spurious} \approx \frac{\mu_\text{end}}{\tau_\text{ref}} = \tilde{\mu}_\text{end}$$

For this to be negligible, $\tilde{\mu}_\text{end}$ must be small compared to the physically meaningful incremental slips on active systems.

#### The $\Delta t$ dependence

The physical incremental slip on an active system scales as:

$$\Delta\gamma^\alpha_\text{phys} \sim \dot{\gamma}^\alpha \cdot \Delta t$$

For the spurious slip to be negligible relative to the physical slip:

$$\tilde{\mu}_\text{end} \ll \dot{\gamma}^\alpha \cdot \Delta t$$

Or equivalently, using $\mu_\text{disc} = \mu_\text{cont} \cdot \Delta t$:

$$\mu_\text{end} \ll \tau_\text{ref} \cdot \dot{\gamma}^\alpha \cdot \Delta t$$

This shows that **$\mu_\text{end}$ should scale with $\Delta t$**: halving the time step requires halving $\mu_\text{end}$ to maintain the same relative accuracy. Equivalently, the normalized quantity $\tilde{\mu}_\text{end} / \Delta\gamma^\alpha_\text{phys}$ is the relevant smallness parameter, not $\tilde{\mu}_\text{end}$ alone.

#### Practical consequence

For a fixed loading rate, $\Delta\gamma^\alpha_\text{phys} \propto \Delta t$. A fixed absolute $\mu_\text{end}$ therefore becomes **relatively larger** as $\Delta t$ decreases. In adaptive time-stepping schemes, $\mu_\text{end}$ should be reduced proportionally to $\Delta t$, or equivalently a relative threshold $\tilde{\mu}_\text{end} / \Delta\gamma_\text{expected}$ should be used instead of an absolute one. For quasi-static simulations with a fixed, moderate time step this coupling is often negligible in practice, but it matters if $\Delta t$ spans many orders of magnitude.

---

### Structural Analogy and Difference with the VP Model

Comparing the two flow rules side by side:

| | IPM | VP (Bingham, $p=1$) |
|---|---|---|
| Flow rule | $\dot{\gamma}^\alpha = \dfrac{\mu}{g^\alpha - \tau^\alpha}$ | $\dot{\gamma}^\alpha = \dfrac{1}{\eta}\langle\tau^\alpha - g^\alpha\rangle$ |
| Active domain | interior: $\Phi^\alpha < 0$ | exterior: $\Phi^\alpha > 0$ |
| Functional form | inverse distance to yield surface | linear overstress |
| Units of parameter | $[\mu] = \text{Pa/s}$ | $[\eta] = \text{Pa·s}$ |
| Rate-independent limit | $\mu \to 0$ | $\eta \to 0$ |

The structural similarity is that both express the slip rate as a ratio involving the yield function and a single scalar parameter. The key differences are:

1. **Opposite sides of the yield surface.** The IPM regularizes from inside the feasible set (overstress is never produced), while the VP model regularizes from outside (yield is violated by design).
2. **Inverse vs. linear functional form.** The IPM flow rate diverges as the yield surface is approached from within ($g^\alpha - \tau^\alpha \to 0$), acting as a repulsive barrier. The VP flow rate grows linearly with how far the state lies outside the yield surface.
3. **Conjugate parameters.** $[\mu] = \text{Pa/s}$ and $[\eta] = \text{Pa·s}$, so $\mu\,\eta$ has units of Pa$^2$ — there is no simple inversion relation between them.

Both models recover the rate-independent KKT conditions in their respective limits and both regularize the same underlying constrained dissipation maximization problem (see `vp_derive.md`), differing only in the choice of regularization function.

---

## Suggested Material-Point Verification Experiments

All experiments use a single crystal with one active slip system and no hardening ($\tau_\text{h} = 0$, $g^\alpha = \tau_0 = \text{const}$) under simple shear loading oriented so that only one slip system is active. The rate-independent exact solution is then known analytically: after yielding, the resolved shear stress is exactly $\tau^\alpha = \tau_0$ for all subsequent steps. Any deviation from this is a measurable artifact of finite $\mu$.

#### Simulation outputs needed

For all three experiments, output per step:

| Quantity | How to obtain |
|---|---|
| $\tau^\alpha = \boldsymbol{Z}^\alpha : \boldsymbol{\sigma}$ | contract Schmid tensor with stress output |
| $\Delta\gamma^\alpha = \gamma^\alpha_{n+1} - \gamma^\alpha_n$ | difference of accumulated slip between steps |

These two quantities appear directly in the perturbed condition and are sufficient to compute all errors. Crucially, $\Delta\gamma^\alpha$ is not equal to the macroscopic strain increment $\Delta\varepsilon$ in general — they are related through the Schmid tensor and crystal geometry.

#### Steady-state formula with finite $\mu_\text{end}$

This result underpins all three experiments. At steady state the stress is constant step-to-step: $\tau^{(n+1)} = \tau^{(n)} = \tau_\text{ss}$. Constant stress means constant elastic strain, so the plastic strain increment equals the applied strain increment as a tensor: $\sum_\alpha \Delta\gamma^\alpha \boldsymbol{Z}^\alpha = \Delta\boldsymbol{\varepsilon}$. For the single active system this determines $\Delta\gamma^\alpha_\text{ss}$ from the loading and crystal geometry. The IPM perturbed condition at convergence then gives:

$$\Delta\gamma^\alpha_\text{ss} \cdot (\tau_0 - \tau_\text{ss}) = \mu_\text{end}$$

$$\boxed{\tau_\text{ss} = \tau_0 - \frac{\mu_\text{end}}{\Delta\gamma^\alpha_\text{ss}}, \qquad \delta\tau = \tau_0 - \tau_\text{ss} = \frac{\mu_\text{end}}{\Delta\gamma^\alpha_\text{ss}}}$$

The IPM steady-state stress is always **below** $\tau_0$. Both $\tau_\text{ss}$ and $\Delta\gamma^\alpha_\text{ss}$ are direct simulation outputs — no inference from macroscopic strain needed.

---

### Experiment 1 — $\mu_\text{end}$ vs step size coupling

**Goal.** Show that a fixed absolute $\mu_\text{end}$ produces an error that grows as the step size shrinks.

**Setup.** Fix total applied strain $\varepsilon_\text{tot}$ and run with different step counts $N$ (e.g. $N = 5, 50, 500$), keeping $\mu_\text{end}$ fixed. There is no physical time — the model is rate-independent and only the per-step increment matters. Finer stepping means smaller $\Delta\gamma^\alpha_\text{ss}$ per step (proportionally, since the geometry is fixed and $\Delta\gamma^\alpha_\text{ss} \propto \varepsilon_\text{tot}/N$).

**Error to plot.** From the steady-state formula: $\delta\tau = \mu_\text{end}/\Delta\gamma^\alpha_\text{ss} \propto N$. Read $\tau_\text{ss}$ from the plateau of the $\tau^\alpha$ output and plot $\tau_0 - \tau_\text{ss}$ against $N$ — it should be linear. You can also verify the perturbed condition directly: $\Delta\gamma^\alpha \cdot (\tau_0 - \tau^\alpha)$ at each converged step should equal $\mu_\text{end}$ regardless of $N$. The paradoxical result is that finer stepping makes the stress error worse.

**Control run.** Repeat with $\mu_\text{end}(N) = \mu_\text{end,ref}/N$. The plateau stress $\tau_\text{ss}$ should now be independent of $N$.

---

### Experiment 2 — Viscosity-like dependence of the flow stress on $\mu_\text{end}$

**Goal.** Show that $\mu_\text{end}$ acts as a viscosity-like parameter: it sets the deviation of the flow stress from $\tau_0$, analogous to how $\eta$ controls the overstress in the VP model.

**Note on time.** In a rate-independent model the time step $\Delta t$ is a purely artificial pseudo-time label with no physical meaning. Varying $\Delta t$ independently of $\Delta\varepsilon$ is therefore not a valid operation: the two are the same discretization and cannot be decoupled. The continuous barrier $\mu_\text{cont}$ [Pa/s] has no physical interpretation in this context. The only meaningful parameter in the discrete problem is $\mu_\text{end}$ [Pa].

**Setup.** Fix the step size (fixed $\Delta\varepsilon$, hence fixed $\Delta\gamma^\alpha_\text{ss}$) and run the same loading with several values of $\mu_\text{end}$ spanning a few orders of magnitude (e.g. $\mu_\text{end} = 10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\,\tau_0$).

**What to observe.** From the steady-state formula $\tau_\text{ss} = \tau_0 - \mu_\text{end}/\Delta\gamma^\alpha_\text{ss}$, the plateau stress should shift linearly with $\mu_\text{end}$. Plot $\tau_\text{ss}$ against $\mu_\text{end}$ — it should be a straight line with slope $-1/\Delta\gamma^\alpha_\text{ss}$ and intercept $\tau_0$. This is the discrete analogue of a viscous flow rule: $\mu_\text{end}$ plays the role of $\eta \cdot \dot{\gamma}^\alpha$ in the VP model, but here it is a numerical parameter rather than a physical one.

**Sign note.** The IPM stress is always *below* $\tau_0$ (interior regularization), while the VP model gives stress *above* $\tau_0$ (exterior regularization). Larger $\mu_\text{end}$ → larger undershoot, while larger $\eta$ in VP → larger overshoot.

**Control run.** Confirm that as $\mu_\text{end} \to 0$ the plateau converges to $\tau_0$, recovering the rate-independent result.

---

### Experiment 3 — Effect of normalization across different stress scales

**Goal.** Show that an absolute $\mu_\text{end}$ [Pa] produces different relative errors for materials with different slip resistances, and that using the normalized $\tilde{\mu}_\text{end} = \mu_\text{end}/\tau_0$ removes this dependence.

**Setup.** Two problems identical in crystal geometry and normalized loading (same $\Delta\varepsilon/(\tau_0/G)$, so the same normalized $\Delta\gamma^\alpha_\text{ss}/(\tau_0/G)$ at steady state) but with different slip resistances: $\tau_0^{(1)} = 10\,\text{MPa}$ and $\tau_0^{(2)} = 1000\,\text{MPa}$. The exact normalized solution is identical for both.

**Error to plot.** The relative stress error from the steady-state formula is:

$$\frac{\delta\tau}{\tau_0} = \frac{\mu_\text{end}}{\Delta\gamma^\alpha_\text{ss}\,\tau_0}$$

Since the normalized loading is the same, $\Delta\gamma^\alpha_\text{ss} \propto \tau_0/G$, so $\delta\tau/\tau_0 \propto \mu_\text{end}/\tau_0^2$. With the same absolute $\mu_\text{end}$, the soft material has a 100× larger relative error. Plot the normalized stress-strain curves $\tau^\alpha/\tau_0$ vs $\varepsilon G/\tau_0$ for both: they diverge despite the normalized problem being identical.

**Control run.** Set $\mu_\text{end}^{(i)} = \tilde{\mu}_\text{end} \cdot \tau_0^{(i)}$ (same dimensionless $\tilde{\mu}_\text{end}$ for both). The normalized curves coincide, confirming that $\tilde{\mu}_\text{end}$ is the meaningful threshold.
