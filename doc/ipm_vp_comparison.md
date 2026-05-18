## Relating the IPM Barrier Parameter $\mu$ to the VP Viscosity $\eta$

Both the IPM and the VP model regularize the same rate-independent constrained dissipation problem.
At steady state, each produces a flow stress that deviates from the yield stress $\tau_0$ by a
model-specific amount. Matching these deviations defines a rate-dependent mapping between $\mu$ and $\eta$.

---

### Steady-State Flow Stress in Each Model

Consider a single active slip system, no hardening ($g^\alpha = \tau_0 = \text{const}$), loaded at a
fixed slip rate $\dot{\gamma}^\alpha$. The steady-state resolved shear stress $\tau_\text{ss}$ and
its deviation $\delta\tau = |\tau_\text{ss} - \tau_0|$ are:

#### IPM (continuous)

The perturbed KKT condition $\dot{\gamma}^\alpha \cdot (\tau_0 - \tau^\alpha) = \mu$ gives at steady state:

$$\tau_\text{ss}^\text{IPM} = \tau_0 - \frac{\mu}{\dot{\gamma}^\alpha}, \qquad \delta\tau^\text{IPM} = \frac{\mu}{\dot{\gamma}^\alpha}$$

The IPM stress lies **below** $\tau_0$ (interior regularization).

#### VP, Bingham ($p = 1$)

The flow rule $\dot{\gamma}^\alpha = \frac{1}{\eta}\langle\tau^\alpha - \tau_0\rangle$ inverted at steady state:

$$\tau_\text{ss}^\text{VP,1} = \tau_0 + \eta\,\dot{\gamma}^\alpha, \qquad \delta\tau^\text{VP,1} = \eta\,\dot{\gamma}^\alpha$$

The VP stress lies **above** $\tau_0$ (exterior regularization).


---

### Matching the Flow Stress Deviation

Setting $\delta\tau^\text{IPM} = \delta\tau^\text{VP}$ (matching magnitudes, ignoring the sign difference)
gives the equivalent barrier parameter $\mu_\text{eq}$ as a function of $\eta$ and $\dot{\gamma}^\alpha$:

#### Bingham ($p = 1$)

$$\frac{\mu}{\dot{\gamma}^\alpha} = \eta\,\dot{\gamma}^\alpha \qquad \Longrightarrow \qquad \boxed{\mu_\text{eq} = \eta\,(\dot{\gamma}^\alpha)^2}$$


---

### Interpretation

| | IPM | VP ($p=1$) | VP ($p=2$) |
|---|---|---|---|
| Steady-state stress | $\tau_0 - \mu/\dot{\gamma}$ | $\tau_0 + \eta\dot{\gamma}$ | $\tau_0 + \tau_\text{ref}\sqrt{\eta\dot{\gamma}}$ |
| Deviation $\delta\tau$ | $\mu/\dot{\gamma}$ | $\eta\dot{\gamma}$ | $\tau_\text{ref}\sqrt{\eta\dot{\gamma}}$ |
| Side of yield surface | below $\tau_0$ | above $\tau_0$ | above $\tau_0$ |
| Equivalent $\mu$ | — | $\eta(\dot{\gamma})^2$ | $\tau_\text{ref}\sqrt{\eta}(\dot{\gamma})^{3/2}$ |

The mapping is **not a constant** — it is intrinsically rate-dependent:

- For Bingham ($p=1$): $\mu \propto (\dot{\gamma})^2$. The IPM deviation decreases with $\dot{\gamma}$ (inverse
  law), while the VP deviation grows linearly. Both scale as $\eta\dot{\gamma}$ in magnitude only when
  $\mu = \eta(\dot{\gamma})^2$.

Because the functional forms differ, a single $\mu$ cannot reproduce the VP behavior across all slip
rates simultaneously — the equivalent $\mu$ must always be re-calibrated to the operating slip rate.

---

### Discrete Form

In the discrete setting (slip increment $\Delta\gamma^\alpha$, time step $\Delta t$), the VP flow rule
gives $\Delta\gamma^\alpha = \frac{\Delta t}{\eta}\langle\Phi^\alpha/\tau_\text{ref}\rangle^p$, so the
discrete steady-state deviations are:

| Model | $\delta\tau$ (discrete) | Equivalent $\mu_\text{end}$ |
|---|---|---|
| Bingham ($p=1$) | $\dfrac{\eta\,\Delta\gamma}{\Delta t}$ | $\dfrac{\eta\,(\Delta\gamma)^2}{\Delta t}$ |

Using $\Delta\gamma \approx \dot{\gamma}\Delta t$ recovers the continuous results above.

---

### A Priori Estimate of $\eta$ from $\mu$ and $\Delta\varepsilon$

In practice $\Delta\gamma^\alpha$ is not known before the simulation — only the applied strain increment
$\Delta\varepsilon$ per step is prescribed. The kinematic constraint at steady state closes the gap:
the plastic strain increment must absorb the applied increment, so

$$\Delta\gamma^\alpha \approx \frac{\Delta\varepsilon}{m}$$

where $m$ is the Schmid factor for the active system ($m = \boldsymbol{Z}^\alpha : \hat{\boldsymbol{e}}\otimes\hat{\boldsymbol{e}}$
for uniaxial loading, $m = 1$ for simple shear aligned with the slip system). This estimate is exact
at steady state when elastic strains are constant.

Because $\Delta t$ is pseudo-time in a rate-independent simulation, $\eta$ and $\Delta t$ always
appear together as the pseudo-viscosity $\hat{\eta} = \eta/\Delta t$ [Pa]. Substituting
$\Delta\gamma^\alpha \approx \Delta\varepsilon/m$ into the matching conditions yields:

| Model | A priori estimate of $\hat{\eta}$ |
|---|---|
| Bingham ($p=1$) | $\hat{\eta} \approx \dfrac{\mu_\text{end}\,m^2}{(\Delta\varepsilon)^2}$ |

For simple shear with a well-oriented single system $m = 1$, these reduce to:

$$\hat{\eta}^\text{Bing} \approx \frac{\mu_\text{end}}{(\Delta\varepsilon)^2}, \qquad \hat{\eta}^\text{Per} \approx \frac{\mu_\text{end}^2}{\tau_\text{ref}^2\,(\Delta\varepsilon)^3}$$

**How to use.** Given $\mu_\text{end}$ and $\Delta\varepsilon$, compute $\hat{\eta}$ from the formula
above. Then set $\eta = \hat{\eta}\cdot\Delta t$ in the VP model (or equivalently use $\hat{\eta}$
directly in the discrete VP flow rule). The two models will then produce approximately the same
magnitude of flow stress deviation at steady state.

**Accuracy.** The estimate is first-order: it assumes that all applied strain is accommodated
plastically ($\Delta\gamma^\alpha \approx \Delta\varepsilon/m$), which holds well after the elastic
transient but breaks down for stiff problems or small overstresses where the elastic correction to
$\Delta\gamma^\alpha$ is not negligible.

---

### Practical Use for Comparison

To run a fair side-by-side comparison of the VP and IPM approaches:

1. **Fix a reference slip rate** $\dot{\gamma}_\text{ref}$ (e.g., the expected steady-state slip rate
   from the loading and geometry).
2. **Choose** $\eta$ for the VP model to set the desired physical overstress:
   $\delta\tau_\text{target} = \eta\,\dot{\gamma}_\text{ref}$ (Bingham) or
   $\delta\tau_\text{target} = \tau_\text{ref}\sqrt{\eta\,\dot{\gamma}_\text{ref}}$ (Perzyna $p=2$).
3. **Set** $\mu_\text{end} = \eta\,(\dot{\gamma}_\text{ref})^2$ (Bingham) or
   $\mu_\text{end} = \tau_\text{ref}\sqrt{\eta}\,(\dot{\gamma}_\text{ref})^{3/2}$ (Perzyna $p=2$)
   so that the IPM produces the same magnitude of flow stress deviation at that slip rate.
4. **Compare** the steady-state plateau stresses from both simulations — they should match in
   magnitude but sit on opposite sides of $\tau_0$.

The residual difference between the two curves then isolates the effect of the functional form
(inverse law vs. power law) rather than merely the choice of regularization strength.
