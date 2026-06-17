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

#### Schur-Complement Reduction and the Reduced Hessian

The $2m\times2m$ system is never solved directly; the slack correction $\delta\mathbf{s}$ is eliminated by a Schur complement, condensing it to an $m\times m$ solve in $\delta\!\Delta\boldsymbol{\gamma}$ alone. Writing the two block rows,
$$\frac{\partial\boldsymbol{\Phi}}{\partial\Delta\boldsymbol{\gamma}}\,\delta\!\Delta\boldsymbol{\gamma} + \delta\mathbf{s} = -\mathbf{R}_g \quad(1), \qquad \operatorname{diag}(\mathbf{s}+\mu)\,\delta\!\Delta\boldsymbol{\gamma} + \operatorname{diag}(\Delta\boldsymbol{\gamma})\,\delta\mathbf{s} = -\mathbf{R}_s \quad(2),$$
solve (1) for the slack correction,
$$\delta\mathbf{s} = -\mathbf{R}_g - \frac{\partial\boldsymbol{\Phi}}{\partial\Delta\boldsymbol{\gamma}}\,\delta\!\Delta\boldsymbol{\gamma},$$
and substitute into (2). This gives the **reduced (condensed) Newton system**
$$\boxed{\mathbf{M}\,\delta\!\Delta\boldsymbol{\gamma} = -\mathbf{R}_\text{mb}}, \qquad \mathbf{M} = \operatorname{diag}(\mathbf{s}+\mu) - \operatorname{diag}(\Delta\boldsymbol{\gamma})\,\frac{\partial\boldsymbol{\Phi}}{\partial\Delta\boldsymbol{\gamma}},$$
followed by the explicit back-substitution for $\delta\mathbf{s}$ above. The right-hand side collapses to the **slack-free reduced residual** (the slacks cancel identically):
$$-\mathbf{R}_s + \Delta\boldsymbol{\gamma}\odot\mathbf{R}_g = -\big(\Delta\gamma^\alpha(\mu - \Phi^\alpha) - \mu\,\lambda^\alpha_k\big) \equiv -R^\alpha_\text{mb},$$
which is exactly the reduced complementarity condition $R^\alpha_\text{mb} = \Delta\gamma^\alpha(\mu - \Phi^\alpha) - \mu\,\lambda^\alpha_k$ obtained by analytically eliminating the slack via $s^\alpha = -\Phi^\alpha$. Consequently $\mathbf{M} = \partial\mathbf{R}_\text{mb}/\partial\Delta\boldsymbol{\gamma}$ — the same Jacobian used for the consistent tangent (`compute_tangent_IFT_mb`).

**Reduced Hessian.** The reduced operator factors as a row scaling of a symmetric positive-definite Hessian,
$$\mathbf{M} = \operatorname{diag}(\Delta\boldsymbol{\gamma})\,\mathbf{H}, \qquad \boxed{\mathbf{H} = -\frac{\partial\boldsymbol{\Phi}}{\partial\Delta\boldsymbol{\gamma}} + \operatorname{diag}\!\left(\frac{s^\alpha + \mu}{\Delta\gamma^\alpha}\right) = \mathbf{K} + \operatorname{diag}\!\left(\frac{s^\alpha + \mu}{\Delta\gamma^\alpha}\right)}$$
where $\mathbf{K} = -\partial\boldsymbol{\Phi}/\partial\Delta\boldsymbol{\gamma} = \boldsymbol{Z}^\top\!\mathbb{C}\,\boldsymbol{Z} + \tau_\text{h}'(A)\,\mathbf{1}\mathbf{1}^\top$ is the symmetric PSD return-map Hessian of the convex QP ($\S$4) and $\tau_\text{h}'(A) = \tau_\infty\,\xi\,e^{-\xi A} \geq 0$ the hardening slope. $\mathbf{H}$ is the Hessian of the reduced modified-barrier objective: the elastoplastic stiffness $\mathbf{K}$ plus a **positive barrier diagonal** $(s^\alpha+\mu)/\Delta\gamma^\alpha > 0$ (every interior iterate has $\Delta\gamma^\alpha>0$ and $s^\alpha+\mu>0$), so $\mathbf{H}\succ 0$. The symmetric form $\mathbf{H}\,\delta\!\Delta\boldsymbol{\gamma} = \mu\lambda^\alpha_k/\Delta\gamma^\alpha - (s^\alpha+\mu)$ follows by left-dividing through by $\operatorname{diag}(\Delta\boldsymbol{\gamma})$.

**Why the unsymmetric $\mathbf{M}$ is solved, not $\mathbf{H}$.** Although $\mathbf{H}$ is the cleaner symmetric Hessian, its barrier diagonal $(s^\alpha+\mu)/\Delta\gamma^\alpha$ **diverges** as $\Delta\gamma^\alpha \to 0$ on systems that deactivate in late outer iterations — the same ill-conditioning that afflicts the $2m\times2m$ form. The row-scaled $\mathbf{M} = \operatorname{diag}(\Delta\boldsymbol{\gamma})\,\mathbf{H}$ absorbs that factor: as $\Delta\gamma^\alpha\to0$ its $\alpha$-row tends to $\operatorname{diag}(\mathbf{s}+\mu)>0$ and stays well-conditioned. The Schur reduction therefore both halves the system size and cures the conditioning, at the cost of a nonsymmetric (but always nonsingular in the interior) $\mathbf{M}$.

---

### Nested-Loop Algorithm

#### Initialization

Elastic predictor at $\Delta\boldsymbol{\gamma} = \mathbf{0}$. If $\Phi^\alpha \leq 0$ for all $\alpha$, the step is purely elastic.

Otherwise:
$$\mu_0 = \mu_\text{init}, \qquad \lambda^\alpha_{k=0} = \lambda^\alpha_\text{prev} > 0, \qquad s^\alpha_0 = \max(-\Phi^\alpha_\text{trial},\,\sqrt{\mu_0}), \qquad \Delta\gamma^\alpha_0 = \sqrt{\mu_0}$$

The multiplier estimates are **warm-started from the converged multipliers of the previous load step** $\lambda^\alpha_\text{prev}$. This provides a near-optimal initial guess for incremental loading, cutting the required outer iterations to near zero for small load increments. On the very first step (no previous history) a uniform positive value $\lambda_\text{init} > 0$ is used as fallback. Initializing at zero is forbidden: the update rule preserves zero as a fixed point and would lock such a system out permanently.

#### Outer Lagrange-Multiplier Loop (index $k$)

**1. Inner Newton solve** at fixed $(\mu_k, \boldsymbol{\lambda}_k)$, iterating the Newton system above with the fraction-to-boundary safeguard
$$\alpha_\text{max} = \min\!\left(1,\; \min_{\delta\!\Delta\gamma^\alpha < 0}\frac{-\tau_\text{min}\,\Delta\gamma^\alpha}{\delta\!\Delta\gamma^\alpha},\; \min_{\delta s^\alpha < 0}\frac{-\tau_\text{min}\,(s^\alpha + \mu)}{\delta s^\alpha}\right)$$
The slack cap is on $s^\alpha + \mu > 0$, not $s^\alpha > 0$: the barrier's singularity sits at $s^\alpha = -\mu$, so the iterate must merely stay in its *domain*, and temporary mild yield violation is allowed during iteration, which makes long Newton steps acceptable. The cap therefore fires on **every** descending slack direction $\delta s^\alpha < 0$ (each can drive $s^\alpha+\mu$ toward the boundary), and on every descending $\delta\!\Delta\gamma^\alpha < 0$ to keep $\Delta\gamma^\alpha > 0$.

**Scaled inner residual and adaptive inner tolerance.** The residual blocks carry different units: $R_g^\alpha = \Phi^\alpha + s^\alpha$ is a stress, while $R_s^\alpha = \Delta\gamma^\alpha(s^\alpha+\mu) - \mu\lambda^\alpha_k$ carries stress $\times$ slip. A leftover $R_s^\alpha$ at termination perturbs the slack — and hence the stress — by $\delta s^\alpha \approx R_s^\alpha/\Delta\gamma^\alpha$: an absolute bound on the raw norm $\|\mathbf{R}\|$ therefore certifies a stress accuracy of only $\texttt{tol}/\Delta\gamma^\alpha$, which **degrades as the load increments (and with them $\Delta\gamma^\alpha$) shrink**. The inner iteration is therefore terminated on the dimensionally consistent scaled norm
$$\bar\lambda_k = \max\Bigl(\max_\alpha \lambda^\alpha_k,\; \max_\alpha \Delta\gamma^\alpha,\; \lambda_{\min}\Bigr), \qquad \|\mathbf{R}\|_{\mathrm{scl}} = \max\!\Bigl(\|\mathbf{R}_g\|_\infty,\; \|\mathbf{R}_s\|_\infty / \bar\lambda_k\Bigr),$$
with a small floor $\lambda_{\min}$ guarding the division, against the **adaptive inner tolerance**
$$\|\mathbf{R}\|_{\mathrm{scl}} \;<\; \texttt{tol}_\text{in}^{(k)} = \max\!\bigl(\theta_\text{in}\,\texttt{tol}_\Phi,\; \theta_\mu\,\mu_k\bigr), \qquad \theta_\text{in},\,\theta_\mu \in [10^{-2}, 10^{-1}].$$
Dividing $\mathbf{R}_s$ by the active-slip scale $\bar\lambda_k$ converts it into the stress error it causes, so the criterion bounds the inner-solve contamination of the yield residual *uniformly in the increment size* — the same logic as the stress-unit outer test (step 3). One shared scale (rather than per-component $\Delta\gamma^\alpha$) avoids demanding absurd accuracy on inactive systems, where a small $R_s^\alpha$ perturbs nothing. The first argument of the tolerance keeps that contamination a fixed factor $\theta_\text{in}$ below the outer target $\texttt{tol}_\Phi$; the second relaxes the inner solve in early outer sweeps, where $\boldsymbol{\lambda}_k$ is still far from converged and polishing the inner solution beyond the $O(\mu)$ move of the next Polyak update is wasted work (the analogue of the classical IPM's $\texttt{tol} = \max(\texttt{tol}_\text{end}, \theta\mu)$, see `ipm.md`). As the APV lowers $\mu_k$, the inner tolerance tightens automatically toward its final value $\theta_\text{in}\,\texttt{tol}_\Phi$.

**Inner-failure / blow-up safeguard.** Two situations spoil an outer step: a *stiff* inner problem (a small $\mu$ shrinks the band $(-\mu,\infty)$ and shortens the admissible steps, so the inner Newton cannot reach $\texttt{tol}_\text{in}^{(k)}$ within `max_inner` steps), and a *blow-up* of the multiplier estimate. The latter is the dangerous one and arises at an **active-set change**: a newly-activating system sits near the shifted boundary $s^\alpha \to -\mu$, where the update $\lambda^\alpha_{k+1} = \mu\lambda^\alpha_k/(s^\alpha+\mu)$ has a near-singular denominator and amplifies $\lambda^\alpha$ violently (in practice $\lambda^\alpha \to \infty$ within a few sweeps, then $\textsf{NaN}$). Both are handled by a **cold restart at a larger $\mu$**: set $\mu_{k+1} = \min(\rho_\uparrow\,\mu_k,\,\mu_\text{ceil})$ with $\rho_\uparrow > 1$ (a wider band moves the boundary away from the activating system), and re-initialize the *entire* iterate from the cold-start values — $s^\alpha$, $\Delta\gamma^\alpha$, **and** the multiplier estimate $\lambda^\alpha_k \leftarrow \lambda_\text{init}$. Resetting $\lambda_k$ is essential: merely raising $\mu$ carries the already-poisoned (or non-finite) estimate forward, and the warm-started $\lambda_k$ is precisely what was inconsistent with the new active set. The dynamic floor is pinned to the raised value, $\mu_\text{lo} \leftarrow \mu_{k+1}$, so the APV ($\S$4) cannot tighten straight back into the regime that just failed. Only if $\mu_k$ reaches $\mu_\text{ceil}$ or the retry budget is spent does the step report failure. (A non-finite residual must be tested explicitly — $\textsf{NaN}$ fails every comparison and would otherwise slip past the stall test and stall the outer loop.)

**2. Multiplier update** from the converged inner state:
$$\boxed{\lambda^\alpha_{k+1} = \frac{\mu_k\,\lambda^\alpha_k}{s^\alpha_* + \mu_k}}$$
This is the stationarity condition rearranged. It preserves positivity, decays on inactive systems ($s^\alpha_* \gg \mu_k \Rightarrow \lambda^\alpha_{k+1} \ll \lambda^\alpha_k$), and remains essentially unchanged on active ones ($s^\alpha_* \approx 0 \Rightarrow \lambda^\alpha_{k+1} \approx \lambda^\alpha_k$).

**Fixed-point / dual-proximal view.** The outer loop is a **contractive fixed-point iteration on the dual variables**, $\boldsymbol{\lambda}_{k+1} = T(\boldsymbol{\lambda}_k)$, where the map is the composition of the inner primal solve and the explicit Polyak update,
$$T \;=\; \underbrace{\Big(s^\alpha \mapsto \tfrac{\mu\,\lambda^\alpha_k}{s^\alpha+\mu}\Big)}_{\text{multiplier update}}\;\circ\;\underbrace{\Big(\boldsymbol{\lambda}_k \mapsto s^\alpha_*(\boldsymbol{\lambda}_k)\Big)}_{\text{inner solve at frozen }\boldsymbol{\lambda}_k}.$$
Freezing $\boldsymbol{\lambda}_k$ renders the inner subproblem a well-posed convex return map with a unique solution $s^\alpha_*(\boldsymbol{\lambda}_k)$, so a single evaluation of $T$ is one inner Newton solve followed by the algebraic update. Its **fixed points are exactly the KKT multipliers**: $\lambda^\alpha_* = T(\lambda^\alpha_*)$ reads $\lambda^\alpha_* = \mu\lambda^\alpha_*/(s^\alpha_*+\mu)$, i.e. $\lambda^\alpha_* s^\alpha_* = 0$, with $\mu$ cancelling identically — the fixed-point set is the rate-independent solution set and is independent of the (finite) shift.

The iteration is a **local contraction** with the rate already quoted in $\S$4. For a single active system the inner solution satisfies $\Delta\gamma_* = \lambda_{k+1}$ and $s_* = -\Phi(\lambda_{k+1})$, so $T$ is defined implicitly by
$$\lambda_{k+1}\big(\mu - \Phi(\lambda_{k+1})\big) = \mu\,\lambda_k.$$
Differentiating at the fixed point ($\Phi(\lambda_*)=0$, $\;\partial\Phi/\partial\Delta\gamma = -a$, $\;a>0$) gives
$$T'(\lambda_*) = \frac{\mathrm{d}\lambda_{k+1}}{\mathrm{d}\lambda_k}\bigg|_* = \frac{\mu}{\mu + a\,\lambda_*} = C \in (0,1),$$
so $\|\boldsymbol{\lambda}_{k+1}-\boldsymbol{\lambda}_*\| \le C\,\|\boldsymbol{\lambda}_k-\boldsymbol{\lambda}_*\|$ (Banach), the **linear** rate that the APV ($\S$4) then accelerates by shrinking $\mu$. Equivalently, $T$ is **dual proximal-point ascent**: each sweep maximizes the (concave) dual functional regularized by a Bregman proximal term of weight $\sim 1/\mu$, so a larger $\mu$ is a stronger proximal penalty (smaller, safer dual steps) and a smaller $\mu$ a weaker one (larger, faster steps) — the dual reading of the same $C(\mu)$ trade-off. This is the IPM analogue of the method-of-multipliers fixed-point map of the augmented-Lagrange VP treatment; only the inner loop is a (locally quadratic) Newton iteration, while the outer loop is this first-order dual fixed-point map. Under Taylor degeneracy ($>5$ active systems, $\operatorname{null}(\boldsymbol{Z})\neq\{\mathbf{0}\}$) $T$ acquires a continuum of fixed points along $\operatorname{null}(\boldsymbol{Z})$ and the contraction holds only on the physically determined quotient $\Delta\boldsymbol{\varepsilon}_p = \sum_\alpha \Delta\gamma^\alpha\boldsymbol{Z}^\alpha$ — precisely why convergence is monitored on $\Delta\boldsymbol{\varepsilon}_p$ rather than on $\boldsymbol{\lambda}$ (see step 3).

**3. Outer convergence**. The raw multiplier change is **not** a usable convergence test. At high-symmetry orientations more than five slip systems go active simultaneously; since the symmetric-deviatoric $\boldsymbol{Z}^\alpha$ span only a 5-dimensional space, the active set is then linearly dependent — there exist $\boldsymbol{c} \neq \mathbf{0}$ with $\sum_\alpha c_\alpha \boldsymbol{Z}^\alpha = \mathbf{0}$ (the **Taylor / Bishop–Hill ambiguity**). The converged multipliers $\Delta\gamma^\alpha = \lambda^\alpha_*$ are then non-unique: they drift along $\operatorname{null}(\boldsymbol{Z})$ without changing $\Delta\boldsymbol{\varepsilon}_p$, $\boldsymbol{\sigma}$, or any $\Phi^\alpha$, so $\max_\alpha|\lambda^\alpha_{k+1}-\lambda^\alpha_k|$ never reaches zero even though the stress has fully converged.

Convergence is therefore measured on the **plastic-strain increment**, which is invariant under that drift, together with a stress-unit KKT measure. Terminate when **both** criteria hold simultaneously:
$$\Big\|\sum_\alpha \big(\Delta\gamma^\alpha_{(k)} - \Delta\gamma^\alpha_{(k-1)}\big)\,\boldsymbol{Z}^\alpha\Big\| < \texttt{tol}_\text{dep} \qquad \textbf{and} \qquad r_\text{KKT} < \texttt{tol}_\Phi.$$

Here $\Delta\gamma_{(k)}$ is the **primal** slip increment from the $k$-th inner solve — the variable that defines $\boldsymbol{\varepsilon}_p$ and the returned stress. It is used rather than the multiplier-estimate change $\lambda^\alpha_{k+1}-\lambda^\alpha_k = -\lambda^\alpha_k\,s^\alpha/(s^\alpha+\mu)$, because the two **decouple when $\mu \ll s$**: at a small shift the estimate change is dominated by inner-residual noise in $s$ and keeps drifting on a degenerate set long after the primal slip (and the stress) has frozen. The first criterion is thus the change in $\Delta\boldsymbol{\varepsilon}_p = \sum_\alpha \Delta\gamma^\alpha \boldsymbol{Z}^\alpha$ across the outer update: a drift with $\sum_\alpha c_\alpha \boldsymbol{Z}^\alpha = \mathbf{0}$ leaves it exactly zero, so it certifies that the *physically determined* part of the slip — hence the stress — has settled, regardless of how the redundant systems share the slip. Note that stress/$\Delta\boldsymbol{\varepsilon}_p$ uniqueness holds for any symmetric positive-semidefinite hardening (perfect, self, isotropic, latent with $q\le 1$); strongly latent hardening ($q>1$, indefinite interaction matrix) can break it, in which case rate-dependent regularization is the standard remedy.

**The natural KKT residual.** The second criterion measures the distance to the *exact* KKT point via the natural (min-form) residual of the complementarity system. For two scalars,
$$\min(a,b) = 0 \quad\Longleftrightarrow\quad a \geq 0, \quad b \geq 0, \quad a\,b = 0,$$
so the componentwise minimum is an exact reformulation of the complementarity conditions. Since the slip increments and the yield function carry different units, each slip increment is converted to a stress scale by the self-hardening modulus of system $\alpha$, the diagonal of the negative yield-function Jacobian
$$a^\alpha = -\,\frac{\partial \Phi^\alpha}{\partial \Delta\gamma^\alpha} = \boldsymbol{Z}^\alpha\!:\!\mathbb{C}\!:\!\boldsymbol{Z}^\alpha + \tau_\text{h}'(A) \;>\; 0,$$
yielding
$$\boxed{\,r_\text{KKT} = \max_\alpha \Bigl|\, \min\!\bigl(a^\alpha\,\Delta\gamma^\alpha,\; -\Phi^\alpha(\Delta\boldsymbol{\gamma})\bigr) \Bigr| \,}$$
built from the **primal** slip (immune to the multiplier decoupling above). $a^\alpha$ is the exchange rate between slip and stress — $a^\alpha\Delta\gamma^\alpha$ is the stress perturbation the slip causes — so the min compares like with like and all three regimes are captured without an explicit active-set classification: on active systems ($s^\alpha \to 0$) the criterion bounds the yield residual $|\Phi^\alpha| < \texttt{tol}_\Phi$ directly; on inactive systems it bounds the stress polluted by spurious slip, $a^\alpha\Delta\gamma^\alpha < \texttt{tol}_\Phi$; a yield violation $\Phi^\alpha > 0$ — transiently admissible up to $\mu$ in the modified-barrier setting — renders the minimum negative and is flagged by the absolute value. The precise value of $a^\alpha$ is uncritical (an $O(1)$-correct conversion suffices; the frozen elastic value is an admissible, robust choice).

**Increment-size uniformity.** This is the decisive property of $r_\text{KKT}$: product-form criteria $\max_\alpha|\lambda^\alpha s^\alpha| < \texttt{tol}$ certify only $|\Phi^\alpha| \lesssim \texttt{tol}/\Delta\gamma^\alpha$, which **degrades in proportion to the number of load increments** ($\Delta\gamma^\alpha \propto 1/n_\text{steps}$) — the solver-tolerance analogue of the classical IPM's $\mu_\text{end}/\Delta\gamma$ bias. Criterion $r_\text{KKT} < \texttt{tol}_\Phi$ instead bounds the distance of the stress state to the yield surface uniformly in the load-increment size: the achieved stress accuracy is $\sim\texttt{tol}_\Phi = \epsilon_\Phi\,\tau_0$ at every $n_\text{steps}$. Both criteria are required (an AND): the first guards settling of the solution, the second guards exact complementarity.

**4. Shift update — Adjusted Parameter Version (APV)**. Driving $\mu_k \to 0$ is **not** required, but $\mu$ is **not** held strictly fixed either; it is adapted from a merit test. Two competing facts set the trade-off:

- *Outer contraction.* Linearizing the multiplier map about the solution (one active system, local slope $a = -\partial\Phi^\alpha/\partial\Delta\gamma^\alpha > 0$) gives the error recursion $\varepsilon_{k+1} = C\,\varepsilon_k$ with
$$C = \frac{\mu}{\mu + a\,\lambda^\alpha_*} \in (0,1).$$
Thus $C \to 0$ as $\mu \to 0$ and $C \to 1$ as $\mu \to \infty$: a **smaller** $\mu$ gives **faster** outer convergence. This is the standard Polyak rate $\|\boldsymbol{\lambda}_{k+1} - \boldsymbol{\lambda}_*\| \le (c/k)\,\|\boldsymbol{\lambda}_k - \boldsymbol{\lambda}_*\|$ with $k = 1/\mu$ (contraction $\propto \mu$), and agrees with the dual-proximal reading of nonlinear rescaling (a larger shift $\mu = 1/k$ is a stronger proximal term, hence smaller dual steps).
- *Inner stiffness.* A smaller $\mu$ shrinks the shifted-feasible band $(-\mu,\infty)$, shortens the fraction-to-boundary steps, and pins the achievable accuracy to the inner tolerance (directly in stress units under the scaled norm of step 1, $|\Phi^\alpha| \lesssim \texttt{tol}_\text{in}^{(k)}$). Too small a $\mu$ therefore makes the **inner** solve fail even though it would *accelerate* the outer loop.

$\mu$ is thus pulled small by the outer rate and large by inner conditioning. The APV resolves this automatically using the **same plastic-strain-increment measure** as the convergence test for its merit:
$$v_k = \Big\|\sum_\alpha \big(\Delta\gamma^\alpha_{(k)} - \Delta\gamma^\alpha_{(k-1)}\big)\,\boldsymbol{Z}^\alpha\Big\|.$$
This quantity is both **$\mu$-independent** (so it is a fair progress measure across $\mu$ changes — unlike the shifted residual $R_s$ or the complementarity gap $|\lambda^\alpha s^\alpha| = \mu|\lambda^\alpha_k - \lambda^\alpha_{k+1}|$, both of which carry an explicit $\mu$ and shrink trivially as $\mu\to 0$, which would otherwise drive $\mu$ to its floor for no real progress) and **Taylor-invariant** (so the slip drift on a degenerate active set does not look like a stall and does not trigger needless $\mu$-tightening). The decision:

- **Sufficient decrease** ($v_{k+1} \le \theta\,v_k$, $0<\theta<1$): keep $\mu$.
- **Stall** ($v_{k+1} > \theta\,v_k$): tighten $\mu_{k+1} = \max(\rho_\downarrow\,\mu_k,\,\mu_\text{lo})$ with $\rho_\downarrow < 1$ to accelerate the outer rate; the next inner solve re-projects the primal iterate onto the shifted-feasible set.

**Complementarity polish.** A subtlety arises once the stress has settled ($\|\Delta\boldsymbol{\varepsilon}_p\|$-merit below $\texttt{tol}_\text{dep}$) but $r_\text{KKT}$ is still above $\texttt{tol}_\Phi$ — e.g. after the inner-failure safeguard ($\S$1) has raised $\mu$. At a fixed active set the residual yield is $|\Phi^\alpha| = |s^\alpha| \sim \mu\,\|\boldsymbol{\lambda}_k - \boldsymbol{\lambda}_{k+1}\|$, so the active branch of $r_\text{KKT}$ scales **linearly with $\mu$** and a larger $\mu$ can floor it above $\texttt{tol}_\Phi$. The remedy is to **shrink $\mu$** ($\mu_{k+1}=\max(\rho_\downarrow\mu_k,\mu_\text{floor})$), which drives the residual down proportionally. This reduction is allowed *below* $\mu_\text{lo}$: with no active-set transition in progress, every active slack sits at $s^\alpha \sim +\mu$ and every inactive one at $s^\alpha \gg 0$, so none is near the shifted boundary $s^\alpha=-\mu$ and the blow-up that forced the earlier raise cannot recur. One decade of reduction typically suffices.

The stall-tightening and the inner-failure raising of step 1 never trigger in the same iteration (the inner solve either fails or it doesn't), so they do not oscillate; the dynamic floor $\mu_\text{lo}$ prevents the *stall*-tightening from re-entering a $\mu$ range that already failed, while the complementarity polish bypasses it deliberately, only once the active set has settled. Together they let $\mu$ settle at the **smallest value the inner solve tolerates** — i.e. the fastest stable outer rate — with no a-priori knowledge of the threshold $\bar k = 1/\mu^*$. Because the local return-map problem is **convex** (convex QP objective $\tfrac12\Delta\boldsymbol{\gamma}^\top\!\boldsymbol{Z}^\top\!\mathbb{C}\,\boldsymbol{Z}\,\Delta\boldsymbol{\gamma}$, convex feasible set $\{\Phi^\alpha \le 0\}$), the linear rate of the APV is sufficient and the Varying-Parameter Version (VPV, $\mu_k\to\infty$ in $k$, required for nonconvexity / a superlinear rate) is not needed.

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

with the corresponding shifted condition $\Delta\gamma^\alpha(s^\alpha + \tilde\mu\,\tau_0) = \tilde\mu\,\tau_0\,\lambda^\alpha_k$. Because $\mu$ is adapted by the APV (lowered for outer speed, raised for inner stability) rather than driven to zero, its precise value is not critical: a small starting $\tilde\mu_\text{init} = O(10^{-2})$ with self-tuning is adequate, and the converged stress/tangent are independent of the value $\mu$ ultimately settles at.

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
| `mu_init` | Initial shift parameter $\mu$ (e.g. $\sim 10^{-2}\tau_0$); the APV adapts it from here |
| `mu_floor` | Hard lower bound for $\mu_k$ |
| `mu_ceil` | Upper bound for $\mu_k$ used by the inner-failure safeguard (e.g. $\sim 10^{-1}\tau_0$) |
| `mu_dec` | APV reduction factor $\rho_\downarrow < 1$ applied to $\mu_k$ on outer stall (smaller $\mu$ = faster outer) |
| `mu_inc` | APV increase factor $\rho_\uparrow > 1$ applied to $\mu_k$ on inner-solve failure |
| `theta` | APV sufficient-decrease factor for the KKT merit ($0 < \theta < 1$) |
| `max_retry` | Maximum consecutive inner-failure $\mu$-increases before the step aborts |
| `lambda_init` | Fallback multiplier estimate for the first load step (warm-start used thereafter) |
| `tol_phi` | Outer KKT tolerance in stress units, $r_\text{KKT} < \texttt{tol}_\Phi = \epsilon_\Phi\,\tau_0$ (increment-size uniform) |
| `theta_in` | Inner safety factor: final inner tolerance $\theta_\text{in}\,\texttt{tol}_\Phi$ below the outer target |
| `theta_mu` | Inner relaxation factor: early-sweep inner tolerance $\sim \theta_\mu\,\mu_k$ |
| `tol_dep` | Outer tolerance on the plastic-strain-increment change $\\|\sum_\alpha \Delta\lambda^\alpha \boldsymbol{Z}^\alpha\\|$ (Taylor-invariant) |
| `tau_min` | Fraction-to-boundary safety factor (applied to $s^\alpha+\mu$, not $s^\alpha$) |
| `k_max` | Maximum outer multiplier iterations |
| `max_inner` | Maximum Newton steps per outer iteration |

---

### References

- B. T. Polyak, *Modified barrier functions: theory and methods*, Mathematical Programming **54** (1992), 177–222.
- R. Polyak & M. Teboulle, *Nonlinear rescaling and proximal-like methods in convex optimization*, Mathematical Programming **76** (1997), 265–284.
- F. Niehüser & J. Mosler, *An interior point method for rate-independent single crystal plasticity*, 2023 (basis of `ipm.md`).

---

### Method at a Glance

The MB-IPM solves the rate-independent return map through a **two-level structure**: an inner solve that satisfies the *shifted* (regularized) optimality conditions for a fixed shift $\mu$ and fixed multiplier estimates $\boldsymbol{\lambda}_k$, wrapped in an outer loop that updates those estimates until the *exact* KKT conditions are met. The shift $\mu$ stays finite throughout; convergence is driven by the multiplier updates, not by $\mu \to 0$.

The essential idea in one line: replace the log-barrier by Polyak's shifted barrier $-\mu\sum_\alpha\lambda^\alpha_k\ln(1+s^\alpha/\mu)$, whose gradient stays finite at the yield surface and therefore yields an **explicit estimate of the Lagrange multiplier** after each solve.

**Conceptual algorithm** (one strain increment $[t_n, t_{n+1}]$):

1. **Elastic predictor.** Evaluate $\Phi^\alpha$ at $\Delta\boldsymbol{\gamma}=\mathbf{0}$. If $\Phi^\alpha\le0$ for all $\alpha$, the step is elastic — return the trial stress.

2. **Initialize.** Choose a shift $\mu$ and warm-start the multiplier estimates $\boldsymbol{\lambda}_k$ from the previous step (strictly positive on every system).

3. **Outer loop** over multiplier estimates $\boldsymbol{\lambda}_k$:
   - **(a) Inner solve** — with $\mu$ and $\boldsymbol{\lambda}_k$ held fixed, solve the shifted KKT system
     $$\Phi^\alpha + s^\alpha = 0, \qquad \Delta\gamma^\alpha(s^\alpha+\mu) = \mu\,\lambda^\alpha_k$$
     for the primal state $(\Delta\boldsymbol{\gamma}, \mathbf{s})$ by Newton iteration.
   - **(b) Multiplier update** — read the new estimate off the converged slacks (the Polyak update):
     $$\lambda^\alpha_{k+1} = \frac{\mu\,\lambda^\alpha_k}{s^\alpha_*+\mu}.$$
   - **(c) Check** the exact KKT conditions ($\Delta\boldsymbol{\varepsilon}_p$ settled and the stress-unit natural residual $r_\text{KKT} = \max_\alpha|\min(a^\alpha\Delta\gamma^\alpha, -\Phi^\alpha)| < \texttt{tol}_\Phi$). If met, stop; otherwise return to (a) with $\boldsymbol{\lambda}_{k+1}$.

4. **Recover** stress and consistent tangent from the converged $\Delta\boldsymbol{\gamma}$ — both free of any $\mu$-bias.

**Why it converges to the exact solution.** Step 3 is a contractive fixed-point iteration on the multipliers (§ outer loop): its fixed point forces $\lambda^\alpha_* s^\alpha_* = 0$, the exact rate-independent complementarity, with the shift $\mu$ cancelling out. The inner solve regularizes the problem (finite rates, enlarged feasible band) without biasing it, and the outer updates remove the regularization error explicitly rather than by shrinking $\mu$. The result is the true rate-independent stress and tangent at a *finite*, numerically comfortable $\mu$ — in contrast to the classical IPM, where the same accuracy requires $\mu\to0$ and pays a flow-stress bias.

*The adaptive choice of $\mu$, the fraction-to-boundary safeguard, and the failure/restart logic — i.e. everything that turns this skeleton into a robust solver — are the subject of the algorithmic part below.*
