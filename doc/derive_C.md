## Deriving the Outer Contraction Rate $C$ of the MB-IPM

This note derives the local convergence rate of the outer multiplier loop of the
Modified-Barrier IPM (see `ipm_mb.md`),
$$\boxed{C = \frac{\mu}{\mu + a\,\lambda^\alpha_*} \in (0,1)},$$
starting from the modified barrier and tracing every step down to the fixed-point
map $\boldsymbol{\lambda}_{k+1} = T(\boldsymbol{\lambda}_k)$. The rate quantifies the
statement *"smaller $\mu$ converges faster, larger $\mu$ slower"* and is the formula
behind the *role of the barrier parameter* discussion.

---

### 1. From the barrier to the shifted complementarity condition

The modified-barrier subproblem replaces the log-barrier by Polyak's shifted variant
$$\mathfrak{L}_\mu = -\mathfrak{D}_\text{red} + \sum_\alpha \lambda^\alpha(\Phi^\alpha + s^\alpha) - \mu\sum_\alpha \lambda^\alpha_k \ln\!\left(1 + \frac{s^\alpha}{\mu}\right),$$
with the multiplier estimates $\lambda^\alpha_k$ **frozen** during the inner solve.
Stationarity in $s^\alpha$,
$$\frac{\partial\mathfrak{L}_\mu}{\partial s^\alpha} = \lambda^\alpha - \frac{\lambda^\alpha_k}{1 + s^\alpha/\mu} = 0,$$
clears to the **shifted complementarity condition**
$$\lambda^\alpha\,(s^\alpha + \mu) = \mu\,\lambda^\alpha_k. \tag{1}$$

This single algebraic relation is the seed of everything below: it both *defines the
inner solution* (with $\lambda^\alpha = \Delta\gamma^\alpha$ from the flow rule) and,
read the other way, *defines the multiplier update*.

---

### 2. The two readings of (1): inner solve and outer update

With the flow-rule identification $\Delta\gamma^\alpha = \lambda^\alpha$, the inner
solve enforces (1) together with $\Phi^\alpha + s^\alpha = 0$, i.e. $s^\alpha = -\Phi^\alpha$:
$$\Delta\gamma^\alpha\,(\mu - \Phi^\alpha(\boldsymbol{\Delta\gamma})) = \mu\,\lambda^\alpha_k. \tag{2}$$

The Polyak update reads the *same* equation (1) as an explicit formula for the new estimate,
evaluated at the converged inner slack $s^\alpha_*$:
$$\lambda^\alpha_{k+1} = \frac{\mu\,\lambda^\alpha_k}{s^\alpha_* + \mu}. \tag{3}$$

Comparing (2) and (3) at the converged inner state ($s^\alpha_* = -\Phi^\alpha$) gives the
key bookkeeping identity
$$\Delta\gamma^\alpha_* = \frac{\mu\,\lambda^\alpha_k}{\mu - \Phi^\alpha} = \lambda^\alpha_{k+1}, \tag{4}$$
i.e. **the converged primal slip of one sweep equals the multiplier estimate fed to the
next.** This is what lets us treat the whole sweep as a map on a single variable.

---

### 3. The outer map $T$ as an implicit scalar relation

Restrict to **one active slip system** (the local rate is governed by the worst single
mode; the multi-system version is in §6). Using (4) to eliminate $\Delta\gamma_*$ in favour
of $\lambda_{k+1}$, equation (2) becomes the **implicit definition of the outer map**
$\lambda_{k+1} = T(\lambda_k)$:
$$\boxed{\;\lambda_{k+1}\,\big(\mu - \Phi(\lambda_{k+1})\big) = \mu\,\lambda_k\;} \tag{5}$$
where $\Phi(\lambda_{k+1})$ means $\Phi$ evaluated at the primal slip $\Delta\gamma = \lambda_{k+1}$.

One full outer sweep is: freeze $\lambda_k$ → solve the (convex, well-posed) inner problem
for $\lambda_{k+1}$ through (5) → repeat. The right-hand side is linear in $\lambda_k$; the
left-hand side is nonlinear in $\lambda_{k+1}$ through the yield function $\Phi$.

---

### 4. The fixed point is the exact KKT solution

A fixed point $\lambda_* = T(\lambda_*)$ sets $\lambda_k = \lambda_{k+1} = \lambda_*$ in (5):
$$\lambda_*\,(\mu - \Phi(\lambda_*)) = \mu\,\lambda_*.$$
For an active system ($\lambda_* > 0$) divide by $\lambda_*$:
$$\mu - \Phi(\lambda_*) = \mu \quad\Longrightarrow\quad \Phi(\lambda_*) = 0. \tag{6}$$
The shift $\mu$ **cancels identically** — the fixed point satisfies the *exact*
rate-independent yield condition $\Phi(\lambda_*) = 0$, with no $\mu$-bias. (If
$\lambda_* = 0$ the system is inactive and (5) is trivially satisfied; either branch
reproduces $\lambda_* s_* = 0$.) This is why $\mu$ may stay finite: it does not enter the
solution, only the *rate* at which we reach it, which we now compute.

---

### 5. Differentiating $T$ at the fixed point

Define
$$g(\lambda_{k+1}) := \lambda_{k+1}\,\big(\mu - \Phi(\lambda_{k+1})\big), \qquad g(\lambda_{k+1}) = \mu\,\lambda_k.$$
Differentiate both sides with respect to $\lambda_k$ (implicit differentiation of (5)):
$$g'(\lambda_{k+1})\,\frac{\mathrm{d}\lambda_{k+1}}{\mathrm{d}\lambda_k} = \mu. \tag{7}$$
By the product rule,
$$g'(\lambda_{k+1}) = \big(\mu - \Phi(\lambda_{k+1})\big) + \lambda_{k+1}\,\big(-\Phi'(\lambda_{k+1})\big),$$
where $\Phi'(\Delta\gamma) = \mathrm{d}\Phi/\mathrm{d}\Delta\gamma$. Evaluate at the fixed
point, using $\Phi(\lambda_*) = 0$ from (6) and introducing the **return-map stiffness**
$$a := -\left.\frac{\partial\Phi}{\partial\Delta\gamma}\right|_{*} > 0,$$
which is positive because increasing the slip increment both relaxes the resolved stress
and raises the hardening, lowering $\Phi$ (this $a$ is the diagonal of
$\mathbf{K} = -\partial\boldsymbol{\Phi}/\partial\boldsymbol{\Delta\gamma} = \boldsymbol{Z}^\top\mathbb{C}\boldsymbol{Z} + \tau_\text{h}'\mathbf{1}\mathbf{1}^\top$, see `ipm_mb.md`). Then
$$g'(\lambda_*) = (\mu - 0) + \lambda_*\,a = \mu + a\,\lambda_*. \tag{8}$$
Insert (8) into (7):
$$\boxed{\;C \equiv T'(\lambda_*) = \frac{\mathrm{d}\lambda_{k+1}}{\mathrm{d}\lambda_k}\bigg|_* = \frac{\mu}{\mu + a\,\lambda_*}\;} \tag{9}$$

Since $a > 0$, $\lambda_* > 0$ and $\mu > 0$, the denominator exceeds the numerator and
$C \in (0,1)$: the map is a **local contraction**, and by the Banach fixed-point theorem
$$|\lambda_{k+1} - \lambda_*| \le C\,|\lambda_k - \lambda_*|,$$
i.e. **linear** (geometric) convergence with ratio $C$.

---

### 6. What $C$ tells you about $\mu$

Rewrite (9) as
$$C = \frac{1}{1 + \dfrac{a\,\lambda_*}{\mu}}.$$

| limit | $C$ | outer convergence |
|---|---|---|
| $\mu \to 0$ | $C \to 0$ | instantaneous (one sweep) |
| $\mu \to \infty$ | $C \to 1$ | arbitrarily slow |

So **a smaller shift $\mu$ gives a faster outer rate** — the formula behind the
trade-off. The opposing pressure (a smaller $\mu$ shrinks the shifted-feasible band
$(-\mu,\infty)$ and stiffens the *inner* Newton solve) is what the APV balances; $C(\mu)$
is the "outer-speed" half of that trade-off.

**Dual-proximal reading.** Writing $k = 1/\mu$, equation (9) is $C = k\,a\lambda_*/(1 + \ldots)^{-1}$
form of Polyak's nonlinear-rescaling rate: $\mu$ acts as the inverse weight of a Bregman
proximal term, so a larger $\mu$ is a *stronger* proximal penalty → smaller, safer dual
steps → $C$ closer to $1$. The contraction (9) is the differential-form statement of that
$O(1/k)$ proximal-point estimate.

---

### 7. Multi-system generalization

For an active set of several systems the map (5) is vector-valued,
$\lambda^\alpha_{k+1}(\mu - \Phi^\alpha(\boldsymbol{\lambda}_{k+1})) = \mu\lambda^\alpha_k$,
and $\Phi^\alpha$ couples all components through $\Delta\gamma^\beta = \lambda^\beta_{k+1}$.
Differentiating as in §5,
$$\big[\mu\,\mathbf{I} + \operatorname{diag}(\boldsymbol{\lambda}_*)\,\mathbf{K}\big]\,\mathbf{T}' = \mu\,\mathbf{I},$$
so the **Jacobian of the outer map** is
$$\boxed{\;\mathbf{T}'(\boldsymbol{\lambda}_*) = \mu\big[\mu\,\mathbf{I} + \operatorname{diag}(\boldsymbol{\lambda}_*)\,\mathbf{K}\big]^{-1} = \big[\mathbf{I} + \tfrac{1}{\mu}\operatorname{diag}(\boldsymbol{\lambda}_*)\,\mathbf{K}\big]^{-1}\;}$$
and convergence is governed by its **spectral radius** $\rho(\mathbf{T}') < 1$. For a single
system this collapses to (9) with $a = K$. Because $\mathbf{K} \succeq 0$ (SPD on the
non-degenerate part), every eigenvalue of $\tfrac{1}{\mu}\operatorname{diag}(\boldsymbol{\lambda}_*)\mathbf{K}$
is $\ge 0$, so every eigenvalue of $\mathbf{T}'$ lies in $(0,1]$. The unit eigenvalues, if
any, correspond exactly to $\operatorname{null}(\boldsymbol{Z})$ — the **Taylor/Bishop–Hill
ambiguity** — where the multipliers drift without changing $\Delta\boldsymbol{\varepsilon}_p$;
this is precisely why convergence is monitored on $\Delta\boldsymbol{\varepsilon}_p$ rather
than on $\boldsymbol{\lambda}$ (see `ipm_mb.md`, §outer convergence).

---

### Summary

| step | result |
|---|---|
| barrier stationarity in $s$ | shifted complementarity $\lambda^\alpha(s^\alpha+\mu)=\mu\lambda^\alpha_k$ |
| flow rule + $s=-\Phi$ | inner relation $\Delta\gamma(\mu-\Phi)=\mu\lambda_k$, and $\Delta\gamma_* = \lambda_{k+1}$ |
| eliminate $\Delta\gamma$ | implicit map $\lambda_{k+1}(\mu-\Phi(\lambda_{k+1}))=\mu\lambda_k$ |
| fixed point | $\Phi(\lambda_*)=0$, exact KKT, $\mu$ cancels |
| implicit differentiation | $C = \mu/(\mu + a\lambda_*) \in (0,1)$ |
| multi-system | $\mathbf{T}' = [\mathbf{I} + \tfrac1\mu\operatorname{diag}(\boldsymbol{\lambda}_*)\mathbf{K}]^{-1}$, rate $= \rho(\mathbf{T}')$ |

The contraction rate is finite and strictly below one for any $\mu > 0$: the outer loop
converges linearly **without** $\mu \to 0$, and faster as $\mu$ is reduced — the exact
content of the *role of the barrier parameter* discussion.

---

### References

- B. T. Polyak, *Modified barrier functions: theory and methods*, Math. Programming **54** (1992), 177–222.
- R. Polyak & M. Teboulle, *Nonlinear rescaling and proximal-like methods in convex optimization*, Math. Programming **76** (1997), 265–284.
