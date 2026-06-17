# Augmented Lagrangian Method for Single Crystal Plasticity

This document outlines the time-discrete Augmented Lagrangian (AL) algorithm for rate-independent single crystal plasticity at small strains. It relies on a fixed-point update recast as a non-smooth root-finding problem to resolve the active slip systems and determine the plastic slip increments.

### Augmented Lagrangian Residual
The standard Karush-Kuhn-Tucker (KKT) conditions are replaced by an equivalent unconstrained non-smooth residual equation using a penalty parameter $\eta > 0$. For a given load step, the residual for each slip system $\alpha$ is defined as:
$$R^\alpha(\Delta \boldsymbol{\gamma}) = \Delta \gamma^\alpha - \max \left( 0, \Delta \gamma^\alpha + \eta \, \Phi^\alpha(\Delta \boldsymbol{\gamma}) \right) = 0$$
where:
* $\Delta \gamma^\alpha$ is the incremental plastic slip.
* $\Phi^\alpha$ is the yield function.
* $\max(0, \bullet)$ enforces the non-negativity of the plastic slip and naturally identifies the active slip systems.

### Active Set and Jacobian
The formulation partitions the slip systems into active ($\mathcal{A}$) and inactive ($\mathcal{I}$) sets based on the argument of the max function:
* **Active** ($\alpha \in \mathcal{A}$): $\Delta \gamma^\alpha + \eta \, \Phi^\alpha > 0$
* **Inactive** ($\alpha \in \mathcal{I}$): $\Delta \gamma^\alpha + \eta \, \Phi^\alpha \leq 0$

The system is solved using a semi-smooth Newton-Raphson method. The Jacobian $J^{\alpha\beta} = \frac{\partial R^\alpha}{\partial \Delta \gamma^\beta}$ is piecewise defined depending on the active set:
$$J^{\alpha\beta} = 
\begin{cases} 
\delta^{\alpha\beta} & \text{if } \alpha \in \mathcal{I} \\[0.5em]
-\eta \, H^{\alpha\beta} & \text{if } \alpha \in \mathcal{A} 
\end{cases}$$
where $H^{\alpha\beta} = \frac{\partial \Phi^\alpha}{\partial \Delta \gamma^\beta}$ is the derivative of the yield function with respect to the plastic slip increments, and $\delta^{\alpha\beta}$ is the Kronecker delta.

*Note: For active systems, the residual simplifies to $R^\alpha = -\eta \, \Phi^\alpha$. For inactive systems, it simplifies to $R^\alpha = \Delta \gamma^\alpha$.*

### Newton System
The iterative Newton update $\delta(\Delta \boldsymbol{\gamma})$ is obtained by solving the linear system:
$$\sum_\beta J^{\alpha\beta} \delta (\Delta \gamma^\beta) = - R^\alpha$$

By separating the slip systems into active and inactive blocks, the linear system can be expressed as:
$$
\begin{bmatrix}
-\eta \mathbf{H}_{\mathcal{A}\mathcal{A}} & -\eta \mathbf{H}_{\mathcal{A}\mathcal{I}} \\[0.5em]
\mathbf{0} & \mathbf{I}_{\mathcal{I}\mathcal{I}}
\end{bmatrix}
\begin{bmatrix}
\delta (\Delta \boldsymbol{\gamma}_{\mathcal{A}}) \\[0.5em]
\delta (\Delta \boldsymbol{\gamma}_{\mathcal{I}})
\end{bmatrix}
= -
\begin{bmatrix}
-\eta \boldsymbol{\Phi}_{\mathcal{A}} \\[0.5em]
\Delta \boldsymbol{\gamma}_{\mathcal{I}}
\end{bmatrix}
$$
where $\mathbf{I}_{\mathcal{I}\mathcal{I}}$ is the identity matrix. 

Solving this system gives the increment update $\Delta \gamma^\alpha \leftarrow \Delta \gamma^\alpha + \delta (\Delta \gamma^\alpha)$. The penalty parameter $\eta$ can be iteratively doubled or updated in an outer loop if the exact KKT conditions are not satisfied to the desired tolerance.

### Convergence Criteria (stress units, increment-size uniform)

Both convergence tests are formulated in stress units, using the slip-to-stress exchange rate
$$a^\alpha = -\frac{\partial \Phi^\alpha}{\partial \Delta\gamma^\alpha} = \boldsymbol{Z}^\alpha\!:\!\mathbb{C}\!:\!\boldsymbol{Z}^\alpha + \tau_\text{h}'(A) > 0$$
(the self-hardening modulus of system $\alpha$): $a^\alpha\,\Delta\gamma^\alpha$ is the stress perturbation caused by the slip $\Delta\gamma^\alpha$.

**Inner (semi-smooth Newton):** terminate when the stress effect of the Newton step is a fixed factor below the outer target,
$$\max_\alpha a^\alpha\,\bigl|\delta(\Delta\gamma^\alpha)\bigr| < \theta_\text{in}\,\texttt{tol}_\Phi .$$
An absolute slip-norm criterion ($\|\delta\Delta\boldsymbol{\gamma}\| < \texttt{tol}$) is increment-size uniform only by accident — its stress meaning $\sim a\,\texttt{tol}$ is implicit; the scaled form makes the conversion explicit and ties the inner accuracy to the outer tolerance.

**Outer (exact KKT):** terminate when the **natural (min-form) KKT residual** is below the stress-unit tolerance,
$$r_\text{KKT} = \max_\alpha \Bigl|\min\!\bigl(a^\alpha\,\Delta\gamma^\alpha,\; -\Phi^\alpha\bigr)\Bigr| < \texttt{tol}_\Phi, \qquad \texttt{tol}_\Phi = \epsilon_\Phi\,\tau_0 .$$
Since $\min(a,b) = 0 \Leftrightarrow a \ge 0,\, b \ge 0,\, ab = 0$, this is an exact reformulation of complementarity: on active systems it bounds the yield residual $|\Phi^\alpha| < \texttt{tol}_\Phi$ directly, on inactive systems the spurious-slip stress $a^\alpha\Delta\gamma^\alpha$, and an *overstress* $\Phi^\alpha > 0$ — the AL iterates approach the yield surface from outside — makes the minimum negative and is flagged by the absolute value.

The natural residual replaces the product test $\|\Delta\boldsymbol{\gamma}\odot\boldsymbol{\Phi}\| < \texttt{tol}$, which has two defects: (i) it certifies only $|\Phi^\alpha| \lesssim \texttt{tol}/\Delta\gamma^\alpha$, an accuracy that degrades in proportion to the number of load increments ($\Delta\gamma^\alpha \propto 1/n_\text{steps}$); and (ii) since *both* factors shrink with the increment size, it can pass on an **unconverged** inner state at small $\Delta\gamma^\alpha$ — unacceptable for a reference solver. The criterion $r_\text{KKT} < \texttt{tol}_\Phi$ bounds the distance of the stress state to the yield surface uniformly in the load-increment size. The same construction is used for the MB-IPM (see `ipm_mb.md`, *Outer convergence*).

| Parameter | Description |
|---|---|
| `eta_init` | Initial penalty parameter $\eta$ (doubled on outer non-convergence) |
| `tol_phi` | Outer KKT tolerance in stress units, $r_\text{KKT} < \texttt{tol}_\Phi = \epsilon_\Phi\,\tau_0$ |
| `theta_in` | Inner step safety factor: Newton step accepted at $\max_\alpha a^\alpha|\delta\Delta\gamma^\alpha| < \theta_\text{in}\,\texttt{tol}_\Phi$ |
| `max_outer` | Maximum outer (penalty) iterations |
| `max_inner` | Maximum semi-smooth Newton iterations per outer step |