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