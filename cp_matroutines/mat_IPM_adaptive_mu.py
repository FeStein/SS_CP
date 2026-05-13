"""
Classical Interior Point Method for rate-independent single crystal plasticity
at small strains.

Implements the nested-loop IPM following Niehüser & Mosler (2023), with:
  - Standard full 2m×2m primal-dual Newton system withouit scaling
  - Adaptive inner Newton tolerance: tol = max(tol_end, theta * mu)
  - Barrier update: mu_{k+1} = max_alpha(dlambda^alpha * s^alpha)^(1+delta)
  - Fraction-to-boundary with adaptive tau = min(tau_min, 1 - r^beta) without
  line search

Derivatives of the yield function are computed via JAX automatic differentiation.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cp_matroutines.cp_base import Material


# ---------------------------------------------------------------------------
# Yield function (JAX-compatible, with hardening) — JIT'd at module level
# ---------------------------------------------------------------------------

@jax.jit
def yield_function(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi):
    r"""
    Compute the yield function for all slip systems:

        Phi^alpha = Z^alpha : sigma(dgamma) - (tau_0 + tau_h)

    with exponential saturation hardening:
        tau_h = tau_inf * (1 - exp(-xi * A)),   A = sum(gamma_n + dgamma)

    Parameters
    ----------
    dgamma  : (m,) jnp array — incremental plastic slips (= Delta lambda)
    eps     : (3,3) jnp array — total strain
    eps_p_n : (3,3) jnp array — plastic strain from previous step
    gamma_n : (m,) jnp array — accumulated slips from previous step
    C       : (3,3,3,3) jnp array — elasticity tensor
    Za      : (m,3,3) jnp array — Schmid tensors Z^alpha
    tau0    : float — initial critical resolved shear stress
    tau_inf : float — saturation shear stress
    xi      : float — hardening saturation exponent

    Returns
    -------
    Phi : (m,) jnp array — yield function values
    """
    eps_p = eps_p_n + jnp.einsum('a,aij->ij', dgamma, Za)
    sig = jnp.einsum('ijkl,kl->ij', C, eps - eps_p)
    tau = jnp.einsum('aij,ij->a', Za, sig)
    A = jnp.sum(gamma_n + dgamma)
    tau_h = tau_inf * (1 - jnp.exp(- xi * A))
    return tau - (tau0 + tau_h)

yield_jacobian = jax.jit(jax.jacfwd(yield_function, argnums=0))


@jax.jit
def residual_scaled(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi, mu):
    r"""
    Scaled complementarity residual for the IPM:

        R^alpha = dgamma^alpha * Phi^alpha(dgamma) + mu

    Parameters
    ----------
    dgamma  : (m,) jnp array — incremental plastic slips
    eps     : (3,3) jnp array — total strain
    eps_p_n : (3,3) jnp array — plastic strain from previous step
    gamma_n : (m,) jnp array — accumulated slips from previous step
    C       : (3,3,3,3) jnp array — elasticity tensor
    Za      : (m,3,3) jnp array — Schmid tensors Z^alpha
    tau0    : float — initial critical resolved shear stress
    tau_inf : float — saturation shear stress
    xi      : float — hardening saturation exponent
    mu      : float — barrier parameter

    Returns
    -------
    R : (m,) jnp array — scaled residual values
    """
    Phi = yield_function(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)
    return dgamma * Phi + mu

# Jacobians for tangent computation
dR_ddgamma = jax.jit(jax.jacfwd(residual_scaled, argnums=0))    # (m, m)
dR_deps    = jax.jit(jax.jacfwd(residual_scaled, argnums=1))    # (m, 3, 3)

def compute_tangent_IFT(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                        tau0, tau_inf, xi, mu):
    r"""
    Consistent elastoplastic tangent via the implicit function theorem.

    At convergence :math:`R(\Delta\gamma, \varepsilon) = 0`, the IFT gives

    .. math::
        \frac{\partial \Delta\gamma}{\partial \varepsilon}
            = -\left(\frac{\partial R}{\partial \Delta\gamma}\right)^{-1}
              \frac{\partial R}{\partial \varepsilon}

    and the consistent tangent follows as

    .. math::
        \mathbb{C}^{ep}_{ijmn}
            = C_{ijmn}
            - C_{ijkl}\,\sum_\alpha Z^\alpha_{kl}\,
              \frac{\partial \Delta\gamma^\alpha}{\partial \varepsilon_{mn}}

    Parameters
    ----------
    dlambda : (m,) jnp array — converged incremental slips
    eps_j   : (3,3) jnp array — total strain
    eps_p_n : (3,3) jnp array — plastic strain from previous step
    gamma_n : (m,) jnp array — accumulated slips from previous step
    C       : (3,3,3,3) jnp array — elasticity tensor
    Za      : (m,3,3) jnp array — Schmid tensors
    tau0    : float — initial CRSS
    tau_inf : float — saturation shear stress
    xi      : float — hardening saturation exponent
    mu      : float — barrier parameter

    Returns
    -------
    C_ep : (3,3,3,3) jnp array — consistent elastoplastic tangent
    """
    # dR/d(dgamma): (m, m)
    J_gamma = dR_ddgamma(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                         tau0, tau_inf, xi, mu)

    # dR/d(eps): (m, 3, 3)
    J_eps = dR_deps(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                    tau0, tau_inf, xi, mu)

    m = dlambda.shape[0]

    # Solve  J_gamma @ X = -J_eps  for  X = d(dgamma)/d(eps)
    # Flatten spatial dims to (m, 9) for a single linear solve
    ddg_deps = jnp.linalg.solve(J_gamma, -J_eps.reshape(m, 9))   # (m, 9)
    ddg_deps = ddg_deps.reshape(m, 3, 3)                          # (m, 3, 3)

    # C_ep = C - C : (sum_alpha  Z^alpha otimes d(dgamma^alpha)/d(eps))
    C_ep = C - jnp.einsum('ijkl,akl,amn->ijmn', C, Za, ddg_deps)

    return C_ep

# ---------------------------------------------------------------------------
# Main material routine
# ---------------------------------------------------------------------------

def compute_stress(eps: np.ndarray, hist: dict, mat: Material,
                   config: dict,
                   ) -> tuple[np.ndarray, dict, int]:
    r"""
    Compute the Cauchy stress via the classical Interior Point Method for
    rate-independent single crystal plasticity at small strains.

    Parameters
    ----------
    eps    : (3,3) array — prescribed total strain tensor
    hist   : dict — material history from the previous load step
    mat    : Material — material parameters (from cp_base)
    config : dict — solver parameters (loaded from TOML [IPM] section)

    Returns
    -------
    sig      : (3,3) array — Cauchy stress tensor
    new_hist : dict — updated material history
    n_iter   : int — total number of IPM iterations (−1 on failure)
    """

    # -- Unpack config ------------------------------------------------------
    k_max     = config["k_max"]
    max_inner = config["max_inner"]
    mu_init   = config["mu_init"]

    # sacled barrier parameter tolerance
    mu_end    = config["mu_end"] * config["dt"] * mat.tau0
    tol_end   = config["tol_end"]
    theta     = config["theta"]
    delta     = config["delta"]
    tau_min   = config["tau_min"]
    verbose   = config["verbose"]

    nSlip = len(mat.Za)

    # -- Convert material data to JAX arrays --------------------------------
    Za      = jnp.array(np.stack(mat.Za))    # (m, 3, 3)
    C       = jnp.array(mat.C)               # (3,3,3,3)
    eps_j   = jnp.array(eps)                 # (3,3)
    eps_p_n = jnp.array(hist["eps_p"])        # (3,3)
    gamma_n = jnp.array(hist["gamma_a"])      # (m,)
    tau0    = mat.tau0
    tau_inf = mat.tau_inf
    xi      = mat.xi

    # Shorthand closures using module-level JIT'd functions
    def Phi_fn(dg):
        return yield_function(dg, eps_j, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)

    def dPhi_fn(dg):
        return yield_jacobian(dg, eps_j, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)

    # ======================================================================
    # 1.  Elastic predictor
    # ======================================================================
    Phi_trial = Phi_fn(jnp.zeros(nSlip))

    if np.all(np.array(Phi_trial) <= 0.0):
        sig_trial = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_n)
        if verbose:
            print("  Elastic predictor successful.")
        hist_el = hist.copy()
        hist_el["C_ep"] = np.array(C)
        hist_el["yield"] = np.array(Phi_trial)
        return np.array(sig_trial), hist_el, 0

    # ======================================================================
    # 2.  Initialization of IPM
    # ======================================================================
    mu = mu_init
    slacks = jnp.maximum(-Phi_trial, jnp.sqrt(mu))
    dlambda = jnp.ones(nSlip) * jnp.sqrt(mu)

    # Initial residual
    Phi = Phi_fn(dlambda)
    R_g = Phi + slacks
    R_s = dlambda * slacks - mu
    R   = jnp.concatenate([R_g, R_s])
    r   = float(jnp.linalg.norm(R))
    r0 = r # store initial residual

    total_newton_iter = 0
    k = 0 #counter for outer iterations
    n = 0 #counter for inner iterations

    # ======================================================================
    # 3.  Outer barrier loop
    # ======================================================================
    while k <= k_max:

        # Adaptive Newton tolerance for inner loop
        newton_tol_inner = max(tol_end, theta * mu)
        if mu <= mu_end:
            newton_tol_inner = tol_end
        n = 0
        # ------------------------------------------------------------------
        # Inner Newton loop for current mu
        # ------------------------------------------------------------------
        while (r > newton_tol_inner) and (n < max_inner):
            # Jacobian of yield function via AD
            dPhi = dPhi_fn(dlambda)            # (m, m)

            # build residual and Jacobian for the full KKT system
            I_m = jnp.eye(nSlip)
            J = jnp.block([
                [dPhi, I_m],
                [jnp.diag(slacks), jnp.diag(dlambda)]
            ])
            rhs = -jnp.concatenate([R_g, R_s])
            delta_x = jnp.linalg.solve(J, rhs)
            d_dlambda = delta_x[:nSlip]
            d_slacks = delta_x[nSlip:]

            # Fraction-to-boundary
            alpha_dl = jnp.where(d_dlambda < 0, -tau_min * dlambda / d_dlambda, jnp.inf)
            alpha_ds = jnp.where(d_slacks  < 0, -tau_min * slacks  / d_slacks,  jnp.inf)
            alpha_max = min(1.0, float(jnp.min(alpha_dl)), float(jnp.min(alpha_ds)))

            # Update primal-dual iterates
            dlambda = dlambda + alpha_max * d_dlambda
            slacks = slacks + alpha_max * d_slacks

            # Recompute residual
            R_g = Phi_fn(dlambda) + slacks
            R_s = dlambda * slacks - mu
            R   = jnp.concatenate([R_g, R_s])
            r   = float(jnp.linalg.norm(R)) /r0

            #increase counters
            n += 1
            total_newton_iter += 1

        # check for Newton divergence
        if n >= max_inner:
            if verbose:
                print(f"  IPM failed to converge for mu = {mu:.2e} after {n} Newton iterations.")
            return np.zeros((3, 3)), hist, -1

        if mu <= mu_end and r < tol_end:
            break

        # adaptive barrier parameter update (Niehüser 2023)
        mu = max(float(jnp.max(dlambda * slacks/r0) ** (1.0 + delta)), mu_end)

        # Recompute residual with new mu
        R_s = dlambda * slacks - mu
        R   = jnp.concatenate([R_g, R_s])
        r   = float(jnp.linalg.norm(R)) / r0

        if verbose:
            print(f"  IPM iter {total_newton_iter}: mu = {mu:.2e}, ||R|| = {r:.2e}")

        k = k + 1

    # ======================================================================
    # 4.  Post-processing
    # ======================================================================
    converged = mu <= mu_end and r < tol_end

    eps_p_new = eps_p_n + jnp.einsum('a,aij->ij', dlambda, Za)
    sig = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_new)

    if verbose:
        status = "converged" if converged else "NOT converged"
        print(f"  IPM {status} in {total_newton_iter} iterations, mu = {mu:.2e}, ||R|| = {r:.2e}")

    if not converged:
        print("WARNING: IPM did not converge to the desired tolerance. Returning trial stress.")
        return np.zeros((3, 3)), hist, -1

    # Update history
    new_hist = hist.copy()
    new_hist["eps"]     = np.array(eps)
    new_hist["eps_p"]   = np.array(eps_p_new)
    new_hist["gamma_a"] = hist["gamma_a"] + np.array(dlambda)
    new_hist["tau_h"]   = hist["tau_h"]          # no hardening update
    new_hist["n_iter"]  = total_newton_iter
    new_hist["yield"]   = np.array(Phi_fn(dlambda))

    # Consistent tangent via IFT
    C_ep = compute_tangent_IFT(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                               tau0, tau_inf, xi, mu)
    new_hist["C_ep"] = np.array(C_ep)

    return np.array(sig), new_hist, total_newton_iter
