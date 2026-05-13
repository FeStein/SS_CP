"""
Augmented Lagrangian method for rate-independent single crystal plasticity
at small strains (Schmidt-Baldassari 2003).

Implements the semi-smooth Newton AL algorithm from doc/al.md:
  - Residual: R^alpha = dgamma^alpha - max(0, dgamma^alpha + eta * Phi^alpha)
  - Active set determines piecewise Jacobian for Newton linearisation
  - Outer loop doubles eta if KKT complementarity conditions are not met
  - Consistent tangent via the implicit function theorem with JAX AD
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cp_matroutines.cp_base import Material
from cp_matroutines.mat_IPM_classic import yield_function, yield_jacobian


# ---------------------------------------------------------------------------
# AL residual and its Jacobians — JIT'd at module level
# ---------------------------------------------------------------------------

@jax.jit
def al_residual(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi, eta):
    r"""
    Augmented Lagrangian residual for each slip system:

        R^alpha = dgamma^alpha - max(0, dgamma^alpha + eta * Phi^alpha(dgamma))

    Roots of R enforce the KKT conditions exactly.
    """
    Phi = yield_function(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)
    return dgamma - jnp.maximum(0.0, dgamma + eta * Phi)


dR_ddgamma_al = jax.jit(jax.jacfwd(al_residual, argnums=0))   # (m, m)
dR_deps_al    = jax.jit(jax.jacfwd(al_residual, argnums=1))   # (m, 3, 3)


def compute_tangent_IFT_al(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                            tau0, tau_inf, xi, eta):
    r"""
    Consistent elastoplastic tangent via the implicit function theorem.

    At convergence R(dgamma, eps) = 0, the IFT gives:

        d(dgamma)/d(eps) = -(dR/d(dgamma))^{-1} dR/d(eps)

    and:

        C_ep = C - C : (sum_alpha Z^alpha otimes d(dgamma^alpha)/d(eps))
    """
    J_gamma = dR_ddgamma_al(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                             tau0, tau_inf, xi, eta)
    J_eps   = dR_deps_al(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                          tau0, tau_inf, xi, eta)
    m = dlambda.shape[0]
    ddg_deps = jnp.linalg.solve(J_gamma, -J_eps.reshape(m, 9))
    ddg_deps = ddg_deps.reshape(m, 3, 3)
    return C - jnp.einsum('ijkl,akl,amn->ijmn', C, Za, ddg_deps)


# ---------------------------------------------------------------------------
# Main material routine
# ---------------------------------------------------------------------------

def compute_stress(eps: np.ndarray, hist: dict, mat: Material,
                   config: dict) -> tuple[np.ndarray, dict, int]:
    r"""
    Compute the Cauchy stress via the Augmented Lagrangian method for
    rate-independent single crystal plasticity at small strains.

    Parameters
    ----------
    eps    : (3,3) array — prescribed total strain tensor
    hist   : dict — material history from the previous load step
    mat    : Material — material parameters (from cp_base)
    config : dict — solver parameters (loaded from TOML [AL] section)

    Returns
    -------
    sig      : (3,3) array — Cauchy stress tensor
    new_hist : dict — updated material history
    n_iter   : int — total Newton iterations (−1 on failure)
    """

    # -- Unpack config -------------------------------------------------------
    eta_init  = config["eta_init"]
    max_outer = config["max_outer"]
    max_inner = config["max_inner"]
    tol_inner = config["tol_inner"]
    tol_kkt   = config["tol_kkt"]
    verbose   = config["verbose"]

    nSlip = len(mat.Za)

    # -- Convert material data to JAX arrays ---------------------------------
    Za      = jnp.array(np.stack(mat.Za))
    C       = jnp.array(mat.C)
    eps_j   = jnp.array(eps)
    eps_p_n = jnp.array(hist["eps_p"])
    gamma_n = jnp.array(hist["gamma_a"])
    tau0    = mat.tau0
    tau_inf = mat.tau_inf
    xi      = mat.xi

    def Phi_fn(dg):
        return yield_function(dg, eps_j, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)

    def dPhi_fn(dg):
        return yield_jacobian(dg, eps_j, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)

    # ========================================================================
    # 1.  Elastic predictor
    # ========================================================================
    Phi_trial = Phi_fn(jnp.zeros(nSlip))
    if np.all(np.array(Phi_trial) <= 0.0):
        sig_trial = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_n)
        if verbose:
            print("  Elastic predictor successful.")
        hist_el = hist.copy()
        hist_el["C_ep"] = np.array(C)
        hist_el["yield"] = np.array(Phi_trial)
        return np.array(sig_trial), hist_el, 0

    # ========================================================================
    # 2.  Initialisation
    # ========================================================================
    dlambda = jnp.zeros(nSlip)
    eta = float(eta_init)
    total_newton_iter = 0
    converged = False

    # ========================================================================
    # 3.  Outer penalty loop
    # ========================================================================
    for outer in range(max_outer):
        inner_converged = False
        n_inner = 0

        # Reference point fixed for the duration of this inner loop
        # (mirrors dlambda_i in the Julia implementation)
        dlambda_i = dlambda

        # --------------------------------------------------------------------
        # Semi-smooth Newton loop for current eta
        # --------------------------------------------------------------------
        for n_inner in range(max_inner):
            Phi  = Phi_fn(dlambda)
            dPhi = dPhi_fn(dlambda)

            # Active set uses the fixed reference dlambda_i plus current Phi
            active = (dlambda_i + eta * Phi) > 0.0

            # FF = I for inactive rows; FF = I - eta * dPhi for active rows
            # (direct translation of the Julia FF construction)
            FF = jnp.where(active[:, None], jnp.eye(nSlip) - eta * dPhi, jnp.eye(nSlip))

            # Residual uses the fixed reference dlambda_i
            R = dlambda - jnp.maximum(0.0, dlambda_i + eta * Phi)

            # Newton step: FF * ddlambda = -R
            ddlambda = jnp.linalg.solve(FF, -R)
            dlambda = dlambda + ddlambda

            total_newton_iter += 1

            if float(jnp.linalg.norm(ddlambda)) < tol_inner:
                inner_converged = True
                break

        if not inner_converged and verbose:
            print(f"  AL outer {outer + 1}: inner Newton did not converge "
                  f"in {max_inner} iterations.")

        # Check KKT complementarity: dlambda^alpha * Phi^alpha = 0 for all alpha
        Phi_cur = Phi_fn(dlambda)
        kkt = float(jnp.linalg.norm(dlambda * Phi_cur))

        if verbose:
            print(f"  AL outer {outer + 1}: eta = {eta:.2e}, "
                  f"KKT = {kkt:.2e}, Newton iters = {n_inner + 1}")

        if kkt < tol_kkt:
            converged = True
            break

        eta *= 2.0

    # ========================================================================
    # 4.  Post-processing
    # ========================================================================
    if not converged:
        print("WARNING: AL did not converge. Returning zero stress.")
        return np.zeros((3, 3)), hist, -1

    eps_p_new = eps_p_n + jnp.einsum('a,aij->ij', dlambda, Za)
    sig       = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_new)

    if verbose:
        print(f"  AL converged in {total_newton_iter} total Newton iterations.")

    new_hist = hist.copy()
    new_hist["eps"]     = np.array(eps)
    new_hist["eps_p"]   = np.array(eps_p_new)
    new_hist["gamma_a"] = hist["gamma_a"] + np.array(dlambda)
    new_hist["tau_h"]   = hist["tau_h"]
    new_hist["n_iter"]  = total_newton_iter
    new_hist["yield"]   = np.array(Phi_fn(dlambda))

    C_ep = compute_tangent_IFT_al(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                                   tau0, tau_inf, xi, eta)
    new_hist["C_ep"] = np.array(C_ep)

    return np.array(sig), new_hist, total_newton_iter
