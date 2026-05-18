"""
Modified-Barrier Interior Point Method for rate-independent single crystal plasticity
at small strains (MB-IPM).

Implements the nested-loop MB-IPM following Polyak's nonlinear-rescaling approach
(Polyak 1992; Polyak & Teboulle 1997):
  - Modified barrier: -mu * lam_k * ln(1 + s/mu)
  - Shifted complementarity: dlambda * (s + mu) = mu * lam_k
  - Explicit Lagrange-multiplier updates (Polyak update rule)
  - Shift parameter mu stays fixed (or decays mildly); convergence via outer mult loop
  - No bias on stress/tangent at convergence (exact KKT recovered)

Derivatives of the yield function are computed via JAX automatic differentiation.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cp_matroutines.cp_base import Material


# ---------------------------------------------------------------------------
# Yield function and Jacobian — JIT'd at module level
# ---------------------------------------------------------------------------

@jax.jit
def yield_function(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi):
    r"""
    Yield function for all slip systems:
        Phi^alpha = Z^alpha : sigma(dgamma) - (tau_0 + tau_h(A))
    with exponential saturation hardening:
        tau_h = tau_inf * (1 - exp(-xi * A)),   A = sum(gamma_n + dgamma)
    """
    eps_p = eps_p_n + jnp.einsum('a,aij->ij', dgamma, Za)
    sig   = jnp.einsum('ijkl,kl->ij', C, eps - eps_p)
    tau   = jnp.einsum('aij,ij->a', Za, sig)
    A     = jnp.sum(gamma_n + dgamma)
    tau_h = tau_inf * (1 - jnp.exp(-xi * A))
    return tau - (tau0 + tau_h)

yield_jacobian = jax.jit(jax.jacfwd(yield_function, argnums=0))


@jax.jit
def residual_mb(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi, mu, lam_k):
    r"""
    Reduced MB-IPM complementarity residual (slacks eliminated via s = -Phi):
        R^alpha = dgamma^alpha * (mu - Phi^alpha) - mu * lam_k^alpha

    Used only for the consistent tangent computation via IFT; lam_k is frozen.
    """
    Phi = yield_function(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)
    return dgamma * (mu - Phi) - mu * lam_k

dR_ddgamma_mb = jax.jit(jax.jacfwd(residual_mb, argnums=0))  # (m, m)
dR_deps_mb    = jax.jit(jax.jacfwd(residual_mb, argnums=1))  # (m, 3, 3)


def compute_tangent_IFT_mb(dlambda, lam_k, eps_j, eps_p_n, gamma_n, C, Za,
                           tau0, tau_inf, xi, mu):
    r"""
    Consistent elastoplastic tangent via the implicit function theorem applied
    to the reduced MB-IPM residual R_mb(dgamma, eps) = 0 (lam_k frozen at
    the converged outer iterate).

    At outer convergence the residual is the exact KKT system, so the tangent
    carries no mu-dependent bias.
    """
    J_gamma = dR_ddgamma_mb(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                            tau0, tau_inf, xi, mu, lam_k)
    J_eps   = dR_deps_mb(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                         tau0, tau_inf, xi, mu, lam_k)
    m = dlambda.shape[0]
    ddg_deps = jnp.linalg.solve(J_gamma, -J_eps.reshape(m, 9))
    ddg_deps = ddg_deps.reshape(m, 3, 3)
    C_ep = C - jnp.einsum('ijkl,akl,amn->ijmn', C, Za, ddg_deps)
    return C_ep


# ---------------------------------------------------------------------------
# Main material routine
# ---------------------------------------------------------------------------

def compute_stress(eps: np.ndarray, hist: dict, mat: Material,
                   config: dict) -> tuple[np.ndarray, dict, int]:
    r"""
    Compute the Cauchy stress via the Modified-Barrier IPM for rate-independent
    single crystal plasticity at small strains.

    Parameters
    ----------
    eps    : (3,3) array — prescribed total strain tensor
    hist   : dict — material history from the previous load step
    mat    : Material — material parameters (from cp_base)
    config : dict — solver parameters from the [MB_IPM] config section

    Returns
    -------
    sig      : (3,3) array — Cauchy stress tensor
    new_hist : dict — updated material history
    n_iter   : int — total Newton iterations (-1 on failure)
    """

    # -- Unpack config ----------------------------------------------------------
    k_max     = config["k_max"]
    max_inner = config["max_inner"]
    mu_init   = float(config["mu_init"])
    mu_floor  = float(config["mu_floor"])
    rho       = float(config["rho"])
    lam_init  = float(config["lambda_init"])
    tol_inner = float(config["tol_inner"])
    tol_mult  = float(config["tol_mult"])
    tol_compl = float(config["tol_compl"])
    tau_min   = float(config["tau_min"])
    verbose   = config["verbose"]

    nSlip = len(mat.Za)

    # -- Convert material data to JAX arrays ------------------------------------
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

    # ==========================================================================
    # 1. Elastic predictor
    # ==========================================================================
    Phi_trial = Phi_fn(jnp.zeros(nSlip))
    if np.all(np.array(Phi_trial) <= 0.0):
        sig_trial = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_n)
        if verbose:
            print("  MB-IPM: elastic predictor successful.")
        hist_el = hist.copy()
        hist_el["C_ep"]  = np.array(C)
        hist_el["yield"] = np.array(Phi_trial)
        return np.array(sig_trial), hist_el, 0

    # ==========================================================================
    # 2. Initialization
    # ==========================================================================
    mu      = mu_init
    lam_k   = jnp.ones(nSlip) * lam_init   # must be > 0 (zero is a fixed point of update)
    slacks  = jnp.maximum(-Phi_trial, jnp.sqrt(mu))
    dlambda = jnp.ones(nSlip) * jnp.sqrt(mu)

    Phi = Phi_fn(dlambda)
    R_g = Phi + slacks
    R_s = dlambda * (slacks + mu) - mu * lam_k
    R   = jnp.concatenate([R_g, R_s])
    r   = float(jnp.linalg.norm(R))

    total_newton_iter = 0
    converged = False

    # ==========================================================================
    # 3. Outer Lagrange-multiplier loop (index k)
    # ==========================================================================
    for k in range(k_max + 1):

        # ----------------------------------------------------------------------
        # 3a. Inner Newton solve at fixed (mu, lam_k)
        # ----------------------------------------------------------------------
        n = 0
        while r > tol_inner and n < max_inner:
            dPhi = dPhi_fn(dlambda)
            I_m  = jnp.eye(nSlip)

            # Full 2m x 2m primal-dual Newton system
            # (2,1) block uses (s + mu) — the MB-IPM shift
            J = jnp.block([
                [dPhi,                  I_m               ],
                [jnp.diag(slacks + mu), jnp.diag(dlambda) ],
            ])
            rhs     = -jnp.concatenate([R_g, R_s])
            delta_x = jnp.linalg.solve(J, rhs)
            d_dlambda = delta_x[:nSlip]
            d_slacks  = delta_x[nSlip:]

            # Fraction-to-boundary: keep dlambda > 0 and slacks + mu > 0
            alpha_dl  = jnp.where(d_dlambda < 0,
                                  -tau_min * dlambda / d_dlambda, jnp.inf)
            alpha_ds  = jnp.where(d_slacks  < 0,
                                  -tau_min * (slacks + mu) / d_slacks, jnp.inf)
            alpha_max = min(1.0, float(jnp.min(alpha_dl)), float(jnp.min(alpha_ds)))

            dlambda = dlambda + alpha_max * d_dlambda
            slacks  = slacks  + alpha_max * d_slacks

            Phi = Phi_fn(dlambda)
            R_g = Phi + slacks
            R_s = dlambda * (slacks + mu) - mu * lam_k
            R   = jnp.concatenate([R_g, R_s])
            r   = float(jnp.linalg.norm(R))

            n += 1
            total_newton_iter += 1

        if n >= max_inner:
            if verbose:
                print(f"  MB-IPM: inner Newton failed at outer k={k}, "
                      f"mu={mu:.2e}, ||R||={r:.2e}")
            return np.zeros((3, 3)), hist, -1

        # ----------------------------------------------------------------------
        # 3b. Multiplier update (Polyak rule): lam_{k+1} = mu * lam_k / (s + mu)
        # ----------------------------------------------------------------------
        lam_k_new   = mu * lam_k / (slacks + mu)
        mult_change = float(jnp.max(
            jnp.abs(lam_k_new - lam_k) / jnp.maximum(1.0, lam_k)))
        compl_gap   = float(jnp.max(jnp.abs(lam_k_new * slacks)))

        lam_k = lam_k_new

        if verbose:
            print(f"  MB-IPM k={k}: mu={mu:.2e}, delta_lam={mult_change:.2e}, "
                  f"compl={compl_gap:.2e}, ||R||={r:.2e}, n_inner={n}")

        # ----------------------------------------------------------------------
        # 3c. Outer convergence check
        # ----------------------------------------------------------------------
        if mult_change < tol_mult or compl_gap < tol_compl:
            converged = True
            break

        # ----------------------------------------------------------------------
        # 3d. Optional mu update (default rho=1.0: fixed shift)
        # ----------------------------------------------------------------------
        mu = max(rho * mu, mu_floor)

        # Recompute residual with updated (lam_k, mu)
        R_s = dlambda * (slacks + mu) - mu * lam_k
        R   = jnp.concatenate([R_g, R_s])
        r   = float(jnp.linalg.norm(R))

    # ==========================================================================
    # 4. Post-processing
    # ==========================================================================
    if not converged:
        print(f"WARNING: MB-IPM did not converge in {k_max} outer iterations.")
        return np.zeros((3, 3)), hist, -1

    eps_p_new = eps_p_n + jnp.einsum('a,aij->ij', dlambda, Za)
    sig       = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_new)

    if verbose:
        print(f"  MB-IPM converged: {total_newton_iter} Newton iters, "
              f"{k + 1} outer iters, mu={mu:.2e}")

    new_hist = hist.copy()
    new_hist["eps"]     = np.array(eps)
    new_hist["eps_p"]   = np.array(eps_p_new)
    new_hist["gamma_a"] = hist["gamma_a"] + np.array(dlambda)
    new_hist["tau_h"]   = hist["tau_h"]
    new_hist["n_iter"]  = total_newton_iter
    new_hist["yield"]   = np.array(Phi_fn(dlambda))

    C_ep = compute_tangent_IFT_mb(dlambda, lam_k, eps_j, eps_p_n, gamma_n, C, Za,
                                   tau0, tau_inf, xi, mu)
    new_hist["C_ep"] = np.array(C_ep)

    return np.array(sig), new_hist, total_newton_iter
