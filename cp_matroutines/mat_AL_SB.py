"""
Augmented Lagrangian method for rate-independent single crystal plasticity
at small strains (Schmidt-Baldassari 2003).

Implements the *classic* augmented Lagrangian algorithm (Algorithm 1) exactly
as stated in the reference paper, so results can be cited against it directly:

  1: initialise i = 0, dlambda^(0) = 0, eta^(0) (penalty)
  2: while i <= i_max:                                   (fixed-point / penalty loop)
  3:     while ||R||_2 > eps_min:                        (semi-smooth Newton loop)
  4:         solve R^a(dlambda^(i+1)) = 0  (Eq. 17, Jacobian Eq. 18)
  5:         (line search / damping -- optional, not used here: full Newton step)
  6:     end while
  7:     if Phi^a <= eps_c and dlambda^a >= -eps_c and |Phi^a dlambda^a| <= eps_c
         for all a:                                       (KKT check, product form)
  8:         converged, exit
  9:     else:
 10:         implicit multiplier update (reference dlambda_i <- dlambda)
 11:         eta^(i+1) = 2 eta^(i), i <- i + 1
 12:     end if
 13: end while

  - Residual: R^alpha = dgamma^alpha - max(0, dgamma^alpha + eta * Phi^alpha)
  - Active set determines the piecewise Jacobian for the Newton linearisation.
  - Inner convergence on the residual 2-norm ||R||_2 < eps_min (Alg. 1, line 3).
  - Outer convergence on the KKT conditions in *product* form (Alg. 1, line 7):
    primal feasibility Phi <= eps_c, dual feasibility dlambda >= -eps_c, and the
    product complementarity |Phi * dlambda| <= eps_c. Note that the product gap
    certifies the yield residual only as |Phi| <~ eps_c / dlambda, so its
    stress-unit accuracy is not uniform in the load-increment size -- this is
    the paper's stated criterion; the increment-uniform natural (min-form)
    residual is available instead in mat_IPM_mb.py / mat_AL_SB_converged.py.
  - Consistent tangent via the implicit function theorem with JAX AD.

Derivatives of the yield function are computed via JAX automatic differentiation.
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
    Compute the Cauchy stress via the classic Augmented Lagrangian algorithm
    (Algorithm 1) for rate-independent single crystal plasticity at small
    strains.

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
    eta_init  = float(config["eta_init"])
    max_outer = int(config["max_outer"])
    max_inner = int(config["max_inner"])
    # eps_c : KKT-check tolerance (Alg. 1, line 7). eps_min : inner Newton
    # residual tolerance (Alg. 1, line 3). Fall back to the previous key names
    # (tol_phi / theta_in) so existing configs keep working.
    eps_c   = float(config.get("eps_c", config.get("tol_phi", 1e-10)))
    eps_min = float(config.get("eps_min",
                               float(config.get("theta_in", 1e-1)) * eps_c))
    verbose = config["verbose"]
    # Opt-in per-iteration tracer (single-step diagnostics only; see
    # examples/ex_single_step.py). None => no recording, zero overhead.
    trace = [] if config.get("trace", False) else None

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
        hist_el["trace"] = trace
        hist_el["cond_final"] = np.nan      # no Newton system solved
        # Penalty-ladder diagnostics (see the block after convergence). No rung
        # was climbed here, so the ladder stays at its entry value.
        hist_el["n_outer"]   = 0
        hist_el["eta_final"] = eta_init
        hist_el["phi_max"]   = float(jnp.max(Phi_trial))
        hist_el["compl_max"] = np.nan       # no multiplier to pair with Phi
        return np.array(sig_trial), hist_el, 0

    # ========================================================================
    # 2.  Initialisation (Algorithm 1, line 1)
    # ========================================================================
    # Cold-start seed: scalar (uniform) or a length-m sequence. dlambda is both
    # the primal slip and the multiplier estimate here, so this is the only
    # handle on which solution is picked when the active set is rank deficient.
    _dg0 = config.get("dgamma_init", None)
    dlambda = (jnp.zeros(nSlip) if _dg0 is None else
               jnp.array(np.broadcast_to(np.asarray(_dg0, dtype=float),
                                         (nSlip,)).copy()))
    eta = eta_init
    total_newton_iter = 0
    converged = False

    # ========================================================================
    # 3.  Fixed-point / penalty loop (Algorithm 1, line 2)
    # ========================================================================
    for outer in range(max_outer):

        # Multiplier reference for this inner solve. Updated to the current
        # iterate at the start of every outer step -- this is the implicit
        # update of Algorithm 1, line 10.
        dlambda_i = dlambda

        # --------------------------------------------------------------------
        # 3a. Semi-smooth Newton loop: iterate while ||R||_2 > eps_min
        #     (Algorithm 1, line 3).
        # --------------------------------------------------------------------
        inner_converged = False
        inner_steps = 0
        for _ in range(max_inner):
            Phi  = Phi_fn(dlambda)
            dPhi = dPhi_fn(dlambda)

            # Active set uses the fixed reference dlambda_i plus current Phi.
            active = (dlambda_i + eta * Phi) > 0.0

            # FF = I for inactive rows; FF = I - eta * dPhi for active rows
            # (the piecewise semi-smooth Newton Jacobian, Eq. 18).
            FF = jnp.where(active[:, None], jnp.eye(nSlip) - eta * dPhi, jnp.eye(nSlip))

            # Residual R^a = dlambda^a - max(0, dlambda_i^a + eta * Phi^a).
            R = dlambda - jnp.maximum(0.0, dlambda_i + eta * Phi)
            res_norm = float(jnp.linalg.norm(R))

            # Inner convergence test (Algorithm 1, line 3, loop condition).
            if not np.isfinite(res_norm):
                break
            if res_norm < eps_min:
                inner_converged = True
                break

            # Newton step: FF * ddlambda = -R   (Eq. 17). Full step; line 5's
            # line search / damping is optional and not used here.
            ddlambda = jnp.linalg.solve(FF, -R)
            dlambda  = dlambda + ddlambda

            total_newton_iter += 1
            inner_steps += 1

            # Per-iteration diagnostics (single-step tracer only). AL has no
            # primal-dual barrier system; the solved matrix is the m x m
            # semi-smooth Newton matrix FF, so cond_solved and cond_kkt both
            # report cond(FF). The continuation parameter is the penalty eta
            # (grows), not a barrier mu. r_abs is the inner residual norm
            # ||R||_2 (AL's inner convergence metric, Alg. 1 line 3). compl_gap
            # is the product gap max_a |dlambda_a * Phi_a| -- the same quantity
            # the KKT check (line 7) tests, and comparable to the IPM traces.
            if trace is not None:
                Phi_new = Phi_fn(dlambda)
                cond_FF = float(jnp.linalg.cond(FF))
                trace.append({
                    "iter": total_newton_iter, "k": outer, "n": inner_steps,
                    "eta": eta, "r_abs": res_norm, "r_rel": res_norm,
                    "compl_gap": float(jnp.max(jnp.abs(dlambda * Phi_new))),
                    "cond_solved": cond_FF, "cond_kkt": cond_FF,
                    "alpha": 1.0,
                    "n_active": int(np.sum(np.array(dlambda) > 1e-10)),
                })

        if not inner_converged and verbose:
            print(f"  AL outer {outer + 1}: inner Newton did not reach "
                  f"||R||_2 < {eps_min:.1e} in {max_inner} iterations.")

        # --------------------------------------------------------------------
        # 3b. KKT check in product form (Algorithm 1, line 7):
        #     Phi^a <= eps_c  and  dlambda^a >= -eps_c  and
        #     |Phi^a * dlambda^a| <= eps_c  for all a.
        # --------------------------------------------------------------------
        Phi_cur   = Phi_fn(dlambda)
        kkt_feas  = float(jnp.max(Phi_cur))                       # primal: want <= eps_c
        kkt_dual  = float(jnp.min(dlambda))                       # dual:   want >= -eps_c
        kkt_compl = float(jnp.max(jnp.abs(Phi_cur * dlambda)))    # complementarity: want <= eps_c

        if verbose:
            print(f"  AL outer {outer + 1}: eta = {eta:.2e}, "
                  f"max Phi = {kkt_feas:.2e}, min dlambda = {kkt_dual:.2e}, "
                  f"max|Phi*dlambda| = {kkt_compl:.2e}, "
                  f"Newton iters = {inner_steps}")

        if (kkt_feas <= eps_c) and (kkt_dual >= -eps_c) and (kkt_compl <= eps_c):
            converged = True
            break

        # Otherwise: double the penalty and continue (Algorithm 1, line 11).
        eta *= 2.0

    # ========================================================================
    # 4.  Post-processing
    # ========================================================================
    if not converged:
        if verbose:
            print("WARNING: AL did not converge.")
        hist["trace"] = trace
        hist["cond_final"] = np.nan
        hist["n_outer"]   = max_outer
        hist["eta_final"] = float(eta)
        hist["phi_max"]   = kkt_feas
        hist["compl_max"] = kkt_compl
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

    # Penalty-ladder diagnostics. eta is reset to eta_init at every load step and
    # doubled once per outer iteration, so the solve terminates on the discrete
    # ladder eta_init * 2^k with k = n_outer - 1. The ladder is entered at a
    # height set by the trial overstress Phi_trial ~ G * dgamma ~ dt (cold start
    # dlambda = 0), which makes the terminal accuracy load-increment dependent:
    # within a fixed k the error inherits that entry height and decays as O(dt),
    # and it jumps back up whenever a smaller increment lets the loop drop a rung.
    # phi_max / compl_max record which KKT condition (Alg. 1, line 7) was binding.
    new_hist["n_outer"]   = outer + 1
    new_hist["eta_final"] = float(eta)
    new_hist["phi_max"]   = kkt_feas
    new_hist["compl_max"] = kkt_compl

    C_ep = compute_tangent_IFT_al(dlambda, eps_j, eps_p_n, gamma_n, C, Za,
                                   tau0, tau_inf, xi, eta)
    new_hist["C_ep"] = np.array(C_ep)
    new_hist["trace"] = trace

    # Conditioning of the final semi-smooth Newton matrix at the converged iterates
    dPhi_final   = dPhi_fn(dlambda)
    active_final = (dlambda_i + eta * Phi_cur) > 0.0
    FF_final     = jnp.where(active_final[:, None],
                             jnp.eye(nSlip) - eta * dPhi_final, jnp.eye(nSlip))
    new_hist["cond_final"] = float(jnp.linalg.cond(FF_final))

    # Final Newton system exported f
    if trace is not None:
        new_hist["newton_final"] = {
            "solved":  np.array(FF_final),
            "kkt":     np.array(FF_final),
            "dPhi":    np.array(dPhi_final),
            "dlambda": np.array(dlambda),
            "Phi":     np.array(Phi_cur),
            "active":  np.array(active_final),
            "eta":     float(eta),
        }

    return np.array(sig), new_hist, total_newton_iter
