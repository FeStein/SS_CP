"""
Modified-Barrier Interior Point Method for rate-independent single crystal plasticity
at small strains (MB-IPM).

Implements the nested-loop MB-IPM following Polyak's nonlinear-rescaling approach
(Polyak 1992; Polyak & Teboulle 1997):
  - Modified barrier: -mu * lam_k * ln(1 + s/mu)
  - Shifted complementarity: dlambda * (s + mu) = mu * lam_k
  - Explicit Lagrange-multiplier updates (Polyak update rule)
  - Adjusted Parameter Version (APV): mu self-tunes via a mu-independent merit
    -- the change in the plastic-strain increment ||sum_a d_lam_a Z_a|| -- so
    smaller mu => faster outer rate (C = mu/(mu + a*lam*)), larger mu =>
    healthier inner solve. mu is lowered on outer stall and raised on
    inner-solve failure, settling at the smallest stable value.
  - Convergence and the merit are measured on the plastic-strain increment, not
    the raw multipliers, so the Taylor (slip non-uniqueness) drift on degenerate
    active sets does not block convergence.
  - Termination is measured in stress units, uniformly in the load-increment
    size: the outer test is the natural (min-form) KKT residual
    r_kkt = max_a |min(a_a * dgamma_a, -Phi_a)| < tol_phi, and the inner
    residual norm scales R_s by the active-slip scale. Product-form criteria
    (|lam*s| < tol, plain ||R|| < tol) certify only |Phi| <~ tol/dgamma and
    degrade as the increments (and with them dgamma) shrink; the stress-unit
    criteria bound the yield residual |Phi| < tol_phi directly at every step
    size. See ipm_mb.md.
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
    mu_ceil   = float(config["mu_ceil"])
    mu_dec    = float(config["mu_dec"])
    mu_inc    = float(config["mu_inc"])
    theta     = float(config["theta"])
    max_retry = int(config["max_retry"])
    lam_init  = float(config["lambda_init"])
    tol_phi   = float(config["tol_phi"])             # outer KKT tolerance in stress units (~ eps_phi * tau0)
    theta_in  = float(config.get("theta_in", 1e-2))  # inner safety factor below tol_phi
    theta_mu  = float(config.get("theta_mu", 1e-2))  # inner relaxation ~ theta_mu * mu in early sweeps
    tol_dep   = float(config["tol_dep"])
    tau_min   = float(config["tau_min"])
    # Inner stall detection: break the inner Newton early when the scaled
    # residual fails to decrease by at least stall_rtol (relative) for
    # stall_patience consecutive steps. On a warm start across an active-set
    # change the inner solve can plateau far above the inner tolerance and burn
    # all of max_inner before the 3b safeguard restarts; catching the stall
    # early triggers that restart sooner.
    stall_patience = int(config.get("inner_stall_patience", 5))
    stall_rtol     = float(config.get("inner_stall_rtol", 1e-3))
    verbose   = config["verbose"]
    debug     = bool(config.get("debug", False))      # warn on primal-dual (lam_k vs dlambda) decoupling
    debug_tol = float(config.get("debug_tol", 1e-2))  # relative drift threshold for that warning
    # Opt-in per-iteration tracer (single-step diagnostics only; see
    # examples/ex_single_step.py). None => no recording, zero overhead.
    trace = [] if config.get("trace", False) else None

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

    # Scaled inner residual (stress units). R_g = Phi + s is already a stress;
    # R_s = dgamma*(s+mu) - mu*lam_k carries stress*slip, and a leftover R_s
    # perturbs the slack by ds ~ R_s/dgamma -- the same 1/dgamma amplification
    # that makes product-form tolerances increment-size dependent. Dividing
    # R_s by the active-slip scale lam_bar converts it into the stress error
    # it causes, so one stress-unit tolerance certifies the yield residual
    # uniformly in the load-increment size. One shared scale (rather than
    # per-component dgamma_a) avoids demanding absurd accuracy on inactive
    # systems, where a small R_s perturbs nothing. np.maximum (not python max)
    # so a NaN propagates into r and is caught by the isfinite safeguards.
    lam_min = 1e-12   # floor guarding the division on barely-plastic steps

    def scaled_residual(R_g, R_s, lam_k_, dlambda_):
        lam_bar = float(np.maximum(np.maximum(jnp.max(lam_k_), jnp.max(dlambda_)),
                                   lam_min))
        return float(np.maximum(jnp.max(jnp.abs(R_g)),
                                jnp.max(jnp.abs(R_s)) / lam_bar))

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
        hist_el["trace"] = trace
        return np.array(sig_trial), hist_el, 0

    # ==========================================================================
    # 2. Initialization
    # ==========================================================================
    mu      = mu_init
    # cold start
    lam_k = jnp.ones(nSlip) * lam_init
    slacks  = jnp.maximum(-Phi_trial, jnp.sqrt(mu))
    dlambda = jnp.ones(nSlip) * jnp.sqrt(mu)

    ## full warm-start
    #lam_k   = jnp.array(hist.get("lam_k", np.ones(nSlip) * lam_init))
    #dlambda = jnp.array(hist.get("lam_k", np.ones(nSlip) * jnp.sqrt(mu)))
    #slacks = -Phi_fn(dlambda)

    Phi = Phi_fn(dlambda)
    R_g = Phi + slacks
    R_s = dlambda * (slacks + mu) - mu * lam_k
    r   = scaled_residual(R_g, R_s, lam_k, dlambda)

    total_newton_iter = 0
    converged  = False
    merit_prev = float("inf")   # APV: previous merit ||d eps_p|| (drives sufficient-decrease test)
    mu_lo      = mu_floor       # APV: dynamic floor on mu, ratcheted up by inner failures
    retries    = 0              # consecutive inner-failure mu-increases

    # ==========================================================================
    # 3. Outer Lagrange-multiplier loop (index k) -- Adjusted Parameter Version
    # ==========================================================================
    for k in range(k_max + 1):

        # ----------------------------------------------------------------------
        # 3a. Inner Newton solve at fixed (mu, lam_k)
        # ----------------------------------------------------------------------
        dgam_prev = dlambda   # primal slip before this outer step (for dep_norm)
        # Adaptive inner tolerance (stress units). theta_in*tol_phi keeps the
        # inner-solve contamination a fixed factor below the outer target
        # tol_phi; theta_mu*mu relaxes early sweeps, where lam_k is still far
        # from converged and polishing the inner solve beyond the O(mu) move
        # of the next Polyak update is wasted work. As the APV lowers mu, the
        # inner tolerance tightens automatically toward theta_in*tol_phi.
        tol_in = max(theta_in * tol_phi, theta_mu * mu)
        n = 0
        r_best = r            # best (lowest) scaled residual seen this inner solve
        stall  = 0            # consecutive steps without sufficient decrease
        while r > tol_in and n < max_inner:
            dPhi = dPhi_fn(dlambda)

            # Schur complement: eliminate d_slacks from the 2m x 2m system.
            # Original block system:
            #   dPhi @ d_dlambda + I @ d_slacks        = -R_g   ... (1)
            #   diag(s+mu) @ d_dlambda + diag(dlambda) @ d_slacks = -R_s   ... (2)
            # From (1): d_slacks = -R_g - dPhi @ d_dlambda
            # Substitute into (2):
            #   M @ d_dlambda = -R_s + dlambda * R_g,  M = diag(s+mu) - diag(dlambda) @ dPhi
            #
            # M is well-conditioned even as dlambda -> 0 (M -> diag(s+mu) > 0),
            # avoiding the ill-conditioning of the 2m x 2m system when inactive
            # slip systems drive dlambda toward zero in late outer iterations.
            M = jnp.diag(slacks + mu) - jnp.diag(dlambda) @ dPhi  # (m, m)
            d_dlambda = jnp.linalg.solve(M, -R_s + dlambda * R_g)
            d_slacks  = -R_g - dPhi @ d_dlambda

            # Fraction-to-boundary: keep dlambda > 0 and slacks + mu > 0.
            # The slack boundary is s = -mu (the barrier's singularity), so the
            # cap must fire on *every* descending slack direction (d_slacks < 0),
            # not only those steeper than -mu.
            alpha_dl  = jnp.where(d_dlambda < 0,
                                  -tau_min * dlambda / d_dlambda, jnp.inf)
            alpha_ds  = jnp.where(d_slacks < 0,
                                  -tau_min * (slacks + mu) / d_slacks, jnp.inf)
            alpha_max = min(1.0, float(jnp.min(alpha_dl)), float(jnp.min(alpha_ds)))

            dlambda = dlambda + alpha_max * d_dlambda
            slacks  = slacks  + alpha_max * d_slacks

            Phi = Phi_fn(dlambda)
            R_g = Phi + slacks
            R_s = dlambda * (slacks + mu) - mu * lam_k
            r   = scaled_residual(R_g, R_s, lam_k, dlambda)

            compl_gap = float(jnp.max(jnp.abs(dlambda * slacks)))

            total_increment = float(jnp.linalg.norm(alpha_max * jnp.concatenate([d_dlambda, d_slacks])))

            # Per-iteration diagnostics (single-step tracer only). cond_solved is
            # the Schur-complement M that is actually solved; cond_kkt is the full
            # 2m x 2m primal-dual matrix, assembled here so it is directly
            # comparable to the classical IPM trace.
            if trace is not None:
                I_m   = jnp.eye(nSlip)
                J_kkt = jnp.block([[dPhi, I_m],
                                   [jnp.diag(slacks + mu), jnp.diag(dlambda)]])
                trace.append({
                    "iter": total_newton_iter + 1, "k": k, "n": n + 1,
                    "mu": mu, "r_abs": r, "r_rel": r,
                    "compl_gap": compl_gap,
                    "cond_solved": float(jnp.linalg.cond(M)),
                    "cond_kkt": float(jnp.linalg.cond(J_kkt)),
                    "alpha": float(alpha_max),
                    "n_active": int(np.sum(np.array(dlambda) > 1e-10)),
                })

            if verbose:
                print(f"    n={n}: r={r:.2e} compl={compl_gap:.2e}, step={total_increment:.2e}")

            n += 1
            total_newton_iter += 1

            # Stall detection (relevant on warm starts across active-set changes):
            # if ||R|| has not dropped by at least stall_rtol relative to the best
            # value seen for stall_patience consecutive steps, the inner solve is
            # plateauing well above tol_in. Break out so the 3b safeguard can
            # restart at a wider mu now rather than after exhausting max_inner.
            if not np.isfinite(r) or r > (1.0 - stall_rtol) * r_best:
                stall += 1
            else:
                stall = 0
            r_best = min(r_best, r) if np.isfinite(r) else r_best
            if stall >= stall_patience:
                if verbose:
                    print(f"    inner stall: no progress on ||R|| for {stall} "
                          f"steps (r={r:.2e}); breaking early.")
                break


        # ----------------------------------------------------------------------
        # 3b. Inner-failure / blow-up safeguard: cold restart at a larger mu.
        #     Triggered by a stiff inner problem (mu too small) OR a non-finite
        #     state. The latter occurs at an active-set change: a newly-activating
        #     system sits near the shifted boundary s -> -mu, where the Polyak
        #     update mu*lam_k/(s+mu) blows up and poisons lam_k (eventually to
        #     inf -> nan). Merely raising mu carries the poison forward, so the
        #     whole iterate is re-initialized from the cold-start values at the
        #     new, wider mu -- including the multiplier estimate lam_k, which is
        #     reset to lam_init (the warm start is what was inconsistent with the
        #     new active set). mu_lo is pinned to the raised mu so the APV (3e)
        #     cannot tighten straight back into the regime that just failed.
        #     (Note: nan fails every '>' test, so it must be caught explicitly or
        #     it traps the outer loop until k_max.)
        # ----------------------------------------------------------------------
        if (not np.isfinite(r)) or r > tol_in:
            if mu >= mu_ceil or retries >= max_retry:
                if verbose:
                    print(f"  MB-IPM: inner Newton failed at outer k={k}, "
                          f"mu={mu:.2e} (ceil/retries spent), ||R||={r:.2e}")
                hist["trace"] = trace
                return np.zeros((3, 3)), hist, -1
            mu       = min(mu * mu_inc, mu_ceil)  # enlarge feasible band
            mu_lo    = mu                         # APV must not undo the raise
            retries += 1
            slacks   = jnp.maximum(-Phi_trial, jnp.sqrt(mu))   # cold restart at new mu
            dlambda  = jnp.ones(nSlip) * jnp.sqrt(mu)
            lam_k    = jnp.ones(nSlip) * lam_init              # drop the poisoned estimate
            Phi = Phi_fn(dlambda)
            R_g = Phi + slacks
            R_s = dlambda * (slacks + mu) - mu * lam_k
            r   = scaled_residual(R_g, R_s, lam_k, dlambda)
            if verbose:
                print(f"  MB-IPM k={k}: inner stall -> cold restart at mu={mu:.2e} "
                      f"(retry {retries}/{max_retry})")
            continue
        retries = 0

        # ----------------------------------------------------------------------
        # 3c. Multiplier update (Polyak rule): lam_{k+1} = mu * lam_k / (s + mu)
        # ----------------------------------------------------------------------
        lam_k_new   = mu * lam_k / (slacks + mu)
        mult_change = float(jnp.max(
            jnp.abs(lam_k_new - lam_k) / jnp.maximum(1.0, lam_k)))   # diagnostic only (Taylor-drifty)

        # ----------------------------------------------------------------------
        # 3c'. (debug) Primal-dual consistency check.
        #   At an exact inner solve R_s = 0, so the Polyak update coincides with
        #   the primal slip: lam_k_new = mu*lam_k/(s+mu) = dlambda. The gap
        #   between them is exactly the inner residual in multiplier space,
        #       dlambda - lam_k_new = R_s / (slacks + mu),
        #   so it is non-zero only because the inner solve was stopped early.
        #   Since the outer convergence test (3d) is built from the *primal*
        #   slip dlambda (natural KKT residual), this drift can no longer make
        #   the convergence test pass for the wrong reason; it remains useful
        #   as a diagnostic that the inner solve is loose relative to the
        #   current mu. Note that under the adaptive inner tolerance
        #   (tol_in ~ theta_mu*mu) some drift in *early* outer sweeps is
        #   expected and harmless. Diagnostic only -- changes nothing.
        # ----------------------------------------------------------------------
        if debug:
            slip_scale = float(jnp.max(jnp.abs(dlambda)))            # active-slip scale
            pd_drift   = float(jnp.max(jnp.abs(lam_k_new - dlambda)))
            rel_drift  = pd_drift / max(slip_scale, 1e-300)
            if rel_drift > debug_tol:
                print(f"  [debug] MB-IPM k={k}: Polyak update decoupled from "
                      f"primal slip by {rel_drift:.2e} (> {debug_tol:.0e}); inner "
                      f"solve loose for mu={mu:.2e} (expected in early sweeps "
                      f"under tol_in={tol_in:.2e}; r={r:.2e}, "
                      f"min lam_k_new={float(jnp.min(lam_k_new)):.2e}).")

        # Outer-progress measure: the change in the plastic-strain increment over
        # this outer step, ||sum_a (dgamma_a - dgamma_prev_a) Z_a||, built from
        # the *primal* slip dlambda (which defines eps_p and the returned stress).
        # This is the physically determined part of the slip: it is invariant
        # under the Taylor null space (drifts with sum_a c_a Z_a = 0 leave it
        # unchanged) and independent of mu, so it serves as both the convergence
        # target and the APV merit. Using the multiplier-estimate change instead
        # (lam_k_new - lam_k = -lam_k*s/(s+mu)) decouples from the primal when
        # mu << s and drifts on a degenerate active set even after the stress has
        # converged. The raw multiplier change (mult_change) has the same defect.
        # See ipm_mb.md.
        d_eps_p   = jnp.einsum('a,aij->ij', dlambda - dgam_prev, Za)
        dep_norm  = float(jnp.linalg.norm(d_eps_p))
        merit     = dep_norm

        # Natural (min-form) KKT residual in stress units, built from the
        # *primal* slip dlambda (not lam_k -- immune to the 3c' decoupling).
        # For scalars: min(a, b) = 0  <=>  a >= 0, b >= 0, a*b = 0, so r_kkt
        # measures the distance to the exact KKT point. The diagonal
        # a_diag = -diag(dPhi/ddgamma) > 0 converts slip to stress so the min
        # compares like with like: on active systems (s -> 0) it bounds the
        # yield residual |Phi| directly; on inactive systems it bounds the
        # stress polluted by spurious slip, a*dgamma; a yield violation
        # Phi > 0 (transiently admissible up to mu in the MB setting) makes
        # the min negative and is flagged by the abs. Unlike the product gap
        # |lam*s| < tol -- which certifies only |Phi| <~ tol/dgamma and
        # degrades as the load increments shrink -- r_kkt < tol_phi bounds the
        # stress error uniformly in the increment size. See ipm_mb.md.
        a_diag = -jnp.diag(dPhi_fn(dlambda))
        r_kkt  = float(jnp.max(jnp.abs(jnp.minimum(a_diag * dlambda, -Phi))))

        lam_k = lam_k_new

        if verbose:
            print(f"  MB-IPM k={k}: mu={mu:.2e}, delta_lam={mult_change:.2e}, "
                  f"kkt={r_kkt:.2e}, dep={dep_norm:.2e}, "
                  f"r={r:.2e}, n_inner={n}")

        # ----------------------------------------------------------------------
        # 3d. Outer convergence check: plastic-strain increment settled AND
        #     exact KKT satisfied to tol_phi (stress units). Using dep_norm
        #     (not the raw multiplier change) filters out the Taylor-ambiguous
        #     slip drift. See ipm_mb.md.
        # ----------------------------------------------------------------------
        if dep_norm < tol_dep and r_kkt < tol_phi:
            converged = True
            break

        # ----------------------------------------------------------------------
        # 3e. mu update.
        #   (i) Complementarity polish. If the stress has settled (dep < tol_dep)
        #       but r_kkt is still above tol_phi, the active set is fixed and
        #       the residual yield is |Phi| = |s| ~ mu * (multiplier drift), so
        #       the active branch of r_kkt scales ~linearly with mu. Shrinking
        #       mu therefore drives it down. This is safe to do *below* mu_lo:
        #       with no active-set transition in progress every active slack
        #       sits at s ~ +mu (and the inactive ones at s >> 0), so none is
        #       near the shifted boundary s = -mu and the blow-up that forced
        #       the earlier mu-raise cannot recur. One decade of reduction is
        #       usually enough.
        #   (ii) Otherwise, APV stall-tightening: tighten mu when the merit (the
        #       plastic-strain-increment change) stalls -- smaller mu => faster
        #       outer rate C = mu/(mu + a*lam*) -- bounded below by mu_lo.
        # ----------------------------------------------------------------------
        if dep_norm < tol_dep and mu > mu_floor:
            mu     = max(mu * mu_dec, mu_floor)   # polish: drive |lam*s| down
            slacks = jnp.maximum(slacks, -(1.0 - tau_min) * mu)
        elif merit > theta * merit_prev and mu > mu_lo:
            mu     = max(mu * mu_dec, mu_lo)
            # mu shrank -> the band s > -mu shrank; project slacks back inside.
            slacks = jnp.maximum(slacks, -(1.0 - tau_min) * mu)
        merit_prev = merit

        # Recompute residual with updated (lam_k, mu, slacks)
        R_g = Phi + slacks
        R_s = dlambda * (slacks + mu) - mu * lam_k
        r   = scaled_residual(R_g, R_s, lam_k, dlambda)

    # ==========================================================================
    # 4. Post-processing
    # ==========================================================================
    if not converged:
        if verbose:
            print(f"WARNING: MB-IPM did not converge in {k_max} outer iterations.")
        hist["trace"] = trace
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
    new_hist["lam_k"]   = np.array(lam_k)

    C_ep = compute_tangent_IFT_mb(dlambda, lam_k, eps_j, eps_p_n, gamma_n, C, Za,
                                   tau0, tau_inf, xi, mu)
    new_hist["C_ep"] = np.array(C_ep)
    new_hist["trace"] = trace

    return np.array(sig), new_hist, total_newton_iter
