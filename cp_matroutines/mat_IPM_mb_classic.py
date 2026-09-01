"""
Modified-Barrier Interior Point Method -- BASE method (Polyak 1992) -- for
rate-independent single crystal plasticity at small strains.

Small-strain Python counterpart of the finite-strain Julia routine
`mb_ipm_classic.jl` (CrystalPlasticity.jl, MBIPMClassic). It is the plain,
textbook version of the nonlinear-rescaling scheme, kept deliberately free of
the convergence-acceleration machinery in `mat_IPM_mb.py` so that the two can
be compared:

  - Modified barrier: -mu * lam_k * ln(1 + s/mu)
  - Shifted complementarity: dlambda * (s + mu) = mu * lam_k
  - Explicit Lagrange-multiplier updates (Polyak rule)
  - mu is FIXED at mu_init for the whole solve (no APV self-tuning, no
    stall-tightening, no inner-failure mu-raise, no cold restarts)
  - fixed inner tolerance tol_in = theta_in * tol_phi (no mu-dependent
    relaxation of early sweeps), no inner stall detection
  - flat cold start lambda = lam_k = lambda_init (no central-path seeding,
    no warm start across load steps)
  - outer convergence measured by the natural (min-form) KKT residual alone,
        r_kkt = max_a |min(a_a * dgamma_a, -Phi_a)| < tol_phi,
    in stress units (no plastic-strain-increment / Taylor-invariance test)
  - inner failure is terminal: there is no safeguard in the base method

Deviations from `mb_ipm_classic.jl` are only those forced by the framework:
the small-strain kinematics and yield function, the JAX (instead of
ForwardDiff) derivatives, the Python return convention (failures return
n_iter = -1 rather than raising), and the opt-in per-iteration tracer plus
consistent tangent that the Python drivers expect.

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

    Used only for the consistent tangent computation via IFT; lam_k is frozen
    at the estimate the converged inner solve was run with.
    """
    Phi = yield_function(dgamma, eps, eps_p_n, gamma_n, C, Za, tau0, tau_inf, xi)
    return dgamma * (mu - Phi) - mu * lam_k

dR_ddgamma_mb = jax.jit(jax.jacfwd(residual_mb, argnums=0))  # (m, m)
dR_deps_mb    = jax.jit(jax.jacfwd(residual_mb, argnums=1))  # (m, 3, 3)


def compute_tangent_IFT_mb(dlambda, lam_k, eps_j, eps_p_n, gamma_n, C, Za,
                           tau0, tau_inf, xi, mu):
    r"""
    Consistent elastoplastic tangent via the implicit function theorem applied
    to the reduced MB-IPM residual R_mb(dgamma, eps) = 0, with (mu, lam_k)
    frozen at the values of the converged inner solve.

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


def mb_inner_residual(R_g, R_s, slacks, mu, a_diag):
    r"""
    Inner residual in stress units (port of `mb_inner_residual` in
    `mb_ipm_common.jl`).

    R_g = Phi + s is already a stress. R_s = dgamma*(s+mu) - mu*lam_k carries
    stress*slip; the second Newton row gives d_dgamma ~ -R_s/(s+mu), so a
    leftover R_s corresponds to the slip error R_s/(s+mu), which the diagonal
    a_diag = -diag(dPhi/ddgamma) > 0 converts into the stress error it causes.
    Both parts are then measured with one stress-unit tolerance, uniformly in
    the load-increment size.
    """
    r_comp = jnp.max(jnp.abs(a_diag * R_s / (slacks + mu)))
    return float(jnp.maximum(jnp.max(jnp.abs(R_g)), r_comp))


# ---------------------------------------------------------------------------
# Main material routine
# ---------------------------------------------------------------------------

def compute_stress(eps: np.ndarray, hist: dict, mat: Material,
                   config: dict) -> tuple[np.ndarray, dict, int]:
    r"""
    Compute the Cauchy stress via the base Modified-Barrier IPM (Polyak 1992)
    for rate-independent single crystal plasticity at small strains.

    Parameters
    ----------
    eps    : (3,3) array — prescribed total strain tensor
    hist   : dict — material history from the previous load step
    mat    : Material — material parameters (from cp_base)
    config : dict — solver parameters from the [MB_IPM_classic] config section

    Returns
    -------
    sig      : (3,3) array — Cauchy stress tensor
    new_hist : dict — updated material history
    n_iter   : int — total Newton iterations (-1 on failure)
    """

    # -- Unpack config ----------------------------------------------------------
    k_max      = int(config["k_max"])
    max_inner  = int(config["max_inner"])
    mu_init    = float(config["mu_init"])
    # Cold-start seed. Scalar (uniform, the base method) or a length-m
    # sequence: a non-uniform seed breaks the symmetry of the iteration and
    # is what selects among the solutions of a Taylor-ambiguous problem.
    lam_init   = np.broadcast_to(np.asarray(config["lambda_init"], dtype=float),
                                 (len(mat.Za),)).copy()
    dgam_init  = np.broadcast_to(np.asarray(config.get("dgamma_init",
                                            config["lambda_init"]), dtype=float),
                                 (len(mat.Za),)).copy()
    tol_phi    = float(config["tol_phi"])        # outer KKT tolerance in stress units
    theta_in   = float(config["theta_in"])       # inner tolerance = theta_in * tol_phi
    tau_min    = float(config["tau_min"])        # fraction-to-boundary factor
    dlam_bound = float(config.get("dlam_bound", 1e-12))  # slips below this count as at the bound
    verbose    = bool(config["verbose"])
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

    # ==========================================================================
    # 1. Elastic predictor
    # ==========================================================================
    Phi_trial = Phi_fn(jnp.zeros(nSlip))
    if np.all(np.array(Phi_trial) <= 0.0):
        sig_trial = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_n)
        if verbose:
            print("  MB-IPM(base): elastic regime, no yielding.")
        hist_el = hist.copy()
        hist_el["C_ep"]  = np.array(C)
        hist_el["yield"] = np.array(Phi_trial)
        hist_el["trace"] = trace
        hist_el["cond_final"]        = np.nan   # no Newton system solved
        hist_el["cond_final_solved"] = np.nan
        return np.array(sig_trial), hist_el, 0

    # ==========================================================================
    # 2. Initialization — plastic computation
    # ==========================================================================
    mu     = mu_init            # FIXED for the whole solve (base method)
    tol_in = theta_in * tol_phi # FIXED inner tolerance
    lam_k   = jnp.array(lam_init)          # multiplier estimate
    dlambda = jnp.array(dgam_init)         # primal slip increment
    Phi     = Phi_fn(dlambda)
    slacks  = jnp.maximum(-Phi, 0.0)

    total_newton_iter = 0
    converged = False
    r_kkt     = float("inf")
    k_final   = 0
    dPhi      = dPhi_fn(dlambda)   # defined even if the inner loop breaks at n = 0

    # ==========================================================================
    # 3. Outer Polyak multiplier loop (index k)
    # ==========================================================================
    for k in range(1, k_max + 1):
        k_final = k

        # ----------------------------------------------------------------------
        # 3a. Inner Newton solve at fixed (mu, lam_k)
        # ----------------------------------------------------------------------
        n = 0
        r = float("inf")
        while True:
            # current residual
            dPhi = dPhi_fn(dlambda)
            R_g  = Phi + slacks
            R_s  = dlambda * (slacks + mu) - mu * lam_k

            # residual in stress units: max(|R_g|, |R_s| scaled to stress units)
            a_diag = -jnp.diag(dPhi)
            r = mb_inner_residual(R_g, R_s, slacks, mu, a_diag)

            if verbose:
                print(f"    MB-IPM(base) k={k} inner n={n}: r={r:.2e} mu={mu:.2e}")

            # tolerance / max-iteration check
            if r <= tol_in or n >= max_inner:
                break

            # Schur complement: eliminate d_slacks from the 2m x 2m system.
            #   dPhi @ d_dlambda + I @ d_slacks                   = -R_g   ... (1)
            #   diag(s+mu) @ d_dlambda + diag(dlambda) @ d_slacks = -R_s   ... (2)
            # From (1): d_slacks = -R_g - dPhi @ d_dlambda; substituting into (2)
            #   M @ d_dlambda = -R_s + dlambda * R_g,  M = diag(s+mu) - diag(dlambda) @ dPhi
            M         = jnp.diag(slacks + mu) - jnp.diag(dlambda) @ dPhi
            d_dlambda = jnp.linalg.solve(M, -R_s + dlambda * R_g)
            d_slacks  = -R_g - dPhi @ d_dlambda

            # Fraction-to-boundary with jitter control: keep dlambda > 0 and
            # s + mu > 0 (the shifted boundary s = -mu is the barrier's
            # singularity, so the cap fires on every descending slack
            # direction). Slips already at the bound (dlambda <= dlam_bound)
            # are exempted from the dlambda cap — otherwise a system parked at
            # zero jams the step length for all the others.
            alpha_max = 1.0
            for a in range(nSlip):
                if d_dlambda[a] < 0 and dlambda[a] > dlam_bound:
                    alpha_max = min(alpha_max,
                                    float(-tau_min * dlambda[a] / d_dlambda[a]))
                if d_slacks[a] < 0:
                    alpha_max = min(alpha_max,
                                    float(-tau_min * (slacks[a] + mu) / d_slacks[a]))
            if verbose:
                print(f"      alpha_max: {alpha_max:.2e}")

            dlambda = jnp.maximum(dlambda + alpha_max * d_dlambda, 0.0)
            slacks  = slacks + alpha_max * d_slacks
            Phi     = Phi_fn(dlambda)

            n += 1
            total_newton_iter += 1

            # Per-iteration diagnostics (single-step tracer only). cond_solved is
            # the Schur complement M that is actually solved; cond_kkt is the full
            # 2m x 2m primal-dual matrix, assembled here so it is directly
            # comparable to the classical IPM trace.
            if trace is not None:
                J_kkt = jnp.block([[dPhi, jnp.eye(nSlip)],
                                   [jnp.diag(slacks + mu), jnp.diag(dlambda)]])
                trace.append({
                    "iter": total_newton_iter, "k": k, "n": n,
                    "mu": mu, "r_abs": r, "r_rel": r,
                    "compl_gap": float(jnp.max(jnp.abs(dlambda * slacks))),
                    "cond_solved": float(jnp.linalg.cond(M)),
                    "cond_kkt": float(jnp.linalg.cond(J_kkt)),
                    "alpha": float(alpha_max),
                    "n_active": int(np.sum(np.array(dlambda) > 1e-10)),
                })

        # ----------------------------------------------------------------------
        # 3b. Inner failure is terminal — the base method has no safeguard
        #     (no mu-raise, no cold restart). Note that a nan fails every '>'
        #     test, so it is caught explicitly.
        # ----------------------------------------------------------------------
        if (not np.isfinite(r)) or r > tol_in:
            if verbose:
                print(f"  MB-IPM(base): inner Newton failed at outer k={k}, "
                      f"mu={mu:.2e}: r={r:.2e} > tol_in={tol_in:.2e} "
                      f"(no safeguard in the base method)")
            hist["trace"] = trace
            hist["cond_final"]        = np.nan
            hist["cond_final_solved"] = np.nan
            return np.zeros((3, 3)), hist, -1

        # ----------------------------------------------------------------------
        # 3c. Polyak multiplier update: lam_{k+1} = mu * lam_k / (s + mu).
        #     lam_k_used is kept for the tangent: the reduced residual whose IFT
        #     gives C_ep is the one the converged inner solve satisfies, i.e. the
        #     pre-update estimate.
        # ----------------------------------------------------------------------
        lam_k_used = lam_k
        lam_k      = mu * lam_k / (slacks + mu)

        # ----------------------------------------------------------------------
        # 3d. KKT residual check: r_kkt = max |min(a_diag * dlambda, -Phi)|
        #     (stress units). Stationarity is already enforced by the inner
        #     Newton; a_diag * dlambda covers complementary slackness (spurious
        #     slip on inactive systems, converted to stress by the diagonal
        #     a_diag = -diag(dPhi/ddgamma) > 0), and -Phi the constraint
        #     violation. For scalars min(a, b) = 0 <=> a, b >= 0, a*b = 0, so
        #     r_kkt measures the distance to the exact KKT point — unlike the
        #     product gap |lam*s| < tol it bounds the stress error uniformly in
        #     the load-increment size.
        # ----------------------------------------------------------------------
        a_diag = -jnp.diag(dPhi)
        r_kkt  = float(jnp.max(jnp.abs(jnp.minimum(a_diag * dlambda, -Phi))))
        if verbose:
            print(f"  MB-IPM(base) k={k}: mu={mu:.2e}, kkt={r_kkt:.2e}, n_inner={n}")

        if r_kkt < tol_phi:
            converged = True
            break

    # ==========================================================================
    # 4. Post-processing
    # ==========================================================================
    if not converged:
        if verbose:
            print(f"WARNING: MB-IPM(base) did not converge in {k_max} outer "
                  f"iterations (r_kkt={r_kkt:.2e}).")
        hist["trace"] = trace
        hist["cond_final"]        = np.nan
        hist["cond_final_solved"] = np.nan
        return np.zeros((3, 3)), hist, -1

    eps_p_new = eps_p_n + jnp.einsum('a,aij->ij', dlambda, Za)
    sig       = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_new)

    if verbose:
        print(f"  MB-IPM(base) converged: {total_newton_iter} Newton iters, "
              f"{k_final} outer iters, mu={mu:.2e}")

    new_hist = hist.copy()
    new_hist["eps"]     = np.array(eps)
    new_hist["eps_p"]   = np.array(eps_p_new)
    new_hist["gamma_a"] = hist["gamma_a"] + np.array(dlambda)
    new_hist["tau_h"]   = hist["tau_h"]
    new_hist["n_iter"]  = total_newton_iter
    new_hist["yield"]   = np.array(Phi_fn(dlambda))
    new_hist["lam_k"]   = np.array(lam_k)

    C_ep = compute_tangent_IFT_mb(dlambda, lam_k_used, eps_j, eps_p_n, gamma_n,
                                  C, Za, tau0, tau_inf, xi, mu)
    new_hist["C_ep"]  = np.array(C_ep)
    new_hist["trace"] = trace

    # Conditioning of the final Newton matrices at the converged iterate
    dPhi_final = dPhi_fn(dlambda)
    M_final    = jnp.diag(slacks + mu) - jnp.diag(dlambda) @ dPhi_final
    J_final    = jnp.block([[dPhi_final, jnp.eye(nSlip)],
                            [jnp.diag(slacks + mu), jnp.diag(dlambda)]])
    new_hist["cond_final_solved"] = float(jnp.linalg.cond(M_final))
    new_hist["cond_final"]        = float(jnp.linalg.cond(J_final))

    # Final Newton system exported f
    if trace is not None:
        new_hist["newton_final"] = {
            "solved":  np.array(M_final),
            "kkt":     np.array(J_final),
            "dPhi":    np.array(dPhi_final),
            "dlambda": np.array(dlambda),
            "slacks":  np.array(slacks),
            "Phi":     np.array(Phi_fn(dlambda)),
            "lam_k":   np.array(lam_k_used),
            "mu":      float(mu),
        }

    return np.array(sig), new_hist, total_newton_iter
