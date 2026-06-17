"""
Augmented Lagrangian method (Schmidt-Baldassari 2003), iterated to numerical
precision -- "converged" variant of mat_AL_SB.

Same semi-smooth Newton AL algorithm as mat_AL_SB (residual, active set,
piecewise Jacobian, eta-continuation, IFT tangent). The ONLY difference is the
stopping rule:

  - mat_AL_SB stops at the *first* iterate below an absolute tolerance
    (inner: max_a a_a*|ddgamma_a| < theta_in*tol_phi; outer: r_kkt < tol_phi).
    Because each step's Newton starts from a load-increment-proportional
    residual and a fast iteration overshoots the gate by a dt-dependent amount,
    the terminal stress error inherits a weak, *staircased* load-increment
    dependence (it sits below tol_phi but is not flat). See the discussion in
    ipm_mb.md / doc/al.md.

  - This variant instead iterates until the residual *stagnates* at the
    round-off floor -- both the inner Newton step and the outer KKT residual
    are driven down until they stop decreasing. The terminal accuracy is then
    the numerical floor, independent of the load-increment size, so the
    load-increment-sensitivity plot shows a flat line rather than a staircase.

Use this when you want AL plotted as a method whose accuracy is set by the
solver (driven to machine), to contrast against the regularization-limited
classical IPM -- without the stopping-rule artifact distracting from that
message. For the matched-absolute-tolerance comparison, use mat_AL_SB instead.

Reads the same [AL] config section; the stagnation parameters below are
optional (sensible defaults) so no config change is required. tol_phi is not
used for termination here.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cp_matroutines.cp_base import Material
from cp_matroutines.mat_IPM_classic import yield_function, yield_jacobian
from cp_matroutines.mat_AL_SB import compute_tangent_IFT_al


# ---------------------------------------------------------------------------
# Main material routine
# ---------------------------------------------------------------------------

def compute_stress(eps: np.ndarray, hist: dict, mat: Material,
                   config: dict) -> tuple[np.ndarray, dict, int]:
    r"""
    Compute the Cauchy stress via the Augmented Lagrangian method, iterated to
    numerical precision (stagnation stop). Signature and history contract are
    identical to mat_AL_SB.compute_stress.
    """

    # -- Unpack config -------------------------------------------------------
    eta_init  = config["eta_init"]
    max_outer = config["max_outer"]
    max_inner = config["max_inner"]
    verbose   = config["verbose"]

    # Stagnation parameters (optional; defaults solve to round-off). A step is
    # "progress" only if it drops below stall_rtol times the best value seen;
    # stall_patience consecutive non-progress steps => we have hit the floor.
    # The absolute *_floor values are fast-exit markers for "numerically zero".
    inner_stall_patience = int(config.get("inner_stall_patience", 3))
    inner_stall_rtol     = float(config.get("inner_stall_rtol", 0.5))
    step_floor           = float(config.get("step_floor", 1e-14))
    outer_stall_patience = int(config.get("outer_stall_patience", 2))
    outer_stall_rtol     = float(config.get("outer_stall_rtol", 0.5))
    r_floor              = float(config.get("r_floor", 1e-14))

    # Opt-in per-iteration tracer (single-step diagnostics only). None => off.
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
        return np.array(sig_trial), hist_el, 0

    # ========================================================================
    # 2.  Initialisation
    # ========================================================================
    dlambda = jnp.zeros(nSlip)
    eta = float(eta_init)
    total_newton_iter = 0
    converged = False
    r_kkt_best = float("inf")   # best (lowest) outer KKT residual seen
    outer_stall = 0             # consecutive outer steps without sufficient decrease

    # ========================================================================
    # 3.  Outer penalty loop
    # ========================================================================
    for outer in range(max_outer):
        inner_converged = False

        # Reference point fixed for the duration of this inner loop.
        dlambda_i = dlambda

        # --------------------------------------------------------------------
        # Semi-smooth Newton loop for current eta -- iterate to stagnation
        # (round-off floor), NOT to an absolute tolerance.
        # --------------------------------------------------------------------
        step_best   = float("inf")
        inner_stall = 0
        for n_inner in range(max_inner):
            Phi  = Phi_fn(dlambda)
            dPhi = dPhi_fn(dlambda)

            active = (dlambda_i + eta * Phi) > 0.0
            FF = jnp.where(active[:, None], jnp.eye(nSlip) - eta * dPhi, jnp.eye(nSlip))
            R  = dlambda - jnp.maximum(0.0, dlambda_i + eta * Phi)

            ddlambda = jnp.linalg.solve(FF, -R)
            dlambda  = dlambda + ddlambda
            total_newton_iter += 1

            # Newton step in stress units: a_diag = -diag(dPhi) > 0.
            a_diag = -jnp.diag(dPhi)
            step   = float(jnp.max(a_diag * jnp.abs(ddlambda)))

            if trace is not None:
                Phi_new = Phi_fn(dlambda)
                cond_FF = float(jnp.linalg.cond(FF))
                trace.append({
                    "iter": total_newton_iter, "k": outer, "n": n_inner + 1,
                    "eta": eta, "r_abs": step, "r_rel": step,
                    "compl_gap": float(jnp.max(jnp.abs(dlambda * Phi_new))),
                    "cond_solved": cond_FF, "cond_kkt": cond_FF,
                    "alpha": 1.0,
                    "n_active": int(np.sum(np.array(dlambda) > 1e-10)),
                })

            # Numerical-precision stop: keep iterating while the Newton step
            # still shrinks; stop once it underflows the floor or stagnates
            # at round-off (no sufficient decrease for inner_stall_patience
            # steps). This is what makes the terminal error load-increment
            # independent. np.maximum-style finite guard: a NaN step is not
            # < floor and not < rtol*best, so it trips the stall counter.
            if (not np.isfinite(step)) or step <= step_floor:
                inner_converged = True
                break
            if step >= inner_stall_rtol * step_best:
                inner_stall += 1
            else:
                inner_stall = 0
            step_best = min(step_best, step)
            if inner_stall >= inner_stall_patience:
                inner_converged = True
                break

        if not inner_converged and verbose:
            print(f"  AL(conv) outer {outer + 1}: inner Newton hit max_inner "
                  f"({max_inner}) without stagnating.")

        # Natural (min-form) KKT residual in stress units (see mat_AL_SB).
        Phi_cur = Phi_fn(dlambda)
        a_cur   = -jnp.diag(dPhi_fn(dlambda))
        r_kkt   = float(jnp.max(jnp.abs(jnp.minimum(a_cur * dlambda, -Phi_cur))))

        if verbose:
            print(f"  AL(conv) outer {outer + 1}: eta = {eta:.2e}, "
                  f"kkt = {r_kkt:.2e}, Newton iters = {n_inner + 1}")

        # --------------------------------------------------------------------
        # Outer stop: terminate when r_kkt can no longer be reduced (round-off
        # floor reached), detected by an absolute floor or by stagnation of
        # r_kkt across eta updates -- NOT at the first crossing of an absolute
        # tolerance. This removes the load-increment staircase.
        # --------------------------------------------------------------------
        if not np.isfinite(r_kkt):
            break  # falls through to the non-converged branch below
        if r_kkt <= r_floor:
            converged = True
            break
        if r_kkt >= outer_stall_rtol * r_kkt_best:
            outer_stall += 1
        else:
            outer_stall = 0
        r_kkt_best = min(r_kkt_best, r_kkt)
        if outer_stall >= outer_stall_patience:
            converged = True
            break

        eta *= 2.0

    # ========================================================================
    # 4.  Post-processing
    # ========================================================================
    if not converged:
        if verbose:
            print("WARNING: AL(conv) did not converge.")
        hist["trace"] = trace
        return np.zeros((3, 3)), hist, -1

    eps_p_new = eps_p_n + jnp.einsum('a,aij->ij', dlambda, Za)
    sig       = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_new)

    if verbose:
        print(f"  AL(conv) converged in {total_newton_iter} total Newton "
              f"iterations (r_kkt = {r_kkt:.2e}).")

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
    new_hist["trace"] = trace

    return np.array(sig), new_hist, total_newton_iter
