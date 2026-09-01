"""
Visco-plastic return mapping for rate-dependent single crystal plasticity
at small strains (24 slip systems, Perzyna-type power-law flow rule).

Uses a standard Newton-Raphson iteration on the residual

    r^alpha = dgamma^alpha / (dt/eta)
              - <(tau^alpha - g^alpha) / tau_ref>^p

where tau^alpha = Z^alpha : sigma(dgamma) is the resolved shear stress,
g^alpha = tau0 is the current slip resistance, tau_ref = tau0 is the
reference stress, eta the viscosity, and p the rate exponent.
<...> denotes the Macaulay bracket (positive part).

Derivatives of the residual are computed via JAX automatic differentiation.

References:
  vp_derive.md — Perzyna-type generalization via power-law penalty
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cp_matroutines.cp_base import Material


# ---------------------------------------------------------------------------
# Residual (JAX-compatible, JIT'd at module level)
# ---------------------------------------------------------------------------

@jax.jit
def _residual(dgamma, eps, eps_p_n, C, Za, tau0, p, dgamma0):
    r"""
    Perzyna-type visco-plastic residual for all slip systems.

    Parameters
    ----------
    dgamma   : (m,) incremental plastic slips
    eps      : (3,3) total strain
    eps_p_n  : (3,3) plastic strain from previous step
    C        : (3,3,3,3) elasticity tensor
    Za       : (m,3,3) Schmid tensors
    tau0     : float — slip resistance (= tau_ref here)
    p        : float — rate exponent
    dgamma0  : float — dt / eta (viscosity-scaled time step)

    Returns
    -------
    r : (m,) residual vector
    """
    eps_p = eps_p_n + jnp.einsum('a,aij->ij', dgamma, Za)
    sig = jnp.einsum('ijkl,kl->ij', C, eps - eps_p)
    tau = jnp.einsum('aij,ij->a', Za, sig)
    overstress = (tau - tau0) / tau0      # Phi^alpha / tau_ref
    return dgamma / dgamma0 - jnp.maximum(overstress, 0.0) ** p

_residual_jacobian = jax.jit(jax.jacfwd(_residual, argnums=0))


# ---------------------------------------------------------------------------
# Main material routine
# ---------------------------------------------------------------------------

def compute_stress(eps: np.ndarray, hist: dict, mat: Material,
                   config: dict,
                   ) -> tuple[np.ndarray, dict, int]:
    r"""
    Compute the Cauchy stress via visco-plastic return mapping for
    rate-dependent single crystal plasticity at small strains.

    Parameters
    ----------
    eps    : (3,3) array — prescribed total strain tensor
    hist   : dict — material history from the previous load step
    mat    : Material — material parameters (from cp_base)
    config : dict — solver parameters (loaded from TOML [VP] section)

    Returns
    -------
    sig      : (3,3) array — Cauchy stress tensor
    new_hist : dict — updated material history
    n_iter   : int — Newton iterations (−1 on failure)
    """

    # -- Unpack config ------------------------------------------------------
    max_iter = config["max_iter"]
    tol      = config["tol"]
    dt       = config["dt"]
    verbose  = config["verbose"]
    # Opt-in per-iteration tracer (single-step diagnostics only; see
    # examples/ex_single_step.py). None => no recording, zero overhead.
    trace = [] if config.get("trace", False) else None

    nSlip    = len(mat.Za)
    eta      = config["eta"]
    dgamma0  = dt / eta                # dt / eta

    # -- Convert to JAX arrays ----------------------------------------------
    Za      = jnp.array(np.stack(mat.Za))
    C       = jnp.array(mat.C)
    eps_j   = jnp.array(eps)
    eps_p_n = jnp.array(hist["eps_p"])
    tau0    = mat.tau0
    p       = config["p"]

    # Closures for module-level JIT'd functions
    def R_fn(dg):
        return _residual(dg, eps_j, eps_p_n, C, Za, tau0, p, dgamma0)

    def dR_fn(dg):
        return _residual_jacobian(dg, eps_j, eps_p_n, C, Za, tau0, p, dgamma0)

    # ======================================================================
    # 1.  Elastic predictor — yield-function check (independent of p / tol)
    # ======================================================================
    sig_trial = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_n)
    tau_trial = jnp.einsum('aij,ij->a', Za, sig_trial)

    if float(jnp.max(tau_trial - tau0)) <= 0.0:
        if verbose:
            print("  VP elastic predictor successful.")
        hist_el = hist.copy()
        hist_el["yield"] = np.array(tau_trial - tau0)
        hist_el["trace"] = trace
        hist_el["cond_final"] = np.nan      # no Newton system solved
        return np.array(sig_trial), hist_el, 0

    # ======================================================================
    # 2.  Newton iteration with backtracking (dgamma >= 0 enforced)
    # ======================================================================
    # Newton seed: scalar (uniform) or a length-m sequence. A non-uniform
    # seed tests whether the regularized problem has a unique solution.
    _dg0 = config.get("dgamma_init", None)
    dgamma = (jnp.zeros(nSlip) if _dg0 is None else
              jnp.array(np.broadcast_to(np.asarray(_dg0, dtype=float),
                                        (nSlip,)).copy()))
    n_iter = 0

    for n in range(1, max_iter + 1):
        # Compute residual and check convergence
        r = R_fn(dgamma)
        r_norm = float(jnp.linalg.norm(r))

        if r_norm < tol:
            n_iter = n
            if verbose:
                print(f"  VP Newton converged in {n} iterations, ||r|| = {r_norm:.2e}")
            break

        # Newton update direction
        dR = dR_fn(dgamma)
        delta = jnp.linalg.solve(dR, -r)

        # Backtracking line search (clamp dgamma >= 0)
        alpha = 1.0
        for _ in range(20):
            dgamma_trial = jnp.maximum(dgamma + alpha * delta, 0.0)
            r_trial_norm = float(jnp.linalg.norm(R_fn(dgamma_trial)))
            if r_trial_norm < r_norm:
                break
            alpha *= 0.5
        dgamma = dgamma_trial

        # Per-iteration diagnostics (single-step tracer only). VP is a penalty
        # (viscoplastic) regularization: no primal-dual or outer multiplier
        # loop, so the solved matrix is the m x m Newton matrix dR and
        # cond_solved = cond_kkt = cond(dR). The "continuation" knob is the
        # viscosity eta (fixed here, not iterated). compl_gap = max_a |dgamma_a *
        # Phi_a| with Phi = tau - tau0 measures the residual KKT violation, which
        # stays finite for eta > 0 (the regularization overstress).
        if trace is not None:
            eps_p_cur = eps_p_n + jnp.einsum('a,aij->ij', dgamma, Za)
            tau_cur   = jnp.einsum('aij,ij->a',
                                   Za, jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_cur))
            Phi_cur   = tau_cur - tau0
            cond_dR   = float(jnp.linalg.cond(dR))
            trace.append({
                "iter": len(trace) + 1, "k": 0, "n": n,
                "eta": eta, "r_abs": r_trial_norm, "r_rel": r_trial_norm,
                "compl_gap": float(jnp.max(jnp.abs(dgamma * Phi_cur))),
                "cond_solved": cond_dR, "cond_kkt": cond_dR,
                "alpha": float(alpha),
                "n_active": int(np.sum(np.array(dgamma) > 1e-10)),
            })
    else:
        if verbose:
            print(f"  VP Newton did not converge after {max_iter} iterations, ||r|| = {r_norm:.2e}")
        hist["trace"] = trace
        hist["cond_final"] = np.nan
        return np.zeros((3, 3)), hist, -1

    # ======================================================================
    # 3.  Post-processing
    # ======================================================================
    eps_p_new = eps_p_n + jnp.einsum('a,aij->ij', dgamma, Za)
    sig = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_new)

    new_hist = hist.copy()
    new_hist["eps"]     = np.array(eps)
    new_hist["eps_p"]   = np.array(eps_p_new)
    new_hist["gamma_a"] = hist["gamma_a"] + np.array(dgamma)
    new_hist["tau_h"]   = hist["tau_h"]          # no hardening update

    tau = jnp.einsum('aij,ij->a', Za, sig)
    new_hist["yield"] = np.array(tau - tau0)
    new_hist["trace"] = trace

    # Conditioning of the final Newton Jacobian dR/ddgamma 
    dR_final = dR_fn(dgamma)
    new_hist["cond_final"] = float(jnp.linalg.cond(dR_final))

    # Final Newton system exported
    if trace is not None:
        dPhi_final = -jnp.einsum('aij,ijkl,bkl->ab', Za, C, Za)
        new_hist["newton_final"] = {
            "solved":  np.array(dR_final),
            "kkt":     np.array(dR_final),
            "dPhi":    np.array(dPhi_final),
            "dlambda": np.array(dgamma),
            "Phi":     np.array(tau - tau0),
            "eta":     float(eta),
            "dgamma0": float(dgamma0),
        }

    return np.array(sig), new_hist, n_iter
