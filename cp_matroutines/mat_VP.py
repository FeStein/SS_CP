"""
Visco-plastic return mapping for rate-dependent single crystal plasticity
at small strains (24 slip systems, power-law flow rule).

Uses a standard Newton-Raphson iteration on the residual

    r^alpha = dgamma^alpha / (gamma0 * dt)
              - |tau^alpha / g^alpha|^(p-1) * (tau^alpha / g^alpha)

where tau^alpha = Z^alpha : sigma(dgamma) is the resolved shear stress,
g^alpha is the current hardening stress, and p is the rate exponent.

Derivatives of the residual are computed via JAX automatic differentiation.

References:
  Scheunemann et al. (2017), Tab. 8.1 / Eq. 4.36-4.39
  Peirce et al. (1982) — hardening law
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
    Visco-plastic residual for all 24 slip systems.

    Parameters
    ----------
    dgamma   : (m,) incremental plastic slips
    eps      : (3,3) total strain
    eps_p_n  : (3,3) plastic strain from previous step
    C        : (3,3,3,3) elasticity tensor
    Za       : (m,3,3) Schmid tensors
    tau0     : float — current (constant) hardening stress
    p        : float — rate exponent
    dgamma0  : float — reference slip rate * dt

    Returns
    -------
    r : (m,) residual vector
    """
    eps_p = eps_p_n + jnp.einsum('a,aij->ij', dgamma, Za)
    sig = jnp.einsum('ijkl,kl->ij', C, eps - eps_p)
    tau = jnp.einsum('aij,ij->a', Za, sig)
    ratio = tau / tau0
    return dgamma / dgamma0 - jnp.maximum(ratio, 0.0) ** p

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

    nSlip    = len(mat.Za)
    dgamma0  = config["gamma0"] * dt

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
    # 1.  Elastic predictor
    # ======================================================================
    r0 = R_fn(jnp.zeros(nSlip))
    if float(jnp.linalg.norm(r0)) < tol:
        sig_trial = jnp.einsum('ijkl,kl->ij', C, eps_j - eps_p_n)
        if verbose:
            print("  VP elastic predictor successful.")
        return np.array(sig_trial), hist, 0

    # ======================================================================
    # 2.  Newton iteration with backtracking (dgamma >= 0 enforced)
    # ======================================================================
    dgamma = jnp.zeros(nSlip)
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
    else:
        if verbose:
            print(f"  VP Newton did not converge after {max_iter} iterations, ||r|| = {r_norm:.2e}")
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

    yield_function = np.max(jnp.einsum('aij,ij->a', Za, sig) - tau0)
    print(f"  VP yield function = {yield_function}")

    return np.array(sig), new_hist, n_iter
