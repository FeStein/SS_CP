"""
Uniaxial tension test for crystal plasticity material routines.

Applies a deformation-controlled uniaxial tension eps_11 = u(t)
scaled linearly from 0 to eps_max over n_steps load steps.

A global Newton iteration enforces the uniaxial stress state:
    sigma_22 = sigma_33 = sigma_12 = sigma_13 = sigma_23 = 0

The consistent tangent C_ep (computed via IFT inside the material
routine) provides the Jacobian for the global Newton.

Outputs gamma_a, sigma (Voigt), and iteration counts per step into out_uniaxial/.
"""

import os
import shutil
import tomllib
import numpy as np

from cp_matroutines.cp_base import Material, ten2voigt, gamma24_to_12
from cp_matroutines.solver import get_solver

# --- Load solver config from TOML -----------------------------------------
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)

compute_stress, solver_cfg = get_solver(config)
print(f"Solver: {config['Solver']['method']}")

# --- Material setup --------------------------------------------------------
mat_cfg = config["Material"]
mat = Material(
    E=mat_cfg["E"],
    nu=mat_cfg["nu"],
    tau0=mat_cfg["tau0"],
    q=mat_cfg["q"],
    xi=mat_cfg["xi"],
    tau_inf=mat_cfg["tau_inf"],
    phi=tuple(np.deg2rad(mat_cfg["phi_deg"])),
)
hist = mat.initialize_history()

# --- Output directory ------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "out_uniaxial")
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
shutil.copy2(config_path, os.path.join(out_dir, "config.toml"))

# --- Global Newton parameters ----------------------------------------------
MAX_GLOBAL_ITER = 20
GLOBAL_TOL = 1e-6

# Free DOF indices in the symmetric strain tensor:
# everything except (0,0) which is prescribed
FREE_IDX = [(1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

# --- Loading: uniaxial tension --------------------------------------------
eps_11_max = 0.01  # maximum axial strain
n_steps = config["Solver"]["n_steps"]

for i, scale in enumerate(np.linspace(0, 1, n_steps + 1)):
    eps_11 = scale * eps_11_max
    print(f"\nStep {i}/{n_steps}: eps_11 = {eps_11:.6f}")

    # Warm-start free strain DOFs from the last converged state
    eps_free = np.array([hist["eps"][ij] for ij in FREE_IDX])

    converged = False
    for g_iter in range(MAX_GLOBAL_ITER):
        # Build full symmetric strain tensor
        eps = np.zeros((3, 3))
        eps[0, 0] = eps_11
        for k, (ii, jj) in enumerate(FREE_IDX):
            eps[ii, jj] = eps_free[k]
            eps[jj, ii] = eps_free[k]

        # Solve material point (inner IPM)
        sig, trial_hist, n_inner = compute_stress(eps, hist, mat, solver_cfg)

        if n_inner < 0:
            print(f"  Global iter {g_iter}: inner solver failed!")
            break

        # Residual: off-axial stress components must vanish
        r = np.array([sig[ij] for ij in FREE_IDX])
        r_norm = np.linalg.norm(r)

        print(f"  Global iter {g_iter}: ||r|| = {r_norm:.2e}")

        if r_norm < GLOBAL_TOL:
            converged = True
            break

        # Consistent tangent from the material routine
        C_ep = trial_hist["C_ep"]

        # Build 5x5 Jacobian: K[a,b] = d(sigma_free[a]) / d(eps_free[b])
        # For off-diagonal strain DOFs (k!=l) the symmetric contribution
        # eps_{kl} = eps_{lk} doubles the tangent entry.
        K = np.zeros((5, 5))
        for a, (ia, ja) in enumerate(FREE_IDX):
            for b, (kb, lb) in enumerate(FREE_IDX):
                K[a, b] = C_ep[ia, ja, kb, lb]
                if kb != lb:
                    K[a, b] += C_ep[ia, ja, lb, kb]

        # Newton update
        deps_free = np.linalg.solve(K, -r)
        eps_free += deps_free

    if not converged:
        print(f"  Global Newton did not converge at step {i}. Aborting.")
        break

    # Accept converged state
    hist = trial_hist

    # --- Output ------------------------------------------------------------
    gamma12 = gamma24_to_12(hist["gamma_a"])
    with open(os.path.join(out_dir, "gamma.csv"), "a") as f:
        f.write(f"{eps_11}," + ",".join(map(str, gamma12)) + "\n")

    vsig = ten2voigt(sig)
    with open(os.path.join(out_dir, "sigma.csv"), "a") as f:
        f.write(f"{eps_11}," + ",".join(map(str, vsig)) + "\n")

    cond = hist.get("cond_final", float("nan"))
    with open(os.path.join(out_dir, "newton.csv"), "a") as f:
        f.write(f"{eps_11},{n_inner},{g_iter},{cond}\n")

print("\nSimulation completed. Results saved in 'out_uniaxial/' directory.")
