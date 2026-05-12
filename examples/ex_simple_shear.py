"""
Simple shear test for crystal plasticity material routines.

Applies a symmetric simple shear strain eps_12 = eps_21 = 0.005
scaled linearly from 0 to 1 over 100 load steps (max shear deformation 0.01).
The solver (IPM or VP) is selected via config.toml [Solver] method.

Outputs gamma_a, sigma (Voigt), and Newton iterations per step into out/.
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
out_dir = os.path.join(os.path.dirname(__file__), "out")
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)
shutil.copy2(config_path, os.path.join(out_dir, "config.toml"))

# --- Loading: symmetric simple shear ---------------------------------------
eps_max = np.array([
    [0.0, 0.005, 0.0],
    [0.005, 0.0,  0.0],
    [0.0,  0.0,   0.0],
])

n_steps = config["Solver"]["n_steps"]

for i, scale in enumerate(np.linspace(0, 1, n_steps + 1)):
    eps = scale * eps_max
    print(f"Step {i}/{n_steps}: scale = {scale:.4f}")

    sig, new_hist, n_iter = compute_stress(eps, hist, mat, solver_cfg)

    if n_iter < 0:
        print("Aborting simulation due to non-convergence.")
        break

    hist = new_hist

    shear = 2 * 0.005 * scale

    #output gamma (consider symmetry)
    gamma12 = gamma24_to_12(hist["gamma_a"])
    with open(os.path.join(out_dir, "gamma.csv"), "a") as f:
        f.write(f"{shear}," + ",".join(map(str, gamma12)) + "\n")

    # output sigma (Voigt)
    vsig = ten2voigt(sig)
    with open(os.path.join(out_dir, "sigma.csv"), "a") as f:
        f.write(f"{shear}," + ",".join(map(str, vsig)) + "\n")

    # output total Newton iterations
    with open(os.path.join(out_dir, "newton.csv"), "a") as f:
        f.write(f"{shear},{n_iter}\n")

    # output yield function values
    with open(os.path.join(out_dir, "yield.csv"), "a") as f:
        f.write(f"{shear}," + ",".join(map(str, hist["yield"])) + "\n")

print("Simulation completed. Results saved in 'out/' directory.")
