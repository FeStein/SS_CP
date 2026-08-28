"""
Parameter sweep for the simple shear test.

Reads [Sweep] from config.toml, overrides one parameter per run, and saves
results in separate output directories.

Example [Sweep] block in config.toml:

    [Sweep]
    param  = "IPM.mu_end"
    values = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    label  = "mu"
    # -> directories: out_mu4, out_mu6, out_mu8, out_mu10, out_mu12

    [Sweep]
    param  = "Solver.n_steps"
    values = [100, 500, 1000, 5000]
    label  = "steps"
    # -> directories: out_steps100, out_steps500, ...

    [Sweep]
    param  = "VP.eta"
    values = [1e-3, 3e-3, 1e-2]
    label  = "eta"
    # -> directories: out_eta3, out_eta00, out_eta2  (exp for pow10, else index)
"""

import copy
import math
import os
import shutil
import tomllib
import numpy as np

from cp_matroutines.cp_base import Material, ten2voigt, gamma24_to_12
from cp_matroutines.solver import get_solver

# --- Load config --------------------------------------------------------------
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
with open(config_path, "rb") as f:
    base_config = tomllib.load(f)

sweep = base_config.get("Sweep")
if sweep is None:
    raise RuntimeError("No [Sweep] section found in config.toml")

param_path = sweep["param"]
values     = sweep["values"]
label      = sweep["label"]

section, key = param_path.split(".", 1)


def _folder_tag(value, idx: int) -> str:
    """Short tag appended to the label to form the output folder name."""
    if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer()):
        return str(int(value))
    if value == 0.0:
        return "0"
    log = math.log10(abs(value))
    if abs(log - round(log)) < 1e-9:
        exp = int(round(log))
        return str(abs(exp))   # 1e-4 -> "4", 1e6 -> "6"
    # Not a clean power of 10: fall back to zero-padded index
    return f"{idx:02d}"


# --- Loading: symmetric simple shear ------------------------------------------
eps_max = np.array([
    [0.0, 0.005, 0.0],
    [0.005, 0.0,  0.0],
    [0.0,  0.0,   0.0],
])

# --- Sweep --------------------------------------------------------------------
for idx, value in enumerate(values):
    config = copy.deepcopy(base_config)
    config[section][key] = value

    tag     = _folder_tag(value, idx)
    out_dir = os.path.join(os.path.dirname(__file__), f"out_{label}{tag}")

    print(f"\n{'='*60}")
    print(f"Sweep: {param_path} = {value}  →  {os.path.basename(out_dir)}/")
    print(f"{'='*60}")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    shutil.copy2(config_path, os.path.join(out_dir, "config.toml"))

    compute_stress, solver_cfg = get_solver(config)

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

    n_steps = config["Solver"]["n_steps"]

    for i, scale in enumerate(np.linspace(0, 1, n_steps + 1)):
        eps = scale * eps_max
        print(f"  Step {i}/{n_steps}: scale = {scale:.4f}", end="\r")

        sig, new_hist, n_iter = compute_stress(eps, hist, mat, solver_cfg)

        if n_iter < 0:
            print(f"\n  Aborting at step {i} due to non-convergence.")
            break

        hist  = new_hist
        shear = 2 * 0.005 * scale

        gamma12 = gamma24_to_12(hist["gamma_a"])
        with open(os.path.join(out_dir, "gamma.csv"), "a") as f:
            f.write(f"{shear}," + ",".join(map(str, gamma12)) + "\n")

        vsig = ten2voigt(sig)
        with open(os.path.join(out_dir, "sigma.csv"), "a") as f:
            f.write(f"{shear}," + ",".join(map(str, vsig)) + "\n")

        cond = hist.get("cond_final", float("nan"))
        with open(os.path.join(out_dir, "newton.csv"), "a") as f:
            f.write(f"{shear},{n_iter},{cond}\n")

        with open(os.path.join(out_dir, "yield.csv"), "a") as f:
            f.write(f"{shear}," + ",".join(map(str, hist["yield"])) + "\n")

    print(f"\n  Done → {out_dir}")

print("\nSweep completed.")
