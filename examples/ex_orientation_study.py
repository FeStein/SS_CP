"""
Crystal-orientation parameter study for the simple shear test.

Reads rotation matrices from initial_rotations.csv (one flattened 3x3 per row),
runs the simple shear simulation for each orientation, and writes results to:

    par_study/
        config.toml          <- one copy of the run configuration
        1/
            gamma.csv
            sigma.csv
            newton.csv
            yield.csv
        2/ ...
        1000/ ...

Parallel execution is controlled via [ParStudy] n_workers in config.toml
(defaults to all CPU cores when absent).

Usage:
    conda run -n ipm-vp python examples/ex_orientation_study.py
"""

import copy
import concurrent.futures
import os
import shutil
import tomllib

import numpy as np

from cp_matroutines.cp_base import Material, ten2voigt, gamma24_to_12
from cp_matroutines.solver import get_solver

_EPS_MAX = np.array([
    [0.0, 0.005, 0.0],
    [0.005, 0.0,  0.0],
    [0.0,  0.0,   0.0],
])


def _run_orientation(args: tuple) -> tuple[int, bool]:
    """Worker: simulate one orientation and write CSV files.

    Returns (1-based index, converged_fully).
    """
    idx, rot_flat, config, out_base = args

    rot     = rot_flat.reshape(3, 3)
    out_dir = os.path.join(out_base, str(idx + 1))
    os.makedirs(out_dir, exist_ok=True)

    compute_stress, solver_cfg = get_solver(config)
    solver_cfg["verbose"] = False   # summary-only output; silence per-step solver prints

    mat_cfg = config["Material"]
    mat = Material(
        E=mat_cfg["E"],
        nu=mat_cfg["nu"],
        tau0=mat_cfg["tau0"],
        q=mat_cfg["q"],
        xi=mat_cfg["xi"],
        tau_inf=mat_cfg["tau_inf"],
        phi=(0.0, 0.0, 0.0),   # overridden by set_orientation below
    )
    mat.set_orientation(rot)
    hist = mat.initialize_history()

    n_steps = config["Solver"]["n_steps"]
    converged = True

    gamma_rows  = []
    sigma_rows  = []
    newton_rows = []
    yield_rows  = []

    for _, scale in enumerate(np.linspace(0, 1, n_steps + 1)):
        eps = scale * _EPS_MAX
        sig, new_hist, n_iter = compute_stress(eps, hist, mat, solver_cfg)

        if n_iter < 0:
            converged = False
            break

        hist  = new_hist
        shear = 2 * 0.005 * scale

        gamma12 = gamma24_to_12(hist["gamma_a"])
        gamma_rows.append(  f"{shear}," + ",".join(map(str, gamma12)))
        sigma_rows.append(  f"{shear}," + ",".join(map(str, ten2voigt(sig))))
        newton_rows.append( f"{shear},{n_iter}")
        yield_rows.append(  f"{shear}," + ",".join(map(str, hist["yield"])))

    for filename, rows in [
        ("gamma.csv",  gamma_rows),
        ("sigma.csv",  sigma_rows),
        ("newton.csv", newton_rows),
        ("yield.csv",  yield_rows),
    ]:
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write("\n".join(rows) + "\n")

    return idx + 1, converged   # 1-based index


if __name__ == "__main__":
    # --- Config ---------------------------------------------------------------
    config_path = os.path.join(os.path.dirname(__file__), "config.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    n_workers = config.get("ParStudy", {}).get("n_workers", os.cpu_count())

    # --- Output directory -----------------------------------------------------
    out_base = os.path.join(os.path.dirname(__file__), "par_study")
    if os.path.exists(out_base):
        shutil.rmtree(out_base)
    os.makedirs(out_base)
    shutil.copy2(config_path, os.path.join(out_base, "config.toml"))

    # --- Load rotation matrices -----------------------------------------------
    rot_csv = os.path.join(os.path.dirname(__file__), "initial_rotations.csv")
    rotations = np.loadtxt(rot_csv, delimiter=",")   # shape (N, 9)
    n_total   = len(rotations)

    print(f"Solver      : {config['Solver']['method']}")
    print(f"Orientations: {n_total}")
    print(f"Workers     : {n_workers}")
    print(f"Output      : {out_base}/")
    print()

    # --- Parallel run ---------------------------------------------------------
    work = [
        (i, rotations[i], copy.deepcopy(config), out_base)
        for i in range(n_total)
    ]

    n_done    = 0
    n_failed  = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_orientation, a): a[0] for a in work}

        for fut in concurrent.futures.as_completed(futures):
            try:
                done_idx, converged = fut.result()
            except Exception as exc:
                orig_idx = futures[fut]
                n_failed += 1
                print(f"\n  ERROR orientation {orig_idx + 1}: {exc}")
            else:
                n_done += 1
                if converged:
                    print(f"ori {done_idx} - successful")
                else:
                    print(f"ori {done_idx} - aborted")

    print(f"\n\nAll done. {n_done} succeeded, {n_failed} failed.")
    print(f"Results in {out_base}/")
