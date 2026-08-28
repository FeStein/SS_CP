"""
Single-step diagnostic driver for the IPM material routines.

Runs ONE plastic load step -- jumping straight to a strain state in the plastic
regime -- and records the local Newton iteration history for one or more solver
methods, so their convergence behaviour can be compared for the manuscript.

For each method the per-(inner-)Newton-iteration trace is written to
out_single/trace_<method>.csv, one row per iteration, with columns:

    iter        global Newton iteration counter (1-based)
    k           outer (barrier / multiplier / penalty) loop index
    n           inner Newton index within the current outer step
    mu / eta    continuation parameter: barrier/shift mu (IPM_classic, IPM_mb),
                penalty eta (AL), or viscosity eta (VP)
    r_abs       method's inner residual norm (||R|| for IPM, Newton step
                ||ddlambda|| for AL, ||r|| for VP)
    r_rel       routine's own (possibly normalized) residual
    compl_gap   complementary gap  max_a |dlambda_a * s_a|  (IPM) /
                max_a |dlambda_a * Phi_a|  (AL, VP)
    cond_solved condition number of the linear system actually solved
                (full 2m x 2m system for IPM_classic, Schur complement M for
                IPM_mb, semi-smooth Newton matrix FF for AL, Newton matrix dR
                for VP)
    cond_kkt    condition number of the full 2m x 2m primal-dual KKT matrix
                (assembled in IPM_classic and IPM_mb -> directly comparable;
                equals the solved m x m matrix for AL and VP, which have no
                primal-dual system)
    alpha       fraction-to-boundary step length (1.0 for AL: full Newton steps)
    n_active    number of slip systems with dlambda > 1e-10

A second file out_single/comparison.csv collects the converged stress (Voigt)
and accumulated slip per method, so the physical results can be checked to agree
across methods; pairwise max differences are also printed.

Tracing is enabled only here (solver_cfg["trace"] = True); ordinary multi-step
simulations run with no per-iteration recording and no overhead.

Usage:
    python ex_single_step.py
    python ex_single_step.py --methods IPM_classic IPM_mb AL
    python ex_single_step.py --scale 1.0
    python ex_single_step.py --orientation 42
"""

import argparse
import copy
import csv
import os
import tomllib
import numpy as np

from cp_matroutines.cp_base import Material, ten2voigt, gamma24_to_12
from cp_matroutines.solver import get_solver

# --- CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--methods", nargs="+", default=["IPM_classic", "IPM_mb", "AL", "VP"],
    help="solver method(s) to trace (default: IPM_classic IPM_mb AL VP). "
         "Note: VP is the rate-dependent penalty (viscoplastic) approach -- it "
         "only matches the rate-independent methods in the eta -> 0 limit.",
)
parser.add_argument(
    "--scale", type=float, default=1.0,
    help="fraction of the maximum shear strain to apply in the single step "
         "(default: 1.0; must be large enough to be plastic)",
)
parser.add_argument(
    "--orientation", type=int, default=None, metavar="N",
    help="1-based row index into initial_rotations.csv (overrides phi_deg in config)",
)
args = parser.parse_args()

# --- Load config --------------------------------------------------------------
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)

# --- Material setup -----------------------------------------------------------
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

if args.orientation is not None:
    rot_csv = os.path.join(os.path.dirname(__file__), "initial_rotations.csv")
    rotations = np.loadtxt(rot_csv, delimiter=",")
    n_total = len(rotations)
    if not (1 <= args.orientation <= n_total):
        raise ValueError(f"--orientation must be between 1 and {n_total}, got {args.orientation}")
    rot = rotations[args.orientation - 1].reshape(3, 3)
    mat.set_orientation(rot)
    print(f"Orientation: row {args.orientation} of initial_rotations.csv")
else:
    print(f"Orientation: Euler angles {mat_cfg['phi_deg']} deg")

# --- Single load step: jump straight to a plastic strain state ----------------
eps_max = np.array([
    [0.0,   0.005, 0.0],
    [0.005, 0.0,   0.0],
    [0.0,   0.0,   0.0],
])
eps = args.scale * eps_max

# --- Output directory ---------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "out_single")
os.makedirs(out_dir, exist_ok=True)

# --- Run each method once, recording the local Newton history -----------------
results = {}   # method -> dict(sig_voigt, gamma12, n_iter, converged)

for method in args.methods:
    cfg = copy.deepcopy(config)
    cfg["Solver"]["method"] = method
    cfg["Solver"]["n_steps"] = 1            # single step -> dt = 1

    compute_stress, solver_cfg = get_solver(cfg)
    solver_cfg["trace"] = True              # enable the per-iteration tracer
    solver_cfg["verbose"] = False           # keep stdout clean; CSV holds the detail

    hist0 = mat.initialize_history()
    sig, new_hist, n_iter = compute_stress(eps, hist0, mat, solver_cfg)

    trace = new_hist.get("trace") or []
    converged = n_iter >= 0
    status = "converged" if converged else "FAILED"

    print(f"\n[{method}] {status}: {n_iter} Newton iterations, "
          f"{len(trace)} traced rows")

    if converged:
        results[method] = {
            "sig": ten2voigt(sig),
            "gamma12": gamma24_to_12(new_hist["gamma_a"]),
            "n_iter": n_iter,
        }

    if not trace:
        print(f"  (no iterations recorded -- elastic predictor or empty trace; "
              f"try a larger --scale than {args.scale})")
        continue

    last = trace[-1]
    param = last.get("mu", last.get("eta"))
    print(f"  final: param={param:.2e}, r_abs={last['r_abs']:.2e}, "
          f"compl_gap={last['compl_gap']:.2e}, cond_kkt={last['cond_kkt']:.2e}")

    # Column order follows the method's own trace records (mu vs eta etc.)
    fieldnames = list(trace[0].keys())
    out_path = os.path.join(out_dir, f"trace_{method}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)
    print(f"  -> {os.path.relpath(out_path, os.path.dirname(__file__))}")

# --- Consistency check: stress and slip must agree across methods -------------
if results:
    # ten2voigt order is [11, 22, 33, 12, 13, 23] with shear scaled by fact=2,
    # so slots 4..6 are 2*s12, 2*s13, 2*s23 (see cp_base.ten2voigt).
    sig_labels   = ["s11", "s22", "s33", "s12", "s13", "s23"]
    gamma_labels = [f"g{a + 1}" for a in range(len(next(iter(results.values()))["gamma12"]))]

    cmp_path = os.path.join(out_dir, "comparison.csv")
    with open(cmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_iter"] + sig_labels + gamma_labels)
        for method, res in results.items():
            writer.writerow(
                [method, res["n_iter"]]
                + [f"{v:.12e}" for v in res["sig"]]
                + [f"{v:.12e}" for v in res["gamma12"]]
            )

    print("\n--- Consistency check (Voigt stress per method) ---")
    print(f"{'method':<14}" + "".join(f"{lbl:>14}" for lbl in sig_labels))
    for method, res in results.items():
        print(f"{method:<14}" + "".join(f"{v:>14.6e}" for v in res["sig"]))

    if len(results) > 1:
        ref_name, ref = next(iter(results.items()))
        print(f"\nMax abs difference vs '{ref_name}':")
        for method, res in results.items():
            if method == ref_name:
                continue
            dsig = float(np.max(np.abs(res["sig"] - ref["sig"])))
            dgam = float(np.max(np.abs(res["gamma12"] - ref["gamma12"])))
            print(f"  {method:<14} d_sigma_max = {dsig:.3e}, d_gamma_max = {dgam:.3e}")
    print(f"-> {os.path.relpath(cmp_path, os.path.dirname(__file__))}")

print("\nSingle-step diagnostics complete. Results saved in 'out_single/'.")
