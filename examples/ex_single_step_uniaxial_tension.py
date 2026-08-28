"""
Single-step Newton-system diagnostics for the four solver methods.

Loads ONE uniaxial-tension step, from a virgin state straight into the plastic
regime, for the reference orientation phi = (0, 0, 0). For that orientation the
tensile axis is the crystal [100] direction, where eight slip systems share the
maximum Schmid factor 1/sqrt(6) = 0.4082 and therefore activate simultaneously:
the classical Taylor ambiguity. The local return-mapping problem is then rank
deficient on the active set, and the point of this driver is to expose how each
method's Newton system responds to that degeneracy.

Methods compared (all solve the same physical problem except VP):

    IPM_classic      classical primal-dual interior point (full 2m x 2m system)
    MB-IPM-classic   modified-barrier IPM, base Polyak method (m x m Schur
                     complement solved, 2m x 2m primal-dual system assembled)
    AL               augmented Lagrangian / semi-smooth Newton (m x m)
    VP               Perzyna visco-plastic penalty regularization (m x m).
                     Rate dependent: it matches the rate-independent methods
                     only in the eta -> 0 limit.

Load level
----------
The step is applied at a prescribed multiple of the yield strain, so it is
plastic by construction. The yield strain is computed from the elastic trial
itself: the trial stress is linear in the strain, so scaling the unit uniaxial
strain by tau0 / max_a tau_a puts the most-stressed slip system exactly on its
yield surface. --scale is that multiple (1.0 = exactly at yield); the default
1.5 gives a trial overstress tau_max / tau0 = 1.5.

Output (all in out_single/)
--------------------------
trace_<method>.csv
    Per-(inner-)Newton-iteration history, one row per iteration:

    iter        global Newton iteration counter (1-based)
    k           outer (barrier / multiplier / penalty) loop index
    n           inner Newton index within the current outer step
    mu / eta    continuation parameter: barrier/shift mu (IPM_classic,
                MB-IPM-classic), penalty eta (AL), or viscosity eta (VP)
    r_abs       method's inner residual norm
    r_rel       routine's own (possibly normalized) residual
    compl_gap   complementary gap  max_a |dlambda_a * s_a|  (IPM) /
                max_a |dlambda_a * Phi_a|  (AL, VP)
    cond_solved condition number of the linear system actually solved
    cond_kkt    condition number of the full 2m x 2m primal-dual KKT matrix
                (IPM_classic, MB-IPM-classic); equals cond_solved for AL and
                VP, which have no primal-dual system
    alpha       fraction-to-boundary / line-search step length
    n_active    number of slip systems with dlambda > 1e-10

newton_<method>.npz
    The final Newton system at the converged iterate, for offline analysis:
    solved (matrix actually factorized), kkt (full primal-dual matrix), dPhi
    (yield-function Jacobian), the converged dlambda / Phi / slacks, the
    singular values of solved and kkt, and the null-space basis of dPhi
    restricted to the active set -- the slip-rate combinations that leave the
    stress unchanged, i.e. the Taylor ambiguity itself.

newton_summary.csv
    One row per method: dimensions, cond, smallest/largest singular value,
    numerical rank and null-space dimension of both the solved and the KKT
    matrix, plus the size of the converged active set.

taylor.npz / printed Taylor block
    Method-independent analysis of the Schmid-Gram matrix A_ab = Z_a : C : Z_b
    on the trial-active set, which is what actually carries the degeneracy.

comparison.csv
    Converged stress (Voigt) and accumulated slip per method, with pairwise max
    differences printed, so the physical results can be checked to agree.

Tracing and the matrix export are enabled only here (solver_cfg["trace"] =
True); ordinary multi-step simulations run with no per-iteration recording and
no overhead.

Usage:
    python ex_single_step_uniaxial_tension.py
    python ex_single_step_uniaxial_tension.py --methods IPM_classic AL
    python ex_single_step_uniaxial_tension.py --scale 2.0
    python ex_single_step_uniaxial_tension.py --orientation 42
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
    "--methods", nargs="+",
    default=["IPM_classic", "MB-IPM-classic", "AL", "VP"],
    help="solver method(s) to trace (default: IPM_classic MB-IPM-classic AL VP). "
         "Note: VP is the rate-dependent penalty (viscoplastic) approach -- it "
         "only matches the rate-independent methods in the eta -> 0 limit.",
)
parser.add_argument(
    "--scale", type=float, default=1.5, metavar="F",
    help="load level as a multiple of the yield strain: the trial overstress is "
         "tau_max / tau0 = F. F = 1.0 sits exactly on the yield surface, so F "
         "must be > 1 to be plastic (default: 1.5)",
)
parser.add_argument(
    "--orientation", type=int, default=None, metavar="N",
    help="1-based row index into initial_rotations.csv (overrides phi_deg in "
         "config). The 8-fold Taylor ambiguity is specific to the default "
         "phi_deg = [0, 0, 0]; other orientations activate fewer systems.",
)
parser.add_argument(
    "--active-tol", type=float, default=1e-10, metavar="TOL",
    help="slip threshold above which a system counts as active (default: 1e-10)",
)
args = parser.parse_args()

# --- Load config --------------------------------------------------------------
config_path = os.path.join(os.path.dirname(__file__), "config_tension.toml")
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
    print(f"Orientation: Euler angles {mat_cfg['phi_deg']} deg (crystal [100] "
          f"tensile axis -- 8-fold Taylor ambiguity expected)")

Za = np.stack(mat.Za)          # (m, 3, 3) Schmid tensors
nSlip = len(mat.Za)
tau0 = mat.tau0

# --- Spectral helpers ---------------------------------------------------------
def spectrum(A):
    """Singular values, condition number, numerical rank and null-space dim."""
    A = np.asarray(A, dtype=float)
    sv = np.linalg.svd(A, compute_uv=False)
    s_max, s_min = float(sv[0]), float(sv[-1])
    # numpy's default matrix_rank tolerance
    tol = s_max * max(A.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(sv > tol))
    return {
        "n": A.shape[0],
        "s_max": s_max,
        "s_min": s_min,
        "cond": (s_max / s_min) if s_min > 0.0 else np.inf,
        "rank": rank,
        "null_dim": A.shape[0] - rank,
        "sv": sv,
    }


def null_space(A):
    """Orthonormal basis of the numerical null space of A (columns)."""
    A = np.asarray(A, dtype=float)
    U, sv, Vt = np.linalg.svd(A)
    tol = sv[0] * max(A.shape) * np.finfo(float).eps
    return Vt[np.count_nonzero(sv > tol):].T


# --- Single load step: scale the unit uniaxial strain to the requested level ---
nu = mat_cfg["nu"]
eps_unit = np.diag([1.0, -nu, -nu])                    # unit uniaxial tension, e1
sig_unit = np.einsum('ijkl,kl->ij', mat.C, eps_unit)
tau_unit = np.einsum('aij,ij->a', Za, sig_unit)

if float(np.max(tau_unit)) <= 0.0:
    raise RuntimeError("No slip system is loaded in tension for this orientation.")

eps_yield = tau0 / float(np.max(tau_unit))             # eps_11 that puts tau_max on tau0
eps = args.scale * eps_yield * eps_unit

sig_trial = np.einsum('ijkl,kl->ij', mat.C, eps)
tau_trial = np.einsum('aij,ij->a', Za, sig_trial)
overstress = float(np.max(tau_trial)) / tau0
trial_active = np.flatnonzero(tau_trial > tau0 - 1e-12 * tau0)

print(f"Load step:   eps_11 = {eps[0, 0]:.6e}  ({args.scale:g} x yield strain "
      f"{eps_yield:.6e})")
print(f"             sig_11_trial = {sig_trial[0, 0]:.6f} MPa, "
      f"tau_max / tau0 = {overstress:.4f}")
print(f"             {len(trial_active)} slip system(s) above yield in the "
      f"elastic trial: {trial_active.tolist()}")
if overstress <= 1.0:
    raise SystemExit("ERROR: the step is elastic -- rerun with --scale > 1.0.")

# --- Output directory ---------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "out_single")
os.makedirs(out_dir, exist_ok=True)


def rel(path):
    return os.path.relpath(path, os.path.dirname(__file__))


# --- Method-independent Taylor analysis of the Schmid-Gram matrix -------------
# A_ab = Z_a : C : Z_b is the Hessian of the local problem; its restriction to
# the active set is the object whose rank deficiency IS the Taylor ambiguity.
A_full = np.einsum('aij,ijkl,bkl->ab', Za, mat.C, Za)
A_act = A_full[np.ix_(trial_active, trial_active)]
sp_full, sp_act = spectrum(A_full), spectrum(A_act)
N_act = null_space(A_act)

print("\n--- Taylor ambiguity (Schmid-Gram matrix A_ab = Z_a : C : Z_b) ---")
print(f"  all {nSlip} systems : rank {sp_full['rank']:2d} / {sp_full['n']:2d}, "
      f"null dim {sp_full['null_dim']:2d}  "
      f"(the Schmid tensors are symmetric and traceless, so they span at most "
      f"the 5-dimensional deviatoric space)")
print(f"  active set ({len(trial_active):2d})  : rank {sp_act['rank']:2d} / "
      f"{sp_act['n']:2d}, null dim {sp_act['null_dim']:2d}, "
      f"cond {sp_act['cond']:.3e}")
print(f"  -> {sp_act['null_dim']} independent slip combination(s) on the active "
      f"set leave the stress unchanged")

taylor_path = os.path.join(out_dir, "taylor.npz")
np.savez(taylor_path,
         A_full=A_full, A_active=A_act, active_idx=trial_active,
         sv_full=sp_full["sv"], sv_active=sp_act["sv"], null_basis=N_act,
         tau_trial=tau_trial, eps=eps, sig_trial=sig_trial)
print(f"  -> {rel(taylor_path)}")

# --- Run each method once, recording the local Newton history -----------------
results = {}      # method -> dict(sig_voigt, gamma12, n_iter, spectra, ...)

for method in args.methods:
    cfg = copy.deepcopy(config)
    cfg["Solver"]["method"] = method
    cfg["Solver"]["n_steps"] = 1            # single step -> dt = 1

    compute_stress, solver_cfg = get_solver(cfg)
    solver_cfg["trace"] = True              # per-iteration tracer + matrix export
    solver_cfg["verbose"] = False           # keep stdout clean; CSV holds the detail

    hist0 = mat.initialize_history()
    sig, new_hist, n_iter = compute_stress(eps, hist0, mat, solver_cfg)

    trace = new_hist.get("trace") or []
    converged = n_iter >= 0
    status = "converged" if converged else "FAILED"

    print(f"\n[{method}] {status}: {n_iter} Newton iterations, "
          f"{len(trace)} traced rows")

    if trace:
        last = trace[-1]
        param = last.get("mu", last.get("eta"))
        print(f"  final iterate: param={param:.2e}, r_abs={last['r_abs']:.2e}, "
              f"compl_gap={last['compl_gap']:.2e}, "
              f"cond_solved={last['cond_solved']:.3e}, "
              f"cond_kkt={last['cond_kkt']:.3e}")

        # Column order follows the method's own trace records (mu vs eta etc.)
        out_path = os.path.join(out_dir, f"trace_{method}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)
        print(f"  -> {rel(out_path)}")
    else:
        print("  (no iterations recorded -- elastic predictor or empty trace)")

    if not converged:
        continue

    dlambda = np.asarray(new_hist["gamma_a"])
    active = np.flatnonzero(dlambda > args.active_tol)

    res = {
        "sig": ten2voigt(sig),
        "gamma12": gamma24_to_12(new_hist["gamma_a"]),
        "n_iter": n_iter,
        "dlambda": dlambda,
        "active": active,
    }

    # ---- Final Newton system: spectra, rank, null space ----------------------
    nf = new_hist.get("newton_final")
    if nf is not None:
        sp_solved = spectrum(nf["solved"])
        sp_kkt = spectrum(nf["kkt"])
        res["sp_solved"], res["sp_kkt"] = sp_solved, sp_kkt

        # dPhi on the converged active set: the method-independent degeneracy
        # as each method's own Jacobian sees it.
        dPhi_act = (nf["dPhi"][np.ix_(active, active)] if len(active)
                    else np.zeros((0, 0)))
        sp_dPhi_act = spectrum(dPhi_act) if len(active) else None
        res["sp_dPhi_act"] = sp_dPhi_act

        print(f"  active set ({len(active)}): {active.tolist()}")
        print(f"    dlambda_active = "
              f"[{', '.join(f'{v:.6e}' for v in dlambda[active])}]")
        print(f"    solved  {sp_solved['n']:2d}x{sp_solved['n']:<2d} "
              f"cond {sp_solved['cond']:.3e}  "
              f"s_min {sp_solved['s_min']:.3e}  s_max {sp_solved['s_max']:.3e}  "
              f"rank {sp_solved['rank']}/{sp_solved['n']}  "
              f"null {sp_solved['null_dim']}")
        print(f"    kkt     {sp_kkt['n']:2d}x{sp_kkt['n']:<2d} "
              f"cond {sp_kkt['cond']:.3e}  "
              f"s_min {sp_kkt['s_min']:.3e}  s_max {sp_kkt['s_max']:.3e}  "
              f"rank {sp_kkt['rank']}/{sp_kkt['n']}  "
              f"null {sp_kkt['null_dim']}")
        if sp_dPhi_act is not None:
            print(f"    dPhi|active {sp_dPhi_act['n']:2d}x{sp_dPhi_act['n']:<2d} "
                  f"cond {sp_dPhi_act['cond']:.3e}  "
                  f"rank {sp_dPhi_act['rank']}/{sp_dPhi_act['n']}  "
                  f"null {sp_dPhi_act['null_dim']}")

        npz_path = os.path.join(out_dir, f"newton_{method}.npz")
        dump = {
            "solved": nf["solved"], "kkt": nf["kkt"], "dPhi": nf["dPhi"],
            "dlambda": nf["dlambda"], "Phi": nf["Phi"],
            "sv_solved": sp_solved["sv"], "sv_kkt": sp_kkt["sv"],
            "active_idx": active,
            "null_dPhi_active": (null_space(dPhi_act) if len(active)
                                 else np.zeros((0, 0))),
            "sig": np.asarray(sig),
        }
        for key in ("slacks", "lam_k", "active", "mu", "eta", "dgamma0"):
            if key in nf:
                dump[f"p_{key}" if key in ("mu", "eta", "dgamma0") else key] = nf[key]
        np.savez(npz_path, **dump)
        print(f"  -> {rel(npz_path)}")

    results[method] = res

# --- Newton-system summary table ---------------------------------------------
summary_rows = [m for m in results if "sp_solved" in results[m]]
if summary_rows:
    sum_path = os.path.join(out_dir, "newton_summary.csv")
    with open(sum_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "n_iter", "n_active",
            "dim_solved", "cond_solved", "smin_solved", "smax_solved",
            "rank_solved", "null_solved",
            "dim_kkt", "cond_kkt", "smin_kkt", "smax_kkt", "rank_kkt", "null_kkt",
            "dim_dPhi_act", "cond_dPhi_act", "rank_dPhi_act", "null_dPhi_act",
        ])
        for method in summary_rows:
            r = results[method]
            s, kk, dp = r["sp_solved"], r["sp_kkt"], r["sp_dPhi_act"]
            writer.writerow([
                method, r["n_iter"], len(r["active"]),
                s["n"], f"{s['cond']:.6e}", f"{s['s_min']:.6e}",
                f"{s['s_max']:.6e}", s["rank"], s["null_dim"],
                kk["n"], f"{kk['cond']:.6e}", f"{kk['s_min']:.6e}",
                f"{kk['s_max']:.6e}", kk["rank"], kk["null_dim"],
                dp["n"] if dp else 0,
                f"{dp['cond']:.6e}" if dp else "",
                dp["rank"] if dp else "",
                dp["null_dim"] if dp else "",
            ])

    print("\n--- Newton system at the converged iterate ---")
    print(f"{'method':<16}{'iters':>6}{'act':>5}"
          f"{'dim':>6}{'cond(solved)':>15}{'rank':>8}"
          f"{'dim':>6}{'cond(kkt)':>15}{'rank':>8}")
    for method in summary_rows:
        r = results[method]
        s, kk = r["sp_solved"], r["sp_kkt"]
        rank_s = f"{s['rank']}/{s['n']}"
        rank_k = f"{kk['rank']}/{kk['n']}"
        print(f"{method:<16}{r['n_iter']:>6}{len(r['active']):>5}"
              f"{s['n']:>6}{s['cond']:>15.4e}{rank_s:>8}"
              f"{kk['n']:>6}{kk['cond']:>15.4e}{rank_k:>8}")
    print(f"-> {rel(sum_path)}")

# --- Consistency check: stress and slip must agree across methods -------------
if results:
    # ten2voigt order is [11, 22, 33, 12, 13, 23] with shear scaled by fact=2,
    # so slots 4..6 are 2*s12, 2*s13, 2*s23 (see cp_base.ten2voigt).
    sig_labels = ["s11", "s22", "s33", "s12", "s13", "s23"]
    gamma_labels = [f"g{a + 1}" for a in range(len(next(iter(results.values()))["gamma12"]))]

    cmp_path = os.path.join(out_dir, "comparison.csv")
    with open(cmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_iter", "n_active"] + sig_labels + gamma_labels)
        for method, res in results.items():
            writer.writerow(
                [method, res["n_iter"], len(res["active"])]
                + [f"{v:.12e}" for v in res["sig"]]
                + [f"{v:.12e}" for v in res["gamma12"]]
            )

    print("\n--- Consistency check (Voigt stress per method) ---")
    print(f"{'method':<16}" + "".join(f"{lbl:>14}" for lbl in sig_labels))
    for method, res in results.items():
        print(f"{method:<16}" + "".join(f"{v:>14.6e}" for v in res["sig"]))

    if len(results) > 1:
        ref_name, ref = next(iter(results.items()))
        print(f"\nMax abs difference vs '{ref_name}':")
        for method, res in results.items():
            if method == ref_name:
                continue
            dsig = float(np.max(np.abs(res["sig"] - ref["sig"])))
            dgam = float(np.max(np.abs(res["gamma12"] - ref["gamma12"])))
            print(f"  {method:<16} d_sigma_max = {dsig:.3e}, d_gamma_max = {dgam:.3e}")
    print(f"-> {rel(cmp_path)}")

print("\nSingle-step diagnostics complete. Results saved in 'out_single/'.")
