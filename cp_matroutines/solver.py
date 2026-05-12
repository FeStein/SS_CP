"""
Solver dispatcher — maps config["Solver"]["method"] to the correct
compute_stress function and its solver-specific config section.
"""

from cp_matroutines.mat_IPM_classic import compute_stress as _ipm_classic
from cp_matroutines.mat_VP import compute_stress as _vp

_SOLVERS = {
    "IPM_classic": (_ipm_classic, "IPM"),
    "IPM_acc": (_ipm_acc, "IPM"),
    "VP":  (_vp,  "VP"),
}

def get_config_key(method: str) -> str:
    """Return the TOML config section key for a solver method."""
    if method not in _SOLVERS:
        raise ValueError(f"Unknown solver method '{method}'. Available: {list(_SOLVERS.keys())}")
    return _SOLVERS[method][1]


def get_solver(config: dict):
    """Return (compute_stress, solver_cfg) based on config["Solver"]["method"]."""
    method = config["Solver"]["method"]
    if method not in _SOLVERS:
        raise ValueError(f"Unknown solver method '{method}'. Available: {list(_SOLVERS.keys())}")
    compute_fn, config_key = _SOLVERS[method]
    solver_cfg = dict(config[config_key])          # shallow copy so we can inject keys
    if method == "VP":
        solver_cfg["dt"] = 1.0 / config["Solver"]["n_steps"]
    return compute_fn, solver_cfg
