#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
physics_dependency_final_v4.py

Mechanistic VQLS-FEM analysis with unit-invariant structural diagnostics.

Why this revision exists
------------------------
The previous version used raw Euclidean norms and the raw stiffness spectrum
for cross-family interpretation. For structural systems with mixed DOF units
(e.g. displacement/rotation), raw ||e||, ||K e||, and kappa(K) depend on the
chosen coordinate units.

This version therefore makes the paper-facing diagnostics invariant to
positive diagonal changes of units by using Jacobi-equilibrated coordinates:

    D       = diag(K)
    K_eq    = D^{-1/2} K D^{-1/2}
    y       = D^{1/2} u
    f_eq    = D^{-1/2} f

Under a diagonal change of variables, K_eq, y, and f_eq are unchanged.

Primary paper-facing quantities
-------------------------------
1) kappa_eq = kappa(D^{-1/2} K D^{-1/2})
2) relative energy-norm error
       eK_rel = sqrt(e^T K e / (u^T K u))
3) equilibrated physical residual
       r_eq = ||D^{-1/2}(K u_q - f)|| / ||D^{-1/2} f||

Raw kappa(K), raw relative L2, and raw physical residual are retained only
as auxiliary diagnostics in the CSV/manifest.

Repeated eigenvalues
--------------------
Modal weights are aggregated over nearly-degenerate eigenspaces before PR
and r99 are computed. This removes arbitrary basis dependence inside repeated
eigenspaces (important for Heat2D).

Figures
-------
fig1_equilibrated_spectral_transfer.*
    Unit-invariant, grouped eigenspace analysis:
      left:  Heat1D vs Beam1D (physics dependency)
      right: Heat1D vs Heat2D (dimensionality dependency)
    Rows:
      spectrum of K_eq
      grouped forcing weights
      cumulative grouped solution weights

fig2_ideal_depth_and_preconditioned_target.*
    Left: ideal infidelity vs ansatz depth, clipped below a configurable floor.
    Right: PR/r99 of x_pre in the eigenbasis of the ACTUAL preconditioned A,
           with kappa_pre and d99 annotations.

fig3_equilibrated_error_amplification.*
    Four families. In the eigenbasis of K_eq, compares
      balanced-L2 contribution          beta_i^2
      energy-error contribution         lambda_i beta_i^2
      equilibrated-residual contribution lambda_i^2 beta_i^2
    where beta_i are coefficients of D^{1/2}(u_q-u_c).

fig4_heat1d_heat2d_noisy_convergence.*
    Same-budget noisy-none convergence for Heat1D vs Heat2D:
      fidelity vs iteration
      balanced L2 vs iteration

Data policy
-----------
The final noisy vector used for paper-facing final metrics is explicitly
dof32_u_q_fis[-1], because the audit showed that this vector reproduces the
last l2_fis row for Heat1D, Heat2D, Beam1D, and Beam2D.

No new VQLS runs are performed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Explicit project layout
# -----------------------------------------------------------------------------

HOME = Path.home()

PHYSICAL_K = {
    "heat1d": HOME / "Heat1D" / "heat1d_32DOF_K.npz",
    "heat2d": HOME / "Heat2D" / "heat2d_32DOF_K.npz",
    "beam1d": HOME / "beam1d_final_results" / "beam1d_32DOF_K.npz",
    "beam2d": HOME / "Beam 2D" / "beam2d_32DOF_K.npz",
}

CLASSICAL_SOL = {
    "heat1d": HOME / "Heat1D" / "sol_heat1d_32_none.npz",
    "heat2d": HOME / "Heat2D" / "sol_heat2d_32_none.npz",
    "beam1d": HOME / "beam1d_final_results" / "sol_beam1d_32_none.npz",
    "beam2d": HOME / "Beam 2D" / "sol_beam2d_32_none.npz",
}

NOISY_ARCHIVE = {
    "heat1d": HOME / "other_families_trex_campaign"
              / "raw_none_seed1001_20260923_124835"
              / "heat1d" / "archive_seed1001.npz",
    "heat2d": HOME / "other_families_trex_campaign"
              / "raw_none_seed1001_20260923_124835"
              / "heat2d" / "archive_seed1001.npz",
    "beam2d": HOME / "other_families_trex_campaign"
              / "raw_none_seed1001_20260923_124835"
              / "beam2d" / "archive_seed1001.npz",
    "beam1d": HOME / "beam1d_final_results"
              / "beam1d_paired_final_20260922_184543"
              / "none" / "archive_seed8103.npz",
}

NOISY_METRICS = {
    "heat1d": HOME / "other_families_trex_campaign"
              / "raw_none_seed1001_20260923_124835"
              / "heat1d" / "metrics_seed1001.csv",
    "heat2d": HOME / "other_families_trex_campaign"
              / "raw_none_seed1001_20260923_124835"
              / "heat2d" / "metrics_seed1001.csv",
    "beam2d": HOME / "other_families_trex_campaign"
              / "raw_none_seed1001_20260923_124835"
              / "beam2d" / "metrics_seed1001.csv",
    "beam1d": HOME / "beam1d_final_results"
              / "beam1d_paired_final_20260922_184543"
              / "none" / "metrics_seed8103.csv",
}

IDEAL_ROOT = HOME / "ideal_preconditioning_32dof"
IDEAL_RUN_HINT = IDEAL_ROOT / "results_full_20260924_164440"
LATEST_IDEAL_POINTER = IDEAL_ROOT / "LATEST_full_RUN.txt"

FAMILIES = ["heat1d", "beam1d", "heat2d", "beam2d"]
FIG1_COMPARISONS = [
    ("Physics dependency: Heat1D vs Beam1D", ("heat1d", "beam1d")),
    ("Dimensionality dependency: Heat1D vs Heat2D", ("heat1d", "heat2d")),
]

PRETTY = {
    "heat1d": "Heat1D",
    "heat2d": "Heat2D",
    "beam1d": "Beam1D",
    "beam2d": "Beam2D",
}

DEFAULT_PRECONDITIONER = "amg"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def as_dense(a):
    if sp.issparse(a):
        return a.toarray()
    return np.asarray(a)


def load_npz_matrix(path: Path, preferred=("K", "A", "arr_0")) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    try:
        keys = set(z.files)
        if {"data", "indices", "indptr", "shape"}.issubset(keys):
            return sp.csr_matrix(
                (z["data"], z["indices"], z["indptr"]),
                shape=tuple(z["shape"]),
            ).toarray().astype(float)

        for k in preferred:
            if k in keys:
                return np.asarray(as_dense(z[k]), dtype=float)

        if len(z.files) == 1:
            return np.asarray(as_dense(z[z.files[0]]), dtype=float)

        raise RuntimeError(f"Cannot identify matrix in {path}; keys={z.files}")
    finally:
        z.close()


def first_vector_from_npz(path: Path, keys: List[str]) -> Tuple[np.ndarray, str]:
    z = np.load(path, allow_pickle=True)
    try:
        for key in keys:
            if key in z.files:
                a = np.asarray(z[key])
                if a.ndim == 1 and a.size > 1 and np.issubdtype(a.dtype, np.number):
                    return np.asarray(a, dtype=float).ravel(), key
        raise RuntimeError(f"No vector among {keys} in {path}; keys={z.files}")
    finally:
        z.close()


def maybe_align_global_sign(uq: np.ndarray, uc: np.ndarray) -> np.ndarray:
    n = min(len(uq), len(uc))
    if float(np.dot(uq[:n], uc[:n])) < 0:
        return -uq
    return uq


def symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def eig_spd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = symmetrize(A)
    lam, V = np.linalg.eigh(A)
    tol = max(1e-14, 1e-12 * max(1.0, float(np.max(np.abs(lam)))))
    keep = lam > tol
    if np.count_nonzero(keep) < 1:
        raise RuntimeError("No positive eigenvalues after SPD filtering.")
    return lam[keep], V[:, keep]


def normalize_prob(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    s = float(np.sum(x))
    if s <= 0:
        return np.zeros_like(x)
    return x / s


def participation_ratio(w):
    w = normalize_prob(w)
    den = float(np.sum(w**2))
    return float(1.0 / den) if den > 0 else np.nan


def effective_rank(w, threshold=0.99):
    w = np.sort(normalize_prob(w))[::-1]
    if len(w) == 0 or np.sum(w) <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(w), threshold) + 1)


def normalized_entropy(w):
    w = normalize_prob(w)
    w = w[w > 0]
    if len(w) <= 1:
        return 0.0
    return float(-np.sum(w * np.log(w)) / math.log(len(w)))


def find_col(df: pd.DataFrame, candidates: List[str], required=False):
    cmap = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    for col in df.columns:
        low = col.lower()
        if any(c.lower() in low for c in candidates):
            return col
    if required:
        raise RuntimeError(f"Could not find {candidates}; columns={list(df.columns)}")
    return None


def last_finite(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    return float(s.iloc[-1]) if len(s) else np.nan


def group_eigenspaces(
    lam: np.ndarray,
    arrays: Dict[str, np.ndarray],
    rtol: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Aggregate quantities over nearly-degenerate eigenvalue groups.

    For a repeated eigenspace, individual eigenvectors are basis-dependent,
    but the SUM of squared projection weights in that subspace is invariant.
    """
    lam = np.asarray(lam, dtype=float)
    if len(lam) == 0:
        return {"lam": lam, **{k: np.asarray(v) for k, v in arrays.items()}}

    groups = []
    start = 0
    for i in range(1, len(lam)):
        scale = max(abs(lam[start]), abs(lam[i]), 1e-300)
        if abs(lam[i] - lam[start]) > rtol * scale:
            groups.append((start, i))
            start = i
    groups.append((start, len(lam)))

    out = {"lam": np.array([np.mean(lam[a:b]) for a, b in groups], dtype=float)}
    out["multiplicity"] = np.array([b - a for a, b in groups], dtype=int)

    for name, arr in arrays.items():
        arr = np.asarray(arr, dtype=float)
        out[name] = np.array([np.sum(arr[a:b]) for a, b in groups], dtype=float)

    return out


def jacobi_equilibrated_coordinates(K: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Unit-invariant diagonal equilibration for a symmetric positive-definite K.

      K_eq = D^{-1/2} K D^{-1/2}
      y    = D^{1/2} u
      f_eq = D^{-1/2} f
    """
    K = symmetrize(K)
    d = np.diag(K).astype(float)
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        bad = np.where((~np.isfinite(d)) | (d <= 0))[0]
        raise RuntimeError(f"Non-positive/invalid diagonal entries in K at indices {bad[:10]}")

    sqrt_d = np.sqrt(d)
    invsqrt_d = 1.0 / sqrt_d
    K_eq = (invsqrt_d[:, None] * K) * invsqrt_d[None, :]

    f = K @ u
    y = sqrt_d * u
    f_eq = invsqrt_d * f

    return {
        "D": d,
        "sqrtD": sqrt_d,
        "invsqrtD": invsqrt_d,
        "K_eq": symmetrize(K_eq),
        "f": f,
        "y": y,
        "f_eq": f_eq,
    }


def recursive_find_numeric(obj: Any, target_names: List[str]) -> Optional[float]:
    targets = {x.lower() for x in target_names}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in targets:
                try:
                    return float(v)
                except Exception:
                    pass
        for v in obj.values():
            hit = recursive_find_numeric(v, target_names)
            if hit is not None:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = recursive_find_numeric(v, target_names)
            if hit is not None:
                return hit
    return None


# -----------------------------------------------------------------------------
# 1) Unit-invariant physical/equilibrated modal study
# -----------------------------------------------------------------------------

UC_KEYS = ["u_c", "u_fem", "u_exact", "x_phys", "u_classical", "uc", "x_classical"]


def physical_modal_analysis(family: str, eig_group_rtol: float, zero_weight_tol: float) -> Dict:
    K_path = PHYSICAL_K[family]
    uc_path = CLASSICAL_SOL[family]

    K = load_npz_matrix(K_path)
    uc, uc_key = first_vector_from_npz(uc_path, UC_KEYS)

    n = min(K.shape[0], len(uc))
    K = symmetrize(K[:n, :n])
    uc = uc[:n]

    # Raw spectrum retained only as an auxiliary diagnostic.
    lam_raw, _ = eig_spd(K)
    kappa_raw = float(lam_raw[-1] / lam_raw[0])

    eq = jacobi_equilibrated_coordinates(K, uc)
    K_eq, y, f_eq = eq["K_eq"], eq["y"], eq["f_eq"]

    lam, V = eig_spd(K_eq)
    c = V.T @ f_eq
    forcing_w = normalize_prob(np.abs(c) ** 2)

    # y = K_eq^{-1} f_eq
    ycoeff = c / lam
    solution_w = normalize_prob(np.abs(ycoeff) ** 2)

    y_direct_coeff = V.T @ y
    solution_w_direct = normalize_prob(np.abs(y_direct_coeff) ** 2)
    y_recon = V @ ycoeff

    groups = group_eigenspaces(
        lam,
        {"forcing": forcing_w, "solution": solution_w},
        rtol=eig_group_rtol,
    )
    gf = normalize_prob(groups["forcing"])
    gu = normalize_prob(groups["solution"])

    zero_mask = gf <= zero_weight_tol * max(1.0, float(np.max(gf)))

    return {
        "family": family,
        "K_path": str(K_path),
        "uc_path": str(uc_path),
        "uc_key": uc_key,
        "K": K,
        "uc": uc,
        "f": eq["f"],
        "D": eq["D"],
        "K_eq": K_eq,
        "y": y,
        "f_eq": f_eq,
        "lam_eq": lam,
        "V_eq": V,
        "raw_kappa_K": kappa_raw,
        "kappa_eq": float(lam[-1] / lam[0]),
        "group_lam_eq": groups["lam"],
        "group_multiplicity": groups["multiplicity"],
        "group_forcing_w": gf,
        "group_solution_w": gu,
        "group_forcing_zero_mask": zero_mask,
        "forcing_PR_grouped": participation_ratio(gf),
        "forcing_r99_grouped": effective_rank(gf),
        "forcing_entropy_grouped": normalized_entropy(gf),
        "solution_PR_grouped": participation_ratio(gu),
        "solution_r99_grouped": effective_rank(gu),
        "solution_entropy_grouped": normalized_entropy(gu),
        "num_eigenvalue_groups": int(len(groups["lam"])),
        "max_group_multiplicity": int(np.max(groups["multiplicity"])),
        "equilibrated_reconstruction_mismatch":
            float(np.linalg.norm(y_recon - y) / max(np.linalg.norm(y), 1e-300)),
        "solution_weight_crosscheck":
            float(np.linalg.norm(solution_w - solution_w_direct)),
    }


# -----------------------------------------------------------------------------
# 2) Ideal VQLS depth + actual preconditioned target structure
# -----------------------------------------------------------------------------

def resolve_ideal_run() -> Path:
    if IDEAL_RUN_HINT.exists():
        return IDEAL_RUN_HINT

    if LATEST_IDEAL_POINTER.exists():
        txt = LATEST_IDEAL_POINTER.read_text().strip()
        p = Path(txt).expanduser()
        if not p.is_absolute():
            p = IDEAL_ROOT / p
        if p.exists():
            return p

    candidates = sorted(IDEAL_ROOT.glob("results_full_*"))
    if not candidates:
        raise FileNotFoundError(f"No ideal full run found under {IDEAL_ROOT}")
    return candidates[-1]


def find_family_pc_dir(run_root: Path, family: str, pc: str) -> Path:
    exact = run_root / family / pc
    if exact.exists():
        return exact

    for d1 in run_root.iterdir():
        if d1.is_dir() and d1.name.lower() == family.lower():
            for d2 in d1.iterdir():
                if d2.is_dir() and d2.name.lower() == pc.lower():
                    return d2

    hits = []
    for p in run_root.rglob("depth_summary.csv"):
        s = str(p.parent).lower()
        if family.lower() in s and pc.lower() in s:
            hits.append(p.parent)
    if hits:
        return sorted(hits)[0]

    raise FileNotFoundError(
        f"Could not find ideal directory for family={family}, pc={pc} under {run_root}"
    )


def load_frozen_system(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    try:
        def arr(key):
            if key not in z.files:
                raise KeyError(f"{key} not in {path}; keys={z.files}")
            return np.asarray(as_dense(z[key]), dtype=float)

        A = arr("A")
        x_pre = arr("x_pre").ravel()
        b = arr("b").ravel()

        n = min(A.shape[0], len(x_pre), len(b))
        A = symmetrize(A[:n, :n])
        x_pre = x_pre[:n]
        b = b[:n]

        return {"A": A, "x_pre": x_pre, "b": b, "keys": list(z.files)}
    finally:
        z.close()


def inspect_ideal_sampling(pc_dir: Path) -> Dict:
    """
    Audit whether the ideal study contains multiple independent seeds.
    We do not fabricate error bars if the files do not support them.
    """
    p = pc_dir / "trajectory_all_restarts.csv"
    if not p.exists():
        return {
            "trajectory_all_restarts_exists": False,
            "seed_column": None,
            "unique_seed_count": np.nan,
            "restart_column": None,
            "unique_restart_count": np.nan,
        }

    df = pd.read_csv(p)
    seed_col = find_col(df, ["theta_seed", "seed", "random_seed"], required=False)
    restart_col = find_col(df, ["restart", "restart_id"], required=False)

    seed_count = (
        int(df[seed_col].nunique(dropna=True))
        if seed_col is not None else np.nan
    )
    restart_count = (
        int(df[restart_col].nunique(dropna=True))
        if restart_col is not None else np.nan
    )

    return {
        "trajectory_all_restarts_exists": True,
        "seed_column": seed_col,
        "unique_seed_count": seed_count,
        "restart_column": restart_col,
        "unique_restart_count": restart_count,
    }


def load_ideal_analysis(
    family: str,
    pc: str,
    eig_group_rtol: float,
) -> Dict:
    run_root = resolve_ideal_run()
    d = find_family_pc_dir(run_root, family, pc)

    depth_path = d / "depth_summary.csv"
    frozen_path = d / "frozen_system.npz"
    manifest_path = d / "manifest.json"

    if not depth_path.exists():
        raise FileNotFoundError(depth_path)
    if not frozen_path.exists():
        raise FileNotFoundError(frozen_path)

    df = pd.read_csv(depth_path)
    depth_col = find_col(df, ["depth", "d"], required=True)
    f_col = find_col(df, ["fidelity", "best_F", "F"], required=False)
    inf_col = find_col(df, ["infidelity", "1-F", "one_minus_f"], required=False)

    depth = pd.to_numeric(df[depth_col], errors="coerce").to_numpy(float)

    if f_col is not None:
        F = pd.to_numeric(df[f_col], errors="coerce").to_numpy(float)
        infidelity = 1.0 - F
    elif inf_col is not None:
        infidelity = pd.to_numeric(df[inf_col], errors="coerce").to_numpy(float)
        F = 1.0 - infidelity
    else:
        raise RuntimeError(f"No fidelity/infidelity in {depth_path}")

    valid = np.isfinite(depth) & np.isfinite(F)
    depth, F, infidelity = depth[valid], F[valid], infidelity[valid]
    order = np.argsort(depth)
    depth, F, infidelity = depth[order], F[order], infidelity[order]

    def first_depth(th):
        idx = np.where(F >= th)[0]
        return float(depth[idx[0]]) if len(idx) else np.nan

    frozen = load_frozen_system(frozen_path)
    A, x_pre, b = frozen["A"], frozen["x_pre"], frozen["b"]

    lam, V = eig_spd(A)
    coeff = V.T @ x_pre
    xw = normalize_prob(np.abs(coeff) ** 2)
    groups = group_eigenspaces(lam, {"x": xw}, rtol=eig_group_rtol)
    gx = normalize_prob(groups["x"])

    axb_residual = float(np.linalg.norm(A @ x_pre - b) / max(np.linalg.norm(b), 1e-300))

    manifest_kappa = np.nan
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            hit = recursive_find_numeric(m, ["kappa_pre", "kappa_preconditioned"])
            if hit is not None:
                manifest_kappa = float(hit)
        except Exception:
            pass

    sampling = inspect_ideal_sampling(d)

    return {
        "family": family,
        "pc": pc,
        "run_root": str(run_root),
        "pc_dir": str(d),
        "depth_summary_path": str(depth_path),
        "frozen_system_path": str(frozen_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "depth": depth,
        "F": F,
        "infidelity": infidelity,
        "depth_F_0p99": first_depth(0.99),
        "depth_F_0p999": first_depth(0.999),
        "best_F": float(np.max(F)),
        "kappa_pre_from_A": float(lam[-1] / lam[0]),
        "kappa_pre_manifest": manifest_kappa,
        "xpre_PR_grouped": participation_ratio(gx),
        "xpre_r99_grouped": effective_rank(gx),
        "xpre_entropy_grouped": normalized_entropy(gx),
        "xpre_group_lam": groups["lam"],
        "xpre_group_weight": gx,
        "xpre_num_eigenvalue_groups": int(len(groups["lam"])),
        "A_xpre_relative_residual": axb_residual,
        **sampling,
    }


# -----------------------------------------------------------------------------
# 3) Noisy final-state error in unit-invariant physical coordinates
# -----------------------------------------------------------------------------

def metrics_final_values(path: Path) -> Dict:
    df = pd.read_csv(path)

    fcol = find_col(df, ["F", "fidelity"], required=False)
    l2col = find_col(df, ["l2_fis", "L2_error_vs_fem", "rel_l2"], required=False)
    rcol = find_col(df, ["res_pre", "residual"], required=False)
    ekcol = find_col(df, ["eK_rel", "energy_error", "rel_energy"], required=False)

    return {
        "F_csv": last_finite(df[fcol]) if fcol else np.nan,
        "l2_csv": last_finite(df[l2col]) if l2col else np.nan,
        "res_pre_csv": last_finite(df[rcol]) if rcol else np.nan,
        "eK_rel_csv": last_finite(df[ekcol]) if ekcol else np.nan,
        "metrics_columns": list(df.columns),
        "selected_columns": {
            "F": fcol, "l2": l2col, "res_pre": rcol, "eK_rel": ekcol
        },
    }


def noisy_error_analysis(
    family: str,
    physical: Dict,
    eig_group_rtol: float,
) -> Dict:
    archive = NOISY_ARCHIVE[family]
    metrics_path = NOISY_METRICS[family]

    z = np.load(archive, allow_pickle=True)
    try:
        if "dof32_u_q_fis" not in z.files:
            raise RuntimeError(f"dof32_u_q_fis not found in {archive}")
        traj = np.asarray(z["dof32_u_q_fis"])
        if traj.ndim < 2 or len(traj) == 0:
            raise RuntimeError(f"Invalid dof32_u_q_fis shape={traj.shape}")
        uq = np.asarray(traj[-1], dtype=float).ravel()

        ufem_archive = (
            np.asarray(z["dof32_u_fem"], dtype=float).ravel()
            if "dof32_u_fem" in z.files else None
        )
    finally:
        z.close()

    uc = np.asarray(physical["uc"], dtype=float)
    K = np.asarray(physical["K"], dtype=float)
    n = min(len(uq), len(uc), K.shape[0])

    uq = maybe_align_global_sign(uq[:n], uc[:n])
    uc = uc[:n]
    K = symmetrize(K[:n, :n])
    e = uq - uc
    f = K @ uc

    # Auxiliary raw metrics (not paper-facing cross-family metrics).
    raw_l2 = float(np.linalg.norm(e) / max(np.linalg.norm(uc), 1e-300))
    raw_rphys = float(np.linalg.norm(K @ e) / max(np.linalg.norm(f), 1e-300))

    # Unit-invariant diagonal-equilibrated coordinates.
    eq = jacobi_equilibrated_coordinates(K, uc)
    sqrtD, invsqrtD = eq["sqrtD"], eq["invsqrtD"]
    K_eq, f_eq, y = eq["K_eq"], eq["f_eq"], eq["y"]
    y_e = sqrtD * e

    # Unit-invariant balanced L2:
    # ||D^{1/2}e|| / ||D^{1/2}u||.
    balanced_l2 = float(
        np.linalg.norm(y_e) / max(np.linalg.norm(y), 1e-300)
    )

    # Unit-invariant energy norm.
    num_E = float(e @ K @ e)
    den_E = float(uc @ K @ uc)
    energy_rel = math.sqrt(max(num_E, 0.0) / max(den_E, 1e-300))

    # Unit-invariant equilibrated physical residual.
    r_eq_vec = K_eq @ y_e
    residual_eq = float(
        np.linalg.norm(r_eq_vec) / max(np.linalg.norm(f_eq), 1e-300)
    )

    lam, V = eig_spd(K_eq)
    beta = V.T @ y_e

    # Three views of the SAME equilibrated error:
    # balanced L2            ~ beta_i^2
    # energy norm            ~ lambda_i beta_i^2
    # equilibrated residual  ~ lambda_i^2 beta_i^2
    l2_mode = np.maximum(np.abs(beta) ** 2, 0.0)
    energy_mode = np.maximum(lam * np.abs(beta) ** 2, 0.0)
    residual_mode = np.maximum((lam ** 2) * np.abs(beta) ** 2, 0.0)

    groups = group_eigenspaces(
        lam,
        {"l2": l2_mode, "energy": energy_mode, "residual": residual_mode},
        rtol=eig_group_rtol,
    )
    gL2 = normalize_prob(groups["l2"])
    gE = normalize_prob(groups["energy"])
    gR = normalize_prob(groups["residual"])

    # Fraction of residual contribution in the upper half of the spectrum.
    midpoint = np.median(groups["lam"])
    stiff_half_fraction = float(np.sum(gR[groups["lam"] >= midpoint]))

    csvm = metrics_final_values(metrics_path)

    ufem_delta = np.nan
    if ufem_archive is not None:
        m = min(len(ufem_archive), len(uc))
        ufem_delta = float(
            np.linalg.norm(ufem_archive[:m] - uc[:m]) /
            max(np.linalg.norm(uc[:m]), 1e-300)
        )

    return {
        "family": family,
        "archive_path": str(archive),
        "metrics_path": str(metrics_path),
        "uq_key": "dof32_u_q_fis[-1]",
        "raw_l2_aux": raw_l2,
        "raw_rphys_aux": raw_rphys,
        "balanced_l2_recomputed": balanced_l2,
        "energy_rel_recomputed": energy_rel,
        "residual_eq_recomputed": residual_eq,
        "kappa_eq": float(lam[-1] / lam[0]),
        "group_lam_eq": groups["lam"],
        "group_balanced_l2_w": gL2,
        "group_energy_error_w": gE,
        "group_equilibrated_residual_w": gR,
        "balanced_l2_PR_grouped": participation_ratio(gL2),
        "energy_error_PR_grouped": participation_ratio(gE),
        "equilibrated_residual_PR_grouped": participation_ratio(gR),
        "balanced_l2_r99_grouped": effective_rank(gL2),
        "energy_error_r99_grouped": effective_rank(gE),
        "equilibrated_residual_r99_grouped": effective_rank(gR),
        "stiff_half_residual_fraction": stiff_half_fraction,
        "archive_ufem_vs_static_uc_delta": ufem_delta,
        "archive_vs_csv_l2_delta": raw_l2 - csvm["l2_csv"]
            if np.isfinite(csvm["l2_csv"]) else np.nan,
        "recomputed_energy_vs_csv_delta": energy_rel - csvm["eK_rel_csv"]
            if np.isfinite(csvm["eK_rel_csv"]) else np.nan,
        **csvm,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_grouped_weight_with_zeros(
    ax,
    x,
    w,
    zero_mask,
    label,
    zero_floor=1e-14,
):
    x = np.asarray(x)
    w = np.asarray(w)
    zero_mask = np.asarray(zero_mask, dtype=bool)

    positive = ~zero_mask
    if np.any(positive):
        ax.plot(x[positive], w[positive], marker="o", ms=3, lw=1.3, label=label)

    if np.any(zero_mask):
        # These are displayed at a plotting floor only; the legend says they
        # are numerical/symmetry zeros, not measurements at this magnitude.
        ax.scatter(
            x[zero_mask],
            np.full(np.count_nonzero(zero_mask), zero_floor),
            marker="x",
            s=26,
            label=f"{label}: zero/orthogonal eigenspace",
        )


def plot_fig1(physical: Dict[str, Dict], outdir: Path):
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.5), constrained_layout=True)

    for col, (title, fams) in enumerate(FIG1_COMPARISONS):
        # A) Equilibrated spectrum, grouped by repeated eigenvalues.
        ax = axes[0, col]
        for fam in fams:
            d = physical[fam]
            lam = d["group_lam_eq"]
            x = np.arange(1, len(lam) + 1)
            ax.plot(
                x,
                lam / np.max(lam),
                marker="o",
                ms=3,
                lw=1.4,
                label=f"{PRETTY[fam]}  κ_eq={d['kappa_eq']:.2e}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Distinct eigenspace index")
        ax.set_ylabel(r"$\lambda_g/\lambda_{\max}$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

        # B) Grouped forcing content. Degenerate eigenspaces are invariant.
        ax = axes[1, col]
        all_positive = []
        for fam in fams:
            d = physical[fam]
            w = d["group_forcing_w"]
            z = d["group_forcing_zero_mask"]
            if np.any(~z):
                all_positive.extend(w[~z].tolist())
        floor = max(min(all_positive) * 0.1, 1e-16) if all_positive else 1e-16

        for fam in fams:
            d = physical[fam]
            x = np.arange(1, len(d["group_forcing_w"]) + 1)
            plot_grouped_weight_with_zeros(
                ax,
                x,
                d["group_forcing_w"],
                d["group_forcing_zero_mask"],
                label=(
                    f"{PRETTY[fam]}  PR={d['forcing_PR_grouped']:.2f}, "
                    f"r99={d['forcing_r99_grouped']}"
                ),
                zero_floor=floor,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Distinct eigenspace index")
        ax.set_ylabel(r"Grouped forcing weight")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

        # C) Cumulative grouped solution content.
        ax = axes[2, col]
        for fam in fams:
            d = physical[fam]
            w = np.sort(d["group_solution_w"])[::-1]
            cum = np.cumsum(w)
            x = np.arange(1, len(cum) + 1)
            ax.plot(
                x,
                cum,
                marker="o",
                ms=3,
                lw=1.4,
                label=(
                    f"{PRETTY[fam]}  PR={d['solution_PR_grouped']:.2f}, "
                    f"r99={d['solution_r99_grouped']}"
                ),
            )
        ax.axhline(0.99, ls="--", lw=1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Number of dominant eigenspaces retained")
        ax.set_ylabel("Cumulative solution weight")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Unit-invariant spectral transfer in the Jacobi-equilibrated physical operator",
        fontsize=14,
    )

    png = outdir / "fig1_equilibrated_spectral_transfer.png"
    pdf = outdir / "fig1_equilibrated_spectral_transfer.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_fig2(
    ideal: Dict[str, Dict],
    pc: str,
    outdir: Path,
    infidelity_floor: float,
):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

    # Left: only the meaningful range; tails below the floor are clipped.
    ax = axes[0]
    for fam in FAMILIES:
        d = ideal[fam]
        y = np.maximum(d["infidelity"], infidelity_floor)
        ax.plot(
            d["depth"],
            y,
            marker="o",
            ms=5,
            lw=1.5,
            label=(
                f"{PRETTY[fam]} "
                f"(d99={d['depth_F_0p99'] if np.isfinite(d['depth_F_0p99']) else '—'}, "
                f"d999={d['depth_F_0p999'] if np.isfinite(d['depth_F_0p999']) else '—'})"
            ),
        )
    ax.set_yscale("log")
    ax.set_ylim(bottom=infidelity_floor)
    ax.set_xlabel("Ansatz depth")
    ax.set_ylabel(r"Ideal infidelity $1-F$")
    ax.set_title(
        f"Ideal depth study ({pc})\n"
        f"values below {infidelity_floor:.0e} clipped"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)

    # Right: what the ansatz actually sees after preconditioning.
    ax = axes[1]
    x = np.arange(len(FAMILIES), dtype=float)
    width = 0.34
    r99 = np.array([ideal[f]["xpre_r99_grouped"] for f in FAMILIES], dtype=float)
    pr = np.array([ideal[f]["xpre_PR_grouped"] for f in FAMILIES], dtype=float)

    ax.bar(x - width/2, r99, width, label=r"$r_{99}(x_{\rm pre})$")
    ax.bar(x + width/2, pr, width, label=r"$PR(x_{\rm pre})$")

    ymax = max(1.0, float(np.nanmax(np.r_[r99, pr])))
    for j, fam in enumerate(FAMILIES):
        d = ideal[fam]
        kp = d["kappa_pre_from_A"]
        d99 = d["depth_F_0p99"]
        ax.text(
            x[j],
            max(r99[j], pr[j]) + 0.06 * ymax,
            f"κpre={kp:.2g}\nd99={d99:g}" if np.isfinite(d99)
            else f"κpre={kp:.2g}\nd99=—",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[f] for f in FAMILIES])
    ax.set_ylabel("Effective number of grouped eigenspaces")
    ax.set_title(r"Preconditioned target $x_{\rm pre}$ in eigenbasis of $A$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle(
        "Ideal representability and structure of the actual preconditioned VQLS target",
        fontsize=14,
    )

    png = outdir / "fig2_ideal_depth_and_preconditioned_target.png"
    pdf = outdir / "fig2_ideal_depth_and_preconditioned_target.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_fig3(noisy: Dict[str, Dict], outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), constrained_layout=True)
    axes = axes.ravel()

    for ax, fam in zip(axes, FAMILIES):
        d = noisy[fam]
        x = np.arange(1, len(d["group_lam_eq"]) + 1)

        ax.plot(
            x,
            np.maximum(d["group_balanced_l2_w"], 1e-16),
            marker="^",
            ms=3,
            lw=1.25,
            label=r"Balanced $L_2$: $\beta_g^2$",
        )
        ax.plot(
            x,
            np.maximum(d["group_energy_error_w"], 1e-16),
            marker="o",
            ms=3,
            lw=1.25,
            label=r"Energy error: $\lambda_g\beta_g^2$",
        )
        ax.plot(
            x,
            np.maximum(d["group_equilibrated_residual_w"], 1e-16),
            marker="s",
            ms=3,
            lw=1.25,
            label=r"Equilibrated residual: $\lambda_g^2\beta_g^2$",
        )

        ax.set_yscale("log")
        ax.set_xlabel("Distinct equilibrated eigenspace index")
        ax.set_ylabel("Normalized contribution")
        ax.set_title(
            f"{PRETTY[fam]}\\n"
            f"$L_{{2,eq}}$={d['balanced_l2_recomputed']:.3g}, "
            f"$e_K$={d['energy_rel_recomputed']:.3g}, "
            f"$r_{{eq}}$={d['residual_eq_recomputed']:.3g}"
        )
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8.2)
    fig.suptitle(
        "Unit-invariant spectral amplification of the final VQLS reconstruction error",
        fontsize=14,
    )

    png = outdir / "fig3_equilibrated_error_amplification.png"
    pdf = outdir / "fig3_equilibrated_error_amplification.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def noisy_trajectory_analysis(family: str, physical: Dict) -> Dict:
    """
    Extract N=32 noisy-none trajectory directly from the audited archive.
    Heat1D and Heat2D both use seed 1001 and the same campaign type/budget.
    """
    archive = NOISY_ARCHIVE[family]
    z = np.load(archive, allow_pickle=True)
    try:
        required = ["dof32_iter", "dof32_F", "dof32_u_q_fis"]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"{archive} missing trajectory keys: {missing}")

        it = np.asarray(z["dof32_iter"], dtype=float).ravel()
        F = np.asarray(z["dof32_F"], dtype=float).ravel()
        UQ = np.asarray(z["dof32_u_q_fis"])
    finally:
        z.close()

    uc = np.asarray(physical["uc"], dtype=float)
    K = np.asarray(physical["K"], dtype=float)
    n = min(K.shape[0], len(uc))
    K = symmetrize(K[:n, :n])
    uc = uc[:n]

    eq = jacobi_equilibrated_coordinates(K, uc)
    sqrtD = eq["sqrtD"]
    y = eq["y"]

    m = min(len(it), len(F), len(UQ))
    it, F, UQ = it[:m], F[:m], UQ[:m]

    balanced_l2 = np.full(m, np.nan, dtype=float)
    for j in range(m):
        uq = np.asarray(UQ[j], dtype=float).ravel()[:n]
        uq = maybe_align_global_sign(uq, uc)
        e = uq - uc
        balanced_l2[j] = (
            np.linalg.norm(sqrtD * e) /
            max(np.linalg.norm(y), 1e-300)
        )

    valid = np.isfinite(it) & np.isfinite(F) & np.isfinite(balanced_l2)

    return {
        "family": family,
        "archive_path": str(archive),
        "iter": it[valid],
        "F": F[valid],
        "balanced_l2": balanced_l2[valid],
        "n_points": int(np.count_nonzero(valid)),
        "iter_min": float(np.min(it[valid])) if np.any(valid) else np.nan,
        "iter_max": float(np.max(it[valid])) if np.any(valid) else np.nan,
    }


def plot_fig4_noisy_heat_convergence(
    trajectories: Dict[str, Dict],
    outdir: Path,
):
    """
    Heat1D vs Heat2D under the same noisy-none optimization budget.
    Left: fidelity vs iteration.
    Right: unit-invariant balanced L2 vs iteration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)

    for fam in ("heat1d", "heat2d"):
        d = trajectories[fam]

        axes[0].plot(
            d["iter"], d["F"],
            lw=1.5, marker="o", ms=2.8,
            label=PRETTY[fam],
        )
        axes[1].plot(
            d["iter"], d["balanced_l2"],
            lw=1.5, marker="o", ms=2.8,
            label=PRETTY[fam],
        )

    axes[0].set_xlabel("Optimization iteration")
    axes[0].set_ylabel("Fidelity")
    axes[0].set_title("Noisy-none fidelity convergence")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Optimization iteration")
    axes[1].set_ylabel(r"Balanced relative $L_2$")
    axes[1].set_title(r"Noisy-none field error in $D^{1/2}$ coordinates")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Heat1D vs Heat2D under the same noisy optimization budget (seed 1001)",
        fontsize=14,
    )

    png = outdir / "fig4_heat1d_heat2d_noisy_convergence.png"
    pdf = outdir / "fig4_heat1d_heat2d_noisy_convergence.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


# -----------------------------------------------------------------------------
# Output tables / manifest
# -----------------------------------------------------------------------------

def write_master_csv(
    physical: Dict[str, Dict],
    ideal: Dict[str, Dict],
    noisy: Dict[str, Dict],
    outdir: Path,
):
    rows = []

    for fam in FAMILIES:
        p, i, n = physical[fam], ideal[fam], noisy[fam]

        rows.append({
            "family": fam,

            # Auxiliary raw diagnostics
            "raw_kappa_K_aux": p["raw_kappa_K"],
            "raw_L2_aux": n["raw_l2_aux"],
            "raw_physical_residual_aux": n["raw_rphys_aux"],

            # Unit-invariant physical diagnostics
            "kappa_equilibrated_K": p["kappa_eq"],
            "forcing_PR_grouped_eq": p["forcing_PR_grouped"],
            "forcing_r99_grouped_eq": p["forcing_r99_grouped"],
            "solution_PR_grouped_eq": p["solution_PR_grouped"],
            "solution_r99_grouped_eq": p["solution_r99_grouped"],
            "num_distinct_eigenspaces_eq": p["num_eigenvalue_groups"],
            "max_eigenspace_multiplicity_eq": p["max_group_multiplicity"],

            # Actual preconditioned VQLS target
            "kappa_pre_from_A": i["kappa_pre_from_A"],
            "kappa_pre_manifest": i["kappa_pre_manifest"],
            "xpre_PR_grouped_A": i["xpre_PR_grouped"],
            "xpre_r99_grouped_A": i["xpre_r99_grouped"],
            "ideal_d99": i["depth_F_0p99"],
            "ideal_d999": i["depth_F_0p999"],
            "ideal_best_F": i["best_F"],
            "ideal_unique_seed_count_detected": i["unique_seed_count"],
            "ideal_unique_restart_count_detected": i["unique_restart_count"],

            # Noisy final-state metrics
            "F_csv": n["F_csv"],
            "l2_fis_csv_aux": n["l2_csv"],
            "balanced_L2_recomputed": n["balanced_l2_recomputed"],
            "res_pre_csv": n["res_pre_csv"],
            "eK_rel_csv": n["eK_rel_csv"],
            "energy_norm_error_recomputed": n["energy_rel_recomputed"],
            "equilibrated_physical_residual": n["residual_eq_recomputed"],
            "balanced_L2_PR_grouped_eq": n["balanced_l2_PR_grouped"],
            "energy_error_PR_grouped_eq": n["energy_error_PR_grouped"],
            "equilibrated_residual_PR_grouped_eq":
                n["equilibrated_residual_PR_grouped"],
            "balanced_L2_r99_grouped_eq": n["balanced_l2_r99_grouped"],
            "energy_error_r99_grouped_eq": n["energy_error_r99_grouped"],
            "equilibrated_residual_r99_grouped_eq":
                n["equilibrated_residual_r99_grouped"],
            "stiff_half_equilibrated_residual_fraction":
                n["stiff_half_residual_fraction"],

            # Consistency checks
            "archive_u_fem_vs_static_u_c_rel":
                n["archive_ufem_vs_static_uc_delta"],
            "archive_last_vector_vs_csv_l2_delta":
                n["archive_vs_csv_l2_delta"],
            "recomputed_energy_vs_csv_eK_delta":
                n["recomputed_energy_vs_csv_delta"],
            "equilibrated_solution_reconstruction_mismatch":
                p["equilibrated_reconstruction_mismatch"],
            "A_xpre_relative_residual":
                i["A_xpre_relative_residual"],
        })

    df = pd.DataFrame(rows)
    path = outdir / "physics_dependency_master_v4.csv"
    df.to_csv(path, index=False)
    return path, df


def serializable(obj):
    if isinstance(obj, dict):
        return {k: serializable(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_manifest(physical, ideal, noisy, args, outdir):
    data = {
        "analysis_version": "v4_unit_invariant_grouped_eigenspaces",
        "choices": {
            "physical_diagnostic_coordinates":
                "Jacobi equilibrated: K_eq=D^-1/2 K D^-1/2",
            "balanced_L2_metric":
                "||D^1/2 e|| / ||D^1/2 u||",
            "primary_error_metric":
                "relative energy norm sqrt(e^T K e / u^T K u)",
            "primary_physical_residual":
                "||D^-1/2(Ku_q-f)|| / ||D^-1/2 f||",
            "noisy_vector":
                "dof32_u_q_fis[-1] (audited to reproduce final l2_fis)",
            "eigenspace_group_rtol": args.eig_group_rtol,
            "forcing_zero_weight_tol_relative": args.zero_weight_tol,
            "ideal_infidelity_plot_floor": args.infidelity_floor,
            "ideal_preconditioner": args.preconditioner,
        },
        "interpretation_guardrails": [
            "Raw kappa(K), raw L2, and raw physical residual are auxiliary only for cross-family structural comparisons.",
            "Grouped eigenspace weights are used when eigenvalues are repeated.",
            "Modal concentration does not prove ansatz representability; the ideal depth study is the direct test.",
            "The depth figure uses best available depth-summary values and is not a seed uncertainty band unless multiple independent seeds are explicitly present.",
        ],
        "physical": {f: serializable(d) for f, d in physical.items()},
        "ideal": {f: serializable(d) for f, d in ideal.items()},
        "noisy": {f: serializable(d) for f, d in noisy.items()},
    }

    path = outdir / "physics_dependency_manifest_v4.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# -----------------------------------------------------------------------------
# Preflight / main
# -----------------------------------------------------------------------------

def preflight(pc: str):
    print("=" * 112)
    print("PREFLIGHT — V4 UNIT-INVARIANT ANALYSIS")
    print("=" * 112)

    ok = True
    for fam in FAMILIES:
        entries = [
            ("K", PHYSICAL_K[fam]),
            ("u_c", CLASSICAL_SOL[fam]),
            ("archive", NOISY_ARCHIVE[fam]),
            ("metrics", NOISY_METRICS[fam]),
        ]
        for label, p in entries:
            exists = p.exists()
            ok &= exists
            print(f"{'OK' if exists else 'MISSING':8s} {fam:8s} {label:9s} {p}")

    try:
        root = resolve_ideal_run()
        print(f"OK       ideal root          {root}")
        for fam in FAMILIES:
            try:
                d = find_family_pc_dir(root, fam, pc)
                for fn in ("depth_summary.csv", "frozen_system.npz"):
                    p = d / fn
                    exists = p.exists()
                    ok &= exists
                    print(f"{'OK' if exists else 'MISSING':8s} {fam:8s} ideal/{pc:8s} {p}")
            except Exception as exc:
                ok = False
                print(f"MISSING  {fam:8s} ideal/{pc:8s} {exc}")
    except Exception as exc:
        ok = False
        print(f"MISSING  ideal root          {exc}")

    print("=" * 112)
    print("PREFLIGHT RESULT:", "PASS" if ok else "FAIL")
    print("=" * 112)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preconditioner", default=DEFAULT_PRECONDITIONER)
    ap.add_argument(
        "--outdir",
        default=str(HOME / "physics_dependency_final_v4"),
    )
    ap.add_argument(
        "--eig-group-rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for grouping nearly repeated eigenvalues.",
    )
    ap.add_argument(
        "--zero-weight-tol",
        type=float,
        default=1e-24,
        help="Relative threshold for treating grouped forcing weights as zero.",
    )
    ap.add_argument(
        "--infidelity-floor",
        type=float,
        default=1e-6,
        help="Plotting floor for ideal infidelity; lower tails are clipped.",
    )
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    passed = preflight(args.preconditioner)
    if args.preflight_only:
        return
    if not passed:
        raise SystemExit("Preflight failed. Fix missing inputs before analysis.")

    print("\n" + "=" * 112)
    print("1) UNIT-INVARIANT PHYSICAL / EIGENSPACE ANALYSIS")
    print("=" * 112)

    physical = {}
    for fam in FAMILIES:
        d = physical_modal_analysis(
            fam, args.eig_group_rtol, args.zero_weight_tol
        )
        physical[fam] = d
        print(
            f"{PRETTY[fam]:7s} "
            f"kappa_raw(aux)={d['raw_kappa_K']:.4e}  "
            f"kappa_eq={d['kappa_eq']:.4e}  "
            f"groups={d['num_eigenvalue_groups']:2d}  "
            f"max_mult={d['max_group_multiplicity']:2d}  "
            f"PRf={d['forcing_PR_grouped']:.3f} r99f={d['forcing_r99_grouped']:2d}  "
            f"PRu={d['solution_PR_grouped']:.3f} r99u={d['solution_r99_grouped']:2d}"
        )

    print("\n" + "=" * 112)
    print("2) IDEAL DEPTH + ACTUAL PRECONDITIONED TARGET")
    print("=" * 112)

    ideal = {}
    for fam in FAMILIES:
        d = load_ideal_analysis(
            fam, args.preconditioner, args.eig_group_rtol
        )
        ideal[fam] = d
        seed_note = (
            str(d["unique_seed_count"])
            if np.isfinite(d["unique_seed_count"]) else "unknown"
        )
        print(
            f"{PRETTY[fam]:7s} "
            f"kappa_pre(A)={d['kappa_pre_from_A']:.4g}  "
            f"PRx={d['xpre_PR_grouped']:.3f} r99x={d['xpre_r99_grouped']:2d}  "
            f"d99={d['depth_F_0p99']} d999={d['depth_F_0p999']}  "
            f"ideal_seeds_detected={seed_note}"
        )

    print("\n" + "=" * 112)
    print("3) FINAL NOISY ERROR — ENERGY NORM + EQUILIBRATED PHYSICAL RESIDUAL")
    print("=" * 112)

    noisy = {}
    for fam in FAMILIES:
        d = noisy_error_analysis(
            fam, physical[fam], args.eig_group_rtol
        )
        noisy[fam] = d
        print(
            f"{PRETTY[fam]:7s} "
            f"L2eq={d['balanced_l2_recomputed']:.6g}  "
            f"eK(recomputed)={d['energy_rel_recomputed']:.6g}  "
            f"eK(csv)={d['eK_rel_csv']:.6g}  "
            f"r_eq={d['residual_eq_recomputed']:.6g}  "
            f"r_pre={d['res_pre_csv']:.6g}  "
            f"rawL2(aux)={d['raw_l2_aux']:.6g}  "
            f"rawRphys(aux)={d['raw_rphys_aux']:.6g}"
        )
        print(
            f"          consistency: ΔL2={d['archive_vs_csv_l2_delta']:.3e}, "
            f"ΔeK={d['recomputed_energy_vs_csv_delta']:.3e}, "
            f"uFEM/archive={d['archive_ufem_vs_static_uc_delta']:.3e}, "
            f"stiff-half residual fraction={d['stiff_half_residual_fraction']:.3f}"
        )

    print("\n" + "=" * 112)
    print("4) WRITING FIGURES / TABLES")
    print("=" * 112)

    trajectories = {
        fam: noisy_trajectory_analysis(fam, physical[fam])
        for fam in ("heat1d", "heat2d")
    }
    print(
        f"Heat1D noisy trajectory: {trajectories['heat1d']['n_points']} points, "
        f"iterations {trajectories['heat1d']['iter_min']:g}..{trajectories['heat1d']['iter_max']:g}"
    )
    print(
        f"Heat2D noisy trajectory: {trajectories['heat2d']['n_points']} points, "
        f"iterations {trajectories['heat2d']['iter_min']:g}..{trajectories['heat2d']['iter_max']:g}"
    )

    f1 = plot_fig1(physical, outdir)
    f2 = plot_fig2(
        ideal, args.preconditioner, outdir, args.infidelity_floor
    )
    f3 = plot_fig3(noisy, outdir)
    f4 = plot_fig4_noisy_heat_convergence(trajectories, outdir)
    csv_path, master = write_master_csv(physical, ideal, noisy, outdir)
    manifest_path = write_manifest(physical, ideal, noisy, args, outdir)

    print("Figure 1:", f1[0])
    print("          ", f1[1])
    print("Figure 2:", f2[0])
    print("          ", f2[1])
    print("Figure 3:", f3[0])
    print("          ", f3[1])
    print("Figure 4:", f4[0])
    print("          ", f4[1])
    print("Master:  ", csv_path)
    print("Manifest:", manifest_path)

    print("\nMASTER SUMMARY")
    cols = [
        "family",
        "kappa_equilibrated_K",
        "solution_PR_grouped_eq",
        "solution_r99_grouped_eq",
        "kappa_pre_from_A",
        "xpre_PR_grouped_A",
        "xpre_r99_grouped_A",
        "ideal_d99",
        "balanced_L2_recomputed",
        "energy_norm_error_recomputed",
        "equilibrated_physical_residual",
        "res_pre_csv",
    ]
    print(master[cols].to_string(index=False))

    print("\nINTERPRETATION GUARDRAILS")
    print(" - Use kappa_eq, balanced L2, eK, and r_eq for cross-family physical claims.")
    print(" - Treat raw kappa(K), raw L2, and raw r_phys as auxiliary diagnostics only.")
    print(" - Grouped eigenspaces remove arbitrary basis dependence from repeated eigenvalues.")
    print(" - x_pre/A analysis is the bridge to what VQLS actually sees after preconditioning.")
    print(" - d99/d999 are representability summaries; do not claim seed-level robustness without independent seeds.")


if __name__ == "__main__":
    main()
