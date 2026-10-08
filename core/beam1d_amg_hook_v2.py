"""Validated Beam1D SA-AMG hook.

Import this module *before* calling ``nisq.prepare(..., precond="amg")``.
It replaces the default AMG constructor only for the current Python process.

The Beam1D cantilever is SPD.  Its relevant algebraically smooth error space
is therefore the cubic Euler--Bernoulli polynomial space, not rigid modes.
"""
from __future__ import annotations

import numpy as np
import pyamg
import scipy.sparse as sp

import nisq


def _beam1d_cubic_basis(n: int, Dv=None, meta=None) -> np.ndarray:
    """Return physical cubic modes mapped to ``prepare``'s AMG coordinates."""
    if n % 2:
        raise ValueError(f"beam1d needs w/theta pairs, received n={n}")
    L = float((meta or {}).get("L", 1.0))
    n_el = n // 2
    xi = np.arange(1, n_el + 1, dtype=float) / n_el
    B = np.zeros((n, 4), dtype=float)
    B[0::2, 0] = 1.0
    B[0::2, 1] = xi
    B[1::2, 1] = 1.0 / L
    B[0::2, 2] = xi**2
    B[1::2, 2] = 2.0 * xi / L
    B[0::2, 3] = xi**3
    B[1::2, 3] = 3.0 * xi**2 / L

    # ``prepare`` first maps u -> y=D^{-1}u.  Supply smooth vectors in y.
    if Dv is not None:
        Dv = np.asarray(Dv, dtype=float).ravel()
        if Dv.size != n or not np.all(np.isfinite(Dv)) or np.any(Dv <= 0):
            raise ValueError("invalid Jacobi-equilibration vector Dv")
        B /= Dv[:, None]
    B /= np.maximum(np.linalg.norm(B, axis=0), 1.0e-300)[None, :]
    return B


def _default_basis(n: int, family: str, Dv=None) -> np.ndarray:
    """Keep the standard smooth spaces for non-Beam1D families."""
    if family == "beam2d":
        B = np.zeros((n, 2), dtype=float)
        B[0::2, 0] = 1.0
        B[1::2, 1] = 1.0
    else:
        B = np.ones((n, 1), dtype=float)
    if Dv is not None:
        Dv = np.asarray(Dv, dtype=float).ravel()
        if Dv.size != n or np.any(Dv <= 0):
            raise ValueError("invalid Jacobi-equilibration vector Dv")
        B /= Dv[:, None]
    B /= np.maximum(np.linalg.norm(B, axis=0), 1.0e-300)[None, :]
    return B


def improved_amg_v2(A, family: str, Dv=None, meta=None) -> np.ndarray:
    """One fixed symmetric SA-AMG V-cycle, returned as a dense SPD inverse."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("AMG requires a square matrix")
    B = (_beam1d_cubic_basis(n, Dv=Dv, meta=meta) if family == "beam1d"
         else _default_basis(n, family, Dv=Dv))
    config = dict(
        smooth="energy",
        presmoother=("gauss_seidel", {"sweep": "symmetric", "iterations": 2}),
        postsmoother=("gauss_seidel", {"sweep": "symmetric", "iterations": 2}),
        max_levels=15,
        max_coarse=10,
        coarse_solver="pinv",
    )
    try:
        ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(A), B=B, **config)
        P = ml.aspreconditioner(cycle="V")
        Minv = np.column_stack([P.matvec(e) for e in np.eye(n)])
    except Exception as exc:
        raise RuntimeError(f"Beam1D SA-AMG v2 failed for n={n}") from exc
    Minv = 0.5 * (Minv + Minv.T)
    eig = np.linalg.eigvalsh(Minv)
    if eig[0] <= 0.0:
        Minv += (abs(eig[0]) + 1.0e-12 * max(eig[-1], 1.0)) * np.eye(n)
    if not np.all(np.isfinite(Minv)) or np.linalg.eigvalsh(Minv)[0] <= 0.0:
        raise RuntimeError("SA-AMG v2 did not produce an SPD inverse")
    S = nisq._sym_sqrt(Minv)
    Ap = 0.5 * (S @ A @ S + (S @ A @ S).T)
    lam = np.linalg.eigvalsh(Ap)
    print(f"[beam1d hook_v2] n={n}, B={B.shape[1]}, levels={len(ml.levels)}, "
          f"kappa_pre={lam[-1] / lam[0]:.6g}", flush=True)
    return Minv


nisq._amg_inverse_operator = improved_amg_v2
print("[beam1d hook_v2] cubic near-kernel installed", flush=True)
