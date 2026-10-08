"""
Central config: paths, whitelist, constants.

Data location resolution (in order):
  1. Environment variable VQLS_DATA
  2. Falls back to $HOME

The repo contains ONLY code. Data files (K matrices, sol files,
metrics CSVs, QEM JSONs, IBM results) live outside the repo.
"""
from __future__ import annotations
import os
import glob
from pathlib import Path

# ---------- data root ----------
DATA = Path(os.environ.get("VQLS_DATA", Path.home())).resolve()

# ---------- data directories (external) ----------
PATHS = {
    "heat1d":      DATA / "Heat1D",
    "heat2d":      DATA / "Heat2D",
    "beam1d":      DATA / "beam1d_final_results",
    "beam2d":      DATA / "Beam 2D",
    "campaign":    DATA / "other_families_trex_campaign",
    "beam1d_opt":  DATA / "beam1d_optimized",
    "ideal":       DATA / "ideal_preconditioning_32dof",
    "ibm":         DATA / "ibm_fixed_theta_vqls",
}

# ---------- families ----------
FAMS = ["heat1d", "heat2d", "beam1d", "beam2d"]
FAM_PRETTY = {"heat1d": "Heat1D", "heat2d": "Heat2D",
              "beam1d": "Beam1D", "beam2d": "Beam2D"}

# ---------- file lookup ----------
def K_file(fam: str, dof: int) -> Path:
    return PATHS[fam] / f"{fam}_{dof}DOF_K.npz"

def sol_file(fam: str, dof: int, kind: str = "none") -> Path:
    """kind in {none, ddd, zne, ideal}."""
    if kind == "ideal":
        return PATHS[fam] / f"sol_ideal_{fam}_{dof}.npz"
    return PATHS[fam] / f"sol_{fam}_{dof}_{kind}.npz"

def frozen_file(fam: str, precond: str, dof: int = 32):
    pat = str(PATHS["ideal"] / "results_full_*" / fam / precond / "frozen_system.npz")
    hits = sorted(glob.glob(pat))
    return Path(hits[0]) if hits else None

# ---------- campaigns whitelist ----------
WL_NONE = {
    "raw_none_seed1001_20260923_124835",
    "raw_none_seed3003_20260923_122930",
    "none_seed4001_20260928_193817",
}
WL_TREX = {
    "trex_only_final_20260921_114421",
    "trex_seed3003_20261002_170940",
    "trex_seed4001_20261002_170940",
}
WL_B1D = {
    "beam1d_paired_final_20260922_184543",
    "beam1d_full_seed8203_20260930_111027",
    "beam1d_full_seed8303_20261001_012345",
}
WL_ALL = WL_NONE | WL_TREX | WL_B1D

# ---------- seeds ----------
SEEDS = {
    "heat1d": {"none": [1001, 3003, 4001], "trex": [1001, 3003, 4001]},
    "heat2d": {"none": [1001, 3003, 4001], "trex": [1001, 3003, 4001]},
    "beam2d": {"none": [1001, 3003, 4001], "trex": [1001, 3003, 4001]},
    "beam1d": {"none": [8103, 8203, 8303], "trex": [8103, 8203, 8303]},
}

# ---------- preconditioners ----------
PRECONDS = ["none", "jacobi", "ichol", "amg"]

# ---------- constants ----------
DOFS   = [8, 16, 32]
SHOTS  = 1024
REPS   = 8
N_CAL  = 24
LEVELS = 3

def nq(dof: int) -> int:
    return dof.bit_length() - 1
