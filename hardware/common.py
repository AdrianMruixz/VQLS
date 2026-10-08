"""Shared, simulator-free utilities for fixed-theta IBM VQLS validation.

This module deliberately reuses only the mathematical constructors in the
project's ``nisq.py``. It never creates an Aer simulator or submits jobs.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector


SCHEMA_VERSION = 2

# These are the complete numerical inputs needed by the VQLS constructions and
# offline metrics.  They are frozen because pyamg's setup is not bitwise
# deterministic across independent calls to ``nisq.prepare``.
_FROZEN_ARRAY_KEYS = (
    "A", "b", "x_pre", "x_phys", "K_pad", "f_pad", "Dv", "Msqrt_inv",
)
_FROZEN_SCALAR_KEYS = (
    "nq", "n_orig", "N", "spd_ok", "kappa_raw", "kappa_pre", "s", "nb",
    "a_norm", "b_norm",
)


def jsonable(value: Any) -> Any:
    """Convert NumPy/Qiskit-adjacent data to JSON-safe Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def import_project_nisq(project_root: Path, hook_module: str | None = None):
    """Import the exact project copy of nisq.py and an optional AMG hook."""
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    if hook_module:
        importlib.import_module(hook_module)
    if "nisq" in sys.modules:
        del sys.modules["nisq"]
    return importlib.import_module("nisq")


def save_frozen_problem(path: Path, sysd: dict[str, Any]) -> None:
    """Persist the one preconditioned/normalized system used for optimization."""
    payload: dict[str, np.ndarray] = {}
    for key in _FROZEN_ARRAY_KEYS:
        value = sysd.get(key)
        payload[key] = (np.empty((0, 0), dtype=float) if value is None
                        else np.asarray(value, dtype=float))
    for key in _FROZEN_SCALAR_KEYS:
        payload[key] = np.asarray(sysd[key])
    payload["precond"] = np.asarray(str(sysd["precond"]))
    payload["meta_json"] = np.asarray(json.dumps(jsonable(sysd.get("meta", {}))))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_frozen_problem(path: Path) -> dict[str, Any]:
    """Load a frozen VQLS system without calling ``nisq.prepare``."""
    if not path.is_file():
        raise FileNotFoundError(f"Frozen numerical problem is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = set(_FROZEN_ARRAY_KEYS + _FROZEN_SCALAR_KEYS +
                       ("precond", "meta_json"))
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"Frozen problem lacks fields: {sorted(missing)}")
        sysd = {key: np.asarray(archive[key], dtype=float)
                for key in _FROZEN_ARRAY_KEYS}
        for key in _FROZEN_SCALAR_KEYS:
            value = archive[key].item()
            sysd[key] = int(value) if key in {"nq", "n_orig", "N", "nb"} else value
        sysd["Msqrt_inv"] = (None if sysd["Msqrt_inv"].size == 0
                              else sysd["Msqrt_inv"])
        sysd["precond"] = str(archive["precond"].item())
        sysd["meta"] = json.loads(str(archive["meta_json"].item()))
    return sysd


def build_problem(manifest: dict[str, Any], project_root: Path, input_dir: Path):
    """Build the circuit/cost from the frozen operator in an input directory."""
    cfg = manifest["configuration"]
    nisq = import_project_nisq(project_root, cfg.get("hook_module") or None)
    frozen = input_dir / manifest["frozen_problem"]["file"]
    if sha256_file(frozen) != manifest["frozen_problem"]["sha256"]:
        raise RuntimeError("Frozen A,b archive hash differs from manifest; do not proceed.")
    sysd = load_frozen_problem(frozen)
    obs, combine, mats, counts = nisq.make_cost_observables(
        sysd["A"], sysd["b"], sysd["nq"],
        atol=float(cfg["pauli_atol"]), cost_type=cfg["cost_type"],
    )
    qc, pv = nisq.build_ansatz(sysd["nq"], int(cfg["depth"]), cfg["entangler"])
    return nisq, sysd, obs, combine, mats, counts, qc, pv


def exact_expectations(qc, theta: np.ndarray, mats: list[np.ndarray]) -> np.ndarray:
    """Exact expectation values of dense cost matrices at a real ansatz theta."""
    state = Statevector(qc.assign_parameters(np.asarray(theta, dtype=float))).data
    if np.max(np.abs(np.imag(state))) > 1e-10:
        raise ValueError("The ansatz state is unexpectedly complex; do not discard phases.")
    state = np.real(state)
    return np.asarray([float(state @ matrix @ state) for matrix in mats], dtype=float)


def validate_theta(theta: np.ndarray, n_parameters: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.size != n_parameters:
        raise ValueError(
            f"Theta has {theta.size} entries but this circuit requires {n_parameters}."
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError("Theta contains NaN or infinity.")
    return theta


def local_cost_and_sem(combine, evs: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    """Compute the local-cost ratio and a first-order independent-error SEM."""
    evs = np.asarray(evs, dtype=float).reshape(-1)
    stds = np.asarray(stds, dtype=float).reshape(-1)
    if evs.size != 2 or stds.size != 2:
        raise ValueError("This fixed-theta workflow expects [<H_L>, <A^2>].")
    cost = float(combine(evs))
    numerator, denominator = evs
    if abs(denominator) < 1e-14:
        raise ZeroDivisionError("Measured <A^2> is too small for a stable VQLS ratio.")
    sem = float(np.sqrt((stds[0] / denominator) ** 2 +
                        (numerator * stds[1] / denominator ** 2) ** 2))
    return cost, sem


def backend_name(backend) -> str:
    name = getattr(backend, "name", None)
    return name() if callable(name) else str(name)


def backend_snapshot(backend) -> dict[str, Any]:
    status = backend.status()
    status_fields = {}
    for field in ("operational", "pending_jobs", "status_msg"):
        if hasattr(status, field):
            status_fields[field] = jsonable(getattr(status, field))
    return {
        "name": backend_name(backend),
        "num_qubits": int(getattr(backend, "num_qubits", -1)),
        "status": status_fields or repr(status),
        "basis_gates": list(getattr(backend, "operation_names", [])),
    }


def input_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "manifest": input_dir / "manifest.json",
        "theta": input_dir / "theta_ideal.npy",
        "reference": input_dir / "ideal_reference.json",
        "frozen": input_dir / "frozen_problem.npz",
    }
