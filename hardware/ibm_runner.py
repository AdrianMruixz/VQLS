#!/usr/bin/env python3
"""Submit one fixed-theta VQLS cost evaluation to IBM EstimatorV2.

Modes are deliberately explicit: raw, TREX, and TREX+ZNE. No SPSA or Adam
appears here: optimization is offline and theta is frozen before submission.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
from qiskit_ibm_runtime.options import EstimatorOptions

from hardware_common import (backend_name, backend_snapshot, build_problem,
                             input_paths, jsonable, read_json, sha256_file,
                             validate_theta, write_json)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--mode", required=True, choices=["raw", "trex", "trex_zne"])
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--trex-randomizations", type=int, default=16)
    ap.add_argument("--trex-shots-per-randomization", type=int, default=64)
    ap.add_argument("--zne-factors", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    ap.add_argument("--zne-extrapolator", default="exponential")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--output-dir", default="ibm_results", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def options_for(mode: str, args) -> EstimatorOptions:
    """Set all mitigation/suppression options explicitly; never use defaults."""
    options = EstimatorOptions()
    options.default_shots = int(args.shots)
    options.resilience_level = 0
    options.dynamical_decoupling.enable = False
    options.twirling.enable_gates = False
    options.twirling.enable_measure = False
    options.resilience.measure_mitigation = False
    options.resilience.zne_mitigation = False

    if mode in ("trex", "trex_zne"):
        # IBM Runtime's measurement-mitigation implementation of TREX.
        options.twirling.enable_measure = True
        options.resilience.measure_mitigation = True
        options.resilience.measure_noise_learning.num_randomizations = int(args.trex_randomizations)
        options.resilience.measure_noise_learning.shots_per_randomization = int(
            args.trex_shots_per_randomization)

    if mode == "trex_zne":
        options.resilience.zne_mitigation = True
        options.resilience.zne.amplifier = "gate_folding"
        options.resilience.zne.noise_factors = list(map(float, args.zne_factors))
        options.resilience.zne.extrapolator = args.zne_extrapolator
    return options


def public_data(pub_data):
    """Persist every public result field, including ZNE diagnostic arrays."""
    out = {}
    for name in dir(pub_data):
        if name.startswith("_"):
            continue
        try:
            value = getattr(pub_data, name)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            out[name] = jsonable(value)
        except TypeError:
            out[name] = repr(value)
    return out


def main():
    args = parse_args()
    if args.shots < 128:
        raise ValueError("Use at least 128 shots per physical measurement configuration.")
    if args.mode == "trex_zne" and args.zne_factors != sorted(args.zne_factors):
        raise ValueError("ZNE factors must be increasing.")

    paths = input_paths(args.input_dir)
    manifest = read_json(paths["manifest"])
    theta = validate_theta(np.load(paths["theta"]), manifest["theta"]["n_parameters"])
    _nisq, _sysd, obs, _combine, _mats, _counts, qc, _pv = build_problem(
        manifest, args.project_root, args.input_dir)

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(qc)
    isa_obs = [operator.apply_layout(isa_circuit.layout) for operator in obs]
    if len(isa_obs) != 2:
        raise RuntimeError("Expected exactly [H_L, A^2] observables for local VQLS cost.")

    result_dir = args.output_dir / manifest["configuration"]["family"] / backend_name(backend)
    result_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = result_dir / f"{args.mode}_rep{args.rep}_{stamp}.json"

    record = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "submitted",
        "timestamp_utc": stamp,
        "mode": args.mode,
        "rep": int(args.rep),
        "backend": backend_snapshot(backend),
        "input_dir": str(args.input_dir.resolve()),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "theta_sha256": manifest["theta"]["sha256"],
        "theta_parameters": int(theta.size),
        "shots_requested": int(args.shots),
        "runtime_options": {
            "raw": args.mode == "raw",
            "measure_mitigation_TREX": args.mode in ("trex", "trex_zne"),
            "zne": args.mode == "trex_zne",
            "zne_factors": args.zne_factors if args.mode == "trex_zne" else [],
            "zne_extrapolator": args.zne_extrapolator if args.mode == "trex_zne" else None,
            "dd": False, "gate_twirling": False,
        },
        "isa": {"depth": isa_circuit.depth(), "count_ops": dict(isa_circuit.count_ops()),
                "layout": str(isa_circuit.layout), "observable_terms": [len(o) for o in isa_obs]},
        "software": {"python": platform.python_version(),
                     "qiskit": importlib.metadata.version("qiskit"),
                     "qiskit_ibm_runtime": importlib.metadata.version("qiskit-ibm-runtime")},
    }
    if args.dry_run:
        write_json(result_path, record)
        print(f"DRY RUN PASSED: {result_path}")
        return

    estimator = EstimatorV2(mode=backend, options=options_for(args.mode, args))
    # One PUB, one logical state, and the two observables required by C_L.
    job = estimator.run([(isa_circuit, isa_obs, [theta.tolist()])])
    record["job_id"] = job.job_id()
    write_json(result_path, record)  # Persist the job ID before waiting.
    print(f"SUBMITTED: job_id={record['job_id']}", flush=True)

    primitive_result = job.result()
    pub = primitive_result[0]
    evs = np.asarray(pub.data.evs, dtype=float).reshape(-1)
    stds = np.asarray(pub.data.stds, dtype=float).reshape(-1)
    if evs.size != 2 or stds.size != 2:
        raise RuntimeError(f"Expected 2 observables, received evs={evs.shape}, stds={stds.shape}.")
    record.update({
        "status": "completed",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "estimate": {"field": "evs", "evs": evs, "stds": stds},
        "pub_metadata": jsonable(pub.metadata),
        "primitive_metadata": jsonable(getattr(primitive_result, "metadata", {})),
        "all_pub_data": public_data(pub.data),
    })
    write_json(result_path, record)
    print(f"COMPLETED: {result_path}")


if __name__ == "__main__":
    main()
