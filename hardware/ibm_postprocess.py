#!/usr/bin/env python3
"""Turn IBM EstimatorV2 result JSON files into auditable VQLS-cost summaries."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from hardware_common import (build_problem, input_paths, local_cost_and_sem,
                             read_json, write_json)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="Directory containing raw_rep*.json, trex_rep*.json, etc.")
    ap.add_argument("--out", default="hardware_cost_summary.json", type=Path)
    return ap.parse_args()


def main():
    args = parse_args()
    paths = input_paths(args.input_dir)
    manifest = read_json(paths["manifest"])
    reference = read_json(paths["reference"])
    _nisq, _sysd, _obs, combine, _mats, _counts, _qc, _pv = build_problem(
        manifest, args.project_root, args.input_dir)

    records = []
    for path in sorted(args.results_dir.glob("*_rep*.json")):
        row = read_json(path)
        if row.get("status") != "completed":
            continue
        evs = np.asarray(row["estimate"]["evs"], dtype=float)
        stds = np.asarray(row["estimate"]["stds"], dtype=float)
        cost, sem = local_cost_and_sem(combine, evs, stds)
        records.append({
            "file": str(path), "mode": row["mode"], "rep": row["rep"],
            "job_id": row.get("job_id"), "C_local": cost,
            "C_local_delta_sem": sem,
            "H_L": float(evs[0]), "A2": float(evs[1]),
            "H_L_sem": float(stds[0]), "A2_sem": float(stds[1]),
            "shots_metadata": row.get("pub_metadata", {}).get("shots"),
        })

    grouped = defaultdict(list)
    for row in records:
        grouped[row["mode"]].append(row["C_local"])
    summary = {}
    exact = float(reference["C_local_exact"])
    for mode, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        summary[mode] = {
            "n_repetitions": int(arr.size), "mean_C_local": float(arr.mean()),
            "sem_between_repetitions": float(arr.std(ddof=1) / np.sqrt(arr.size))
            if arr.size > 1 else None,
            "bias_vs_exact_theta_cost": float(arr.mean() - exact),
        }
    payload = {
        "manifest": str(paths["manifest"]),
        "offline_reference": reference,
        "hardware_records": records,
        "summary_by_mode": summary,
        "interpretation": (
            "C_local is measured on IBM. Fidelity, residual, L2, energy and QoI "
            "inside offline_reference describe the ideal state at the executed theta "
            "and are not direct hardware measurements."),
    }
    out = args.out if args.out.is_absolute() else args.results_dir / args.out
    write_json(out, payload)
    print(f"Saved: {out}")
    for mode, row in summary.items():
        print(f"{mode:9s} C_L={row['mean_C_local']:.6e}; "
              f"bias={row['bias_vs_exact_theta_cost']:+.3e}; n={row['n_repetitions']}")


if __name__ == "__main__":
    main()
