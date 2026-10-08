#!/usr/bin/env python3
"""CLI: run figures from the repo.

Usage:
    python scripts/make_figs.py --fig all
    python scripts/make_figs.py --fig trajectories
    python scripts/make_figs.py --fig costs --outdir ~/figs
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "figs"

FIGURES = {
    "trajectories":   "trajectories.py",
    "dimensionality": "dimensionality.py",
    "physics":        "physics.py",
    "costs":          "costs.py",
    "ibm":            "ibm.py",
    "ideal":          "ideal.py",
    "entropy":        "entropy.py",
}


def run_fig(name, outdir):
    script = FIGS / FIGURES[name]
    print(f"\n{'='*60}")
    print(f"  Running: {name}  ({script.name})")
    print(f"{'='*60}")
    # try to pass --outdir; if the script doesn't accept it, run without
    try:
        subprocess.run([sys.executable, str(script), "--outdir", str(outdir)], check=True)
    except subprocess.CalledProcessError:
        print(f"  [warn] {name} failed with --outdir, retrying without")
        subprocess.run([sys.executable, str(script)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="all", choices=["all"] + list(FIGURES.keys()))
    ap.add_argument("--outdir", default=str(Path.home() / "figs"))
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    todo = list(FIGURES.keys()) if args.fig == "all" else [args.fig]
    for name in todo:
        run_fig(name, outdir)

    print(f"\n{'='*60}")
    print(f"  DONE  figures in {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
