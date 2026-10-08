#!/usr/bin/env python3
"""Entanglement entropy of x_pre  strict recipe."""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 6.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.linewidth": 0.8, "axes.grid": False, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 3.0,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "xtick.direction": "in", "ytick.direction": "in",
})

NQ = 5
N_DOF = 2 ** NQ
ROOT = Path.home() / "ideal_preconditioning_32dof"
PC = "amg"

FAM_COLORS = {"heat1d": "#0072B2", "heat2d": "#E69F00",
              "beam1d": "#D55E00", "beam2d": "#009E73"}
FAM_PRETTY = {"heat1d": "Heat1D", "heat2d": "Heat2D",
              "beam1d": "Beam1D", "beam2d": "Beam2D"}

CTRL = {"heat1d": 0.094, "heat2d": 0.776}
TOL = 0.02

def load_xpre(fam, pc=PC):
    for base in sorted(ROOT.glob("results_full_*")):
        f = base / fam / pc / "frozen_system.npz"
        if f.is_file():
            z = np.load(f, allow_pickle=True)
            return np.asarray(z["x_pre"], dtype=float)
    raise FileNotFoundError(f"x_pre missing for {fam}/{pc}")

def entropy_profile(psi):
    psi = psi / np.linalg.norm(psi)
    assert psi.shape == (N_DOF,), f"x_pre must be size {N_DOF}"
    T = psi.reshape([2] * NQ)
    S = []
    for k in range(1, NQ):
        M = T.reshape(2**k, 2**(NQ - k))
        sigma = np.linalg.svd(M, compute_uv=False)
        p = sigma ** 2
        p = p / p.sum()
        p = p[p > 1e-15]
        Sk = -np.sum(p * np.log2(p))
        S.append(Sk)
    return np.array(S)

print("=" * 72)
print("ENTROPY OF x_pre (normalize -> reshape -> SVD -> -sum p log2 p)")
print("=" * 72)

results = {}
for fam in ("heat1d", "heat2d", "beam1d"):
    try:
        x = load_xpre(fam)
    except FileNotFoundError as e:
        print(f"[skip] {e}"); continue
    S = entropy_profile(x)
    results[fam] = S
    Smax = S.max()
    karg = int(np.argmax(S)) + 1
    print(f"\n{FAM_PRETTY[fam]}")
    print(f"  |x_pre| = {np.linalg.norm(x):.6f}")
    for k in range(1, NQ):
        print(f"  S(k={k}) = {S[k-1]:.4f} bits")
    print(f"  S_max  = {Smax:.4f} bits (at k={karg})")

print("\n" + "=" * 72)
print("CONTROLS")
print("=" * 72)
ok = True
for fam, target in CTRL.items():
    if fam not in results:
        print(f"  [skip] {fam}: no data"); continue
    got = results[fam].max()
    delta = abs(got - target)
    status = "OK" if delta <= TOL else "FAIL"
    if delta > TOL: ok = False
    print(f"  {FAM_PRETTY[fam]:8s} expected ~ {target:.3f}  got = {got:.4f}  "
          f"|diff| = {delta:.4f}  -> {status}")

if not ok:
    print("\n" + "!" * 72)
    print("CONTROLS FAILED - no plot generated.")
    print("Likely causes:")
    print("  - wrong x_pre (physical solution, not preconditioned)")
    print("  - wrong preconditioner (not AMG)")
    print("  - wrong reshape / qubit-order convention")
    print("  - x_pre not normalized before reshaping")
    print("!" * 72)
    sys.exit(1)

print("\nControls OK -> generating plot.\n")

ks = np.arange(1, NQ)
fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), layout="constrained")
for fam in ("heat1d", "heat2d"):
    if fam not in results: continue
    S = results[fam]
    ax.plot(ks, S, marker="o", color=FAM_COLORS[fam],
            markeredgecolor=FAM_COLORS[fam], markerfacecolor="white",
            markeredgewidth=0.9, linewidth=1.4, label=FAM_PRETTY[fam])

ax.set_xticks(ks)
ax.set_xlabel(r"Bipartition  $k\,|\,(5-k)$")
ax.set_ylabel(r"Entanglement entropy  $S(k)$  [bits]")
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax.legend(loc="upper left")
ax.set_box_aspect(1)
ax.tick_params(direction="in", length=3, width=0.8)
for sp in ("top", "right", "bottom", "left"):
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.8)

fig.savefig("fig_entanglement_xpre.pdf")
fig.savefig("fig_entanglement_xpre.png", dpi=600)
plt.close(fig)
print("wrote fig_entanglement_xpre.pdf / .png")

print("\n" + "=" * 72)
print("TABLE")
print("=" * 72)
print(f"{'family':8s}  {'S(1)':>7s}  {'S(2)':>7s}  {'S(3)':>7s}  {'S(4)':>7s}  "
      f"{'S_max':>7s}  {'k*':>3s}")
print("-" * 60)
for fam in ("heat1d", "heat2d", "beam1d"):
    if fam not in results: continue
    S = results[fam]
    kstar = int(np.argmax(S)) + 1
    print(f"{FAM_PRETTY[fam]:8s}  "
          f"{S[0]:>7.4f}  {S[1]:>7.4f}  {S[2]:>7.4f}  {S[3]:>7.4f}  "
          f"{S.max():>7.4f}  {kstar:>3d}")
