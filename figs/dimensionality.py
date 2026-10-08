#!/usr/bin/env python3
"""dimensionality.pdf  2x2 combined dimensionality figure."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 6.5,
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

FAM_COLORS = {"heat1d": "#0072B2", "heat2d": "#E69F00",
              "beam1d": "#D55E00", "beam2d": "#009E73"}
FAM_PRETTY = {"heat1d": "Heat1D", "heat2d": "Heat2D",
              "beam1d": "Beam1D", "beam2d": "Beam2D"}
FAMS = ["heat1d", "heat2d", "beam1d", "beam2d"]
NQ = 5
N_DOF = 2 ** NQ
ROOT = Path.home() / "ideal_preconditioning_32dof"
PC = "amg"

# 3-seed archives (same as in mechanism figure)
ARCHIVES_3SEEDS = {
    "heat1d": [
        ("raw_none_seed1001_20260923_124835", "heat1d", 1001),
        ("raw_none_seed3003_20260923_122930", "heat1d", 3003),
        ("none_seed4001_20260928_193817",     "heat1d", 4001),
    ],
    "heat2d": [
        ("raw_none_seed1001_20260923_124835", "heat2d", 1001),
        ("raw_none_seed3003_20260923_122930", "heat2d", 3003),
        ("none_seed4001_20260928_193817",     "heat2d", 4001),
    ],
}

def _sci_fmt(v, _pos=None):
    if v <= 0: return "0"
    s = f"{v:.1e}"; ms, es = s.split("e"); e = int(es); m = float(ms)
    if abs(m-1.0) < 1e-6:
        if e == 0: return "1"
        return r"$10^{%d}$" % e
    return r"$%.1f\times10^{%d}$" % (m, e)

def style_axis(ax):
    ax.set_box_aspect(1)
    ax.tick_params(axis="x", which="major", bottom=True, top=False,
                   labelbottom=True, labeltop=False,
                   direction="in", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", left=True, right=False,
                   labelleft=True, labelright=False,
                   direction="in", length=3, width=0.8)
    ax.tick_params(axis="y", which="minor", left=True, right=False,
                   direction="in", length=1.5, width=0.6)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_linewidth(0.8)


def force_top_tick(ax, nice=True):
    """Fija el tope del eje Y a un valor 'nice' >= max, con tick etiquetado."""
    lo, hi = ax.get_ylim()
    ticks = sorted(t for t in ax.get_yticks() if lo <= t <= hi)
    if not nice:
        if not any(abs(t - hi) < 1e-9 * max(1, abs(hi)) for t in ticks):
            ticks = sorted(ticks + [hi])
        ax.set_yticks(ticks); ax.set_ylim(lo, hi)
        return
    # elegir un top "redondo" por encima de hi
    # 1) intentar multiple de la separacion tipica
    if len(ticks) >= 2:
        step = ticks[1] - ticks[0]
    else:
        step = hi / 5
    import math
    # nice step: 1,2,2.5,5,10,20,25,50,...
    def nice_step(s):
        exp = math.floor(math.log10(s))
        base = s / 10**exp
        for cand in (1, 2, 2.5, 5, 10):
            if base <= cand:
                return cand * 10**exp
        return 10**(exp+1)
    nstep = nice_step(step)
    top = math.ceil(hi / nstep) * nstep
    # armar grid de ticks desde lo hasta top
    start = math.floor(lo / nstep) * nstep
    new_ticks = []
    v = start
    while v <= top + 1e-9:
        if v >= lo - 1e-9:
            new_ticks.append(round(v, 10))
        v += nstep
    if top not in new_ticks:
        new_ticks.append(top)
    ax.set_yticks(sorted(set(new_ticks)))
    ax.set_ylim(lo, top)


def panel_label(ax, letter):
    ax.text(0.97, 0.97, f"{letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="right")

def load_depth(fam, pc=PC):
    for base in sorted(ROOT.glob("results_full_*")):
        f = base / fam / pc / "depth_summary.csv"
        if f.is_file(): return pd.read_csv(f).sort_values("depth")
    return None

def load_xpre(fam, pc=PC):
    for base in sorted(ROOT.glob("results_full_*")):
        f = base / fam / pc / "frozen_system.npz"
        if f.is_file():
            z = np.load(f, allow_pickle=True)
            return np.asarray(z["x_pre"], dtype=float)
    return None

def entropy_profile(psi):
    psi = psi / np.linalg.norm(psi)
    T = psi.reshape([2]*NQ)
    S = []
    for k in range(1, NQ):
        M = T.reshape(2**k, 2**(NQ-k))
        s = np.linalg.svd(M, compute_uv=False)
        p = s**2; p = p / p.sum(); p = p[p > 1e-15]
        S.append(-np.sum(p * np.log2(p)))
    return np.array(S)

# ---------------- load L2 trajectories (3-seed mean) ----------------
import physics_dependency_final_v5 as core
print("Loading physical quantities ...")
physical = {fam: core.physical_modal_analysis(fam, 1e-8, 1e-24)
            for fam in ("heat1d", "heat2d", "beam1d")}
traj_curves = {}
for fam, specs in ARCHIVES_3SEEDS.items():
    curves = []
    for camp, fam_dir, seed in specs:
        p = Path.home()/"other_families_trex_campaign"/camp/fam_dir/f"archive_seed{seed}.npz"
        if not p.is_file(): continue
        old = core.NOISY_ARCHIVE.get(fam, None)
        core.NOISY_ARCHIVE[fam] = str(p)
        try:
            d = core.noisy_trajectory_analysis(fam, physical[fam])
            it = np.asarray(d["iter"], dtype=float)
            y  = np.asarray(d["balanced_l2"], dtype=float)
            ok = np.isfinite(it) & np.isfinite(y)
            if ok.any(): curves.append((it[ok], y[ok]))
        except Exception as e:
            print(f"  warn {fam} seed={seed}: {e}")
        finally:
            if old is not None: core.NOISY_ARCHIVE[fam] = old
    traj_curves[fam] = curves

def mean_over_seeds(curves):
    if not curves: return None
    it_min = max(c[0][0] for c in curves)
    it_max = min(c[0][-1] for c in curves)
    if it_max <= it_min: return None
    grid = np.arange(it_min, it_max + 1)
    interp = [np.interp(grid, it, y) for it, y in curves]
    C = np.vstack(interp)
    return dict(grid=grid, mu=C.mean(0), sd=C.std(0, ddof=0), n=len(curves))

# ---------------- figure ----------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), layout="constrained")

# (a) PR and r99 (from physics_dependency)
ax = axes[0, 0]
fams_pa = ["heat1d", "heat2d"]
x = np.arange(len(fams_pa))
w = 0.36
# PR / r99 de x_pre precondicionado (mismo que fig_dimensionality_mechanism)
ideal_vals = {f: core.load_ideal_analysis(f, "amg", 1e-8) for f in fams_pa}
pr_vals  = [ideal_vals[f]["xpre_PR_grouped"]  for f in fams_pa]
r99_vals = [ideal_vals[f]["xpre_r99_grouped"] for f in fams_pa]
colors_pa = [FAM_COLORS[f] for f in fams_pa]
ax.bar(x - w/2, pr_vals,  w, color=colors_pa, edgecolor="black",
       linewidth=0.6, label=r"$PR$")
ax.bar(x + w/2, r99_vals, w, color=colors_pa, alpha=0.55,
       edgecolor="black", linewidth=0.6, label=r"$r_{99}$")
for xi, v in zip(x - w/2, pr_vals):
    ax.text(xi, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
for xi, v in zip(x + w/2, r99_vals):
    ax.text(xi, v + 0.15, f"{int(v)}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([FAM_PRETTY[f] for f in fams_pa])
ax.set_ylabel(r"Eigenspaces")
ax.legend(loc="upper left")
ax.set_ylim(0, max(r99_vals) * 1.25)
force_top_tick(ax)
style_axis(ax)
panel_label(ax, "a")

# (b) Entanglement S(k)
ax = axes[0, 1]
ks = np.arange(1, NQ)
for fam in ("heat1d", "heat2d"):
    x_pre = load_xpre(fam)
    if x_pre is None: continue
    S = entropy_profile(x_pre)
    ax.plot(ks, S, marker="o", color=FAM_COLORS[fam],
            markeredgecolor=FAM_COLORS[fam], markerfacecolor="white",
            markeredgewidth=0.9, linewidth=1.4, label=FAM_PRETTY[fam])
ax.set_xlim(1, 4)
ax.set_xticks(ks)
ax.set_xticklabels(["1", "2", "3", "4"])
ax.set_xlabel(r"Bipartition  $k\,|\,(5-k)$")
ax.set_ylabel(r"Entanglement entropy  $S(k)$  [bits]")
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
# panel (b) ya tiene ticks nice [0,0.25,...,1]
ax.legend(loc="upper left")
style_axis(ax)
panel_label(ax, "b")

# (c) d_min (bars) for 4 families
ax = axes[1, 0]
fams_d = ["heat1d", "heat2d"]
x = np.arange(len(fams_d))
dmin_vals = []
for fam in fams_d:
    d = load_depth(fam)
    if d is None: dmin_vals.append(np.nan); continue
    ok = d[d["F_pre"] > 0.99]
    dmin_vals.append(int(ok["depth"].min()) if not ok.empty else 6.5)
colors_d = [FAM_COLORS[f] for f in fams_d]
ax.bar(x, dmin_vals, 0.6, color=colors_d, edgecolor="black", linewidth=0.6)
for xi, v in zip(x, dmin_vals):
    if np.isfinite(v):
        ax.text(xi, v + 0.1, f"{int(v)}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([FAM_PRETTY[f] for f in fams_d])
ax.set_ylabel(r"$d_{\min}$  for  $F > 0.99$  (AMG)")
ax.set_ylim(0, max(6, int(np.nanmax(dmin_vals)) + 1))
force_top_tick(ax)
style_axis(ax)
panel_label(ax, "c")

# (d) L2 trajectory (mean ± std over 3 seeds)
ax = axes[1, 1]
rows = []
for fam in ("heat1d", "heat2d"):
    m = mean_over_seeds(traj_curves.get(fam, []))
    if m is None: continue
    col = FAM_COLORS[fam]
    ax.plot(m["grid"], m["mu"], marker="o", ms=2.0, color=col,
            label=FAM_PRETTY[fam])
ax.set_xlabel("Optimization iteration")
ax.set_ylabel(r"$L_2$  (mean)")
ax.set_xlim(0, 250)
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.set_ylim(bottom=0)
force_top_tick(ax)
ax.legend(loc="upper left")
style_axis(ax)
panel_label(ax, "d")

fig.savefig("dimensionality.pdf")
fig.savefig("dimensionality.png", dpi=600)
plt.close(fig)
print("wrote dimensionality.pdf / .png")
