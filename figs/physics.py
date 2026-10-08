#!/usr/bin/env python3
"""physics.pdf  2x2: spectrum, PR/r99 cumulative, error distribution per family."""
from __future__ import annotations
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter, MaxNLocator

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 6.2,
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
METRIC_COLORS = {"L2": "#009E73", "eK": "#0072B2", "r_eq": "#D55E00"}
METRIC_LABELS = {
    "L2":   r"$\beta_g^2$",
    "eK":   r"$\lambda_g\,\beta_g^2$",
    "r_eq": r"$\lambda_g^2\,\beta_g^2$",
}



def set_xlim_auto(ax, n_max, n_ticks=6):
    """xlim 1..n_max con n_ticks equiespaciados, terminando exacto en n_max."""
    ax.set_xlim(1, n_max)
    # ticks equiespaciados: incluye SIEMPRE 1 y n_max
    ticks = np.linspace(1, n_max, n_ticks).round().astype(int).tolist()
    ticks = sorted(set(ticks))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


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
    for sp in ("top","right","bottom","left"):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_linewidth(0.8)

def panel_label(ax, letter):
    ax.text(0.97, 0.97, f"{letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="right")

def log_y(ax):
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    lo_exp = int(np.floor(np.log10(lo)))
    hi_exp = int(np.ceil(np.log10(hi)))
    step = max(1, (hi_exp - lo_exp) // 4)
    dec = [10.0**e for e in range(lo_exp, hi_exp+1, step)]
    ax.set_yticks(dec)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.2,0.5), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(_sci_fmt))

# ---------- load ----------
import physics_dependency_final_v5 as core
print("Loading physical quantities ...")
physical = {fam: core.physical_modal_analysis(fam, 1e-8, 1e-24)
            for fam in ("heat1d", "beam1d")}
noisy = {fam: core.noisy_error_analysis(fam, physical[fam], 1e-8)
         for fam in ("heat1d", "beam1d")}
print("  ok")

# ---------- figure ----------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), layout="constrained")

# (a) Spectrum: both families overlaid
ax = axes[0, 0]
for fam in ("heat1d", "beam1d"):
    lam = np.asarray(physical[fam]["group_lam_eq"], dtype=float)
    x = np.arange(1, len(lam)+1)
    keq = physical[fam]["kappa_eq"]
    ax.plot(x, lam/lam.max(), marker="o",
            color=FAM_COLORS[fam], markeredgecolor=FAM_COLORS[fam],
            markerfacecolor="white", markeredgewidth=0.9,
            label=rf"{FAM_PRETTY[fam]}  ($\kappa_{{\rm eq}}$={keq:.2g})")
nmax = max(len(physical[f]["group_lam_eq"]) for f in ("heat1d","beam1d"))
set_xlim_auto(ax, nmax, n_ticks=6)
ax.set_xlabel("Distinct eigenspace index")
ax.set_ylabel(r"$\lambda_g/\lambda_{\max}$")
log_y(ax)
ax.legend(loc="lower left", fontsize=6.2)
style_axis(ax)
panel_label(ax, "a")

# (b) Cumulative solution weight with PR and r99
ax = axes[0, 1]
for fam in ("heat1d", "beam1d"):
    w = np.sort(np.asarray(physical[fam]["group_solution_w"], dtype=float))[::-1]
    cum = np.cumsum(w)
    x = np.arange(1, len(cum)+1)
    pr  = physical[fam]["solution_PR_grouped"]
    r99 = physical[fam]["solution_r99_grouped"]
    ax.plot(x, cum, marker="o",
            color=FAM_COLORS[fam], markeredgecolor=FAM_COLORS[fam],
            markerfacecolor="white", markeredgewidth=0.9,
            label=rf"{FAM_PRETTY[fam]}  ($PR$={pr:.2f},  $r_{{99}}$={int(r99)})")
ax.axhline(0.99, ls="--", lw=0.9, alpha=0.7, color="gray")
nmax = max(len(physical[f]["group_solution_w"]) for f in ("heat1d","beam1d"))
set_xlim_auto(ax, nmax, n_ticks=6)
ax.set_ylim(0, 1.02)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax.set_xlabel("Dominant eigenspaces retained")
ax.set_ylabel("Cumulative solution weight")
ax.legend(loc="lower right", fontsize=6.2)
style_axis(ax)
panel_label(ax, "b")

# (c) Error distribution  Heat1D
ax = axes[1, 0]
d = noisy["heat1d"]
x = np.arange(1, len(d["group_lam_eq"])+1)
ax.plot(x, np.maximum(d["group_balanced_l2_w"], 1e-16),
        marker="^", color=METRIC_COLORS["L2"],
        markeredgecolor=METRIC_COLORS["L2"], markerfacecolor="white",
        markeredgewidth=0.9, label=METRIC_LABELS["L2"])
ax.plot(x, np.maximum(d["group_energy_error_w"], 1e-16),
        marker="o", color=METRIC_COLORS["eK"],
        markeredgecolor=METRIC_COLORS["eK"], markerfacecolor="white",
        markeredgewidth=0.9, label=METRIC_LABELS["eK"])
ax.plot(x, np.maximum(d["group_equilibrated_residual_w"], 1e-16),
        marker="s", color=METRIC_COLORS["r_eq"],
        markeredgecolor=METRIC_COLORS["r_eq"], markerfacecolor="white",
        markeredgewidth=0.9, label=METRIC_LABELS["r_eq"])
set_xlim_auto(ax, len(d["group_lam_eq"]), n_ticks=6)
ax.set_xlabel("Equilibrated eigenspace index")
ax.set_ylabel("Normalized contribution")
log_y(ax)
ax.legend(loc="lower left", fontsize=6.0, title="Heat1D")
style_axis(ax)
panel_label(ax, "c")

# (d) Error distribution  Beam1D
ax = axes[1, 1]
d = noisy["beam1d"]
x = np.arange(1, len(d["group_lam_eq"])+1)
ax.plot(x, np.maximum(d["group_balanced_l2_w"], 1e-16),
        marker="^", color=METRIC_COLORS["L2"],
        markeredgecolor=METRIC_COLORS["L2"], markerfacecolor="white",
        markeredgewidth=0.9, label=METRIC_LABELS["L2"])
ax.plot(x, np.maximum(d["group_energy_error_w"], 1e-16),
        marker="o", color=METRIC_COLORS["eK"],
        markeredgecolor=METRIC_COLORS["eK"], markerfacecolor="white",
        markeredgewidth=0.9, label=METRIC_LABELS["eK"])
ax.plot(x, np.maximum(d["group_equilibrated_residual_w"], 1e-16),
        marker="s", color=METRIC_COLORS["r_eq"],
        markeredgecolor=METRIC_COLORS["r_eq"], markerfacecolor="white",
        markeredgewidth=0.9, label=METRIC_LABELS["r_eq"])
set_xlim_auto(ax, len(d["group_lam_eq"]), n_ticks=6)
ax.set_xlabel("Equilibrated eigenspace index")
ax.set_ylabel("Normalized contribution")
log_y(ax)
ax.legend(loc="lower left", fontsize=6.0, title="Beam1D")
style_axis(ax)
panel_label(ax, "d")

fig.savefig("physics.pdf")
fig.savefig("physics.png", dpi=600)
plt.close(fig)
print("wrote physics.pdf / .png")
