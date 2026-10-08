#!/usr/bin/env python3
"""fig_ideal_validation.pdf  2x2: Fidelity & L2, all 4 families x all 4 preconditioners."""
from __future__ import annotations
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
    "lines.linewidth": 1.2, "lines.markersize": 2.6,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "xtick.direction": "in", "ytick.direction": "in",
})

# Nature-safe Okabe-Ito
FAM_COLORS = {"heat1d": "#0072B2",   # blue
              "heat2d": "#E69F00",   # orange
              "beam1d": "#D55E00",   # vermillion
              "beam2d": "#009E73"}   # green
FAM_PRETTY = {"heat1d": "Heat1D", "heat2d": "Heat2D",
              "beam1d": "Beam1D", "beam2d": "Beam2D"}
PRECOND_STYLE = {
    "amg":    dict(ls="-",  marker="o", label="AMG"),
    "ichol":  dict(ls="--", marker="s", label="ichol"),
    "jacobi": dict(ls=":",  marker="^", label="jacobi"),
    "none":   dict(ls="-.", marker="D", label="none"),
}
ROOT = Path.home() / "ideal_preconditioning_32dof"
FAMS = ["heat1d", "heat2d", "beam1d", "beam2d"]
PCS  = ["amg", "ichol", "jacobi", "none"]

def load(fam, pc):
    for base in sorted(ROOT.glob("results_full_*")):
        f = base / fam / pc / "depth_summary.csv"
        if f.is_file():
            df = pd.read_csv(f)
            return df.sort_values("depth").reset_index(drop=True)
    return None

def _sci_fmt(v, _pos=None):
    if v <= 0: return "0"
    s = f"{v:.1e}"; ms, es = s.split("e"); e = int(es); m = float(ms)
    if abs(m-1.0) < 1e-6:
        if e == 0: return "1"
        return r"$10^{%d}$" % e
    return r"$%.1f\times10^{%d}$" % (m, e)

def style_axis(ax):
    ax.set_xlim(1, 6)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
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

def panel_label(ax, letter):
    ax.text(0.03, 0.97, f"{letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left")

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

# ---- load all ----
data = {}
for fam in FAMS:
    for pc in PCS:
        d = load(fam, pc)
        if d is not None:
            data[(fam, pc)] = d

# ---- figure 2x2 ----
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), layout="constrained")

# (a) Fidelity vs depth: color=family, ls=precond
ax = axes[0, 0]
for fam in FAMS:
    for pc in PCS:
        d = data.get((fam, pc))
        if d is None: continue
        st = PRECOND_STYLE[pc]
        ax.plot(d["depth"], d["F_pre"],
                color=FAM_COLORS[fam], linestyle=st["ls"],
                marker=st["marker"], markersize=2.4,
                markeredgecolor=FAM_COLORS[fam], markerfacecolor="white",
                markeredgewidth=0.8, linewidth=1.0)
ax.set_xlabel("Depth  $d$")
ax.set_ylabel(r"Fidelity  $F$")
ax.set_ylim(0.0, 1.02)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
style_axis(ax)
panel_label(ax, "a")

# (b) L2 vs depth: color=family, ls=precond
ax = axes[0, 1]
for fam in FAMS:
    for pc in PCS:
        d = data.get((fam, pc))
        if d is None: continue
        st = PRECOND_STYLE[pc]
        ax.plot(d["depth"], d["l2_fis"],
                color=FAM_COLORS[fam], linestyle=st["ls"],
                marker=st["marker"], markersize=2.4,
                markeredgecolor=FAM_COLORS[fam], markerfacecolor="white",
                markeredgewidth=0.8, linewidth=1.0)
ax.set_xlabel("Depth  $d$")
ax.set_ylabel(r"$L_2$ error")
# tope del eje en 1 (los valores >1 se recortan visualmente)
_lo = min(
    float(data[(f, pc)]["l2_fis"].min())
    for f in FAMS for pc in PCS if (f, pc) in data
)
ax.set_ylim(_lo / 2, 1.0)
log_y(ax)
style_axis(ax)
panel_label(ax, "b")

# (c) Fidelity, zoom: only AMG, one curve per family
ax = axes[1, 0]
for fam in FAMS:
    d = data.get((fam, "amg"))
    if d is None: continue
    ax.plot(d["depth"], d["F_pre"], marker="o",
            color=FAM_COLORS[fam], markeredgecolor=FAM_COLORS[fam],
            markerfacecolor="white", markeredgewidth=0.9,
            linewidth=1.4, label=FAM_PRETTY[fam])
ax.set_xlabel("Depth  $d$")
ax.set_ylabel(r"Fidelity  $F$  (AMG only)")
ax.set_ylim(0.0, 1.02)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax.legend(loc="lower right", ncol=2)
style_axis(ax)
panel_label(ax, "c")

# (d) L2, zoom: only AMG, one curve per family
ax = axes[1, 1]
for fam in FAMS:
    d = data.get((fam, "amg"))
    if d is None: continue
    ax.plot(d["depth"], d["l2_fis"], marker="o",
            color=FAM_COLORS[fam], markeredgecolor=FAM_COLORS[fam],
            markerfacecolor="white", markeredgewidth=0.9,
            linewidth=1.4, label=FAM_PRETTY[fam])
ax.set_xlabel("Depth  $d$")
ax.set_ylabel(r"$L_2$ error  (AMG only)")
_lo_d = min(
    float(data[(f, "amg")]["l2_fis"].min())
    for f in FAMS if (f, "amg") in data
)
ax.set_ylim(_lo_d / 2, 1.0)
log_y(ax)
ax.legend(loc="lower left", ncol=2)
style_axis(ax)
panel_label(ax, "d")

# ---- legends ----
fam_handles = [Line2D([0],[0], color=FAM_COLORS[f], lw=1.4,
                      label=FAM_PRETTY[f]) for f in FAMS]
pc_handles  = [Line2D([0],[0], color="#333333", lw=1.2,
                      linestyle=PRECOND_STYLE[p]["ls"],
                      marker=PRECOND_STYLE[p]["marker"],
                      markersize=3.0, markerfacecolor="white",
                      markeredgecolor="#333333", markeredgewidth=0.8,
                      label=PRECOND_STYLE[p]["label"]) for p in PCS]

fig.legend(handles=fam_handles + pc_handles,
           loc="upper center", ncol=8, frameon=False,
           bbox_to_anchor=(0.5, 1.03), fontsize=7)

fig.savefig("fig_ideal_validation.pdf")
fig.savefig("fig_ideal_validation.png", dpi=600)
plt.close(fig)
print("wrote fig_ideal_validation.pdf / .png")

# ---- tabla resumen ----
print("\n=== F(d=6) por familia/precondicionador ===")
print(f"{'family':8s}  " + "  ".join(f"{pc:>9s}" for pc in PCS))
print("-"*60)
for fam in FAMS:
    row = [f"{data[(fam,pc)]['F_pre'].iloc[-1]:>9.4f}"
           if (fam,pc) in data else f"{'--':>9s}" for pc in PCS]
    print(f"{FAM_PRETTY[fam]:8s}  " + "  ".join(row))

print("\n=== L2(d=6) por familia/precondicionador ===")
print(f"{'family':8s}  " + "  ".join(f"{pc:>9s}" for pc in PCS))
print("-"*60)
for fam in FAMS:
    row = [f"{data[(fam,pc)]['l2_fis'].iloc[-1]:>9.2e}"
           if (fam,pc) in data else f"{'--':>9s}" for pc in PCS]
    print(f"{FAM_PRETTY[fam]:8s}  " + "  ".join(row))
