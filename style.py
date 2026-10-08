"""
Nature Portfolio figure style + shared plotting helpers.

Import with: from style import *
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

# ---------- rcParams (Nature-compliant) ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.linewidth": 0.8, "axes.grid": False, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 2.8,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "xtick.direction": "in", "ytick.direction": "in",
})

# ---------- colors ----------
FAM_COLORS = {"heat1d": "#0072B2", "heat2d": "#E69F00",
              "beam1d": "#D55E00", "beam2d": "#009E73"}
ARM_COLORS = {"none": "#4D4D4D", "trex": "#0072B2", "trex_zne": "#D55E00"}
ARM_LABELS = {"none": "none", "trex": "TREX", "trex_zne": "TREX+ZNE"}
ARM_STYLE  = {"none": dict(ls="-",  marker="o", label="none"),
              "trex": dict(ls="--", marker="D", label="TREX")}
METRIC_COLORS = {"L2": "#009E73", "eK": "#0072B2", "r_eq": "#D55E00"}
METRIC_LABELS = {"L2":   r"$\beta_g^2$",
                 "eK":   r"$\lambda_g\,\beta_g^2$",
                 "r_eq": r"$\lambda_g^2\,\beta_g^2$"}

# ---------- helpers ----------
def sci_fmt(v, _pos=None):
    if v <= 0: return "0"
    s = f"{v:.1e}"; ms, es = s.split("e"); e = int(es); m = float(ms)
    if abs(m - 1.0) < 1e-6:
        return "1" if e == 0 else r"$10^{%d}$" % e
    return r"$%.1f\times10^{%d}$" % (m, e)

def panel_label(ax, letter, loc="right"):
    if loc == "right":
        ax.text(0.97, 0.97, f"{letter})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="right")
    else:
        ax.text(0.03, 0.97, f"{letter})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

def style_axis(ax, box=True):
    if box:
        for sp in ("top","right","bottom","left"):
            ax.spines[sp].set_visible(True); ax.spines[sp].set_linewidth(0.8)
    else:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="major", bottom=True, top=False,
                   labelbottom=True, labeltop=False,
                   direction="in", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", left=True, right=False,
                   labelleft=True, labelright=False,
                   direction="in", length=3, width=0.8)
    ax.tick_params(axis="y", which="minor", left=True, right=False,
                   direction="in", length=1.5, width=0.6)
    ax.set_box_aspect(1)

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
    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))

def force_top_tick(ax):
    lo, hi = ax.get_ylim()
    ticks = sorted(t for t in ax.get_yticks() if lo <= t <= hi)
    step = (ticks[1] - ticks[0]) if len(ticks) >= 2 else (hi / 5)
    def nice_step(s):
        exp = math.floor(math.log10(s)); base = s / 10**exp
        for c in (1, 2, 2.5, 5, 10):
            if base <= c: return c * 10**exp
        return 10**(exp+1)
    nstep = nice_step(step)
    top = math.ceil(hi / nstep) * nstep
    start = math.floor(lo / nstep) * nstep
    new_ticks = []; v = start
    while v <= top + 1e-9:
        if v >= lo - 1e-9: new_ticks.append(round(v, 10))
        v += nstep
    if top not in new_ticks: new_ticks.append(top)
    ax.set_yticks(sorted(set(new_ticks)))
    ax.set_ylim(lo, top)
