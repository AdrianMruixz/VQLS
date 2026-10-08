#!/usr/bin/env python3
"""ibm_hardware.pdf  raw/none vs TREX vs TREX+ZNE on IBM Kingston."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

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

FAM_PRETTY = {"beam1d": "Beam1D", "heat2d": "Heat2D"}
FAMS = ["beam1d", "heat2d"]

# Display name: raw -> "none"
MODE_LABELS = {"raw": "none", "trex": "TREX", "trex_zne": "TREX+ZNE"}
MODE_COLORS = {"raw": "#4D4D4D", "trex": "#0072B2", "trex_zne": "#D55E00"}
MODES = ["raw", "trex", "trex_zne"]

IBM = {
    "beam1d": Path.home()/"ibm_fixed_theta_vqls/ibm_fixed_theta_vqls/ibm_results_beam1d/beam1d/ibm_kingston",
    "heat2d": Path.home()/"ibm_fixed_theta_vqls/ibm_fixed_theta_vqls/ibm_results_heat2d/heat2d/ibm_kingston",
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
    for sp in ("top","right","bottom","left"):
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

# ---------- load IBM JSONs ----------
# (fam, mode) -> list of C values (one per rep)
data = defaultdict(list)
for fam, root in IBM.items():
    if not root.is_dir():
        print(f"  ! missing {root}"); continue
    for jp in sorted(root.glob("*.json")):
        try: j = json.loads(jp.read_text())
        except Exception: continue
        mode = j.get("mode")
        if mode not in MODES: continue
        evs = (j.get("estimate") or {}).get("evs")
        if not evs: continue
        try:
            C = evs[0]
            if C is None: continue
            while isinstance(C, (list, tuple)) and len(C) > 0:
                C = C[-1]
            C = float(C)
        except Exception: continue
        if np.isfinite(C):
            data[(fam, mode)].append(C)

print("=== IBM Kingston: C por (familia, modo) ===")
for fam in FAMS:
    for mode in MODES:
        vs = data.get((fam, mode), [])
        if vs:
            print(f"  {FAM_PRETTY[fam]:8s} {MODE_LABELS[mode]:10s}  n={len(vs)}  "
                  f"mean={np.mean(vs):.3e}  vals={[f'{v:.3e}' for v in vs]}")

def mean(x): return float(np.mean(x)) if x else np.nan

# ---------- plot ----------
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.0), layout="constrained")

x = np.arange(len(FAMS))
w = 0.8 / len(MODES)

for i, mode in enumerate(MODES):
    vals = [mean(data.get((f, mode), [])) for f in FAMS]
    vals_plot = [v if np.isfinite(v) else 0.0 for v in vals]
    ax.bar(x + (i - (len(MODES)-1)/2)*w, vals_plot, w,
           color=MODE_COLORS[mode], edgecolor="black", linewidth=0.5,
           label=MODE_LABELS[mode])

ax.set_xticks(x)
ax.set_xticklabels([FAM_PRETTY[f] for f in FAMS])
ax.set_ylabel(r"Cost function  $C$  (IBM Kingston)")
log_y(ax)
ax.legend(loc="upper right")
style_axis(ax)

fig.savefig("ibm_hardware.pdf")
fig.savefig("ibm_hardware.png", dpi=600)
plt.close(fig)
print("\nwrote ibm_hardware.pdf / .png")

# ---------- resumen ----------
print("\n=== resumen IBM (mean over reps) ===")
print(f"{'family':8s}  {'none':>12s}  {'TREX':>12s}  {'TREX+ZNE':>12s}")
print("-"*55)
for f in FAMS:
    row = []
    for mode in MODES:
        v = mean(data.get((f, mode), []))
        row.append(f"{v:.3e}" if np.isfinite(v) else "--")
    print(f"{FAM_PRETTY[f]:8s}  " + "  ".join(f"{x:>12s}" for x in row))

print("\n=== reducciones ===")
for f in FAMS:
    cn = mean(data.get((f, "raw"), []))
    ct = mean(data.get((f, "trex"), []))
    cz = mean(data.get((f, "trex_zne"), []))
    if np.isfinite(cn) and np.isfinite(ct) and cn > 0:
        print(f"  {FAM_PRETTY[f]:8s}  none->TREX: {(cn-ct)/cn*100:+6.1f}%   "
              f"none->TREX+ZNE: {(cn-cz)/cn*100:+6.1f}%")
