#!/usr/bin/env python3
"""costs_stages.pdf  2 panels, 4 families, mean only (no error bars)."""
from __future__ import annotations
import re, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
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

FAM_PRETTY = {"heat1d": "Heat1D", "heat2d": "Heat2D",
              "beam1d": "Beam1D", "beam2d": "Beam2D"}
FAMS = ["heat1d", "heat2d", "beam1d", "beam2d"]

ARM_COLORS = {"none": "#4D4D4D", "trex": "#0072B2", "trex_zne": "#D55E00"}
ARM_LABELS = {"none": "none", "trex": "TREX", "trex_zne": "TREX+ZNE"}

HOME = Path.home()
CAMPAIGN   = HOME/"other_families_trex_campaign"
BEAM1D_FIN = HOME/"beam1d_final_results"
BEAM1D_OPT = HOME/"beam1d_optimized"

# Whitelist family campaigns (same as plot_nature_figs.py)
WL_NONE = {"raw_none_seed1001_20260923_124835",
           "raw_none_seed3003_20260923_122930",
           "none_seed4001_20260928_193817"}
WL_TREX = {"trex_only_final_20260921_114421",
           "trex_seed3003_20261002_170940",
           "trex_seed4001_20261002_170940"}
# Beam1D campaigns (same as figures)
WL_B1D = {"beam1d_paired_final_20260922_184543",
          "beam1d_full_seed8203_20260930_111027",
          "beam1d_full_seed8303_20261001_012345"}

def canon(x): return re.sub(r"[^a-z0-9]","",str(x).lower())
def find_col(df, names):
    cols = {canon(c): c for c in df.columns}
    for n in names:
        if canon(n) in cols: return cols[canon(n)]
    return None
def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0],
        errors="coerce")
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
def mean(x): return float(np.mean(x)) if x else np.nan

# =====================================================================
# Panel (a): training-time from CSVs (C_spsa_center)
# =====================================================================
data_a = defaultdict(dict)   # (fam, arm) -> {seed: C}

# --- familias (heat1d/heat2d/beam2d) ---
for p in CAMPAIGN.rglob("metrics_seed*.csv"):
    sp = str(p).lower()
    if any(t in sp for t in ("smoke","_ckpt","test","__pycache__")): continue
    camp = p.parent.parent.name
    if camp in WL_NONE:   arm = "none"
    elif camp in WL_TREX: arm = "trex"
    else: continue
    try: head = pd.read_csv(p, nrows=1)
    except Exception: continue
    if "family" not in head.columns: continue
    fam = str(head["family"].iloc[0]).lower()
    if fam not in ("heat1d","heat2d","beam2d"): continue
    try: seed = int(head["seed"].iloc[0])
    except Exception: continue
    df = pd.read_csv(p); df.columns = [str(c).strip() for c in df.columns]
    c_dof = find_col(df, ["dof","ndof","n_dof","level","nivel"])
    c_it  = find_col(df, ["iter","iteration","it","step"])
    c_C   = find_col(df, ["C_spsa_center","C_exact","C_full_exact","C_local_pauli"])
    if None in (c_dof, c_it, c_C): continue
    df["_dof"]=to_num(df[c_dof]); df["_it"]=to_num(df[c_it]); df["_C"]=to_num(df[c_C])
    g = df[df._dof == 32].sort_values("_it")
    if not g.empty and np.isfinite(g["_C"].iloc[-1]):
        data_a[(fam, arm)][seed] = float(g["_C"].iloc[-1])

# --- Beam1D ---
for root in [BEAM1D_FIN, BEAM1D_OPT]:
    if not root.is_dir(): continue
    for p in root.rglob("metrics_seed*.csv"):
        sp = str(p).lower()
        if any(t in sp for t in ("smoke","_ckpt","test","__pycache__")): continue
        # arm = parent dir name (none / trex)
        arm = p.parent.name.lower()
        if arm not in ("none","trex"): continue
        # camp: si parent parent es "none"/"trex", subir un nivel
        camp = p.parent.parent.name
        if camp not in WL_B1D: continue
        try: head = pd.read_csv(p, nrows=1)
        except Exception: continue
        if "family" not in head.columns: continue
        fam = str(head["family"].iloc[0]).lower()
        if fam != "beam1d": continue
        try: seed = int(head["seed"].iloc[0])
        except Exception: continue
        df = pd.read_csv(p); df.columns = [str(c).strip() for c in df.columns]
        c_dof = find_col(df, ["dof","ndof","n_dof","level","nivel"])
        c_it  = find_col(df, ["iter","iteration","it","step"])
        c_C   = find_col(df, ["C_spsa_center","C_exact","C_full_exact","C_local_pauli"])
        if None in (c_dof, c_it, c_C): continue
        df["_dof"]=to_num(df[c_dof]); df["_it"]=to_num(df[c_it]); df["_C"]=to_num(df[c_C])
        g = df[df._dof == 32].sort_values("_it")
        if not g.empty and np.isfinite(g["_C"].iloc[-1]):
            data_a[(fam, arm)][seed] = float(g["_C"].iloc[-1])

# =====================================================================
# Panel (b): post-hoc QEM from JSONs
# =====================================================================
data_b = defaultdict(dict)   # (fam, mode) -> {seed: C}

# familias
for jp in CAMPAIGN.rglob("qem_seed*.json"):
    sp = str(jp).lower()
    if "smoke" in sp or "test" in sp: continue
    if jp.parent.parent.name not in WL_TREX: continue
    try: j = json.loads(jp.read_text())
    except Exception: continue
    fam = str(j.get("family","")).lower()
    if fam not in ("heat1d","heat2d","beam2d"): continue
    seed_m = re.search(r"seed(\d+)", jp.stem)
    seed = int(seed_m.group(1)) if seed_m else None
    if seed is None: continue
    for mode in ("trex","trex_zne"):
        e = j.get("modes",{}).get(mode)
        if not e: continue
        est = e.get("estimate",{})
        if "mean" in est and np.isfinite(est["mean"]):
            data_b[(fam, mode)][seed] = float(est["mean"])

# Beam1D
for root in [BEAM1D_FIN, BEAM1D_OPT]:
    if not root.is_dir(): continue
    for jp in root.rglob("qem_seed*.json"):
        sp = str(jp).lower()
        if "smoke" in sp or "test" in sp: continue
        # camp: parent.parent (none/trex) .parent (campaign)
        camp = jp.parent.parent.name
        if camp not in WL_B1D: continue
        try: j = json.loads(jp.read_text())
        except Exception: continue
        # beam1d json no tiene family  sacar del path
        fam = "beam1d" if "beam1d" in sp else None
        if fam is None: continue
        seed_m = re.search(r"seed(\d+)", jp.stem)
        seed = int(seed_m.group(1)) if seed_m else None
        if seed is None: continue
        for mode in ("trex","trex_zne"):
            e = j.get("modes",{}).get(mode)
            if not e: continue
            est = e.get("estimate",{})
            if "mean" in est and np.isfinite(est["mean"]):
                data_b[(fam, mode)][seed] = float(est["mean"])

print("=== (a) training-time C (CSV, C_spsa_center, last iter dof=32) ===")
for fam in FAMS:
    for arm in ("none","trex"):
        d = data_a.get((fam, arm), {})
        if not d: continue
        vals = "  ".join(f"{s}:{v:.3e}" for s,v in sorted(d.items()))
        print(f"  {FAM_PRETTY[fam]:8s} {arm:5s}  {vals}   mean={mean(list(d.values())):.3e}")

print("\n=== (b) post-hoc QEM C (JSON) ===")
for fam in FAMS:
    for mode in ("trex","trex_zne"):
        d = data_b.get((fam, mode), {})
        if not d: continue
        vals = "  ".join(f"{s}:{v:.3e}" for s,v in sorted(d.items()))
        print(f"  {FAM_PRETTY[fam]:8s} {mode:9s}  {vals}   mean={mean(list(d.values())):.3e}")

# =====================================================================
# Figure 1×2  mean only, no error bars
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), layout="constrained")

# --- (a) ---
ax = axes[0]
x = np.arange(len(FAMS)); w = 0.36
for i, arm in enumerate(("none","trex")):
    vals = [mean(list(data_a.get((f, arm), {}).values())) for f in FAMS]
    vals_plot = [v if np.isfinite(v) else 0.0 for v in vals]
    ax.bar(x + (i-0.5)*w, vals_plot, w,
           color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5,
           label=ARM_LABELS[arm])
ax.set_xticks(x); ax.set_xticklabels([FAM_PRETTY[f] for f in FAMS])
ax.set_ylabel(r"Training-time  $C$  (mean)")
log_y(ax)
ax.legend(loc="upper right")
style_axis(ax)
panel_label(ax, "a")

# --- (b) ---
ax = axes[1]
for i, mode in enumerate(("trex","trex_zne")):
    vals = [mean(list(data_b.get((f, mode), {}).values())) for f in FAMS]
    vals_plot = [v if np.isfinite(v) else 0.0 for v in vals]
    ax.bar(x + (i-0.5)*w, vals_plot, w,
           color=ARM_COLORS[mode], edgecolor="black", linewidth=0.5,
           label=ARM_LABELS[mode])
ax.set_xticks(x); ax.set_xticklabels([FAM_PRETTY[f] for f in FAMS])
ax.set_ylabel(r"Post-hoc QEM  $C$  (mean)")
log_y(ax)
ax.legend(loc="upper right")
style_axis(ax)
panel_label(ax, "b")

fig.savefig("costs_training_and_qem.pdf")
fig.savefig("costs_training_and_qem.png", dpi=600)
plt.close(fig)
print("\nwrote costs_stages.pdf / .png")

# =====================================================================
# resumen
# =====================================================================
print("\n=== (a) resumen training-time (mean over seeds) ===")
print(f"{'family':8s}  {'none':>12s}  {'trex':>12s}  {'reduction':>10s}")
print("-"*50)
for f in FAMS:
    n = list(data_a.get((f,"none"), {}).values())
    t = list(data_a.get((f,"trex"), {}).values())
    mn, mt = mean(n), mean(t)
    red = (mn-mt)/mn*100 if np.isfinite(mn) and np.isfinite(mt) and mn>0 else np.nan
    print(f"{FAM_PRETTY[f]:8s}  {mn:>12.3e}  {mt:>12.3e}  {red:>9.1f}%")

print("\n=== (b) resumen post-hoc QEM (mean over seeds) ===")
print(f"{'family':8s}  {'trex':>12s}  {'trex+zne':>12s}  {'reduction':>10s}")
print("-"*50)
for f in FAMS:
    t  = list(data_b.get((f,"trex"), {}).values())
    tz = list(data_b.get((f,"trex_zne"), {}).values())
    mt, mtz = mean(t), mean(tz)
    red = (mt-mtz)/mt*100 if np.isfinite(mt) and np.isfinite(mtz) and mt>0 else np.nan
    print(f"{FAM_PRETTY[f]:8s}  {mt:>12.3e}  {mtz:>12.3e}  {red:>9.1f}%")
