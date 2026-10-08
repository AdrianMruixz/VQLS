#!/usr/bin/env python3
"""Figuras Nature npj QI  4 y 5 qubits, 4 paneles, cuadros cerrados."""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

# ============ WHITELIST ============
SEED_WHITELIST = {
    "Heat1D": {"none": [1001, 3003, 4001], "trex": [1001, 3003, 4001]},
    "Heat2D": {"none": [1001, 3003, 4001], "trex": [1001, 3003, 4001]},
    "Beam2D": {"none": [1001, 3003, 4001], "trex": [1001, 3003, 4001]},
    "Beam1D": {"none": [8103, 8203, 8303], "trex": [8103, 8203, 8303]},
}
# ===================================

plt.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":8,"axes.labelsize":8,"axes.titlesize":8,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
    "axes.spines.top":True,"axes.spines.right":True,
    "axes.spines.left":True,"axes.spines.bottom":True,
    "axes.linewidth":0.8,"axes.grid":False,"axes.axisbelow":True,
    "axes.edgecolor":"black",
    "lines.linewidth":1.2,"lines.markersize":2.8,
    "legend.frameon":False,"figure.facecolor":"white",
    "savefig.facecolor":"white","savefig.bbox":"tight",
    "savefig.pad_inches":0.02,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.major.size":3,"ytick.major.size":3,
    "xtick.minor.size":1.5,"ytick.minor.size":1.5,
})

HOME = Path.home()
OUT = Path.cwd()/"none_trex_trajectory_figures"
OUT.mkdir(exist_ok=True)

FAMILY_STYLE = {"Heat1D":"#0072B2","Heat2D":"#E69F00",
                "Beam2D":"#009E73","Beam1D":"#CC79A7"}
ARM_STYLE = {"none":dict(ls="-",marker="o",label="none"),
             "trex":dict(ls="--",marker="D",label="TREX")}
FAM_CANON = {"heat1d":"Heat1D","heat2d":"Heat2D",
             "beam2d":"Beam2D","beam1d":"Beam1D"}
CAMPAIGN   = HOME/"other_families_trex_campaign"
BEAM1D_FIN = HOME/"beam1d_final_results"
BEAM1D_OPT = HOME/"beam1d_optimized"

def canon(x): return re.sub(r"[^a-z0-9]","",str(x).lower())
def find_col(df,names):
    cols={canon(c):c for c in df.columns}
    for n in names:
        if canon(n) in cols: return cols[canon(n)]
    return None
def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0],
        errors="coerce")
def infer_arm(p):
    if p.parent.name.lower() in ("none","trex"): return p.parent.name.lower()
    for q in p.parts:
        ql=q.lower()
        if ql.startswith("none_seed") or ql.startswith("raw_none_"): return "none"
        if ql.startswith("trex_seed") or ql.startswith("trex_only_"): return "trex"
    return None

# ---------------- Load ----------------
by_key={}
for root in [CAMPAIGN,BEAM1D_FIN,BEAM1D_OPT]:
    if not root.exists(): continue
    for p in root.rglob("metrics_seed*.csv"):
        sp=str(p).lower()
        if any(t in sp for t in ("smoke","_ckpt","test","__pycache__")): continue
        try: head=pd.read_csv(p,nrows=1)
        except Exception: continue
        if "family" not in head.columns or "seed" not in head.columns: continue
        fam=FAM_CANON.get(str(head["family"].iloc[0]).lower())
        if fam is None: continue
        arm=infer_arm(p)
        if arm is None: continue
        try: seed=int(head["seed"].iloc[0])
        except Exception: continue
        if seed not in SEED_WHITELIST.get(fam,{}).get(arm,[]): continue
        mt=p.stat().st_mtime; k=(fam,arm,seed)
        if k not in by_key or mt>by_key[k][0]: by_key[k]=(mt,p)

print(f"[discovery] {len(by_key)} CSVs")
by_fa=defaultdict(list)
for (f,a,s) in by_key: by_fa[(f,a)].append(s)
for k in sorted(by_fa): print(f"  {k[0]:8s} {k[1]:5s}  seeds={sorted(by_fa[k])}")

frames=[]
for (fam,arm,seed),(_,path) in by_key.items():
    df=pd.read_csv(path); df.columns=[str(c).strip() for c in df.columns]
    c_dof=find_col(df,["dof","ndof","n_dof","level","nivel"])
    c_it =find_col(df,["iter","iteration","it","step"])
    c_F  =find_col(df,["F_coherent_theta","F","fidelity"])
    c_r  =find_col(df,["res_pre_coherent_theta","res_pre","residual"])
    c_l2 =find_col(df,["l2_fis_coherent_theta","l2_fis","L2","l2_pre"])
    c_C  =find_col(df,["C_spsa_center","C_exact","C_full_exact","C_local_pauli"])
    if None in (c_dof,c_it,c_F,c_r,c_l2):
        print(f"  ! skip {path}"); continue
    clean=pd.DataFrame({
        "family":fam,"arm":arm,"seed":seed,
        "dof":to_num(df[c_dof]),"iteration":to_num(df[c_it]),
        "F":to_num(df[c_F]),"residual":to_num(df[c_r]),
        "L2":to_num(df[c_l2]),
        "C":to_num(df[c_C]) if c_C else np.nan,
    }).dropna(subset=["dof","iteration","F","residual","L2"])
    frames.append(clean)

data=pd.concat(frames,ignore_index=True)
data=data[data.dof.isin([16,32])].copy()
data["iteration"]=data["iteration"]-data.groupby(
    ["family","arm","seed"])["iteration"].transform("min")
data=data.sort_values(["family","arm","seed","iteration"])
print(f"[data] {len(data)} rows")

# ---------------- mean ± std ----------------
def mean_over_seeds(metric, dof_fixed):
    rows=[]
    for fam in FAMILY_STYLE:
        for arm in ("none","trex"):
            s=data[(data.family==fam)&(data.arm==arm)&(data.dof==dof_fixed)]
            if s.empty: continue
            seeds=sorted(s["seed"].unique())
            it_max=int(s["iteration"].max())
            if it_max<=0: continue
            grid=np.arange(0,it_max+1)
            all_y=[]
            for sd in seeds:
                g=s[s.seed==sd].sort_values("iteration")
                x=g["iteration"].values.astype(float)
                y=g[metric].values.astype(float)
                ok=np.isfinite(x)&np.isfinite(y)
                if metric in ("residual","L2"): ok&=(y>0)
                x,y=x[ok],y[ok]
                if len(x)<2: continue
                sd_max=float(x[-1])
                yi=np.full(len(grid), np.nan)
                mask=grid<=sd_max
                if metric in ("residual","L2"):
                    yi[mask]=np.exp(np.interp(grid[mask],x,np.log(y)))
                else:
                    yi[mask]=np.interp(grid[mask],x,y)
                all_y.append(yi)
            if not all_y: continue
            C=np.vstack(all_y)
            mu=np.nanmean(C,axis=0)
            sd_=np.nanstd(C,axis=0,ddof=0)
            n_ok=np.sum(~np.isnan(C),axis=0)
            for it,m,ss,nn in zip(grid,mu,sd_,n_ok):
                if not np.isfinite(m): continue
                rows.append(dict(family=fam,arm=arm,dof=dof_fixed,
                                 iteration=int(it),mean=float(m),
                                 std=float(ss),n=int(nn)))
    return pd.DataFrame(rows)

# ---------------- style helpers ----------------
def _style_axes(ax, x_max_data, logy=False, dof=None):
    # X: 180 (4q) o 250 (5q), termina exacto
    if dof == 16:
        x_max = 180
        xticks = [0, 45, 90, 135, 180]
    elif dof == 32:
        x_max = 250
        xticks = [0, 50, 100, 150, 200, 250]
    else:
        x_max = x_max_data
        xticks = None
    ax.set_xlim(0, x_max)
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])

    # Y: incluye limites superior e inferior con número
    y_lo, y_hi = ax.get_ylim()
    if logy:
        lo_exp = int(np.floor(np.log10(y_lo)))
        hi_exp = int(np.ceil(np.log10(y_hi)))
        decades = [10.0**e for e in range(lo_exp, hi_exp+1)
                   if 10.0**e >= y_lo and 10.0**e <= y_hi]
        if not decades:
            decades = [y_lo]
        if not any(abs(y_hi - d)/d < 1e-9 for d in decades):
            decades = decades + [y_hi]
        ax.set_yticks(sorted(decades))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=(0.2,0.5), numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # décadas limpias: "10^-1"; valores intermedios: "9.0x10^-1"
        def _sci_fmt(v, _pos=None):
            if v <= 0: return "0"
            s = f"{v:.1e}"          # "1.0e-01"
            m_str, e_str = s.split("e")
            e = int(e_str)
            m = float(m_str)
            if abs(m - 1.0) < 1e-6:
                if e == 0: return "1"
                return r"$10^{%d}$" % e
            return r"$%.1f\times10^{%d}$" % (m, e)
        ax.yaxis.set_major_formatter(FuncFormatter(_sci_fmt))
    else:
        n = 4
        step = (y_hi - y_lo) / n if y_hi > y_lo else 1.0
        ax.set_yticks([y_lo + i*step for i in range(n+1)])

    # Solo bottom/left con ticks; top/right sin marcas
    ax.tick_params(axis="x", which="major", bottom=True, top=False,
                   labelbottom=True, labeltop=False,
                   direction="in", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", left=True, right=False,
                   labelleft=True, labelright=False,
                   direction="in", length=3, width=0.8)
    ax.tick_params(axis="x", which="minor", bottom=True, top=False,
                   direction="in", length=1.5, width=0.6)
    ax.tick_params(axis="y", which="minor", left=True, right=False,
                   direction="in", length=1.5, width=0.6)
    for sp in ("top","right","bottom","left"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.8)
    ax.set_box_aspect(1)

def _panel_label(ax, letter):
    ax.text(0.97, 0.97, f"{letter})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="right")

def draw_mean_panel(ax, dof, metric, ylab, logy, fixed_ylim, letter):
    mdf = mean_over_seeds(metric, dof)
    if mdf.empty: ax.set_visible(False); return
    if fixed_ylim is not None: y_lo,y_hi=fixed_ylim
    else:
        v=mdf["mean"].dropna()
        if logy: v=v[v>0]
        if v.empty: ax.set_visible(False); return
        y_lo,y_hi=float(v.min()),float(v.max())
        pad=0.08*(y_hi-y_lo) if y_hi>y_lo else 0.1
        y_lo-=pad; y_hi+=pad
        if logy and y_lo<=0: y_lo=float(v[v>0].min())/1.5
    for fam in FAMILY_STYLE:
        for arm in ("none","trex"):
            g=mdf[(mdf.family==fam)&(mdf.arm==arm)].sort_values("iteration")
            if logy: g=g[g["mean"]>0]
            if g.empty: continue
            st=ARM_STYLE[arm]
            ax.plot(g["iteration"],g["mean"],
                    color=FAMILY_STYLE[fam],linestyle=st["ls"],
                    linewidth=1.2,marker=st["marker"],markersize=2.5,
                    markevery=max(1,len(g)//8),
                    markerfacecolor="white",markeredgewidth=0.8)
            lo=g["mean"]-g["std"]; hi=g["mean"]+g["std"]
            if logy: lo=np.clip(lo,1e-30,None)
            ax.fill_between(g["iteration"],lo,hi,
                            color=FAMILY_STYLE[fam],alpha=0.15,linewidth=0)
    if logy: ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel(ylab)
    ax.set_ylim(y_lo,y_hi)
    _style_axes(ax, int(mdf["iteration"].max()), logy=logy, dof=dof)
    _panel_label(ax, letter)

def draw_spag_panel(ax, dof, metric, ylab, logy, fixed_ylim, letter):
    sub=data[(data.dof==dof)][["iteration",metric,"family","arm","seed"]].dropna()
    if logy: sub=sub[sub[metric]>0]
    if sub.empty: ax.set_visible(False); return
    if fixed_ylim is not None: y_lo,y_hi=fixed_ylim
    else:
        v=sub[metric]
        y_lo,y_hi=float(v.min()),float(v.max())
        pad=0.08*(y_hi-y_lo) if y_hi>y_lo else 0.1
        y_lo-=pad; y_hi+=pad
        if logy and y_lo<=0: y_lo=float(v[v>0].min())/1.5
    for fam in FAMILY_STYLE:
        for arm in ("none","trex"):
            g_all=sub[(sub.family==fam)&(sub.arm==arm)]
            if g_all.empty: continue
            for sd in sorted(g_all["seed"].unique()):
                g=g_all[g_all.seed==sd].sort_values("iteration")
                if g.empty: continue
                ax.plot(g["iteration"],g[metric],
                        color=FAMILY_STYLE[fam],
                        linestyle=ARM_STYLE[arm]["ls"],
                        linewidth=0.7,alpha=0.55)
    if logy: ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel(ylab)
    ax.set_ylim(y_lo,y_hi)
    _style_axes(ax, int(data[data.dof==dof]["iteration"].max()), logy=logy, dof=dof)
    _panel_label(ax, letter)

def _handles():
    fam_h=[Line2D([0],[0],color=FAMILY_STYLE[f],lw=1.2,label=f) for f in FAMILY_STYLE]
    arm_h=[Line2D([0],[0],color="#333333",lw=1.2,
                  linestyle=ARM_STYLE[a]["ls"],
                  marker=ARM_STYLE[a]["marker"],markersize=2.8,
                  markerfacecolor="white",markeredgewidth=0.8,
                  label=ARM_STYLE[a]["label"]) for a in ("none","trex")]
    return fam_h,arm_h

PANELS = [
    ("F",        "Fidelity",   False, (0.0,1.0)),
    ("residual", "Residual",   True,  None),
    ("L2",       "L2",         True,  None),
    ("C",        r"Cost function  $C$", True, None),
]

def make_figure(dof, mode, fname):
    fig,axes=plt.subplots(2,2,figsize=(7.5,7.5),layout="constrained")
    axes=axes.ravel()
    for k,(metric,ylab,logy,ylim) in enumerate(PANELS):
        ax=axes[k]
        letter=chr(97+k)
        if mode=="mean":
            draw_mean_panel(ax,dof,metric,ylab,logy,ylim,letter)
        else:
            draw_spag_panel(ax,dof,metric,ylab,logy,ylim,letter)
    fam_h,arm_h=_handles()
    fig.legend(handles=fam_h+arm_h,loc="upper center",ncol=6,
               frameon=False,bbox_to_anchor=(0.5,1.02),fontsize=8)
    fig.savefig(OUT/fname); plt.close(fig)
    print(f"wrote {fname}")

make_figure(16, "mean", "4_qubits_trajectories_mean.pdf")
make_figure(32, "mean", "5_qubits_trajectories_mean.pdf")
make_figure(16, "spag", "4_qubits_trajectories_all_seeds.pdf")
make_figure(32, "spag", "5_qubits_trajectories_all_seeds.pdf")

print(f"\nSaved in:\n{OUT}")
