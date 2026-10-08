#!/usr/bin/env python3
"""Post-hoc none/TREX/TREX+ZNE assessment of one warm-start final theta."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import beam1d_amg_hook_v2
import nisq
from beam1d32_athens_local_campaign import fake_args, ideal_args, estimate, jsonable, trex_zne_evs

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--shots",type=int,default=1024); ap.add_argument("--reps",type=int,default=8)
    ap.add_argument("--n-cal",type=int,default=24); ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--datadir",type=Path,default=Path("."))
    ap.add_argument("--require-cluster-files",action="store_true")
    a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    theta=np.asarray(np.load(a.theta),float).ravel(); sysd=nisq.prepare(
      "beam1d",32,precond="amg",datadir=str(a.datadir))
    src=(sysd.get("meta",{}) or {}).get("src","?")
    if a.require_cluster_files and src=="builtin_fem":
      raise RuntimeError(f"cluster K/f files not found in {a.datadir.resolve()}")
    obs,comb,mats,counts=nisq.make_cost_observables(sysd["A"],sysd["b"],sysd["nq"],atol=0.,cost_type="local")
    obg,cg,mg,_=nisq.make_cost_observables(sysd["A"],sysd["b"],sysd["nq"],atol=0.,cost_type="global")
    exact=nisq.Evaluator(5,5,obs,comb,mats,ideal_args(),seed=a.seed)
    eg=nisq.Evaluator(5,5,obg,cg,mg,ideal_args(),seed=a.seed)
    if theta.size != exact.qc_log.num_parameters: raise ValueError("theta is not Beam1D32 brick d=5")
    result={"theta_file":str(a.theta),"matrix_source":src,"exact":{"C_local_dense":float(comb(exact.exact_evs(theta))),
      "C_local_pauli":float(comb(exact.exact_evs_observables(theta))),"C_global_dense":float(cg(eg.exact_evs(theta))),
      "metrics":nisq.metrics_state(exact.statevector(theta),sysd,tag="warmstart_final")},"modes":{}}
    for mode in ("none","trex","trex_zne"):
      ev=nisq.Evaluator(5,5,obs,comb,mats,fake_args(a.shots,mode,[1.,1.5,2.],8),seed=a.seed+100*len(result["modes"]))
      if mode in ("trex","trex_zne"): gamma=ev.calibrate_trex(n_cal=a.n_cal)
      else: gamma=None
      if mode=="none": f=lambda:float(comb(ev.raw_evs(theta)))
      elif mode=="trex": f=lambda:float(comb(ev.trex_evs(theta)))
      else: f=lambda:float(comb(trex_zne_evs(ev,theta,[1.,1.5,2.],a.shots)))
      stat=estimate(f,a.reps); result["modes"][mode]={"estimate":stat,"bias_vs_exact_pauli":stat["mean"]-result["exact"]["C_local_pauli"],"trex_gamma":gamma}
      print(f"{mode}: C_L={stat['mean']:.6e} +/- {stat['sem']:.2e} bias={result['modes'][mode]['bias_vs_exact_pauli']:+.3e}",flush=True)
    a.out.write_text(json.dumps(jsonable(result),indent=2,allow_nan=False)+"\n"); print(f"Saved: {a.out}")
if __name__=="__main__": main()
