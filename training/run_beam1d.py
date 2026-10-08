#!/usr/bin/env python3
"""Beam1D-32 / FakeAthens campaign with a LOCAL VQLS objective.

The solver uses only C_L (full Pauli representation, ``atol=0``).  Each
SPSA two-point gradient is averaged and passed to Adam with a conservative
learning rate.  C_G and the FEM metrics are diagnostics only: they never
steer the optimizer.

Modes are deliberately limited to the measurements supported by the active
FakeAthens baseline:
  none       raw Pauli expectations;
  trex       readout twirling + TREX;
  trex_zne   TREX at every folded scale, then linear ZNE per observable.

DD and gate/Pauli twirling remain OFF.  The active ``nisq.Evaluator`` rejects
them in FakeAthens mode, and enabling them by modifying that guard would no
longer be a controlled comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

import beam1d_amg_hook_v2  # installs the reviewed cubic Beam1D AMG hook
import nisq


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def fake_args(shots: int, mode: str, scales, meas_rand: int) -> SimpleNamespace:
    """Arguments consumed by the current active ``nisq.Evaluator``."""
    uses_trex = mode in {"trex", "trex_zne"}
    return SimpleNamespace(
        entangler="brick", noise_model="fake-athens",
        # Suppression intentionally absent from every mode.
        dd=False, dd_sequence="XY4", twirl=False, twirl_randomizations=1,
        meas_twirl=uses_trex, meas_twirl_rand=meas_rand,
        noiseless=False, p1=0.0, p2=0.0, p_ro=0.0, ro_asym=0.0,
        shots=shots,
        # Used only by provenance; ZNE itself is implemented below so that
        # TREX is correctly performed at every scale and every support.
        zne_scales=list(map(float, scales)), zne_amplifier="fold",
        zne_fit="linear", zne_folding="global", mitiq=False,
    )


def ideal_args() -> SimpleNamespace:
    return SimpleNamespace(
        entangler="brick", noise_model="uniform",
        dd=False, dd_sequence="XY4", twirl=False, twirl_randomizations=1,
        meas_twirl=False, meas_twirl_rand=1,
        noiseless=True, p1=0.0, p2=0.0, p_ro=0.0, ro_asym=0.0, shots=256,
    )


def require_fake_athens_environment() -> dict:
    try:
        import qiskit
        import qiskit_aer
        import qiskit_ibm_runtime
        from qiskit_ibm_runtime.fake_provider import FakeAthensV2
    except ImportError as exc:
        raise RuntimeError(
            "Activate mitiq_py311_device before running this campaign."
        ) from exc
    return {
        "python": sys.version.split()[0],
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
        "fake_backend": FakeAthensV2().name,
    }


def estimate(cost_fn, n: int) -> dict:
    values = np.asarray([float(cost_fn()) for _ in range(n)], dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("a sampled cost was non-finite")
    return {
        "mean": float(values.mean()),
        "sem": float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "values": values.tolist(),
    }


def spsa_adam(cost, x0, *, iters: int, c: float, lr: float, avg: int,
              seed: int, log_every: int) -> dict:
    """SPSA gradient estimator -> Adam.

    ``avg`` averages independent *measurements* for each SPSA side before
    constructing the gradient.  It is not parameter averaging and it is not
    a mitigation method.  Adam's per-coordinate first update has size near
    ``lr``; for 35 parameters, 0.005 is intentionally conservative.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    for k in range(1, iters + 1):
        delta = rng.choice((-1.0, 1.0), size=x.size)
        plus = np.asarray([float(cost(x + c * delta)) for _ in range(avg)])
        minus = np.asarray([float(cost(x - c * delta)) for _ in range(avg)])
        fp, fm = float(plus.mean()), float(minus.mean())
        if not np.isfinite(fp + fm):
            raise FloatingPointError(f"non-finite SPSA cost at iteration {k}")
        grad = ((fp - fm) / (2.0 * c)) * delta
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        mh = m / (1.0 - 0.9 ** k)
        vh = v / (1.0 - 0.999 ** k)
        step = lr * mh / (np.sqrt(vh) + 1.0e-8)
        x = x - step
        # Rotation angles are periodic.  Wrapping prevents numerical drift
        # while preserving the ansatz exactly.
        x = (x + np.pi) % (2.0 * np.pi) - np.pi
        row = {
            "iteration": k, "f_plus": fp, "f_minus": fm,
            "paired_mean": 0.5 * (fp + fm),
            "side_sem_plus": float(plus.std(ddof=1) / np.sqrt(avg)) if avg > 1 else 0.0,
            "side_sem_minus": float(minus.std(ddof=1) / np.sqrt(avg)) if avg > 1 else 0.0,
            "gradient_norm": float(np.linalg.norm(grad)),
            "adam_step_norm": float(np.linalg.norm(step)),
        }
        history.append(row)
        if k == 1 or k % log_every == 0 or k == iters:
            print(f"  iter {k:03d}: Cpair={row['paired_mean']:.6g} "
                  f"|g|={row['gradient_norm']:.3g} |step|={row['adam_step_norm']:.3g}",
                  flush=True)
    return {"theta": x, "history": history}


def _folded_trex_evs(ev, circuit, shots: int) -> np.ndarray:
    """TREX expectation of every observable for one already-folded circuit."""
    if ev.trex_gamma is None:
        raise RuntimeError("TREX must be calibrated before TREX+ZNE")
    nrand = max(2, int(getattr(ev.args, "meas_twirl_rand", 8)))
    shots_per_randomization = max(int(shots) // nrand, 32)
    simulator = ev._backend(1.0)
    support_sum = None
    for _ in range(nrand):
        flip = int(ev.rng.integers(1 << ev.pe.nq))
        circuits = ev.pe.build(circuit, flip=flip)
        circuits = ev._compile_execution_circuits(circuits)
        result = simulator.run(
            circuits, shots=shots_per_randomization,
            seed_simulator=int(ev.rng.integers(2 ** 31)),
        ).result()
        current, _ = ev.pe.evs_per_support(result, flip=flip)
        support_sum = current if support_sum is None else {
            key: support_sum[key] + current[key] for key in support_sum
        }
    out = np.zeros(len(ev.obs), dtype=float)
    for (observable_index, support), value in support_sum.items():
        gamma = ev.trex_gamma.get(support, 1.0) if support else 1.0
        out[observable_index] += value / nrand / gamma
    # The native Evaluator maintains this accounting for raw/TREX calls.
    ev.total_shots += nrand * shots_per_randomization * ev.n_bases
    ev.n_calls += 1
    return out


def trex_zne_evs(ev, theta, scales, shots: int) -> np.ndarray:
    """TREX at every scale; linear ZNE of EACH observable; then combine."""
    try:
        from mitiq.zne.scaling import fold_global
    except ImportError as exc:
        raise RuntimeError("TREX+ZNE requires mitiq in mitiq_py311_device") from exc
    base = ev._bound(theta, False)
    observations = []
    for scale in scales:
        folded = fold_global(base, scale_factor=float(scale))
        observations.append(_folded_trex_evs(ev, folded, shots))
    y = np.asarray(observations, dtype=float)
    x = np.asarray(scales, dtype=float)
    # A ratio is nonlinear: fitting the scalar local cost would bias the
    # result.  Extrapolate numerator and denominator separately instead.
    return np.asarray([np.polyfit(x, y[:, j], deg=1)[1]
                       for j in range(y.shape[1])], dtype=float)


def make_reference(ev_exact, combine, path: Path, restarts: int, maxiter: int) -> np.ndarray:
    """Find a near-zero *ideal* local-cost control; never use it to start SPSA."""
    rng = np.random.default_rng(99173)
    best = None
    print("Building ideal d=5 reference control (L-BFGS-B; not an optimizer start)...",
          flush=True)
    for attempt in range(restarts):
        x0 = rng.uniform(-np.pi, np.pi, size=ev_exact.qc_log.num_parameters)
        result = minimize(
            lambda x: float(combine(ev_exact.exact_evs_observables(x))), x0,
            method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1.0e-14},
        )
        val = float(result.fun)
        if best is None or val < best[0]:
            best = (val, np.asarray(result.x, dtype=float))
        print(f"  ideal restart {attempt + 1}/{restarts}: C_L={val:.6e}", flush=True)
        if best[0] < 1.0e-10:
            break
    if best is None or not np.isfinite(best[0]):
        raise RuntimeError("ideal reference optimization failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, best[1])
    print(f"Saved ideal control theta: {path} (C_L={best[0]:.6e})", flush=True)
    return best[1]


def exact_bundle(theta, ev_local, ev_global, sysd) -> dict:
    """Correct, separate ideal-state diagnostics for one parameter vector."""
    return {
        "C_local_dense": float(ev_local.combine(ev_local.exact_evs(theta))),
        "C_local_pauli": float(ev_local.combine(ev_local.exact_evs_observables(theta))),
        "C_global_dense": float(ev_global.combine(ev_global.exact_evs(theta))),
        "C_global_pauli": float(ev_global.combine(ev_global.exact_evs_observables(theta))),
        # This is the FEM solution reconstructed from the IDEAL state prepared
        # by theta.  It is not a claim that a noisy density matrix has a
        # unique classical displacement field.
        "ideal_parameter_state_metrics": nisq.metrics_state(
            ev_local.statevector(theta), sysd, tag="ideal_parameter_state"
        ),
    }


def run_mode(args, sysd, obs_l, comb_l, mats_l, obs_g, comb_g, mats_g,
             reference_theta, environment) -> dict:
    uses_trex = args.mode in {"trex", "trex_zne"}
    cfg = fake_args(args.shots, args.mode, args.zne_scales, args.meas_rand)
    probe = nisq.Evaluator(sysd["nq"], args.depth, obs_l, comb_l, mats_l, cfg, seed=47001)
    ev_global = nisq.Evaluator(sysd["nq"], args.depth, obs_g, comb_g, mats_g,
                               ideal_args(), seed=47002)
    if reference_theta.size != probe.qc_log.num_parameters:
        raise ValueError("reference theta does not match brick depth")
    if uses_trex:
        print(f"Calibrating TREX ({args.n_cal_trex} circuits) for mode={args.mode}...", flush=True)
        gamma = probe.calibrate_trex(n_cal=args.n_cal_trex)
    else:
        gamma = None

    def one_cost(theta: np.ndarray) -> float:
        if args.mode == "none":
            return float(comb_l(probe.raw_evs(theta)))
        if args.mode == "trex":
            return float(comb_l(probe.trex_evs(theta)))
        if args.mode == "trex_zne":
            return float(comb_l(trex_zne_evs(probe, theta, args.zne_scales, args.shots)))
        raise ValueError(args.mode)

    # QEM control: a near-exact local solution evaluates mitigation bias where
    # the answer is known.  It is NOT used to initialize noisy optimization.
    control = {
        "exact": exact_bundle(reference_theta, probe, ev_global, sysd),
        "estimated": estimate(lambda: one_cost(reference_theta), args.reevals),
    }
    control["estimated_bias_vs_exact_pauli"] = (
        control["estimated"]["mean"] - control["exact"]["C_local_pauli"]
    )

    runs = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        theta0 = rng.normal(0.0, args.init_scale, size=probe.qc_log.num_parameters)
        # Recreate per-seed evaluator so all modes see identical parameter
        # starts yet independent, reproducible sampling streams.
        ev = nisq.Evaluator(sysd["nq"], args.depth, obs_l, comb_l, mats_l, cfg,
                            seed=100000 + seed)
        if uses_trex:
            ev.trex_gamma = dict(gamma)

        def seeded_cost(theta: np.ndarray) -> float:
            if args.mode == "none":
                return float(comb_l(ev.raw_evs(theta)))
            if args.mode == "trex":
                return float(comb_l(ev.trex_evs(theta)))
            return float(comb_l(trex_zne_evs(ev, theta, args.zne_scales, args.shots)))

        print(f"mode={args.mode}, seed={seed}: local SPSA-gradient -> Adam "
              f"({args.iters} iters, avg={args.avg}, lr={args.adam_lr})", flush=True)
        initial = {
            "theta": theta0.tolist(),
            "estimated": estimate(lambda: seeded_cost(theta0), args.reevals),
            "exact": exact_bundle(theta0, ev, ev_global, sysd),
        }
        sol = spsa_adam(seeded_cost, theta0, iters=args.iters, c=args.spsa_c,
                        lr=args.adam_lr, avg=args.avg, seed=seed,
                        log_every=args.log_every)
        thetaf = sol["theta"]
        final = {
            "theta": thetaf.tolist(),
            "estimated": estimate(lambda: seeded_cost(thetaf), args.reevals),
            "exact": exact_bundle(thetaf, ev, ev_global, sysd),
        }
        initial_m = initial["exact"]["ideal_parameter_state_metrics"]
        final_m = final["exact"]["ideal_parameter_state_metrics"]
        print(f"  final: C_L(est) {initial['estimated']['mean']:.5g} -> "
              f"{final['estimated']['mean']:.5g}; F {initial_m['F']:.5f} -> "
              f"{final_m['F']:.5f}; res {initial_m['res_pre']:.3e} -> "
              f"{final_m['res_pre']:.3e}", flush=True)
        np.save(args.out.parent / f"theta_{args.mode}_seed{seed}.npy", thetaf)
        runs.append({"seed": int(seed), "initial": initial, "final": final,
                     "spsa_adam_history": sol["history"],
                     "sampling": {"n_calls": int(ev.n_calls),
                                  "total_shots_accounted": int(ev.total_shots)}})

    return {
        "experiment": "beam1d32_fake_athens_local_spsa_to_adam",
        "environment": environment,
        "configuration": {
            "family": "beam1d", "dof": 32, "nq": int(sysd["nq"]),
            "preconditioner": "beam1d_amg_hook_v2_cubic_equilibrated",
            "kappa_pre": float(sysd["kappa_pre"]),
            "cost_used_by_optimizer": "local", "cost_type": "local",
            "entangler": "brick", "depth": args.depth,
            "pauli_atol": 0.0, "backend": "FakeAthensV2 / AerSimulator.from_backend",
            "shots_per_cost_estimate": args.shots,
            "dd": False, "gate_pauli_twirling": False,
            "mode": args.mode, "measurement_twirling_trex": uses_trex,
            "zne": args.mode == "trex_zne",
            "zne_scales": list(map(float, args.zne_scales)) if args.mode == "trex_zne" else [],
            "zne_fit": "linear per observable (not scalar ratio)",
            "pauli_terms": [{"full": int(a), "kept": int(b)} for a, b in
                            nisq.make_cost_observables(sysd["A"], sysd["b"], sysd["nq"],
                                                       atol=0.0, cost_type="local")[3]],
        },
        "optimizer": {
            "method": "two-point SPSA gradient -> Adam",
            "spsa_c": args.spsa_c, "adam_lr": args.adam_lr,
            "adam_beta1": 0.9, "adam_beta2": 0.999,
            "averages_per_spsa_side": args.avg, "iterations": args.iters,
            "start": "zero-centred normal; never the ideal control theta",
            "init_scale": args.init_scale,
        },
        "trex_gamma": gamma,
        "control_near_ideal_theta": control,
        "runs": runs,
        "metric_interpretation": (
            "F/res_pre/l2_fis/eK_rel are reconstructed from the ideal state for the final theta. "
            "They are independent physical validation metrics, not noisy-cost inputs."
        ),
        "acceptance_rule": {
            "target_F_min": args.target_fidelity,
            "target_res_pre_max": args.target_res,
            "target_l2_fis_max": args.target_l2,
            "local_certificate_for_Cglobal_le_0.01": 0.01 / sysd["nq"],
            "rule": (
                "A mode is useful only if it reduces bias on the near-ideal control and "
                "its final theta improves physical F, residual and L2.  Lower C_L alone is never success."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["none", "trex", "trex_zne"], default="none")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--avg", type=int, default=2)
    ap.add_argument("--spsa-c", type=float, default=0.10)
    ap.add_argument("--adam-lr", type=float, default=0.005)
    ap.add_argument("--init-scale", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7101, 7102])
    ap.add_argument("--reevals", type=int, default=6)
    ap.add_argument("--n-cal-trex", type=int, default=24)
    ap.add_argument("--meas-rand", type=int, default=8)
    ap.add_argument("--zne-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--reference-theta", type=Path, default=None)
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--ideal-restarts", type=int, default=5)
    ap.add_argument("--ideal-maxiter", type=int, default=1000)
    ap.add_argument("--target-fidelity", type=float, default=0.99)
    ap.add_argument("--target-res", type=float, default=0.10)
    ap.add_argument("--target-l2", type=float, default=0.10)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.depth < 1 or args.shots < 64 or args.iters < 1 or args.avg < 1
            or args.spsa_c <= 0 or args.adam_lr <= 0 or args.init_scale <= 0
            or args.reevals < 2 or args.meas_rand < 2 or args.log_every < 1):
        raise ValueError("invalid campaign parameters")
    if args.mode == "trex_zne" and len(args.zne_scales) < 2:
        raise ValueError("TREX+ZNE needs at least two noise scales")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    environment = require_fake_athens_environment()
    sysd = nisq.prepare("beam1d", 32, precond="amg")
    obs_l, comb_l, mats_l, _ = nisq.make_cost_observables(
        sysd["A"], sysd["b"], sysd["nq"], atol=0.0, cost_type="local")
    obs_g, comb_g, mats_g, _ = nisq.make_cost_observables(
        sysd["A"], sysd["b"], sysd["nq"], atol=0.0, cost_type="global")
    ev_exact = nisq.Evaluator(sysd["nq"], args.depth, obs_l, comb_l, mats_l,
                              ideal_args(), seed=44001)

    ref_path = args.reference_theta or (args.out.parent / "theta_ideal_control.npy")
    if ref_path.exists():
        reference_theta = np.asarray(np.load(ref_path), dtype=float).ravel()
        print(f"Loaded ideal control theta: {ref_path}", flush=True)
    else:
        reference_theta = make_reference(ev_exact, comb_l, ref_path,
                                         args.ideal_restarts, args.ideal_maxiter)
    if reference_theta.size != ev_exact.qc_log.num_parameters:
        raise ValueError("ideal control theta does not match current depth")
    ref_metrics = nisq.metrics_state(ev_exact.statevector(reference_theta), sysd,
                                     tag="ideal_control")
    ref_cost = float(comb_l(ev_exact.exact_evs_observables(reference_theta)))
    print(f"CONFIG: Beam1D-32 | brick d={args.depth} | local cost | atol=0 | "
          f"hook_v2 kappa={sysd['kappa_pre']:.6g}", flush=True)
    print(f"IDEAL CONTROL (diagnostic only): C_L={ref_cost:.6e}, "
          f"F={ref_metrics['F']:.8f}, res={ref_metrics['res_pre']:.3e}", flush=True)
    if args.build_reference:
        payload = {"status": "reference-built", "reference_theta": str(ref_path),
                   "reference_cost_local_pauli": ref_cost,
                   "reference_metrics": ref_metrics,
                   "kappa_pre": float(sysd["kappa_pre"]), "environment": environment}
        args.out.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")
        return

    payload = run_mode(args, sysd, obs_l, comb_l, mats_l, obs_g, comb_g, mats_g,
                       reference_theta, environment)
    payload["reference_theta"] = str(ref_path)
    args.out.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
