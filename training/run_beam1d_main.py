#!/usr/bin/env python3
# Definitive Beam1D FakeAthens warm-start runner.
"""Multigrid VQLS with provenance-preserving noisy diagnostics.

The optimizer always uses the selected *measured* Pauli objective.  Exact
statevector quantities are saved only as offline diagnostics of the theta
found by that optimizer; they are never used to update it.
"""
import argparse, json, time, csv, os, numpy as np
from scipy.optimize import minimize
import nisq
import beam1d_amg_hook_v2  # install the validated cubic Beam1D SA-AMG hook

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="beam1d", choices=["beam1d"])
    p.add_argument("--dof-target", type=int, default=32)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--pauli-atol", type=float, default=0.0)
    p.add_argument("--cost-type", default="local")
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr-levels", type=float, nargs=3, default=None,
                   metavar=("LR8", "LR16", "LR32"),
                   help="Optional Adam learning rates for the 8, 16 and 32 DOF "
                        "levels. Overrides --lr level by level.")
    p.add_argument("--iters-coarse", type=int, default=40)
    p.add_argument("--iters-refine", type=int, default=80)
    p.add_argument("--iters-levels", type=int, nargs=3, default=None,
                   metavar=("IT8", "IT16", "IT32"),
                   help="Optional SPSA+Adam iteration counts for 8, 16 and 32 "
                        "DOF. Overrides --iters-coarse/--iters-refine level by level.")
    p.add_argument("--jitter", type=float, default=0.02)
    p.add_argument("--max-step", type=float, default=0.03,
                   help="Maximum L2 norm of one Adam update.")
    p.add_argument("--coarse-init", default="lbfgs",
                   choices=["lbfgs", "normal"],
                   help="8-DOF initialization. L-BFGS minimizes only the exact "
                        "local VQLS objective; it never uses FEM fidelity/L2.")
    p.add_argument("--lbfgs-restarts", type=int, default=4)
    p.add_argument("--lbfgs-maxiter", type=int, default=500)
    p.add_argument("--mitigation", default="none",
                   choices=["none", "zne", "trex", "trex_zne"],
                   help="Estimador de mitigacion. Independiente de DD/twirling "
                        "(que son SUPRESION, no mitigacion). "
                        "'zne' = ZNE sin correccion de lectura; "
                        "'trex_zne' = TREX calibrado y DESPUES extrapolacion.")
    p.add_argument("--noise-model", default="fake-athens",
                   choices=["uniform", "fake-athens"],
                   help="'fake-athens' usa el snapshot FakeAthensV2 de 5 qubits "
                        "(via AerSimulator.from_backend). 'uniform' usa make_noise().")
    p.add_argument("--zne-scales", type=int, nargs="+", default=[1, 3, 5],
                   help="Factores de folding (enteros IMPARES: fold_cx exige 2k+1).")
    p.add_argument("--zne-fit", default="exp_fixed",
                   choices=["linear", "exp_fixed", "richardson"],
                   help="Extrapolador. richardson con 3 puntos es un polinomio de "
                        "grado 2: minimo sesgo pero MAXIMA varianza bajo ruido de "
                        "disparo. exp_fixed ancla la asintota en Tr(O)/2^n (punto "
                        "fijo del canal despolarizante) y es el mas estable aqui.")
    p.add_argument("--datadir", default=".",
                   help="Carpeta con los .npz de K y f del cluster.")
    p.add_argument("--require-cluster-files", action="store_true",
                   help="ABORTA si nisq.load_system cae al FEM interno en vez de "
                        "cargar tus .npz. Uselo SIEMPRE para resultados del paper.")
    p.add_argument("--n-cal-trex", type=int, default=24,
                   help="Circuitos de calibracion TREX (estados de base "
                        "computacional aleatorios). Coste aparte del optimizador.")
    p.add_argument("--dry-run", action="store_true",
                   help="Imprime el presupuesto de disparos y sale sin optimizar.")
    p.add_argument("--allow-inloop-zne", action="store_true",
                   help="Explicit override for research-only in-loop ZNE. The "
                        "definitive campaign keeps ZNE post hoc.")
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--theta-seed", type=int, default=1001)
    p.add_argument("--mode", default="noisy_trex",
                   choices=["ideal", "noisy_none", "noisy_trex"])
    p.add_argument("--dd", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable synthetic XY4 dynamical decoupling (recorded as suppression, not mitigation).")
    p.add_argument("--gate-twirl", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable Pauli gate twirling (recorded as suppression, not mitigation).")
    p.add_argument("--meas-twirl", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable measurement twirling; required by the in-repo TREX calibration model.")
    p.add_argument("--spsa-avg", type=int, default=8)
    p.add_argument("--spsa-c0", type=float, default=0.20)
    p.add_argument("--spsa-gamma", type=float, default=0.101)
    p.add_argument("--json-out", default=None)
    p.add_argument("--theta-out", default=None,
                   help="Save the returned target-DOF theta as a .npy file.")

    # Diagnostics do NOT affect the optimizer.
    # They are evaluated from the exact statevector at the
    # parameters found by noisy SPSA+Adam.
    p.add_argument(
        "--metrics-csv",
        default=None,
        help="CSV output for physical/solver diagnostics"
    )
    p.add_argument(
        "--diag-every",
        type=int,
        default=5,
        help="Evaluate and save physical diagnostics every N iterations"
    )

    p.add_argument(
        "--archive-npz",
        default=None,
        help="NPZ archive containing theta, psi, physical solutions and diagnostics"
    )

    return p.parse_args()

def make_evaluator_args(cli):
    """Construye el namespace que consume `nisq.Evaluator`.

    Con `--noise-model fake-athens` se fuerzan los guardarrailes del snapshot;
    no son preferencias de estilo, son condiciones que `Evaluator` valida y que
    de otro modo (a) lanzan ValueError o (b) producen un estimado "mitigado"
    que en realidad no vio el ruido del dispositivo.
    """
    class A: pass
    a = A()
    a.entangler, a.dd_sequence = "brick", "XY4"
    a.noise_model = cli.noise_model
    a.noiseless = (cli.mode == "ideal")
    a.p1, a.p2, a.p_ro, a.ro_asym = 5e-4, 3.5e-3, 1.5e-2, 0.3
    a.shots = int(cli.shots)
    a.dd = bool(cli.dd)
    a.twirl = bool(cli.gate_twirl)
    a.twirl_randomizations = 8
    a.meas_twirl, a.meas_twirl_rand = bool(cli.meas_twirl), 8
    a.mitigation = cli.mitigation
    a.zne_scales = list(cli.zne_scales)
    a.zne_amplifier = "fold"
    a.zne_fit = cli.zne_fit
    a.zne_folding, a.mitiq, a.pec_samples = "global", False, 16

    if cli.noise_model == "fake-athens":
        # Evaluator.__init__ lanza ValueError si dd o twirl estan activos con
        # el snapshot: ambos insertan puertas y contaminan el baseline.
        a.dd = False
        a.twirl = False
        # Evaluator._backend prohibe el escalado analitico con from_backend().
        a.zne_amplifier = "fold"
        # La ruta Mitiq de zne_evs no transpila las puertas de medicion al
        # target del snapshot (_compile_execution_circuits no se invoca), asi
        # que Aer no les aplica ruido. Se usa la ruta propia con fold_cx.
        a.mitiq = False
        # TREX del repo exige twirl de medicion para que <Z^m> -> gamma_m<Z^m>
        # sea exacto con lectura asimetrica (van den Berg et al. 2022).
        a.meas_twirl = True

    for s in a.zne_scales:
        if int(s) % 2 != 1 or int(s) < 1:
            raise ValueError(
                f"--zne-scales debe ser entero impar >=1 (fold_cx usa 2k+1); "
                f"recibido {s}.")
    return a


def make_logical_ideal_args():
    """Evaluator arguments with no device layout and no noise.

    FakeAthens maps 3/4 logical qubits onto a five-qubit target.  Exact Pauli
    diagnostics must therefore use a separate logical evaluator, otherwise an
    8- or 16-component state is multiplied by a 32-dimensional observable.
    """
    class A: pass
    a = A()
    a.entangler, a.dd_sequence = "brick", "XY4"
    a.noise_model = "uniform"
    a.noiseless = True
    a.p1 = a.p2 = a.p_ro = a.ro_asym = 0.0
    a.shots = 256
    a.dd = False
    a.twirl = False
    a.twirl_randomizations = 1
    a.meas_twirl = False
    a.meas_twirl_rand = 1
    a.mitigation = "none"
    a.zne_scales = [1, 3, 5]
    a.zne_amplifier = "fold"
    a.zne_fit = "linear"
    a.zne_folding = "global"
    a.mitiq = False
    a.pec_samples = 1
    return a

def _sync_mode_mit(cli):
    # `mode` used to silently overwrite `--mitigation`, making `--mitigation
    # zne` impossible to execute.  Keep the legacy modes as unambiguous
    # aliases, but reject contradictory requests rather than relabelling them.
    expected = {"ideal": "none", "noisy_none": "none", "noisy_trex": "trex"}[cli.mode]
    # `trex_zne` y `zne` son brazos de la campana, no alias de un modo legado:
    # ambos corren bajo --mode noisy_none y no deben chocar con `expected`.
    if cli.mitigation in ("zne", "trex_zne"):
        if cli.mode != "noisy_none":
            raise ValueError(
                f"--mitigation {cli.mitigation!r} requiere --mode noisy_none "
                f"(recibido {cli.mode!r}).")
        cli.suppression = {"dd": bool(cli.dd), "gate_twirl": bool(cli.gate_twirl),
                           "measurement_twirl": bool(cli.meas_twirl)}
        return cli
    if cli.mitigation != "none" and cli.mitigation != expected:
        raise ValueError(
            f"Contradictory --mode {cli.mode!r} and --mitigation {cli.mitigation!r}. "
            "Use --mode noisy_none with --mitigation zne for noisy ZNE, or "
            "use the legacy noisy_trex mode only with --mitigation trex."
        )
    if cli.mode == "noisy_trex":
        cli.mitigation = "trex"
    elif cli.mode == "ideal":
        cli.mitigation = "none"
    cli.suppression = {
        "dd": bool(cli.dd),
        "gate_twirl": bool(cli.gate_twirl),
        "measurement_twirl": bool(cli.meas_twirl),
    }
    return cli

def dof_ladder(dof_target):
    dofs, d = [], 8
    while d <= dof_target:
        dofs.append(d); d *= 2
    if not dofs or dofs[-1] != dof_target:
        raise ValueError("--dof-target must be 8 * 2^k for this multigrid ladder")
    return dofs


def level_schedule(cli, dof, index):
    """Returns the configured Adam rate and iteration count for one level."""
    lr = (float(cli.lr_levels[index]) if cli.lr_levels is not None
          else float(cli.lr))
    iters = (int(cli.iters_levels[index]) if cli.iters_levels is not None
             else int(cli.iters_coarse if index == 0 else cli.iters_refine))
    return lr, iters

def prolong(theta_c, nq_c, depth, jitter, rng):
    nq_f = nq_c + 1
    P_f = (depth + 1) * nq_f
    theta_f = np.zeros(P_f)
    for layer in range(depth + 1):
        s_c = layer * nq_c
        s_f = layer * nq_f
        theta_f[s_f:s_f+nq_c] = theta_c[s_c:s_c+nq_c]
        theta_f[s_f+nq_c] = rng.normal(0, jitter)
    return theta_f


DIAG_FIELDS = [
    "family",
    "seed",
    "dof",
    "iter",

    "F",
    "F_coherent_theta",
    "infidelity",

    "res_pre",
    "res_pre_coherent_theta",
    "l2_pre",

    "l2_fis",
    "l2_fis_coherent_theta",
    "linf_fis",

    "eK",
    "eK_rel",

    "cos_fis",

    "qoi_name",
    "qoi_q",
    "qoi_fem",
    "qoi_ana",
    "qoi_ana_nodal",
    "qoi_rel_vs_fem",
    "qoi_rel_vs_ana",

    "l2_vs_analitica",
    "l2_fem_vs_analitica",

    "C_exact",
    "C_full_exact",
    "C_trunc_exact",
    "C_spsa_center",
    "C_spsa_sem",

    "grad_norm",
    "spsa_avg",
    "step_norm",
    "time_s",
]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def append_diag_csv(path, row):
    """
    Save one diagnostic checkpoint.

    This function is diagnostic only:
    it never feeds information back to SPSA or Adam.
    """
    if path is None:
        return

    path = os.path.expanduser(path)
    folder = os.path.dirname(os.path.abspath(path))

    if folder:
        os.makedirs(folder, exist_ok=True)

    exists = os.path.exists(path)

    clean = {}

    for key in DIAG_FIELDS:
        value = row.get(key, "")

        if key in ("family", "qoi_name"):
            clean[key] = value
        elif key in ("seed", "dof", "iter", "spsa_avg"):
            try:
                clean[key] = int(value)
            except Exception:
                clean[key] = value
        else:
            clean[key] = _safe_float(value)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=DIAG_FIELDS
        )

        if not exists:
            writer.writeheader()

        writer.writerow(clean)


def optimize_level(sysd, nq, depth, theta0, cli, prefix="", level_lr=None,
                   level_iters=None):
    import hashlib
    _h = int(hashlib.md5(prefix.encode()).hexdigest()[:8], 16)
    rng_spsa = np.random.default_rng(cli.seed + _h % 100000)
    A = sysd["A"]
    b = sysd["b"]
    x_pre = sysd["x_pre"]
    obs, combine, mats, counts = nisq.make_cost_observables(
        A, b, nq, atol=cli.pauli_atol, cost_type=cli.cost_type, atol_mode="rel")
    ev = nisq.Evaluator(nq, depth, obs, combine, mats,
                        make_evaluator_args(cli), seed=cli.seed)
    # Independent logical observables: never reuse FakeAthens-laid-out Pauli
    # operators for exact diagnostics at 3 or 4 logical qubits.
    obs_exact, combine_exact, mats_exact, _ = nisq.make_cost_observables(
        A, b, nq, atol=cli.pauli_atol, cost_type=cli.cost_type,
        atol_mode="rel")
    ev_exact = nisq.Evaluator(
        nq, depth, obs_exact, combine_exact, mats_exact,
        make_logical_ideal_args(), seed=cli.seed)
    
    # ----------------------------------------------------------------
    # SELECCION DEL ESTIMADOR                      [PATCH fakeathens+trex_zne]
    #
    # BUG CORREGIDO AQUI: `Evaluator.zne_evs` (ruta no-Mitiq) llama
    # internamente a `trex_evs`, y `trex_evs` hace fallback SILENCIOSO a
    # `raw_evs` si `trex_gamma is None`.  La version anterior solo llamaba a
    # `calibrate_trex()` en el brazo "trex", de modo que el brazo "zne"
    # publicaba ZNE SIN correccion de lectura bajo una etiqueta que sugeria
    # la pila completa.  Ahora la calibracion es explicita por brazo:
    #
    #   none      -> raw_evs                      (sin gamma)
    #   trex      -> calibrate + trex_evs         (con gamma)
    #   zne       -> zne_evs SIN gamma            (control: aisla ZNE)
    #   trex_zne  -> calibrate + zne_evs          (gamma aplicada en cada
    #                                              factor de folding y DESPUES
    #                                              se extrapola)
    # ----------------------------------------------------------------
    gamma_trex = None
    if cli.mode != "ideal" and cli.mitigation in ("trex", "trex_zne"):
        gamma_trex = ev.calibrate_trex(n_cal=cli.n_cal_trex)
        if gamma_trex is None:
            raise RuntimeError(
                f"--mitigation {cli.mitigation!r} pidio TREX pero "
                "calibrate_trex() devolvio None (sin ruido de lectura). "
                "Rehusando etiquetar un estimado crudo como mitigado.")
        bad = {m: g for m, g in gamma_trex.items()
               if not np.isfinite(g) or not (0.05 <= g <= 1.0)}
        if bad:
            raise RuntimeError(f"factores TREX invalidos: {bad}")
        print(f"{prefix}  TREX calibrado: "
              f"{len(gamma_trex)} soportes, "
              f"gamma in [{min(gamma_trex.values()):.4f}, "
              f"{max(gamma_trex.values()):.4f}]")

    if cli.mode == "ideal":
        # Ideal mode retains the *same Pauli-truncated objective* as noisy
        # mode, isolating noise without changing the optimization problem.
        evs_fn = lambda th: np.asarray(
            ev_exact.exact_evs_observables(th), float)
    elif cli.mitigation == "trex":
        evs_fn = lambda th: np.asarray(ev.trex_evs(th, shots=cli.shots), float)
    elif cli.mitigation in ("zne", "trex_zne"):
        evs_fn = lambda th: np.asarray(ev.zne_evs(th, shots=cli.shots), float)
    else:  # noisy_none
        evs_fn = lambda th: np.asarray(ev.raw_evs(th, shots=cli.shots), float)

    if theta0 is None:
        rng_init = np.random.default_rng(cli.theta_seed)
        if cli.coarse_init == "lbfgs":
            best = None
            bounds = [(-np.pi, np.pi)] * len(ev_exact.pv)
            for restart in range(max(1, int(cli.lbfgs_restarts))):
                x0 = rng_init.normal(0.0, 0.10, len(ev_exact.pv))
                result = minimize(
                    lambda th: float(combine_exact(ev_exact.exact_evs(th))),
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": int(cli.lbfgs_maxiter),
                             "ftol": 1.0e-12, "gtol": 1.0e-8},
                )
                value = float(combine_exact(ev_exact.exact_evs(result.x)))
                if best is None or value < best[0]:
                    best = (value, np.asarray(result.x, float).copy(), restart)
            theta = best[1]
            print(f"{prefix}  coarse L-BFGS: C_local={best[0]:.6e} "
                  f"restart={best[2]}")
        else:
            theta = rng_init.normal(0.0, 0.10, len(ev_exact.pv))
        maxiter = int(level_iters if level_iters is not None else cli.iters_coarse)
        F_init = nisq.fidelity(ev_exact.statevector(theta), x_pre)
    else:
        theta = np.array(theta0, float)
        maxiter = int(level_iters if level_iters is not None else cli.iters_refine)
        F_init = nisq.fidelity(ev_exact.statevector(theta), x_pre)

    m_ad = np.zeros_like(theta)
    v_ad = np.zeros_like(theta)

    # ========================================================
    # SAFE SPSA + ADAM STATE
    # ========================================================
    # Best point estimated from SPSA perturbations in noisy mode.  In ideal
    # mode the objective at theta is free and must be used instead: a mean of
    # C(theta +/- c Delta) is a different, c-dependent smoothed objective.
    # No exact fidelity/residual is used by the optimizer.
    best_measured_theta = theta.copy()
    best_measured_cost = np.inf
    bad_steps = 0

    # Maximum total Adam displacement in parameter space.
    max_step_norm = float(cli.max_step)

    history = []

    # ========================================================
    # COMPLETE OFFLINE ARCHIVE
    #
    # Analysis only. Nothing stored here is ever fed back
    # into SPSA, Adam, TREX or SAFE rollback.
    # ========================================================
    archive = {
        "iter": [],
        "theta": [],
        "psi": [],
        "u_q_pre": [],
        "u_q_fis": [],

        "F": [],
        "infidelity": [],
        "res_pre": [],
        "l2_pre": [],
        "l2_fis": [],
        "linf_fis": [],
        "eK": [],
        "eK_rel": [],
        "cos_fis": [],
        "qoi_q": [],
        "qoi_fem": [],
        "qoi_rel_vs_fem": [],

        "C_exact": [],
        "C_full_exact": [],
        "C_trunc_exact": [],
        "C_spsa_center": [],
        "C_spsa_sem": [],
        "grad_norm": [],
        "step_norm": [],
        "theta": [],
        "spsa_avg": [],
        "time_s": [],
    }

    print(f"{prefix}  nq={nq}  P={len(theta)}  Pauli={counts[0][1]}+{counts[1][1]}  "
          f"iters={maxiter}  lr={float(level_lr if level_lr is not None else cli.lr):.4g}  "
          f"mitigation={cli.mitigation}")
    if F_init is not None:
        print(f"{prefix}  F_init_prolong={F_init:.6f}")

    for it in range(1, maxiter + 1):
        t0 = time.time()
        P_ = len(theta)

        # ====================================================
        # SPSA GRADIENT ESTIMATION
        #
        # Number of independent SPSA perturbations is controlled
        # directly by --spsa-avg.
        #
        # With --spsa-avg 8:
        #     avg_now = 8
        # for every DOF level.
        #
        # Each SPSA perturbation requires TWO noisy cost
        # evaluations: C(theta+c*Delta), C(theta-c*Delta).
        # ====================================================
        avg_now = max(1, int(cli.spsa_avg))

        c_k = cli.spsa_c0 / ((it + 1)**cli.spsa_gamma)

        _grads = []
        _centers = []

        for _ in range(avg_now):
            delta = 2*rng_spsa.integers(0, 2, size=P_) - 1

            cp = float(
                combine(
                    np.asarray(
                        evs_fn(theta + c_k*delta),
                        float
                    )
                )
            )

            cm = float(
                combine(
                    np.asarray(
                        evs_fn(theta - c_k*delta),
                        float
                    )
                )
            )

            # SPSA gradient.
            # delta_i = +/-1, therefore 1/delta_i = delta_i.
            _grads.append(
                ((cp - cm)/(2.0*c_k))*delta
            )

            # Free estimate of cost near current theta.
            # Reuses Cp and Cm: ZERO extra quantum evaluations.
            _centers.append(
                0.5*(cp + cm)
            )

        grad = np.mean(_grads, axis=0)
        gnorm = float(np.linalg.norm(grad))

        C_center = float(np.mean(_centers))

        if len(_centers) > 1:
            C_center_sem = float(
                np.std(_centers, ddof=1) /
                np.sqrt(len(_centers))
            )
        else:
            C_center_sem = 0.0

        # ====================================================
        # DRIFT DETECTOR
        #
        # Noisy mode: C_center is the only free estimate available and is
        # explicitly a perturbation-centred SPSA surrogate.
        # Ideal mode: evaluate C(theta) exactly.  This is free of sampling and
        # avoids falsely treating a c-dependent smoothed cost as C(theta).
        # ====================================================
        if cli.mode == "ideal":
            C_accept = float(combine(evs_fn(theta)))
            C_accept_sem = 0.0
            acceptance_estimator = "exact_current_pauli_objective"
        else:
            C_accept = C_center
            C_accept_sem = C_center_sem
            acceptance_estimator = "spsa_perturbation_center_surrogate"

        if not np.isfinite(best_measured_cost):
            best_measured_cost = C_accept
            best_measured_theta = theta.copy()
            bad_steps = 0

        else:
            noise_margin = max(
                2.0*C_accept_sem,
                0.02*max(abs(best_measured_cost), 1.0e-12)
            )

            if C_accept < best_measured_cost:
                best_measured_cost = C_accept
                best_measured_theta = theta.copy()
                bad_steps = 0

            elif C_accept > best_measured_cost + noise_margin:
                bad_steps += 1

            else:
                bad_steps = 0

        # If three consecutive measured points are clearly worse,
        # restore the best measured point and damp Adam momentum.
        if bad_steps >= 3:
            theta = best_measured_theta.copy()

            m_ad *= 0.25
            v_ad *= 0.25

            bad_steps = 0

            print(
                f"{prefix}    [SAFE SPSA] rollback at it={it}: "
                f"Cbest({acceptance_estimator})={best_measured_cost:.3e}"
            )

            continue

        # ====================================================
        # ADAM
        # ====================================================
        b1, b2, eps = 0.9, 0.999, 1e-8

        m_ad = b1*m_ad + (1.0-b1)*grad
        v_ad = b2*v_ad + (1.0-b2)*(grad*grad)

        m_hat = m_ad / (1.0-b1**it)
        v_hat = v_ad / (1.0-b2**it)

        step = float(level_lr if level_lr is not None else cli.lr) * m_hat / (np.sqrt(v_hat) + eps)

        # ====================================================
        # TRUST REGION / STEP CLIPPING
        #
        # Prevent Adam from destroying the warm start.
        # No extra circuit evaluations.
        # ====================================================
        step_norm = float(np.linalg.norm(step))

        if step_norm > max_step_norm:
            step *= max_step_norm / step_norm
            step_norm = max_step_norm

        theta = theta - step

        # ====================================================
        # EXACT COHERENT DIAGNOSTICS AT THE PARAMETERS FOUND
        # BY THE NOISY OPTIMIZER
        #
        # IMPORTANT:
        # This does NOT alter SPSA/Adam and does NOT add noisy
        # circuit evaluations or extra shots.
        #
        # psi is the exact statevector corresponding to the
        # current theta found under noisy optimization.
        # ====================================================
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
        psi = ev_exact.statevector(theta)

        F = nisq.fidelity(psi, x_pre)
        res = nisq.norm_residual(psi, A, b)

        # Two distinct exact diagnostics are required.  The noisy estimator
        # sees the Pauli-truncated operators in `obs`; `mats` are the full
        # dense operators.  Conflating them can manufacture an apparent
        # objective improvement when pauli-atol is nonzero.
        c_full_exact = float(combine_exact(ev_exact.exact_evs(theta)))
        c_trunc_exact = float(
            combine_exact(ev_exact.exact_evs_observables(theta)))

        elapsed = time.time() - t0

        row = {
            "iter": int(it),
            "F": float(F),
            "residual": float(res),

            # Legacy field retained for old parsers; it is explicitly the
            # full dense objective, not necessarily the measured objective.
            "C_exact": float(c_full_exact),
            "C_full_exact": float(c_full_exact),
            "C_trunc_exact": float(c_trunc_exact),
            "C_spsa_center": float(C_center),
            "C_spsa_sem": float(C_center_sem),
            "C_acceptance": float(C_accept),
            "acceptance_estimator": acceptance_estimator,

            "grad_norm": float(gnorm),
            "spsa_avg": int(avg_now),
            "step_norm": float(step_norm),
            "theta": [float(v) for v in theta],

            "time_s": float(elapsed),
        }

        # ----------------------------------------------------
        # Physical diagnostics only at requested checkpoints.
        #
        # metrics_state() already performs:
        #
        #   preconditioned-space scaling
        #   physical reconstruction
        #   FEM comparison
        #   energy-norm evaluation
        #   QoI evaluation
        #
        # We intentionally reuse nisq.metrics_state instead
        # of duplicating those formulas here.
        # ----------------------------------------------------
        do_diag = (
            it == 1
            or it == maxiter
            or it % max(int(cli.diag_every), 1) == 0
        )

        if do_diag:

            mdiag = nisq.metrics_state(
                psi,
                sysd,
                tag=f"{cli.family}_coherent_noisy_theta"
            )

            dof_here = int(
                sysd.get(
                    "n_orig",
                    len(sysd["b"])
                )
            )

            diag_row = {
                "family": cli.family,
                "seed": int(cli.seed),
                "dof": dof_here,
                "iter": int(it),

                "F": float(mdiag["F"]),
                "F_coherent_theta": float(mdiag["F"]),
                "infidelity":
                    float(1.0 - mdiag["F"]),

                "res_pre":
                    float(mdiag["res_pre"]),

                "res_pre_coherent_theta":
                    float(mdiag["res_pre"]),

                "l2_pre":
                    float(mdiag["l2_pre"]),

                "l2_fis":
                    float(mdiag["l2_fis"]),

                "l2_fis_coherent_theta":
                    float(mdiag["l2_fis"]),

                "linf_fis":
                    float(mdiag["linf_fis"]),

                "eK":
                    float(mdiag["eK"]),

                "eK_rel":
                    float(mdiag["eK_rel"]),

                "cos_fis":
                    float(mdiag["cos_fis"]),

                "qoi_name":
                    mdiag.get("qoi_name", ""),

                "qoi_q":
                    mdiag.get(
                        "qoi_q",
                        float("nan")
                    ),

                "qoi_fem":
                    mdiag.get(
                        "qoi_fem",
                        float("nan")
                    ),

                "qoi_ana":
                    mdiag.get(
                        "qoi_ana",
                        float("nan")
                    ),

                "qoi_ana_nodal":
                    mdiag.get(
                        "qoi_ana_nodal",
                        float("nan")
                    ),

                "qoi_rel_vs_fem":
                    mdiag.get(
                        "qoi_rel_vs_fem",
                        float("nan")
                    ),

                "qoi_rel_vs_ana":
                    mdiag.get(
                        "qoi_rel_vs_ana",
                        float("nan")
                    ),

                "l2_vs_analitica":
                    mdiag.get(
                        "l2_vs_analitica",
                        float("nan")
                    ),

                "l2_fem_vs_analitica":
                    mdiag.get(
                        "l2_fem_vs_analitica",
                        float("nan")
                    ),

                "C_exact": float(c_full_exact),
                "C_full_exact": float(c_full_exact),
                "C_trunc_exact": float(c_trunc_exact),
                "C_spsa_center":
                    float(C_center),
                "C_spsa_sem":
                    float(C_center_sem),

                "grad_norm":
                    float(gnorm),

                "spsa_avg":
                    int(avg_now),

                "step_norm":
                    float(step_norm),

                "time_s":
                    float(elapsed),
            }

            # Also store diagnostics in JSON history.
            row.update({
                "diagnostic_state": "exact_coherent_statevector_at_theta_found_by_noisy_optimizer",

                "F_coherent_theta":
                    diag_row["F_coherent_theta"],
                "infidelity":
                    diag_row["infidelity"],

                "res_pre":
                    diag_row["res_pre"],

                "res_pre_coherent_theta":
                    diag_row["res_pre_coherent_theta"],

                "l2_pre":
                    diag_row["l2_pre"],

                "l2_fis":
                    diag_row["l2_fis"],

                "l2_fis_coherent_theta":
                    diag_row["l2_fis_coherent_theta"],

                "linf_fis":
                    diag_row["linf_fis"],

                "eK":
                    diag_row["eK"],

                "eK_rel":
                    diag_row["eK_rel"],

                "cos_fis":
                    diag_row["cos_fis"],

                "qoi_name":
                    diag_row["qoi_name"],

                "qoi_q":
                    diag_row["qoi_q"],

                "qoi_fem":
                    diag_row["qoi_fem"],

                "qoi_ana":
                    diag_row["qoi_ana"],

                "qoi_ana_nodal":
                    diag_row[
                        "qoi_ana_nodal"
                    ],

                "qoi_rel_vs_fem":
                    diag_row[
                        "qoi_rel_vs_fem"
                    ],

                "qoi_rel_vs_ana":
                    diag_row[
                        "qoi_rel_vs_ana"
                    ],

                "l2_vs_analitica":
                    diag_row[
                        "l2_vs_analitica"
                    ],

                "l2_fem_vs_analitica":
                    diag_row[
                        "l2_fem_vs_analitica"
                    ],
            })

            append_diag_csv(
                cli.metrics_csv,
                diag_row
            )


            # =================================================
            # COMPLETE STATE RECONSTRUCTION FOR OFFLINE ANALYSIS
            #
            # These use exactly the same definitions as
            # nisq.metrics_state().
            # =================================================

            p_state = np.real(
                np.asarray(psi)
            )

            p_state = p_state / max(
                np.linalg.norm(p_state),
                1.0e-300
            )

            # Preconditioned-space VQLS solution including
            # optimal least-squares amplitude.
            alpha_pre = nisq.alpha_ls(
                p_state,
                sysd["A"],
                sysd["b"]
            )

            u_q_pre = (
                alpha_pre * p_state
            )

            # Physical FEM solution. to_physical() performs
            # amplitude reconstruction and inverse mapping
            # from preconditioned coordinates.
            u_q_fis = np.asarray(
                nisq.to_physical(
                    p_state,
                    sysd
                ),
                dtype=float
            )

            archive["iter"].append(
                int(it)
            )

            archive["theta"].append(
                np.asarray(
                    theta,
                    dtype=float
                ).copy()
            )

            archive["psi"].append(
                np.asarray(
                    psi
                ).copy()
            )

            archive["u_q_pre"].append(
                np.asarray(
                    u_q_pre,
                    dtype=float
                ).copy()
            )

            archive["u_q_fis"].append(
                np.asarray(
                    u_q_fis,
                    dtype=float
                ).copy()
            )

            archive["F"].append(
                float(mdiag["F"])
            )

            archive["infidelity"].append(
                float(
                    1.0 - mdiag["F"]
                )
            )

            archive["res_pre"].append(
                float(mdiag["res_pre"])
            )

            archive["l2_pre"].append(
                float(mdiag["l2_pre"])
            )

            archive["l2_fis"].append(
                float(mdiag["l2_fis"])
            )

            archive["linf_fis"].append(
                float(mdiag["linf_fis"])
            )

            archive["eK"].append(
                float(mdiag["eK"])
            )

            archive["eK_rel"].append(
                float(mdiag["eK_rel"])
            )

            archive["cos_fis"].append(
                float(mdiag["cos_fis"])
            )

            archive["qoi_q"].append(
                float(
                    mdiag.get(
                        "qoi_q",
                        np.nan
                    )
                )
            )

            archive["qoi_fem"].append(
                float(
                    mdiag.get(
                        "qoi_fem",
                        np.nan
                    )
                )
            )

            archive[
                "qoi_rel_vs_fem"
            ].append(
                float(
                    mdiag.get(
                        "qoi_rel_vs_fem",
                        np.nan
                    )
                )
            )

            archive["C_exact"].append(
                float(c_full_exact)
            )

            archive["C_full_exact"].append(
                float(c_full_exact)
            )

            archive["C_trunc_exact"].append(
                float(c_trunc_exact)
            )

            archive[
                "C_spsa_center"
            ].append(
                float(C_center)
            )

            archive[
                "C_spsa_sem"
            ].append(
                float(C_center_sem)
            )

            archive["grad_norm"].append(
                float(gnorm)
            )

            archive["step_norm"].append(
                float(step_norm)
            )

            archive["spsa_avg"].append(
                int(avg_now)
            )

            archive["time_s"].append(
                float(elapsed)
            )

        history.append(row)

        # ====================================================
        # COMPACT LOG
        # ====================================================
        if do_diag:
            print(
                f"{prefix}    it {it:3d}: "
                f"F={mdiag['F']:.5f}  "
                f"inf={1.0-mdiag['F']:.5f}  "
                f"res={mdiag['res_pre']:.5f}  "
                f"L2={mdiag['l2_fis']:.5f}  "
                f"eK={mdiag['eK_rel']:.5f}  "
                f"Linf={mdiag['linf_fis']:.5f}  "
                f"QoI={mdiag['qoi_rel_vs_fem']:.5f}  "
                f"Cfull={c_full_exact:.3e}  "
                f"Ctrunc={c_trunc_exact:.3e}  "
                f"|g|={gnorm:.2e}  "
                f"avg={avg_now}  "
                f"|dth|={step_norm:.3e}  "
                f"t={elapsed:.1f}s"
            )

    # A rollback can occur on the final loop iteration, in which case the
    # last history row describes a discarded point.  Compute final diagnostics
    # from the returned theta itself; never use history[-1] as its proxy.
    # Return the best point seen by the selected measured objective, rather
    # than blindly returning the final Adam step.  No exact FEM metric enters
    # this choice.
    if cli.mode != "ideal" and np.isfinite(best_measured_cost):
        theta = best_measured_theta.copy()
        print(f"{prefix}  returning best measured theta: "
              f"Csurrogate={best_measured_cost:.6e}")
    theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
    psi_final = ev_exact.statevector(theta)
    final_mdiag = nisq.metrics_state(
        psi_final, sysd,
        tag="exact_coherent_statevector_at_returned_theta"
    )
    final_c_full = float(combine_exact(ev_exact.exact_evs(theta)))
    final_c_trunc = float(
        combine_exact(ev_exact.exact_evs_observables(theta)))

    # ========================================================
    # STATIC REFERENCE SOLUTIONS
    # ========================================================

    p_final = np.real(
        np.asarray(psi_final)
    )

    p_final = p_final / max(
        np.linalg.norm(p_final),
        1.0e-300
    )

    u_fem = np.linalg.solve(
        sysd["K_pad"],
        sysd["f_pad"]
    )[:sysd["n_orig"]]

    try:
        u_analytic, qoi_analytic, qoi_name = (
            nisq.analytic_solution(sysd)
        )
    except Exception:
        u_analytic = None
        qoi_analytic = np.nan
        qoi_name = ""

    archive["theta_final"] = np.asarray(
        theta,
        dtype=float
    ).copy()

    archive["psi_final"] = np.asarray(
        psi_final
    ).copy()

    archive["u_q_fis_final"] = np.asarray(
        nisq.to_physical(
            p_final,
            sysd
        ),
        dtype=float
    )

    archive["u_q_pre_final"] = (
        nisq.alpha_ls(
            p_final,
            sysd["A"],
            sysd["b"]
        ) * p_final
    )

    archive["u_fem"] = np.asarray(
        u_fem,
        dtype=float
    )

    archive["x_pre_exact"] = np.asarray(
        np.linalg.solve(
            sysd["A"],
            sysd["b"]
        ),
        dtype=float
    )

    archive["x_pre_target"] = np.asarray(
        sysd["x_pre"],
        dtype=float
    )

    if u_analytic is not None:
        archive["u_analytic"] = np.asarray(
            u_analytic,
            dtype=float
        )
    else:
        archive["u_analytic"] = np.array(
            [],
            dtype=float
        )

    archive["qoi_analytic"] = float(
        qoi_analytic
    )

    archive["qoi_name"] = str(
        qoi_name
    )

    archive["kappa_pre"] = float(
        sysd["kappa_pre"]
    )

    archive["nq"] = int(nq)

    archive["n_orig"] = int(
        sysd["n_orig"]
    )

    archive["final_F_coherent_theta"] = float(final_mdiag["F"])
    archive["final_res_pre_coherent_theta"] = float(final_mdiag["res_pre"])
    archive["final_l2_fis_coherent_theta"] = float(final_mdiag["l2_fis"])
    archive["final_eK_rel_coherent_theta"] = float(final_mdiag["eK_rel"])
    archive["final_C_full_exact"] = final_c_full
    archive["final_C_trunc_exact"] = final_c_trunc

    return (
        theta,
        history,
        F_init,
        psi_final,
        archive,
        final_mdiag,
    )



# =====================================================================
# PRESUPUESTO DE DISPAROS                       [PATCH fakeathens+trex_zne]
# =====================================================================
def _preflight_systems(cli):
    """Resolve the actual FEM files and count measured Pauli bases."""
    rows = []
    for dof in dof_ladder(cli.dof_target):
        sysd = nisq.prepare(
            cli.family, dof, precond="amg", datadir=cli.datadir)
        src = (sysd.get("meta", {}) or {}).get("src", "?")
        if cli.require_cluster_files and src == "builtin_fem":
            raise RuntimeError(
                f"[DOF={dof}] cluster K/f files were not found in "
                f"{os.path.abspath(cli.datadir)!r}; refusing builtin_fem.")
        obs, combine, mats, counts = nisq.make_cost_observables(
            sysd["A"], sysd["b"], sysd["nq"], atol=cli.pauli_atol,
            cost_type=cli.cost_type, atol_mode="rel")
        ev = nisq.Evaluator(
            sysd["nq"], cli.depth, obs, combine, mats,
            make_evaluator_args(cli), seed=cli.seed + dof)
        rows.append({
            "dof": dof,
            "nq": int(sysd["nq"]),
            "kappa_pre": float(sysd["kappa_pre"]),
            "src": src,
            "n_bases": int(ev.n_bases),
            "pauli_terms": int(sum(kept for _full, kept in counts)),
        })
    return rows


def _print_shot_budget(cli, preflight_rows):
    """Estimacion a priori del costo cuantico del run.

    Contabilidad (por nivel del multigrid):
      1 evaluacion de evs_fn = n_bases circuitos x shots
      meas_twirl multiplica por meas_twirl_rand randomizaciones
        (reparte los mismos `shots`, asi que NO multiplica los disparos,
         pero SI multiplica el numero de trabajos enviados)
      ZNE multiplica por len(zne_scales)
      1 iteracion SPSA = 2 * spsa_avg evaluaciones de evs_fn
      TREX cobra aparte n_cal_trex circuitos x shots, una sola vez
    `n_bases` is measured from the actual observable and transpiled evaluator
    during the zero-shot preflight; no guessed typical count is used.
    """
    zne_mult = len(cli.zne_scales) if cli.mitigation in ("zne", "trex_zne") else 1

    dofs = dof_ladder(cli.dof_target)
    total = 0
    print("-" * 72)
    print(f"{'nivel':>8}  {'bases':>7}  {'iters':>6}  {'disparos':>14}")
    for i, dof in enumerate(dofs):
        n_bases = next(r["n_bases"] for r in preflight_rows
                       if r["dof"] == dof)
        per_eval = n_bases * cli.shots * zne_mult
        per_iter = 2 * cli.spsa_avg * per_eval
        _lr, iters = level_schedule(cli, dof, i)
        lvl = iters * per_iter
        if cli.mitigation in ("trex", "trex_zne"):
            lvl += cli.n_cal_trex * cli.shots
        total += lvl
        print(f"{dof:>8}  {n_bases:>7}  {iters:>6}  {lvl:>14,}")
    print(f"{'TOTAL':>8}  {'':>7}  {'':>6}  {total:>14,}")
    print(f"  ({2*cli.spsa_avg} evaluaciones de costo por iteracion SPSA"
          + (f" x {zne_mult} factores de folding" if zne_mult > 1 else "") + ")")
    print("-" * 72)
    if total > 5.0e8:
        print("  [AVISO] > 5e8 disparos. En simulador es tiempo de CPU; en")
        print("  hardware real es tiempo de cola inviable. Baje --spsa-avg o")
        print("  --iters-refine, o corra el brazo trex_zne con menos iteraciones.")


def main():
    cli = _sync_mode_mit(parse_args())
    if cli.mitigation in ("zne", "trex_zne") and not cli.allow_inloop_zne:
        raise ValueError(
            "In-loop ZNE is disabled in the definitive campaign. Train with "
            "TREX, then evaluate none/TREX/TREX+ZNE post hoc. Use "
            "--allow-inloop-zne only for an explicitly separate ablation.")
    if cli.pauli_atol != 0.0:
        print("[WARNING] pauli_atol is nonzero: the trained operator is "
              "intentionally truncated.")
    _level_lrs = (list(cli.lr_levels) if cli.lr_levels is not None
                  else [cli.lr] * 3)
    _level_iters = (list(cli.iters_levels) if cli.iters_levels is not None
                    else [cli.iters_coarse, cli.iters_refine, cli.iters_refine])
    if min(cli.shots, cli.spsa_avg, *_level_iters,
           cli.lbfgs_restarts, cli.lbfgs_maxiter) < 1:
        raise ValueError("shots, iterations, SPSA averages and L-BFGS limits "
                         "must be positive")
    if any(not (0.0 < float(lr) <= 0.02) for lr in _level_lrs):
        raise ValueError("Each definitive noisy Adam learning rate must be in (0, 0.02]")
    if not (0.0 < cli.max_step <= 0.06):
        raise ValueError("max_step must be in (0, 0.06]")
    print("=" * 72)
    print(f"MULTIGRID v3 | family={cli.family} | dof={cli.dof_target} | mitigation={cli.mitigation}")
    suppression_label = ", ".join(k for k, v in cli.suppression.items() if v) or "none"
    print(f"SUPPRESSION: {suppression_label}")
    print(f"NOISE MODEL: {cli.noise_model}")
    print(f"ESTIMATOR  : {cli.mitigation}"
          + (f"  (folding {cli.zne_scales}, fit={cli.zne_fit})"
             if cli.mitigation in ("zne", "trex_zne") else ""))
    print("SCHEDULE   : " + ", ".join(
        f"DOF={dof}: lr={level_schedule(cli, dof, i)[0]:.4g}, "
        f"iters={level_schedule(cli, dof, i)[1]}"
        for i, dof in enumerate(dof_ladder(cli.dof_target))))
    print("DIAGNOSTICS: exact coherent statevector at theta found by the selected noisy estimator")
    print("=" * 72)
    preflight_rows = _preflight_systems(cli)
    for row in preflight_rows:
        print(f"PREFLIGHT DOF={row['dof']} nq={row['nq']} "
              f"kappa_pre={row['kappa_pre']:.6g} "
              f"bases={row['n_bases']} Pauli={row['pauli_terms']} "
              f"K/f<-{row['src']}")
    _print_shot_budget(cli, preflight_rows)
    if cli.dry_run:
        print("--dry-run: presupuesto impreso, no se optimiza.")
        return

    dofs = dof_ladder(cli.dof_target)
    theta = None
    all_history = {}
    all_final = {}
    all_archives = {}
    t0 = time.time()
    rng_j = np.random.default_rng(cli.theta_seed + 42)

    for i, dof in enumerate(dofs):
        prefix = f"[DOF={dof}]"
        level_lr, level_iters = level_schedule(cli, dof, i)
        if level_iters == 0:
            print(f"{prefix} SKIP\n")
            continue

        sysd = nisq.prepare(cli.family, dof, precond="amg", datadir=cli.datadir)
        nq = sysd["nq"]
        kappa_pre = float(sysd["kappa_pre"])
        src = (sysd.get("meta", {}) or {}).get("src", "?")
        # nisq.load_system cae al FEM interno SIN avisar si no encuentra los
        # .npz.  Publicar eso como "resultado del cluster" seria un error de
        # procedencia, no un detalle de configuracion.
        if cli.require_cluster_files and src == "builtin_fem":
            raise RuntimeError(
                f"{prefix} nisq.load_system no encontro K/f en "
                f"{os.path.abspath(cli.datadir)!r} y cayo al FEM interno. "
                f"Nombres que busca para dof={dof}: "
                f"{cli.family}_{dof}DOF_K.npz / _f.npz, "
                f"{cli.family}_{dof}_K.npz / _f.npz, "
                f"viga_{dof}DOF_K.npz / _f.npz, "
                f"{cli.family}_dof{dof}_K.npz / _f.npz.")
        print(f"{prefix} nq={nq}  kappa_pre={kappa_pre:.3e}  K/f<-{src}")
        if src == "builtin_fem":
            print(f"{prefix} [AVISO] FEM interno, NO son tus matrices del cluster.")

        if i > 0:
            theta_seed = prolong(theta, int(np.log2(dofs[i-1])), cli.depth, cli.jitter, rng_j)
            print(f"{prefix} prolongado: {len(theta)} -> {len(theta_seed)} params")
        else:
            theta_seed = None

        theta, history, F_init, psi_final, level_archive, final_mdiag = optimize_level(
            sysd,
            nq,
            cli.depth,
            theta_seed,
            cli,
            prefix,
            level_lr=level_lr,
            level_iters=level_iters,
        )

        all_history[str(dof)] = history
        all_archives[str(dof)] = level_archive
        _x_classical = np.linalg.solve(sysd["A"], sysd["b"])
        _x_classical_n = _x_classical / (np.linalg.norm(_x_classical) + 1e-30)
        _psi_n = psi_final / (np.linalg.norm(psi_final) + 1e-30)
        _F_vs_classical = float(abs(np.dot(_psi_n, _x_classical_n))**2)
        # Solución analítica (closed-form) donde exista
        try:
            _u_ana, _qoi_ana, _qoi_name = nisq.analytic_solution(sysd)
        except Exception:
            _u_ana, _qoi_ana, _qoi_name = None, float("nan"), ""
        # Physical solution: use the sole reconstruction map used by
        # metrics_state().  The old manual path omitted normalization and
        # equilibration factors and therefore was not a physical FEM field.
        try:
            _x_phys_q = np.asarray(nisq.to_physical(psi_final, sysd), float)
        except Exception:
            _x_phys_q = None
        _u_fem = np.linalg.solve(sysd["K_pad"], sysd["f_pad"])[:sysd["n_orig"]]
        _L2_fem = (float(np.linalg.norm(_x_phys_q - _u_fem) /
                         max(np.linalg.norm(_u_fem), 1e-300))
                   if _x_phys_q is not None else float("nan"))
        # Error L2 vs analítica (si aplica)
        if _u_ana is not None and _x_phys_q is not None and len(_u_ana) == len(_x_phys_q):
            _L2_err = float(np.linalg.norm(_x_phys_q - _u_ana) /
                            (np.linalg.norm(_u_ana) + 1e-30))
        else:
            _L2_err = float("nan")
        all_final[str(dof)] = {
            "F": float(final_mdiag["F"]), "F_coherent_theta": float(final_mdiag["F"]),
            "residual": float(final_mdiag["res_pre"]),
            "residual_pre_coherent_theta": float(final_mdiag["res_pre"]),
            "C_exact": level_archive["final_C_full_exact"],
            "C_full_exact": level_archive["final_C_full_exact"],
            "C_trunc_exact": level_archive["final_C_trunc_exact"],
            "kappa_pre": kappa_pre,
            "P": len(theta), "F_init": F_init,
            "psi_final": psi_final.tolist(),
            "theta_final": theta.tolist(),
            "x_pre": sysd["x_pre"].tolist(),
            "x_classical": _x_classical.tolist(),
            "F_vs_classical": _F_vs_classical,
            "F_vs_classical_definition": "coherent_statevector_fidelity_in_normalized_preconditioned_coordinates",
            "u_analytic": (_u_ana.tolist() if _u_ana is not None else None),
            "qoi_analytic": _qoi_ana,
            "qoi_name": _qoi_name,
            "x_phys_quantum": (_x_phys_q.tolist() if _x_phys_q is not None else None),
            "L2_error_vs_analytic": _L2_err,
            "L2_error_vs_fem": _L2_fem,
            "physical_solution_definition": "to_physical(exact_coherent_statevector_at_theta_found_by_noisy_optimizer)",
            "diagnostic_state": "exact_coherent_statevector_at_theta_found_by_noisy_optimizer",
            "mode": cli.mode, "seed": int(cli.seed),
            "spsa_avg": int(cli.spsa_avg),
            "spsa_c0": float(cli.spsa_c0),
            "spsa_gamma": float(cli.spsa_gamma),
            "adam_lr": float(level_lr),
            "optimizer_iterations": int(level_iters),
        }
        print()

    # ========================================================
    # SAVE COMPLETE NUMERICAL ARCHIVE
    # ========================================================

    if cli.archive_npz:

        archive_path = os.path.expanduser(
            cli.archive_npz
        )

        archive_dir = os.path.dirname(
            os.path.abspath(
                archive_path
            )
        )

        os.makedirs(
            archive_dir,
            exist_ok=True
        )

        npz_data = {}

        for dof_key, ar in all_archives.items():

            prefix_npz = f"dof{dof_key}_"

            # Time-history arrays.
            for key in [
                "iter",
                "theta",
                "psi",
                "u_q_pre",
                "u_q_fis",
                "F",
                "infidelity",
                "res_pre",
                "l2_pre",
                "l2_fis",
                "linf_fis",
                "eK",
                "eK_rel",
                "cos_fis",
                "qoi_q",
                "qoi_fem",
                "qoi_rel_vs_fem",
                "C_exact",
                "C_full_exact",
                "C_trunc_exact",
                "C_spsa_center",
                "C_spsa_sem",
                "grad_norm",
                "step_norm",
                "spsa_avg",
                "time_s",
            ]:
                npz_data[
                    prefix_npz + key
                ] = np.asarray(
                    ar[key]
                )

            # Final/reference arrays.
            for key in [
                "theta_final",
                "psi_final",
                "u_q_pre_final",
                "u_q_fis_final",
                "u_fem",
                "u_analytic",
                "x_pre_exact",
                "x_pre_target",
            ]:
                npz_data[
                    prefix_npz + key
                ] = np.asarray(
                    ar[key]
                )

            for key in [
                "qoi_analytic",
                "kappa_pre",
                "nq",
                "n_orig",
                "final_F_coherent_theta",
                "final_res_pre_coherent_theta",
                "final_l2_fis_coherent_theta",
                "final_eK_rel_coherent_theta",
                "final_C_full_exact",
                "final_C_trunc_exact",
            ]:
                npz_data[
                    prefix_npz + key
                ] = np.asarray(
                    ar[key]
                )

            npz_data[
                prefix_npz +
                "qoi_name"
            ] = np.asarray(
                ar["qoi_name"]
            )

            # ------------------------------------------------
            # OFFLINE BEST CHECKPOINT INDICES
            #
            # These are NEVER used during optimization.
            # ------------------------------------------------

            if len(ar["iter"]) > 0:

                best_defs = {
                    "best_F":
                        int(
                            np.argmax(
                                ar["F"]
                            )
                        ),

                    "best_res":
                        int(
                            np.argmin(
                                ar["res_pre"]
                            )
                        ),

                    "best_L2":
                        int(
                            np.argmin(
                                ar["l2_fis"]
                            )
                        ),

                    "best_eK":
                        int(
                            np.argmin(
                                ar["eK_rel"]
                            )
                        ),

                    "best_QoI":
                        int(
                            np.nanargmin(
                                ar[
                                    "qoi_rel_vs_fem"
                                ]
                            )
                        ),

                    "best_C_exact":
                        int(
                            np.argmin(
                                ar["C_full_exact"]
                            )
                        ),

                    "best_C_trunc_exact":
                        int(
                            np.argmin(
                                ar["C_trunc_exact"]
                            )
                        ),
                }

                for name, idx in best_defs.items():

                    npz_data[
                        prefix_npz +
                        name +
                        "_archive_index"
                    ] = np.asarray(idx)

                    npz_data[
                        prefix_npz +
                        name +
                        "_iter"
                    ] = np.asarray(
                        ar["iter"][idx]
                    )

                    npz_data[
                        prefix_npz +
                        name +
                        "_theta"
                    ] = np.asarray(
                        ar["theta"][idx]
                    )

                    npz_data[
                        prefix_npz +
                        name +
                        "_psi"
                    ] = np.asarray(
                        ar["psi"][idx]
                    )

                    npz_data[
                        prefix_npz +
                        name +
                        "_u_q_fis"
                    ] = np.asarray(
                        ar[
                            "u_q_fis"
                        ][idx]
                    )

                    npz_data[
                        prefix_npz +
                        name +
                        "_u_q_pre"
                    ] = np.asarray(
                        ar[
                            "u_q_pre"
                        ][idx]
                    )

        np.savez_compressed(
            archive_path,
            **npz_data
        )

        print(
            f"OK Archive NPZ: "
            f"{archive_path}"
        )

    print("=" * 72)
    print(f"{'DOF':>6}  {'kappa_pre':>10}  {'F_init':>10}  {'F_final':>10}  {'res':>10}")
    for dof in dofs:
        if str(dof) not in all_final:
            continue
        r = all_final[str(dof)]
        f_init_str = f"{r['F_init']:.5f}" if r['F_init'] is not None else " random"
        print(f"{dof:>6}  {r['kappa_pre']:>10.3e}  {f_init_str:>10}  {r['F']:>10.5f}  {r['residual']:>10.5f}")
    print(f"Tiempo: {time.time()-t0:.1f}s\n")

    if cli.json_out:
        with open(cli.json_out, "w") as f:
            json.dump({"config": vars(cli), "dofs": dofs, "history": all_history,
                       "final": all_final, "total_time_s": time.time()-t0}, f, indent=2)
        print(f"OK Guardado: {cli.json_out}")

    if cli.theta_out:
        theta_path = os.path.expanduser(cli.theta_out)
        os.makedirs(os.path.dirname(os.path.abspath(theta_path)), exist_ok=True)
        np.save(theta_path, np.asarray(theta, dtype=float))
        print(f"OK Theta final: {theta_path}")

if __name__ == "__main__":
    main()
