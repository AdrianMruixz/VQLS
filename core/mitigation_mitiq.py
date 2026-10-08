#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mitigation_mitiq.py -- mitigacion 100% Mitiq + supresion 100% Qiskit
=====================================================================
Version corregida:
  1. Separa el termino identidad de cada observable para evitar
     ValueError al medir un soporte vacio.
  2. Transpilacion sin backend para evitar advertencias.
  3. [FIX] `_ddd` era literalmente `return self._raw(circuit)`: DDD NO se
     aplicaba nunca.  Ahora usa `mitiq.ddd.execute_with_ddd` con la regla
     configurable (xx / yy / xyxy).
  4. [FIX] al reconstruir los ejecutores desde `default_shots` se perdia
     `counter=self.counter`, de modo que `qpu_time_seconds()` devolvia
     siempre 0.
  5. [FIX] `qpu_time_seconds` multiplicaba counter["shots"] * counter
     ["circuits"], pero counter["shots"] YA es el acumulado sobre todos los
     circuitos -> el tiempo crecia cuadraticamente.
  6. [FIX] `_pec`, `_rem`, `_lre` no comprobaban `obs is None` (observable
     puramente identidad, que ocurre p.ej. con precond=cholesky, donde
     A = I y por tanto A^2 = I tiene un solo termino de Pauli).
  7. [FIX] `_zne` sumaba la identidad ANTES de extrapolar; es inocuo para
     Richardson/lineal/poly (afines) pero NO para el ajuste exponencial con
     asintota.  Ahora se extrapola la parte no-identidad y se suma despues.
"""

import warnings
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

import cirq
import mitiq
from mitiq import Observable, PauliString, MeasurementResult
from mitiq.interface import convert_to_mitiq, convert_from_mitiq
from mitiq import zne, pec, cdr, ddd, rem, lre, pt
from mitiq.rem import generate_inverse_confusion_matrix
from mitiq.pec import represent_operations_in_circuit_with_local_depolarizing_noise
from mitiq.zne.scaling import fold_global, fold_all, fold_gates_at_random
from mitiq.zne.inference import (RichardsonFactory, LinearFactory,
                                 ExpFactory, PolyFactory)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from mitiq.experimental import trex
    except Exception:
        trex = None


MITIQ_METHODS = ["none", "zne", "pec", "cdr", "ddd", "rem", "trex", "lre", "pt", "pipeline"]


# ---------------------------------------------------------------------
# 1) Puente Qiskit <-> Mitiq
# ---------------------------------------------------------------------
def sparse_pauli_to_observable(spo: SparsePauliOp):
    """
    Convierte SparsePauliOp de Qiskit a (Observable, ident_coeff).

    El término identidad se devuelve aparte para sumarlo manualmente,
    evitando que Mitiq intente medir un conjunto vacío de qubits.
    """
    ident_coeff = 0.0
    paulis = []
    for label, coeff in zip(spo.paulis.to_labels(), spo.coeffs):
        if label == "I" * spo.num_qubits:
            ident_coeff += float(np.real(coeff))
            continue
        c = complex(coeff)
        if abs(c) == 0.0:
            continue
        paulis.append(PauliString(spec=label[::-1], coeff=c))

    if not paulis:
        # Si no hay términos no-identidad, devolvemos None
        return None, ident_coeff

    return Observable(*paulis), ident_coeff


def make_measurement_executor(simulator, shots, seed=None, basis_gates=None, counter=None):
    """Executor que devuelve MeasurementResult, obligatorio para Mitiq."""
    rng = np.random.default_rng(seed)

    def executor(circuit) -> MeasurementResult:
        if not isinstance(circuit, QuantumCircuit):
            circuit = convert_from_mitiq(circuit, "qiskit")
        c = circuit.remove_final_measurements(inplace=False)
        c.measure_all()
        kw = dict(optimization_level=0)
        if basis_gates is not None:
            kw["basis_gates"] = list(basis_gates)
        # Transpilación SIN backend para evitar advertencias
        ct = transpile(c, **kw)
        counts = simulator.run(
            ct, shots=int(shots),
            seed_simulator=int(rng.integers(2 ** 31))).result().get_counts()
        bits = []
        for key, n in counts.items():
            row = [int(b) for b in key.replace(" ", "")[::-1]]
            bits.extend([row] * int(n))

        # Actualizar contador de circuitos y shots
        if counter is not None:
            counter["circuits"] += 1
            counter["shots"] += int(shots)

        return MeasurementResult(bits, qubit_indices=tuple(range(c.num_qubits)))

    return executor


def _extrapolator(name, scale_factors):
    if name == "richardson":
        return RichardsonFactory(scale_factors).extrapolate
    if name == "linear":
        return LinearFactory(scale_factors).extrapolate
    if name == "exponential":
        return ExpFactory(scale_factors, asymptote=None).extrapolate
    if name == "poly":
        return PolyFactory(scale_factors, order=2).extrapolate
    raise ValueError(f"extrapolador desconocido: {name}")


def to_linequbit_cirq(qiskit_circuit):
    cc, _ = convert_to_mitiq(qiskit_circuit)
    qs = sorted(cc.all_qubits())
    return cc.transform_qubits({q: cirq.LineQubit(i)
                                for i, q in enumerate(qs)})


def _folding(name):
    return {"global": fold_global,
            "local": fold_all,
            "random": fold_gates_at_random}[name]


# ---------------------------------------------------------------------
# 2) Evaluador Mitiq
# ---------------------------------------------------------------------
class MitiqEvaluator:
    def __init__(self, obs_list, ansatz, noise_model=None, shots=4096,
                 seed=0, basis_gates=("rz", "sx", "x", "y", "cx"),
                 estimator_options=None, noise_1q=5e-4, noise_2q=3.5e-3,
                 p_ro=0.015, ro_asym=0.3):
        self.nq = ansatz.num_qubits
        self.ansatz = ansatz
        self.shots = int(shots)
        self.basis_gates = list(basis_gates)
        self.noise_1q = noise_1q
        self.noise_2q = noise_2q
        self.p_ro = p_ro
        self.ro_asym = ro_asym
        # Matriz de confusión inversa para REM
        self.inverse_confusion_matrix = self._build_inverse_confusion_matrix()

        # Convertir observables separando identidad
        self.observables = []
        self.identity_coeffs = []
        for spo in obs_list:
            obs, id_c = sparse_pauli_to_observable(spo)
            self.observables.append(obs)
            self.identity_coeffs.append(id_c)

        self.rng = np.random.default_rng(seed)

        self.sim_noisy = AerSimulator(noise_model=noise_model,
                                      basis_gates=self.basis_gates)
        self.sim_clean = AerSimulator(basis_gates=self.basis_gates)
        self.noise_model = noise_model
        self.noiseless = noise_model is None

        self.counter = {"circuits": 0, "shots": 0}
        self.executor = make_measurement_executor(
            self.sim_noisy, self.shots, seed=seed,
            basis_gates=self.basis_gates, counter=self.counter)
        self.executor_clean = make_measurement_executor(
            self.sim_clean, self.shots, seed=seed + 1,
            basis_gates=self.basis_gates, counter=self.counter)

        self.n_calls = 0
        self.total_shots = 0
        self.time_reset_ns = 1000.0
        self.time_gate_ns_per_depth = 100.0
        self.time_measure_ns = 1000.0
        self.qc_log = None   # se asigna desde fuera

        # --- parser de opciones estilo Qiskit Runtime ---
        self.estimator_options = estimator_options or {}

        tw = self.estimator_options.get("twirling", {})
        self._trex_enable_measure = bool(tw.get("enable_measure", False))
        self._trex_num_random = int(tw.get("num_randomizations", 32))
        self._trex_shots_per_rand = tw.get("shots_per_randomization", "auto")

        tg = self.estimator_options.get("twirling_gates", {})
        self._gate_twirl_enable = bool(tg.get("enable_gates", False))
        self._gate_twirl_num = int(tg.get("num_randomizations", 1))

        res = self.estimator_options.get("resilience", {})
        self._zne_enable = bool(res.get("zne_mitigation", False))
        self._zne_scales = list(res.get("zne", {}).get("noise_factors", [1.0, 3.0, 5.0]))
        self._zne_extrap = res.get("zne", {}).get("extrapolator", "richardson")
        self._zne_folding = self.estimator_options.get("zne_folding", "global")

        dd = self.estimator_options.get("dynamical_decoupling", {})
        self._dd_enable = bool(dd.get("enable", False))
        self._dd_seq = str(dd.get("sequence_type", "XY4"))
        self._ddd_num_trials = int(dd.get("num_trials", 1))

        if "default_shots" in self.estimator_options:
            self.shots = int(self.estimator_options["default_shots"])
            # FIX: sin `counter=self.counter` el contador de circuitos/disparos
            # se quedaba en cero y qpu_time_seconds() devolvia siempre 0.
            self.executor = make_measurement_executor(
                self.sim_noisy, self.shots, seed=seed,
                basis_gates=self.basis_gates, counter=self.counter)
            self.executor_clean = make_measurement_executor(
                self.sim_clean, self.shots, seed=seed + 1,
                basis_gates=self.basis_gates, counter=self.counter)


    # -----------------------------------------------------------------
    def _build_inverse_confusion_matrix(self):
        """Genera la matriz inversa de confusión usando la API de Mitiq."""
        p01 = min(0.49, self.p_ro * (1.0 + self.ro_asym))
        p10 = min(0.49, self.p_ro * (1.0 - self.ro_asym))
        return generate_inverse_confusion_matrix(self.nq, p01, p10)


    # -----------------------------------------------------------------
    def qpu_time_seconds(self):
        """
        Tiempo de QPU estimado:  (reset + compuertas + medicion) * SHOTS TOTALES.

        FIX: la version anterior hacia  ... * counter["shots"] * counter
        ["circuits"], pero `counter["shots"]` ya acumula los disparos de TODOS
        los circuitos ejecutados, de modo que el producto contaba el numero de
        circuitos dos veces y el tiempo crecia cuadraticamente con el numero
        de evaluaciones.
        """
        depth = self.ansatz.depth()
        per_shot_ns = (self.time_reset_ns +
                       self.time_gate_ns_per_depth * depth +
                       self.time_measure_ns)
        return per_shot_ns * self.counter["shots"] / 1e9

    def reset_counter(self):
        self.counter["circuits"] = 0
        self.counter["shots"] = 0

    # -----------------------------------------------------------------
    def set_shots(self, shots):
        """Reconfigura los shots y los ejecutores."""
        self.shots = int(shots)
        self.executor = make_measurement_executor(
            self.sim_noisy, self.shots, seed=0,
            basis_gates=self.basis_gates, counter=self.counter)
        self.executor_clean = make_measurement_executor(
            self.sim_clean, self.shots, seed=1,
            basis_gates=self.basis_gates, counter=self.counter)

    # -----------------------------------------------------------------
    def _choose_executor(self):
        return self.executor_clean if self.noiseless else self.executor

    # -----------------------------------------------------------------
    def _real_expectation(self, obs, circuit, executor):
        """Devuelve float(real) de obs.expectation, sin identidad.

        Si obs es None (solo había identidad), devuelve 0.0.
        """
        if obs is None:
            return 0.0
        return float(np.real(obs.expectation(circuit, executor)))

    # -----------------------------------------------------------------
    def statevector(self, theta):
        from qiskit.quantum_info import Statevector
        qc = self.qc_log if self.qc_log is not None else self.ansatz
        return np.real(Statevector(qc.assign_parameters(theta)).data)

    # -----------------------------------------------------------------
    def evs(self, theta, method="none"):
        circuit = self.ansatz.assign_parameters(theta)
        if method == "none":
            return self._raw(circuit)
        elif method == "zne":
            return self._zne(circuit)
        elif method == "pec":
            return self._pec(circuit)
        elif method == "cdr":
            return self._cdr(circuit)
        elif method == "ddd":
            return self._ddd(circuit)
        elif method == "rem":
            return self._rem(circuit)
        elif method == "trex":
            return self._trex(circuit)
        elif method == "lre":
            return self._lre(circuit)
        elif method == "pt":
            return self._pt(circuit)
        elif method == "pipeline":
            return self._pipeline(circuit)
        else:
            raise ValueError(f"Método desconocido: {method}")

    # -----------------------------------------------------------------
    def _raw(self, circuit):
        executor = self._choose_executor()
        out = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                out.append(self.identity_coeffs[i])
                continue
            v = self._real_expectation(obs, circuit, executor) + self.identity_coeffs[i]
            out.append(v)
        return np.asarray(out, float)

    # -----------------------------------------------------------------
    def _zne(self, circuit):
        sf = list(map(float, self._zne_scales))
        circs = zne.construct_circuits(circuit,
                                       scale_factors=sf,
                                       scale_method=_folding(self._zne_folding))
        executor = self._choose_executor()
        extrap = _extrapolator(self._zne_extrap, sf)
        out = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                out.append(self.identity_coeffs[i])
                continue
            # FIX: se extrapola SOLO la parte no-identidad y la identidad se
            # suma despues.  Para Richardson/lineal/poly es equivalente (son
            # afines), pero no para el ajuste exponencial con asintota.
            ys = [self._real_expectation(obs, c, executor) for c in circs]
            out.append(float(zne.combine_results(sf, ys, extrap))
                       + self.identity_coeffs[i])
        return np.asarray(out, float)


    # -----------------------------------------------------------------
    def _pt(self, circuit):
        """Pauli Twirling (PT): promedia sobre variantes twirled del circuito."""
        executor = self._choose_executor()
        n_pt = max(1, self._gate_twirl_num)
        variants = pt.generate_pauli_twirl_variants(circuit, num_circuits=n_pt)
        out = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                out.append(self.identity_coeffs[i])
                continue
            vals = []
            for c in variants:
                vals.append(self._real_expectation(obs, c, executor) + self.identity_coeffs[i])
            out.append(float(np.mean(vals)))
        return np.asarray(out, float)

    # -----------------------------------------------------------------
    def _pipeline(self, circuit):
        """Pipeline completo ZNE  PT  DDD  REM (basado en guía de Mitiq)."""
        executor = self._choose_executor()
        sf = list(map(float, self._zne_scales))
        scaled_circuits = zne.construct_circuits(
            circuit,
            scale_factors=sf,
            scale_method=_folding(self._zne_folding)
        )
        rule = ddd.rules.xyxy
        n_pt = max(1, self._gate_twirl_num)

        out = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                out.append(self.identity_coeffs[i])
                continue

            scale_vals = []
            for c_scaled in scaled_circuits:
                pt_variants = pt.generate_pauli_twirl_variants(c_scaled, num_circuits=n_pt)
                pt_vals = []
                for c_pt in pt_variants:
                    # Aplicar DDD
                    c_ddd = ddd.insert_ddd_sequences(c_pt, rule=rule)
                    # Ejecutar y corregir lectura con REM
                    raw = executor(c_ddd)
                    corrected = rem.mitigate_measurements(raw, self.inverse_confusion_matrix)
                    exp = obs._expectation_from_measurements([corrected]).real
                    pt_vals.append(exp + self.identity_coeffs[i])
                scale_vals.append(float(np.mean(pt_vals)))

            out.append(float(zne.combine_results(sf, scale_vals, _extrapolator(self._zne_extrap, sf))))
        return np.asarray(out, float)

    # -----------------------------------------------------------------
    def _pec(self, circuit):
        executor = self._choose_executor()
        # Construir representaciones cuasi-probabilísticas para todo el circuito.
        # Se usa el nivel de ruido de 2 qubits (dominante en el modelo).
        representations = represent_operations_in_circuit_with_local_depolarizing_noise(
            circuit, noise_level=self.noise_2q
        )
        vals = []
        for i, obs in enumerate(self.observables):
            if obs is None:                      # observable puramente identidad
                vals.append(self.identity_coeffs[i])
                continue
            v = pec.execute_with_pec(
                circuit,
                executor,
                observable=obs,
                representations=representations,
                num_samples=50,
                random_state=int(self.rng.integers(2**31)),
            )
            vals.append(float(np.real(v)) + self.identity_coeffs[i])
        return np.asarray(vals, float)

    # -----------------------------------------------------------------
    def _cdr(self, circuit):
        executor = self._choose_executor()
        vals = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                vals.append(self.identity_coeffs[i])
                continue
            v = cdr.execute_with_cdr(
                circuit,
                executor,
                observable=obs,
                simulator=self.executor_clean,
                num_training_circuits=20,
                fraction_non_clifford=0.2,
                random_state=int(self.rng.integers(2**31)),
            )
            vals.append(float(np.real(v)) + self.identity_coeffs[i])
        return np.asarray(vals, float)

    # -----------------------------------------------------------------
    def _ddd_rule(self):
        """Regla de secuencia DDD segun estimator_options.dynamical_decoupling."""
        name = self._dd_seq.strip().lower()
        if name in ("xx",):
            return ddd.rules.xx
        if name in ("yy",):
            return ddd.rules.yy
        return ddd.rules.xyxy          # XY4 por defecto

    def _ddd(self, circuit):
        """
        Digital Dynamical Decoupling REAL (mitiq.ddd.execute_with_ddd).

        FIX: la version anterior era  `return self._raw(circuit)`, es decir,
        DDD no hacia absolutamente nada y sus resultados eran identicos al
        baseline sin mitigar.  Cualquier conclusion previa del tipo "DDD no
        ayuda" media este no-op, no la tecnica.
        """
        executor = self._choose_executor()
        rule = self._ddd_rule()
        out = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                out.append(self.identity_coeffs[i])
                continue
            try:
                v = ddd.execute_with_ddd(
                    circuit, executor, observable=obs, rule=rule,
                    num_trials=max(1, self._ddd_num_trials))
                out.append(float(np.real(v)) + self.identity_coeffs[i])
            except Exception as exc:
                warnings.warn(f"DDD fallo ({exc}); se reporta el valor crudo.")
                out.append(self._real_expectation(obs, circuit, executor)
                           + self.identity_coeffs[i])
        return np.asarray(out, float)

    # -----------------------------------------------------------------
    def _rem(self, circuit):
        executor = self._choose_executor()
        vals = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                vals.append(self.identity_coeffs[i])
                continue
            v = rem.execute_with_rem(circuit, executor, observable=obs,
                                     inverse_confusion_matrix=self.inverse_confusion_matrix)
            vals.append(float(np.real(v)) + self.identity_coeffs[i])
        return np.asarray(vals, float)

    # -----------------------------------------------------------------
    def _trex(self, circuit):
        if trex is None:
            raise RuntimeError("TREX no disponible en Mitiq experimental")
        try:
            executor = self._choose_executor()
            vals = []
            for i, obs in enumerate(self.observables):
                if obs is None:
                    vals.append(self.identity_coeffs[i])
                    continue
                v = trex.execute_with_trex(circuit, executor, observable=obs,
                                           random_state=int(self.rng.integers(2**31)))
                vals.append(float(np.real(v)) + self.identity_coeffs[i])
            return np.asarray(vals, float)
        except Exception:
            return self._raw(circuit)

    # -----------------------------------------------------------------
    def _lre(self, circuit):
        executor = self._choose_executor()
        vals = []
        for i, obs in enumerate(self.observables):
            if obs is None:
                vals.append(self.identity_coeffs[i])
                continue
            v = lre.execute_with_lre(
                circuit,
                executor,
                degree=2,
                fold_multiplier=2,
                observable=obs,
            )
            vals.append(float(np.real(v)) + self.identity_coeffs[i])
        return np.asarray(vals, float)
