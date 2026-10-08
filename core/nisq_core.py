#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nisq_core.py -- piezas de nisq.py sin dependencia de Qiskit
=============================================================
Optimizadores clasicos (SPSA, COBYLA, L-BFGS-B), el extrapolador de ZNE, y
la logica de seguridad numerica (epsilon escalado / clip de denominador)
usados por `nisq.py`.  Viven en un modulo separado, importado por `nisq.py`
ANTES de sus imports de Qiskit, para que:

  1) `nisq.spsa_minimize`, `nisq.cobyla_minimize`, `nisq.lbfgs_minimize`,
     `nisq._extrapolate` sigan funcionando exactamente igual (compatibilidad
     hacia atras: solo `run_mitigation_study.py` hace `import nisq` en todo
     el repo, y sigue viendo estos nombres como atributos del modulo).
  2) Estas funciones sean testeables con `pytest` en un entorno SIN Qiskit
     instalado (requisito explicito de la tarea: tests no dependientes de
     Qiskit para los wrappers de optimizador y para la logica eps/clip).
"""
from __future__ import annotations

import warnings

import numpy as np


# =====================================================================
# 1) SEGURIDAD NUMERICA:  epsilon escalado + combinacion segura de razones
# =====================================================================
def scaled_eps(a_norm, floor=1e-12, rel=1e-12):
    """
    eps = max(floor, ||A||_2 * rel).

    Los costos VQLS (`local`, `global`, `energy`) son razones de valores
    esperados cuyo denominador depende del estado variacional actual
    (<x|A'A|x> o <x|A|x>). Un `eps` FIJO (p.ej. 1e-6) esta mal escalado si
    ||A||_2 no es ~1 (pasa con los precondicionadores `amg`/`cheby`, que no
    preservan la norma espectral tras `prepare()`), asi que se escala con
    la norma real de A.
    """
    a_norm = float(abs(a_norm))
    return max(float(floor), a_norm * float(rel))


def safe_ratio_combine(num, den, a_norm, scale=1.0, warn_rel=1e-8,
                        eps=None, label="cost"):
    """
    Devuelve `scale * num / den_clipped`, SIEMPRE finito.

    - El denominador se recorta a `eps = scaled_eps(a_norm)` preservando su
      signo (evita que ratios con denominador->0 disparen a +-inf).
    - Si `|den|` cae por debajo de `warn_rel * max(a_norm, eps_floor)` ANTES
      de aplicar el recorte, se emite un `RuntimeWarning`: es la senal de
      que el optimizador esta explotando un polo numerico del costo, no una
      caracteristica fisica real del ansatz.
    - `den == 0.0` exactamente se trata como `den >= 0` (recorta a +eps),
      igual que hacia el codigo original.

    Se usa para los tres tipos de costo (`local`, `global`, `energy`) de
    `nisq.make_cost_observables`, sustituyendo los "floor" ad-hoc que tenia
    cada uno (1e-12 fijo en local/global, 1e-6 fijo en energy) por una unica
    ruta auditada y testeada (ver tests/test_numerics.py).
    """
    if eps is None:
        eps = scaled_eps(a_norm)
    den = float(den)
    warn_thresh = float(warn_rel) * max(float(a_norm), eps)
    if abs(den) < warn_thresh:
        warnings.warn(
            f"{label}: |denominador|={abs(den):.3e} por debajo del umbral "
            f"de seguridad {warn_thresh:.3e} (||A||_2={a_norm:.3e}); "
            f"se recorta a eps={eps:.3e} para evitar +-inf.",
            RuntimeWarning, stacklevel=2)
    if abs(den) < eps:
        den = eps if den >= 0.0 else -eps
    return float(scale) * float(num) / den


# =====================================================================
# 2) EXTRAPOLADOR ZNE (usado con y sin Mitiq)
# =====================================================================
def _extrapolate(scales, ys, mode, c_fix):
    """
    Extrapolacion a ruido cero.

    "richardson"           : interpolacion polinomica exacta de grado
                              n_scales-1, evaluada en scale=0.
    "linear"                : ajuste lineal, intercepto en scale=0.
    "exp_fixed"/"fixedpoint": exponencial con asintota FIJA en c_fix
                              (= Tr(O)/2^n, punto fijo del canal
                              depolarizante global). "fixedpoint" es el
                              alias usado por `run_mitigation_study.py`;
                              ambos nombres son equivalentes aqui.
    """
    if mode in ("exp_fixed", "fixedpoint"):
        z = ys - c_fix
        if np.all(z > 1e-14) or np.all(z < -1e-14):
            s = np.sign(z[0])
            m, q = np.polyfit(scales, np.log(np.abs(z)), 1)
            if m < 0.0:            # exponencial DECRECIENTE (antes: m < 0.05,
                                   # que aceptaba ajustes crecientes)
                return float(c_fix + s * np.exp(q))
        return float(np.polyfit(scales, ys, 1)[1])
    if mode == "richardson":
        c = np.polyfit(scales, ys, len(scales) - 1)
        return float(np.polyval(c, 0.0))
    return float(np.polyfit(scales, ys, 1)[1])          # linear


# =====================================================================
# 3) VALIDACION DE MITIQ
# =====================================================================
def check_mitiq_version(mitiq_module, minimum=(1, 0)):
    """
    Valida que `mitiq` este instalado y con version >= `minimum`.

    `mitiq_module` es el modulo ya importado (o None si el import fallo en
    el llamador). No usa `packaging` para no anadir una dependencia nueva:
    parsea manualmente los dos primeros componentes de `__version__`.
    Lanza RuntimeError con instrucciones de instalacion si no se cumple.
    """
    if mitiq_module is None:
        raise RuntimeError(
            "--mitiq fue solicitado pero el paquete 'mitiq' no esta "
            "instalado. Instale con:\n"
            "    pip install 'mitiq>=1.0.0'")
    ver = getattr(mitiq_module, "__version__", "0.0.0")
    try:
        parts = tuple(int(p) for p in ver.split(".")[:2])
    except ValueError:
        parts = (0, 0)
    if parts < tuple(minimum):
        min_str = ".".join(str(m) for m in minimum)
        raise RuntimeError(
            f"Se requiere mitiq>={min_str}.0; version instalada: {ver}. "
            f"Actualice con:\n"
            f"    pip install -U 'mitiq>={min_str}.0'")
    return ver


# =====================================================================
# 4) OPTIMIZADORES  (SPSA, COBYLA, L-BFGS-B)
# =====================================================================
def spsa_minimize(cost, x0, maxiter=500, a=0.30, c=0.15, alpha=0.602,
                  gamma=0.101, seed=0, callback=None, n_resamples=1,
                  calibrate=True, target_step=0.20, smooth_window=5,
                  final_reeval=True):
    """
    SPSA (Spall 1998, IEEE Trans. Aerosp. Electron. Syst. 34, 817). 2
    evaluaciones de costo por iteracion, independiente del numero de
    parametros -> el optimizador estandar para VQA con ruido.

    CALIBRACION DE `a` (Spall, "Implementation of the simultaneous
    perturbation algorithm", 1998, regla practica #4):
        a = target_step * (A+1)^alpha / |g_0|
    de modo que el PRIMER paso valga aproximadamente `target_step` radianes
    con independencia de la escala del costo.

    BUG ANTERIOR 1: `gbar` se calculaba (5 pares de evaluaciones, coste real) y
    NUNCA se usaba; `a` se quedaba en 0.30 para todos los costos.  Como
    `cost_energy` devuelve valores de orden 1e2-1e3 y `cost_local` de orden 1,
    el mismo `a` produce pasos que difieren en 3 ordenes de magnitud.  Medido
    en una cuadratica f = s*||x||^2 con x0 = 1:
        s=1e-3 -> reduccion 0.984 (practicamente nula)
        s=1    -> reduccion 5e-7  (converge)
        s=1e3  -> reduccion 1.000 (estancado)
    Es decir, el estancamiento se leia como "fallo de entrenabilidad" cuando
    era un tamano de paso mal escalado.  Con `calibrate=True` los tres casos
    convergen igual.

    BUG ANTERIOR 2 (CORREGIDO AQUI):  el seguimiento del mejor punto hacia

        x = x - ak*(fp-fm)/(2*ck)*d      # x YA actualizado
        f = 0.5*(fp+fm)                  # costo medido en x_ANTERIOR +- ck*d
        if f < best_f: best_f, best_x = f, x.copy()

    es decir, emparejaba el x NUEVO con el costo del x VIEJO: `best_x`
    terminaba siendo un punto que nunca se evaluo.  Ademas, quedarse con el
    argmin de una historia RUIDOSA selecciona la realizacion de ruido mas
    favorable, no el mejor parametro (sesgo a la baja de ~2*sigma con 40
    iteraciones).  Correccion:
      * `f_center` se asocia al x que lo genero (x_prev), no al actualizado;
      * el criterio de mejor punto usa una MEDIA MOVIL (`smooth_window`) del
        costo, que promedia el ruido antes de comparar;
      * `best_f` se re-evalua al final sobre `best_x` para que el valor
        reportado corresponda de verdad a los parametros devueltos.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float).copy()
    P = len(x)
    # helper: average multiple cost evaluations to reduce stochastic noise
    def eval_avg(xx):
        s = 0.0
        for _ in range(max(1, int(n_resamples))):
            s += float(cost(xx))
        return s / max(1, int(n_resamples))
    # calibracion de a
    g0 = []
    for _ in range(5):
        d = rng.choice([-1.0, 1.0], size=P)
        g0.append(abs(eval_avg(x + c * d) - eval_avg(x - c * d)) / (2 * c))
    gbar = float(np.mean(g0)) + 1e-12
    f0 = eval_avg(x)
    best_x, best_f = x.copy(), f0
    hist = [f0]
    win = []                       # ventana movil para suavizar el ruido
    A = 0.1 * maxiter
    if calibrate:
        a = float(target_step) * (A + 1.0) ** alpha / gbar
    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        d = rng.choice([-1.0, 1.0], size=P)
        x_prev = x.copy()                       # <-- punto que genera fp,fm
        fp, fm = eval_avg(x_prev + ck * d), eval_avg(x_prev - ck * d)
        x = x_prev - ak * (fp - fm) / (2 * ck) * d
        f = 0.5 * (fp + fm)                     # costo asociado a x_prev
        hist.append(f)
        win.append(f)
        if len(win) > max(1, int(smooth_window)):
            win.pop(0)
        f_smooth = float(np.mean(win))
        # el mejor punto se asocia SIEMPRE al x que produjo la medicion
        if f_smooth < best_f:
            best_f, best_x = f_smooth, x_prev.copy()
        if callback is not None:
            callback(k, x, f)
    fx = eval_avg(x)
    if fx < best_f:
        best_f, best_x = fx, x.copy()
    if final_reeval:
        # el valor reportado debe corresponder a los parametros devueltos
        best_f = eval_avg(best_x)
    return dict(x=best_x, fun=best_f, history=np.array(hist))


def cobyla_minimize(cost, x0, maxiter=200, rhobeg=0.5, callback=None):
    """
    Simple COBYLA wrapper that records cost history via callback.
    Returns dict consistent with `spsa_minimize` for downstream use.
    """
    from scipy.optimize import minimize
    hist = []

    def _cb(xk):
        try:
            hist.append(float(cost(xk)))
        except Exception:
            hist.append(np.nan)

    # initial cost
    try:
        hist.append(float(cost(x0)))
    except Exception:
        hist.append(np.nan)

    r = minimize(cost, x0, method="COBYLA",
                 options=dict(maxiter=int(maxiter), rhobeg=float(rhobeg)),
                 callback=_cb)
    return dict(x=r.x, fun=float(r.fun), history=np.array(hist))


def lbfgs_minimize(cost, x0, maxiter=200, callback=None):
    """
    L-BFGS-B wrapper using `scipy.optimize.minimize`.
    Uses numerical gradients (finite differences) provided by SciPy.
    Records cost history via callback and returns dict like other optimizers.
    NOTE: L-BFGS is gradient-based and is sensitive to noise; it's best used
    after a noise-free pre-optimization or with smoothing/averaging.
    """
    from scipy.optimize import minimize
    hist = []

    def _cb(xk):
        try:
            hist.append(float(cost(xk)))
        except Exception:
            hist.append(np.nan)

    # initial cost
    try:
        hist.append(float(cost(x0)))
    except Exception:
        hist.append(np.nan)

    r = minimize(cost, x0, method="L-BFGS-B",
                 options=dict(maxiter=int(maxiter), disp=False),
                 callback=_cb)
    return dict(x=r.x, fun=float(r.fun), history=np.array(hist))


# =====================================================================
# 5) GRADIENTE POR PARAMETER-SHIFT  +  ADAM
# =====================================================================
def parameter_shift_grad(evs_fn, combine, x, shift=np.pi / 2, jac_eps=1e-6):
    """
    Gradiente EXACTO de un costo VQLS por la regla de desplazamiento de
    parametro.

    POR QUE NO SE APLICA LA REGLA DIRECTAMENTE AL COSTO
    ---------------------------------------------------
    Los tres costos de `nisq.make_cost_observables` son RAZONES de valores
    esperados,  C = g(E_0, E_1)  con  E_k(th) = <psi(th)| M_k |psi(th)>:

        local  : C = E_0 / E_1
        global : C = 1 - E_0 / E_1
        energy : C = -0.5 ||b||^2 E_0 / E_1

    La regla de parameter-shift de dos terminos
        dE/dth_i = [ E(th + pi/2 e_i) - E(th - pi/2 e_i) ] / 2
    es exacta para generadores con dos autovalores distintos (+-1/2), que es
    el caso de RY (Mitarai et al., PRA 98, 032309 (2018); Schuld et al., PRA
    99, 032331 (2019)).  Pero una RAZON de dos de esas funciones NO es un
    polinomio trigonometrico, asi que aplicar la regla a C directamente da un
    gradiente SESGADO.  La forma correcta es la regla de la cadena:

        dC/dth_i = SUM_k  (dg/dE_k)  *  (dE_k/dth_i)

    Aqui `dE_k/dth_i` sale del parameter-shift (exacto, 2P evaluaciones del
    VECTOR de valores esperados) y `dg/dE_k` se obtiene por diferencias
    centradas sobre `combine`, que es una funcion escalar EXPLICITA de dos
    numeros: no tiene ruido de disparo, asi que la diferencia finita es
    determinista y exacta a ~1e-10.  No se introduce varianza extra.

    Parametros
    ----------
    evs_fn : callable(x) -> np.ndarray
        Devuelve el vector de valores esperados (p.ej. `Evaluator.raw_evs` o
        cualquier variante mitigada).
    combine : callable(np.ndarray) -> float
        El `combine` devuelto por `make_cost_observables`.

    Coste: 2 * n_parametros evaluaciones del vector de observables.
    """
    x = np.asarray(x, float)
    ev0 = np.asarray(evs_fn(x), float)
    J = np.empty(len(ev0))
    for k in range(len(ev0)):
        h = float(jac_eps) * max(1.0, abs(ev0[k]))
        ep, em = ev0.copy(), ev0.copy()
        ep[k] += h
        em[k] -= h
        J[k] = (float(combine(ep)) - float(combine(em))) / (2.0 * h)
    g = np.empty(len(x))
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += shift
        xm[i] -= shift
        dE = 0.5 * (np.asarray(evs_fn(xp), float) -
                    np.asarray(evs_fn(xm), float))
        g[i] = float(J @ dE)
    return g


def adam_minimize(cost, grad, x0, maxiter=200, lr=0.05, b1=0.9, b2=0.999,
                  eps=1e-8, callback=None):
    """
    Adam (Kingma & Ba, ICLR 2015) sobre un gradiente estocastico.

    Es el companero natural de `parameter_shift_grad`: el gradiente de
    parameter-shift bajo un numero finito de disparos es un estimador
    INSESGADO pero ruidoso, y Adam esta disenado exactamente para eso (a
    diferencia de L-BFGS-B, que asume un gradiente determinista y se detiene
    de inmediato con objetivos estocasticos).
    """
    x = np.asarray(x0, float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    f0 = float(cost(x))
    hist = [f0]
    best_x, best_f = x.copy(), f0
    for t in range(1, int(maxiter) + 1):
        g = np.asarray(grad(x), float)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
        f = float(cost(x))
        hist.append(f)
        if f < best_f:
            best_f, best_x = f, x.copy()
        if callback is not None:
            callback(t, x, f)
    return dict(x=best_x, fun=best_f, history=np.array(hist))
