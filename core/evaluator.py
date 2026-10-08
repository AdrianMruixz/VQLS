#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vqls_nisq.py -- VQLS escalable NISQ para viga 1D/2D y calor 1D/2D
==================================================================
Pila completa, verificada con Qiskit 2.5 / Aer 0.17:

  ANSATZ      hardware-efficient RY + entrelazador BRICKWORK (par-impar).
              Mismo numero de CX que la cadena lineal pero profundidad de
              2 capas en vez de n-1  ->  respuesta directa al punto 2 del
              correo de Ethan (reordenar compuertas para que el transpilador
              las lea como paralelas).

  COSTO       LOCAL de Bravo-Prieto et al., Quantum 7, 1188 (2023):
                  C_L = <x|H_L|x> / <x|A'A|x>
                  H_L = A' U (1 - (1/n) SUM_j |0_j><0_j| (x) 1_jbar) U' A
              Evita los barren plateaus dependientes del costo (Cerezo 2021)
              que SI tiene el costo global. Tambien disponibles "global" y
              "energy" (Sato 2021) para comparacion.

  PRECOND     none / jacobi / ichol / amg / cheby.  TODOS clasicos y
              simetricos (M^-1/2 A M^-1/2) -> SPD preservado, sin coste de
              profundidad de circuito (Hosaka et al., arXiv:2312.15657).

  RUIDO       ``uniform``: depolarizante 1q/2q + relajacion termica + error
              de lectura asimetrico. ``fake-athens``: snapshot calibrado de
              IBM FakeAthensV2 (topologia, duraciones y errores por qubit/
              arista), simulado localmente con Aer.

  MITIGACION  DD, Pauli twirling, TREX, ZNE (folding o model-exact/PEA),
              CDR, GDEM, PEC.  Ver el bloque MITIGACION mas abajo.

  OPTIMIZADOR SPSA con warm start encadenado (profundidad y n_qubits).

Autor: asistencia de tesis A. Magana.  Licencia: uso academico.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nisq_core import (spsa_minimize, cobyla_minimize, lbfgs_minimize,   # noqa: F401
                       adam_minimize, parameter_shift_grad,
                       _extrapolate, scaled_eps, safe_ratio_combine,
                       check_mitiq_version)

try:
    import pyamg
    _HAVE_PYAMG = True
except ImportError:
    _HAVE_PYAMG = False

try:
    import mitiq
    # NOTA: `fold_local` NO existe en la API publica de Mitiq >=1.0
    # (verificado: mitiq.zne.scaling solo expone fold_global, fold_all,
    # fold_gates_at_random, layer_folding...). El nombre historico mas
    # cercano a un "folding local determinista por compuerta" es fold_all
    # (cada compuerta se pliega round((scale-1)/2) veces, igual que el
    # fold_cx propio del repo). Un import roto aqui hacia que _HAVE_MITIQ
    # quedara en False SIEMPRE aunque mitiq estuviera instalado -- bug
    # corregido.
    from mitiq.zne.scaling import fold_global, fold_all as fold_local
    from mitiq.zne.inference import RichardsonFactory
    try:
        from mitiq.zne.inference import LinearFactory
        _HAVE_LINEAR_FACTORY = True
    except ImportError:                      # pragma: no cover
        _HAVE_LINEAR_FACTORY = False
    from mitiq.zne import execute_with_zne
    _HAVE_MITIQ = True
except Exception:
    _HAVE_MITIQ = False
    _HAVE_LINEAR_FACTORY = False


def _load_fake_athens_v2():
    """Carga perezosa del snapshot IBM para no afectar el modo ``uniform``."""
    try:
        from qiskit_ibm_runtime.fake_provider import FakeAthensV2
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "--noise-model fake-athens requiere qiskit-ibm-runtime. "
            "Instala una version que incluya FakeAthensV2."
        ) from exc
    return FakeAthensV2()


# =====================================================================
# 1) SISTEMAS FEM  (verificacion analitica incluida)
# =====================================================================
def beam1d_system(n_dof, E=210e9, I=8.33e-6, L_total=1.0, P=1000.0):
    """
    Viga en voladizo Euler-Bernoulli, elementos Hermite cubicos.
    2 GDL por nodo (w, theta).  Empotrada en x=0, carga puntual P en la punta.
    n_dof = 2 * n_elem  ->  n_dof in {2,4,8,16,32,64} = n_elem {1,2,4,8,16,32}

    VERIFICACION: w_punta debe ser exactamente P L^3 / (3 E I).
    """
    assert n_dof % 2 == 0, "n_dof debe ser par (2 GDL por nodo)"
    n_el = n_dof // 2
    Le = L_total / n_el
    n_nodes = n_el + 1
    K = np.zeros((2 * n_nodes, 2 * n_nodes))
    c = E * I / Le ** 3
    ke = c * np.array([
        [12.0,      6 * Le,      -12.0,     6 * Le],
        [6 * Le,  4 * Le ** 2,  -6 * Le,  2 * Le ** 2],
        [-12.0,    -6 * Le,      12.0,    -6 * Le],
        [6 * Le,  2 * Le ** 2,  -6 * Le,  4 * Le ** 2],
    ])
    for e in range(n_el):
        idx = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[np.ix_(idx, idx)] += ke
    f = np.zeros(2 * n_nodes)
    f[2 * (n_nodes - 1)] = P                      # carga transversal en la punta
    # empotramiento: elimina w0, theta0
    K = K[2:, 2:]
    f = f[2:]
    meta = dict(kind="beam1d", E=E, I=I, L=L_total, P=P, n_el=n_el,
                w_tip_analytic=P * L_total ** 3 / (3.0 * E * I))
    return K, f, meta


def heat1d_system(n_dof, k=1.0, L_total=1.0, q=1.0):
    """
    Conduccion 1D estacionaria, P1 lineales, Dirichlet homogeneo en ambos
    extremos, fuente uniforme q.  K = (k/h) tridiag(-1,2,-1).
    VERIFICACION: T_max analitico = q L^2 / (8 k).
    """
    n = n_dof
    h = L_total / (n + 1)
    K = (k / h) * (np.diag(2.0 * np.ones(n)) +
                   np.diag(-1.0 * np.ones(n - 1), 1) +
                   np.diag(-1.0 * np.ones(n - 1), -1))
    f = q * h * np.ones(n)
    meta = dict(kind="heat1d", k=k, L=L_total, q=q, h=h,
                T_max_analytic=q * L_total ** 2 / (8.0 * k))
    return K, f, meta


def heat2d_system(n_dof, k=1.0, L_total=1.0, q=1.0):
    """Poisson 2D, 5 puntos, malla m x m con m^2 = n_dof."""
    m = int(round(np.sqrt(n_dof)))
    assert m * m == n_dof, "heat2d requiere n_dof cuadrado perfecto"
    h = L_total / (m + 1)
    T = (np.diag(2.0 * np.ones(m)) + np.diag(-1.0 * np.ones(m - 1), 1)
         + np.diag(-1.0 * np.ones(m - 1), -1))
    Im = np.eye(m)
    K = (k / h ** 2) * (np.kron(T, Im) + np.kron(Im, T))
    f = q * np.ones(m * m)
    return K, f, dict(kind="heat2d", m=m, h=h)


def beam2d_system(n_dof, E=210e9, nu=0.3, Lx=1.0, Ly=0.1, P=1000.0):
    """
    Viga 2D en voladizo, Q4 tension plana, malla nx x ny.
    Se elige la malla mas cuadrada posible tal que los GDL libres = n_dof.
    Si no hay ajuste exacto se rellena con d*I (padding SPD) aguas arriba.
    """
    best = None
    for ny in range(1, 33):
        for nx in range(1, 65):
            ndf = 2 * (nx) * (ny + 1)          # nodos libres: columnas 1..nx
            if ndf == n_dof:
                cand = (abs(nx / max(ny, 1) - 8.0), nx, ny)
                if best is None or cand < best:
                    best = cand
    if best is None:
        raise ValueError(f"beam2d: no hay malla Q4 con exactamente {n_dof} GDL")
    _, nx, ny = best
    hx, hy = Lx / nx, Ly / ny
    G = E / (2 * (1 + nu))
    D = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0],
                                      [0, 0, (1 - nu) / 2]])
    gp = 1.0 / np.sqrt(3.0)
    ke = np.zeros((8, 8))
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            dN = 0.25 * np.array([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
            J = np.array([[hx / 2, 0.0], [0.0, hy / 2]])
            dNx = np.linalg.solve(J, dN)
            B = np.zeros((3, 8))
            B[0, 0::2] = dNx[0]
            B[1, 1::2] = dNx[1]
            B[2, 0::2] = dNx[1]
            B[2, 1::2] = dNx[0]
            ke += B.T @ D @ B * np.linalg.det(J)
    nnode = (nx + 1) * (ny + 1)
    K = np.zeros((2 * nnode, 2 * nnode))
    nid = lambda i, j: j * (nx + 1) + i
    for j in range(ny):
        for i in range(nx):
            n0, n1 = nid(i, j), nid(i + 1, j)
            n2, n3 = nid(i + 1, j + 1), nid(i, j + 1)
            idx = np.array([2 * n0, 2 * n0 + 1, 2 * n1, 2 * n1 + 1,
                            2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1])
            K[np.ix_(idx, idx)] += ke
    f = np.zeros(2 * nnode)
    f[2 * nid(nx, ny // 2) + 1] = -P
    free = np.array([d for d in range(2 * nnode)
                     if (d // 2) % (nx + 1) != 0])          # x=0 empotrado
    K = K[np.ix_(free, free)]
    f = f[free]
    _ = G
    return K, f, dict(kind="beam2d", nx=nx, ny=ny)


SYSTEM_BUILDERS = {
    "beam1d": beam1d_system,
    "beam2d": beam2d_system,
    "heat1d": heat1d_system,
    "heat2d": heat2d_system,
}


def load_system(family, n_dof, datadir="."):
    """
    Intenta cargar K,f desde .npz (los sistemas reales de FEniCSx del alumno).
    Si no existen, genera el sistema FEM interno equivalente.
    Convenciones probadas: <family>_<ndof>DOF_K.npz / _f.npz  y variantes.
    """
    cands = [
        (f"{family}_{n_dof}DOF_K.npz", f"{family}_{n_dof}DOF_f.npz"),
        (f"{family}_{n_dof}_K.npz", f"{family}_{n_dof}_f.npz"),
        (f"viga_{n_dof}DOF_K.npz", f"viga_{n_dof}DOF_f.npz"),
        (f"beam1d_dof{n_dof}_K.npz", f"beam1d_dof{n_dof}_f.npz"),
    ]
    for kp, fp in cands:
        kpath, fpath = os.path.join(datadir, kp), os.path.join(datadir, fp)
        if os.path.exists(kpath) and os.path.exists(fpath):
            def _get(p):
                z = np.load(p, allow_pickle=True)
                if "arr_0" in z:
                    a = z["arr_0"]
                else:
                    a = z[list(z.keys())[0]]
                a = np.asarray(a)
                return a.todense() if sp.issparse(a) else a
            K = np.asarray(_get(kpath), float)
            f = np.asarray(_get(fpath), float).ravel()
            # A bare K/f archive carries no boundary conditions, units or
            # continuum parameters.  It is valid for algebraic FEM-error
            # metrics, but must never be silently compared with a default
            # closed-form continuum solution.
            return K, f, dict(kind=family, src=kp, analytic_available=False)
    K, f, meta = SYSTEM_BUILDERS[family](n_dof)
    meta["src"] = "builtin_fem"
    return K, f, meta


def _validate_spd_system(K, f, family, rtol=1e-10):
    """Fail early instead of silently repairing an invalid FEM system."""
    K = np.asarray(K, float)
    f = np.asarray(f, float).ravel()
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"{family}: K must be square; got shape {K.shape}")
    if len(f) != K.shape[0]:
        raise ValueError(f"{family}: len(f)={len(f)} but K is {K.shape}")
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(f))):
        raise ValueError(f"{family}: K/f contain NaN or Inf")
    asym = float(np.linalg.norm(K - K.T, ord=np.inf))
    scale = max(float(np.linalg.norm(K, ord=np.inf)), 1.0)
    if asym > rtol * scale:
        raise ValueError(
            f"{family}: K is not symmetric (relative infinity-norm asymmetry "
            f"{asym / scale:.3e}); refusing to symmetrize physical input."
        )
    lam_min = float(np.linalg.eigvalsh(0.5 * (K + K.T))[0])
    if not lam_min > 0.0:
        raise ValueError(f"{family}: K is not SPD (lambda_min={lam_min:.6e})")


# =====================================================================
# 2) PRECONDICIONADORES  (todos clasicos, simetricos, SPD-preservantes)
# =====================================================================
def _ichol0(A, eps=1e-10):
    """IC(0): Cholesky incompleta con el patron de dispersion de A."""
    n = A.shape[0]
    L = np.zeros_like(A)
    for j in range(n):
        s = A[j, j] - np.sum(L[j, :j] ** 2)
        L[j, j] = np.sqrt(max(s, eps))
        for i in range(j + 1, n):
            if A[i, j] != 0.0:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L


def _sym_inv_sqrt(M, floor=1e-14):
    """M^{-1/2} para M SPD (por descomposicion espectral)."""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.maximum(w, floor * max(w.max(), 1.0))
    return (V * (w ** -0.5)) @ V.T


def _sym_sqrt(M, floor=1e-14):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.maximum(w, floor * max(w.max(), 1.0))
    return (V * np.sqrt(w)) @ V.T


def _chebyshev_inverse_poly(A, degree):
    """
    p_d(A) ~ A^{-1} por serie de Chebyshev en [lmin, lmax].
    Devuelve p_d(A) SPD (se proyecta si algun autovalor sale <= 0).
    Como p_d(A) conmuta con A, el sistema precondicionado simetrico es
    exactamente  A_pre = p_d(A) A,  b_pre = p_d(A) b.
    """
    lam, V = np.linalg.eigh(0.5 * (A + A.T))
    lo, hi = float(max(lam[0], 1e-300)), float(lam[-1])
    d = max(1, int(degree))

    def _T(dd, x):
        """T_dd(x) valido dentro y fuera de [-1,1]."""
        x = np.asarray(x, float)
        out = np.empty_like(x)
        inn = np.abs(x) <= 1.0
        out[inn] = np.cos(dd * np.arccos(x[inn]))
        xo = x[~inn]
        out[~inn] = np.sign(xo) ** dd * np.cosh(dd * np.arccosh(np.abs(xo)))
        return out

    # Polinomio residual optimo en norma-sup sobre [lo,hi]:
    #   r_d(l) = T_d( (hi+lo-2l)/(hi-lo) ) / T_d( (hi+lo)/(hi-lo) )
    # El espectro precondicionado es  l*p(l) = 1 - r_d(l)  in  [1-rho, 1+rho],
    # con rho = 1/|T_d((hi+lo)/(hi-lo))|.  De aqui sale la ley d ~ O(sqrt(kappa)).
    w0 = (hi + lo) / (hi - lo) if hi > lo else 2.0
    Td0 = float(_T(d, np.array([w0]))[0])
    r = _T(d, (hi + lo - 2.0 * lam) / (hi - lo)) / Td0
    pl = (1.0 - r) / np.where(np.abs(lam) > 1e-300, lam, 1e-300)
    return (V * pl) @ V.T


def _amg_inverse_operator(A, family, Dv=None, meta=None):
    """
    Construye densamente M^{-1} = un ciclo V de AMG (agregacion suavizada),
    con el espacio nulo cercano correcto por familia. Se simetriza.
    Nota de escala: aqui se densifica porque n<=64 (benchmark). El setup de
    AMG es O(nnz); la densificacion es solo para poder formar M^{-1/2}.
    """
    if not _HAVE_PYAMG:
        raise RuntimeError("pyamg no instalado:  pip install pyamg")
    n = A.shape[0]
    Acsr = sp.csr_matrix(A)
    if family == "beam1d":
        # GDL ordenados (w_1, th_1, w_2, th_2, ...) en nodos x_i = i*h.
        # Modos rigidos exactos: traslacion (w=1, th=0) y rotacion (w=x, th=1).
        nn = len(B_idx_w := np.arange(0, n, 2))
        x = np.arange(1, nn + 1, dtype=float)
        B = np.zeros((n, 2))
        B[B_idx_w, 0] = 1.0                     # traslacion
        B[B_idx_w, 1] = x                       # rotacion: w = x
        B[np.arange(1, n, 2), 1] = 1.0          # rotacion: theta = 1
        B[:, 1] /= max(np.linalg.norm(B[:, 1]), 1e-30)
    elif family == "beam2d":
        B = np.zeros((n, 2))
        B[0::2, 0] = 1.0
        B[1::2, 1] = 1.0
    else:
        B = np.ones((n, 1))
    if Dv is not None:
        Dv = np.asarray(Dv, float).ravel()
        if len(Dv) != n or np.any(Dv <= 0.0):
            raise ValueError("invalid Dv passed to AMG near-kernel")
        B = B / Dv[:, None]
    for j in range(B.shape[1]):
        B[:, j] /= max(np.linalg.norm(B[:, j]), 1e-300)
    ml = pyamg.smoothed_aggregation_solver(
        Acsr, B=B, smooth="jacobi",
        presmoother=("gauss_seidel", {"sweep": "symmetric"}),
        postsmoother=("gauss_seidel", {"sweep": "symmetric"}),
        max_coarse=4)
    P = ml.aspreconditioner(cycle="V")
    Minv = np.column_stack([P.matvec(e) for e in np.eye(n)])
    Minv = 0.5 * (Minv + Minv.T)
    w = np.linalg.eigvalsh(Minv)
    if w[0] <= 0:                                # garantiza SPD
        Minv += (abs(w[0]) + 1e-12 * w[-1]) * np.eye(n)
    return Minv


def prepare(family, n_dof, precond="amg", cheby_degree=8, datadir=".",
            equilibrate=None):
    """
    Devuelve el sistema listo para VQLS:
      A (SPD, ||A||_2 = 1), b (unitario), nq, kappa_raw, kappa_pre, y todo lo
      necesario para volver al espacio fisico.

    Cadena:  K,f  ->  padding SPD (d*I)  ->  equilibrado Jacobi  ->  precond
             ->  normalizacion espectral.
    kappa_raw se mide SOBRE LA MATRIZ CRUDA (antes de cualquier normalizacion),
    que es la unica lectura que hace testable la hipotesis de la tesis.
    """
    K, f, meta = load_system(family, n_dof, datadir)
    _validate_spd_system(K, f, family)
    n0 = K.shape[0]
    nq = int(np.ceil(np.log2(max(n0, 2))))
    N = 1 << nq

    # --- padding SPD (d*I, NO ceros: los ceros hacen singular la matriz) ---
    d_mean = float(np.mean(np.diag(K))) or 1.0
    Kp = np.eye(N) * d_mean
    Kp[:n0, :n0] = K
    fp = np.zeros(N)
    fp[:n0] = f
    # Input symmetry was checked above; this only removes roundoff from the
    # explicitly constructed padded block.
    Kp = 0.5 * (Kp + Kp.T)
    kappa_raw = float(np.linalg.cond(Kp))
    x_phys = np.linalg.solve(Kp, fp)

    # --- equilibrado diagonal simetrico (Jacobi) ---
    # POLITICA (corregida): `equilibrate=None` -> automatico.
    #   precond="none"  -> SIN equilibrado  (kappa_pre == kappa_raw: el control
    #                      honesto del experimento, "k raw").
    #   precond="jacobi"-> SOLO equilibrado D^-1/2 A D^-1/2.
    #   resto           -> equilibrado + precondicionador (equilibrar primero
    #                      es estandar y mejora ICHOL/AMG).
    # BUG ANTERIOR: `equilibrate=True` fijo hacia que "none" y "jacobi"
    # produjeran EXACTAMENTE la misma matriz (verificado: ||A_none-A_jac||=0),
    # de modo que el punto de referencia sin precondicionar no existia.
    if equilibrate is None:
        equilibrate = (precond != "none")
    if equilibrate:
        dg = np.diag(Kp).copy()
        dg[dg <= 0] = 1.0
        Dv = 1.0 / np.sqrt(dg)
        Ke = (Kp * Dv[None, :]) * Dv[:, None]
        fe = Dv * fp
    else:
        Dv = np.ones(N)
        Ke, fe = Kp.copy(), fp.copy()
    Ke = 0.5 * (Ke + Ke.T)
    # --- normalizacion PROVISIONAL (solo para construir el precondicionador
    #     en una escala sana: ICHOL usa desplazamientos alpha absolutos).
    #     La normalizacion DEFINITIVA se aplica DESPUES del precondicionador.
    s_pre = float(np.linalg.norm(Ke, 2))
    norm_fe_pre = float(np.linalg.norm(fe))
    Ke = Ke / s_pre
    fe = fe / norm_fe_pre

    # --- precondicionador ---
    Msqrt_inv = None                    # M^{-1/2} en el espacio equilibrado
    if precond in ("none", "jacobi"):
        A_, b_ = Ke, fe
    elif precond == "ichol":
        best = None
        for al in [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            L = _ichol0(Ke + al * np.eye(N))
            if not np.all(np.isfinite(L)) or np.any(np.diag(L) <= 0):
                continue
            try:
                Li = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(Li)):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                Aa = Li @ Ke @ Li.T
            if not np.all(np.isfinite(Aa)):
                continue
            Aa = 0.5 * (Aa + Aa.T)
            ww = np.linalg.eigvalsh(Aa)
            if ww[0] <= 0:
                continue
            kk = float(ww[-1] / ww[0])
            if best is None or kk < best[0]:
                best = (kk, Aa, Li @ fe, Li.T)
        if best is None:
            raise RuntimeError("ICHOL irrecuperable")
        _, A_, b_, Msqrt_inv = best
    elif precond == "cholesky":
        L = np.linalg.cholesky(Ke)
        Li = np.linalg.inv(L)
        A_, b_, Msqrt_inv = Li @ Ke @ Li.T, Li @ fe, Li.T
    elif precond == "amg":
        Minv = _amg_inverse_operator(Ke, meta["kind"], Dv=Dv, meta=meta)
        S = _sym_sqrt(Minv)                       # M^{-1/2}
        A_ = 0.5 * (S @ Ke @ S + (S @ Ke @ S).T)
        b_ = S @ fe
        Msqrt_inv = S
    elif precond == "cheby":
        # p_d(A) es un polinomio en A -> conmuta con A -> p_d(A)A ya es
        # simetrico. NO hace falta (ni conviene) la raiz cuadrada: tomarla
        # amplifica los autovalores donde el ajuste de 1/x se vuelve negativo.
        Pd = _chebyshev_inverse_poly(Ke, cheby_degree)
        A_ = 0.5 * (Pd @ Ke + (Pd @ Ke).T)
        b_ = Pd @ fe
        Msqrt_inv = None            # la solucion ya vive en el espacio equilibrado
    else:
        raise ValueError(f"precond desconocido: {precond}")

    A_ = 0.5 * (A_ + A_.T)
    ww = np.linalg.eigvalsh(A_)
    spd_ok = bool(ww[0] > 0)
    # kappa es invariante de escala -> se puede medir aqui sin sesgo
    kappa_pre = (float(ww[-1] / ww[0]) if spd_ok
                 else float(np.abs(ww).max() / max(np.abs(ww).min(), 1e-300)))

    # --- NORMALIZACION DEFINITIVA: **DESPUES** del precondicionador ---------
    # BUG ANTERIOR: se normalizaba solo ANTES, asi que tras ichol/cholesky/amg
    # la matriz devuelta NO cumplia ||A||_2 = 1 ni ||b|| = 1 (medido en
    # beam1d 16GDL: ||b|| = 74.2 con cholesky, 62.4 con amg, 3.7 con ichol).
    # Consecuencias: (i) `--pauli-atol` es un umbral ABSOLUTO aplicado sobre
    # una escala arbitraria distinta en cada familia/precondicionador, de modo
    # que el truncado no era comparable entre corridas; (ii) `b` deja de ser
    # un estado cuantico legitimo. La reconstruccion en `to_physical` sigue
    # siendo valida porque se acumulan ambos factores en `s` y `nb`.
    s2 = float(np.linalg.norm(A_, 2))
    nb2 = float(np.linalg.norm(b_))
    A = A_ / (s2 if s2 > 0 else 1.0)
    b = b_ / (nb2 if nb2 > 0 else 1.0)
    s_tot = s_pre * (s2 if s2 > 0 else 1.0)
    nb_tot = norm_fe_pre * (nb2 if nb2 > 0 else 1.0)

    x_pre = np.linalg.solve(A, b)
    x_pre = x_pre / np.linalg.norm(x_pre)

    return dict(A=A, b=b, nq=nq, n_orig=n0, N=N, spd_ok=spd_ok,
                kappa_raw=kappa_raw, kappa_pre=kappa_pre,
                x_pre=x_pre, x_phys=x_phys, K_pad=Kp, f_pad=fp,
                Dv=Dv, Msqrt_inv=Msqrt_inv, s=s_tot, nb=nb_tot,
                a_norm=float(np.linalg.norm(A, 2)),
                b_norm=float(np.linalg.norm(b)),
                precond=precond, meta=meta)


def to_physical(psi, sysd):
    """
    Reconstruye el desplazamiento fisico a partir del estado cuantico.
    psi (normalizado, espacio precondicionado+normalizado)
      -> alpha por minimos cuadrados en el espacio precondicionado
      -> deshace M^{-1/2}, normalizacion y equilibrado.
    """
    A, b = sysd["A"], sysd["b"]
    p = psi / np.linalg.norm(psi)
    Ap = A @ p
    den = float(Ap @ Ap)
    alpha = float(b @ Ap) / den if den > 1e-300 else 0.0
    y = alpha * p                                    # sol. del sistema normalizado
    y = y * (sysd["nb"] / sysd["s"])                 # deshace normalizacion
    if sysd["Msqrt_inv"] is not None:
        y = sysd["Msqrt_inv"] @ y                    # espacio equilibrado
    y = sysd["Dv"] * y                               # espacio fisico
    return y[:sysd["n_orig"]]


# =====================================================================
# 3) ANSATZ  --  reduccion de profundidad (punto 2 del correo de Ethan)
# =====================================================================
from qiskit import QuantumCircuit, transpile                       # noqa: E402
from qiskit.circuit import ParameterVector                          # noqa: E402
from qiskit.circuit.library import XGate, YGate, ZGate, IGate       # noqa: E402
from qiskit.quantum_info import (SparsePauliOp, Operator,           # noqa: E402
                                 Statevector)
from qiskit.transpiler import CouplingMap, PassManager, InstructionDurations
from qiskit.transpiler.basepasses import TransformationPass         # noqa: E402
from qiskit.transpiler.passes import (ALAPScheduleAnalysis,         # noqa: E402
                                      PadDynamicalDecoupling)
from qiskit.dagcircuit import DAGCircuit                            # noqa: E402
from qiskit_aer import AerSimulator                                 # noqa: E402
from qiskit_aer.primitives import EstimatorV2                       # noqa: E402
from qiskit_aer.noise import (NoiseModel, depolarizing_error,       # noqa: E402
                              thermal_relaxation_error, ReadoutError)

CLIFF = np.array([0.0, np.pi / 2, -np.pi / 2, np.pi])


def build_ansatz(nq, depth, entangler="brick"):
    """
    RY hardware-efficient con capa RY final.  (depth+1)*nq parametros.

    entangler="linear": CX(0,1),CX(1,2),...,CX(n-2,n-1)  -> profundidad n-1
                        por capa (todas las CX comparten qubit con la anterior,
                        el transpilador NO puede paralelizarlas).
    entangler="brick" : primero los pares (0,1),(2,3),... y luego (1,2),(3,4),...
                        MISMO numero de CX, mismo mapa lineal, pero cada grupo
                        actua sobre qubits DISJUNTOS -> profundidad 2 por capa.
                        Esto es exactamente la "reordenacion de compuertas para
                        que el transpilador las interprete como operaciones
                        paralelas" que sugiere Ethan.  Es gratis.
    """
    pv = ParameterVector(f"t{nq}_{depth}", (depth + 1) * nq)
    qc = QuantumCircuit(nq)
    for d in range(depth):
        for i in range(nq):
            qc.ry(pv[d * nq + i], i)
        if entangler == "linear":
            for i in range(nq - 1):
                qc.cx(i, i + 1)
        else:                                   # brick (par-impar)
            for i in range(0, nq - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, nq - 1, 2):
                qc.cx(i, i + 1)
    for i in range(nq):
        qc.ry(pv[depth * nq + i], i)
    return qc, pv


def two_qubit_depth(qc):
    """Profundidad contando SOLO compuertas de 2 qubits (la metrica de NIBP)."""
    return qc.depth(lambda inst: inst.operation.num_qubits == 2)


# =====================================================================
# 4) FUNCIONES DE COSTO  (local de Bravo-Prieto incluida)
# =====================================================================
def _householder_prep(b):
    """
    U real ortogonal con U|0..0> = |b>.  Reflexion de Householder:
        U = I - 2 v v^T ,  v = (e0 - b)/||e0 - b||
    Es simetrica y ortogonal, y esta definida sin ambiguedad -> el costo
    local es reproducible exactamente (depende de U, no solo de U|0>).
    """
    n = len(b)
    e0 = np.zeros(n)
    e0[0] = 1.0
    bb = b / np.linalg.norm(b)
    v = e0 - bb
    nv = np.linalg.norm(v)
    if nv < 1e-14:
        return np.eye(n)
    v = v / nv
    return np.eye(n) - 2.0 * np.outer(v, v)


def local_projector(nq):
    """P_L = (1/n) SUM_j |0><0|_j (x) I_resto   (operador de Bravo-Prieto)."""
    N = 1 << nq
    P = np.zeros((N, N))
    idx = np.arange(N)
    for j in range(nq):
        # qiskit: qubit j es el bit j (little-endian)
        diag = ((idx >> j) & 1) == 0
        P[idx[diag], idx[diag]] += 1.0
    return P / nq


def make_cost_observables(A, b, nq, atol=1e-3, cost_type="local",
                          atol_mode="rel"):
    """
    Devuelve (obs, combine, mats, counts).

    cost_type="local"  -- Bravo-Prieto et al., Quantum 7, 1188 (2023), Eq.(9):
        C_L = <x|H_L|x> / <x|A'A|x>,
        H_L = A' U (I - P_L) U' A   con  U|0> = |b>,  P_L = (1/n) SUM |0_j><0_j|.
        C_L = 0  <=>  C_G = 0, y  C_G <= n*C_L.  Evita el barren plateau
        dependiente del costo del costo global (Cerezo et al. 2021).

    cost_type="global" -- C_G = 1 - |<b|A|x>|^2 / (<x|A'A|x> ||b||^2).
        SOLO para comparacion: tiene BP dependiente del costo incluso a
        profundidad O(1).

    cost_type="energy" -- Sato et al. PRA 2021 (minimo negativo finito).
    """
    counts = []

    def op(M):
        """
        Descomposicion de Pauli con truncado.

        atol_mode="rel" (por defecto): el umbral es RELATIVO al coeficiente
        de Pauli mas grande del propio observable,  thr = atol * max|c_k|.
        Con el umbral ABSOLUTO original el mismo --pauli-atol descartaba una
        fraccion completamente distinta del operador en cada familia y para
        cada precondicionador (los coeficientes van de ~7e-2 con `none` a
        ~5e-1 con `cholesky`), de modo que el sesgo de truncado no era
        comparable entre corridas.  atol_mode="abs" recupera el comportamiento
        anterior.
        """
        # Qiskit's conversion and simplify methods have their own default
        # numerical tolerances.  Passing atol=0 to this function must mean
        # *no intentional Pauli pruning*, so disable both internal thresholds
        # explicitly before applying the user-requested threshold below.
        full = SparsePauliOp.from_operator(
            Operator(M), atol=0.0, rtol=0.0
        ).simplify(atol=0.0, rtol=0.0)
        nf = len(full)
        if atol and atol > 0:
            cmax = float(np.max(np.abs(full.coeffs))) if nf else 1.0
            thr = atol * cmax if atol_mode == "rel" else atol
            red = full.simplify(atol=thr, rtol=0.0)
        else:
            red = full
        if len(red) == 0:
            red = SparsePauliOp.from_list([("I" * nq, 0.0)])
        counts.append((nf, len(red)))
        return red

    bb = float(b @ b)
    A2 = A @ A
    # eps escalado por ||A||_2: los tres tipos de costo son razones de
    # valores esperados con denominador dependiente del estado (<x|A'A|x> o
    # <x|A|x>); un floor FIJO esta mal calibrado si ||A||_2 != 1 (pasa con
    # los precondicionadores amg/cheby, que no preservan la norma espectral
    # tras `prepare()`). safe_ratio_combine() (nisq_core.py) recorta el
    # denominador a max(1e-12, ||A||_2*1e-12) y avisa si el optimizador se
    # acerca demasiado al polo -- as-i el costo NUNCA puede irse a +-inf por
    # manipulacion numerica (ver tests/test_numerics.py).
    a_norm = float(np.linalg.norm(A, 2))

    if cost_type == "local":
        U = _householder_prep(b)
        PL = local_projector(nq)
        HL = A @ U @ (np.eye(len(b)) - PL) @ U.T @ A
        HL = 0.5 * (HL + HL.T)
        mats = [HL, A2]
        obs = [op(HL), op(A2)]
        combine = (lambda ev: safe_ratio_combine(
            ev[0], ev[1], a_norm, scale=1.0, label="cost_local"))
    elif cost_type == "global":
        Ab = A @ b
        M0 = np.outer(Ab, Ab) / bb
        mats = [M0, A2]
        obs = [op(M0), op(A2)]
        combine = (lambda ev: 1.0 - safe_ratio_combine(
            ev[0], ev[1], a_norm, scale=1.0, label="cost_global"))
    elif cost_type == "energy":
        Pb = np.outer(b, b) / bb
        # C_energy = -0.5*||b||^2 * <Pb>/<A>  (Sato et al. PRA 2021).
        # Sin proteccion, <A> -> 0 hace que el costo se dispare a -inf y el
        # optimizador puede "hacer trampa" explotando ese polo numerico en
        # vez de minimizar el residual fisico.
        mats = [Pb, A]
        obs = [op(Pb), op(A)]
        combine = (lambda ev: safe_ratio_combine(
            ev[0], ev[1], a_norm, scale=-0.5 * bb, label="cost_energy"))
    else:
        raise ValueError(cost_type)
    return obs, combine, mats, counts


def identity_coeff(spo, nq):
    """Tr(O)/2^n : punto fijo del canal depolarizante global."""
    iden = "I" * nq
    for p, c in zip(spo.paulis.to_labels(), spo.coeffs):
        if p == iden:
            return float(np.real(c))
    return 0.0


# ---- metricas ----------------------------------------------------------
def fidelity(psi, x):
    return float(abs(np.dot(psi / np.linalg.norm(psi),
                            x / np.linalg.norm(x))) ** 2)


def norm_residual(psi, A, b):
    """||A u_q - b|| / ||b||  con u_q = alpha*psi ajustado por minimos cuadrados."""
    p = psi / np.linalg.norm(psi)
    Ap = A @ p
    den = float(Ap @ Ap)
    alpha = float(b @ Ap) / den if den > 1e-300 else 0.0
    return float(np.linalg.norm(A @ (alpha * p) - b) / np.linalg.norm(b))



# ---------------------------------------------------------------------
# 4bis) METRICAS BAJO RUIDO  --  bloque anadido en la revision de metricas
# ---------------------------------------------------------------------
# MOTIVO.  `run_mitigation_study.py` calculaba la fidelidad ruidosa asi:
#
#       rho  = <matriz densidad ruidosa>
#       psi  = np.real(np.diag(rho))          # <-- INCORRECTO
#       psi /= np.linalg.norm(psi)
#
# diag(rho)[i] = |c_i|^2 son PROBABILIDADES, no amplitudes: (a) pierde todos
# los signos -- x_pre de la viga 1D tiene 7 de 16 entradas negativas -- y
# (b) eleva al cuadrado los modulos, cambiando la direccion del vector.
# Verificacion: aplicando esa formula al estado IDEAL EXACTO (F real =
# 1.000000) se obtiene F = 0.0195 y residual = 0.905.  Es decir, el numero
# reportado no medía ruido; medía el error de calculo.
#
# Definiciones usadas aqui:
#   F_mix  = <x|rho|x>          fidelidad de estado mixto contra estado puro
#   F_coh  = |<x|v_1>|^2        v_1 = autovector dominante de rho (componente
#                               coherente).  Separa "el estado se mezclo" de
#                               "el estado apunta a otro sitio".
#   p_eff  = (1-F_mix)/(1-2^-n) intensidad efectiva del canal despolarizante
#   res    = ||A u_q - b||/||b||  con u_q = alpha*v (alpha por minimos cuadr.)
#   L2_pre / L2_fis / L2_ana    ver `metrics_state`.
#
# COTA VERIFICADA (ajuste sobre perturbaciones aleatorias de la solucion):
#       res_norm  <~  kappa_pre * sqrt(1 - F)
# => con kappa_pre = 2.2e3 hace falta F > 0.999999998 para res < 0.1.  Un
# residual alto con F aparentemente buena NO es un bug: es kappa.


def alpha_ls(v, A, b):
    """Escala optima por minimos cuadrados: alpha = (b.Av)/(Av.Av)."""
    Av = A @ v
    den = float(Av @ Av)
    return (float(b @ Av) / den) if den > 1e-300 else 0.0


def analytic_solution(sysd):
    """
    (u_ana, qoi_ana, nombre_qoi) en el espacio fisico de GDL libres, o
    (None, nan, "") si la familia no tiene solucion cerrada implementada.

    viga 1D (Euler-Bernoulli, voladizo, carga P en la punta):
        w(x)     = P x^2 (3L - x) / (6 E I)
        theta(x) = P x (2L - x)   / (2 E I)
        w_punta  = P L^3 / (3 E I)
    calor 1D (Poisson, Dirichlet homogeneo, fuente q uniforme):
        T(x)     = q x (L - x) / (2 k)
        T_max    = q L^2 / (8 k)
    """
    m = sysd.get("meta", {}) or {}
    kind = m.get("kind", "")
    n0 = sysd["n_orig"]
    if kind == "beam1d" and "L" in m and m.get("analytic_available", True):
        E, I, L, P, n_el = m["E"], m["I"], m["L"], m["P"], m["n_el"]
        xs = np.linspace(0.0, L, n_el + 1)[1:]
        u = np.zeros(n0)
        w = P * xs ** 2 * (3 * L - xs) / (6 * E * I)
        th = P * xs * (2 * L - xs) / (2 * E * I)
        u[0::2] = w[:len(u[0::2])]
        u[1::2] = th[:len(u[1::2])]
        return u, float(m["w_tip_analytic"]), "w_punta"
    if kind == "heat1d" and {"k", "L", "q"} <= set(m):
        # -k T'' = q, T(0)=T(L)=0.  This is the same PDE and BCs assembled
        # by heat1d_system(); the previous linear profile was unrelated.
        k, L, q = float(m["k"]), float(m["L"]), float(m["q"])
        h = float(m.get("h", L / (n0 + 1)))
        xs = np.arange(1, n0 + 1) * h
        u = q * xs * (L - xs) / (2.0 * k)
        return u, float(q * L**2 / (8.0 * k)), "T_max"
    # beam2d_system is a Q4 plane-stress model, not an Euler--Bernoulli
    # continuum model.  Do not attach a fabricated beam formula to it.
    return None, float("nan"), ""


def qoi_of(u_phys, sysd):
    """Cantidad de interes: w de la punta (viga) o temperatura maxima."""
    kind = (sysd.get("meta", {}) or {}).get("kind", "")
    if kind == "beam1d":
        return float(u_phys[-2])            # ..., w_n, theta_n
    if kind in ("heat1d", "heat2d"):
        return float(np.max(u_phys))
    return float(np.max(np.abs(u_phys)))


def metrics_state(v, sysd, tag=""):
    """
    Metricas completas de un ESTADO PURO (caso ideal / statevector).

    Niveles de error L2, deliberadamente separados:
      l2_pre          vs A^-1 b   -- espacio precondicionado
      l2_fis          vs K^-1 f   -- ERROR DEL SOLVER en espacio fisico
                                     (es la que valida el VQLS)
      l2_vs_analitica vs u(x) EDP -- solver + error de discretizacion FEM
    """
    A, b = sysd["A"], sysd["b"]
    Kp, fp = sysd["K_pad"], sysd["f_pad"]
    n0 = sysd["n_orig"]
    v = np.asarray(v, complex)
    imag_rel = float(np.linalg.norm(np.imag(v)) / max(np.linalg.norm(v), 1e-300))
    if imag_rel > 1e-12:
        raise ValueError(
            "metrics_state is a real-amplitude FEM reconstruction; received "
            f"a state with relative imaginary norm {imag_rel:.3e}."
        )
    p = np.real(v) / max(np.linalg.norm(v), 1e-300)

    x_ex_pre = np.linalg.solve(A, b)
    u_q_pre = alpha_ls(p, A, b) * p
    res = float(np.linalg.norm(A @ u_q_pre - b) / np.linalg.norm(b))
    l2_pre = float(np.linalg.norm(u_q_pre - x_ex_pre) /
                   max(np.linalg.norm(x_ex_pre), 1e-300))

    u_q = to_physical(p, sysd)
    u_fem = np.linalg.solve(Kp, fp)[:n0]
    e = u_q - u_fem
    nrm = max(np.linalg.norm(u_fem), 1e-300)
    Kff = Kp[:n0, :n0]
    eK = float(np.sqrt(max(e @ (Kff @ e), 0.0)))
    eK_ref = float(np.sqrt(max(u_fem @ (Kff @ u_fem), 1e-300)))

    u_ana, qoi_ana, qoi_name = analytic_solution(sysd)
    qoi_q, qoi_fem = qoi_of(u_q, sysd), qoi_of(u_fem, sysd)
    # QoI analitica EVALUADA EN LOS MISMOS NODOS: separa el error real de
    # discretizacion del artefacto de muestreo (el maximo de T(x) cae en
    # x = L/2, que no es nodo si n es par).
    qoi_ana_nod = qoi_of(u_ana, sysd) if u_ana is not None else float("nan")

    out = dict(
        tag=tag,
        F=fidelity(p, sysd["x_pre"]),
        res_pre=res, l2_pre=l2_pre,
        l2_fis=float(np.linalg.norm(e) / nrm),
        linf_fis=float(np.max(np.abs(e)) / max(np.max(np.abs(u_fem)), 1e-300)),
        eK=eK, eK_rel=eK / eK_ref,
        cos_fis=float(u_q @ u_fem / (np.linalg.norm(u_q) * nrm + 1e-300)),
        qoi_name=qoi_name, qoi_q=qoi_q, qoi_fem=qoi_fem,
        qoi_ana=qoi_ana, qoi_ana_nodal=qoi_ana_nod,
        qoi_rel_vs_fem=abs(qoi_q - qoi_fem) / max(abs(qoi_fem), 1e-300),
        qoi_rel_vs_ana=(abs(qoi_q - qoi_ana) / max(abs(qoi_ana), 1e-300)
                        if np.isfinite(qoi_ana) else float("nan")),
    )
    if u_ana is not None:
        na = max(np.linalg.norm(u_ana), 1e-300)
        out["l2_vs_analitica"] = float(np.linalg.norm(u_q - u_ana) / na)
        out["l2_fem_vs_analitica"] = float(np.linalg.norm(u_fem - u_ana) / na)
    else:
        out["l2_vs_analitica"] = float("nan")
        out["l2_fem_vs_analitica"] = float("nan")
    return out


def density_matrix_of(qc_log, theta, noise_model,
                      basis_gates=("rz", "sx", "x", "y", "cx"),
                      optimization_level=0):
    """rho EXACTA (sin ruido de disparos) del ansatz logico bajo el modelo."""
    qt = transpile(qc_log.assign_parameters(theta),
                   basis_gates=list(basis_gates),
                   optimization_level=optimization_level)
    qt.save_density_matrix()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    return np.asarray(sim.run(qt).result().data(0)["density_matrix"])


def metrics_density(rho, sysd, tag="ruidoso"):
    """
    Diagnostics from rho.

    F_mix and purity are density-matrix observables.  Residual/L2 are instead
    computed for the principal eigenvector and are therefore a *coherent
    component proxy*, not a unique physical solution of the mixed state.
    """
    rho = np.asarray(rho)
    N = rho.shape[0]
    xh = sysd["x_pre"] / np.linalg.norm(sysd["x_pre"])

    F_mix = float(np.real(np.vdot(xh, rho @ xh)))
    purity = float(np.real(np.trace(rho @ rho)))
    p_eff = float(np.clip((1.0 - F_mix) / (1.0 - 1.0 / N), 0.0, 1.0))

    w, V = np.linalg.eigh(rho)
    v = V[:, -1]
    j = int(np.argmax(np.abs(v)))
    v = np.real(v * np.exp(-1j * np.angle(v[j])))       # fija la fase global
    m = metrics_state(v, sysd, tag=tag)
    m.update(F_mix=F_mix, F_coh=m.pop("F"), purity=purity, p_eff=p_eff,
             lam_max_rho=float(w[-1]),
             state_interpretation="principal_eigenvector_coherent_component_proxy")
    return m


def solver_limit(sysd, p_eff, cost_type="local"):
    """
    LIMITE DEL SOLVER.  Bajo ruido, todo valor esperado adquiere un piso
    independiente del estado:
            <O>_ruidoso ~ (1-p) <O>_ideal + p * Tr(O)/2^n
    El costo VQLS es una razon <O1>/<O2>.  Evaluado EN LA SOLUCION:
        local/global :  <A^2>_sol = ||A x||^2/||x||^2 ~ lambda_min^2 ~ kappa^-2
        energy       :  <A>_sol                       ~ lambda_min   ~ kappa^-1
    Si ese valor cae por debajo del piso p*Tr(O2)/2^n, el denominador pasa a
    ser 100% ruido y el MINIMO GLOBAL del costo se mueve fuera de la solucion
    (la razon baja SUBIENDO el denominador hacia lambda_max, no resolviendo).
    Ninguna tecnica de mitigacion corrige esto: haria falta recuperar
    <A^2> ~ 1e-6 desde un valor ruidoso ~1e-1, o sea 5-7 cifras
    significativas, muy por debajo del piso de disparos M^(-1/2).

    Exponentes verificados con ajuste log-log (R^2 = 0.997):
        <A^2>_sol/piso ~ kappa^-1.72 (viga) / kappa^-1.95 (calor)   [teoria -2]
        <A>_sol/piso   ~ kappa^-0.97 (viga) / kappa^-0.98 (calor)   [teoria -1]

    Criterio operativo:  SNR_den >= 1 utilizable, >= 10 comodo.
    Regla practica:  kappa_pre <~ 1/sqrt(p_eff)  (local/global)
                     kappa_pre <~ 1/p_eff        (energy)
    """
    A, nq = sysd["A"], sysd["nq"]
    N = 1 << nq
    xh = sysd["x_pre"] / np.linalg.norm(sysd["x_pre"])
    O2 = (A @ A) if cost_type in ("local", "global") else A
    den_sol = float(xh @ O2 @ xh)
    floor = float(p_eff * np.trace(O2) / N)
    snr = den_sol / max(floor, 1e-300)
    expo = 2.0 if cost_type in ("local", "global") else 1.0
    return dict(cost_type=cost_type, p_eff=p_eff, den_sol=den_sol,
                den_floor=floor, snr_den=snr, kappa_pre=sysd["kappa_pre"],
                kappa_max_est=float(sysd["kappa_pre"] * snr ** (1.0 / expo)),
                verdict=("OK" if snr >= 10 else
                         "MARGINAL" if snr >= 1 else "COSTO CIEGO"))


def exact_noisy_optimum(sysd, p_eff, cost_type="local", iters=400):
    """
    MINIMO GLOBAL EXACTO del costo ruidoso sobre TODO el espacio de Hilbert.

        opt de  [(1-p) x'O1 x + p a] / [(1-p) x'O2 x + p c]   con ||x||=1

    es un programa fraccional cuasiconvexo:
        razon <= t   <=>   lambda_min[(1-p)(O1 - t O2)] <= p (t c - a)
    resuelto por biseccion en t + eigh.  NO depende del ansatz, ni del
    optimizador, ni de los disparos: es la cota superior absoluta de lo que
    cualquier VQLS puede alcanzar con ese costo y ese ruido.
    Devuelve (valor, F_opt, res_opt).
    """
    A, b, nq = sysd["A"], sysd["b"], sysd["nq"]
    N = 1 << nq
    _, _, mats, _ = make_cost_observables(A, b, nq, atol=0.0,
                                          cost_type=cost_type)
    O1, O2 = mats
    a, c = float(np.trace(O1) / N), float(np.trace(O2) / N)
    maximize = (cost_type == "energy")      # C_energy = -k * <Pb>/<A>

    def slack(t):
        M = (1.0 - p_eff) * (O1 - t * O2)
        w, V = np.linalg.eigh(M)
        if maximize:
            return w[-1] - p_eff * (t * c - a), V[:, -1]
        return w[0] - p_eff * (t * c - a), V[:, 0]

    lo, hi = -1e4, 1e4
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val, _ = slack(mid)
        if maximize:
            lo, hi = (mid, hi) if val >= 0 else (lo, mid)
        else:
            lo, hi = (lo, mid) if val <= 0 else (mid, hi)
    t = lo if maximize else hi
    v = np.real(slack(t)[1])
    return float(t), fidelity(v, sysd["x_pre"]), norm_residual(v, A, b)


def residual_bound(kappa, F):
    """Cota verificada: res_norm <~ kappa*sqrt(1-F), saturada en 1."""
    return float(min(1.0, kappa * np.sqrt(max(1.0 - F, 0.0))))


def fidelity_required(kappa, res_target=0.1):
    """Fidelidad minima necesaria para alcanzar `res_target` dado kappa."""
    return float(1.0 - (res_target / max(kappa, 1e-300)) ** 2)


# =====================================================================
# 5) RUIDO
# =====================================================================
def make_noise(p1, p2, t1_us=120.0, t2_us=90.0, tg1_ns=35.0, tg2_ns=400.0,
               p_ro=0.0, ro_asym=0.0, noise_factor=1.0):
    """
    Modelo tipo IBM: depolarizante 1q/2q + relajacion termica + lectura.

    noise_factor k (para ZNE "model-exact" / estilo PEA):
      - depolarizante:  D_lam compuesto k veces = D_{1-(1-lam)^k}   (EXACTO)
      - relajacion:     tiempo de compuerta -> k * t_gate           (EXACTO)
    Esto amplifica el ruido REAL del modelo, sin los artefactos del gate
    folding (que tambien duplica compuertas de 1q y no escala T1/T2 igual).

    p_ro>0 activa error de lectura (sin el, TREX es un no-op).
    ro_asym en [0,1): asimetria  p01 = p_ro*(1+a), p10 = p_ro*(1-a).
    """
    k = float(max(noise_factor, 1e-12))
    t1, t2 = t1_us * 1e3, t2_us * 1e3
    nm = NoiseModel()
    lam1 = 1.0 - (1.0 - p1) ** k
    lam2 = 1.0 - (1.0 - p2) ** k
    e1 = depolarizing_error(lam1, 1).compose(
        thermal_relaxation_error(t1, t2, tg1_ns * k))
    nm.add_all_qubit_quantum_error(e1, ["sx", "x", "y", "id"])
    r2 = thermal_relaxation_error(t1, t2, tg2_ns * k).tensor(
        thermal_relaxation_error(t1, t2, tg2_ns * k))
    e2 = depolarizing_error(lam2, 2).compose(r2)
    nm.add_all_qubit_quantum_error(e2, ["cx", "ecr"])
    if p_ro > 0:
        p01 = min(0.49, p_ro * (1.0 + ro_asym))
        p10 = min(0.49, p_ro * (1.0 - ro_asym))
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))
    return nm


# =====================================================================
# 6) SUPRESION:  Pauli twirling  +  dynamical decoupling
# =====================================================================
_TWIRL_TABLE = None


def _build_twirl_table():
    """
    Tabla de twirling de Pauli para CX:  (a,b) -> (c,d) tal que
        (C (x) D) CX (A (x) B) = CX     salvo fase global.
    Se construye por fuerza bruta con Operator (16 combinaciones).
    Referencia: IBM Quantum, "Write a custom transpiler pass" (PauliTwirl).
    """
    global _TWIRL_TABLE
    if _TWIRL_TABLE is not None:
        return _TWIRL_TABLE
    paulis = [IGate(), XGate(), YGate(), ZGate()]
    cx = Operator(QuantumCircuit(2, name="cx").compose(
        QuantumCircuit(2)))  # placeholder
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    cx = Operator(qc)
    table = []
    for ia, A_ in enumerate(paulis):
        for ib, B_ in enumerate(paulis):
            # CONVENCION (corregida): el indice `ia` se aplica al qubit 0
            # (control de la CX) y `ib` al qubit 1 (blanco), que es como los
            # usa `pauli_twirl_circuit`.  En Qiskit `Op(X).tensor(Op(Y))` es
            # X (x) Y y coloca Y en el qubit 0, asi que el operador con `ia`
            # en q0 es  B_ (x) A_  --  NO  A_ (x) B_.
            # BUG ANTERIOR: la tabla se construia con A_ (x) B_ y se aplicaba
            # con ia->q0, invirtiendo control y blanco: 15 de las 16 entradas
            # dejaban de ser una identidad logica y el "twirl" cambiaba el
            # estado preparado (verificado: F(psi_twirl, psi_0) ~ 0).
            pre = Operator(B_).tensor(Operator(A_))
            M = cx @ pre                      # CX (A_ en q0, B_ en q1)
            for ic, C_ in enumerate(paulis):
                for idd, D_ in enumerate(paulis):
                    post = Operator(D_).tensor(Operator(C_))
                    if (post @ M).equiv(cx):
                        table.append((ia, ib, ic, idd))
                        break
                else:
                    continue
                break
    _TWIRL_TABLE = table
    return table


_PAULI_GATES = {0: "id", 1: "x", 2: "y", 3: "z"}


def pauli_twirl_circuit(qc, rng):
    """
    Twirling de Pauli sobre las CX: cada CX se rodea de Paulis aleatorios
    (A(x)B) ... (C(x)D) elegidos de la tabla de 16 combinaciones que dejan la
    CX invariante salvo fase global.  Convierte ruido coherente (que se acumula
    ~cuadraticamente) en ruido de Pauli (que se acumula ~linealmente), y es
    prerequisito de PEC.  Referencia: IBM Quantum, "Write a custom transpiler
    pass" (PauliTwirl); Wallman & Emerson, PRA 94, 052325 (2016).

    Se reconstruye el circuito directamente (no via DAG) porque
    substitute_node_with_dag es inestable en Qiskit 2.5 con circuitos con layout.
    """
    table = _build_twirl_table()
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    if qc.layout is not None:
        out._layout = qc.layout
    for inst in qc.data:
        if inst.operation.name != "cx" or len(inst.qubits) != 2:
            out.append(inst)
            continue
        ia, ib, ic, idd = table[int(rng.integers(len(table)))]
        q0, q1 = inst.qubits
        for g, q in ((ia, q0), (ib, q1)):
            if g:
                getattr(out, _PAULI_GATES[g])(q)
        out.append(inst)
        for g, q in ((ic, q0), (idd, q1)):
            if g:
                getattr(out, _PAULI_GATES[g])(q)
    return out


def apply_dd(qc_t, tg1_ns=35.0, tg2_ns=400.0, readout_ns=1000.0,
             sequence="XY4"):
    """
    Desacoplamiento dinamico sobre los qubits ociosos.  Con el entrelazador
    brickwork la mitad de los qubits queda ociosa en cada media capa, asi que
    DD SI tiene ventanas donde actuar (con la cadena lineal casi no las hay).
    """
    durs = InstructionDurations(
        [("sx", None, int(tg1_ns)), ("x", None, int(tg1_ns)),
         ("y", None, int(tg1_ns)), ("z", None, 0), ("id", None, int(tg1_ns)),
         ("rz", None, 0), ("cx", None, int(tg2_ns)),
         ("ecr", None, int(tg2_ns)), ("measure", None, int(readout_ns)),
         ("reset", None, int(readout_ns)), ("barrier", None, 0)], dt=1e-9)
    seq = ([XGate(), XGate()] if sequence == "XX"
           else [XGate(), YGate(), XGate(), YGate()])
    pm = PassManager([ALAPScheduleAnalysis(durations=durs),
                      PadDynamicalDecoupling(durations=durs, dd_sequence=seq)])
    try:
        return pm.run(qc_t)
    except Exception as exc:                        # pragma: no cover
        print(f"    [DD] omitido ({exc.__class__.__name__}: {exc})")
        return qc_t


# =====================================================================
# 7) EVALUADOR + PILA DE MITIGACION
# =====================================================================
def fold_cx(qc_t, scale):
    """ZNE por gate folding local:  CX -> CX (CX' CX)^k,  scale = 2k+1."""
    assert scale % 2 == 1 and scale >= 1
    k = (scale - 1) // 2
    if k == 0:
        return qc_t
    out = QuantumCircuit(*qc_t.qregs, *qc_t.cregs)
    for inst in qc_t.data:
        out.append(inst)
        if inst.operation.name in ("cx", "ecr", "cz"):
            for _ in range(k):
                out.barrier(inst.qubits)
                out.append(inst)
                out.barrier(inst.qubits)
                out.append(inst)
    return out


def _support_mask(label):
    """Bitmask de los qubits en el soporte de una cadena de Pauli."""
    m = 0
    for i, ch in enumerate(reversed(label)):
        if ch != "I":
            m |= (1 << i)
    return m
class MitiqEvaluator:
    def __init__(self, obs_list, ansatz, noise_model=None, shots=4096,
                 seed=0, basis_gates=("rz","sx","x","y","cx"),
                 estimator_options=None):
        ...
        self.estimator_options = estimator_options or {}
        # leer opciones
        tw = self.estimator_options.get("twirling", {})
        self._trex_enable_measure = bool(tw.get("enable_measure", False))
        self._trex_num_random = int(tw.get("num_randomizations", 32))
        # gate twirl
        self._gate_twirl_enable = bool(self.estimator_options.get("twirling_gates", {}).get("enable_gates", False))
        self._gate_twirl_num = int(self.estimator_options.get("twirling_gates", {}).get("num_randomizations", 1))
        # ZNE
        res = self.estimator_options.get("resilience", {})
        self._zne_enable = bool(res.get("zne_mitigation", False))
        self._zne_scales = list(res.get("zne", {}).get("noise_factors", [1.0,3.0,5.0]))
        self._zne_extrap = res.get("zne", {}).get("extrapolator", "richardson")
        # DD
        dd = self.estimator_options.get("dynamical_decoupling", {})
        self._dd_enable = bool(dd.get("enable", False))
        self._dd_seq = dd.get("sequence_type", "XY4")
        # shots default
        self.shots = int(self.estimator_options.get("default_shots", self.shots))



class Evaluator:
    """
    Evaluador con toda la pila de mitigacion. Cada tecnica es independiente y
    se activa por bandera, para poder medir su contribucion por separado (que
    es exactamente lo que pide Ethan: probarlas TODAS y estar seguros).

    Estimacion por MUESTREO FISICO (PauliEstimator), no por snapshot analitico.
    """

    def __init__(self, nq, depth, obs, combine, mats, args, seed=0):
        self.nq, self.depth, self.combine, self.mats = nq, depth, combine, mats
        self.args = args
        self.rng = np.random.default_rng(seed)
        self.basis = ["rz", "sx", "x", "y", "cx"]
        self.noise_model_name = getattr(args, "noise_model", "uniform")
        if self.noise_model_name not in ("uniform", "fake-athens"):
            raise ValueError(f"noise_model desconocido: {self.noise_model_name}")

        # FakeAthensV2 es un snapshot de cinco qubits.  No se mezclan DD ni
        # gate twirling en este baseline: ambos insertan puertas adicionales
        # y ya se comprobó que no forman parte de una comparación limpia.
        self.device_backend = None
        if self.noise_model_name == "fake-athens":
            if nq > 5:
                raise ValueError(
                    f"FakeAthensV2 tiene 5 qubits; el problema requiere nq={nq}.")
            if getattr(args, "dd", False) or getattr(args, "twirl", False):
                raise ValueError(
                    "fake-athens baseline requiere --dd y --twirl desactivados; "
                    "usa solo raw o TREX en la primera campaña.")
            self.device_backend = _load_fake_athens_v2()
            self.cmap = None
        else:
            self.cmap = CouplingMap.from_line(nq) if nq > 1 else None
        self.has_readout_noise = (self.noise_model_name == "fake-athens" or
                                  float(getattr(args, "p_ro", 0.0)) > 0.0)

        qc, self.pv = build_ansatz(nq, depth, args.entangler)
        self.qc_log = qc
        if self.device_backend is not None:
            # El layout y las puertas nativas proceden del snapshot, no de la
            # cadena uniforme usada por el modo de sensibilidad controlada.
            qt = transpile(qc, backend=self.device_backend,
                           optimization_level=3, seed_transpiler=7)
        else:
            qt = transpile(qc, coupling_map=self.cmap, basis_gates=self.basis,
                           optimization_level=3, seed_transpiler=7)
        self.qc_bare = qt
        self.qc_t = apply_dd(qt, sequence=args.dd_sequence) if args.dd else qt

        self.obs = [o.apply_layout(qt.layout) if qt.layout else o for o in obs]
        self.c_fix = np.array([identity_coeff(o, qt.num_qubits)
                               for o in self.obs])
        self.pe = PauliEstimator(self.obs, qt.num_qubits)
        self.n_bases = self.pe.n_bases
        self.n_cx = qt.count_ops().get("cx", 0)
        self.dep2q = two_qubit_depth(qt)
        self.depth_total = self.qc_t.depth()

        self.trex_gamma = None
        self.cdr_coef = None
        self.f_hat = 1.0
        self.n_calls = 0
        self.total_shots = 0
        self._sim = {}
        # Cache de capas de medicion transpiladas, keyed by (basis, flip).
        self._meas_cache = {}

    # ---------------- backend ------------------------------------
    def _backend(self, k=1.0):
        key = round(float(k), 6)
        if key not in self._sim:
            a = self.args
            if self.device_backend is not None:
                if not np.isclose(k, 1.0):
                    raise ValueError(
                        "FakeAthensV2 no admite el escalado analitico del "
                        "modelo uniforme. Para ZNE use folding de Mitiq.")
                self._sim[key] = AerSimulator.from_backend(
                    self.device_backend,
                    max_parallel_experiments=npar,
                    max_parallel_threads=npar,
                )
            else:
                nm = (None if a.noiseless else
                      make_noise(a.p1, a.p2, p_ro=a.p_ro, ro_asym=a.ro_asym,
                                 noise_factor=k))
                self._sim[key] = AerSimulator(
                    noise_model=nm,
                    basis_gates=self.basis,
                    max_parallel_experiments=npar,
                    max_parallel_threads=npar,
                )
        return self._sim[key]

    def _compile_execution_circuits(self, circuits):
        """Compila ansatz *y medición* al target de FakeAthens.

        PauliEstimator añade H/Sdg/X después de compilar el ansatz. Dejarlas
        sin transpilar haría que sus puertas no fueran las nativas del
        snapshot y que Aer no aplicara el modelo de ruido correcto.
        """
        if self.device_backend is None:
            return circuits
        return transpile(circuits, backend=self.device_backend,
                         optimization_level=0, seed_transpiler=7)

    def _bound(self, theta, twirl):
        qc = self.qc_t.assign_parameters(np.asarray(theta, float))
        return pauli_twirl_circuit(qc, self.rng) if twirl else qc

    # ---------------- evaluacion cruda ----------------------------
    def _get_meas_transpiled(self, basis, flip):
        """Devuelve la capa de medicion transpilada, con cache por (basis,flip)."""
        key = (tuple(basis), int(flip))
        cached = self._meas_cache.get(key)
        if cached is not None:
            return cached
        mc = self.pe._meas_circuit(list(basis), flip=flip)
        if self.device_backend is not None:
            mc = transpile(mc, backend=self.device_backend,
                           optimization_level=0, seed_transpiler=7)
        self._meas_cache[key] = mc
        return mc

    def _build_cached_circuits(self, qc_bound, flip):
        """Compone el ansatz bound con las capas de medicion cacheadas."""
        out = []
        for gl in self.pe.groups:
            for basis, _ in gl:
                mc = self._get_meas_transpiled(basis, flip)
                out.append(qc_bound.compose(mc, front=False))
        return out

    def raw_evs(self, theta, shots=None, k=1.0, twirl=None, per_support=False,
                noiseless=False):
        sh = int(shots or self.args.shots)
        tw = (self.args.twirl if twirl is None else twirl)
        nrand = max(1, self.args.twirl_randomizations) if tw else 1
        sh_r = max(sh // nrand, 32)
        sim = (AerSimulator(basis_gates=self.basis) if noiseless
               else self._backend(k))
        # --- twirl de medicion (TREX) -------------------------------------
        # Se activa con `--meas-twirl` y solo tiene sentido con error de
        # lectura.  Cada randomizacion usa una mascara de volteo distinta; el
        # signo se corrige clasicamente en PauliEstimator.  El promedio sobre
        # mascaras hace que el canal de lectura sea diagonal en la base de
        # Pauli-Z, que es la HIPOTESIS que TREX necesita para que
        # <Z^m> -> gamma_m <Z^m> sea exacto (van den Berg et al. 2022).
        mt = (getattr(self.args, "meas_twirl", False)
              and not noiseless and self.has_readout_noise)
        if mt and nrand == 1:
            nrand = max(2, int(getattr(self.args, "meas_twirl_rand", 8)))
            sh_r = max(sh // nrand, 32)
        acc = None
        for _ in range(nrand):
            flip = int(self.rng.integers(1 << self.pe.nq)) if mt else 0
            qc_bound = self._bound(theta, tw)
            circs = self._build_cached_circuits(qc_bound, flip)
            res = sim.run(circs, shots=sh_r,
                          seed_simulator=int(self.rng.integers(2 ** 31))
                          ).result()
            if per_support:
                cur, _ = self.pe.evs_per_support(res, flip=flip)
                acc = cur if acc is None else {kk: acc[kk] + cur[kk]
                                               for kk in acc}
            else:
                cur, _ = self.pe.evs_from_results(res, flip=flip)
                acc = cur if acc is None else acc + cur
            self.n_calls += 1
            self.total_shots += sh_r * self.n_bases
        if per_support:
            return {kk: vv / nrand for kk, vv in acc.items()}
        return acc / nrand

    def exact_evs(self, theta):
        """Exact expectations of the full dense diagnostic matrices.

        This is deliberately distinct from ``exact_evs_observables``: the
        latter is the zero-shot expectation of the Pauli-truncated observable
        that the estimator actually measures.
        """
        psi = np.real(Statevector(
            self.qc_log.assign_parameters(np.asarray(theta, float))).data)
        return np.array([psi @ M @ psi for M in self.mats])

    def exact_evs_observables(self, theta):
        """Zero-shot expectation of the Pauli observables seen by raw/TREX/ZNE.

        Keeping this separate from dense ``mats`` exposes the deterministic
        bias introduced by ``pauli_atol`` before any shot or channel noise.
        """
        psi = np.real(Statevector(
            self.qc_log.assign_parameters(np.asarray(theta, float))).data)
        return np.asarray([
            float(np.real(psi @ (spo.to_matrix() @ psi)))
            for spo in self.obs
        ])

    def statevector(self, theta):
        return np.real(Statevector(
            self.qc_log.assign_parameters(np.asarray(theta, float))).data)

    # ---------------- TREX ----------------------------------------
    def calibrate_trex(self, n_cal=24):
        """
        Aprende la atenuacion gamma_m del canal de lectura por SOPORTE m,
        preparando estados de base computacional ALEATORIOS y midiendo Z^(m).
        Promediar sobre entradas aleatorias es equivalente al twirling de
        medicion de TREX: diagonaliza la matriz de transferencia de lectura.
        Ref.: van den Berg, Minev, Temme, PRA 105, 032620 (2022).
        """
        if not self.has_readout_noise or self.args.noiseless:
            self.trex_gamma = None
            return None
        masks = sorted({m for (_i, m) in
                        self.raw_evs(np.zeros(self.qc_log.num_parameters),
                                     shots=32, per_support=True)} - {0})
        acc = {m: [] for m in masks}
        sim = self._backend(1.0)
        for _ in range(n_cal):
            z = self.rng.integers(0, 2, size=self.nq)
            qc = QuantumCircuit(self.nq)
            for j, zz in enumerate(z):
                if zz:
                    qc.x(j)
            qc.measure_all()
            if self.device_backend is not None:
                qt = transpile(qc, backend=self.device_backend,
                               optimization_level=1, seed_transpiler=7)
            else:
                qt = transpile(qc, basis_gates=self.basis,
                               optimization_level=1)
            res = sim.run(qt, shots=self.args.shots,
                          seed_simulator=int(self.rng.integers(2 ** 31))
                          ).result()
            counts = res.get_counts()
            tot = sum(counts.values())
            keys = np.array([int(kk.replace(" ", ""), 2) for kk in counts])
            vals = np.array(list(counts.values()), float)
            for m in masks:
                par = np.array([bin(int(kk) & m).count("1") & 1 for kk in keys])
                meas = float(np.sum(vals * (1 - 2 * par)) / tot)
                ideal = (-1.0) ** int(sum(z[j] for j in range(self.nq)
                                          if (m >> j) & 1))
                acc[m].append(meas / ideal)
        self.trex_gamma = {m: float(np.clip(np.mean(a), 0.05, 1.0))
                           for m, a in acc.items()}
        return self.trex_gamma

    def trex_evs(self, theta, shots=None, k=1.0):
        if self.trex_gamma is None:
            return self.raw_evs(theta, shots, k)
        g = self.raw_evs(theta, shots, k, per_support=True)
        out = np.zeros(len(self.obs))
        for (i, m), v in g.items():
            out[i] += v / (self.trex_gamma.get(m, 1.0) if m else 1.0)
        return out

    # ---------------- ZNE ------------------------------------------
    def zne_evs(self, theta, shots=None):
        """
        ZNE sobre los observables del costo.
          amplifier="fold"  : gate folding digital (lo que hace IBM Runtime).
          amplifier="model" : recompone el modelo exacto  D_lam^k  y  t->k*t.
                              Sin los artefactos del folding (que tambien
                              duplica compuertas de 1q y no escala T1/T2 igual).
                              Es el analogo local de PEA: se amplifica el ruido
                              REAL aprendido/conocido, no una aproximacion.

        Ruta Mitiq (args.mitiq=True) -- decisiones documentadas:
          * folding global vs local: `--zne-folding` selecciona entre
            `mitiq.zne.scaling.fold_global` (pliega la unitaria completa del
            circuito, U -> U(U'U)^k; mas uniforme para el ansatz superficial
            de este benchmark) y `fold_all` (pliega CADA compuerta
            individualmente, equivalente local/determinista -- analogo de
            Mitiq al `fold_cx` propio del repo). Mitiq >=1.0 NO expone
            `fold_local`; el nombre historico mas cercano es `fold_all`.
          * `--zne-fit` selecciona la extrapolacion:
              richardson -> `mitiq.zne.inference.RichardsonFactory`
              linear     -> `mitiq.zne.inference.LinearFactory` si esta
                            disponible en la version de Mitiq instalada;
                            si no, fallback a la implementacion propia
                            (`_extrapolate(..., "linear", ...)`) sobre los
                            circuitos escalados por el folding de Mitiq.
              fixedpoint -> Mitiq no tiene una factory equivalente al
                            extrapolador de ASINTOTA FIJA en c_fix =
                            Tr(O)/2^n (punto fijo del canal depolarizante
                            global) que ya tiene el repo, asi que se usa el
                            folding de Mitiq para generar los circuitos
                            escalados y `_extrapolate(..., "exp_fixed", ...)`
                            del repo para la extrapolacion.
          * semilla: `seed_simulator` sale de `self.rng`, el generador propio
            del `Evaluator` (fijado una vez en `__init__` con la seed del
            experimento) -> reproducible dentro de una corrida, distinto
            entre corridas con distinta seed.
          * paralelizacion: ninguna. `AerSimulator.run` ya agrupa en una sola
            llamada todos los circuitos de base de medicion de un factor de
            escala; a los tamanos de qubit de este benchmark (<=6 qubits) el
            overhead de multiprocessing no compensa, y mantiene el codigo
            determinista/facil de depurar.
        """
        a = self.args
        if self.device_backend is not None and getattr(a, "zne_amplifier", "fold") == "model":
            raise ValueError(
                "fake-athens no permite zne_amplifier='model': ese escalado "
                "solo existe para make_noise(). Use folding de Mitiq.")
        if getattr(a, 'mitiq', False):
            check_mitiq_version(mitiq if _HAVE_MITIQ else None)
            bound = self._bound(theta, False)
            sh = int(shots or a.shots)
            scales = list(map(float, a.zne_scales))
            fold_fn = fold_global if getattr(a, 'zne_folding', 'global') == 'global' else fold_local
            fit = getattr(a, 'zne_fit', 'richardson')
            fit = 'fixedpoint' if fit == 'exp_fixed' else fit

            sh = int(shots or a.shots)
            n_rand = max(1, int(getattr(a, 'twirl_randomizations', 1))) if a.twirl else 1

            def _run_zne_on_circuit(circ, idx):
                # executor that returns the expectation for observable idx
                def exec_fn(c):
                    sim = self._backend(1.0) if not a.noiseless else AerSimulator(basis_gates=self.basis)
                    circs = self.pe.build(c)
                    circs = self._compile_execution_circuits(circs)
                    res = sim.run(circs, shots=max(32, sh),
                                  seed_simulator=int(self.rng.integers(2 ** 31))).result()
                    cur, _ = self.pe.evs_from_results(res)
                    return float(cur[idx])

                if fit == 'fixedpoint':
                    ys = np.array([exec_fn(fold_fn(circ, s)) for s in scales])
                    return _extrapolate(np.array(scales), ys, 'exp_fixed', self.c_fix[idx])

                if fit == 'linear' and not _HAVE_LINEAR_FACTORY:
                    ys = np.array([exec_fn(fold_fn(circ, s)) for s in scales])
                    return _extrapolate(np.array(scales), ys, 'linear', self.c_fix[idx])

                if fit == 'richardson':
                    factory_cls = RichardsonFactory
                elif fit == 'linear':
                    factory_cls = LinearFactory
                else:
                    raise ValueError(f"zne_fit desconocido: {fit}")

                factory = factory_cls(scale_factors=scales)
                # mitiq.execute_with_zne accepts a unitary (QuantumCircuit) and an executor
                val = execute_with_zne(circ, lambda c: exec_fn(c), factory=factory, scale_noise=fold_fn)
                return float(val)

            out = np.zeros(len(self.obs))
            # If Pauli Twirling + Randomized Error Mitigation selected, average over twirls
            for i in range(len(self.obs)):
                vals = []
                for rr in range(n_rand):
                    # build bounded circuit (with DD already in self.qc_t) and optionally twirl
                    bound_circ = self._bound(theta, False)
                    if a.twirl:
                        bound_circ = pauli_twirl_circuit(bound_circ, self.rng)
                    v = _run_zne_on_circuit(bound_circ, i)
                    vals.append(v)
                out[i] = float(np.mean(vals))
            return out

        # Fall back: original implementation without Mitiq
        ys = []
        for k in a.zne_scales:
            if a.zne_amplifier == "fold":
                keep = self.qc_t
                self.qc_t = fold_cx(keep, int(k))
                pe_keep, self.pe = self.pe, PauliEstimator(
                    self.obs, self.qc_t.num_qubits)
                ys.append(self.trex_evs(theta, shots, k=1.0))
                self.qc_t, self.pe = keep, pe_keep
            else:
                ys.append(self.trex_evs(theta, shots, k=float(k)))
        Y = np.array(ys)
        return np.array([_extrapolate(np.array(a.zne_scales, float), Y[:, i],
                                      a.zne_fit, self.c_fix[i])
                         for i in range(Y.shape[1])])

    # ---------------- CDR + GDEM -----------------------------------
    def calibrate_cdr(self, theta_ref, n_cal=32, frac_swap=0.5):
        """
        Clifford Data Regression (Czarnik et al., Quantum 5, 592 (2021)):
        se construye un conjunto de entrenamiento de circuitos NEAR-CLIFFORD
        alrededor de theta_ref (simulables clasicamente en tiempo polinomico),
        se mide <O>_ruidoso y se ajusta  <O>_exacto = a*<O>_ruidoso + b.
        GDEM = el caso particular de un solo parametro (modelo depolarizante
        global):  <O>_mit = c + (<O>_ruidoso - c)/f,  c = Tr(O)/2^n.
        """
        P = self.qc_log.num_parameters
        th_near = CLIFF[np.argmin(np.abs(
            (np.asarray(theta_ref)[:, None] - CLIFF[None, :] + np.pi)
            % (2 * np.pi) - np.pi), axis=1)]
        Yn, Ye = [], []
        for _ in range(n_cal):
            t = th_near.copy()
            ns = max(1, int(frac_swap * P * self.rng.random()))
            idx = self.rng.choice(P, size=ns, replace=False)
            t[idx] = self.rng.choice(CLIFF, size=ns)
            Yn.append(self.trex_evs(t))
            Ye.append(self.exact_evs(t))
        Yn, Ye = np.array(Yn), np.array(Ye)
        self.cdr_coef = []
        for i in range(Yn.shape[1]):
            X = np.vstack([Yn[:, i], np.ones(len(Yn))]).T
            c_, *_ = np.linalg.lstsq(X, Ye[:, i], rcond=None)
            self.cdr_coef.append(c_)
        c = self.c_fix
        num = float(np.sum((Yn - c) * (Ye - c)))
        den = float(np.sum((Ye - c) ** 2))
        self.f_hat = num / den if abs(den) > 1e-14 else 1.0
        pred = np.stack([np.vstack([Yn[:, i], np.ones(len(Yn))]).T
                         @ self.cdr_coef[i] for i in range(Yn.shape[1])], 1)
        self.cdr_rms = float(np.sqrt(np.mean((pred - Ye) ** 2))
                             / (float(np.std(Ye)) + 1e-12))
        return self.f_hat, self.cdr_rms

    def cdr_evs(self, theta, shots=None):
        y = self.trex_evs(theta, shots)
        if self.cdr_coef is None:
            return y
        return np.array([self.cdr_coef[i][0] * y[i] + self.cdr_coef[i][1]
                         for i in range(len(y))])

    def gdem_evs(self, theta, shots=None):
        y = self.trex_evs(theta, shots)
        f = self.f_hat if abs(self.f_hat) > 1e-6 else 1.0
        return self.c_fix + (y - self.c_fix) / f

    # ---------------- PEC ------------------------------------------
    def pec_overhead(self):
        """
        gamma de muestreo de PEC para el canal depolarizante de 2 qubits.
        D_lam^-1 = D_mu con mu = -lam/(1-lam).  Cuasi-probabilidades:
          q_I = (1-mu) + mu/16 ,  q_P = mu/16 (P != I)
          gamma = |q_I| + 15|q_P|
        El sobrecoste TOTAL en disparos es gamma^(2*n_cx): crece
        EXPONENCIALMENTE con la profundidad.  Ese es, medido, el teorema de
        Wang et al. (Quantum 8, 1287, 2024) y de Takagi et al. (npj QI 2022).
        """
        lam = self.args.p2
        mu = -lam / (1.0 - lam)
        qI = (1.0 - mu) + mu / 16.0
        qP = mu / 16.0
        g = abs(qI) + 15.0 * abs(qP)
        return float(g), float(g ** self.n_cx), float(g ** (2 * self.n_cx))

    def pec_evs(self, theta, shots=None, n_samples=None):
        """
        PEC por muestreo de cuasi-probabilidad: por cada CX se inserta un Pauli
        de 2 qubits muestreado de |q_P|/gamma y se acumula el signo.
        Estimador insesgado del canal depolarizante:
            <O>_PEC = gamma^n_cx * E[ signo * <O>_ruidoso ]
        NOTA HONESTA: cancela EXACTAMENTE la parte depolarizante; la relajacion
        termica es no-unital y NO es un canal de Pauli, asi que su residuo
        permanece.  Es el procedimiento estandar (twirl -> canal Pauli -> PEC).
        """
        ns = int(n_samples or self.args.pec_samples)
        lam = self.args.p2
        mu = -lam / (1.0 - lam)
        q = np.array([(1.0 - mu) + mu / 16.0] + [mu / 16.0] * 15)
        p = np.abs(q) / np.sum(np.abs(q))
        gamma = float(np.sum(np.abs(q)))
        P2 = [(a, b) for a in range(4) for b in range(4)]
        sh = max(int((shots or self.args.shots) // max(ns, 1)), 32)
        sim = self._backend(1.0)
        acc, wsum = np.zeros(len(self.obs)), 0.0
        for _ in range(ns):
            qc0 = self.qc_t.assign_parameters(np.asarray(theta, float))
            out = QuantumCircuit(*qc0.qregs, *qc0.cregs)
            if qc0.layout is not None:
                out._layout = qc0.layout
            sign = 1.0
            for inst in qc0.data:
                out.append(inst)
                if inst.operation.name == "cx" and len(inst.qubits) == 2:
                    j = int(self.rng.choice(16, p=p))
                    sign *= float(np.sign(q[j]))
                    a_, b_ = P2[j]
                    for gi, qb in ((a_, inst.qubits[0]), (b_, inst.qubits[1])):
                        if gi:
                            getattr(out, _PAULI_GATES[gi])(qb)
            circs = self.pe.build(out)
            res = sim.run(circs, shots=sh,
                          seed_simulator=int(self.rng.integers(2 ** 31))
                          ).result()
            v, _ = self.pe.evs_from_results(res)
            acc += sign * v
            wsum += 1.0
            self.n_calls += 1
            self.total_shots += sh * self.n_bases
        return (gamma ** self.n_cx) * acc / max(wsum, 1.0)

    # ---------------- costo segun la pila elegida -------------------
    def cost(self, theta, method=None, shots=None):
        m = method or self.args.mitigation
        if m == "none":
            v = self.raw_evs(theta, shots)
        elif m == "trex":
            v = self.trex_evs(theta, shots)
        elif m == "zne":
            v = self.zne_evs(theta, shots)
        elif m == "cdr":
            v = self.cdr_evs(theta, shots)
        elif m == "gdem":
            v = self.gdem_evs(theta, shots)
        elif m == "pec":
            v = self.pec_evs(theta, shots)
        elif m == "exact":
            v = self.exact_evs(theta)
        else:
            raise ValueError(m)
        return float(self.combine(v))


# NOTA: `_extrapolate` vive ahora en nisq_core.py (importado arriba) para
# que sea testeable sin Qiskit; se deja re-exportado como `nisq._extrapolate`
# por el import del encabezado.


# =====================================================================
# 8) ESTIMADOR CON MUESTREO FISICO REAL
# =====================================================================
# MOTIVO (verificado en Aer 0.17.2):  qiskit_aer.primitives.EstimatorV2 NO
# mide: inserta save_expectation_value (snapshot analitico sobre la matriz
# densidad) y luego suma ruido gaussiano de ancho fijo = default_precision.
# Por tanto (a) el ReadoutError NUNCA se aplica -> TREX/M3 son inmedibles, y
# (b) el "ruido de disparo" no es binomial -> el SNR reportado no es el fisico.
# Aqui se agrupa en bases qubit-wise commuting, se anaden las rotaciones de
# base y las mediciones, y se muestrea con AerSimulator.  Esto SI reproduce
# el ruido de lectura y el ruido de disparo correctos.
# =====================================================================
class PauliEstimator:
    """Estimador por muestreo: agrupa en bases QWC, mide, y reconstruye <O>."""

    def __init__(self, obs_list, nq):
        self.nq = nq
        self.groups = []            # [[(basis, [(mask, coef), ...]), ...], ...]
        self.n_bases = 0
        for spo in obs_list:
            gl = []
            for g in spo.group_commuting(qubit_wise=True):
                labels = g.paulis.to_labels()
                basis = ["I"] * nq
                for lab in labels:
                    for j, ch in enumerate(reversed(lab)):
                        if ch != "I":
                            basis[j] = ch
                terms = [(_support_mask(l), float(np.real(c)))
                         for l, c in zip(labels, g.coeffs)]
                gl.append((tuple(basis), terms))
            self.groups.append(gl)
            self.n_bases += len(gl)

    def _meas_circuit(self, basis, flip=0):
        """
        Rotacion a la base de medicion + (opcional) twirl de medicion.

        `flip` es una mascara de bits: se aplica X en cada qubit marcado JUSTO
        ANTES de medir, es decir DESPUES de la rotacion de base.  Ese es el
        twirling de medicion de TREX (van den Berg, Minev, Temme, PRA 105,
        032620 (2022)): promediar sobre `flip` aleatorios diagonaliza la
        matriz de transferencia de lectura, de modo que
            <Z^m>_ruidoso = gamma_m <Z^m>_ideal
        se cumple EXACTAMENTE (tambien con error de lectura asimetrico).  Sin
        este twirl en la fase de estimacion, dividir entre gamma_m es un
        estimador sesgado en cuanto p01 != p10.
        """
        mc = QuantumCircuit(self.nq)
        for j, ch in enumerate(basis):
            if ch == "X":
                mc.h(j)
            elif ch == "Y":
                mc.sdg(j)
                mc.h(j)
        for j in range(self.nq):
            if (flip >> j) & 1:
                mc.x(j)
        mc.measure_all()
        return mc

    def build(self, bound_qc, flip=0):
        """Circuitos listos para AerSimulator (uno por base de medicion)."""
        return [bound_qc.compose(self._meas_circuit(basis, flip), front=False)
                for gl in self.groups for basis, _ in gl]

    def evs_from_results(self, result, n_circ_offset=0, flip=0):
        out = np.zeros(len(self.groups))
        idx = n_circ_offset
        for i, gl in enumerate(self.groups):
            for _, terms in gl:
                counts = result.get_counts(idx)
                idx += 1
                tot = sum(counts.values())
                keys = np.array([int(k.replace(" ", ""), 2)
                                 for k in counts.keys()])
                vals = np.array(list(counts.values()), float)
                for mask, cf in terms:
                    if mask == 0:
                        out[i] += cf
                        continue
                    # correccion del twirl: el bit medido es x_j XOR flip_j,
                    # asi que la paridad medida sobre `mask` esta invertida
                    # cuando |flip & mask| es impar.
                    sgn = 1.0 - 2.0 * (bin(int(flip) & mask).count("1") & 1)
                    par = np.array([bin(int(k) & mask).count("1") & 1
                                    for k in keys])
                    out[i] += cf * sgn * float(
                        np.sum(vals * (1 - 2 * par)) / tot)
        return out, idx

    def evs_per_support(self, result, n_circ_offset=0, flip=0):
        """Igual, pero desglosado por mascara de soporte (para TREX)."""
        out = {}
        idx = n_circ_offset
        for i, gl in enumerate(self.groups):
            for _, terms in gl:
                counts = result.get_counts(idx)
                idx += 1
                tot = sum(counts.values())
                keys = np.array([int(k.replace(" ", ""), 2)
                                 for k in counts.keys()])
                vals = np.array(list(counts.values()), float)
                for mask, cf in terms:
                    if mask == 0:
                        out[(i, 0)] = out.get((i, 0), 0.0) + cf
                        continue
                    sgn = 1.0 - 2.0 * (bin(int(flip) & mask).count("1") & 1)
                    par = np.array([bin(int(k) & mask).count("1") & 1
                                    for k in keys])
                    v = cf * sgn * float(np.sum(vals * (1 - 2 * par)) / tot)
                    out[(i, mask)] = out.get((i, mask), 0.0) + v
        return out, idx


# =====================================================================
# 9) OPTIMIZADOR SPSA + WARM START
# =====================================================================
# NOTA: `spsa_minimize`, `cobyla_minimize`, `lbfgs_minimize` viven ahora en
# nisq_core.py (importados arriba) para que sean testeables sin Qiskit; se
# re-exportan como `nisq.spsa_minimize`, etc. via el import del encabezado.


def lbfgs_ideal(A, b, nq, depth, entangler, cost_type, seeds=(0, 1, 2, 3),
                maxiter=2000, atol=0.0):
    """
    Pre-optimizacion EXACTA por statevector (sin ruido, sin disparos).
    Sirve para (i) el warm start y (ii) medir F_max = expresividad del ansatz,
    que es la unica forma de separar 'el ansatz no puede' de 'el optimizador
    no encuentra'.
    """
    from scipy.optimize import minimize
    qc, _ = build_ansatz(nq, depth, entangler)
    _, combine, mats, _ = make_cost_observables(A, b, nq, atol=atol,
                                                cost_type=cost_type)

    def f(th):
        psi = np.real(Statevector(qc.assign_parameters(th)).data)
        return float(combine(np.array([psi @ M @ psi for M in mats])))

    best = None
    P = (depth + 1) * nq
    for s in seeds:
        rng = np.random.default_rng(s)
        x0 = rng.uniform(-np.pi, np.pi, P)
        r = minimize(f, x0, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    psi = np.real(Statevector(qc.assign_parameters(best.x)).data)
    return best.x, float(best.fun), psi


def warm_start(theta_prev, nq, depth_prev, depth_new, rng, jitter=0.02):
    """
    Warm start encadenado en profundidad: se copian las capas ya optimizadas
    y las nuevas se inicializan cerca de 0 (identidad) + jitter pequeno.
    Inicializar las capas nuevas en ~0 hace que el circuito nuevo sea
    inicialmente EQUIVALENTE al viejo -> el costo no empeora al anadir
    profundidad, y se evita caer de golpe en la region de meseta.
    (Grant et al., Quantum 3, 214 (2019): identity block initialization.)
    """
    P_new = (depth_new + 1) * nq
    x = rng.normal(0.0, jitter, P_new)
    if theta_prev is None:
        return rng.uniform(-np.pi, np.pi, P_new)
    n_copy = min(len(theta_prev) - nq, depth_prev * nq)
    x[:n_copy] = theta_prev[:n_copy]
    x[-nq:] = theta_prev[-nq:]                 # capa RY final
    return x


# =====================================================================
# 10) DIAGNOSTICO NIBP:  el experimento que zanja la discusion
# =====================================================================
def nibp_diagnostic(ev, methods, n_theta=12, n_rep=6, seed=0, spsa_c=0.15):
    """
    Para cada tecnica de mitigacion mide, sobre theta aleatorios:

      |dC|      = |C(th+c*d) - C(th-c*d)| / (2c)   (senal de gradiente SPSA)
      sigma     = desviacion estandar del ESTIMADOR de C a theta fijo
      SNR       = mean|dC| / sigma
      sesgo     = mean( C_mitigado - C_exacto )

    LECTURA:
      * Si la mitigacion sube el SNR por encima de 1  -> Ethan tiene razon:
        el problema era sesgo y la mitigacion lo arregla.
      * Si la mitigacion baja el sesgo pero el SNR NO sube (o baja), estamos
        en el teorema de Wang, Czarnik, Arrasmith, Cerezo, Cincio & Coles,
        "Can Error Mitigation Improve Trainability of Noisy VQAs?",
        Quantum 8, 1287 (2024): la concentracion exponencial del costo no se
        puede resolver sin gastar recursos exponenciales en otro sitio, y eso
        vale para ZNE, PEC, CDR y destilacion virtual POR IGUAL.
      Este es exactamente el numero que hay que ensenarle a Ethan.
    """
    rng = np.random.default_rng(seed)
    P = ev.qc_log.num_parameters
    out = {}
    for m in methods:
        dC, sig, bias = [], [], []
        for _ in range(n_theta):
            th = rng.uniform(-np.pi, np.pi, P)
            d = rng.choice([-1.0, 1.0], size=P)
            cp = ev.cost(th + spsa_c * d, method=m)
            cm = ev.cost(th - spsa_c * d, method=m)
            dC.append(abs(cp - cm) / (2 * spsa_c))
            reps = [ev.cost(th, method=m) for _ in range(n_rep)]
            sig.append(float(np.std(reps, ddof=1)))
            bias.append(float(np.mean(reps)) -
                        float(ev.combine(ev.exact_evs(th))))
        s = float(np.mean(sig)) + 1e-15
        out[m] = dict(grad_mean=float(np.mean(dC)),
                      grad_var=float(np.var(dC)),
                      sigma=s,
                      snr=float(np.mean(dC)) / s,
                      bias=float(np.mean(bias)),
                      abs_bias=float(np.mean(np.abs(bias))))
    return out


# =====================================================================
# 11) MAIN
# =====================================================================
def run(args):
    import pandas as pd
    rows, out_csv = [], f"{args.out_prefix}.csv"
    if args.fresh and os.path.exists(out_csv):
        os.remove(out_csv)

    print("=" * 96)
    print(f"vqls_nisq | familia={args.family} | precond={args.precond} | "
          f"costo={args.cost_type} | entangler={args.entangler}")
    if getattr(args, "noise_model", "uniform") == "fake-athens":
        print(f"ruido=fake-athens (snapshot IBM calibrado) | "
              f"shots={args.shots} | estudio={args.study}")
    else:
        print(f"ruido=uniform p1={args.p1} p2={args.p2} p_ro={args.p_ro} | "
              f"shots={args.shots} | estudio={args.study}")
    print(f"mitigacion: DD={args.dd} twirl={args.twirl} "
          f"ZNE({args.zne_amplifier},{args.zne_fit},{args.zne_scales})")
    print("=" * 96)

    for n_dof in args.dofs:
        try:
            sysd = prepare(args.family, n_dof, precond=args.precond,
                           cheby_degree=args.cheby_degree,
                           datadir=args.datadir)
        except Exception as exc:
            print(f"[SKIP] {args.family} {n_dof}DOF: {exc}")
            continue
        A, b, nq = sysd["A"], sysd["b"], sysd["nq"]
        print(f"\n--- {args.family} {n_dof} GDL -> {nq} qubits | "
              f"kappa_cruda={sysd['kappa_raw']:.3e} -> "
              f"kappa_pre={sysd['kappa_pre']:.3e} "
              f"({'SPD' if sysd['spd_ok'] else 'NO SPD!'}) "
              f"| fuente={sysd['meta'].get('src')}")

        theta_prev, depth_prev = None, 0
        rng = np.random.default_rng(args.seeds[0])
        for depth in args.depths:
            t0 = time.time()
            obs, combine, mats, cnt = make_cost_observables(
                A, b, nq, atol=args.pauli_atol, cost_type=args.cost_type)

            # --- techo de expresividad (statevector exacto) --------------
            th_id, c_id, psi_id = lbfgs_ideal(
                A, b, nq, depth, args.entangler, args.cost_type,
                seeds=tuple(args.seeds), maxiter=args.maxiter_ideal,
                atol=args.pauli_atol)
            F_max = fidelity(psi_id, sysd["x_pre"])
            res_id = norm_residual(psi_id, A, b)

            ev = Evaluator(nq, depth, obs, combine, mats, args,
                           seed=int(rng.integers(2 ** 31)))
            g_pec, ov_pec, sh_pec = ev.pec_overhead()
            print(f"  d={depth}: CX={ev.n_cx} dep2q={ev.dep2q} "
                  f"depth={ev.depth_total} bases={ev.n_bases} "
                  f"Pauli={cnt[0][1]}+{cnt[1][1]} | F_max={F_max:.4f} "
                  f"C_ideal={c_id:.2e} | PEC gamma^ncx={ov_pec:.2f}")

            if args.p_ro > 0 and not args.noiseless:
                ev.calibrate_trex(n_cal=args.n_cal_trex)
            if any(m in args.methods for m in ("cdr", "gdem")):
                f_hat, rms = ev.calibrate_cdr(th_id, n_cal=args.n_cal_cdr)
                print(f"        CDR: f_hat={f_hat:.4f} rms_ajuste={rms:.4f}")
            else:
                f_hat, rms = np.nan, np.nan

            base = dict(family=args.family, n_dof=n_dof, nq=nq, depth=depth,
                        entangler=args.entangler, precond=args.precond,
                        cheby_degree=args.cheby_degree,
                        cost_type=args.cost_type,
                        kappa_raw=sysd["kappa_raw"],
                        kappa_pre=sysd["kappa_pre"], spd=sysd["spd_ok"],
                        n_cx=ev.n_cx, depth_2q=ev.dep2q,
                        depth_total=ev.depth_total, n_bases=ev.n_bases,
                        n_pauli=cnt[0][1] + cnt[1][1],
                        p1=args.p1, p2=args.p2, p_ro=args.p_ro,
                        shots=args.shots, F_max=F_max, cost_ideal=c_id,
                        residual_ideal=res_id, f_hat=f_hat, cdr_rms=rms,
                        pec_gamma=g_pec, pec_overhead_shots=sh_pec)

            # ---------- ESTUDIO 1: diagnostico NIBP ---------------------
            if args.study in ("nibp", "both"):
                diag = nibp_diagnostic(ev, args.methods,
                                       n_theta=args.n_theta, n_rep=args.n_rep,
                                       seed=int(rng.integers(2 ** 31)))
                print(f"        {'metodo':8s} {'|dC|':>10s} {'sigma':>10s} "
                      f"{'SNR':>8s} {'sesgo':>10s}")
                for m, d in diag.items():
                    flag = "  <-- entrenable" if d["snr"] > 1.0 else ""
                    print(f"        {m:8s} {d['grad_mean']:10.2e} "
                          f"{d['sigma']:10.2e} {d['snr']:8.2f} "
                          f"{d['bias']:+10.2e}{flag}")
                    rows.append(dict(base, study="nibp", method=m, **d))

            # ---------- ESTUDIO 2: resolver de verdad -------------------
            if args.study in ("solve", "both"):
                for m in args.methods:
                    x0 = warm_start(theta_prev, nq, depth_prev, depth, rng)
                    if args.warm_ideal:
                        x0 = th_id.copy()
                    if args.optimizer == "spsa":
                        r = spsa_minimize(lambda t: ev.cost(t, method=m), x0,
                                          maxiter=args.maxiter,
                                          seed=int(rng.integers(2 ** 31)))
                    elif args.optimizer == "cobyla":
                        r = cobyla_minimize(lambda t: ev.cost(t, method=m), x0,
                                             maxiter=args.maxiter)
                    elif args.optimizer == "lbfgs":
                        r = lbfgs_minimize(lambda t: ev.cost(t, method=m), x0,
                                           maxiter=args.maxiter)
                    else:
                        raise ValueError(f"optimizer desconocido: {args.optimizer}")
                    psi = ev.statevector(r["x"])
                    F = fidelity(psi, sysd["x_pre"])
                    res = norm_residual(psi, A, b)
                    # precondicionado: reconstruir u_q en espacio precondicionado
                    p = psi / np.linalg.norm(psi)
                    Ap = A @ p
                    den = float(Ap @ Ap)
                    alpha = float(b @ Ap) / den if den > 1e-300 else 0.0
                    u_q_pre = alpha * p
                    l2_pre = float(np.linalg.norm(u_q_pre - sysd["x_pre"]) /
                                   max(np.linalg.norm(sysd["x_pre"]), 1e-300))
                    u_q = to_physical(psi, sysd)
                    u_c = sysd["x_phys"][:sysd["n_orig"]]
                    l2 = float(np.linalg.norm(u_q - u_c) /
                               max(np.linalg.norm(u_c), 1e-300))
                    # Save solution vectors for plotting/analysis
                    ts = int(time.time() * 1000) % 1000000000
                    sol_name = (f"sol_{args.family}_{n_dof}_d{depth}_" \
                                f"{m}_{args.optimizer}_{ts}.npz")
                    np.savez(sol_name, u_q=u_q, u_c=u_c, u_q_pre=u_q_pre,
                             psi=psi, A_dot_p=float(b @ (A @ (psi / np.linalg.norm(psi)))))
                    print(f"        [{m:5s}] F={F:.4f} (techo {F_max:.4f}) "
                          f"res={res:.4f} L2_fis={l2:.4f} L2_pre={l2_pre:.4f} "
                          f"C_fin={r['fun']:.3e}")
                    rows.append(dict(base, study="solve", method=m,
                                     F_noisy=F, degradation=F_max - F,
                                     residual_noisy=res, l2_physical=l2,
                                     l2_preconditioned=l2_pre,
                                     cost_final=r["fun"],
                                     cost_history=json.dumps(r.get("history", [] ).tolist() if hasattr(r.get("history", []), "tolist") else list(r.get("history", []))),
                                     optimizer=args.optimizer,
                                     n_calls=ev.n_calls,
                                     total_shots=ev.total_shots))
            theta_prev, depth_prev = th_id, depth
            print(f"        ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv),
              index=False)
    print(f"\n>> {len(rows)} filas -> {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="VQLS NISQ: precond + costo local + mitigacion completa")
    ap.add_argument("--family", default="beam1d",
                    choices=["beam1d", "beam2d", "heat1d", "heat2d"])
    ap.add_argument("--dofs", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--datadir", default=".")
    # --- sistema / precondicionador ---
    ap.add_argument("--precond", default="amg",
                    choices=["none", "jacobi", "ichol", "cholesky", "amg",
                             "cheby"])
    ap.add_argument("--cheby-degree", type=int, default=60)
    ap.add_argument("--cost-type", default="energy",
                    choices=["local", "global", "energy"])
    ap.add_argument("--pauli-atol", type=float, default=1e-3)
    # --- ansatz ---
    ap.add_argument("--entangler", default="linear", choices=["brick", "linear"])
    # --- ruido ---
    ap.add_argument("--p1", type=float, default=5e-4)
    ap.add_argument("--p2", type=float, default=3.5e-3)
    ap.add_argument("--p-ro", type=float, default=1.5e-2)
    ap.add_argument("--ro-asym", type=float, default=0.0)
    ap.add_argument("--noise-model", default="uniform",
                    choices=["uniform", "fake-athens"],
                    help=("uniform: estudio controlado; fake-athens: "
                          "snapshot IBM de cinco qubits simulado con Aer"))
    ap.add_argument("--noiseless", action="store_true")
    # --- mitigacion ---
    ap.add_argument("--methods", nargs="+",
                    default=["none", "trex", "zne", "cdr", "gdem", "pec"])
    ap.add_argument("--mitigation", default="none")
    ap.add_argument("--dd", action="store_true")
    ap.add_argument("--dd-sequence", default="XY4", choices=["XX", "XY4"])
    ap.add_argument("--twirl", action="store_true")
    ap.add_argument("--twirl-randomizations", type=int, default=8)
    ap.add_argument("--meas-twirl", action="store_true",
                    help="twirling de medicion (requisito de TREX)")
    ap.add_argument("--meas-twirl-rand", type=int, default=8)
    ap.add_argument("--zne-scales", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--zne-amplifier", default="model",
                    choices=["model", "fold"])
    ap.add_argument("--zne-fit", default="exp_fixed",
                    choices=["linear", "exp_fixed", "richardson"])
    ap.add_argument("--pec-samples", type=int, default=16)
    ap.add_argument("--n-cal-trex", type=int, default=24)
    ap.add_argument("--n-cal-cdr", type=int, default=32)
    # --- optimizacion ---
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--maxiter-ideal", type=int, default=2000)
    ap.add_argument("--warm-ideal", action="store_true",
                    help="arranca SPSA en theta* ideal (mide degradacion pura)")
    ap.add_argument("--optimizer", default="spsa",
                    choices=["spsa", "cobyla", "lbfgs"],
                    help="optimzador para la fase con ruido")
    # --- estudio ---
    ap.add_argument("--study", default="nibp",
                    choices=["nibp", "solve", "both"])
    ap.add_argument("--n-theta", type=int, default=12)
    ap.add_argument("--n-rep", type=int, default=6)
    ap.add_argument("--out-prefix", default="vqls_nisq")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()


# ============ PARCHE: AMG MEJORADO PARA BEAM1D ============
def _improved_amg_beam1d(A):
    """Precondicionador AMG mejorado para beam1d con estimación de modo de movimiento rígido."""
    import numpy as np
    import scipy.sparse as sp
    import pyamg
    
    n = A.shape[0]
    Acsr = sp.csr_matrix(A)
    
    # Estimar modos de movimiento rígido para beam1d
    nn = len(np.arange(0, n, 2))
    x = np.arange(1, nn + 1, dtype=float)
    B = np.zeros((n, 2))
    B[np.arange(0, n, 2), 0] = 1.0
    B[np.arange(0, n, 2), 1] = x
    B[np.arange(1, n, 2), 1] = 1.0
    B[:, 1] /= max(np.linalg.norm(B[:, 1]), 1e-30)
    
    cfg = {
        "smooth": "energy",
        "presmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 2}),
        "postsmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 2}),
        "max_levels": 10,
        "max_coarse": 10,
        "coarse_solver": "pinv",
    }
    
    best_kappa = np.inf
    best_Minv = None
    
    # Intenta smoothed aggregation con modos rígidos
    try:
        ml = pyamg.smoothed_aggregation_solver(Acsr, B=B, **cfg)
        P = ml.aspreconditioner(cycle="V")
        Minv = np.column_stack([P.matvec(e) for e in np.eye(n)])
        Minv = 0.5 * (Minv + Minv.T)
        w = np.linalg.eigvalsh(Minv)
        if w[0] <= 0:
            Minv += (abs(w[0]) + 1e-12 * w[-1]) * np.eye(n)
        S = _sym_sqrt(Minv)
        A_pre = S @ A @ S
        A_pre = 0.5 * (A_pre + A_pre.T)
        wa = np.linalg.eigvalsh(A_pre)
        best_kappa = wa[-1] / wa[0]
        best_Minv = Minv
    except Exception:
        pass
    
    # Fallback a Ruge-Stuben si la anterior falla o º > 10
    if best_Minv is None or best_kappa > 10:
        try:
            ml_rs = pyamg.ruge_stuben_solver(Acsr)
            P_rs = ml_rs.aspreconditioner(cycle="V")
            Minv_rs = np.column_stack([P_rs.matvec(e) for e in np.eye(n)])
            Minv_rs = 0.5 * (Minv_rs + Minv_rs.T)
            w = np.linalg.eigvalsh(Minv_rs)
            if w[0] <= 0:
                Minv_rs += (abs(w[0]) + 1e-12 * w[-1]) * np.eye(n)
            S_rs = _sym_sqrt(Minv_rs)
            A_pre_rs = S_rs @ A @ S_rs
            A_pre_rs = 0.5 * (A_pre_rs + A_pre_rs.T)
            wa_rs = np.linalg.eigvalsh(A_pre_rs)
            kappa_rs = wa_rs[-1] / wa_rs[0]
            if kappa_rs < best_kappa:
                best_Minv = Minv_rs
        except Exception:
            pass
    
    return best_Minv



