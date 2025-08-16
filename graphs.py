

from __future__ import annotations

import numpy as np
import cmath, math
import matplotlib.pyplot as plt

# Estilo visual (opcional)
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
except Exception:
    pass

# Airy: SciPy si disponible; si no, mpmath
try:
    from scipy.special import airy as scipy_airy
    _USE_SCIPY = True
except Exception:
    import mpmath as mp
    _USE_SCIPY = False

# Configuración de fuentes y mathtext (sin LaTeX externo)
plt.rcParams.update({
    "text.usetex": False,          # NO usar LaTeX externo
    "mathtext.fontset": "stix",   # MathText nativo
    "font.family": "STIXGeneral",
})

# ----------------------------- Constantes SI -----------------------------
hbar = 1.054_571_817e-34      # J*s
q_e  = 1.602_176_634e-19      # C
m_e  = 9.109_383_7015e-31     # kg
EV   = q_e                    # 1 eV en J
NM   = 1e-9                   # m
KVPCM_TO_VPM = 1e5            # 1 kV/cm = 1e5 V/m

# ----------------------------- Parámetros base ---------------------------
params = dict(
    # Geometría (nm)
    p_nm = 3.0,
    q_nm = 7.0,

    # Potenciales (eV)
    a_eV   = 0.008,   # barreras NM
    c_eV   = 0.008,  # splitting total c

    # Campo eléctrico en región 3 (kV/cm)
    F_kVcm = 20.0,

    # Masas relativas (× m_e)
    m1_rel = 0.067,
    m2_rel = 0.067,
    m3_rel = 0.50,
    m4_rel = 0.067,
    m5_rel = 0.067,

    # Energía (eV)
    E_min_eV = 1e-5,
    E_max_eV = 0.40,
    N_E      = 1600,

    # Archivos de salida
    out_bloch_pdf     = 'bloch_minibandas.pdf',
    out_potential_pdf = 'perfil_potencial.pdf',
)

EPS_K = 1e-15  # regularización de divisiones

# --------------------------- utilidades numéricas ------------------------

def airy_all(z: complex):
    """Devuelve (Ai, Ai', Bi, Bi') en z (complejo)."""
    if _USE_SCIPY:
        Ai, Aip, Bi, Bip = scipy_airy(z)
        return complex(Ai), complex(Aip), complex(Bi), complex(Bip)
    ai, aip, bi, bip = mp.airy(z)  # mpmath retorna (Ai, Ai', Bi, Bi')
    return complex(ai), complex(aip), complex(bi), complex(bip)


def k_val(E_J: complex, m: float, V_J: float) -> complex:
    return cmath.sqrt(2.0*m*(E_J - V_J)) / hbar


def safe_div(num, den):
    if abs(den) < EPS_K:
        den = den + 0j + EPS_K
    return num/den

# -------------------------- matrices de interfaz -------------------------

def build_interfaces(E_eV: float, P: dict, sigma: int):
    p = P['p_nm']*NM; q = P['q_nm']*NM
    a = P['a_eV']*EV
    c = P['c_eV']*EV
    F = P['F_kVcm']*KVPCM_TO_VPM

    m1 = P['m1_rel']*m_e; m2 = P['m2_rel']*m_e; m3 = P['m3_rel']*m_e
    m4 = P['m4_rel']*m_e; m5 = P['m5_rel']*m_e

    E  = E_eV*EV

    # Regiones planas 1 y 5: V=a; regiones 2 y 4: V=0
    U_a = a

    # Región 3: V_3(x) = a - sigma*c/2 + e F x
    b_sigma = a - sigma*(c/2.0)

    # vectores de onda
    k1 = k_val(E, m1, U_a)
    k2 = k_val(E, m2, 0.0)
    k4 = k_val(E, m4, 0.0)
    k5 = k_val(E, m5, U_a)

    # razones de masas
    l = m3/m4; n = m2/m3; o = m4/m5

    # parámetros de interfaz
    mu    = safe_div((m1/m2)*k2, k1)      # (m1*k2)/(m2*k1)
    mu_45 = safe_div(o*k5, k4)            # (m4*k5)/(m5*k4)

    # Airy en región 3
    alpha3 = (2.0*m3*q_e*F/(hbar**2))**(1/3)
    x0 = (E - b_sigma)/(q_e*F)     # E = b_sigma + e F x0
    X = p + q                      # entrada de R3
    Y = q + 2.0*p                  # salida de R3

    z_L = alpha3*(X - x0)
    z_R = alpha3*(Y - x0)

    AiL, AipL, BiL, BipL = airy_all(z_L)
    AiR, AipR, BiR, BipR = airy_all(z_R)

    # factores de acoplamiento
    nu  = safe_div(n*alpha3, k2)
    lam = safe_div(l*k4, alpha3)

    # exponentes de propagación
    e_k1m, e_k1p = cmath.exp(-1j*k1*p), cmath.exp(+1j*k1*p)
    e_k2m, e_k2p = cmath.exp(-1j*k2*p), cmath.exp(+1j*k2*p)
    e_q2m, e_q2p = cmath.exp(-1j*k2*q), cmath.exp(+1j*k2*q)
    e_Y4p, e_Y4m = cmath.exp(+1j*k4*Y), cmath.exp(-1j*k4*Y)
    e_q4m, e_q4p = cmath.exp(-1j*k4*q), cmath.exp(+1j*k4*q)

    # I12: (R1 -> R2)
    I12 = 0.5*np.array([
        [ e_k1m*(1+mu)*e_k2p,  e_k1m*(1-mu)*e_k2m ],
        [ e_k1p*(1-mu)*e_k2p,  e_k1p*(1+mu)*e_k2m ],
    ], dtype=complex)

    # I23: (R2 -> R3)
    I23 = 0.5*np.array([
        [ e_q2m*(AiL - 1j*nu*AipL),  e_q2m*(BiL - 1j*nu*BipL) ],
        [ e_q2p*(AiL + 1j*nu*AipL),  e_q2p*(BiL + 1j*nu*BipL) ],
    ], dtype=complex)

    # I34: (R3 -> R4)
    I34 = math.pi*np.array([
        [ e_Y4p*(BipR - 1j*lam*BiR),   e_Y4m*(BipR + 1j*lam*BiR) ],
        [ e_Y4p*(-AipR + 1j*lam*AiR),  e_Y4m*(-AipR - 1j*lam*AiR) ],
    ], dtype=complex)

    # I45: (R4 -> R5)
    c_plus, c_minus = (1+mu_45), (1-mu_45)
    I45 = 0.5*np.array([
        [ e_q4m*c_plus,  e_q4m*c_minus ],
        [ e_q4p*c_minus, e_q4p*c_plus  ],
    ], dtype=complex)

    return I12, I23, I34, I45

# ---------------------- ensamblado y g(E) ----------------------

def M_celda(E_eV: float, P: dict, sigma: int) -> np.ndarray:
    I12, I23, I34, I45 = build_interfaces(E_eV, P, sigma)
    return I12 @ I23 @ I34 @ I45


def g_rhs(E_eV: float, P: dict, sigma: int) -> complex:
    M = M_celda(E_eV, P, sigma)
    return 0.5*(M[0,0] + M[1,1])

# ------------- detección y ancho de minibandas -------------

def bands_from_g(E_grid_eV, g_vals, tol=1e-6):
    g_real = np.real(g_vals)
    mask = np.abs(g_real) <= 1.0 + tol
    bands = []
    if not np.any(mask):
        return bands
    idx = np.where(mask)[0]
    starts = [idx[0]]; ends = []
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            ends.append(idx[i-1]); starts.append(idx[i])
    ends.append(idx[-1])
    for s, e in zip(starts, ends):
        EL = E_grid_eV[s]
        ER = E_grid_eV[e]
        if ER > EL:
            bands.append((EL, ER))
    return bands

# ---------------- perfil de potencial (por periodo) ----------------

def potential_profile(P: dict, sigma: int, Npts_per_seg: int = 600):
    p_m = P['p_nm']*NM; q_m = P['q_nm']*NM
    a_eV = P['a_eV']; c_eV = P['c_eV']
    F_SI = P['F_kVcm']*KVPCM_TO_VPM  # V/m

    x0m = 0.0
    x1m = p_m
    x2m = x1m + q_m
    x3m = x2m + p_m
    x4m = x3m + q_m
    x5m = x4m + p_m

    def linspace_m(a, b, N):
        xm = np.linspace(a, b, N)
        return xm, xm/NM

    # Barrera NM izq: V=a (GRIS)
    x_bL_m, x_bL_nm = linspace_m(x0m, x1m, Npts_per_seg)
    V_bL = np.full_like(x_bL_nm, a_eV)
    # Pozo izq: V=0 (VERDE)
    x_wL_m, x_wL_nm = linspace_m(x1m, x2m, Npts_per_seg)
    V_wL = np.zeros_like(x_wL_nm)
    # Barrera M lineal: V = a - sigma*c/2 + (q_e F x)/EV (CELESTE CLARO)
    x_M_m, x_M_nm = linspace_m(x2m, x3m, Npts_per_seg)
    b_sigma_eV = a_eV - sigma*(c_eV/2.0)
    V_M = b_sigma_eV + (q_e*F_SI*x_M_m)/EV
    # Pozo der (VERDE)
    x_wR_m, x_wR_nm = linspace_m(x3m, x4m, Npts_per_seg)
    V_wR = np.zeros_like(x_wR_nm)
    # Barrera NM der (GRIS)
    x_bR_m, x_bR_nm = linspace_m(x4m, x5m, Npts_per_seg)
    V_bR = np.full_like(x_bR_nm, a_eV)

    x_nm = np.concatenate([x_bL_nm, x_wL_nm, x_M_nm, x_wR_nm, x_bR_nm])
    V_eV = np.concatenate([V_bL,    V_wL,    V_M,    V_wR,    V_bR   ])

    regions = [
        (x_bL_nm[0], x_bL_nm[-1], 'Barrera plana',   '#dddddd'),  # gris
        (x_wL_nm[0], x_wL_nm[-1], 'Pozo',            '#a8e6a3'),  # verde
        (x_M_nm[0],  x_M_nm[-1],  'Barrera lineal',  '#c9e6ff'),  # celeste claro
        (x_wR_nm[0], x_wR_nm[-1], 'Pozo',            '#a8e6a3'),
        (x_bR_nm[0], x_bR_nm[-1], 'Barrera plana',   '#dddddd'),
    ]
    return x_nm, V_eV, regions

# ------------------------------- utilidades de formato -------------------------------

def format_bandwidths(bands, max_count=6):
    """Texto compacto con anchos ΔE en eV."""
    dEs = [ER-EL for (EL, ER) in bands]
    if len(dEs) == 0:
        return "—"
    dEs = dEs[:max_count]
    return ", ".join(f"{d:.4f}" for d in dEs) + " eV"

# ------------------------------- main plot -------------------------------

def run_and_plot(P: dict):
    E_grid = np.linspace(P['E_min_eV'], P['E_max_eV'], P['N_E'])

    # Espines: ↑ azul, ↓ rojo
    spins = [ (+1, 'uparrow', 'blue', '↑'), (-1, 'downarrow', 'red', '↓') ]

    # ======= Figura 1: condición de Bloch y minibandas =======
    fig1 = plt.figure(figsize=(9.2, 6.2))
    ax1 = fig1.add_subplot(111)
    lines = []
    legend_lines = []

    bands_info = {}

    for sigma, s_macro, color, arrow_txt in spins:
        g_vals = np.array([g_rhs(float(Ee), P, sigma) for Ee in E_grid], dtype=complex)
        g_real = np.real(g_vals)
        bands  = bands_from_g(E_grid, g_vals)

        label_txt = f"(1/2) Tr(M_celda) {arrow_txt}"
        ln, = ax1.plot(E_grid, g_real, lw=1.8, color=color, label=label_txt)
        lines.append(ln); legend_lines.append(ln)

        for (EL, ER) in bands:
            ax1.axvspan(EL, ER, color=color, alpha=0.15)
            ax1.axvline(EL, color=color, lw=0.9, alpha=0.9)
            ax1.axvline(ER, color=color, lw=0.9, alpha=0.9)

        bands_info[sigma] = format_bandwidths(bands)

    # Guías ±1
    ax1.axhline(1.0,  ls='--', lw=1.0, color='k', alpha=0.7)
    ax1.axhline(-1.0, ls='--', lw=1.0, color='k', alpha=0.7)

    # Límites Y con margen
    y_all = np.concatenate([ln.get_ydata(orig=False) for ln in lines])
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    pad = 0.15*(y_max - y_min if y_max>y_min else 1.0)
    ax1.set_ylim(y_min - pad, y_max + pad)

    # Etiquetas/título compatibles con mathtext del backend PDF
    ax1.set_xlabel('Energía (eV)')
    ax1.set_ylabel(r"$\frac{1}{2}\,\mathrm{Tr}(M_{\mathrm{celda}}(E))$")
    ax1.set_title(r"Condición de Bloch: $\cos(KT)=\frac{1}{2}\,\mathrm{Tr}(M_{\mathrm{celda}})$")
    ax1.grid(True, alpha=0.25)
    ax1.legend(handles=legend_lines, loc='best', frameon=True)

    # Cajas con ΔE
    box_kw = dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3')
    ax1.text(0.985, 0.98,
             r"$\Delta E$ (↑): " + bands_info.get(+1, "—"),
             transform=ax1.transAxes, ha='right', va='top',
             color='blue', fontsize=10, bbox=box_kw)
    ax1.text(0.985, 0.90,
             r"$\Delta E$ (↓): " + bands_info.get(-1, "—"),
             transform=ax1.transAxes, ha='right', va='top',
             color='red', fontsize=10, bbox=box_kw)

    try:
        fig1.tight_layout()
    except Exception:
        pass
    fig1.savefig(P['out_bloch_pdf'])

    # ======= Figura 2: perfil de potencial por periodo =======
    fig2 = plt.figure(figsize=(10.0, 5.4))
    ax2 = fig2.add_subplot(111)

    # sombrear regiones fijas
    _, _, regions = potential_profile(P, +1)
    for (xa, xb, name, color_fill) in regions:
        ax2.axvspan(xa, xb, alpha=0.35, color=color_fill)
        ax2.text(0.5*(xa+xb), 0.97, name, ha='center', va='top',
                 transform=ax2.get_xaxis_transform(), fontsize=9)

    # trazas ↑ (azul) y ↓ (rojo)
    for sigma, s_macro, color, arrow_txt in spins:
        x_nm, V_eV, _ = potential_profile(P, sigma)
        ax2.plot(x_nm, V_eV, lw=2.0, color=color, label=f"Perfil V(x) {arrow_txt}")

    ax2.set_xlabel('Posición x dentro del periodo (nm)')
    ax2.set_ylabel('V(x) (eV)')
    ax2.set_title(r"Perfil de potencial por periodo: $V_3(x)=a-\sigma\,c/2+eFx$")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='best', frameon=True)

    try:
        fig2.tight_layout()
    except Exception:
        pass
    fig2.savefig(P['out_potential_pdf'])


if __name__ == "__main__":
    run_and_plot(params)
