#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot I vs λ, I vs θ e razões I_target/I_Ref vs λ para LiF em diferentes alvos
(Ag, In, Mo, Nb), usando Lei de Bragg e descartando arquivos muito pequenos.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- parâmetros do experimento --------
d_lif_nm = 0.201           # distância interplanar LiF (100)/(200) em nm
d_nacl_nm = 0.282 

d_lif_m  = d_lif_nm * 1e-9  # em m
d_nacl_m  = d_nacl_nm * 1e-9  # em m


# -------- diretórios --------
script_dir = Path(__file__).resolve().parent
dados_dir  = script_dir / 'Dados'

# -------- Mapeamento alvo → arquivo --------

# --- controle: usar arquivos alinhados ou não ---
aligned = False  # ou False, se quiser os .dat sem "_aligned"

# --- mapa de "alvo" → nome base do arquivo (sem extensão) ---
bases = {
    'Ag':  'ag1',
    'In':   'in1',
    'Mo':   'mo',
    'Sr':   'sr1',
    'Zr':   'zr1',
    'Nb':   'nb1',
    'Sn':   'sn',
    'RefLif':  'lifref',
    'Ref':  'ref'
}
# --- monta o sufixo conforme aligned ---
suffix = '_aligned' if aligned else ''
# --- monta o mapeamento final "alvo" → arquivo ---
#alvos = {
#    alvo: f'Lif{base}{suffix}.txt'
#    for alvo, base in bases.items()
#}

alvos = {
    alvo: f'nacl{base}{suffix}.txt'
    for alvo, base in bases.items()
}
# -------- coleta e processamento dos dados --------
resultados = {}
for nome, fname in alvos.items():
    arq = dados_dir / fname
    # validações de existência e tamanho omitidas...

    dados = []
    with open(arq, 'r', encoding='latin-1', errors='ignore') as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith('#'):
                continue
            partes = linha.split()
            if len(partes) < 2:
                continue
            try:
                dois_theta = float(partes[0])
                cont       = float(partes[1])
            except ValueError:
                continue

            # (1) θ é metade de 2θ
            theta     = dois_theta / 2.0                               # θ = 2θ/2 :contentReference[oaicite:0]{index=0}
            theta_rad = np.radians(theta)                              # converte para radianos :contentReference[oaicite:1]{index=1}

            # (2) escolhe d conforme o alvo
            if nome in ('Sn', 'RefLif'):
                d_m = d_lif_m                                       
            else:
                d_m = d_nacl_m

            # (3) Bragg: λ = 2·d·sin(θ)
            lam_nm = 2 * d_m * np.sin(theta_rad) * 1e9               # em nm :contentReference[oaicite:2]{index=2}

            logcont = np.log(cont)
            dados.append((theta, lam_nm, cont, logcont))

    if dados:
        arr = np.array(dados)
        resultados[nome] = {
            'theta':     arr[:, 0],
            'lambda':    arr[:, 1],
            'intens':    arr[:, 2],
            'logintens': arr[:, 3]
        }

# -------- plot 1: I vs θ --------
fig2, ax2 = plt.subplots(figsize=(8, 5))
for nome, vals in resultados.items():
    if nome == 'Ref':
        ax2.plot(vals['theta'], vals['intens'], '--', label='Ref', alpha=0.7)
    else:
        ax2.plot(vals['theta'], vals['intens'], label=nome)
ax2.set_xlabel(r"Ângulo de difração $\theta$ (graus)")
ax2.set_ylabel("Intensidade (u.a.)")
ax2.set_title(r"Curvas LiF: Intensidade vs Ângulo $\theta$")
ax2.grid(True, ls="--", alpha=0.4)
ax2.legend(loc="best", fontsize="small", framealpha=0.8)
plt.tight_layout()
plt.savefig("figures/p1-I-vs-theta.pdf", format="pdf")





# -------- plot 2: Log(I) vs θ --------
fig2, ax2 = plt.subplots(figsize=(8, 5))
for nome, vals in resultados.items():
    if nome == 'Ref':
        ax2.plot(vals['theta'], vals['logintens'], '--', label='Ref', alpha=0.7)
    else:
        ax2.plot(vals['theta'], vals['logintens'], label=nome)
ax2.set_xlabel(r"Ângulo de difração $\theta$ (graus)")
ax2.set_ylabel("Log(Intensidade (u.a.))")
ax2.set_title(r"Curvas LiF na escala logaritmica: Log(Intensidade) vs Ângulo $\theta$")
ax2.grid(True, ls="--", alpha=0.4)
ax2.legend(loc="best", fontsize="small", framealpha=0.8)
plt.tight_layout()
plt.savefig("figures/p2-LogI-vs-theta.pdf", format="pdf")

# -------- plot 3: I vs λ --------
fig1, ax1 = plt.subplots(figsize=(8, 5))
for nome, vals in resultados.items():
    if nome == 'Ref':
        ax1.plot(vals['lambda'], vals['intens'], '--', label='Ref', alpha=0.7)
    else:
        ax1.plot(vals['lambda'], vals['intens'], label=nome)

ax1.axvline(0.0711, color='red', ls='--', lw=1, label=r'$\lambda=0.0711\,$nm')
ax1.set_xlabel(r"Comprimento de onda $\lambda$ (nm)")
ax1.set_ylabel("Intensidade (u.a.)")
ax1.set_title(r"Curvas LiF: Intensidade vs Comprimento de onda $\lambda$")
ax1.grid(True, ls="--", alpha=0.4)
ax1.legend(loc="best", fontsize="small", framealpha=0.8)
plt.tight_layout()
plt.savefig("figures/p3-I-vs-lambda.pdf", format="pdf")

# -------- plot 4: Razões I_target/I_Ref vs λ --------
if 'Ref' in resultados:
    lam_ref = resultados['Ref']['lambda']
    int_ref = resultados['Ref']['intens']
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for nome, vals in resultados.items():
        if nome == 'Ref':
            continue
        int_i  = np.interp(lam_ref, vals['lambda'], vals['intens'])
        ratio  = int_i / int_ref
        ax3.plot(lam_ref, ratio, label=nome)
    ax3.set_xlabel(r"Comprimento de onda $\lambda$ (nm)")
    ax3.set_ylabel(r"$I_{target}/I_{Ref}$")
    ax3.set_title(r"Razão $I_{target}/I_{Ref}$ vs Comprimento de onda")
    ax3.grid(True, ls="--", alpha=0.4)
    ax3.legend(loc="best", fontsize="small", framealpha=0.8)
    plt.tight_layout()
    plt.savefig("figures/p4-I_Iref-vs-lambda.pdf", format="pdf")
else:
    print("⚠️ Referência 'Ref' não encontrada: pulando cálculo de razões.")


# -------- Plot 5: Suavizado Log(I) vs θ com Savitzky–Golay --------
from scipy.signal import savgol_filter, find_peaks

# parâmetros da suavização
window_length = 6  # tamanho da janela (ímpar) – deve ser <= tamanho de vals['theta']
polyorder     = 3    # ordem do polinômio para ajuste

fig_smooth, ax_smooth = plt.subplots(figsize=(8, 5))
for nome, vals in resultados.items():
    # aplica só em alvos (não em Ref), mas você pode incluir Ref se quiser
    y_raw = vals['logintens']
    x = vals['theta']

    # 1) encontra picos altos para preservá-los
    peaks, _ = find_peaks(y_raw, height=np.mean(y_raw) + 2*np.std(y_raw))
    
    # 2) aplica filtro Savitzky–Golay
    y_smooth = savgol_filter(y_raw, window_length=window_length, polyorder=polyorder, mode='interp')  # :contentReference[oaicite:0]{index=0}

    # 3) restaura valor original nos picos (para não suavizar picos experimentais)
    y_smooth[peaks] = y_raw[peaks]  

    # plota
    if nome == 'Ref':
        ax_smooth.plot(x, y_smooth, '--', label='Ref suavizado', alpha=0.7)
    else:
        ax_smooth.plot(x, y_smooth, label=f'{nome} suavizado')

ax_smooth.set_xlabel(r"Ângulo de difração $\theta$ (graus)")
ax_smooth.set_ylabel("Log(Intensidade) suavizado (u.a.)")
ax_smooth.set_title("Suavização Savitzky–Golay: Log(I) vs Ângulo θ")  
ax_smooth.grid(True, ls="--", alpha=0.4)
ax_smooth.legend(loc="best", fontsize="small", framealpha=0.8)
plt.tight_layout()
plt.savefig("figures/p5-LogI-vs-theta-suavizado.pdf", format="pdf")

# -------- plot 6: Razões suavizadas I_target/I_Ref vs λ --------

# parâmetros da suavização
window_length = 9  # tamanho da janela (ímpar, <= len(lam_ref))
polyorder     = 3   # ordem do polinômio

fig_ratio, ax_ratio = plt.subplots(figsize=(8, 5))

# dados de referência
lam_ref = resultados['Ref']['lambda']
int_ref = resultados['Ref']['intens']

for nome, vals in resultados.items():
    if nome == 'Ref':
        continue

    # 1) interpola intensidade do alvo em lam_ref
    int_i = np.interp(lam_ref, vals['lambda'], vals['intens'])
    ratio = int_i / int_ref

    # 2) detecta picos altos para preservá-los
    peaks, _ = find_peaks(ratio, height=np.mean(ratio) + 2*np.std(ratio))  

    # 3) aplica Savitzky–Golay ao ratio
    ratio_smooth = savgol_filter(ratio, window_length, polyorder, mode='interp') 

    # 4) restaura valores originais nos picos
    ratio_smooth[peaks] = ratio[peaks] 

    # plota a curva suavizada
    ax_ratio.plot(lam_ref, ratio_smooth, label=f'{nome} suavizado')

ax_ratio.set_xlabel(r"Comprimento de onda $\lambda$ (nm)")
ax_ratio.set_ylabel(r"Razão suavizada $I_{target}/I_{Ref}$")
ax_ratio.set_title("Razões suavizadas $I_{target}/I_{Ref}$ vs Comprimento de onda")  
ax_ratio.grid(True, ls="--", alpha=0.4)
ax_ratio.legend(loc="best", fontsize="small", framealpha=0.8)
plt.tight_layout()
plt.savefig("figures/p6-I_Iref-lambda-suavizado.pdf", format="pdf")

# -------- plot 7: Razões suavizadas I_target/I_Ref vs λ (filtro duplo) --------
from scipy.signal import savgol_filter, find_peaks

# parâmetros do Savitzky–Golay
window_length = 9   # janela ímpar
polyorder     = 3    # ordem do polinômio

# obtém dados de referência
lam_ref = resultados['Ref']['lambda']
int_ref = resultados['Ref']['intens']

# 1) suaviza I_ref
int_ref_smooth = savgol_filter(int_ref, window_length, polyorder, mode='interp')  # :contentReference[oaicite:0]{index=0}

fig_duplo, ax_duplo = plt.subplots(figsize=(8, 5))

for nome, vals in resultados.items():
    if nome == 'Ref':
        continue

    # intensidade bruta do alvo
    int_i = np.interp(lam_ref, vals['lambda'], vals['intens'])
    
    # 2) suaviza I_target
    int_i_smooth = savgol_filter(int_i, window_length, polyorder, mode='interp')  

    # 3) calcula razão entre as séries suavizadas
    ratio = int_i_smooth / int_ref_smooth

    # 4) detecta picos altos na razão (para preservar)
    peaks, _ = find_peaks(ratio, height=np.mean(ratio) + 2*np.std(ratio))  

    # 5) suaviza a razão
    ratio_smooth = savgol_filter(ratio, window_length, polyorder, mode='interp') 

    # 6) restaura picos originais
    ratio_smooth[peaks] = ratio[peaks] 

    # plota
    ax_duplo.plot(lam_ref, ratio_smooth, label=f'{nome} (duplo filtro)')

ax_duplo.set_xlabel(r"Comprimento de onda $\lambda$ (nm)")
ax_duplo.set_ylabel(r"Razão suavizada $I_{target}/I_{Ref}$")
ax_duplo.set_title("Razão dupla suavização: $I_{target}/I_{Ref}$ vs λ")
ax_duplo.grid(True, ls="--", alpha=0.4)
ax_duplo.legend(loc="best", fontsize="small", framealpha=0.8)
plt.tight_layout()
plt.savefig("figures/p7-I_Iref-lambda-duplo-suavizado.pdf", format="pdf")

# -------- Plot 8: Razões suavizadas I_target/I_Ref vs λ (com intervalo e pausa de primeira suavização) --------
from scipy.signal import savgol_filter, find_peaks

# --- parâmetros de corte em θ (graus) ---
theta_i = 2.5   # início da análise
theta_f = 15.0  # fim da análise
theta_a = 8.75 # início da “zona sem primeira suavização”
theta_b = 11.25 # fim da “zona sem primeira suavização”

# --- parâmetros do Savitzky–Golay ---
window_length = 9  # janela ímpar; ajuste conforme seu tamanho de amostra
polyorder     = 3    # ordem do polinômio

# extrai dados de Ref e filtra por θ_i–θ_f
theta_ref_full = resultados['Ref']['theta']
mask_ref = (theta_ref_full >= theta_i) & (theta_ref_full <= theta_f)
lam_ref = resultados['Ref']['lambda'][mask_ref]
int_ref = resultados['Ref']['intens'][mask_ref]

# 1) primeira suavização de I_ref, exceto na zona [theta_a,theta_b]
mask_no_smooth = (theta_ref_full >= theta_a) & (theta_ref_full <= theta_b)
mask_no_smooth = mask_no_smooth & mask_ref  # aplica também theta_i–theta_f

int_ref_first = int_ref.copy()
# índice dentro de lam_ref onde pode suavizar
idx_smooth = ~mask_no_smooth[mask_ref]
int_ref_first[idx_smooth] = savgol_filter(
    int_ref[idx_smooth], window_length, polyorder, mode='interp'
)

fig_duplo2, ax_duplo2 = plt.subplots(figsize=(8, 5))

for nome, vals in resultados.items():
    if nome == 'Ref':
        continue

    # extrai e interpola I_target no mesmo lam_ref
    theta_i_full = vals['theta']
    mask_tgt = (theta_i_full >= theta_i) & (theta_i_full <= theta_f)
    lam_tgt = vals['lambda'][mask_tgt]
    int_tgt = vals['intens'][mask_tgt]
    int_i = np.interp(lam_ref, lam_tgt, int_tgt)

    # 2) primeira suavização de I_target, exceto na zona [theta_a,theta_b]
    int_i_first = int_i.copy()
    idx_smooth_i = ~mask_no_smooth[mask_ref]
    int_i_first[idx_smooth_i] = savgol_filter(
        int_i[idx_smooth_i], window_length, polyorder, mode='interp'
    )

    # 3) calcula razão entre as primeiras suavizações
    ratio_first = int_i_first / int_ref_first

    # 4) detecta picos altos na razão para preservar
    peaks, _ = find_peaks(ratio_first, height=np.mean(ratio_first) + 2*np.std(ratio_first))

    # 5) segunda suavização (final) na razão inteira
    ratio_smooth = savgol_filter(ratio_first, window_length, polyorder, mode='interp')
    ratio_smooth[peaks] = ratio_first[peaks]  # restaura picos

    # 6) plota
    ax_duplo2.plot(lam_ref, ratio_smooth, label=f'{nome} (duplo filtro c/ intervalo)')

ax_duplo2.set_xlabel(r"Comprimento de onda $\lambda$ (nm)")
ax_duplo2.set_ylabel(r"Razão suavizada $I_{target}/I_{Ref}$")
ax_duplo2.set_title("Razão dupla suavização com pausa em subintervalo de θ") 
ax_duplo2.grid(True, ls="--", alpha=0.4)
ax_duplo2.legend(loc="best", fontsize="small", framealpha=0.8)
plt.tight_layout()
plt.savefig("figures/p8-I_Iref-lambda-duplo-suavizado-selectivo.pdf", format="pdf")



# -------- Plot 9: μ/ρ vs λ e μ/ρ vs E a partir dos dados .dat --------


# -------- ordem de plotagem, espessuras e densidades --------
order = ['sr', 'zr', 'nb', 'mo', 'ag', 'in', 'sn']

thickness_mm = {
    'ag': 0.05, 'sn': 0.12, 'in': 0.10,
    'mo': 0.10, 'nb': 0.16, 'zr': 0.05, 'sr': 0.47
}
densities = {
    'ag': 10.49, 'sn': 7.289, 'in': 7.31,
    'mo': 10.28, 'nb': 8.582, 'zr': 6.511, 'sr': 2.64
}

# -------- referências --------
# Ref padrão (NaCl) e Ref LiF
# key → nome da chave em resultados
default_ref = 'Ref'      # usa NaCl
lif_ref     = 'RefLif'   # usa LiF

# mapear cada elemento à sua referência (se não estiver aqui, usa default_ref)
ref_map = {
    'sn': lif_ref,
    # adicione aqui 'seuElemento': lif_ref_ou_default_ref,
}

# -------- extrai todos os resultados uma vez --------
res = resultados  # suposição: `resultados` já preenchido anteriormente

# -------- figura 1: μ/ρ vs λ com múltiplas curvas --------
fig1, ax1 = plt.subplots(figsize=(8, 5))

for nome in order:
    key_sample = nome if nome in res else nome.capitalize()
    if key_sample not in res:
        print(f"⚠️ dados para '{nome}' não encontrados; pulando.")
        continue

    # decide qual referência usar
    ref_key = ref_map.get(nome, default_ref)
    lam_ref = res[ref_key]['lambda']
    int_ref = res[ref_key]['intens']

    vals = res[key_sample]
    # interpola intensidade do alvo em cada λ do ref
    int_i = np.interp(lam_ref, vals['lambda'], vals['intens'])
    T     = int_i / int_ref
    t_cm  = thickness_mm[nome] / 10.0
    rho   = densities[nome]
    mu_rho = -np.log(T) / (rho * t_cm)

    ax1.plot(lam_ref, mu_rho, label=f"{nome.upper()} (t={thickness_mm[nome]} mm)")

ax1.set_xlabel(r"Comprimento de onda $\lambda$ (nm)")
ax1.set_ylabel(r"Coeficiente de massa $\mu/\rho$ (cm$^2$/g)")
ax1.set_title(r"$\mu/\rho$ vs Comprimento de onda para vários alvos$")
ax1.grid(True, ls="--", alpha=0.4)
ax1.legend(loc='best', fontsize='small', framealpha=0.8)
plt.tight_layout()
fig1.savefig("figures/p9-mu_rho-multiple-lambda.pdf", bbox_inches='tight')


# -------- figura 2: subplots individuais, com linha em λ=0.0711 nm --------
present = [nome for nome in order
           if (nome in res) or (nome.capitalize() in res)]
n = len(present)
fig2, axs = plt.subplots(n, 1, sharex=True,
                         figsize=(8, 2.5*n),
                         constrained_layout=True)

for ax, nome in zip(axs, present):
    key_sample = nome if nome in res else nome.capitalize()
    vals = res[key_sample]

    # referência para este elemento
    ref_key = ref_map.get(nome, default_ref)
    lam_ref = res[ref_key]['lambda']
    int_ref = res[ref_key]['intens']

    # cálculo de μ/ρ
    int_i  = np.interp(lam_ref, vals['lambda'], vals['intens'])
    T      = int_i / int_ref
    t_cm   = thickness_mm[nome] / 10.0
    rho    = densities[nome]
    mu_rho = -np.log(T) / (rho * t_cm)

    ax.plot(lam_ref, mu_rho, linewidth=1.5, label=nome.upper())
    ax.axvline(0.0711, ls='--', lw=1,
               label=r'$\lambda=0.0711\,$nm')

    ax.set_ylabel(r'$\mu/\rho$ (cm$^2$/g)')
    ax.set_title(nome.upper())
    ax.grid(True, ls='--', alpha=0.3)

# só o último eixo recebe xlabel
axs[-1].set_xlabel(r'Comprimento de onda $\lambda$ (nm)')

# legendas sem duplicatas em cada subplot
for ax in axs:
    h, l = ax.get_legend_handles_labels()
    by_label = dict(zip(l, h))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize='small', framealpha=0.8)

fig2.savefig("figures/p10-mu_rho-subplots-lambda.pdf",
             format="pdf", bbox_inches='tight')
#     Plot 9:  Sub plot vs energy converte λ[nm] em energia E[MeV]


# -------- ordem, espessuras e densidades (mesmos de antes) --------
order = ['sr', 'zr', 'nb', 'mo', 'ag', 'in', 'sn']
thickness_mm = {
    'ag': 0.05, 'sn': 0.12, 'in': 0.10,
    'mo': 0.10, 'nb': 0.16, 'zr': 0.05, 'sr': 0.47
}
densities = {
    'ag': 10.49, 'sn': 7.289, 'in': 7.31,
    'mo': 10.28, 'nb': 8.582, 'zr': 6.511, 'sr': 2.64
}

# -------- referências --------
default_ref = 'Ref'     # NaCl
lif_ref     = 'RefLif'  # LiF
ref_map = {
    'sn': lif_ref,      # Sn usa LiF
    # adicione outros: 'elemento': lif_ref_ou_default_ref
}

res = resultados  # resultados já preenchido

# -------- calcula E a partir de cada lam_ref possível --------
# mas é melhor calcular E dentro do loop para cada ref distinto,
# pois lam_ref muda conforme ref_map.

# --- figura 3: μ/ρ vs energia (múltiplas curvas) ---
fig3, ax3 = plt.subplots(figsize=(8, 5))

for nome in order:
    key_sample = nome if nome in res else nome.capitalize()
    if key_sample not in res:
        continue

    # escolhe referência
    ref_key  = ref_map.get(nome, default_ref)
    lam_ref  = res[ref_key]['lambda']   # em nm
    int_ref  = res[ref_key]['intens']

    # converte λ→E
    E_eV   = 1240.0 / lam_ref
    E_MeV  = E_eV * 1e-6

    vals   = res[key_sample]
    int_i  = np.interp(lam_ref, vals['lambda'], vals['intens'])
    T      = int_i / int_ref
    t_cm   = thickness_mm[nome] / 10.0
    rho    = densities[nome]
    mu_rho = -np.log(T) / (rho * t_cm)

    ax3.plot(E_MeV, mu_rho, label=nome.upper())

ax3.set_xlabel("Energia do fóton (MeV)")
ax3.set_ylabel(r"Coeficiente de massa $\mu/\rho$ (cm$^2$/g)")
ax3.set_title(r"$\mu/\rho$ vs Energia do fóton para vários alvos$")
ax3.grid(True, ls="--", alpha=0.4)
ax3.legend(loc='best', fontsize='small', framealpha=0.8)
plt.tight_layout()
fig3.savefig("figures/p11-mu_rho-multiple-energy.pdf", bbox_inches='tight')


# --- figura 4: subplots individuais μ/ρ vs energia ---
present = [nome for nome in order
           if (nome in res) or (nome.capitalize() in res)]
n = len(present)
fig4, axs = plt.subplots(n, 1, sharex=True,
                         figsize=(8, 2.5*n),
                         constrained_layout=True)

for ax, nome in zip(axs, present):
    key_sample = nome if nome in res else nome.capitalize()
    # referência
    ref_key  = ref_map.get(nome, default_ref)
    lam_ref  = res[ref_key]['lambda']
    int_ref  = res[ref_key]['intens']

    # converte λ→E
    E_eV   = 1240.0 / lam_ref
    E_MeV  = E_eV * 1e-6

    vals   = res[key_sample]
    int_i  = np.interp(lam_ref, vals['lambda'], vals['intens'])
    T      = int_i / int_ref
    t_cm   = thickness_mm[nome] / 10.0
    rho    = densities[nome]
    mu_rho = -np.log(T) / (rho * t_cm)

    ax.plot(E_MeV, mu_rho, linewidth=1.5, label=nome.upper())
    # linha de referência (opcional), aqui no valor médio de E_MeV
    ax.axvline(np.mean(E_MeV), ls='--', lw=0.8, label=r'$\langle E\rangle$')

    ax.set_ylabel(r'$\mu/\rho$ (cm$^2$/g)')
    ax.set_title(nome.upper())
    ax.grid(True, ls='--', alpha=0.3)

# xlabel apenas no último
axs[-1].set_xlabel("Energia do fóton (MeV)")

# legendas sem duplicatas
for ax in axs:
    h, l = ax.get_legend_handles_labels()
    by_label = dict(zip(l, h))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize='small', framealpha=0.8)

fig4.savefig("figures/p12-mu_rho-subplots-energy.pdf",
             format="pdf", bbox_inches='tight')
# -------- Plot 10 vs Energia do fóton (MeV) usando Bohr model para linhas Kα --------

# ordem de plotagem
order = ['sr', 'zr', 'nb', 'mo', 'ag', 'in', 'sn']

# espessuras (mm) e densidades (g/cm³)
thickness_mm = {'ag':0.05,'sn':0.12,'in':0.10,'mo':0.10,'nb':0.16,'zr':0.05,'sr':0.47}
densities    = {'ag':10.49,'sn':7.289,'in':7.31,'mo':10.28,'nb':8.582,'zr':6.511,'sr':2.64}

# números atômicos z e constantes de blindagem σ para linhas kα
z = {'sr':38,'zr':40,'nb':41,'mo':42,'ag':47,'in':49,'sn':50}
sigma = {'sr':1.0,'zr':1.0,'nb':1.0,'mo':1.0,'ag':1.0,'in':1.0,'sn':1.0}

# rydberg energy (ev)
r_inf = 13.605693009  # energia de rydberg para hidrogênio 

# extrai λ e i_ref de 'ref'
lam_ref = resultados['ref']['lambda']
int_ref = resultados['ref']['intens']

# converte λ[nm] em energia contínua e_photon (mev)
e_ev_cont = 1240.0 / lam_ref        
e_mev_cont = e_ev_cont * 1e-6

# calcula energias kα via bohr model (ev) e converte para mev
# e_kα = r_inf * (z - σ)^2 * (1/1^2 - 1/2^2) = (3/4) r_inf (z - σ)^2
e_ka_ev = {nome: 0.75 * r_inf * (z[nome] - sigma[nome])**2 for nome in order}
e_ka_mev = {nome: e/1e6 for nome, e in e_ka_ev.items()}

# --- figura 1: múltiplas curvas μ/ρ vs energia (mev) com linhas kα ---
fig3, ax3 = plt.subplots(figsize=(8,5))

for nome in order:
    key = nome if nome in resultados else nome.capitalize()
    if key not in resultados:
        continue
    vals = resultados[key]
    int_i = np.interp(lam_ref, vals['lambda'], vals['intens'])
    t     = int_i / int_ref
    t_cm  = thickness_mm[nome]/10.0
    rho   = densities[nome]
    mu_rho = np.log(t**-1)/(rho*t_cm)
    ax3.plot(e_mev_cont, mu_rho, label=f"{nome.upper()}")

    # linha vertical na energia kα
    ax3.axvline(e_ka_mev[nome],
                color=ax3.lines[-1].get_color(),
                ls='--', lw=1,
                label=f"{nome.upper()} kα ({e_ka_mev[nome]:.3e} mev)")

ax3.set_xlabel("energia do fóton (mev)")
ax3.set_ylabel(r"$\mu/\rho$ (cm$^2$/g)")
ax3.set_title(r"$\mu/\rho$ vs energia com linhas kα (bohr)")
ax3.grid(true, ls="--", alpha=0.4)
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles, labels, loc='best', fontsize='small', framealpha=0.8)
plt.tight_layout()
fig3.savefig("figures/p13-mu_rho-multiple-energy.pdf", bbox_inches='tight')

# --- figura 2: subplots individuais μ/ρ vs energia (mev) com kα ---
present = [nome for nome in order if (nome in resultados) or (nome.capitalize() in resultados)]
n = len(present)
fig4, axs = plt.subplots(n,1,sharex=true,figsize=(8,2.5*n),constrained_layout=true)

for ax, nome in zip(axs, present):
    key = nome if nome in resultados else nome.capitalize()
    vals = resultados[key]
    int_i = np.interp(lam_ref, vals['lambda'], vals['intens'])
    t     = int_i / int_ref
    t_cm  = thickness_mm[nome]/10.0
    rho   = densities[nome]
    mu_rho = -np.log(t)/(rho*t_cm)

    ax.plot(e_mev_cont, mu_rho, color='c0', linewidth=1.5)
    # linha kα
    ax.axvline(e_ka_mev[nome], color='red', ls='--', lw=1)
    ax.text(e_ka_mev[nome], ax.get_ylim()[1]*0.8,
            f"kα={e_ka_mev[nome]:.2e} mev",
            rotation=90, va='top', ha='right', color='red', fontsize='x-small') 

    ax.set_ylabel(r'$\mu/\rho$ (cm$^2$/g)')
    ax.set_title(nome.upper())
    ax.grid(true, ls='--', alpha=0.3)

axs[-1].set_xlabel("energia do fóton (mev)")
fig4.savefig("figures/p14-mu_rho-subplots-energy.pdf", bbox_inches='tight')

print(r"""
          \O/      
           |       
          / \      

    {═════════════}
    { Tutto posto }
    {═════════════}
""")

