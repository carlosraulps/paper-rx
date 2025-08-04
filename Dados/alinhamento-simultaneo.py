
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realinhamento simultâneo de CJ_LiF_*_aligned.dat em 2θ,
baseado no pico de referência calculado para λ = 0.07107 nm (Mo Kα).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- parâmetros de Bragg e λ alvo corrigido ---
d_lif_nm         = 0.201       # nm (LiF (100)/(200))
lambda_target_nm = 0.07107     # nm (Mo Kα)

# cálculo de 2θ alvo: θ = arcsin(λ/(2d)), 2θ = 2θ
arg = lambda_target_nm / (2 * d_lif_nm)
if not 0 < arg <= 1:
    raise ValueError(f"λ/2d = {arg:.3f} fora de [0,1]! Verifique λ e d.")
theta_target_rad  = np.arcsin(arg)
two_theta_target  = 2 * np.degrees(theta_target_rad)
print(f"2θ alvo (λ={lambda_target_nm} nm): {two_theta_target:.4f}°")

# --- diretórios ---
base_dir  = Path(__file__).resolve().parent
dados_dir = base_dir / "."
out_dir   = base_dir / "alinhamento-simultaneo"
out_dir.mkdir(exist_ok=True)

# --- identifica arquivos já alinhados ---
aligned_files = sorted(dados_dir.glob("CJ_LiF_*_aligned.dat"))
if not aligned_files:
    aligned_files = sorted(dados_dir.glob("CJ_LiF_*[!.aligned].dat"))
    print("Usando arquivos originais como alinhados:", aligned_files)

# --- lê referência ---
# prioriza o _aligned, senão cai no .dat original
ref_path_candidates = [
    dados_dir / "CJ_LiF_Ref_aligned.dat",
    dados_dir / "CJ_LiF_Ref.dat"
]
for ref_path in ref_path_candidates:
    if ref_path.exists():
        ref_data = np.genfromtxt(ref_path, comments="#")
        print(f"Usando referência: {ref_path.name}")
        break
else:
    raise FileNotFoundError("Arquivo de referência CJ_LiF_Ref(.aligned).dat não encontrado")

two_theta_ref = ref_data[:,0]
I_ref         = ref_data[:,1]
# filtra apenas valores finitos
mask_ref      = np.isfinite(two_theta_ref) & np.isfinite(I_ref)
peak_idx_ref  = np.nanargmax(I_ref[mask_ref])
peak_ref_2t   = two_theta_ref[mask_ref][peak_idx_ref]
print(f"Pico Ref em 2θ = {peak_ref_2t:.4f}°")

# --- loop de realinhamento ---
plot_data = {}
for arq in aligned_files:
    stem = arq.stem.replace("_aligned","")
    data = np.genfromtxt(arq, comments="#")
    if data.ndim<2 or data.shape[1]<2:
        print(f"⚠️ Ignorando formato inválido: {arq.name}")
        continue

    two_theta = data[:,0]
    I         = data[:,1]
    mask      = np.isfinite(two_theta) & np.isfinite(I)
    if not np.any(mask):
        print(f"⚠️ Sem dados válidos em {arq.name}")
        continue

    # pico neste arquivo (somente pontos finitos)
    idx_peak   = np.nanargmax(I[mask])
    peak_2t    = two_theta[mask][idx_peak]
    delta      = two_theta_target - peak_2t
    two_theta2 = two_theta + delta

    # salvar arquivo realinhado
    out_name = f"{stem}_re_aligned.dat"
    out_path = out_dir / out_name
    np.savetxt(
        out_path,
        np.column_stack([two_theta2, I]),
        header="two_theta(deg)    Intensidade",
        fmt="%.6f    %.6f"
    )
    print(f"{out_name}: pico={peak_2t:.4f}°, Δ={delta:.4f}°")

    # armazena para plot
    label = stem.replace("CJ_LiF_","")
    plot_data[label] = (two_theta2, I)

# --- plot único de I vs 2θ realinhado ---
plt.figure(figsize=(8,6))
for label, (xx, yy) in plot_data.items():
    plt.plot(xx, yy, label=label)
plt.xlabel(r"Ângulo duplo de difração $2\theta$ (°)")
plt.ylabel("Intensidade (u.a.)")
plt.title(f"I vs 2θ (picos realinhados a {two_theta_target:.4f}°)")
plt.legend(loc="best", fontsize="small", framealpha=0.8)
plt.grid(ls="--", alpha=0.4)
plt.tight_layout()
plt.show()

