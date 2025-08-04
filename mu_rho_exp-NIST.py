
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_experimental_data(dados_dir: Path, aligned: bool = True):
    """
    Carrega dados experimentais de difração (NaCl e LiF) para múltiplos alvos.
    Retorna dict: resultados[chave] = {'lambda': [...], 'intens': [...]}, contendo 'Ref' (NaCl) e 'RefLif'.
    """
    d_lif_m   = 0.201e-9
    d_nacl_m  = 0.282e-9
    suffix    = "_aligned" if aligned else ""
    bases     = {
        'Ag': 'ag1', 'In': 'in1', 'Mo': 'mo',
        'Sr': 'sr1', 'Zr': 'zr1', 'Nb': 'nb1',
        'Sn': 'sn', 'RefLif': 'lifref', 'Ref': 'ref'
    }
    ext       = '.txt'
    lif_keys  = {'Sn', 'RefLif'}
    resultados = {}

    for chave, base in bases.items():
        # tenta arquivos aligned e sem aligned
        for suf in ([suffix] if not aligned else [suffix, ""]):
            fname = f'nacl{base}{suf}{ext}'
            path  = dados_dir / fname
            if path.exists():
                arq_path = path
                if suf == "":
                    print(f"ℹ️ usando sem aligned para {chave}: {fname}")
                break
        else:
            print(f"⚠️ não encontrado: nacl{base}{suffix}{ext}")
            continue

        data = []
        with open(arq_path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                parts = line.split()
                if len(parts) < 2: continue
                try:
                    two_theta = float(parts[0])
                    intensity = float(parts[1])
                except ValueError:
                    continue
                theta  = two_theta / 2.0
                d_val  = d_lif_m if chave in lif_keys else d_nacl_m
                lam_nm = 2 * d_val * np.sin(np.radians(theta)) * 1e9
                data.append((lam_nm, intensity))

        arr = np.array(data)
        resultados[chave] = {'lambda': arr[:,0], 'intens': arr[:,1]}

    return resultados


def plot_energy_subplots(
    resultados, data_dir, order,
    thickness_mm, densities, z, sigma,
    emin, emax
):
    """
    Subplots de μ/ρ vs energia (MeV): NIST, experimental, kα.
    """
    hc    = 1239.841984
    r_inf = 13.605693009

    present = [nome for nome in order if nome.capitalize() in resultados]
    n = len(present)
    if n == 0:
        print("⚠️ Nenhum dado experimental encontrado.")
        return

    fig, axs = plt.subplots(
        n, 1,
        sharex=True,
        figsize=(8, 2.5*n),
        constrained_layout=True
    )
    if n == 1:
        axs = [axs]

    for ax, nome in zip(axs, present):
        key = nome.capitalize()

        # 1) NIST
        for nist_file in sorted(data_dir.glob(f'NIST-{z[nome]}*.dat')):
            df = pd.read_csv(nist_file, sep=r'\s+', comment='#',
                             names=['energia','mu_rho'])
            df = df[(df.energia>=emin)&(df.energia<=emax)]
            if not df.empty:
                ax.plot(df.energia, df.mu_rho, '--', label='NIST')
            break

        # 2) Experimental
        ref_key   = 'RefLif' if nome=='sn' else 'Ref'
        lam_ref   = resultados[ref_key]['lambda']
        int_ref   = resultados[ref_key]['intens']
        E_MeV_ref = (hc/lam_ref)*1e-6

        vals  = resultados[key]
        int_i = np.interp(lam_ref, vals['lambda'], vals['intens'])
        T     = int_i / int_ref
        mu_rho = -np.log(T) / (densities[nome]*(thickness_mm[nome]/10))
        mask   = (E_MeV_ref>=emin)&(E_MeV_ref<=emax)
        ax.plot(E_MeV_ref[mask], mu_rho[mask], '-', label='Exp.')

        # 3) Linha kα
        Ek_MeV = 0.75*r_inf*(z[nome]-sigma[nome])**2*1e-6
        if emin<=Ek_MeV<=emax:
            ax.axvline(Ek_MeV, ls=':', lw=1, color='red', label='kα')

        ax.set_title(key)
        ax.set_ylabel(key + '\n' + r'$\mu/\rho$ (cm$^2$/g)')
        ax.grid(True, ls='--', alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by = dict(zip(labels, handles))
        ax.legend(by.values(), by.keys(), fontsize='x-small', framealpha=0.8)

    axs[-1].set_xlabel('Energia do fóton (MeV)')
    fig.savefig('figures/pnm1-mu_rho-subplots-energy.pdf', bbox_inches='tight')


def plot_wavelength_subplots(
    resultados, order,
    thickness_mm, densities, z, sigma,
    emin_nm=0.01, emax_nm=0.1
):
    """
    Subplots de μ/ρ vs comprimento de onda (nm) para cada elemento:
      experimental + kα convertida em nm.
    """
    hc    = 1239.841984  # eV·nm
    r_inf = 13.605693009

    present = [nome for nome in order if nome.capitalize() in resultados]
    n = len(present)
    if n == 0:
        print("⚠️ Nenhum dado experimental para λ.")
        return

    fig, axs = plt.subplots(
        n, 1,
        sharex=True,
        figsize=(8, 2.5*n),
        constrained_layout=True
    )
    if n == 1:
        axs = [axs]

    for ax, nome in zip(axs, present):
        key  = nome.capitalize()
        vals = resultados[key]
        lam   = vals['lambda']
        inten = vals['intens']

        # Experimental
        ref_key = 'RefLif' if nome=='sn' else 'Ref'
        lam_ref = resultados[ref_key]['lambda']
        int_ref = resultados[ref_key]['intens']
        int_i   = np.interp(lam_ref, lam, inten)
        T       = int_i / int_ref
        mu_rho  = -np.log(T) / (densities[nome]*(thickness_mm[nome]/10))
        mask    = (lam_ref>=emin_nm)&(lam_ref<=emax_nm)
        ax.plot(lam_ref[mask], mu_rho[mask], '-', label='Exp.')

        # kα em nm
        Ek_eV = 0.75*r_inf*(z[nome]-sigma[nome])**2
        lam_ka= hc / Ek_eV
        if emin_nm<=lam_ka<=emax_nm:
            ax.axvline(lam_ka, ls=':', lw=1, color='red', label='kα')

        ax.set_title(key)
        ax.set_ylabel(key + '\n' + r'$\mu/\rho$ (cm$^2$/g)')
        ax.grid(True, ls='--', alpha=0.3)
        h, l = ax.get_legend_handles_labels(); by=dict(zip(l,h))
        ax.legend(by.values(), by.keys(), fontsize='x-small', framealpha=0.8)

    axs[-1].set_xlabel('Comprimento de onda (nm)')
    fig.savefig('figures/pnm1-mu_rho-subplots-lambda.pdf', bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(
        description='μ/ρ vs energia e comprimento de onda'
    )
    parser.add_argument(
        '-d','--data-dir', type=Path,
        default=Path('Dados'),
        help='Diretório de dados (.txt e .dat)'
    )
    parser.add_argument(
        '--emin', type=float, default=0.01,
        help='Energia mínima (MeV)'
    )
    parser.add_argument(
        '--emax', type=float, default=0.07,
        help='Energia máxima (MeV)'
    )
    parser.add_argument(
        '--no-aligned', action='store_true',
        help='Ignorar arquivos aligned'
    )
    args = parser.parse_args()

    order = ['sr','zr','nb','mo','ag','in','sn']
    thickness_mm = {
        'sr':0.35, 'zr':0.05, 'nb':0.16,'mo':0.10,
        'ag':0.05,'in':0.10,'sn':0.12
    }
    densities = {
        'sr':2.64,'zr':6.511,'nb':8.582,'mo':10.28,
        'ag':10.49,'in':7.31,'sn':7.289
    }
    z     = {'sr':38,'zr':40,'nb':41,'mo':42,'ag':47,'in':49,'sn':50}
    sigma = {k:1.0 for k in z}

    resultados = load_experimental_data(
        args.data_dir,
        aligned=not args.no_aligned
    )
    plot_energy_subplots(
        resultados, args.data_dir, order,
        thickness_mm, densities, z, sigma,
        args.emin, args.emax
    )
    plot_wavelength_subplots(
        resultados, order,
        thickness_mm, densities, z, sigma,
        emin_nm=0.005, emax_nm=0.05
    )


if __name__ == '__main__':
    main()

