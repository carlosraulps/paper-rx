
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 1) Lista dos arquivos originais (sem sufixo "_aligned")
files = [
    'alinhamento-simultaneo/CJ_LiF_KBr_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_SrCO3_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_Zr_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_Ag_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_Sn_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_In01_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_Mo_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_Nb016_re_aligned.dat',
    'alinhamento-simultaneo/CJ_LiF_Ref_re_aligned.dat',
]

# 2) Cria diretório de saída
out_dir = 'alinhamento-manual'
os.makedirs(out_dir, exist_ok=True)

def load_two_columns(fname):
    xs, ys = [], []
    with open(fname, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)

# 3) Carrega todos os dados
data = {}
for fname in files:
    x, y = load_two_columns(fname)
    data[fname] = {'x': x, 'y': y, 'dx': 0.0}

# 4) Configura figura e eixos
n = len(files)
fig = plt.figure(figsize=(10, 6 + n*0.3))
ax_main = fig.add_axes([0.1, 0.3 + 0.02*n, 0.85, 0.65 - 0.02*n])
lines = {}

# plota cada série com deslocamento zero inicial
for fname, d in data.items():
    line, = ax_main.plot(d['x'] + d['dx'], d['y'], label=os.path.splitext(fname)[0])
    lines[fname] = line

ax_main.set_title('Alinhamento Comparativo Manual')
ax_main.set_xlabel('X')
ax_main.set_ylabel('Y')
ax_main.grid(alpha=0.3)
ax_main.legend(loc='upper right', fontsize='small')

# 5) Cria sliders um abaixo do outro
slider_axes = {}
sliders = {}
h_slider = 0.02
for i, fname in enumerate(files):
    bottom = 0.25 - i*(h_slider + 0.005)
    ax = fig.add_axes([0.1, bottom, 0.75, h_slider], facecolor='lightgray')
    slider = Slider(ax, os.path.splitext(fname)[0],
                    -10.0, 10.0, valinit=0.0, valstep=0.01)
    slider_axes[fname] = ax
    sliders[fname] = slider

    def update_factory(name):
        def update(val):
            data[name]['dx'] = sliders[name].val
            lines[name].set_xdata(data[name]['x'] + data[name]['dx'])
            ax_main.relim(); ax_main.autoscale_view()
            fig.canvas.draw_idle()
        return update

    sliders[fname].on_changed(update_factory(fname))

# 6) Botão “Salvar Tudo”
ax_btn = fig.add_axes([0.87, 0.02, 0.1, 0.04])
btn = Button(ax_btn, 'Salvar Tudo', color='lightgray', hovercolor='0.9')

def save_all(event):
    for fname, d in data.items():
        aligned = np.column_stack([d['x'] + d['dx'], d['y']])
        outname = os.path.join(out_dir,
            os.path.splitext(fname)[0] + '_aligned.dat')
        np.savetxt(outname, aligned, fmt='%.6e',
                   header='# X_shifted    Y', comments='')
    print(f'→ Todos os arquivos salvos em "{out_dir}/" com sufixo "_aligned.dat".')

btn.on_clicked(save_all)

plt.show()

