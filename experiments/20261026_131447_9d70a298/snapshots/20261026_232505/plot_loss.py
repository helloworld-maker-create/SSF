"""Render recorded global training losses; use rank 0 once, without rank duplication."""
from pathlib import Path
import csv
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
summary = json.loads((here / 'summary.json').read_text())
with (here / 'updates.csv').open(newline='') as stream:
    rows = [r for r in csv.DictReader(stream) if r['rank'] == '0']
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, constrained_layout=True)
for ax, kind, label, color in zip(axes, ['G', 'C'], ['Generator DMD loss', 'Critic loss'], ['#2563eb', '#b45309']):
    selected = [r for r in rows if r['kind'] == kind]
    ax.plot([int(r['outer_before']) for r in selected],
            [float(r['loss_global_mean']) for r in selected],
            marker='.', markersize=4, linewidth=1.1, color=color)
    ax.set_ylabel(label)
    ax.grid(alpha=.2)
    ax.set_xlim(0, summary['counts']['outer'])
axes[-1].set_xlabel('Outer iteration before the update (zero-based)')
fig.suptitle(f"Baseline in progress: {summary['counts']['outer']}/600 outer iterations")
fig.supxlabel('Training losses only; video quality and SSF comparison are not available.', fontsize=9)
fig.savefig(here / 'loss.png', dpi=170)
plt.close(fig)
