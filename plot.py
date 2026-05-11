import pandas as pd, matplotlib.pyplot as plt, numpy as np

before = pd.read_csv('results_before.csv')
after  = pd.read_csv('results_after.csv')
domains = [l.split('_')[0] for l in before['label']]
x = np.arange(len(domains))

fig, axes = plt.subplots(1,2,figsize=(12,5))
for ax, metric, title in zip(axes, ['ECE','OWA'],
    ['ECE Before vs After QLoRA','OWA Before vs After QLoRA']):
    ax.bar(x-0.2, before[metric], 0.35, label='Before', color='steelblue')
    ax.bar(x+0.2, after[metric],  0.35, label='After',  color='coral')
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_title(title); ax.legend()
plt.suptitle('Gemma 2B Calibration Change After QLoRA', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('calibration_results.png', dpi=150, bbox_inches='tight')
print('Saved: calibration_results.png')
