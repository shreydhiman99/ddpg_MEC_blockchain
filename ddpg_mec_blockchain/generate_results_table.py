"""
Generate a clean results table (LaTeX) with mean ± std across 3 seeds.
"""
import json, glob, numpy as np, os

agents_labels = [
    ('ddpg',      'DDPG (Ours)'),
    ('td3',       'TD3'),
    ('qlearning', 'Q-Learning'),
    ('greedy',    'Greedy'),
    ('random',    'Random'),
]

DATA = 'results/data'
rows = []
for ag, label in agents_labels:
    files = sorted(glob.glob(f'{DATA}/{ag}_s20_t50_*.json'))
    data  = [json.load(open(f)) for f in files]
    n     = len(data)

    r  = [d['mean_reward']     for d in data]
    d  = [d['mean_delay']      for d in data]
    e  = [d['mean_energy']     for d in data]
    th = [d['mean_throughput'] for d in data]

    rows.append({
        'label':     label,
        'r_mean':    np.mean(r),  'r_std':  np.std(r, ddof=1),
        'd_mean':    np.mean(d),  'd_std':  np.std(d, ddof=1),
        'e_mean':    np.mean(e),  'e_std':  np.std(e, ddof=1),
        'th_mean':   np.mean(th), 'th_std': np.std(th, ddof=1),
        'n': n,
    })

# find best per column
best_r  = max(rows, key=lambda x: x['r_mean'])['r_mean']
best_d  = min(rows, key=lambda x: x['d_mean'])['d_mean']
best_e  = min(rows, key=lambda x: x['e_mean'])['e_mean']
best_th = max(rows, key=lambda x: x['th_mean'])['th_mean']

def fmt(mean, std, best, hi=True):
    s = f'{mean:.3f} $\\pm$ {std:.3f}'
    if (hi and abs(mean - best) < 1e-9) or (not hi and abs(mean - best) < 1e-9):
        s = f'\\textbf{{{s}}}'
    return s

lines = []
lines.append(r'\begin{table}[ht]')
lines.append(r'\centering')
lines.append(r'\caption{Comparative performance (mean\,$\pm$\,std, 3 seeds, 20 servers, 50 tasks)}')
lines.append(r'\label{tab:results}')
lines.append(r'\begin{tabular}{lcccc}')
lines.append(r'\toprule')
lines.append(r'Algorithm & Cumulative Reward & End-to-End Delay (s) & Energy (J) & Throughput \\')
lines.append(r'\midrule')

for row in rows:
    r_str  = fmt(row['r_mean'],  row['r_std'],  best_r,  hi=True)
    d_str  = fmt(row['d_mean'],  row['d_std'],  best_d,  hi=False)
    e_str  = fmt(row['e_mean'],  row['e_std'],  best_e,  hi=False)
    th_str = fmt(row['th_mean'], row['th_std'], best_th, hi=True)
    lines.append(f'{row["label"]} & {r_str} & {d_str} & {e_str} & {th_str} \\\\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

tex = '\n'.join(lines)
print(tex)

out = 'results/figures/results_table.tex'
with open(out, 'w', encoding='utf-8') as f:
    f.write(tex)
print(f'\nSaved: {out}')
