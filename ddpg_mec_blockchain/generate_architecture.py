"""
Draw.io-style system architecture — minimal, clean, simple.
Flat boxes, thin borders, pastel fills, plain arrows.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Draw.io palette (pastel fill, darker border) ───────────────────────────
UE_F,  UE_E  = '#DAE8FC', '#6C8EBF'
ME_F,  ME_E  = '#D5E8D4', '#82B366'
BC_F,  BC_E  = '#FFE6CC', '#D79B00'
RL_F,  RL_E  = '#E1D5E7', '#9673A6'
TX     = '#333333'

# ── helpers ────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, fc, ec, lines):
    r = plt.Rectangle((cx-w/2, cy-h/2), w, h,
                       facecolor=fc, edgecolor=ec,
                       linewidth=1.8, zorder=3)
    ax.add_patch(r)
    step = h / (len(lines) + 1)
    for i, (txt, fs, bold) in enumerate(lines):
        ax.text(cx, cy + h/2 - step*(i+1),
                txt, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal',
                color=TX, zorder=4)

def harrow(x1, y, x2, label, ec='#555555', lw=1.6, dashed=False,
           above=True, fs=8):
    ls = (0, (5, 3)) if dashed else 'solid'
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=ec, lw=lw,
                                linestyle=ls))
    dy = 0.22 if above else -0.24
    ax.text((x1+x2)/2, y+dy, label,
            ha='center', va='center', fontsize=fs, color=ec)

def varrow(x, y1, y2, ec='#555555', lw=1.4):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=ec, lw=lw))

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
ax.text(8, 8.65,
        'Proposed Blockchain-Secured MEC Resource Allocation Framework with DDPG',
        ha='center', va='center', fontsize=13, fontweight='bold', color=TX)

# ══════════════════════════════════════════════════════════════════════════
# ROW LABELS  (simple grey text above each row)
# ══════════════════════════════════════════════════════════════════════════
for lbl, x in [('Mobile Devices', 1.7), ('MEC Servers', 5.2),
               ('Blockchain Network', 9.8), ('DDPG Agent', 13.9)]:
    ax.text(x, 8.20, lbl, ha='center', va='center',
            fontsize=9, color='#666666', fontweight='bold')
    ax.plot([x-1.4, x+1.4], [8.0, 8.0], color='#CCCCCC', lw=0.8)

# ══════════════════════════════════════════════════════════════════════════
# MOBILE DEVICES  (col x=1.7)
# ══════════════════════════════════════════════════════════════════════════
UE_YS = [7.0, 5.0, 3.0]
for i, y in enumerate(UE_YS):
    box(1.7, y, 2.6, 0.85, UE_F, UE_E, [
        (f'UE {i+1}  —  Mobile Device', 9, True),
        (f'Task k{i+1}  |  Deadline ψ{i+1}', 7.5, False),
    ])

# ══════════════════════════════════════════════════════════════════════════
# MEC SERVERS  (col x=5.2)
# ══════════════════════════════════════════════════════════════════════════
SV_YS = [7.0, 5.0, 3.0]
for i, y in enumerate(SV_YS):
    box(5.2, y, 2.6, 0.85, ME_F, ME_E, [
        (f'MEC Server  c{i+1}', 9, True),
        (f'CPU f{i+1}  |  BW B{i+1}  |  Power p{i+1}', 7.5, False),
    ])

# ══════════════════════════════════════════════════════════════════════════
# BLOCKCHAIN  (col x=9.8,  full-height single box)
# ══════════════════════════════════════════════════════════════════════════
box(9.8, 5.0, 3.6, 5.8, BC_F, BC_E, [
    ('Consortium Blockchain', 10.5, True),
    ('PBFT Consensus', 9, False),
    ('', 1, False),
    ('• Resource Announcement Ledger', 8, False),
    ('• Allocation Decision Ledger', 8, False),
    ('• Task Result Audit Ledger', 8, False),
    ('• Smart Contract  &  Payment', 8, False),
    ('• Byzantine Fault Tolerance', 8, False),
])

# ══════════════════════════════════════════════════════════════════════════
# DDPG AGENT  (col x=13.9)
# ══════════════════════════════════════════════════════════════════════════
box(13.9, 5.0, 3.6, 5.8, RL_F, RL_E, [
    ('DDPG Agent', 10.5, True),
    ('Cross-Attention Actor', 9, False),
    ('', 1, False),
    ('• Actor Network', 8, False),
    ('• Critic Network  Q(s,a)', 8, False),
    ('• Replay Buffer  (s,a,r,s\')', 8, False),
    ('• Target Networks  τ=0.005', 8, False),
    ('• OU Noise Exploration', 8, False),
])

# ══════════════════════════════════════════════════════════════════════════
# ARROWS  — clean, minimal
# ══════════════════════════════════════════════════════════════════════════

# ① UE → MEC  (all 3 rows)
for y in UE_YS:
    harrow(3.0, y, 3.9, '', ec=UE_E, lw=1.6)
harrow(3.0, UE_YS[1], 3.9, '① Task Offload', ec=UE_E, above=True)

# ② MEC → Blockchain
for y in SV_YS:
    harrow(6.5, y+0.18, 8.0, '', ec=ME_E, lw=1.5)
harrow(6.5, SV_YS[0]+0.18, 8.0,
       '② Resource Announce', ec=ME_E, above=True)

# ⑤ Blockchain → MEC
for y in SV_YS:
    harrow(8.0, y-0.18, 6.5, '', ec=BC_E, lw=1.5)
harrow(8.0, SV_YS[2]-0.18, 6.5,
       '⑤ Task Assignment', ec=BC_E, above=False)

# ⑥ MEC → Blockchain  result (dashed, middle row only)
harrow(6.5, SV_YS[1], 8.0,
       '⑥ Result  (dashed)', ec='#999999', lw=1.3,
       dashed=True, above=False)

# ③ Blockchain → DDPG  state
harrow(11.6, 6.5, 12.1, '③ State s(t)', ec='#555555', above=True)

# ④ DDPG → Blockchain  action
harrow(12.1, 5.8, 11.6, '④ Action a(t)', ec=RL_E, above=False)

# ⑦ Blockchain → DDPG  reward (dashed)
harrow(11.6, 5.1, 12.1, '⑦ Reward r(t)', ec=BC_E,
       lw=1.3, dashed=True, above=False)

# ══════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════
handles = [
    mpatches.Patch(facecolor=UE_F, edgecolor=UE_E, linewidth=1.5,
                   label='Mobile Device (UE)'),
    mpatches.Patch(facecolor=ME_F, edgecolor=ME_E, linewidth=1.5,
                   label='MEC Server'),
    mpatches.Patch(facecolor=BC_F, edgecolor=BC_E, linewidth=1.5,
                   label='Blockchain (PBFT)'),
    mpatches.Patch(facecolor=RL_F, edgecolor=RL_E, linewidth=1.5,
                   label='DDPG Agent'),
]
ax.legend(handles=handles, loc='lower center', ncol=4,
          fontsize=9, frameon=True, edgecolor='#CCCCCC',
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(pad=0.3)
for ext in ('pdf', 'png'):
    out = f'D:/Masters Research Work/system_architecture.{ext}'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
