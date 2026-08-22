"""
Generate all paper figures with confidence intervals.
Run after: python experiments/run_all_baselines.py
Output: results/figures/*.png and results/figures/*.pdf
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "results", "figures"), exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})

COLORS = {
    "ddpg":      "#E8593C",
    "td3":       "#5C7CDE",
    "qlearning": "#A374DB",
    "greedy":    "#F4A623",
    "random":    "#73726C",
}
MARKERS = {"ddpg": "x", "td3": "D", "qlearning": "^", "greedy": "s", "random": "o"}
LABELS  = {"ddpg": "DDPG (Ours)", "td3": "TD3", "qlearning": "Q-Learning", "greedy": "Greedy", "random": "Random"}


def data_dir():
    return os.path.join(BASE_DIR, "results", "data")


def fig_dir():
    return os.path.join(BASE_DIR, "results", "figures")


def load_results(pattern):
    files = glob.glob(os.path.join(data_dir(), pattern))
    return [json.load(open(f)) for f in sorted(files)]


def ci(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return 0
    se = stats.sem(data)
    return se * stats.t.ppf((1 + confidence) / 2., n - 1)


def smooth(rewards, window=20):
    if len(rewards) < window:
        return np.array(rewards)
    return np.convolve(rewards, np.ones(window) / window, mode='valid')


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir(), f"{name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir(), f"{name}.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {name}")


def fig1_rewards_vs_servers():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    server_range = [10, 20, 30, 40, 50, 60]

    for agent in ["ddpg", "td3", "qlearning", "greedy", "random"]:
        means, cis = [], []
        for n_servers in server_range:
            results = load_results(f"{agent}_s{n_servers}_t50_seed*.json")
            if not results:
                means.append(np.nan); cis.append(0)
            else:
                vals = [r["mean_reward"] for r in results]
                means.append(np.mean(vals))
                cis.append(ci(vals))

        ax.errorbar(server_range, means, yerr=cis,
                    label=LABELS[agent], color=COLORS[agent],
                    marker=MARKERS[agent], capsize=4, capthick=1.5)

    ax.set_xlabel("No. of MEC Servers")
    ax.set_ylabel("Total Rewards")
    ax.set_title("Reward vs. Number of MEC Servers")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig1_rewards_vs_servers")


def fig2_rewards_vs_tasks():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    task_range = [50, 60, 70, 80, 90, 100]

    for agent in ["ddpg", "td3", "qlearning", "greedy", "random"]:
        means, cis = [], []
        for n_tasks in task_range:
            results = load_results(f"{agent}_s20_t{n_tasks}_seed*.json")
            if not results:
                means.append(np.nan); cis.append(0)
            else:
                vals = [r["mean_reward"] for r in results]
                means.append(np.mean(vals))
                cis.append(ci(vals))

        ax.errorbar(task_range, means, yerr=cis,
                    label=LABELS[agent], color=COLORS[agent],
                    marker=MARKERS[agent], capsize=4, capthick=1.5)

    ax.set_xlabel("No. of Offloading Tasks")
    ax.set_ylabel("Total Rewards")
    ax.set_title("Reward vs. Number of Offloading Tasks")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig2_rewards_vs_tasks")


def fig3_unit_price():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_labels = [f"{int(p*100)}%" for p in pcts]

    for agent in ["ddpg", "qlearning"]:
        rewards = []
        for pct in pcts:
            results = load_results(f"{agent}_unit_price_{pct:.1f}.json")
            if results:
                rewards.append(np.mean([r["mean_reward"] for r in results]))
            else:
                rewards.append(np.nan)
        ax.plot(range(len(pcts)), rewards, label=LABELS[agent],
                color=COLORS[agent], marker=MARKERS[agent])

    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_xlabel("Unit Price Increase (%)")
    ax.set_ylabel("Total Rewards")
    ax.set_title("Effect of Unit Price on Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig3_unit_price")


def fig4_data_size():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_labels = [f"{int(p*100)}%" for p in pcts]

    for agent in ["ddpg", "qlearning"]:
        rewards = []
        for pct in pcts:
            results = load_results(f"{agent}_data_size_{pct:.1f}.json")
            if results:
                rewards.append(np.mean([r["mean_reward"] for r in results]))
            else:
                rewards.append(np.nan)
        ax.plot(range(len(pcts)), rewards, label=LABELS[agent],
                color=COLORS[agent], marker=MARKERS[agent])

    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_xlabel("Data Size Increase (%)")
    ax.set_ylabel("Total Rewards")
    ax.set_title("Effect of Data Size on Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig4_data_size")


def fig5_end_to_end_delay():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    task_range = list(range(59, 145, 5))

    for agent in ["ddpg", "qlearning"]:
        delays = []
        for t in task_range:
            results = load_results(f"{agent}_delay_tasksize_{t}.json")
            if results:
                delays.append(np.mean([r["mean_delay"] for r in results]))
            else:
                delays.append(np.nan)
        ax.plot(task_range, delays, label=LABELS[agent],
                color=COLORS[agent], marker=MARKERS[agent])

    ax.set_xlabel("Task Size (Mbps)")
    ax.set_ylabel("Delay (s)")
    ax.set_title("End-to-End Delay: DDPG vs Q-Learning")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig5_delay")


def fig6_convergence_learning_rate():
    """
    Fig 6 per paper caption: 'The Learning rate α decreased by a factor of 0.01'
    Shows DDPG vs Q-Learning convergence at fixed α=0.01.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent in ["ddpg", "qlearning"]:
        results = load_results(f"{agent}_lr_0.01.json")
        if results:
            rewards = results[0]["rewards"]
            smoothed = smooth(rewards)
            ax.plot(smoothed, label=LABELS[agent], color=COLORS[agent])

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Utility")
    ax.set_title("Convergence: Learning Rate α = 0.01")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig6_convergence_lr")


def fig7_convergence_discount():
    """
    Fig 7 per paper caption: 'The Discount factor γ increased by a factor of 0.5'
    Shows DDPG vs Q-Learning convergence at fixed γ=0.5.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent in ["ddpg", "qlearning"]:
        results = load_results(f"{agent}_gamma_0.5.json")
        if results:
            rewards = results[0]["rewards"]
            smoothed = smooth(rewards)
            ax.plot(smoothed, label=LABELS[agent], color=COLORS[agent])

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Utility")
    ax.set_title("Convergence: Discount Factor γ = 0.5")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig7_convergence_gamma")


def fig6b_qlearning_convergence_lr():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for i, alpha in enumerate(alphas):
        results = load_results(f"qlearning_lr_{alpha}.json")
        if results:
            rewards = results[0]["rewards"]
            smoothed = smooth(rewards)
            ax.plot(smoothed, label=f"α={alpha}",
                    color=COLORS["qlearning"],
                    linestyle=line_styles[i % len(line_styles)],
                    alpha=0.85)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Utility")
    ax.set_title("Q-Learning Convergence — Varying Learning Rate α")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig6b_qlearning_convergence_lr")


def fig_convergence_comparison():
    """Training curve comparison: DDPG vs all baselines (default config)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for agent in ["ddpg", "td3", "qlearning", "greedy", "random"]:
        results = load_results(f"{agent}_s20_t50_seed42.json")
        if results:
            rewards = results[0]["episode_rewards"]
            smoothed = smooth(rewards, window=15)
            ax.plot(smoothed, label=LABELS[agent], color=COLORS[agent])

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Episode Reward (smoothed)")
    ax.set_title("Training Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    save_fig("fig_convergence_comparison")


def generate_results_table():
    table_rows = []
    for agent in ["ddpg", "td3", "qlearning", "greedy", "random"]:
        results = load_results(f"{agent}_s20_t50_seed*.json")
        if not results:
            continue
        rewards   = [r["mean_reward"]     for r in results]
        delays    = [r["mean_delay"]      for r in results]
        energies  = [r["mean_energy"]     for r in results]
        thru      = [r["mean_throughput"] for r in results]
        table_rows.append({
            "agent":      LABELS[agent],
            "reward":     f"{np.mean(rewards):.1f} ± {ci(rewards):.1f}",
            "delay":      f"{np.mean(delays):.4f} ± {ci(delays):.4f}",
            "energy":     f"{np.mean(energies):.4f} ± {ci(energies):.4f}",
            "throughput": f"{np.mean(thru):.3f} ± {ci(thru):.3f}",
        })

    latex = (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Comparative Performance Results (mean $\\pm$ 95\\% CI, 5 runs)}\n"
        "\\begin{tabular}{lcccc}\n\\hline\n"
        "Algorithm & Total Reward & Avg Delay (s) & Energy & Throughput \\\\\n\\hline\n"
    )
    for row in table_rows:
        latex += f"{row['agent']} & {row['reward']} & {row['delay']} & {row['energy']} & {row['throughput']} \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}"

    out = os.path.join(fig_dir(), "results_table.tex")
    with open(out, "w") as f:
        f.write(latex)
    print("Saved results_table.tex")
    return table_rows


if __name__ == "__main__":
    print("Generating figures...")
    fig1_rewards_vs_servers()
    fig2_rewards_vs_tasks()
    fig3_unit_price()
    fig4_data_size()
    fig5_end_to_end_delay()
    fig6_convergence_learning_rate()
    fig6b_qlearning_convergence_lr()
    fig7_convergence_discount()
    fig_convergence_comparison()
    rows = generate_results_table()

    print("\n=== Summary Table ===")
    print(f"{'Algorithm':<18} {'Reward':>20} {'Delay':>22} {'Throughput':>18}")
    print("-" * 80)
    for row in rows:
        print(f"{row['agent']:<18} {row['reward']:>20} {row['delay']:>22} {row['throughput']:>18}")

    print(f"\nAll figures saved to {os.path.join('results', 'figures')}/")
