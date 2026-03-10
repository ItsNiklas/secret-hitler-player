#!/usr/bin/env python3
"""
Plot 2 – Approval Rate Line Chart (per-round resolution).

X-axis is round number (1, 2, 3, …), y-axis is Alice's yes-vote rate.
Each model is a distinct line.  Rounds are cut off when fewer than 10 %%
of games reach that round.

Models sorted by overall win rate; baselines below a divider in the
legend.  Models with < MIN_GAMES games are dropped.

Usage:
    python approval_line_chart.py                        # all non-abliterated
    python approval_line_chart.py --include-abliterated  # include abliterated
"""

import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import plot_config
from plot_config import (
    FIG_WIDTH,
    MIN_GAMES,
    setup_plot_style,
    extract_model_name,
    get_model_color,
    get_markerdata_for_model,
    get_plot_path,
    load_summary_file,
    collect_model_keys,
    compute_win_rate,
    sort_models_by_winrate,
)

setup_plot_style()

EVAL_DIR = Path(__file__).parent


# ------------------------------------------------------------------
# Per-round approval rate computation
# ------------------------------------------------------------------

def compute_per_round_rates(folder: Path) -> dict | None:
    """
    Return {round_num: {"yes": int, "total": int, "rate": float}} or None.
    Round numbers are 1-based.
    """
    json_files = list(folder.glob("*.json"))
    if len(json_files) < MIN_GAMES:
        return None

    counts = defaultdict(lambda: {"yes": 0, "total": 0})
    n_games = 0

    for fpath in json_files:
        summary = load_summary_file(fpath)
        if summary is None:
            continue
        logs = summary.get("logs", [])
        if not logs:
            continue
        n_games += 1
        for round_idx, round_data in enumerate(logs):
            votes = round_data.get("votes")
            if not votes or not isinstance(votes, list):
                continue
            if 0 >= len(votes) or votes[0] is None:
                continue
            rnd = round_idx + 1
            counts[rnd]["total"] += 1
            if bool(votes[0]):
                counts[rnd]["yes"] += 1

    if n_games < MIN_GAMES:
        return None

    # Find a reasonable cutoff: last round where ≥10 % of games reach it
    result = {}
    for rnd in sorted(counts.keys()):
        total = counts[rnd]["total"]
        if total < max(3, n_games * 0.10):
            break  # stop here
        yes = counts[rnd]["yes"]
        result[rnd] = {"yes": yes, "total": total, "rate": yes / total}
    return result if result else None


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_approval_lines(round_data: dict, baseline_names: set):
    """
    Single-panel line chart: approval rate by round.
    """
    models = list(round_data.keys())
    n = len(models)
    if n == 0:
        print("No data to plot.")
        return

    # Determine global max round
    max_round = max(max(rd.keys()) for rd in round_data.values()) - 5

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.8))

    for model in models:
        rd = round_data[model]
        rounds = sorted(rd.keys())
        rates = [rd[r]["rate"] * 100 for r in rounds]
        m_style, ms = get_markerdata_for_model(model)
        ax.plot(
            rounds, rates,
            marker=m_style, color=get_model_color(model),
            linewidth=1.0, markersize=ms * 0.65,
            markeredgecolor="white", markeredgewidth=0.4,
            label=model, zorder=3,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(r"Approval rate (\%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0.5, max_round + 0.5)
    ax.set_xticks(range(1, max_round + 1))
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend below
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.6),
        ncol=3,
        framealpha=0,
        handlelength=2,
        handletextpad=0.8,
        columnspacing=1.0,
    )

    out = get_plot_path("approval_rate_line.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Approval-rate per-round line chart")
    parser.add_argument("--include-abliterated", action="store_true")
    args = parser.parse_args()

    keys = collect_model_keys(include_abliterated=args.include_abliterated)

    round_data = {}
    win_rates = {}

    for key in keys:
        folder = EVAL_DIR / key
        if not folder.is_dir():
            continue
        name = extract_model_name(key)

        rd = compute_per_round_rates(folder)
        if rd is None:
            continue
        wr = compute_win_rate(folder)
        if wr is None:
            continue

        round_data[name] = rd
        win_rates[name] = wr
        max_r = max(rd.keys())
        print(f"{name:30s}  rounds 1-{max_r}  WR={wr:.0f}%")

    if round_data:
        ordered, bl = sort_models_by_winrate(round_data, win_rates)
        plot_approval_lines(ordered, bl)


if __name__ == "__main__":
    main()
