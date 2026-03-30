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
from matplotlib.offsetbox import AnnotationBbox

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
    ordered_models = list(round_data.keys())
    n = len(ordered_models)
    if n == 0:
        print("No data to plot.")
        return

    # Determine global max round
    max_round = 10

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.3))
    lines = []

    for model in ordered_models:
        rd = round_data[model]
        rounds = sorted(rd.keys())
        rates = [rd[r]["rate"] * 100 for r in rounds]
        m, ms = get_markerdata_for_model(model)
        (line,) = ax.plot(
            rounds, rates,
            marker=m, color=get_model_color(model),
            linewidth=2, markersize=ms, label=model,
            markeredgecolor="white", markeredgewidth=1,
        )
        lines.append((model, line))

    ax.set_xlabel("")
    ax.set_ylabel(r"Approval rate (\%)")
    ax.grid(True, alpha=0.4)
    ax.set_ylim(None, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: fr"{int(y)}\%"))
    ax.set_xticks(range(1, max_round + 1))
    ax.set_xlim(0.85, max_round - 0.6)

    # Place "Round" label to the left of tick "1" to save vertical space
    ax.annotate("Round", xy=(1, 0), xycoords=("data", "axes fraction"),
                xytext=(-15, -7), textcoords="offset points",
                ha="right", va="top", fontsize=plt.rcParams["axes.labelsize"])

    legend = ax.legend(
        framealpha=0,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        handlelength=0,
        handletextpad=1.9,
        ncol=1,
    )

    # Add model icons to legend
    for model, handle in zip([m for m, _ in lines], legend.legend_handles):
        imagebox = plot_config.get_model_imagebox(model)
        if imagebox is not None:
            imagebox.set_zoom(imagebox.get_zoom() * 0.8)
            ab = AnnotationBbox(
                imagebox, (0.5, 0.5), xybox=(10, 0),
                xycoords=handle, boxcoords="offset points",
                frameon=False, box_alignment=(0.5, 0.5), zorder=10,
            )
            fig.add_artist(ab)

    out = get_plot_path("approval_rate_line.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {out}")
    plt.close(fig)


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Approval-rate per-round line chart")
    parser.add_argument("--include-abliterated", action="store_true")
    args = parser.parse_args()

    keys = collect_model_keys(include_abliterated=args.include_abliterated, include_baselines=False)

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
