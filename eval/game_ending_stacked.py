#!/usr/bin/env python3
"""
Plot 3 – Game-Ending Conditions: 100 %% Stacked Horizontal Bar Chart (Table 6).

Each model gets one horizontal bar totalling 100 %% of its games, divided into
four coloured segments for the four win conditions.

Sorted by overall win rate; baselines below a divider.
Models with < MIN_GAMES games are dropped.

Usage:
    python game_ending_stacked.py                        # all non-abliterated
    python game_ending_stacked.py --include-abliterated  # include abliterated
    python game_ending_stacked.py runsF2-GEMMA           # single model
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.offsetbox import AnnotationBbox

import plot_config
from plot_config import (
    FIG_WIDTH,
    MIN_GAMES,
    ROLE_COLORS,
    setup_plot_style,
    extract_model_name,
    get_model_imagebox,
    get_plot_path,
    load_games_from_folder,
    compute_win_rate,
    collect_model_keys,
    sort_models_by_winrate,
)
from gamestats import analyze_win_conditions

setup_plot_style()

EVAL_DIR = Path(__file__).parent
BAR_HEIGHT = 0.50

CONDITIONS = ["liberal_policies", "hitler_killed", "hitler_chancellor", "fascist_policies"]
COND_LABELS = {
    "liberal_policies":  "Liberal policies (5 enacted)",
    "hitler_killed":     "Hitler killed",
    "hitler_chancellor": "Hitler elected chancellor",
    "fascist_policies":  "Fascist policies (6 enacted)",
}
COND_COLORS = {
    "liberal_policies":  ROLE_COLORS["liberal"],
    "hitler_killed":     "#8FAEC1",
    "hitler_chancellor": ROLE_COLORS["hitler"],
    "fascist_policies":  ROLE_COLORS["fascist"],
}


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def compute_ending_fractions(folder: Path):
    games = load_games_from_folder(folder)
    if len(games) < MIN_GAMES:
        return None
    analysis = analyze_win_conditions(games)
    total = analysis["total_games"]
    if total == 0:
        return None
    return {c: analysis["win_conditions"].get(c, 0) / total * 100 for c in CONDITIONS}


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_stacked_endings(model_data: dict, baseline_names: set):
    models = list(model_data.keys())
    n = len(models)
    if n == 0:
        print("No data to plot.")
        return

    fig_height = max(1.6, 0.25 * n + 0.6)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    y_pos = np.arange(n)

    # Detect baseline divider position
    baseline_divider_idx = None
    for i, m in enumerate(models):
        if m in baseline_names:
            baseline_divider_idx = i
            break

    # Build stacked bars
    lefts = np.zeros(n)
    for cond in CONDITIONS:
        widths = np.array([model_data[m][cond] for m in models])
        ax.barh(
            y_pos, widths, left=lefts, height=BAR_HEIGHT,
            color=COND_COLORS[cond], label=COND_LABELS[cond], zorder=3,
        )
        for i, (w, l) in enumerate(zip(widths, lefts)):
            if w >= 8:
                ax.text(
                    l + w / 2, y_pos[i] + 0.04, f"{w:.0f}\\%",
                    ha="center", va="center", color="white",
                    fontweight="bold", zorder=4, fontsize=7,
                )
        lefts += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"Share of games (\%)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}\\%"))
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Baseline divider
    if baseline_divider_idx is not None:
        ax.axhline(baseline_divider_idx - 0.5, color="0.65", linewidth=0.6,
                   linestyle="--", zorder=1)

    # Logos
    has_icon = any(get_model_imagebox(m) is not None for m in models)
    if has_icon:
        ax.tick_params(axis="y", pad=12)
    for i, m in enumerate(models):
        ib = get_model_imagebox(m)
        if ib is not None:
            ab = AnnotationBbox(
                ib, xy=(0, y_pos[i]),
                xycoords=("axes fraction", "data"),
                xybox=(-8, 0), boxcoords="offset points",
                frameon=False, box_alignment=(0.5, 0.5), zorder=10,
            )
            ax.add_artist(ab)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2, framealpha=0,
        handlelength=1.5, handletextpad=0.6,
        columnspacing=1.2,
    )

    out = get_plot_path("game_ending_stacked.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Game-ending conditions stacked bar")
    parser.add_argument("folder", nargs="?", default=None)
    parser.add_argument("--include-abliterated", action="store_true")
    args = parser.parse_args()

    if args.folder:
        keys = [args.folder]
    else:
        keys = collect_model_keys(include_abliterated=args.include_abliterated)

    model_data = {}
    win_rates = {}
    for key in keys:
        folder = EVAL_DIR / key
        if not folder.is_dir():
            continue
        name = extract_model_name(key)
        fracs = compute_ending_fractions(folder)
        if fracs is None:
            continue
        wr = compute_win_rate(folder)
        if wr is None:
            continue
        model_data[name] = fracs
        win_rates[name] = wr
        parts = "  ".join(f"{c[:3]}={fracs[c]:.0f}%" for c in CONDITIONS)
        print(f"{name:30s}  {parts}  WR={wr:.0f}%")

    if model_data:
        ordered, bl = sort_models_by_winrate(model_data, win_rates)
        plot_stacked_endings(ordered, bl)


if __name__ == "__main__":
    main()
