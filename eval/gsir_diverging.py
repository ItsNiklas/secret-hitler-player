#!/usr/bin/env python3
"""
Plot 1 – GSIR Horizontal Diverging Bar Chart (Table 2).

For each model, computes the Game-State Impact Rating (GSIR) broken down by
Alice's role (Liberal, Fascist, Hitler).  Each role gets its own sub-bar so
that a negative Liberal GSIR is never hidden behind a Fascist bar.

Models are sorted by overall win rate; baselines appear below a divider.
Models with fewer than MIN_GAMES games are dropped.

Usage:
    python gsir_diverging.py                          # all non-abliterated models
    python gsir_diverging.py --include-abliterated    # include abliterated variants
    python gsir_diverging.py runsF2-GEMMA             # single model
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.offsetbox import AnnotationBbox

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
from gamestats import analyze_alice_game_state_impact

setup_plot_style()

EVAL_DIR = Path(__file__).parent

# Three sub-rows per model, tightly packed
SUB_ROLES = ["liberal", "fascist", "hitler"]
SUB_HEIGHT = 0.18  # height of each thin bar
SUB_SPACING = 0.22  # vertical distance between sub-bar centres
GROUP_GAP = 0.30  # extra gap between model groups


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def compute_gsir_for_folder(folder: Path):
    games = load_games_from_folder(folder)
    if len(games) < MIN_GAMES:
        return None
    impact = analyze_alice_game_state_impact(games)
    if impact["total_actions"] == 0:
        return None
    return {
        role: impact["cumulative_mean_by_role"].get(role, 0.0) * 100
        for role in SUB_ROLES
    }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_gsir_diverging(model_data: dict, baseline_names: set):
    models = list(model_data.keys())
    n = len(models)
    if n == 0:
        print("No data to plot.")
        return

    # Build y-positions: 3 sub-rows per model, extra gap before baselines
    y_positions = {}
    y_current = 0.0
    first_baseline = True
    baseline_divider_y = None
    for m in models:
        if m in baseline_names and first_baseline:
            baseline_divider_y = y_current - GROUP_GAP * 0.5
            y_current += GROUP_GAP
            first_baseline = False
        y_positions[m] = y_current
        y_current += SUB_SPACING * (len(SUB_ROLES) - 1) + GROUP_GAP

    total_height = y_current
    fig_height = max(4, total_height * 0.12 + 0.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    for model in models:
        vals = model_data[model]
        centre = y_positions[model]
        for j, role in enumerate(SUB_ROLES):
            y = centre + (j - 1) * SUB_SPACING
            ax.barh(
                y, vals[role], height=SUB_HEIGHT,
                color=ROLE_COLORS[role], zorder=3,
                label=role.capitalize() if model == models[0] else None,
            )

    # y-axis: one tick per model at centre
    ax.set_yticks([y_positions[m] for m in models])
    ax.set_yticklabels(models)
    ax.invert_yaxis()

    # Horizontal divider before baselines
    if baseline_divider_y is not None:
        ax.axhline(baseline_divider_y, color="0.65", linewidth=0.6,
                   linestyle="--", zorder=1)

    ax.axvline(0, color="0.4", linewidth=0.8, zorder=5)
    ax.set_xlabel(r"Cumulative GSIR (centiscore per game)")

    def _fmt(x, _):
        if abs(x) < 1e-9:
            return "0"
        return f"{x:+.1f}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Logos
    has_icon = any(get_model_imagebox(m) is not None for m in models)
    if has_icon:
        ax.tick_params(axis="y", pad=12)
    for m in models:
        ib = get_model_imagebox(m)
        if ib is not None:
            ab = AnnotationBbox(
                ib, xy=(0, y_positions[m]),
                xycoords=("axes fraction", "data"),
                xybox=(-8, 0), boxcoords="offset points",
                frameon=False, box_alignment=(0.5, 0.5), zorder=10,
            )
            ax.add_artist(ab)

    ax.legend(loc="lower right", framealpha=0.9, ncol=3)

    out = get_plot_path("gsir_diverging.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GSIR Diverging Bar Chart")
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
        gsir = compute_gsir_for_folder(folder)
        if gsir is None:
            continue
        wr = compute_win_rate(folder)
        if wr is None:
            continue
        model_data[name] = gsir
        win_rates[name] = wr
        print(f"{name:30s}  lib={gsir['liberal']:+.2f}  fas={gsir['fascist']:+.2f}  hit={gsir['hitler']:+.2f}  WR={wr:.0f}%")

    if model_data:
        ordered, bl = sort_models_by_winrate(model_data, win_rates)
        plot_gsir_diverging(ordered, bl)


if __name__ == "__main__":
    main()
