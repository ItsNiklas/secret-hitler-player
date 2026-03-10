#!/usr/bin/env python3
"""
Plot 4 – Scatter Plot of Overall GSIR vs. Overall Win Rate.

Each dot is one model.  A linear trendline is overlaid together with
the Pearson correlation coefficient.

Baselines are excluded.  Models with < MIN_GAMES games are dropped.

Usage:
    python gsir_vs_winrate.py                        # non-abliterated models
    python gsir_vs_winrate.py --include-abliterated  # include abliterated
"""

import argparse
from pathlib import Path

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
    load_games_from_folder,
    collect_model_keys,
)
from gamestats import analyze_alice_performance, analyze_alice_game_state_impact

setup_plot_style()

EVAL_DIR = Path(__file__).parent


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def compute_gsir_and_winrate(folder: Path):
    games = load_games_from_folder(folder)
    if len(games) < MIN_GAMES:
        return None
    impact = analyze_alice_game_state_impact(games)
    if impact["total_actions"] == 0:
        return None
    overall_gsir = impact["cumulative_mean"] * 100
    perf = analyze_alice_performance(games)
    if perf["total_games"] == 0:
        return None
    return overall_gsir, perf["win_rate"]


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_scatter(data: dict[str, tuple[float, float]]):
    models = list(data.keys())
    xs = np.array([data[m][0] for m in models])
    ys = np.array([data[m][1] for m in models])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))

    for m, x, y in zip(models, xs, ys):
        m_style, ms = get_markerdata_for_model(m)
        ax.scatter(
            x, y,
            marker=m_style, s=ms ** 2,
            color=get_model_color(m),
            edgecolors="white", linewidths=0.4,
            zorder=5,
        )

    # Trendline
    if len(xs) >= 3:
        coeffs = np.polyfit(xs, ys, 1)
        r = np.corrcoef(xs, ys)[0, 1]
        x_line = np.linspace(xs.min() - 2, xs.max() + 2, 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, "--", color="0.55", linewidth=0.9, zorder=2)
        ax.text(
            0.03, 0.96, f"$r = {r:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="0.85",
                      boxstyle="round,pad=0.3"),
        )

    ax.set_xlabel(r"Overall GSIR (centiscore per game)")
    ax.set_ylabel(r"Overall win rate (\%)")
    ax.grid(True, alpha=0.25, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate points
    for m, x, y in zip(models, xs, ys):
        ax.annotate(
            m, (x, y),
            textcoords="offset points", xytext=(3, -8),
            ha="left", va="bottom", fontsize=6,
        )

    out = get_plot_path("gsir_vs_winrate.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GSIR vs Win Rate scatter plot")
    parser.add_argument("--include-abliterated", action="store_true")
    args = parser.parse_args()

    # No baselines for this plot
    keys = collect_model_keys(
        include_abliterated=args.include_abliterated,
        include_baselines=False,
    )

    scatter_data: dict[str, tuple[float, float]] = {}
    for key in keys:
        folder = EVAL_DIR / key
        if not folder.is_dir():
            continue
        name = extract_model_name(key)
        result = compute_gsir_and_winrate(folder)
        if result is not None:
            gsir, wr = result
            scatter_data[name] = (gsir, wr)
            print(f"{name:30s}  GSIR={gsir:+.2f}  WinRate={wr:.1f}%")

    if scatter_data:
        plot_scatter(scatter_data)


if __name__ == "__main__":
    main()
