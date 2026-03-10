#!/usr/bin/env python3
"""
Plot 5 – Abliteration Dumbbell / Paired-Dot Plot (Table 5 deltas).

For each standard → abliterated model pair, two dots connected by a
horizontal line show the performance shift on key metrics (Win Rate,
Vote Accuracy, GSIR).

Usage:
    python abliteration_dumbbell.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_config
from plot_config import (
    FIG_WIDTH,
    setup_plot_style,
    extract_model_name,
    get_plot_path,
    load_summary_file,
    load_games_from_folder,
)
from gamestats import analyze_alice_performance, analyze_alice_game_state_impact

setup_plot_style()

EVAL_DIR = Path(__file__).parent

# (standard_folder, abliterated_folder, display_label)
PAIRS = [
    ("runsF2-GEMMA",        "runsF2-AMORALGEMMA",              "Gemma 3 27B"),
    ("runsF2-GPTOSS120B",   "runsF2-GPTOSS120B-DERESTRICTED",  "GPT-OSS 120B"),
    ("runsF2-MISTRALSMALL", "runsF2-DOLPHINVENICE",            "Mistral Small 24B"),
    ("runsF2-LLAMA3170B",   "runsF2-NOUSHERMES4",              "Llama 3.1 70B"),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _vote_accuracy(folder: Path) -> float:
    json_files = list(folder.glob("*.json"))
    total_inst = total_succ = 0
    for fpath in json_files:
        summary = load_summary_file(fpath)
        if summary is None:
            continue
        players = summary.get("players", [])
        logs = summary.get("logs", [])
        if not players or not logs:
            continue
        roles = {i: p["role"] for i, p in enumerate(players)}
        fp = 0
        executed = set()
        for rd in logs:
            votes = rd.get("votes", [])
            pid = rd.get("presidentId")
            cid = rd.get("chancellorId")
            if not votes or pid is None or cid is None:
                if rd.get("enactedPolicy") == "fascist":
                    fp += 1
                if "execution" in rd:
                    executed.add(rd["execution"])
                continue
            in_hz = fp >= 3
            dangerous = roles.get(pid) in ("fascist", "hitler") or roles.get(cid) == "hitler"
            if in_hz and dangerous:
                if 0 < len(votes) and votes[0] is not None and 0 not in executed and roles.get(0) == "liberal":
                    total_inst += 1
                    if votes[0] is False:
                        total_succ += 1
            if rd.get("enactedPolicy") == "fascist":
                fp += 1
            if "execution" in rd:
                executed.add(rd["execution"])
    return total_succ / total_inst * 100 if total_inst else float("nan")


def compute_metrics(folder: Path) -> dict[str, float] | None:
    games = load_games_from_folder(folder)
    if not games:
        return None
    perf = analyze_alice_performance(games)
    impact = analyze_alice_game_state_impact(games)
    win_rate = perf["win_rate"]
    gsir = impact["cumulative_mean"] * 100 if impact["total_actions"] else float("nan")
    vacc = _vote_accuracy(folder)
    return {"Win Rate": win_rate, "Vote Accuracy": vacc, "GSIR": gsir}


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

METRICS = ["Win Rate", "Vote Accuracy", "GSIR"]


def plot_dumbbell(pair_data):
    n_pairs = len(pair_data)
    n_metrics = len(METRICS)
    if n_pairs == 0:
        print("No pairs to plot.")
        return

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(FIG_WIDTH, max(1.4, 0.25 * n_pairs + 0.5)),
        sharey=True,
    )
    if n_metrics == 1:
        axes = [axes]

    y_pos = np.arange(n_pairs)
    pair_labels = [p[0] for p in pair_data]

    standard_color = plot_config.LOGOBLAU
    abliterated_color = "#D95F02"

    for col_idx, metric in enumerate(METRICS):
        ax = axes[col_idx]
        for i, (label, std_m, abl_m) in enumerate(pair_data):
            v_std = std_m.get(metric, float("nan"))
            v_abl = abl_m.get(metric, float("nan"))
            if np.isnan(v_std) or np.isnan(v_abl):
                continue

            ax.plot(
                [v_std, v_abl], [y_pos[i], y_pos[i]],
                color="0.65", linewidth=1.5, zorder=2,
            )
            ax.scatter(
                v_std, y_pos[i], s=40, color=standard_color,
                edgecolors="white", linewidths=0.4, zorder=4,
                label="Standard" if i == 0 and col_idx == 0 else None,
            )
            ax.scatter(
                v_abl, y_pos[i], s=40, color=abliterated_color,
                edgecolors="white", linewidths=0.4, zorder=4, marker="D",
                label="Abliterated" if i == 0 and col_idx == 0 else None,
            )

            delta = v_abl - v_std
            mid_x = (v_std + v_abl) / 2
            sign = "+" if delta >= 0 else ""
            if metric == "GSIR":
                delta_text = f"{sign}{delta:.1f}"
            else:
                delta_text = f"{sign}{delta:.0f}pp"
            ax.text(
                mid_x, y_pos[i] - 0.4, delta_text,
                ha="center", va="top", fontsize=7, color="0.4",
            )

        ax.set_xlabel(metric + (r" (\%)" if metric != "GSIR" else " (cs)"))
        ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col_idx == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels)
            ax.invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, -0.36),
            ncol=2, framealpha=0, handletextpad=0.5,
        )

    out = get_plot_path("abliteration_dumbbell.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ------------------------------------------------------------------

def main():
    pair_results = []
    for std_key, abl_key, label in PAIRS:
        std_folder = EVAL_DIR / std_key
        abl_folder = EVAL_DIR / abl_key
        if not std_folder.is_dir() or not abl_folder.is_dir():
            print(f"Skipping {label}: folder(s) missing")
            continue

        std_m = compute_metrics(std_folder)
        abl_m = compute_metrics(abl_folder)
        if std_m is None or abl_m is None:
            print(f"Skipping {label}: no data")
            continue

        print(f"{label}")
        for metric in METRICS:
            sv = std_m.get(metric, float("nan"))
            av = abl_m.get(metric, float("nan"))
            delta = av - sv if not (np.isnan(sv) or np.isnan(av)) else float("nan")
            unit = "cs" if metric == "GSIR" else "%"
            print(f"  {metric:16s}  std={sv:6.1f}{unit}  abl={av:6.1f}{unit}  delta={delta:+.1f}")

        pair_results.append((label, std_m, abl_m))

    if pair_results:
        plot_dumbbell(pair_results)


if __name__ == "__main__":
    main()
