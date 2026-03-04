#!/usr/bin/env python3
"""
Presidential Endorsement Rate (Agreeableness).

How often do the *other* players vote Yes for Alice's (Player 0)
chancellor nomination when she is president?

Broken down by the role of the voting player:
  - All  (every other player)
  - Liberal voters
  - Fascist voters
  - Hitler voters

Usage:
    python agreeableness.py                        # compare all model folders
    python agreeableness.py runsF2-GEMMA           # single model folder
    python agreeableness.py --crawl                # include human data (crawl/)
    python agreeableness.py --include-abliterated  # include abliterated/derestricted models
"""

import argparse
import glob
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.ndimage import rotate as rotate_image

import plot_config

plot_config.setup_plot_style()

EVAL_DIR = Path(__file__).parent
ALICE_ID = 0  # Alice is always Player 0

# Columns = Alice's role as president
ALICE_ROLES = ["all", "liberal", "fascist", "hitler"]
ALICE_ROLE_LABELS = {
    "all":     "All",
    "liberal": "Liberal",
    "fascist": "Fascist",
    "hitler":  "Hitler",
}
ALICE_ROLE_COLORS = {
    "all":     plot_config.LOGOBLAU,
    "liberal": plot_config.ROLE_COLORS["liberal"],
    "fascist": plot_config.ROLE_COLORS["fascist"],
    "hitler":  plot_config.ROLE_COLORS["hitler"],
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_endorsement_rate(folder: Path) -> dict | None:
    """
    Compute presidential endorsement rates for all games in *folder*.

    Returns a dict keyed by Alice's role as president
    ("all", "liberal", "fascist", "hitler"), each value being a dict
    {"yes": int, "total": int, "rate": float}.
    Returns None if no valid rounds were found.
    """
    json_files = list(folder.glob("*.json"))
    if not json_files:
        return None

    counts = {role: {"yes": 0, "total": 0} for role in ALICE_ROLES}

    for fpath in json_files:
        summary = plot_config.load_summary_file(fpath)
        if summary is None:
            continue

        players = summary.get("players", [])
        logs = summary.get("logs", [])
        if not players or not logs:
            continue

        # Alice's role in this game
        alice_role = players[ALICE_ID].get("role", "unknown") if len(players) > ALICE_ID else "unknown"
        n_players = len(players)

        for round_data in logs:
            pres_id = round_data.get("presidentId")
            votes = round_data.get("votes")

            # Only rounds where Alice is president and votes exist
            if pres_id != ALICE_ID:
                continue
            if not votes or not isinstance(votes, list):
                continue

            # Count yes-votes from every other player
            yes_count = 0
            total_count = 0
            for voter_id in range(n_players):
                if voter_id == ALICE_ID:
                    continue
                if voter_id >= len(votes):
                    continue
                vote = votes[voter_id]
                if vote is None:
                    continue  # abstain / not voting this round
                total_count += 1
                if bool(vote):
                    yes_count += 1

            if total_count == 0:
                continue

            # Accumulate into "all" and into Alice's specific role bucket
            counts["all"]["total"] += total_count
            counts["all"]["yes"] += yes_count
            if alice_role in ("liberal", "fascist", "hitler"):
                counts[alice_role]["total"] += total_count
                counts[alice_role]["yes"] += yes_count

    # Build result only if we have at least some data
    if counts["all"]["total"] == 0:
        return None

    result = {}
    for role in ALICE_ROLES:
        total = counts[role]["total"]
        yes = counts[role]["yes"]
        result[role] = {
            "yes": yes,
            "total": total,
            "rate": yes / total if total > 0 else float("nan"),
        }
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_x_positions(n_models: int, n_groups: int = 4, gap: float = 0.25):
    """
    Return (group_centers, bar_offsets, bar_width) for a grouped bar chart.
    """
    bar_width = (1.0 - gap) / n_models
    offsets = np.arange(n_models) * bar_width - (n_models - 1) * bar_width / 2
    group_centers = np.arange(n_groups, dtype=float)
    return group_centers, offsets, bar_width


def plot_comparison(model_data: dict[str, dict]):
    """
    Produce a grouped bar chart comparing endorsement rates across models.

    model_data: {model_display_name: {role: {"rate": float, "total": int}}}
    """
    models = list(model_data.keys())
    n_models = len(models)
    n_groups = len(ALICE_ROLES)   # 4 groups: all, liberal, fascist, hitler

    group_centers, offsets, bar_width = _bar_x_positions(n_models, n_groups)

    fig, ax = plt.subplots(figsize=(plot_config.FIG_WIDTH, 3.2))

    for i, model_name in enumerate(models):
        data = model_data[model_name]
        color = plot_config.get_model_color(model_name)
        rates = [data[role]["rate"] * 100 if not np.isnan(data[role]["rate"]) else 0
                 for role in ALICE_ROLES]
        totals = [data[role]["total"] for role in ALICE_ROLES]

        x_positions = group_centers + offsets[i]
        bars = ax.bar(
            x_positions,
            rates,
            width=bar_width * 0.9,
            color=color,
            label=model_name,
            alpha=0.85,
            zorder=3,
        )


    # Reference lines
    ax.axhline(50, color="0.6", linewidth=0.8, linestyle="--", zorder=2)

    # Axes decoration
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [ALICE_ROLE_LABELS[r] for r in ALICE_ROLES],
    )
    ax.set_ylabel("Yes-vote rate (\\%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Colour-coded x-tick labels to match role colours
    for tick, role in zip(ax.get_xticklabels(), ALICE_ROLES):
        tick.set_color(ALICE_ROLE_COLORS[role])

    # Legend at the bottom
    legend = ax.legend(
        framealpha=0,
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        handlelength=2,
        handletextpad=1.6,
        ncol=2,
    )

    for model_name, legend_handle in zip(models, legend.legend_handles):
        imagebox = plot_config.get_model_imagebox(model_name)
        if imagebox is None:
            continue
        ab = AnnotationBbox(
            imagebox,
            (0.5, 0.5),
            xycoords=legend_handle,
            frameon=False,
            clip_on=False,
            box_alignment=(-1.5, 0.5),
            zorder=10,
        )
        fig.add_artist(ab)

    out_path = plot_config.get_plot_path("agreeableness.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_single(model_name: str, data: dict):
    """
    Simple bar chart for a single model, coloured by voter role.
    """
    fig, ax = plt.subplots(figsize=(plot_config.FIG_WIDTH * 0.65, 3.0))

    x = np.arange(len(ALICE_ROLES))
    rates = [data[role]["rate"] * 100 if not np.isnan(data[role]["rate"]) else 0
             for role in ALICE_ROLES]
    totals = [data[role]["total"] for role in ALICE_ROLES]
    colors = [ALICE_ROLE_COLORS[r] for r in ALICE_ROLES]

    bars = ax.bar(x, rates, color=colors, alpha=0.85, zorder=3)

    for bar, total, rate in zip(bars, totals, rates):
        if total > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}\\%\nn={total}",
                ha="center",
                va="bottom",
            )

    ax.axhline(50, color="0.6", linewidth=0.8, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([ALICE_ROLE_LABELS[r] for r in ALICE_ROLES])
    for tick, role in zip(ax.get_xticklabels(), ALICE_ROLES):
        tick.set_color(ALICE_ROLE_COLORS[role])

    ax.set_ylabel("Yes-vote rate (\\%)")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "-")
    out_path = plot_config.get_plot_path(f"agreeableness_{safe_name}.pdf")
    fig.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(model_data: dict[str, dict]):
    col_w = 12
    header = f"{'Model':<32}" + "".join(f"{ALICE_ROLE_LABELS[r]:>{col_w}}" for r in ALICE_ROLES)
    print()
    print("=" * len(header))
    print("PRESIDENTIAL ENDORSEMENT RATE  (yes-vote % from other players, by Alice's role)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, data in model_data.items():
        row = f"{model_name:<32}"
        for role in ALICE_ROLES:
            rate = data[role]["rate"]
            total = data[role]["total"]
            if np.isnan(rate) or total == 0:
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{rate*100:>{col_w-4}.1f}% (n={total:{'>'}3})" if False else \
                       f"{rate*100:>{col_w}.1f}"
        print(row)
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Presidential Endorsement Rate: how often others vote Yes "
                    "for Alice's nominations when she is president."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Single model folder to analyse (e.g. runsF2-GEMMA). "
             "Omit to compare all model folders.",
    )
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Include the crawl/ (human) folder.",
    )
    parser.add_argument(
        "--include-abliterated",
        action="store_true",
        help="Include abliterated / derestricted model variants.",
    )
    args = parser.parse_args()

    model_data: dict[str, dict] = {}

    if args.folder:
        # ---- Single-folder mode ----
        folder = EVAL_DIR / args.folder
        if not folder.is_dir():
            print(f"Error: folder not found: {folder}")
            return
        data = compute_endorsement_rate(folder)
        if data is None:
            print(f"No valid presidential rounds found in {folder}")
            return
        model_name = plot_config.extract_model_name(args.folder)
        model_data[model_name] = data
        print_table(model_data)
        plot_single(model_name, data)
    else:
        # ---- Multi-folder mode: iterate over MODEL_REGISTRY order ----
        candidate_keys = list(plot_config.MODEL_REGISTRY.keys())
        if not args.crawl:
            candidate_keys = [k for k in candidate_keys if k != "crawl"]
        if not args.include_abliterated:
            candidate_keys = [
                k for k in candidate_keys
                if not plot_config.MODEL_REGISTRY[k].get("abliterated", False)
            ]

        for key in candidate_keys:
            folder = EVAL_DIR / key
            if not folder.is_dir():
                continue
            data = compute_endorsement_rate(folder)
            if data is None:
                continue
            model_name = plot_config.extract_model_name(key)
            model_data[model_name] = data

        if not model_data:
            print("No data found. Run from the eval/ directory.")
            return

        print_table(model_data)
        plot_comparison(model_data)


if __name__ == "__main__":
    main()
