#!/usr/bin/env python3
"""
Token usage aggregator for Secret Hitler evaluation runs.
Only counts tokens used by Alice (Player 0).

Reads game filenames from a runs folder, finds corresponding raw token-stats
JSONL files (matched by closest timestamp within tolerance), filters to Alice's
requests only, and prints per-game averages overall and broken down by stage.

Each token-stats file is matched to at most one game (greedy closest-first
assignment) so that no file is double-counted.

Usage:
  python aggregate_tokens.py <eval_dir>   — report for one run
  python aggregate_tokens.py              — report for every runs* folder
                                            and plot avg completion tokens
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox
from scipy.ndimage import rotate as rotate_image
from matplotlib.offsetbox import OffsetImage

from plot_config import (
    FIG_WIDTH,
    MODEL_REGISTRY,
    extract_model_name,
    get_model_color,
    get_model_imagebox,
    get_plot_path,
    setup_plot_style,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_STATS_DIR = SCRIPT_DIR / "token_stats"
TIMESTAMP_TOLERANCE_S = 900          # 15 minutes — covers parallel-game drift
ALICE_NAME = "Alice"


def parse_timestamp(ts_str: str) -> datetime:
    """Parse a YYYYMMDD_HHMMSS string into a datetime."""
    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")


def build_token_stats_index():
    """Build a dict mapping timestamp strings to JSONL file paths."""
    index = {}
    for path in TOKEN_STATS_DIR.glob("tokens_*.jsonl"):
        m = re.search(r"tokens_(\d{8}_\d{6})\.jsonl$", path.name)
        if m:
            index[m.group(1)] = path
    return index


def match_games_to_tokens(game_timestamps, index):
    """Greedy 1-to-1 matching: pair each game with the closest token file.

    Pairs are sorted by absolute time delta (smallest first).  Once a token
    file or game is claimed it cannot be reused, preventing double-counting.
    Only pairs within TIMESTAMP_TOLERANCE_S are accepted.

    Returns a list of (game_ts, token_path) tuples and the count of unmatched
    games.
    """
    # Build all candidate pairs
    pairs = []
    for game_ts in game_timestamps:
        game_dt = parse_timestamp(game_ts)
        for tok_ts, tok_path in index.items():
            delta = abs((parse_timestamp(tok_ts) - game_dt).total_seconds())
            if delta <= TIMESTAMP_TOLERANCE_S:
                pairs.append((delta, game_ts, tok_ts, tok_path))

    # Sort by delta ascending — greedily assign closest first
    pairs.sort(key=lambda x: x[0])
    used_games = set()
    used_tokens = set()
    matches = []
    for _delta, game_ts, tok_ts, tok_path in pairs:
        if game_ts in used_games or tok_ts in used_tokens:
            continue
        used_games.add(game_ts)
        used_tokens.add(tok_ts)
        matches.append((game_ts, tok_path))

    missing = len(game_timestamps) - len(matches)
    return matches, missing


def load_alice_stats(jsonl_path: Path) -> dict:
    """Read a JSONL token file and return aggregated stats for Alice only."""
    total_prompt = 0
    total_completion = 0
    total_requests = 0
    by_stage = {}

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("player") != ALICE_NAME:
                continue
            prompt = record.get("prompt_tokens", 0)
            completion = record.get("completion_tokens", 0)
            total_prompt += prompt
            total_completion += completion
            total_requests += 1

            stage = record.get("stage", "unknown")
            if stage not in by_stage:
                by_stage[stage] = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
            by_stage[stage]["prompt_tokens"] += prompt
            by_stage[stage]["completion_tokens"] += completion
            by_stage[stage]["requests"] += 1

    return {
        "total_requests": total_requests,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "by_stage": by_stage,
    }


def aggregate_folder(eval_dir: Path, index: dict, verbose: bool = True):
    """Aggregate Alice's token stats for one eval folder.

    Returns a dict with per-game averages, or None if no data.
    """
    game_files = sorted(eval_dir.glob("Game_*_summary.json"))
    if not game_files:
        if verbose:
            print(f"No Game_*_summary.json files found in {eval_dir}")
        return None

    ts_pattern = re.compile(r"Game_(\d{8}_\d{6})_summary\.json$")
    game_timestamps = []
    for gf in game_files:
        m = ts_pattern.search(gf.name)
        if m:
            game_timestamps.append(m.group(1))

    matches, missing = match_games_to_tokens(game_timestamps, index)
    summaries = [load_alice_stats(tok_path) for _, tok_path in matches]

    if verbose:
        print(f"Matched {len(summaries)}/{len(game_timestamps)} games "
              f"to token stats (skipped {missing})")
        print(f"Showing tokens for: {ALICE_NAME} (Player 0) only\n")

    if not summaries:
        if verbose:
            print("No token stats found for any game in this folder.")
        return None

    total_games = len(summaries)
    total_requests = sum(s["total_requests"] for s in summaries)
    total_prompt = sum(s["total_prompt_tokens"] for s in summaries)
    total_completion = sum(s["total_completion_tokens"] for s in summaries)
    total_tokens = sum(s["total_tokens"] for s in summaries)

    by_stage = defaultdict(lambda: {"prompt": 0, "completion": 0, "requests": 0, "games": 0})
    for summary in summaries:
        for stage, stats in summary.get("by_stage", {}).items():
            by_stage[stage]["prompt"] += stats["prompt_tokens"]
            by_stage[stage]["completion"] += stats["completion_tokens"]
            by_stage[stage]["requests"] += stats["requests"]
            by_stage[stage]["games"] += 1

    result = {
        "total_games": total_games,
        "avg_requests": total_requests / total_games,
        "avg_prompt": total_prompt / total_games,
        "avg_completion": total_completion / total_games,
        "avg_total": total_tokens / total_games,
        "by_stage": dict(by_stage),
    }

    if verbose:
        print_report(result)

    return result


def print_report(result):
    """Print a single-folder token report."""
    print(f"=== ALICE TOKEN AVERAGES PER GAME ===")
    print(f"Total Games: {result['total_games']}")
    print(f"Avg Requests/game: {result['avg_requests']:.1f}")
    print(f"Avg Prompt tokens/game: {result['avg_prompt']:,.1f}")
    print(f"Avg Completion tokens/game: {result['avg_completion']:,.1f}")
    print(f"Avg Total tokens/game: {result['avg_total']:,.1f}")

    print("\n=== BY STAGE (avg per game that had this stage) ===")
    for stage, stats in sorted(
        result["by_stage"].items(),
        key=lambda x: x[1]["prompt"] / x[1]["games"] if x[1]["games"] else 0,
        reverse=True,
    ):
        avg_prompt = stats["prompt"] / stats["games"]
        avg_completion = stats["completion"] / stats["games"]
        avg_requests = stats["requests"] / stats["games"]
        print(
            f"{stage}: {stats['games']} games, "
            f"avg {avg_requests:.1f} reqs, "
            f"{avg_prompt:,.1f} prompt, "
            f"{avg_completion:,.1f} completion"
        )


def plot_completion_tokens(results: dict):
    """Vertical bar chart of avg completion tokens/game across all models, sorted."""
    setup_plot_style(use_latex=True)

    # Collect entries that exist in both results and MODEL_REGISTRY
    entries = []
    for folder_key, res in results.items():
        entries.append((folder_key, res))

    if not entries:
        print("Nothing to plot.")
        return

    # Sort by avg_completion descending
    entries.sort(key=lambda x: x[1]["avg_completion"], reverse=True)

    names = [extract_model_name(k) for k, _ in entries]
    values = [r["avg_completion"] for _, r in entries]
    colors = [get_model_color(n) for n in names]

    BAR_WIDTH = 0.6
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.0))

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, values, color=colors, width=BAR_WIDTH, zorder=5)

    # Value labels on top of bars
    y_max = max(values) * 1.15
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + y_max * 0.01,
            f"{val:,.0f}", ha="center", va="bottom",
            fontsize=9
        )

    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Avg Completion Tokens / Game")

    # Model logos above each bar label (below the x-axis)
    ax.tick_params(axis="x", color="0.85", labelcolor="0", pad=0)
    ax.tick_params(axis="y", color="0.85", labelcolor="0")

    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()

    # Render once so tick-label bounding boxes are available
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_ax = ax.transAxes.inverted()

    for i, name in enumerate(names):
        imagebox = get_model_imagebox(name)
        if imagebox is not None:
            label = ax.get_xticklabels()[i]
            bbox = label.get_window_extent(renderer)

            # The label is rotated 35° with ha="right", so the text starts
            # at the lower-left corner of its axis-aligned bounding box.
            # Convert that point to axes-fraction coords (stable across savefig).
            end_x, end_y = inv_ax.transform((bbox.x0, bbox.y0))

            # Rotate the image to match the 35° label angle
            img_data = imagebox.get_data()
            rotated_data = rotate_image(img_data, 35, reshape=True, order=1,
                                        mode='constant', cval=255)
            rotated_imagebox = OffsetImage(rotated_data, zoom=imagebox.get_zoom())

            # Place the icon at the text-start end of the label
            ab = AnnotationBbox(
                rotated_imagebox,
                xy=(end_x, end_y),
                xycoords="axes fraction",
                frameon=False,
                clip_on=False,
                box_alignment=(0.85, 0.4),
                zorder=1,
            )
            ax.add_artist(ab)

    out = get_plot_path("token_completion_comparison.pdf")
    plt.savefig(out)
    print(f"\nPlot saved to: {out}")
    plt.close()


def main():
    index = build_token_stats_index()

    if len(sys.argv) >= 2:
        # ── Single-folder mode ──
        eval_dir = Path(sys.argv[1])
        if not eval_dir.is_absolute():
            eval_dir = SCRIPT_DIR / eval_dir
        if not eval_dir.is_dir():
            print(f"Error: {eval_dir} is not a directory")
            sys.exit(1)
        result = aggregate_folder(eval_dir, index, verbose=True)
        if result is None:
            sys.exit(1)
    else:
        # ── All runs* folders → aggregate + plot ──
        run_dirs = sorted(
            d for d in SCRIPT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("runs")
        )
        if not run_dirs:
            print("No runs* directories found next to this script.")
            sys.exit(1)

        results = {}   # folder_key → result dict
        for rd in run_dirs:
            print(f"{'─'*60}")
            print(f"  {rd.name}")
            print(f"{'─'*60}")
            res = aggregate_folder(rd, index, verbose=True)
            if res is not None:
                results[rd.name] = res
            print()

        if results:
            plot_completion_tokens(results)


if __name__ == "__main__":
    main()
