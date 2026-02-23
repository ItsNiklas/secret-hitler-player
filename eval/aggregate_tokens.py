#!/usr/bin/env python3
"""
Token usage aggregator for Secret Hitler evaluation runs.

Reads game filenames from a runs folder, finds corresponding token-stats
summaries (matched by closest timestamp within 60 s), and prints per-game
averages overall and broken down by game stage.

Usage: python aggregate_tokens.py <eval_dir>
  eval_dir  Directory containing Game_*_summary.json files (e.g. runsF2-GEMMA)
"""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_STATS_DIR = SCRIPT_DIR / "token_stats"
TIMESTAMP_TOLERANCE_S = 60


def parse_timestamp(ts_str: str) -> datetime:
    """Parse a YYYYMMDD_HHMMSS string into a datetime."""
    return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")


def build_token_stats_index():
    """Build a dict mapping timestamp strings to summary file paths."""
    index = {}
    for path in TOKEN_STATS_DIR.glob("summary_*.json"):
        m = re.search(r"summary_(\d{8}_\d{6})\.json$", path.name)
        if m:
            index[m.group(1)] = path
    return index


def find_matching_token_stats(game_ts: str, index: dict):
    """Find the token-stats file whose timestamp is closest to game_ts (within tolerance)."""
    game_dt = parse_timestamp(game_ts)
    best_path = None
    best_delta = float("inf")
    for ts_str, path in index.items():
        delta = abs((parse_timestamp(ts_str) - game_dt).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_path = path
    if best_delta <= TIMESTAMP_TOLERANCE_S:
        return best_path
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_tokens.py <eval_dir>")
        sys.exit(1)

    eval_dir = Path(sys.argv[1])
    if not eval_dir.is_absolute():
        eval_dir = SCRIPT_DIR / eval_dir

    if not eval_dir.is_dir():
        print(f"Error: {eval_dir} is not a directory")
        sys.exit(1)

    # Find game files and extract timestamps
    game_files = sorted(eval_dir.glob("Game_*_summary.json"))
    if not game_files:
        print(f"No Game_*_summary.json files found in {eval_dir}")
        sys.exit(1)

    ts_pattern = re.compile(r"Game_(\d{8}_\d{6})_summary\.json$")
    game_timestamps = []
    for gf in game_files:
        m = ts_pattern.search(gf.name)
        if m:
            game_timestamps.append(m.group(1))

    # Build token stats index and match
    index = build_token_stats_index()
    summaries = []
    missing = 0
    for ts in game_timestamps:
        stats_path = find_matching_token_stats(ts, index)
        if stats_path:
            with open(stats_path) as f:
                summaries.append(json.load(f))
        else:
            missing += 1

    print(f"Matched {len(summaries)}/{len(game_timestamps)} games "
          f"to token stats (skipped {missing})\n")

    if not summaries:
        print("No token stats found for any game in this folder.")
        sys.exit(1)

    # Aggregate
    total_games = len(summaries)
    total_requests = sum(s.get("total_requests", 0) for s in summaries)
    total_prompt = sum(s.get("total_prompt_tokens", 0) for s in summaries)
    total_completion = sum(s.get("total_completion_tokens", 0) for s in summaries)
    total_tokens = sum(s.get("total_tokens", 0) for s in summaries)

    by_stage = defaultdict(lambda: {"prompt": 0, "completion": 0, "requests": 0, "games": 0})
    for summary in summaries:
        for stage, stats in summary.get("by_stage", {}).items():
            by_stage[stage]["prompt"] += stats.get("prompt_tokens", 0)
            by_stage[stage]["completion"] += stats.get("completion_tokens", 0)
            by_stage[stage]["requests"] += stats.get("requests", 0)
            by_stage[stage]["games"] += 1

    # Print results
    print("=== AVERAGES PER GAME ===")
    print(f"Total Games: {total_games}")
    print(f"Avg Requests/game: {total_requests / total_games:.1f}")
    print(f"Avg Prompt tokens/game: {total_prompt / total_games:,.1f}")
    print(f"Avg Completion tokens/game: {total_completion / total_games:,.1f}")
    print(f"Avg Total tokens/game: {total_tokens / total_games:,.1f}")

    print("\n=== BY STAGE (avg per game that had this stage) ===")
    for stage, stats in sorted(
        by_stage.items(),
        key=lambda x: x[1]["prompt"] / x[1]["games"],
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


if __name__ == "__main__":
    main()
