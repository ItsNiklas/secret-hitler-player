"""
Vote-decision analysis for Secret Hitler games.

Compares Alice's votes against the majority decision in each government
proposal and plots agreement rates per round.

Usage: python decisions.py <eval_dir>
  eval_dir  Directory containing game JSON files (e.g. runsF1-Qwen3)
"""

from functools import cache
import os
import glob
import json
import sys
from typing import List, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from plot_config import setup_plot_style, extract_model_name, get_plot_path, UNIBLAU

setup_plot_style()

EVAL_DIR = sys.argv[1] 


def load_eval_run_filenames() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(f"{current_dir}/{EVAL_DIR}/*.json")


def load_crawl_summary_filenames() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return glob.glob(f"{parent_dir}/crawl/summaries/*.json")


def extract_game_id_from_filename(filename: str) -> str:
    base_filename = os.path.basename(filename)

    if "_20" in base_filename and base_filename.count("_") >= 3:
        return base_filename.split("_")[0]
    elif "_summary.json" in base_filename:
        return base_filename.replace("_summary.json", "")
    return None


def get_matching_game_ids() -> Set[str]:
    eval_filenames = load_eval_run_filenames()
    crawl_filenames = load_crawl_summary_filenames()

    eval_game_ids = {extract_game_id_from_filename(f) for f in eval_filenames}
    crawl_game_ids = {extract_game_id_from_filename(f) for f in crawl_filenames}

    return eval_game_ids.intersection(crawl_game_ids)


@cache
def load_matching_run_data():
    matching_ids = get_matching_game_ids()
    matching_count = len(matching_ids)
    eval_filenames = load_eval_run_filenames()

    log_count_distribution = defaultdict(int)
    log_count_games = defaultdict(list)

    for filename in eval_filenames:
        game_id = extract_game_id_from_filename(filename)
        if game_id in matching_ids:
            with open(filename, "r") as f:
                data = json.load(f)
                logs_length = len(data.get("logs", []))
                log_count_distribution[logs_length] += 1
                log_count_games[logs_length].append(game_id)

    print("\nLog count distribution:")
    for count in sorted(log_count_distribution.keys()):
        games = log_count_distribution[count]
        percentage = (games / matching_count) * 100 if matching_count > 0 else 0
        print(f"Games with {count} logs: {games} ({percentage:.2f}%)")

    return log_count_distribution, log_count_games


def get_last_log_for_single_round_games():
    """
    For games with round count of 1, fetch the corresponding game from summaries/
    and extract the last element of their logs array.
    Then compare with the only log element in the eval run file.
    Only print games where the chancellor is the same in both logs.
    """
    matching_ids = get_matching_game_ids()
    matching_count = len(matching_ids)
    _, log_count_games = load_matching_run_data()

    # Get games with only 1 log entry
    single_log_games = log_count_games.get(1, [])

    if not single_log_games:
        print("\nNo games with just 1 log entry found.")
        return

    single_log_count = len(single_log_games)
    percentage = (single_log_count / matching_count) * 100 if matching_count > 0 else 0
    print(f"\nFound {single_log_count} games with just 1 log entry ({percentage:.2f}%)")

    # HACKY FILTER: Drop games with low lib ELO
    filtered_single_log_games = []
    for game_id in single_log_games:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        summary_path = f"{parent_dir}/crawl/summaries/{game_id}_summary.json"
        
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
                lib_elo = summary_data.get("libElo", {}).get("overall", 0)
                if lib_elo >= 0: # !!!
                    filtered_single_log_games.append(game_id)
    
    filtered_count = len(filtered_single_log_games)
    filtered_percentage = (filtered_count / single_log_count) * 100 if single_log_count > 0 else 0
    print(f"After filtering for high lib ELO: {filtered_count} games ({filtered_percentage:.2f}% of single log games)")
    
    single_log_games = filtered_single_log_games  # Replace the original list with filtered one
    single_log_count = filtered_count  # Update count

    # For each game with 1 log, get the corresponding summary file and eval run file
    comparison_data = []
    same_chancellor_count = 0
    same_chancellor_role_count = 0
    same_chancellor_affiliation_count = 0
    same_voting_result_count = 0
    same_voting_result_same_affiliation_count = 0

    # Helper function to determine party affiliation
    def get_party_affiliation(role):
        if role == "liberal":
            return "liberal"
        elif role in ["fascist", "hitler"]:
            return "fascist"
        return "unknown"

    for game_id in single_log_games:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        summary_path = f"{parent_dir}/crawl/summaries/{game_id}_summary.json"
        eval_run_path = None

        # Find the corresponding eval run file
        for run_file in load_eval_run_filenames():
            if extract_game_id_from_filename(run_file) == game_id:
                eval_run_path = run_file
                break

        if os.path.exists(summary_path) and eval_run_path:
            # Get summary data
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
                summary_logs = summary_data.get("logs", [])
                # Create a role mapping based on player index
                summary_players_roles = {}
                summary_players_names = {}
                for i, player in enumerate(summary_data.get("players", [])):
                    summary_players_roles[i] = player.get("role")
                    summary_players_names[i] = player.get("username")

                # Get eval run data
                with open(eval_run_path, "r") as f:
                    eval_data = json.load(f)
                    eval_logs = eval_data.get("logs", [])
                    # Create a role mapping based on player index
                    eval_players_roles = {}
                    eval_players_names = {}
                    for i, player in enumerate(eval_data.get("players", [])):
                        eval_players_roles[i] = player.get("role")
                        eval_players_names[i] = player.get("username")

                if summary_logs and eval_logs:
                    # Get the last log entry from summary and the only log entry from eval
                    summary_last_log = summary_logs[-1]
                    eval_only_log = eval_logs[0] if eval_logs else None

                    # Check if the chancellor is the same in both logs
                    summary_chancellor = summary_last_log.get("chancellorId") if summary_last_log else None
                    eval_chancellor = eval_only_log.get("chancellorId") if eval_only_log else None

                    if summary_chancellor == eval_chancellor and summary_chancellor is not None:
                        same_chancellor_count += 1

                    # Check if the chancellor has the same role in both logs
                    summary_chancellor_role = summary_players_roles.get(summary_chancellor, "unknown")
                    eval_chancellor_role = eval_players_roles.get(eval_chancellor, "unknown")

                    # Check if the chancellor has the same party affiliation in both logs
                    summary_chancellor_affiliation = get_party_affiliation(summary_chancellor_role)
                    eval_chancellor_affiliation = get_party_affiliation(eval_chancellor_role)
                    
                    # Get voting results once to avoid repeating code
                    summary_votes = summary_last_log.get("votes", [])
                    eval_votes = eval_only_log.get("votes", [])
                    
                    summary_vote_success = None
                    eval_vote_success = None
                    
                    if summary_votes and eval_votes:
                        summary_vote_success = sum(1 for v in summary_votes if v) > len(summary_votes) / 2
                        eval_vote_success = sum(1 for v in eval_votes if v) > len(eval_votes) / 2
                    
                    # Same role check
                    if summary_chancellor_role == eval_chancellor_role and summary_chancellor_role != "unknown":
                        same_chancellor_role_count += 1
                        
                        # Check if voting results match for same role
                        if summary_vote_success is not None and eval_vote_success is not None:
                            if summary_vote_success == eval_vote_success:
                                same_voting_result_count += 1
                    
                    # Same party affiliation check
                    if summary_chancellor_affiliation == eval_chancellor_affiliation and summary_chancellor_affiliation != "unknown":
                        same_chancellor_affiliation_count += 1

                        # Check if voting results match for same affiliation
                        if summary_vote_success is not None and eval_vote_success is not None:
                            if summary_vote_success == eval_vote_success:
                                same_voting_result_same_affiliation_count += 1

                    comparison_data.append(
                        {
                            "game_id": game_id,
                            "summary_logs_count": len(summary_logs),
                            "summary_last_log": summary_last_log,
                            "eval_only_log": eval_only_log,
                            "summary_players_roles": summary_players_roles,
                            "eval_players_roles": eval_players_roles,
                            "summary_players_names": summary_players_names,
                            "eval_players_names": eval_players_names,
                        }
                    )

    # Print comparison of findings
    same_chancellor_percentage = (same_chancellor_count / single_log_count) * 100 if single_log_count > 0 else 0
    same_role_percentage = (same_chancellor_role_count / single_log_count) * 100 if single_log_count > 0 else 0
    same_affiliation_percentage = (same_chancellor_affiliation_count / single_log_count) * 100 if single_log_count > 0 else 0

    print(f"\nFound {same_chancellor_count} games with the same chancellor in both logs ({same_chancellor_percentage:.2f}% of single log games)")
    print(f"Found {same_chancellor_role_count} games where the chancellor has the same role (L/F/H) in both logs ({same_role_percentage:.2f}% of single log games)")
    print(f"Found {same_chancellor_affiliation_count} games where the chancellor has the same party affiliation (L/F) in both logs ({same_affiliation_percentage:.2f}% of single log games)")

    # Calculate percentage of games where voting results match among games with same chancellor role
    same_vote_of_same_role_percentage = (same_voting_result_count / same_chancellor_role_count) * 100 if same_chancellor_role_count > 0 else 0
    print(
        f"When a chancellor of the same role was chosen, eval voted the same as summary in {same_vote_of_same_role_percentage:.2f}% of times ({same_voting_result_count}/{same_chancellor_role_count} games)"
    )
    
    # Calculate percentage of games where voting results match among games with same chancellor affiliation
    same_vote_of_same_affiliation_percentage = (same_voting_result_same_affiliation_count / same_chancellor_affiliation_count) * 100 if same_chancellor_affiliation_count > 0 else 0
    print(
        f"When a chancellor of the same affiliation was chosen, eval voted the same as summary in {same_vote_of_same_affiliation_percentage:.2f}% of times ({same_voting_result_same_affiliation_count}/{same_chancellor_affiliation_count} games)"
    )

    # --- Plot: Decision agreement metrics ---
    metrics = {
        'Same Chancellor': same_chancellor_percentage,
        'Same Role': same_role_percentage,
        'Same Affiliation': same_affiliation_percentage,
        'Vote Agree\n(same role)': same_vote_of_same_role_percentage,
        'Vote Agree\n(same aff.)': same_vote_of_same_affiliation_percentage,
    }
    fig, ax = plt.subplots(figsize=(6.46, 3))
    bars = ax.bar(metrics.keys(), metrics.values(), color=UNIBLAU, zorder=5)
    ax.set_ylabel('Agreement Rate')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}\\%'))
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}\\%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    model_slug = extract_model_name(EVAL_DIR).replace(' ', '_').lower()
    out_path = get_plot_path(f'decisions_{model_slug}.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {out_path}")

    return comparison_data


if __name__ == "__main__":
    eval_files = load_eval_run_filenames()
    crawl_files = load_crawl_summary_filenames()

    print(f"Found {len(eval_files)} files in eval/runs")
    print(f"Found {len(crawl_files)} files in crawl/summaries")

    matching_ids = get_matching_game_ids()
    matching_count = len(matching_ids)
    print(f"Found {matching_count} matching game IDs between eval/runs and crawl/summaries")

    log_distribution, _ = load_matching_run_data()

    # For games with 1 log, get the last log element from summaries
    get_last_log_for_single_round_games()
