"""
Belief-accuracy questionnaire evaluation.

Assesses how accurately players (or Alice alone) identify other players'
roles based on in-game beliefs logged per round.

Usage: python questionaire.py <eval_dir>
  eval_dir  Directory containing game JSON files (e.g. runsF1-Qwen3)
"""

import os
import glob
import json
import sys
from typing import List, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from plot_config import FIG_WIDTH, setup_plot_style, extract_model_name, get_plot_path, ROLE_COLORS

setup_plot_style()

EVAL_DIR = sys.argv[1] if len(sys.argv) > 1 else None
ALICE_ONLY = True  # Always evaluate Alice (Player 0) only


def load_eval_run_filenames() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(f"{current_dir}/{EVAL_DIR}/*.json")


def parse_game_data(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse game data to extract true roles and belief assessments."""

    # Extract true roles
    true_roles = {}
    player_id_to_username = {}
    player_id_to_index = {}

    for i, player in enumerate(game_data["players"]):
        username = player["username"]
        role = player["role"]
        player_id = player["_id"]

        true_roles[username] = role.lower()  # Normalize to lowercase
        player_id_to_username[player_id] = username
        player_id_to_index[player_id] = i

    # Extract belief assessments from logs
    belief_data = []

    for log_entry in game_data["logs"]:
        if "rapidAssessments" in log_entry:
            round_beliefs = {}

            for assessor_idx, assessment_str in log_entry["rapidAssessments"].items():
                assessor_idx = int(assessor_idx)
                assessor_username = game_data["players"][assessor_idx]["username"]

                # Parse the assessment string
                beliefs = {}
                valid_beliefs = {"unknown", "fascist", "liberal", "hitler"}

                for line in assessment_str.strip().split("\n"):
                    if ":" in line:
                        target, belief = line.split(":", 1)
                        target = target.strip()
                        belief = belief.strip()

                        # Filter out self-referential or invalid beliefs
                        if belief.lower() not in valid_beliefs:
                            continue

                        # Skip if player is assessing their own role
                        if target == assessor_username:
                            continue

                        beliefs[target] = belief

                round_beliefs[assessor_username] = beliefs

            belief_data.append(
                {
                    "log_id": log_entry["_id"],
                    "beliefs": round_beliefs,
                    "president_id": log_entry.get("presidentId"),
                    "chancellor_id": log_entry.get("chancellorId"),
                    "enacted_policy": log_entry.get("enactedPolicy"),
                }
            )

    return {"game_id": game_data["_id"], "true_roles": true_roles, "players": [p["username"] for p in game_data["players"]], "belief_data": belief_data}


def calculate_belief_accuracy(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate accuracy statistics for each player's beliefs."""

    true_roles = parsed_data["true_roles"]
    players = parsed_data["players"]

    # Initialize statistics for each player
    player_stats = {}
    for player in players:
        player_stats[player] = {
            "total_beliefs": 0,
            "correct_beliefs": 0,
            "incorrect_beliefs": 0,
            "by_round": [],
            "by_target_role": {"fascist": {"correct": 0, "total": 0}, "liberal": {"correct": 0, "total": 0}, "hitler": {"correct": 0, "total": 0}},
            "belief_distribution": {"Unknown": 0, "Fascist": 0, "Liberal": 0, "Hitler": 0},
        }

    # Analyze each round
    for round_idx, round_data in enumerate(parsed_data["belief_data"]):
        round_stats = {}

        for assessor, beliefs in round_data["beliefs"].items():
            if assessor not in player_stats:
                continue

            round_correct = 0
            round_total = 0

            for target, belief in beliefs.items():
                if target not in true_roles:
                    continue

                true_role = true_roles[target]
                round_total += 1
                player_stats[assessor]["total_beliefs"] += 1
                player_stats[assessor]["belief_distribution"][belief] += 1

                # Check if belief is correct
                if belief.lower() == true_role.lower():
                    round_correct += 1
                    player_stats[assessor]["correct_beliefs"] += 1
                    player_stats[assessor]["by_target_role"][true_role.lower()]["correct"] += 1
                else:
                    player_stats[assessor]["incorrect_beliefs"] += 1

                player_stats[assessor]["by_target_role"][true_role.lower()]["total"] += 1

            # Store round statistics
            accuracy = round_correct / round_total if round_total > 0 else 0
            player_stats[assessor]["by_round"].append({"round": round_idx + 1, "correct": round_correct, "total": round_total, "accuracy": accuracy})
            round_stats[assessor] = accuracy

    # Calculate overall accuracies
    for player in player_stats:
        stats = player_stats[player]
        stats["overall_accuracy"] = stats["correct_beliefs"] / stats["total_beliefs"] if stats["total_beliefs"] > 0 else 0

        # Calculate accuracy by target role
        for role in stats["by_target_role"]:
            role_stats = stats["by_target_role"][role]
            role_stats["accuracy"] = role_stats["correct"] / role_stats["total"] if role_stats["total"] > 0 else 0

    return player_stats


def calculate_hitler_identification_timing(all_games_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate how early Hitler was correctly identified in games where Alice was Liberal."""

    if not ALICE_ONLY:
        return {}

    hitler_identification_rounds = []
    games_analyzed = 0

    for game_data in all_games_data:
        true_roles = game_data["true_roles"]

        # Only analyze games where Alice was Liberal
        if true_roles.get("Alice", "").lower() != "liberal":
            continue

        games_analyzed += 1

        # Find who Hitler is in this game
        hitler_player = None
        for player, role in true_roles.items():
            if role.lower() == "hitler":
                hitler_player = player
                break

        if not hitler_player:
            continue

        # Look through rounds to find first correct Hitler identification
        first_correct_round = None

        for round_idx, round_data in enumerate(game_data["belief_data"]):
            for assessor, beliefs in round_data["beliefs"].items():
                # Check all players' beliefs, not just Alice
                if hitler_player in beliefs:
                    belief = beliefs[hitler_player]
                    if belief.lower() == "hitler":
                        first_correct_round = round_idx + 1  # 1-indexed
                        break

            if first_correct_round is not None:
                break

        if first_correct_round is not None:
            hitler_identification_rounds.append(first_correct_round)

    result = {}
    if hitler_identification_rounds:
        result["average_round"] = sum(hitler_identification_rounds) / len(hitler_identification_rounds)
        result["total_games_with_identification"] = len(hitler_identification_rounds)
        result["total_liberal_alice_games"] = games_analyzed
        result["identification_rate"] = len(hitler_identification_rounds) / games_analyzed if games_analyzed > 0 else 0
    else:
        result["average_round"] = None
        result["total_games_with_identification"] = 0
        result["total_liberal_alice_games"] = games_analyzed
        result["identification_rate"] = 0

    return result


def print_accuracy_statistics(all_games_data: List[Dict[str, Any]]):
    """Print comprehensive accuracy statistics across all games."""

    # Aggregate statistics across all games
    aggregated_stats = defaultdict(
        lambda: {
            "total_beliefs": 0,
            "correct_beliefs": 0,
            "games_played": 0,
            "by_target_role": {"fascist": {"correct": 0, "total": 0}, "liberal": {"correct": 0, "total": 0}, "hitler": {"correct": 0, "total": 0}},
            "belief_distribution": {"Unknown": 0, "Fascist": 0, "Liberal": 0, "Hitler": 0},
            "by_own_role": defaultdict(
                lambda: {"correct": 0, "total": 0, "by_target_role": {"fascist": {"correct": 0, "total": 0}, "liberal": {"correct": 0, "total": 0}, "hitler": {"correct": 0, "total": 0}}}
            ),
        }
    )

    for game_data in all_games_data:
        game_stats = calculate_belief_accuracy(game_data)
        true_roles = game_data["true_roles"]

        for player, stats in game_stats.items():
            # Filter for Alice only if toggle is enabled
            if ALICE_ONLY and player != "Alice":
                continue

            if stats["total_beliefs"] > 0:  # Only count players who made assessments
                aggregated_stats[player]["total_beliefs"] += stats["total_beliefs"]
                aggregated_stats[player]["correct_beliefs"] += stats["correct_beliefs"]
                aggregated_stats[player]["games_played"] += 1

                # Aggregate by target role
                for role in stats["by_target_role"]:
                    aggregated_stats[player]["by_target_role"][role]["correct"] += stats["by_target_role"][role]["correct"]
                    aggregated_stats[player]["by_target_role"][role]["total"] += stats["by_target_role"][role]["total"]

                # Aggregate belief distribution
                for belief_type in stats["belief_distribution"]:
                    aggregated_stats[player]["belief_distribution"][belief_type] += stats["belief_distribution"][belief_type]

                # Track accuracy by player's own role (correctly per game)
                player_true_role = true_roles.get(player, "unknown")
                aggregated_stats[player]["by_own_role"][player_true_role]["correct"] += stats["correct_beliefs"]
                aggregated_stats[player]["by_own_role"][player_true_role]["total"] += stats["total_beliefs"]

                # Track target role breakdown for this player's own role
                for target_role in ["fascist", "liberal", "hitler"]:
                    aggregated_stats[player]["by_own_role"][player_true_role]["by_target_role"][target_role]["correct"] += stats["by_target_role"][target_role]["correct"]
                    aggregated_stats[player]["by_own_role"][player_true_role]["by_target_role"][target_role]["total"] += stats["by_target_role"][target_role]["total"]

    # Print results
    print("\n" + "=" * 80)
    if ALICE_ONLY:
        print("BELIEF ACCURACY STATISTICS - ALICE ONLY")
    else:
        print("BELIEF ACCURACY STATISTICS")
    print("=" * 80)

    # Sort players by overall accuracy
    players_by_accuracy = sorted(aggregated_stats.items(), key=lambda x: x[1]["correct_beliefs"] / x[1]["total_beliefs"] if x[1]["total_beliefs"] > 0 else 0, reverse=True)

    # Print belief distribution summary
    print(f"\nBELIEF DISTRIBUTION BY PLAYER:")
    print("This table shows the proportion of each type of belief each player made.")
    print("Values are proportions (0.000 to 1.000) showing how often they guessed each role.")
    print(f"\n{'Player':<12} {'Unknown':<8} {'Fascist':<8} {'Liberal':<8} {'Hitler':<8}")
    print("-" * 50)
    for player, stats in players_by_accuracy:
        if stats["total_beliefs"] == 0:
            continue
        dist = stats["belief_distribution"]
        total = sum(dist.values())
        if total > 0:
            print(f"{player:<12} {dist['Unknown']/total:.3f}    {dist['Fascist']/total:.3f}    {dist['Liberal']/total:.3f}    {dist['Hitler']/total:.3f}")

    # Add performance by own role analysis
    print(f"\n\nPERFORMANCE BY PLAYER'S OWN ROLE:")
    print("This shows how well players performed when they themselves had different roles.")

    role_performance = defaultdict(
        lambda: {"correct": 0, "total": 0, "players": set(), "by_target_role": {"fascist": {"correct": 0, "total": 0}, "liberal": {"correct": 0, "total": 0}, "hitler": {"correct": 0, "total": 0}}}
    )

    for player, stats in aggregated_stats.items():
        if stats["total_beliefs"] == 0:
            continue
        for role, performance in stats["by_own_role"].items():
            if performance["total"] > 0:
                role_performance[role]["correct"] += performance["correct"]
                role_performance[role]["total"] += performance["total"]
                role_performance[role]["players"].add(player)

                # Add target role breakdown
                for target_role in ["fascist", "liberal", "hitler"]:
                    role_performance[role]["by_target_role"][target_role]["correct"] += performance["by_target_role"][target_role]["correct"]
                    role_performance[role]["by_target_role"][target_role]["total"] += performance["by_target_role"][target_role]["total"]

    print(f"\n{'Role':<12} {'Players':<8} {'Total':<7} {'Correct':<8} {'Accuracy':<10} {'vs Fascist':<12} {'vs Liberal':<12} {'vs Hitler':<10}")
    print("-" * 90)
    for role, perf in role_performance.items():
        if perf["total"] > 0:
            accuracy = perf["correct"] / perf["total"]

            # Calculate accuracy vs each target role type
            fasc_acc = perf["by_target_role"]["fascist"]["correct"] / perf["by_target_role"]["fascist"]["total"] if perf["by_target_role"]["fascist"]["total"] > 0 else 0
            lib_acc = perf["by_target_role"]["liberal"]["correct"] / perf["by_target_role"]["liberal"]["total"] if perf["by_target_role"]["liberal"]["total"] > 0 else 0
            hitler_acc = perf["by_target_role"]["hitler"]["correct"] / perf["by_target_role"]["hitler"]["total"] if perf["by_target_role"]["hitler"]["total"] > 0 else 0

            print(f"{role.capitalize():<12} {len(perf['players']):<8} {perf['total']:<7} {perf['correct']:<8} {accuracy:.3f}     {fasc_acc:.3f}       {lib_acc:.3f}       {hitler_acc:.3f}")

    print("\nExplanation:")
    print("- 'Players': Number of unique players who had this role and made assessments")
    print("- 'Total': Total number of role guesses made by players with this role")
    print("- 'Correct': Number of correct role identifications")
    print("- 'Accuracy': Overall proportion of correct guesses (Correct/Total)")
    print("- 'vs Fascist/Liberal/Hitler': Accuracy when identifying players of those specific roles")

    # Add Hitler identification timing analysis for Alice mode
    if ALICE_ONLY:
        hitler_timing = calculate_hitler_identification_timing(all_games_data)
        print(f"\n\nHITLER IDENTIFICATION TIMING (Alice as Liberal):")
        print("This shows how early Hitler was correctly identified in games where Alice was Liberal.")
        print("-" * 70)

        if hitler_timing["total_liberal_alice_games"] > 0:
            print(f"Total games where Alice was Liberal: {hitler_timing['total_liberal_alice_games']}")
            print(f"Games where Hitler was identified: {hitler_timing['total_games_with_identification']}")
            print(f"Hitler identification rate: {hitler_timing['identification_rate']:.3f}")

            if hitler_timing["average_round"] is not None:
                print(f"Average round of first correct Hitler identification: {hitler_timing['average_round']:.2f}")
            else:
                print("Hitler was never correctly identified in these games")
        else:
            print("No games found where Alice was Liberal")


if __name__ == "__main__":
    if EVAL_DIR is None:
        print("Usage: python questionaire.py <EVAL_DIR> [alice]")
        sys.exit(1)

    eval_files = load_eval_run_filenames()
    print(f"Found {len(eval_files)} files in eval/{EVAL_DIR}")
    if ALICE_ONLY:
        print("Mode: Evaluating Alice only")
    else:
        print("Mode: Evaluating all players")

    # Load and parse each game file
    all_games_data = []

    for file_path in eval_files:
        if 'annotat' in file_path.lower():
            continue

        with open(file_path, "r") as f:
            game_data = json.load(f)

        parsed_data = parse_game_data(game_data)
        all_games_data.append(parsed_data)

    print(f"\nTotal games processed: {len(all_games_data)}")

    # Print accuracy statistics across all games
    print_accuracy_statistics(all_games_data)

    # --- Plot: Belief accuracy by own role ---
    # Aggregate accuracy by role across all games
    role_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
    for game_data in all_games_data:
        game_stats = calculate_belief_accuracy(game_data)
        true_roles = game_data['true_roles']
        for player, stats in game_stats.items():
            if ALICE_ONLY and player != 'Alice':
                continue
            if stats['total_beliefs'] == 0:
                continue
            own_role = true_roles.get(player, 'unknown')
            role_acc[own_role]['correct'] += stats['correct_beliefs']
            role_acc[own_role]['total'] += stats['total_beliefs']

    if role_acc:
        roles_to_plot = [r for r in ['liberal', 'fascist', 'hitler'] if role_acc[r]['total'] > 0]
        accuracies = [role_acc[r]['correct'] / role_acc[r]['total'] * 100 for r in roles_to_plot]
        colors = [ROLE_COLORS.get(r, '#999999') for r in roles_to_plot]

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3))
        bars = ax.bar([r.capitalize() for r in roles_to_plot], accuracies, color=colors, zorder=5)
        ax.set_ylabel('Belief Accuracy')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}\\%'))
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}\\%', ha='center', va='bottom')
        plt.tight_layout()
        model_slug = extract_model_name(EVAL_DIR).replace(' ', '_').lower()
        mode_suffix = '_alice' if ALICE_ONLY else '_all'
        out_path = get_plot_path(f'questionaire_{model_slug}{mode_suffix}.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {out_path}")
