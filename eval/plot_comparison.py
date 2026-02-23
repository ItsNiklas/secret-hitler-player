"""
Multi-model belief-accuracy comparison plot.

Loads game data from several evaluation directories (configured in
EVAL_DIRS) and plots liberal-player role-identification accuracy per round,
comparing across models. Uses functions from questionaire.py.
No CLI arguments; edit EVAL_DIRS and ALICE_ONLY in-source.
"""

import os
import glob
import json
from typing import List, Dict, Any
from collections import defaultdict
from matplotlib.offsetbox import AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np

from questionaire import parse_game_data, calculate_belief_accuracy
from plot_config import get_model_imagebox, setup_plot_style, extract_model_name, get_model_color, get_markerdata_for_model, get_plot_path

# Apply shared plotting configuration
setup_plot_style()

# Configuration: Add the folders you want to compare here
EVAL_DIRS = [
    "runsF1-G3-12B",
    "runsF1-G3-27B",
    "runsF1-Llama33-70B",
    "runsF1-Q3",
    "runsF1-R1Distill-70B",
    # Add more directories here as needed
]

# Set to True to analyze only Alice, False for all players
ALICE_ONLY = True


def load_eval_run_filenames(eval_dir: str) -> List[str]:
    """Load all JSON files from the specified evaluation directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(f"{current_dir}/{eval_dir}/*summary.json")


def get_liberal_accuracy_by_round(all_games_data: List[Dict[str, Any]]) -> Dict[int, List[float]]:
    """Get accuracy by round for players who are Liberal in their games."""
    
    liberal_accuracy_by_round = defaultdict(list)

    for game_data in all_games_data:
        game_stats = calculate_belief_accuracy(game_data)
        true_roles = game_data["true_roles"]

        for player, stats in game_stats.items():
            # Filter for Alice only if toggle is enabled
            if ALICE_ONLY and player != "Alice":
                continue

            # Only analyze Liberal players
            player_true_role = true_roles.get(player, "unknown").lower()
            if player_true_role != "liberal":
                continue

            # Extract round-by-round accuracy for this Liberal player
            for round_data in stats["by_round"]:
                round_num = round_data["round"]
                accuracy = round_data["accuracy"]
                liberal_accuracy_by_round[round_num].append(accuracy)

    return liberal_accuracy_by_round


def plot_comparison():
    """Plot accuracy comparison across multiple evaluation directories."""
    
    plt.figure(figsize=(5.50, 3.5))
    
    # Colors for different models
    
    all_rounds = set()
    model_data = {}
    
    for eval_dir in EVAL_DIRS:
        print(f"Processing {eval_dir}...")
        
        eval_files = load_eval_run_filenames(eval_dir)
        if not eval_files:
            print(f"No files found in {eval_dir}, skipping...")
            continue
            
        print(f"Found {len(eval_files)} files in {eval_dir}")
        
        # Load and parse each game file
        all_games_data = []
        for file_path in eval_files:
            try:
                with open(file_path, "r") as f:
                    game_data = json.load(f)
                parsed_data = parse_game_data(game_data)
                all_games_data.append(parsed_data)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_games_data:
            print(f"No valid games found in {eval_dir}, skipping...")
            continue
            
        print(f"Processed {len(all_games_data)} games from {eval_dir}")
        
        # Get liberal accuracy data
        liberal_accuracy_by_round = get_liberal_accuracy_by_round(all_games_data)
        
        if not liberal_accuracy_by_round:
            print(f"No Liberal player data found in {eval_dir}, skipping...")
            continue
        
        # Calculate total number of Liberal players (from round 1)
        total_liberal_players = len(liberal_accuracy_by_round.get(1, []))
        if total_liberal_players == 0:
            print(f"No Liberal players found in round 1 for {eval_dir}, skipping...")
            continue
        
        # Filter rounds: keep only rounds where at least 10% of Liberal players reached that round
        rounds = sorted(liberal_accuracy_by_round.keys())
        filtered_rounds = []
        avg_accuracies = []
        
        for round_num in rounds:
            accuracies = liberal_accuracy_by_round[round_num]
            num_players_in_round = len(accuracies)
            percentage_reached = (num_players_in_round / total_liberal_players) * 100
            
            if percentage_reached >= 10.0:  # Keep rounds with at least 10% of players
                filtered_rounds.append(round_num)
                avg_accuracy = np.mean(accuracies)
                avg_accuracies.append(avg_accuracy)
            else:
                print(f"  Filtering out round {round_num} for {eval_dir}: only {num_players_in_round}/{total_liberal_players} ({percentage_reached:.1f}%) players reached this round")
        
        if not filtered_rounds:
            print(f"No rounds with sufficient data found in {eval_dir}, skipping...")
            continue
        
        all_rounds.update(filtered_rounds)
        model_data[eval_dir] = (filtered_rounds, avg_accuracies)
        
        # Extract model name from directory for cleaner labels
        model_name = extract_model_name(eval_dir)
        color = get_model_color(model_name)
        
        # Get marker style for this model
        m, ms = get_markerdata_for_model(model_name)
        
        # Convert to percentages and plot this model's data
        avg_accuracies_percent = [acc * 100 for acc in avg_accuracies]
        plt.plot(filtered_rounds, avg_accuracies_percent, marker=m, label=f"{model_name}", 
                 linewidth=2, markersize=ms, color=color, markeredgecolor='white', markeredgewidth=1)
        
        print(f"  Plotted {len(filtered_rounds)} rounds for {eval_dir} (rounds {min(filtered_rounds)}-{max(filtered_rounds)})")
    
    if not model_data:
        print("No data to plot!")
        return
    
    plt.xlabel("Round")
    plt.ylabel("Role Identification Accuracy")
    plt.ylim(0)
    # plt.title("Liberal Player Accuracy by Round - Model Comparison" + (" (Alice Only)" if ALICE_ONLY else " (All Players)"))
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    all_rounds_list = sorted(list(all_rounds))
    if all_rounds_list:
        plt.xlim(min(all_rounds_list) - 0.5, max(all_rounds_list) + 0.5)
        plt.xticks(all_rounds_list)
    
    # Format y-axis to show percentage ticks
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}\\%'))
    # Add legend
    legend = plt.gca().legend(framealpha=0, 
                      bbox_to_anchor=(0.5, -0.2), loc='upper center',
                      handlelength=2, handletextpad=1.4, ncol=3)
    
    # Add logos to legend - need to draw first to get positions
    plt.gcf().canvas.draw()
    
    # Add logos to legend
    for model, legend_handle in zip(map(extract_model_name, EVAL_DIRS), legend.legend_handles):
        imagebox = get_model_imagebox(model)
        if not imagebox:
            continue
        
        # Get the position of the legend handle (the line)
        # Position the logo at the start of each legend entry
        ab = AnnotationBbox(imagebox, (0.5, 0.5), 
                           xybox=(19, 0),  # Offset to the left of text
                           xycoords=legend_handle,
                           boxcoords="offset points",
                           frameon=False,
                           box_alignment=(0.5, 0.5),  # Center the imagebox
                           zorder=10)  # High zorder to appear in front
        plt.gcf().add_artist(ab)
    
    # Adjust layout to prevent clipping (?)
    plt.tight_layout()
    
    # Save the plot
    mode_suffix = "_alice" if ALICE_ONLY else "_all"
    output_filename = f"plot_comparison{mode_suffix}.pdf"
    out_path = get_plot_path(output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved to: {out_path}")
    
    # Print summary statistics
    print(f"\nComparison Summary (filtered data):")
    for eval_dir in EVAL_DIRS:
        if eval_dir in model_data:
            rounds, accuracies = model_data[eval_dir]
            model_name = eval_dir.split('-')[1] if '-' in eval_dir else eval_dir
            print(f"{model_name}: Avg accuracy = {np.mean(accuracies)*100:.1f}%, "
                  f"Best = {max(accuracies)*100:.1f}% (R{rounds[np.argmax(accuracies)]}), "
                  f"Worst = {min(accuracies)*100:.1f}% (R{rounds[np.argmin(accuracies)]}), "
                  f"Rounds plotted: {len(rounds)}")


if __name__ == "__main__":
    print("Model Comparison Plotting Tool")
    print("=" * 50)
    print(f"Comparing directories: {', '.join(EVAL_DIRS)}")
    print(f"Mode: {'Alice only' if ALICE_ONLY else 'All players'}")
    print()
    
    plot_comparison()
