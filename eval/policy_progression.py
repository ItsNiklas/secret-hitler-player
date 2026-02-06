"""
Policy Progression Comparison Tool

Compares liberal policy progression across multiple runsF1 models and Human data.
Similar to plot_comparison.py but focused on liberal policies over rounds.
Imports and reuses functions from gamestats.py.
"""

import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox

# Import functions from gamestats.py
from gamestats import enhanced_parse_game_data, calculate_policy_counts_by_round
from plot_config import setup_plot_style, ROLE_COLORS, extract_model_name, get_model_imagebox, get_model_color, get_markerdata_for_model

# Apply shared plotting configuration
setup_plot_style()

# Configuration: Add the folders you want to compare here
# 5 runsF1 models + Human data
EVAL_DIRS = [
    "runsF1-G3-12B",
    "runsF1-G3-27B",
    "runsF1-Llama33-70B",
    "runsF1-Q3",
    "runsF1-R1Distill-70B",
    "../crawl/summaries",  # Human data
]

LEGEND = False


def load_and_process_data(eval_dir, policy_type):
    """Load and process game data for a specific evaluation directory and policy type.
    
    Args:
        eval_dir: Directory containing game summary files
        policy_type: 'liberal' or 'fascist'
        
    Returns:
        tuple: (filtered_rounds, filtered_means) or (None, None) if no data
    """
    print(f"Processing {eval_dir}...")

    eval_files = glob.glob(f"{os.path.dirname(os.path.abspath(__file__))}/{eval_dir}/*summary.json")
    if not eval_files:
        print(f"No files found in {eval_dir}, skipping...")
        return None, None

    print(f"Found {len(eval_files)} files in {eval_dir}")

    # Load and parse each game file
    all_games_data = []

    for file_path in eval_files:
        # Skip annotation files
        if "annotat" in file_path.lower():
            continue
        try:
            with open(file_path, "r") as f:
                game_data = json.load(f)

            parsed_data = enhanced_parse_game_data(game_data)
            if "runs" in eval_dir:
                # For runsF1 models, filter to only include games where Alice is a specific role if needed
                if parsed_data["players"][0]["role"] == "liberal":
                    if policy_type != "liberal":
                        continue
                elif parsed_data["players"][0]["role"] in ["fascist", "hitler"]:
                    if policy_type != "fascist":
                        continue
            all_games_data.append(parsed_data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if not all_games_data:
        print(f"No valid games found in {eval_dir}, skipping...")
        return None, None

    print(f"Processed {len(all_games_data)} games from {eval_dir}")

    # Calculate policy progression using the refactored function
    policy_data = calculate_policy_counts_by_round(all_games_data, policy_type=policy_type)

    if not policy_data["rounds"]:
        print(f"No round data found in {eval_dir}, skipping...")
        return None, None

    # Filter rounds based on cutoff
    cutoff_round = policy_data["cutoff_round"]
    filtered_rounds = []
    filtered_means = []

    for round_num, mean_val in zip(policy_data["rounds"], policy_data["means"]):
        if round_num <= cutoff_round or cutoff_round >= policy_data["max_rounds"]:
            filtered_rounds.append(round_num)
            filtered_means.append(mean_val)

    if not filtered_rounds:
        print(f"No rounds with sufficient data found in {eval_dir}, skipping...")
        return None, None

    print(f"  Plotted {len(filtered_rounds)} rounds for {eval_dir} (rounds {min(filtered_rounds)}-{max(filtered_rounds)})")
    print(f"  Mean {policy_type} policies: {np.mean(filtered_means):.2f}")
    
    return filtered_rounds, filtered_means


def plot_policy_progression_comparison():
    """Plot policy progression comparison with subplots for liberal and fascist policies."""

    fig, (ax_lib, ax_fas) = plt.subplots(2, 1, figsize=(5.50, 4.5), sharex=True)

    all_rounds = set()
    liberal_data = {}
    fascist_data = {}

    # Process data for both policy types
    for eval_dir in EVAL_DIRS:
        # Load liberal data
        lib_rounds, lib_means = load_and_process_data(eval_dir, "liberal")
        if lib_rounds is not None:
            liberal_data[eval_dir] = (lib_rounds, lib_means)
            all_rounds.update(lib_rounds)
        
        # Load fascist data
        fas_rounds, fas_means = load_and_process_data(eval_dir, "fascist")
        if fas_rounds is not None:
            fascist_data[eval_dir] = (fas_rounds, fas_means)
            all_rounds.update(fas_rounds)

    if not liberal_data and not fascist_data:
        print("No data to plot!")
        return

    # Plot liberal policies (top subplot)
    for eval_dir in EVAL_DIRS:
        if eval_dir not in liberal_data:
            continue
            
        filtered_rounds, filtered_means = liberal_data[eval_dir]
        model_name = extract_model_name(eval_dir)
        color = get_model_color(model_name)
        m, ms = get_markerdata_for_model(model_name)
        
        ax_lib.plot(filtered_rounds, filtered_means, marker=m, linewidth=2, markersize=ms, 
                    color=color, markeredgecolor='white', markeredgewidth=1)

    ax_lib.set_ylabel("Avg. Liberal Policies", y=0.5)
    ax_lib.grid(True, alpha=0.3)
    ax_lib.set_yticks([1, 2, 3, 4, 5, 6])
    ax_lib.set_ylim(0, 6.6)
    ax_lib.axhline(y=5, color=ROLE_COLORS["liberal"], linestyle="--", alpha=0.8, zorder=-1)

    # Plot fascist policies (bottom subplot)
    for eval_dir in EVAL_DIRS:
        if eval_dir not in fascist_data:
            continue
            
        filtered_rounds, filtered_means = fascist_data[eval_dir]
        model_name = extract_model_name(eval_dir)
        color = get_model_color(model_name)
        m, ms = get_markerdata_for_model(model_name)
        
        ax_fas.plot(filtered_rounds, filtered_means, marker=m, linewidth=2, markersize=ms, 
                    color=color, markeredgecolor='white', markeredgewidth=1)

    ax_fas.set_ylabel("Avg. Fascist Policies", y=0.5)
    ax_fas.set_xlabel("Round")
    ax_fas.grid(True, alpha=0.3)
    ax_fas.set_yticks([1, 2, 3, 4, 5, 6])
    ax_fas.set_ylim(0, 6.6)
    ax_fas.axhline(y=6, color=ROLE_COLORS["fascist"], linestyle="--", alpha=0.8, zorder=-1)
    ax_fas.axhline(y=3, color=ROLE_COLORS["fascist"], linestyle=":", alpha=0.8, zorder=-1)

    # Set shared x-axis properties
    all_rounds_list = sorted(list(all_rounds))
    ax_fas.set_xticks(all_rounds_list)
    ax_fas.set_xlim(min(all_rounds_list) - 0.5, 11 + 0.5)

    # Add legend at the bottom with 3 columns
    # Create dummy lines for legend (use data from first available model)
    handles = []
    labels = []
    for eval_dir in EVAL_DIRS:
        if eval_dir in liberal_data or eval_dir in fascist_data:
            model_name = extract_model_name(eval_dir)
            color = get_model_color(model_name)
            m, ms = get_markerdata_for_model(model_name)
            line = plt.Line2D([0], [0], marker=m, color=color, linewidth=2, markersize=ms,
                            markeredgecolor='white', markeredgewidth=1)
            handles.append(line)
            labels.append(model_name)
    
    # Add legend below the bottom subplot
    legend = fig.legend(handles, labels, loc='lower center', ncol=3, 
                       framealpha=0, handlelength=2, handletextpad=1.4,
                       bbox_to_anchor=(0.5, -0.02))

    # Add logos to legend - need to draw first to get positions
    fig.canvas.draw()

    # Add logos to legend
    for eval_dir, legend_handle in zip(EVAL_DIRS, legend.legend_handles):
        if eval_dir not in liberal_data and eval_dir not in fascist_data:
            continue
        model_name = extract_model_name(eval_dir)
        imagebox = get_model_imagebox(model_name)
        if not imagebox:
            continue

        # Position the logo at the start of each legend entry
        ab = AnnotationBbox(
            imagebox,
            (0.5, 0.5),
            xybox=(19, 0),  # Offset to the left of text
            xycoords=legend_handle,
            boxcoords="offset points",
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=10,
        )
        fig.add_artist(ab)

    # Adjust layout to prevent clipping, leaving space for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save the plot
    dirs_suffix = "_".join([d.replace("/", "_").replace("\\", "_").split("/")[-1] for d in EVAL_DIRS])
    output_filename = f"policy_progression_comparison_{dirs_suffix}.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved as: {output_filename}")

    # Print summary statistics
    print("\nLiberal Policy Progression Summary:")
    for eval_dir in EVAL_DIRS:
        if eval_dir in liberal_data:
            rounds, means = liberal_data[eval_dir]
            model_name = extract_model_name(eval_dir)
            print(f"{model_name}: Avg liberal policies = {np.mean(means):.2f}, "
                  f"Final round mean = {means[-1]:.2f} (R{rounds[-1]}), "
                  f"Rounds plotted: {len(rounds)}")
    
    print("\nFascist Policy Progression Summary:")
    for eval_dir in EVAL_DIRS:
        if eval_dir in fascist_data:
            rounds, means = fascist_data[eval_dir]
            model_name = extract_model_name(eval_dir)
            print(f"{model_name}: Avg fascist policies = {np.mean(means):.2f}, "
                  f"Final round mean = {means[-1]:.2f} (R{rounds[-1]}), "
                  f"Rounds plotted: {len(rounds)}")


if __name__ == "__main__":
    print("Policy Progression Comparison Tool")
    print("=" * 50)
    print(f"Comparing directories: {', '.join(EVAL_DIRS)}")
    print()

    # Generate combined subplot plot
    print(f"\n{'='*50}")
    print(f"Generating policy progression plot (Liberal + Fascist)")
    print(f"{'='*50}")
    plot_policy_progression_comparison()
