#!/usr/bin/env python3
"""
Multi-folder gamestate evaluation comparison (2x2 grid).

Compares Alice's per-round gamestate evaluation scores across four
evaluation runs in a 2x2 subplot layout.

Usage: python stateeval_plot_max.py <folder1> <folder2> <folder3> <folder4>
  folder1..4  Paths to folders with *_summary.json files
"""

import json
from collections import defaultdict
from pathlib import Path
import argparse
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, Akima1DInterpolator, PchipInterpolator
from plot_config import extract_model_name, get_model_imagebox, setup_plot_style, load_summary_file, ROLE_COLORS, get_plot_path
from stateeval_plot import get_alice_role, extract_gamestate_scores

# Apply shared plotting configuration
setup_plot_style()

def process_folder(folder_path):
    """Process a single folder and return games_by_role data."""
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return None
    
    # Find all JSON summary files
    json_files = list(folder_path.glob("*_summary.json"))
    
    if not json_files:
        print(f"No summary files found in {folder_path}")
        return None
    
    print(f"Found {len(json_files)} summary files in {folder_path.name}")
    
    games_by_role = defaultdict(list)
    files_processed = 0
    
    # Process each file
    for file_path in json_files:
        summary = load_summary_file(file_path)
        if summary is None:
            continue
            
        alice_role = get_alice_role(summary)
        if alice_role is None:
            print(f"Warning: Could not find Alice's role in {file_path.name}")
            continue
            
        scores = extract_gamestate_scores(summary)
        if not scores:
            print(f"Warning: No gamestate scores found in {file_path.name}")
            continue
            
        games_by_role[alice_role].append({
            'filename': file_path.name,
            'scores': scores
        })
        files_processed += 1
    
    print(f"Successfully processed {files_processed} files from {folder_path.name}")
    return games_by_role


def plot_single_subplot(ax, games_by_role, folder_path, show_legend=False):
    """Plot gamestate evaluations for a single subplot."""
    # Calculate total number of games to adjust alpha
    total_games = sum(len(games) for games in games_by_role.values())
    # Normalize alpha: for 297 games alpha=0.1, scale inversely with number of games
    # Formula: alpha = (0.1 * 297) / total_games, clamped between 0.05 and 0.3
    adaptive_alpha = np.clip((0.1 * 297) / total_games, 0.05, 0.25)
    
    # Plot each game as a line, grouped by Alice's role
    for role, games in games_by_role.items():
        game_color = ROLE_COLORS.get(role, '#CCCCCC')
        
        # Collect data for mean calculation per role
        role_rounds_for_mean = []
        role_scores_for_mean = []
        
        for game in games:
            rounds = [s['round'] for s in game['scores']]
            scores = [s['score'] for s in game['scores']]
            
            bspl = PchipInterpolator(rounds, scores)
            x_smooth = np.linspace(min(rounds), max(rounds), 100)
            ax.plot(x_smooth, bspl(x_smooth), color=game_color, alpha=adaptive_alpha, linewidth=0.8)
        
            # Collect data for this role's mean calculation
            role_rounds_for_mean.extend(rounds)
            role_scores_for_mean.extend(scores)
        
        # Calculate and plot mean line for this role
        if role_rounds_for_mean and role_scores_for_mean:
            # Group scores by round for mean calculation
            scores_by_round = defaultdict(list)
            for round_num, score in zip(role_rounds_for_mean, role_scores_for_mean):
                scores_by_round[round_num].append(score)
            
            # Calculate mean for each round
            mean_rounds = sorted(scores_by_round.keys())
            mean_scores = [np.mean(scores_by_round[r]) for r in mean_rounds]
            
            # Plot mean line for this role (high z-order, not bold)
            ax.plot(mean_rounds, mean_scores, color=game_color, linewidth=2, marker="o", markersize=6, zorder=10, markeredgecolor='white', markeredgewidth=1)
    
    # Formatting
    ax.set_xlim(1, 9.5)
    ax.set_xticks(range(1, 10))
    ax.set_ylim(-1, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'+{v:.1f}' if v > 0 else (f'$-${abs(v):.1f}' if v < 0 else '0.0')))
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='0.85', linestyle='--', alpha=0.5, linewidth=0.5)

    model_name = extract_model_name(folder_path)
    imagebox = get_model_imagebox(model_name)
    
    if imagebox:
        # Draw the legend for model name
        legend2 = ax.legend(handles=[Line2D([0], [0], color='none', label=model_name)], 
                          loc='lower right', framealpha=1, handletextpad=-0.4)
        ax.add_artist(legend2)  # Add the new legend without removing the first one
        
        plt.gcf().canvas.draw()
        ab = AnnotationBbox(imagebox, (0.5, 0.5), xybox=(-3, 0), 
                           xycoords=legend2.legend_handles[0], boxcoords="offset points",
                           frameon=False, box_alignment=(0.5, 0.5), zorder=10)
        ax.add_artist(ab)
    
    if show_legend:
        # Create legend with role counts
        legend_elements = []
        for role in reversed(games_by_role.keys()):
            color = ROLE_COLORS.get(role, '#CCCCCC')
            legend_elements.append(Line2D([0], [0], color=color, lw=2, marker='o', markersize=6,
                                            markeredgecolor='white', markeredgewidth=1,
                                            label=f'{role.capitalize()}'))
        
        ax.legend(handles=legend_elements, loc='lower right', framealpha=1)
        
        # Add text annotations explaining the score
        ax.text(0.02, 0.98, r'$\uparrow$ Liberal Advantage', transform=ax.transAxes, 
                 verticalalignment='top', fontsize=9, alpha=0.7)
        ax.text(0.02, 0.02, r'$\downarrow$ Fascist Advantage', transform=ax.transAxes, 
                 verticalalignment='bottom', fontsize=9, alpha=0.7)


def plot_gamestate_evaluations(folders):
    """Plot gamestate evaluation scores for Alice across all games in four folders."""
    if len(folders) != 4:
        print(f"Error: Expected 4 folders, got {len(folders)}")
        return
    
    folder_paths = [Path(f) for f in folders]
    
    # Process all folders
    all_games_by_role = []
    for folder_path in folder_paths:
        games_by_role = process_folder(folder_path)
        if games_by_role is None:
            return
        all_games_by_role.append(games_by_role)
    
    # Create 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(5.50, 4), sharex=True, sharey=True)
    axs = axs.flatten()
    
    # Plot each folder in a subplot
    for idx, (ax, games_by_role, folder_path) in enumerate(zip(axs, all_games_by_role, folder_paths)):
        # Show legend only in the first subplot (top-left)
        show_legend = False#(idx == 0)
        plot_single_subplot(ax, games_by_role, folder_path, show_legend=show_legend)
    
    # Add common labels
    fig.text(0.5, 0.04, 'Round', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Game State Evaluation Score', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    
    # Save the plot as PDF
    out_path = get_plot_path("stateeval_plot_max.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot gamestate evaluation scores for Alice from Secret Hitler game summary files in 2x2 subplots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stateeval_plot_max.py runsA1-Base/ runsA2-CoT/ runsA3-Memory/ runsA4-RoleMsg/
  python stateeval_plot_max.py folder1/ folder2/ folder3/ folder4/
        """
    )
    
    parser.add_argument(
        'folders',
        nargs=4,
        help='Paths to four folders containing *_summary.json files'
    )
    
    args = parser.parse_args()
    
    plot_gamestate_evaluations(args.folders)


if __name__ == "__main__":
    main()