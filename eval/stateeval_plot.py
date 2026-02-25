#!/usr/bin/env python3
"""
Gamestate evaluation plotter.

Plots Alice's per-round gamestate evaluation scores (split by role) from
JSON summary files, with smoothed trend lines.

Usage: python stateeval_plot.py <summaries_folder>
  summaries_folder  Path to folder with *_summary.json files
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
from plot_config import FIG_WIDTH, extract_model_name, get_model_imagebox, setup_plot_style, load_summary_file, ROLE_COLORS, get_plot_path

# Apply shared plotting configuration
setup_plot_style()


def get_alice_role(summary):
    """Extract Alice's role from the game summary."""
    if not summary or 'players' not in summary:
        return None
    
    for player in summary['players']:
        if player.get('username') == 'Alice':
            return player.get('role')
    return None


def extract_gamestate_scores(summary):
    """Extract gamestate evaluation scores by round from a summary file."""
    if not summary or 'logs' not in summary:
        return []
    
    scores = []
    for round_idx, round_data in enumerate(summary['logs'], 1):
        if 'gameStateScore' in round_data:
            scores.append({
                'round': round_idx,
                'score': round_data['gameStateScore']
            })
    return scores


def plot_gamestate_evaluations(summaries_folder):
    """Plot gamestate evaluation scores for Alice across all games in the folder."""
    folder_path = Path(summaries_folder)
    
    if not folder_path.exists():
        print(f"Error: Folder {summaries_folder} does not exist")
        return
    
    # Find all JSON summary files
    json_files = list(folder_path.glob("*_summary.json"))
    
    if not json_files:
        print(f"No summary files found in {summaries_folder}")
        return
    
    print(f"Found {len(json_files)} summary files")
    
    # Create the plot
    plt.figure(figsize=(FIG_WIDTH, 3))
    
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
    
    print(f"Successfully processed {files_processed} files")
    
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
            plt.plot(x_smooth, bspl(x_smooth), color=game_color, alpha=0.1, linewidth=0.8)
        
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
            plt.plot(mean_rounds, mean_scores, color=game_color, linewidth=2, marker="o", markersize=6, zorder=10, markeredgecolor='white', markeredgewidth=1)
    
    # Create legend with role counts (3 fields)
    legend_elements = []
    for role in reversed(games_by_role.keys()):
        color = ROLE_COLORS.get(role, '#CCCCCC')
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, marker='o', markersize=6,
                                        markeredgecolor='white', markeredgewidth=1,
                                        label=f'{role.capitalize()}'))
    
    # Formatting
    plt.xlabel('Round')
    plt.ylabel('Game State Evaluation Score')
    plt.xlim(1, 9.5)
    plt.xticks(range(1, 10))  # Tick everything from 1 to 11
    plt.ylim(-1, 1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'+{v:.1f}' if v > 0 else (f'$-${abs(v):.1f}' if v < 0 else '0.0')))
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='0.85', linestyle='--', alpha=0.5, linewidth=0.5)

    model_name = extract_model_name(folder_path)
    imagebox = get_model_imagebox(model_name)
    
    plt.gcf().canvas.draw()

    if imagebox:
        # Draw the legend for roles
        legend2 = plt.gca().legend(handles=[Line2D([0], [0], color='none', label=model_name)], 
                          loc='upper right', framealpha=1, handletextpad=-0.4)
        plt.gca().add_artist(legend2)  # Add the new legend without removing the first one
        
        ab = AnnotationBbox(imagebox, (0.5, 0.5), xybox=(-3, 0), 
                           xycoords=legend2.legend_handles[0], boxcoords="offset points",
                           frameon=False, box_alignment=(0.5, 0.5), zorder=10)
        plt.gcf().add_artist(ab)
    
    plt.legend(handles=legend_elements, loc='lower right', framealpha=1)
    # Add text annotations explaining the score
    plt.text(0.02, 0.98, r'$\uparrow$ Liberal Advantage', transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=9, alpha=0.7)
    plt.text(0.02, 0.02, r'$\downarrow$ Fascist Advantage', transform=plt.gca().transAxes, 
             verticalalignment='bottom', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot as PDF
    model_slug = extract_model_name(folder_path).replace(' ', '_').lower()
    output_filename = f"stateeval_plot_{model_slug}.pdf"
    out_path = get_plot_path(output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot gamestate evaluation scores for Alice from Secret Hitler game summary files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stateeval_plot.py ../crawl/summaries/
  python stateeval_plot.py /path/to/summaries/folder/
  python stateeval_plot.py runsA2-CoT/
        """
    )
    
    parser.add_argument(
        'summaries_folder',
        help='Path to the folder containing *_summary.json files'
    )
    
    args = parser.parse_args()
    
    plot_gamestate_evaluations(args.summaries_folder)


if __name__ == "__main__":
    main()