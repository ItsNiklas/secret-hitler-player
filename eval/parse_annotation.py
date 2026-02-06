import argparse
import json
import os
from collections import defaultdict
from turtle import color
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from gamestats import parse_game_data
from matplotlib.offsetbox import AnnotationBbox
from plot_config import (
    setup_plot_style, ROLE_COLORS, UNIBLAU, GAMMA, ETA, 
    extract_model_name, get_model_imagebox, perform_chi_square_test, load_techniques
)

# Apply shared plotting configuration
setup_plot_style()

# Load techniques from JSON file
TECHNIQUES = load_techniques("persuasion_cialdini.jsonl") + load_techniques("persuasion_sh.jsonl") + load_techniques("persuasion_jailbreak.jsonl")


def parse_speaker(text):
    """Extracts the speaker's name from a chat message."""
    if ":" in text:
        return text.split(":", 1)[0]
    return None


def perform_chi_square_test_by_role(by_role_df, role_message_counts):
    """Perform chi-square test for homogeneity on annotation counts by role.
    
    Aggregates fascist and hitler roles together and performs chi-square test 
    to test if annotation patterns differ significantly between liberals and fascists.
    
    Args:
        by_role_df (pd.DataFrame): DataFrame with roles as index and annotations as columns
        role_message_counts (dict): Dictionary with message counts per role
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = by_role_df.copy()
    
    # Aggregate fascist and hitler roles together
    if 'fascist' in df.index and 'hitler' in df.index:
        fascist_totals = df.loc['fascist'] + df.loc['hitler']
        df = df.drop(['fascist', 'hitler'])
        df.loc['fascist_combined'] = fascist_totals
    elif 'fascist' in df.index:
        df = df.rename(index={'fascist': 'fascist_combined'})
    elif 'hitler' in df.index:
        df = df.rename(index={'hitler': 'fascist_combined'})
    
    # Get the contingency table for the test
    contingency_table = df.loc[['liberal', 'fascist_combined']]
    
    perform_chi_square_test(contingency_table, "Chi-Square Test: Role Analysis", "liberal", "fascist")

def aggregate_annotations(folder_path):
    """Aggregates annotations from JSON files in a given folder.

    Args:
        folder_path (str): The path to the folder containing the data files.
    """
    annotation_counts = defaultdict(int)
    role_annotation_counts = defaultdict(lambda: defaultdict(int))
    all_unique_reasons = set()
    role_message_counts = defaultdict(int)
    role_total_techniques = defaultdict(int)
    role_player_counts = defaultdict(int)

    winning_annotation_counts = defaultdict(int)
    losing_annotation_counts = defaultdict(int)
    high_elo_annotation_counts = defaultdict(int)
    low_elo_annotation_counts = defaultdict(int)

    num_winning_players = 0
    num_losing_players = 0
    num_high_elo_games = 0
    num_low_elo_games = 0

    for filename in os.listdir(folder_path):
        if filename.endswith("-chat-annotated.json"):
            annotated_filepath = os.path.join(folder_path, filename)
            summary_filename = filename.replace("-chat-annotated.json", ".json").replace("xhr_data", "summary")
            summary_filepath = os.path.join(folder_path, summary_filename)

            if not os.path.exists(summary_filepath):
                # Check if the summary file exists in the parent directory under replay_data
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(folder_path)))
                replay_data_path = os.path.join(parent_dir, "summaries", summary_filename)
                if os.path.exists(replay_data_path):
                    summary_filepath = replay_data_path
                else:
                    print(f"Warning: Corresponding summary file not found for {filename} {replay_data_path}")
                    continue

            with open(summary_filepath, 'r') as f:
                summary_data = json.load(f)
            
            with open(annotated_filepath, 'r') as f:
                annotated_data = json.load(f)

            game_stats = parse_game_data(summary_data)
            winner = game_stats['winner']
            num_players = len(summary_data.get('players', []))
            num_fascists = sum(1 for p in summary_data.get('players', []) if p['role'] in ['fascist', 'hitler'])
            num_liberals = num_players - num_fascists

            # print(f"Processing {filename}: Winner = {winner}, Players = {num_players} (Liberals: {num_liberals})")

            if winner == 'liberals':
                num_winning_players += num_liberals
                num_losing_players += num_fascists
            elif winner == 'fascists':
                num_winning_players += num_fascists
                num_losing_players += num_liberals

            # ELO calculation
            lib_elo = summary_data.get("libElo", {}).get("overall", 0)
            fas_elo = summary_data.get("fasElo", {}).get("overall", 0)
            avg_elo = (lib_elo + fas_elo) / 2 if (lib_elo + fas_elo) > 0 else 0
            is_high_elo = avg_elo > 1650

            if is_high_elo:
                num_high_elo_games += 1
            else:
                num_low_elo_games += 1

            # Create a mapping from username to role
            role_map = {player['username']: player['role'] for player in summary_data.get('players', [])}
            
            # Count players by role
            for player in summary_data.get('players', []):
                role_player_counts[player['role']] += 1

            for item in annotated_data:
                text = item.get("text")
                if not text:
                    continue
                
                speaker = parse_speaker(text)
                if not speaker or speaker not in role_map:
                    continue

                role = role_map[speaker]
                role_message_counts[role] += 1
                annotations = item.get("annotation", [])

                # Flatten list in case of nesting and get unique annotations
                flat_annotations = []
                for ann in annotations:
                    if isinstance(ann, list):
                        flat_annotations.extend(ann)
                    else:
                        flat_annotations.append(ann)
                
                # Filter annotations to only include valid techniques
                unique_annotations = {ann for ann in flat_annotations if ann in TECHNIQUES}
                role_total_techniques[role] += len(unique_annotations)
                all_unique_reasons.update(unique_annotations)

                is_winner = (winner == 'liberals' and role == 'liberal') or \
                            (winner == 'fascists' and role in ['fascist', 'hitler'])

                for annotation in unique_annotations:
                    annotation_counts[annotation] += 1
                    role_annotation_counts[role][annotation] += 1
                    if is_winner:
                        winning_annotation_counts[annotation] += 1
                    else:
                        losing_annotation_counts[annotation] += 1
                    
                    if is_high_elo:
                        high_elo_annotation_counts[annotation] += 1
                    else:
                        low_elo_annotation_counts[annotation] += 1

    # Create pandas DataFrames for better visualization
    # Overall counts
    overall_df = pd.DataFrame.from_dict(annotation_counts, orient='index', columns=['count'])
    overall_df = overall_df.sort_values(by='count', ascending=False)

    # By role counts
    by_role_df = pd.DataFrame.from_dict(role_annotation_counts, orient='index')
    by_role_df = by_role_df.fillna(0).astype(int)
    by_role_df = by_role_df.sort_index()


    print("--- Annotation Counts ---")
    print(overall_df)
    print("\n--- Annotation Counts by Role ---")
    print(by_role_df)
    perform_chi_square_test_by_role(by_role_df, role_message_counts)

    print("\n--- Average Persuasion Techniques per Message by Role ---")
    for role in sorted(role_message_counts.keys()):
        if role_message_counts[role] > 0:
            average = role_total_techniques[role] / role_message_counts[role]
            print(f"{role}: {average:.3f} techniques listed per message")



    # Get top 10 annotations and define colors
    top_10_df = overall_df.head(10)
    top_10_annotations = top_10_df.index.tolist()

    # 1. Plot for overall counts (Top 10)
    fig, ax = plt.subplots(figsize=(5.50, 3))
    sorted_data = top_10_df['count'].sort_values()
    bars = ax.barh(range(len(sorted_data)), sorted_data.values, height=0.65, color=UNIBLAU, zorder=5)
    ax.set_yticks(range(len(sorted_data)))
    ax.set_yticklabels(sorted_data.index)
    ax.grid(axis='x', zorder=-5, color='0.85')
    # ax.set_title('Top Annotation Categories (Overall)')
    ax.set_xlabel('Count')
    
    # Add numbers to bars
    max_value = sorted_data.max()
    for i, (bar, value) in enumerate(zip(bars, sorted_data.values)):
        if value == max_value:
            # Place number inside the bar with white color for the largest bar
            ax.text(value - 0.01 * max_value, bar.get_y() + bar.get_height()/2.2, 
                   f'{int(value)}', va='center', ha='right', color='white', fontsize=8, zorder=10)
        else:
            # Place number outside the bar for other bars
            ax.text(value + 0.01 * max_value, bar.get_y() + bar.get_height()/2.2, 
                   f'{int(value)}', va='center', ha='left', fontsize=8)
    
    # Add model name with icon as a fake legend in bottom right corner
    model_name = extract_model_name(folder_path)
    imagebox = get_model_imagebox(model_name)
    if imagebox:
        legend = ax.legend(handles=[Line2D([0], [0], color='none', label=model_name)], 
                        loc='lower right', framealpha=1, handletextpad=-0.4)
        fig.canvas.draw()
        
        ab = AnnotationBbox(imagebox, (0.5, 0.5), xybox=(-3, 0), 
                        xycoords=legend.legend_handles[0], boxcoords="offset points",
                        frameon=False, box_alignment=(0.5, 0.5), zorder=10)
        fig.add_artist(ab)
    
    plt.tight_layout()
    plt.savefig(f'{folder_path.split("/")[-2]}_overall_annotation_counts.pdf')
    plt.close()
    print("\nSaved overall annotation counts plot to 'overall_annotation_counts.pdf'")


    # 2. Plot for by-role counts (Top 10 categories, normalized)
    # Normalize by role player counts
    by_role_normalized_df = by_role_df.copy().astype(float)
    for role in by_role_normalized_df.index:
        if role_player_counts[role] > 0:
            by_role_normalized_df.loc[role] = by_role_normalized_df.loc[role] / role_player_counts[role]
        else:
            by_role_normalized_df.loc[role] = 0
    
    # Filter to only include top 10 annotations
    by_role_top_10_df = by_role_normalized_df[top_10_annotations]
    
    # Prepare data for plotting: techniques sorted by total usage
    # Calculate total usage for sorting (sum across all roles)
    technique_totals = by_role_top_10_df.sum(axis=0).sort_values(ascending=True)
    sorted_techniques = technique_totals.index.tolist()
    
    # Create the plot
    fig = plt.figure(figsize=(5.50, 4))
    ax = fig.add_subplot(111)
    
    # Set up bar positions
    bar_height = 0.25
    y_pos = np.arange(len(sorted_techniques))
    
    # Plot bars for each role
    for i, role in enumerate(['liberal', 'fascist', 'hitler']):
        if role in by_role_top_10_df.index:
            values = [by_role_top_10_df.loc[role, tech] if tech in by_role_top_10_df.columns else 0 
                     for tech in sorted_techniques]
            # Offset: liberal at top (-1), fascist in middle (0), hitler at bottom (+1)
            offset = -(i - 1) * bar_height
            ax.barh(y_pos + offset, values, bar_height, 
                   label=role.capitalize(), color=ROLE_COLORS[role], zorder=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_techniques)
    ax.set_xlabel('Average Uses of Technique Per Player')
    ax.grid(axis='x', zorder=-5, color='0.85')
    
    # Add model name with icon as a fake legend in corner
    model_name = extract_model_name(folder_path)
    imagebox = get_model_imagebox(model_name)
    
    fig.canvas.draw()

    if imagebox:
        # Draw the legend for roles
        legend2 = ax.legend(handles=[Line2D([0], [0], color='none', label=model_name)], 
                          loc='lower right', framealpha=1, handletextpad=-0.4)
        ax.add_artist(legend2)  # Add the new legend without removing the first one
        
        ab = AnnotationBbox(imagebox, (0.5, 0.5), xybox=(-3, 0), 
                           xycoords=legend2.legend_handles[0], boxcoords="offset points",
                           frameon=False, box_alignment=(0.5, 0.5), zorder=10)
        fig.add_artist(ab)
        
    # Re-add the role legend (it gets overwritten by the second legend call)
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.23), framealpha=1)
    
    plt.tight_layout()
    plt.savefig(f'{folder_path.split("/")[-2]}_by_role_annotation_counts.pdf')
    plt.close()
    print("Saved by-role annotation counts plot to 'by_role_annotation_counts.pdf'")




    # 3. Plot for winning vs losing players (Normalized)
    winning_df = pd.DataFrame.from_dict(winning_annotation_counts, orient='index', columns=['count'])
    if num_winning_players > 0:
        winning_df['normalized_count'] = winning_df['count'] / num_winning_players
    else:
        winning_df['normalized_count'] = 0

    losing_df = pd.DataFrame.from_dict(losing_annotation_counts, orient='index', columns=['count'])
    if num_losing_players > 0:
        losing_df['normalized_count'] = losing_df['count'] / num_losing_players
    else:
        losing_df['normalized_count'] = 0
    
    # Combine winning and losing into one DataFrame for easier plotting
    combined_df = pd.concat([winning_df.rename(columns={'normalized_count': 'Winning'}), 
                             losing_df.rename(columns={'normalized_count': 'Losing'})], axis=1).fillna(0)
    combined_df = combined_df.loc[top_10_annotations].sort_values(by='Winning', ascending=True)

    fig, ax = plt.subplots(figsize=(5.50, 3))
    combined_df[['Losing', 'Winning']].plot(kind='barh', ax=ax, zorder=5, color=[GAMMA, ETA])
    ax.grid(axis='x', zorder=-5, color='0.85')
    # ax.set_title('Normalized Annotation Counts for Winning vs. Losing Players (Top 10 Overall)')
    ax.set_xlabel('Average Uses of Technique Per Player')
    
    # Add model name with icon as a fake legend in corner
    model_name = extract_model_name(folder_path)
    imagebox = get_model_imagebox(model_name)
    
    fig.canvas.draw()

    if imagebox:
        # Draw the legend for model name
        legend2 = ax.legend(handles=[Line2D([0], [0], color='none', label=model_name)], 
                          loc='lower right', framealpha=1, handletextpad=-0.4)
        ax.add_artist(legend2)  # Add the new legend without removing the first one
        
        ab = AnnotationBbox(imagebox, (0.5, 0.5), xybox=(-3, 0), 
                           xycoords=legend2.legend_handles[0], boxcoords="offset points",
                           frameon=False, box_alignment=(0.5, 0.5), zorder=10)
        fig.add_artist(ab)
        
    # Re-add the winning/losing legend (it gets overwritten by the second legend call)
    # Reverse the legend order to match the visual bar order (top to bottom)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='center right', bbox_to_anchor=(1.0, 0.29), framealpha=1)
    
    plt.tight_layout()
    plt.savefig(f'{folder_path.split("/")[-2]}_win_lose_annotation_counts.pdf')
    plt.close()
    print("Saved winning/losing annotation counts plot to 'win_lose_annotation_counts.pdf'")

    # Statistical test for winning vs losing players
    winning_losing_df = pd.DataFrame({
        'Winning': pd.Series(winning_annotation_counts),
        'Losing': pd.Series(losing_annotation_counts)
    }).fillna(0).astype(int)
    perform_chi_square_test(winning_losing_df.T, "Chi-Square Test: Winning vs Losing Players", "winning", "losing")

    if num_high_elo_games != 0 and num_low_elo_games != 0:
        # 4. Plot for high vs low ELO (Normalized)

        high_elo_df = pd.DataFrame.from_dict(high_elo_annotation_counts, orient='index', columns=['count'])
        high_elo_df['normalized_count'] = high_elo_df['count'] / num_high_elo_games


        low_elo_df = pd.DataFrame.from_dict(low_elo_annotation_counts, orient='index', columns=['count'])
        low_elo_df['normalized_count'] = low_elo_df['count'] / num_low_elo_games

        
        combined_elo_df = pd.concat([high_elo_df.rename(columns={'normalized_count': r'High ELO ($>$1650)'}), 
                                    low_elo_df.rename(columns={'normalized_count': r'Low ELO ($\leq$1650)'})], axis=1).fillna(0)
        combined_elo_df = combined_elo_df.loc[top_10_annotations].sort_values(by=r'High ELO ($>$1650)', ascending=True)

        fig, ax = plt.subplots(figsize=(5.50, 3))
        combined_elo_df[[r'High ELO ($>$1650)', r'Low ELO ($\leq$1650)']].plot(kind='barh', ax=ax, zorder=5, color=[GAMMA, ETA])
        ax.grid(axis='x', zorder=-5, color='0.85')
        # ax.set_title('Normalized Annotation Counts for High vs. Low ELO Games (Top 10 Overall)')
        ax.set_xlabel('Average Uses of Technique Per Game')
        # ax.set_ylabel('Persuasion Technique')
        
        # Add model name with icon as a fake legend in corner
        model_name = extract_model_name(folder_path)
        imagebox = get_model_imagebox(model_name)
        
        fig.canvas.draw()

        if imagebox:
            # Draw the legend for model name
            legend2 = ax.legend(handles=[Line2D([0], [0], color='none', label=model_name)], 
                              loc='lower right', framealpha=1, handletextpad=-0.4)
            ax.add_artist(legend2)  # Add the new legend without removing the first one
            
            ab = AnnotationBbox(imagebox, (0.5, 0.5), xybox=(-3, 0), 
                               xycoords=legend2.legend_handles[0], boxcoords="offset points",
                               frameon=False, box_alignment=(0.5, 0.5), zorder=10)
            fig.add_artist(ab)
            
        # Re-add the ELO legend (it gets overwritten by the second legend call)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='center right', bbox_to_anchor=(1.0, 0.29), framealpha=1)
        
        plt.tight_layout()
        plt.savefig(f'{folder_path.split("/")[-2]}_elo_annotation_counts.pdf')
        plt.close()
        print("Saved ELO-based annotation counts plot to 'elo_annotation_counts.pdf'")

        # Statistical test for high-elo vs low-elo players
        high_low_elo_df = pd.DataFrame({
            'High_ELO': pd.Series(high_elo_annotation_counts),
            'Low_ELO': pd.Series(low_elo_annotation_counts)
        }).fillna(0).astype(int)
        perform_chi_square_test(high_low_elo_df.T, "Chi-Square Test: High ELO vs Low ELO Players", "high ELO", "low ELO")

def main():
    parser = argparse.ArgumentParser(description="Aggregate annotations from Secret Hitler game logs.")
    parser.add_argument("folder", type=str, help="Path to the folder containing the JSON game logs.")
    args = parser.parse_args()

    aggregate_annotations(args.folder)

if __name__ == "__main__":
    main()
