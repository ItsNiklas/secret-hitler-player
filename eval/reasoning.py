from functools import cache
import os
import glob
import json
import sys
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

# EVAL_DIR = "runs2-Llama318"
# EVAL_DIR = "runs1-Qwen3"
# EVAL_DIR = "runs3-Llama3370"
# EVAL_DIR = "runs4-Random"
# EVAL_DIR = "runs5-Rule1"

EVAL_DIR = sys.argv[1] if len(sys.argv) > 1 else None

# Predefined reasoning categories mapping
REASONING_CATEGORIES = {
    "A": "Recent policy (e.g., laws passed, voting outcomes)",
    "B": "Probability-based reasoning (e.g., statistical likelihood, pattern recognition)",
    "C": "Statements made by other players",
    "D": "Random guess / intuition",
    "NONE": "Doesn't fit — propose a new category"
}

# For legend display - shorter versions
REASONING_LABELS = {
    "A": "Recent policy",
    "B": "Probability-based reasoning", 
    "C": "Player statements",
    "D": "Random guess / intuition",
    "NONE": "Other / Doesn't fit"
}

from plot_config import setup_plot_style

# Apply shared plotting configuration
setup_plot_style()


def load_eval_run_filenames() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(f"{current_dir}/{EVAL_DIR}/*.json")


def extract_final_reason(reflection_text: str) -> Optional[str]:
    """Extract the final reason from a reflection text and map to predefined categories A, B, C, D, NONE"""
    if not reflection_text:
        return None
    
    # Look for patterns like "Reasoning Category: X" at the end of the text
    match = re.search(r'Reasoning Category:\s*([^.\n]+)', reflection_text, re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        
        # Clean up the reason text
        # Remove trailing quotes
        reason = re.sub(r'["\'`"]+$', '', reason)
        # Remove trailing stars
        reason = re.sub(r'\*+$', '', reason)
        # Remove prefix stars
        reason = re.sub(r'^\*+', '', reason)
        # Remove content in parentheses that appears to be cut off (ends with single letter)
        reason = re.sub(r'\s*\([a-zA-Z]\s*$', '', reason)
        # Remove any trailing whitespace again after cleanup
        reason = reason.strip().upper()
        
        if not reason:
            return None
            
        # Map to predefined categories (A, B, C, D, NONE)
        if reason in REASONING_CATEGORIES:
            return reason
        
        # If no exact match, return None
        return None
    
    return None


def extract_final_reason(reflection_text: str) -> Optional[str]:
    """Extract the final reason from a reflection text that ends with 'Reason: X'"""
    if not reflection_text:
        return None
    
    # Look for patterns like "Reason: X" at the end of the text
    match = re.search(r'Reasoning Category:\s*([^.\n]+)', reflection_text, re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        
        # Clean up the reason text
        # Remove trailing quotes
        reason = re.sub(r'["\'`”]+$', '', reason)

        # Remove trailing stars
        reason = re.sub(r'\*+$', '', reason)

        # Remove prefix stars
        reason = re.sub(r'^\*+', '', reason)
        
        # Remove content in parentheses that appears to be cut off (ends with single letter)
        reason = re.sub(r'\s*\([a-zA-Z]\s*$', '', reason)
        
        # Remove any trailing whitespace again after cleanup
        reason = reason.strip()
        
        return reason if reason else None
    
    return None


def analyze_reflections(game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and analyze reflections from a game's logs, filtering for Liberal players only"""
    reflections_data = []
    
    # Create mapping from player index to role
    player_roles = {}
    for index, player in enumerate(game_data.get('players', [])):
        player_roles[str(index)] = player['role']
    
    logs = game_data.get('logs', [])
    total_rounds = len(logs)
    
    for round_num, log_entry in enumerate(logs):
        reflections = log_entry.get('reflections', {})
        
        for player_id, reflection_text in reflections.items():
            # Only process reflections from Liberal players
            if player_id in player_roles and player_roles[player_id] == 'liberal':
                if reflection_text:  # Only process non-empty reflections
                    final_reason = extract_final_reason(reflection_text)
                    
                    reflections_data.append({
                        'player_id': player_id,
                        'reflection_text': reflection_text,
                        'final_reason': final_reason,
                        'president_id': log_entry.get('presidentId'),
                        'chancellor_id': log_entry.get('chancellorId'),
                        'enacted_policy': log_entry.get('enactedPolicy'),
                        'round_number': round_num + 1,  # 1-indexed
                        'total_rounds': total_rounds,
                    })
    
    return reflections_data


def generate_reasoning_statistics(all_reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistics about reasoning patterns for predefined categories A, B, C, D, NONE"""
    stats = {
        'total_reflections': len(all_reflections),
        'final_reasons_counter': Counter(),
        'reflections_with_reasons': 0,
        'reasoning_by_round': defaultdict(lambda: {
            'final_reasons_counter': Counter(),
            'total_reflections': 0
        })
    }
    
    for reflection in all_reflections:
        final_reason = reflection['final_reason']
        round_number = reflection['round_number']
        
        # Count reflections by round
        stats['reasoning_by_round'][round_number]['total_reflections'] += 1
        
        # Process final reasons (only count predefined categories)
        if final_reason and final_reason in REASONING_CATEGORIES:
            stats['reflections_with_reasons'] += 1
            stats['final_reasons_counter'][final_reason] += 1
            
            # Round statistics
            stats['reasoning_by_round'][round_number]['final_reasons_counter'][final_reason] += 1
    
    return stats


def plot_reasoning_over_rounds(stats: Dict[str, Any], eval_dir: str):
    """Create a matplotlib plot showing reasoning patterns over rounds for categories A, B, C, D, NONE"""
    
    # Use predefined categories in a specific order
    predefined_categories = ["A", "B", "C", "D", "NONE"]
    
    # Get all rounds that have data
    rounds = sorted([r for r in stats['reasoning_by_round'].keys() if stats['reasoning_by_round'][r]['total_reflections'] > 0])

    if not rounds:
        print("No data available for plotting")
        return
    
    # Calculate percentages for each category by round
    reason_percentages = {reason: [] for reason in predefined_categories}
    
    for round_num in rounds:
        round_data = stats['reasoning_by_round'][round_num]
        total_reasons_in_round = sum(round_data['final_reasons_counter'].values())
        
        for reason in predefined_categories:
            if total_reasons_in_round > 0:
                percentage = (round_data['final_reasons_counter'][reason] / total_reasons_in_round) * 100
            else:
                percentage = 0
            reason_percentages[reason].append(percentage)
    
    # Create the plot
    plt.figure(figsize=(6.46, 4))
    
    # Plot each reason as a line
    for i, reason in enumerate(predefined_categories):
        # Use the readable label for the legend
        legend_name = f"{reason}: {REASONING_LABELS[reason]}"
        plt.plot(rounds, reason_percentages[reason], 
                marker='o', linewidth=2, markersize=4, 
                label=legend_name, alpha=0.8)
    
    plt.xlabel('Round Number')
    plt.ylabel('Percentage of Reflections (\%)')
    # plt.title(f'Liberal Players Reasoning Patterns Over Game Rounds - {eval_dir}')
    plt.grid()
    plt.legend(loc='best')
    
    # Set x-axis to show integer round numbers
    if rounds:
        plt.xticks(rounds)
        plt.xlim(0.5, max(rounds) + 0.5)
    
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    # Save the plot with sanitized filename
    sanitized_eval_dir = eval_dir.replace('/', '_').replace('\\', '_')
    output_filename = f"liberal_reasoning_over_rounds_{sanitized_eval_dir}.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_filename}")
    
    # Show the plot
    plt.show()


def print_statistics(stats: Dict[str, Any]):
    """Print formatted statistics about reasoning patterns for categories A, B, C, D, NONE"""
    print(f"\n=== LIBERAL PLAYERS REASONING ANALYSIS STATISTICS ===")
    print(f"Total reflections found (Liberal players only): {stats['total_reflections']}")
    print(f"Reflections with final reasons: {stats['reflections_with_reasons']}")
    
    print(f"\n--- Reasoning Categories (A, B, C, D, NONE) ---")
    for category in ["A", "B", "C", "D", "NONE"]:
        count = stats['final_reasons_counter'][category]
        description = REASONING_CATEGORIES[category]
        print(f"{category}: {count} times - {description}")
    
    print(f"\n--- Category Definitions ---")
    for category, description in REASONING_CATEGORIES.items():
        print(f"{category}: {description}")


if __name__ == "__main__":

    if not EVAL_DIR:
        print("Please provide evaluation directory as argument")
        print("Usage: python reasoning.py <eval_dir>")
        sys.exit(1)

    eval_files = load_eval_run_filenames()
    print(f"Found {len(eval_files)} files in eval/{EVAL_DIR}")

    # Load and parse each game file
    all_games_data = []
    all_reflections = []

    for file_path in eval_files:
        with open(file_path, "r") as f:
            game_data = json.load(f)
            all_games_data.append(game_data)
            
            # Extract reflections from this game
            game_reflections = analyze_reflections(game_data)
            all_reflections.extend(game_reflections)

    print(f"Total games processed: {len(all_games_data)}")
    print(f"Total reflections extracted: {len(all_reflections)}")
    
    # Generate and display statistics
    stats = generate_reasoning_statistics(all_reflections)
    print_statistics(stats)
    
    # Create and save the plot
    plot_reasoning_over_rounds(stats, EVAL_DIR)
