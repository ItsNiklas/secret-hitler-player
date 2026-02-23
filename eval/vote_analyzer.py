#!/usr/bin/env python3
"""
Vote analysis for Secret Hitler game summaries.

Aggregates yes/no vote statistics, per-round voting patterns, and Elo
ratings from JSON summary files.

Usage: python vote_analyzer.py <summaries_folder>
  summaries_folder  Path to folder with *_summary.json files
"""

import json
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from plot_config import setup_plot_style, extract_model_name, get_plot_path, load_summary_file, UNIBLAU, GAMMA, ETA

# Apply shared plotting configuration
setup_plot_style()


def get_combined_elo(summary):
    """Extract and compute combined ELO from libElo and fasElo if available.
    
    Args:
        summary: The parsed JSON summary data
    
    Returns:
        float or None: Average of overall libElo and fasElo, or None if not available
    """
    if not summary:
        return None
    
    lib_elo = summary.get('libElo', {}).get('overall')
    fas_elo = summary.get('fasElo', {}).get('overall')
    
    if lib_elo is not None and fas_elo is not None:
        return (lib_elo + fas_elo) / 2
    return None


def extract_votes_from_summary(summary, player_position=None):
    """Extract all votes from a summary file.
    
    Args:
        summary: The parsed JSON summary data
        player_position: Optional int, if specified only count votes from this player position (0-indexed)
    """
    if not summary or 'logs' not in summary:
        return []
    
    combined_elo = get_combined_elo(summary)
    
    all_votes = []
    for round_idx, round_data in enumerate(summary['logs'], 1):
        if 'votes' in round_data and isinstance(round_data['votes'], list):
            votes = round_data['votes']
            
            if player_position is not None:
                # Filter to only the specified player position
                if player_position < len(votes):
                    player_vote = votes[player_position]
                    all_votes.append({
                        'round': round_idx,
                        'votes': [player_vote],
                        'yes_count': 1 if player_vote is True else 0,
                        'no_count': 1 if player_vote is False else 0,
                        'total_players': 1,
                        'elo': combined_elo
                    })
            else:
                # Include all players' votes
                all_votes.append({
                    'round': round_idx,
                    'votes': votes,
                    'yes_count': sum(1 for v in votes if v is True),
                    'no_count': sum(1 for v in votes if v is False),
                    'total_players': len(votes),
                    'elo': combined_elo
                })
    return all_votes


def perform_elo_chi_square_test(high_elo_stats, low_elo_stats, elo_threshold):
    """Perform chi-square test comparing high vs low ELO voting patterns.
    
    Args:
        high_elo_stats (dict): Statistics for high ELO games
        low_elo_stats (dict): Statistics for low ELO games
        elo_threshold (int): The ELO threshold used to split the data
    """
    print("\n" + "=" * 60)
    print("CHI-SQUARE TEST: HIGH ELO vs LOW ELO VOTING PATTERNS")
    print("=" * 60)
    
    # Check if we have data for both groups
    if high_elo_stats['total_votes'] == 0 or low_elo_stats['total_votes'] == 0:
        print("Warning: Insufficient data for chi-square test")
        return
    
    # Test 1: Overall yes/no pattern
    print("\n--- TEST 1: Overall Yes/No Pattern ---")
    contingency_overall = np.array([
        [high_elo_stats['yes_votes'], high_elo_stats['no_votes']],
        [low_elo_stats['yes_votes'], low_elo_stats['no_votes']]
    ])
    
    print(f"\nContingency Table:")
    print(f"{'Group':<20} {'Yes Votes':<15} {'No Votes':<15} {'Total':<15} {'Yes %':<10}")
    print("-" * 75)
    
    high_total = high_elo_stats['total_votes']
    high_yes_pct = (high_elo_stats['yes_votes'] / high_total * 100) if high_total > 0 else 0
    print(f"High ELO (>{elo_threshold})    {high_elo_stats['yes_votes']:<15,} "
          f"{high_elo_stats['no_votes']:<15,} {high_total:<15,} {high_yes_pct:<10.1f}")
    
    low_total = low_elo_stats['total_votes']
    low_yes_pct = (low_elo_stats['yes_votes'] / low_total * 100) if low_total > 0 else 0
    print(f"Low ELO (≤{elo_threshold})     {low_elo_stats['yes_votes']:<15,} "
          f"{low_elo_stats['no_votes']:<15,} {low_total:<15,} {low_yes_pct:<10.1f}")
    
    try:
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_overall)
        
        print(f"\nChi-square statistic: {chi2_stat:.4f}")
        print(f"P-value: {p_value:.6e}" if p_value < 0.001 else f"P-value: {p_value:.6f}")
        
        n = contingency_overall.sum()
        min_dim = min(contingency_overall.shape[0], contingency_overall.shape[1]) - 1
        if min_dim > 0:
            cramers_v = np.sqrt(chi2_stat / (n * min_dim))
            print(f"Cramer's V (effect size): {cramers_v:.4f}")
        
        if p_value < 0.05:
            print(f"Result: SIGNIFICANT - Overall voting rates differ between ELO groups")
        else:
            print(f"Result: NOT SIGNIFICANT - Similar overall voting rates")
            
    except ValueError as e:
        print(f"Test failed: {e}")
    
    # Test 2: Game phase patterns (early/mid/late)
    print("\n--- TEST 2: Game Phase Patterns (Early/Mid/Late) ---")
    
    # Calculate yes votes for each phase
    high_early_yes = sum(high_elo_stats['round_stats'][r]['yes'] for r in range(1, 4) if r in high_elo_stats['round_stats'])
    high_mid_yes = sum(high_elo_stats['round_stats'][r]['yes'] for r in range(4, 8) if r in high_elo_stats['round_stats'])
    high_late_yes = sum(high_elo_stats['round_stats'][r]['yes'] for r in range(8, 20) if r in high_elo_stats['round_stats'])
    
    high_early_total = sum(high_elo_stats['round_stats'][r]['total'] for r in range(1, 4) if r in high_elo_stats['round_stats'])
    high_mid_total = sum(high_elo_stats['round_stats'][r]['total'] for r in range(4, 8) if r in high_elo_stats['round_stats'])
    high_late_total = sum(high_elo_stats['round_stats'][r]['total'] for r in range(8, 20) if r in high_elo_stats['round_stats'])
    
    low_early_yes = sum(low_elo_stats['round_stats'][r]['yes'] for r in range(1, 4) if r in low_elo_stats['round_stats'])
    low_mid_yes = sum(low_elo_stats['round_stats'][r]['yes'] for r in range(4, 8) if r in low_elo_stats['round_stats'])
    low_late_yes = sum(low_elo_stats['round_stats'][r]['yes'] for r in range(8, 20) if r in low_elo_stats['round_stats'])
    
    low_early_total = sum(low_elo_stats['round_stats'][r]['total'] for r in range(1, 4) if r in low_elo_stats['round_stats'])
    low_mid_total = sum(low_elo_stats['round_stats'][r]['total'] for r in range(4, 8) if r in low_elo_stats['round_stats'])
    low_late_total = sum(low_elo_stats['round_stats'][r]['total'] for r in range(8, 20) if r in low_elo_stats['round_stats'])
    
    # Create contingency table: rows = ELO group, cols = game phase (yes/no for each)
    contingency_phase = np.array([
        [high_early_yes, high_early_total - high_early_yes, 
         high_mid_yes, high_mid_total - high_mid_yes,
         high_late_yes, high_late_total - high_late_yes],
        [low_early_yes, low_early_total - low_early_yes,
         low_mid_yes, low_mid_total - low_mid_yes,
         low_late_yes, low_late_total - low_late_yes]
    ])
    
    print(f"\nContingency Table (Yes/No for each phase):")
    print(f"{'Group':<20} {'Early Yes':<12} {'Early No':<12} {'Mid Yes':<12} {'Mid No':<12} {'Late Yes':<12} {'Late No':<12}")
    print("-" * 100)
    print(f"High ELO (>{elo_threshold})    {high_early_yes:<12,} {high_early_total-high_early_yes:<12,} "
          f"{high_mid_yes:<12,} {high_mid_total-high_mid_yes:<12,} "
          f"{high_late_yes:<12,} {high_late_total-high_late_yes:<12,}")
    print(f"Low ELO (≤{elo_threshold})     {low_early_yes:<12,} {low_early_total-low_early_yes:<12,} "
          f"{low_mid_yes:<12,} {low_mid_total-low_mid_yes:<12,} "
          f"{low_late_yes:<12,} {low_late_total-low_late_yes:<12,}")
    
    print(f"\nYes Vote Percentages by Phase:")
    print(f"{'Group':<20} {'Early (1-3)':<15} {'Mid (4-7)':<15} {'Late (8+)':<15}")
    print("-" * 65)
    
    high_early_pct = (high_early_yes / high_early_total * 100) if high_early_total > 0 else 0
    high_mid_pct = (high_mid_yes / high_mid_total * 100) if high_mid_total > 0 else 0
    high_late_pct = (high_late_yes / high_late_total * 100) if high_late_total > 0 else 0
    
    low_early_pct = (low_early_yes / low_early_total * 100) if low_early_total > 0 else 0
    low_mid_pct = (low_mid_yes / low_mid_total * 100) if low_mid_total > 0 else 0
    low_late_pct = (low_late_yes / low_late_total * 100) if low_late_total > 0 else 0
    
    print(f"High ELO (>{elo_threshold})    {high_early_pct:<15.1f} {high_mid_pct:<15.1f} {high_late_pct:<15.1f}")
    print(f"Low ELO (≤{elo_threshold})     {low_early_pct:<15.1f} {low_mid_pct:<15.1f} {low_late_pct:<15.1f}")
    
    try:
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_phase)
        
        print(f"\nChi-square Test Results:")
        print(f"Null hypothesis: Voting patterns across game phases are independent of ELO level")
        print(f"Alternative hypothesis: Game phase voting patterns differ significantly between ELO levels")
        print(f"Chi-square statistic: {chi2_stat:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"P-value: {p_value:.6e}" if p_value < 0.001 else f"P-value: {p_value:.6f}")
        
        n = contingency_phase.sum()
        min_dim = min(contingency_phase.shape[0] - 1, contingency_phase.shape[1] - 1)
        if min_dim > 0:
            cramers_v = np.sqrt(chi2_stat / (n * min_dim))
            print(f"Cramer's V (effect size): {cramers_v:.4f}")
            
            if cramers_v < 0.1:
                effect_interpretation = "negligible"
            elif cramers_v < 0.3:
                effect_interpretation = "small"
            elif cramers_v < 0.5:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            print(f"Effect size interpretation: {effect_interpretation}")
        
        if p_value < 0.05:
            print(f"\nResult: SIGNIFICANT (p < 0.05)")
            print("→ Game phase voting patterns differ significantly between high and low ELO players")
            
            # Analyze which phases show the biggest differences
            print("\nPhase-by-phase differences:")
            if high_early_total > 0 and low_early_total > 0:
                early_diff = high_early_pct - low_early_pct
                print(f"  Early game: {abs(early_diff):.1f}% difference ({'High ELO votes YES more' if early_diff > 0 else 'Low ELO votes YES more'})")
            if high_mid_total > 0 and low_mid_total > 0:
                mid_diff = high_mid_pct - low_mid_pct
                print(f"  Mid game:   {abs(mid_diff):.1f}% difference ({'High ELO votes YES more' if mid_diff > 0 else 'Low ELO votes YES more'})")
            if high_late_total > 0 and low_late_total > 0:
                late_diff = high_late_pct - low_late_pct
                print(f"  Late game:  {abs(late_diff):.1f}% difference ({'High ELO votes YES more' if late_diff > 0 else 'Low ELO votes YES more'})")
        else:
            print(f"\nResult: NOT SIGNIFICANT (p ≥ 0.05)")
            print("→ No significant difference in game phase voting patterns between ELO levels")
            
    except ValueError as e:
        print(f"Chi-square test failed: {e}")


def analyze_votes(summaries_folder, player_position=None):
    """Analyze votes from all summary files in the given folder."""
    folder_path = Path(summaries_folder)
    
    if not folder_path.exists():
        print(f"Error: Folder {summaries_folder} does not exist")
        return
    
    # Find all JSON summary files
    json_files = list(folder_path.glob("*_summary.json"))

    if not json_files:
        json_files = list(folder_path.glob("*.json"))
    
    if not json_files:
        print(f"No summary files found in {summaries_folder}")
        return
    
    print(f"Found {len(json_files)} summary files")
    if player_position is not None:
        print(f"Filtering votes for player position {player_position} only")
    print("=" * 60)
    
    # Initialize statistics - overall
    total_votes = 0
    total_yes_votes = 0
    total_no_votes = 0
    round_stats = defaultdict(lambda: {'games': 0, 'yes': 0, 'no': 0, 'total': 0})
    player_count_stats = defaultdict(int)
    files_processed = 0
    
    # Initialize ELO-based statistics
    elo_threshold = 1650
    high_elo_stats = {
        'total_votes': 0,
        'yes_votes': 0,
        'no_votes': 0,
        'round_stats': defaultdict(lambda: {'games': 0, 'yes': 0, 'no': 0, 'total': 0})
    }
    low_elo_stats = {
        'total_votes': 0,
        'yes_votes': 0,
        'no_votes': 0,
        'round_stats': defaultdict(lambda: {'games': 0, 'yes': 0, 'no': 0, 'total': 0})
    }
    games_with_elo = 0
    
    # Process each file
    for file_path in json_files:
        summary = load_summary_file(file_path)
        if summary is None:
            continue
            
        files_processed += 1
        game_votes = extract_votes_from_summary(summary, player_position)
        
        for vote_round in game_votes:
            total_players = vote_round['total_players']
            if total_players == 0:
                break
            round_num = vote_round['round']
            yes_count = vote_round['yes_count']
            no_count = vote_round['no_count']
            elo = vote_round.get('elo')
            
            # Update overall statistics
            total_votes += total_players
            total_yes_votes += yes_count
            total_no_votes += no_count
            
            # Update round statistics
            round_stats[round_num]['games'] += 1
            round_stats[round_num]['yes'] += yes_count
            round_stats[round_num]['no'] += no_count
            round_stats[round_num]['total'] += total_players
            
            # Track player count distribution
            player_count_stats[total_players] += 1
            
            # Update ELO-based statistics if ELO data is available
            if elo is not None:
                if round_num == 1:  # Count games only once per game
                    games_with_elo += 1
                
                if elo > elo_threshold:
                    # High ELO
                    high_elo_stats['total_votes'] += total_players
                    high_elo_stats['yes_votes'] += yes_count
                    high_elo_stats['no_votes'] += no_count
                    high_elo_stats['round_stats'][round_num]['games'] += 1
                    high_elo_stats['round_stats'][round_num]['yes'] += yes_count
                    high_elo_stats['round_stats'][round_num]['no'] += no_count
                    high_elo_stats['round_stats'][round_num]['total'] += total_players
                else:
                    # Low ELO
                    low_elo_stats['total_votes'] += total_players
                    low_elo_stats['yes_votes'] += yes_count
                    low_elo_stats['no_votes'] += no_count
                    low_elo_stats['round_stats'][round_num]['games'] += 1
                    low_elo_stats['round_stats'][round_num]['yes'] += yes_count
                    low_elo_stats['round_stats'][round_num]['no'] += no_count
                    low_elo_stats['round_stats'][round_num]['total'] += total_players
    
    # Print overall statistics
    print(f"FILES PROCESSED: {files_processed}")
    print(f"TOTAL VOTES CAST: {total_votes:,}")
    print(f"YES VOTES: {total_yes_votes:,} ({total_yes_votes/total_votes*100:.1f}%)")
    print(f"NO VOTES: {total_no_votes:,} ({total_no_votes/total_votes*100:.1f}%)")
    print()
    
    # Print player count distribution
    print("PLAYER COUNT DISTRIBUTION:")
    for player_count in sorted(player_count_stats.keys()):
        count = player_count_stats[player_count]
        print(f"  {player_count} players: {count} rounds")
    print()
    
    # Print per-round statistics
    print("PER-ROUND STATISTICS:")
    print(f"{'Round':<6} {'Games':<8} {'Total Votes':<12} {'Yes Votes':<12} {'No Votes':<12} {'Yes %':<8} {'No %':<8} {'Avg Votes/Game':<15}")
    print("-" * 95)
    
    for round_num in sorted(round_stats.keys()):
        stats = round_stats[round_num]
        games = stats['games']
        total_round_votes = stats['total']
        yes_votes = stats['yes']
        no_votes = stats['no']
        
        yes_pct = (yes_votes / total_round_votes * 100) if total_round_votes > 0 else 0
        no_pct = (no_votes / total_round_votes * 100) if total_round_votes > 0 else 0
        avg_votes_per_game = total_round_votes / games if games > 0 else 0
        
        print(f"{round_num:<6} {games:<8} {total_round_votes:<12} {yes_votes:<12} {no_votes:<12} "
              f"{yes_pct:<8.1f} {no_pct:<8.1f} {avg_votes_per_game:<15.1f}")
    
    print()
    
    # Summary by round ranges
    print("ROUND GROUPINGS:")
    early_rounds = sum(round_stats[r]['total'] for r in range(1, 4) if r in round_stats)
    mid_rounds = sum(round_stats[r]['total'] for r in range(4, 8) if r in round_stats)
    late_rounds = sum(round_stats[r]['total'] for r in range(8, 20) if r in round_stats)
    
    early_yes = sum(round_stats[r]['yes'] for r in range(1, 4) if r in round_stats)
    mid_yes = sum(round_stats[r]['yes'] for r in range(4, 8) if r in round_stats)
    late_yes = sum(round_stats[r]['yes'] for r in range(8, 20) if r in round_stats)
    
    if early_rounds > 0:
        print(f"Early rounds (1-3): {early_yes}/{early_rounds} ({early_yes/early_rounds*100:.1f}% yes)")
    if mid_rounds > 0:
        print(f"Mid rounds (4-7):   {mid_yes}/{mid_rounds} ({mid_yes/mid_rounds*100:.1f}% yes)")
    if late_rounds > 0:
        print(f"Late rounds (8+):   {late_yes}/{late_rounds} ({late_yes/late_rounds*100:.1f}% yes)")
    
    # ELO-based analysis
    if games_with_elo > 0:

        # --- Plot: Yes vote % by round, high vs low ELO ---
        high_rounds = sorted(high_elo_stats['round_stats'].keys())
        low_rounds = sorted(low_elo_stats['round_stats'].keys())
        all_plot_rounds = sorted(set(high_rounds) | set(low_rounds))

        if all_plot_rounds:
            high_pcts = []
            low_pcts = []
            for r in all_plot_rounds:
                hs = high_elo_stats['round_stats'].get(r, {'yes': 0, 'total': 0})
                ls = low_elo_stats['round_stats'].get(r, {'yes': 0, 'total': 0})
                high_pcts.append(hs['yes'] / hs['total'] * 100 if hs['total'] > 0 else 0)
                low_pcts.append(ls['yes'] / ls['total'] * 100 if ls['total'] > 0 else 0)

            fig, ax = plt.subplots(figsize=(6.46, 3))
            x = np.arange(len(all_plot_rounds))
            width = 0.35
            ax.bar(x - width/2, high_pcts, width, label=f'High ELO (>{elo_threshold})', color=GAMMA, zorder=5)
            ax.bar(x + width/2, low_pcts, width, label=f'Low ELO ($\\leq${elo_threshold})', color=ETA, zorder=5)
            ax.set_xlabel('Round')
            ax.set_ylabel('Yes Vote Rate')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}\\%'))
            ax.set_xticks(x)
            ax.set_xticklabels(all_plot_rounds)
            ax.grid(axis='y', alpha=0.3)
            ax.legend(framealpha=1)
            plt.tight_layout()
            model_slug = extract_model_name(summaries_folder).replace(' ', '_').lower()
            out_path = get_plot_path(f'vote_analyzer_{model_slug}.pdf')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nPlot saved to: {out_path}")

        print("\n" + "=" * 60)
        print(f"ELO-BASED ANALYSIS ({games_with_elo} games with ELO data)")
        print("=" * 60)
        
        # Perform chi-square test first
        perform_elo_chi_square_test(high_elo_stats, low_elo_stats, elo_threshold)
        
        # High ELO statistics
        if high_elo_stats['total_votes'] > 0:
            print(f"\nHIGH ELO (>{elo_threshold}):")
            print(f"TOTAL VOTES: {high_elo_stats['total_votes']:,}")
            print(f"YES VOTES: {high_elo_stats['yes_votes']:,} ({high_elo_stats['yes_votes']/high_elo_stats['total_votes']*100:.1f}%)")
            print(f"NO VOTES: {high_elo_stats['no_votes']:,} ({high_elo_stats['no_votes']/high_elo_stats['total_votes']*100:.1f}%)")
            
            print("\nPER-ROUND STATISTICS (High ELO):")
            print(f"{'Round':<6} {'Games':<8} {'Total Votes':<12} {'Yes Votes':<12} {'No Votes':<12} {'Yes %':<8} {'No %':<8}")
            print("-" * 75)
            
            for round_num in sorted(high_elo_stats['round_stats'].keys()):
                stats = high_elo_stats['round_stats'][round_num]
                games = stats['games']
                total_round_votes = stats['total']
                yes_votes = stats['yes']
                no_votes = stats['no']
                
                yes_pct = (yes_votes / total_round_votes * 100) if total_round_votes > 0 else 0
                no_pct = (no_votes / total_round_votes * 100) if total_round_votes > 0 else 0
                
                print(f"{round_num:<6} {games:<8} {total_round_votes:<12} {yes_votes:<12} {no_votes:<12} "
                      f"{yes_pct:<8.1f} {no_pct:<8.1f}")
            
            # Round groupings for high ELO
            print("\nROUND GROUPINGS (High ELO):")
            high_early = sum(high_elo_stats['round_stats'][r]['total'] for r in range(1, 4) if r in high_elo_stats['round_stats'])
            high_mid = sum(high_elo_stats['round_stats'][r]['total'] for r in range(4, 8) if r in high_elo_stats['round_stats'])
            high_late = sum(high_elo_stats['round_stats'][r]['total'] for r in range(8, 20) if r in high_elo_stats['round_stats'])
            
            high_early_yes = sum(high_elo_stats['round_stats'][r]['yes'] for r in range(1, 4) if r in high_elo_stats['round_stats'])
            high_mid_yes = sum(high_elo_stats['round_stats'][r]['yes'] for r in range(4, 8) if r in high_elo_stats['round_stats'])
            high_late_yes = sum(high_elo_stats['round_stats'][r]['yes'] for r in range(8, 20) if r in high_elo_stats['round_stats'])
            
            if high_early > 0:
                print(f"Early rounds (1-3): {high_early_yes}/{high_early} ({high_early_yes/high_early*100:.1f}% yes)")
            if high_mid > 0:
                print(f"Mid rounds (4-7):   {high_mid_yes}/{high_mid} ({high_mid_yes/high_mid*100:.1f}% yes)")
            if high_late > 0:
                print(f"Late rounds (8+):   {high_late_yes}/{high_late} ({high_late_yes/high_late*100:.1f}% yes)")
        
        # Low ELO statistics
        if low_elo_stats['total_votes'] > 0:
            print(f"\nLOW ELO (≤{elo_threshold}):")
            print(f"TOTAL VOTES: {low_elo_stats['total_votes']:,}")
            print(f"YES VOTES: {low_elo_stats['yes_votes']:,} ({low_elo_stats['yes_votes']/low_elo_stats['total_votes']*100:.1f}%)")
            print(f"NO VOTES: {low_elo_stats['no_votes']:,} ({low_elo_stats['no_votes']/low_elo_stats['total_votes']*100:.1f}%)")
            
            print("\nPER-ROUND STATISTICS (Low ELO):")
            print(f"{'Round':<6} {'Games':<8} {'Total Votes':<12} {'Yes Votes':<12} {'No Votes':<12} {'Yes %':<8} {'No %':<8}")
            print("-" * 75)
            
            for round_num in sorted(low_elo_stats['round_stats'].keys()):
                stats = low_elo_stats['round_stats'][round_num]
                games = stats['games']
                total_round_votes = stats['total']
                yes_votes = stats['yes']
                no_votes = stats['no']
                
                yes_pct = (yes_votes / total_round_votes * 100) if total_round_votes > 0 else 0
                no_pct = (no_votes / total_round_votes * 100) if total_round_votes > 0 else 0
                
                print(f"{round_num:<6} {games:<8} {total_round_votes:<12} {yes_votes:<12} {no_votes:<12} "
                      f"{yes_pct:<8.1f} {no_pct:<8.1f}")
            
            # Round groupings for low ELO
            print("\nROUND GROUPINGS (Low ELO):")
            low_early = sum(low_elo_stats['round_stats'][r]['total'] for r in range(1, 4) if r in low_elo_stats['round_stats'])
            low_mid = sum(low_elo_stats['round_stats'][r]['total'] for r in range(4, 8) if r in low_elo_stats['round_stats'])
            low_late = sum(low_elo_stats['round_stats'][r]['total'] for r in range(8, 20) if r in low_elo_stats['round_stats'])
            
            low_early_yes = sum(low_elo_stats['round_stats'][r]['yes'] for r in range(1, 4) if r in low_elo_stats['round_stats'])
            low_mid_yes = sum(low_elo_stats['round_stats'][r]['yes'] for r in range(4, 8) if r in low_elo_stats['round_stats'])
            low_late_yes = sum(low_elo_stats['round_stats'][r]['yes'] for r in range(8, 20) if r in low_elo_stats['round_stats'])
            
            if low_early > 0:
                print(f"Early rounds (1-3): {low_early_yes}/{low_early} ({low_early_yes/low_early*100:.1f}% yes)")
            if low_mid > 0:
                print(f"Mid rounds (4-7):   {low_mid_yes}/{low_mid} ({low_mid_yes/low_mid*100:.1f}% yes)")
            if low_late > 0:
                print(f"Late rounds (8+):   {low_late_yes}/{low_late} ({low_late_yes/low_late*100:.1f}% yes)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze yes/no votes from Secret Hitler game summary files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vote_analyzer.py ../crawl/summaries/
  python vote_analyzer.py /path/to/summaries/folder/
  python vote_analyzer.py ../crawl/summaries/ --position 0
  python vote_analyzer.py ../crawl/summaries/ -p 2
        """
    )
    
    parser.add_argument(
        'summaries_folder',
        help='Path to the folder containing *_summary.json files'
    )
    
    parser.add_argument(
        '--position', '-p',
        type=int,
        help='Filter votes to only include the specified player position (0-indexed). If not specified, analyzes all players.'
    )
    
    args = parser.parse_args()
    
    analyze_votes(args.summaries_folder, args.position)


if __name__ == "__main__":
    main()