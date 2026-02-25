#!/usr/bin/env python3
"""
Vote analysis for Secret Hitler game summaries.

Aggregates yes/no vote statistics and per-round voting patterns
from JSON summary files.

Usage: python vote_analyzer.py <summaries_folder>
  summaries_folder  Path to folder with *_summary.json files

Always evaluates Alice (Player 0).
"""

import json
from collections import defaultdict
from pathlib import Path
import argparse
from plot_config import setup_plot_style, load_summary_file

# Apply shared plotting configuration
setup_plot_style()


def extract_votes_from_summary(summary):
    """Extract Alice's (Player 0) votes from a summary file."""
    if not summary or 'logs' not in summary:
        return []
    
    all_votes = []
    for round_idx, round_data in enumerate(summary['logs'], 1):
        if 'votes' in round_data and isinstance(round_data['votes'], list):
            votes = round_data['votes']
            
            # Always filter to Alice (Player 0)
            if 0 < len(votes):
                player_vote = votes[0]
                all_votes.append({
                    'round': round_idx,
                    'votes': [player_vote],
                    'yes_count': 1 if player_vote is True else 0,
                    'no_count': 1 if player_vote is False else 0,
                    'total_players': 1,
                })
    return all_votes


def analyze_vote_accuracy(summaries_folder):
    """Analyze vote accuracy: how often Alice correctly votes nein in the hitler zone
    when a fascist is president or hitler is chancellor.

    Instance: Alice is Liberal AND game is in hitler zone (3+ fascist policies enacted)
              AND (Fascist is up for President OR Hitler is up for Chancellor)
    Success:  Alice votes nein (False)
    """
    folder_path = Path(summaries_folder)

    if not folder_path.exists():
        print(f"Error: Folder {summaries_folder} does not exist")
        return

    json_files = list(folder_path.glob("*_summary.json"))
    if not json_files:
        json_files = list(folder_path.glob("*.json"))
    if not json_files:
        print(f"No summary files found in {summaries_folder}")
        return

    total_instances = 0
    total_successes = 0
    round_stats = defaultdict(lambda: {'instances': 0, 'successes': 0})
    files_processed = 0

    for file_path in json_files:
        summary = load_summary_file(file_path)
        if summary is None:
            continue

        players = summary.get('players', [])
        logs = summary.get('logs', [])
        if not players or not logs:
            continue

        files_processed += 1

        # Build role lookup by player index
        roles = {idx: p['role'] for idx, p in enumerate(players)}

        # Determine hitler zone threshold (default 3)
        hitler_zone = 3

        # Track cumulative fascist policies and executed players
        fascist_policies = 0
        executed = set()

        for round_idx, round_data in enumerate(logs, 1):
            votes = round_data.get('votes', [])
            pres_id = round_data.get('presidentId')
            chanc_id = round_data.get('chancellorId')

            if not votes or pres_id is None or chanc_id is None:
                # Still track enacted policy even if we skip analysis
                if round_data.get('enactedPolicy') == 'fascist':
                    fascist_policies += 1
                if 'execution' in round_data:
                    executed.add(round_data['execution'])
                continue

            # Check hitler zone condition BEFORE this round's policy is enacted
            in_hitler_zone = fascist_policies >= hitler_zone

            # Check government composition
            pres_role = roles.get(pres_id)
            chanc_role = roles.get(chanc_id)
            fascist_president = pres_role in ('fascist', 'hitler')
            hitler_chancellor = chanc_role == 'hitler'

            dangerous_government = fascist_president or hitler_chancellor

            if in_hitler_zone and dangerous_government:
                # Check Alice's (Player 0) vote
                if 0 < len(votes) and votes[0] is not None and 0 not in executed and roles.get(0) == 'liberal':
                    total_instances += 1
                    round_stats[round_idx]['instances'] += 1

                    if votes[0] is False:  # nein
                        total_successes += 1
                        round_stats[round_idx]['successes'] += 1

            # Update state AFTER evaluating votes
            if round_data.get('enactedPolicy') == 'fascist':
                fascist_policies += 1
            if 'execution' in round_data:
                executed.add(round_data['execution'])

    # Print results
    print()
    print("=" * 60)
    print("VOTE ACCURACY (Alice nein in hitler zone vs dangerous gov)")
    print("=" * 60)
    print(f"Files processed: {files_processed}")
    print(f"Total instances: {total_instances}")

    if total_instances > 0:
        accuracy = total_successes / total_instances * 100
        print(f"Correct nein votes: {total_successes}")
        print(f"Vote accuracy: {accuracy:.1f}%")
    else:
        print("No qualifying instances found.")

    if round_stats:
        print()
        print(f"{'Round':<8} {'Instances':<12} {'Correct':<12} {'Accuracy':<10}")
        print("-" * 42)
        for round_num in sorted(round_stats.keys()):
            s = round_stats[round_num]
            acc = s['successes'] / s['instances'] * 100 if s['instances'] > 0 else 0
            print(f"{round_num:<8} {s['instances']:<12} {s['successes']:<12} {acc:<10.1f}")


def analyze_votes(summaries_folder):
    """Analyze Alice's (Player 0) votes from all summary files in the given folder."""
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
    print(f"Evaluating Alice (Player 0)")
    print("=" * 60)
    
    # Initialize statistics - overall
    total_votes = 0
    total_yes_votes = 0
    total_no_votes = 0
    round_stats = defaultdict(lambda: {'games': 0, 'yes': 0, 'no': 0, 'total': 0})
    player_count_stats = defaultdict(int)
    files_processed = 0
    
    # Process each file
    for file_path in json_files:
        summary = load_summary_file(file_path)
        if summary is None:
            continue
            
        files_processed += 1
        game_votes = extract_votes_from_summary(summary)
        
        for vote_round in game_votes:
            total_players = vote_round['total_players']
            if total_players == 0:
                break
            round_num = vote_round['round']
            yes_count = vote_round['yes_count']
            no_count = vote_round['no_count']
            
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

    # Vote accuracy analysis
    analyze_vote_accuracy(summaries_folder)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Alice's (Player 0) votes from Secret Hitler game summary files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vote_analyzer.py ../crawl/summaries/
  python vote_analyzer.py /path/to/summaries/folder/
        """
    )
    
    parser.add_argument(
        'summaries_folder',
        help='Path to the folder containing *_summary.json files'
    )
    
    args = parser.parse_args()
    
    analyze_votes(args.summaries_folder)


if __name__ == "__main__":
    main()