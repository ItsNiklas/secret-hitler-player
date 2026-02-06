import os
import glob
import json
import sys
from typing import List, Dict, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from scipy.stats import mannwhitneyu
from plot_config import setup_plot_style, ROLE_COLORS, extract_model_name, get_model_imagebox

# Apply shared plotting configuration
setup_plot_style()

EVAL_DIR = sys.argv[1] if len(sys.argv) > 1 else None
ALICE_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # Player ID to analyze (default 0 for Alice)

def print_usage():
    print("Usage: python gamestats.py <EVAL_DIR>")
    print("  EVAL_DIR: Directory name containing evaluation JSON files (e.g., 'runs1-Qwen3')")
    print(f"  Note: Analyzing player {ALICE_ID} (Alice). To analyze a different player, modify ALICE_ID variable.")
    sys.exit(1)


def load_eval_run_filenames() -> List[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(f"{current_dir}/{EVAL_DIR}/*.json")


def parse_game_data(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse game data and determine win condition and other game statistics."""
    
    # Extract basic game info
    players = game_data.get('players', [])
    logs = game_data.get('logs', [])
    
    # Count policies enacted
    liberal_policies = 0
    fascist_policies = 0
    hitler_player_id = None
    
    # Find Hitler player
    for i, player in enumerate(players):
        if player.get('role') == 'hitler':
            hitler_player_id = i
            break
    
    # Count enacted policies and check for Hitler chancellorship
    hitler_chancellor_win = False
    
    for log in logs:
        # Check if Hitler became chancellor when there are already 3+ fascist policies
        chancellor_id = log.get('chancellorId')
        if chancellor_id == hitler_player_id and fascist_policies >= 3:
            hitler_chancellor_win = True
            break
            
        # Count the enacted policy for this round
        enacted_policy = log.get('enactedPolicy')
        if enacted_policy == 'liberal':
            liberal_policies += 1
        elif enacted_policy == 'fascist':
            fascist_policies += 1
    
    # Determine win condition
    win_condition = None
    winner = None
    
    if hitler_chancellor_win:
        win_condition = "hitler_chancellor"
        winner = "fascists"
    elif liberal_policies >= 5:
        win_condition = "liberal_policies"
        winner = "liberals"
    elif fascist_policies >= 6:
        win_condition = "fascist_policies"
        winner = "fascists"
    else:
        # Default to liberal win (typically means Hitler was killed)
        win_condition = "hitler_killed"
        winner = "liberals"
    
    # Determine Alice's (player ALICE_ID) role and whether she won
    alice_role = None
    alice_won = False
    if len(players) > ALICE_ID:
        alice_role = players[ALICE_ID].get('role')
        if alice_role in ['liberal'] and winner == 'liberals':
            alice_won = True
        elif alice_role in ['fascist', 'hitler'] and winner == 'fascists':
            alice_won = True

    return {
        'game_id': game_data.get('_id', 'unknown'),
        'players': players,
        'liberal_policies': liberal_policies,
        'fascist_policies': fascist_policies,
        'hitler_player_id': hitler_player_id,
        'win_condition': win_condition,
        'winner': winner,
        'total_rounds': len(logs),
        'alice_role': alice_role,
        'alice_won': alice_won
    }


def analyze_alice_performance(games_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Alice's (player ALICE_ID) win rate and performance statistics."""
    
    alice_games = []
    alice_wins = 0
    alice_as_liberal = 0
    alice_as_fascist = 0
    alice_as_hitler = 0
    alice_wins_as_liberal = 0
    alice_wins_as_fascist = 0
    alice_wins_as_hitler = 0
    
    for game in games_data:
        if game['alice_role'] is not None:
            alice_games.append(game)
            
            # Count role occurrences
            if game['alice_role'] == 'liberal':
                alice_as_liberal += 1
                if game['alice_won']:
                    alice_wins_as_liberal += 1
            elif game['alice_role'] == 'fascist':
                alice_as_fascist += 1
                if game['alice_won']:
                    alice_wins_as_fascist += 1
            elif game['alice_role'] == 'hitler':
                alice_as_hitler += 1
                if game['alice_won']:
                    alice_wins_as_hitler += 1
            
            if game['alice_won']:
                alice_wins += 1
    
    total_alice_games = len(alice_games)
    
    if total_alice_games == 0:
        return {
            'total_games': 0,
            'win_rate': 0.0,
            'role_distribution': {},
            'win_rate_by_role': {}
        }
    
    # Calculate overall win rate
    overall_win_rate = (alice_wins / total_alice_games) * 100
    
    # Calculate win rates by role
    liberal_win_rate = (alice_wins_as_liberal / alice_as_liberal * 100) if alice_as_liberal > 0 else 0
    fascist_win_rate = (alice_wins_as_fascist / alice_as_fascist * 100) if alice_as_fascist > 0 else 0
    hitler_win_rate = (alice_wins_as_hitler / alice_as_hitler * 100) if alice_as_hitler > 0 else 0
    
    return {
        'total_games': total_alice_games,
        'total_wins': alice_wins,
        'win_rate': overall_win_rate,
        'role_distribution': {
            'liberal': alice_as_liberal,
            'fascist': alice_as_fascist,
            'hitler': alice_as_hitler
        },
        'wins_by_role': {
            'liberal': alice_wins_as_liberal,
            'fascist': alice_wins_as_fascist,
            'hitler': alice_wins_as_hitler
        },
        'win_rate_by_role': {
            'liberal': liberal_win_rate,
            'fascist': fascist_win_rate,
            'hitler': hitler_win_rate
        }
    }


def print_alice_analysis(alice_stats: Dict[str, Any]):
    """Print detailed Alice performance analysis."""
    print("\n" + "="*60)
    print(f"ALICE (PLAYER {ALICE_ID}) PERFORMANCE ANALYSIS")
    print("="*60)
    
    total = alice_stats['total_games']
    wins = alice_stats['total_wins']
    win_rate = alice_stats['win_rate']
    
    print(f"Total games played by Alice: {total}")
    print(f"Total wins: {wins}")
    print(f"Overall Win Rate: {win_rate:.1f}%")
    
    print(f"\nRole Distribution:")
    role_dist = alice_stats['role_distribution']
    wins_by_role = alice_stats['wins_by_role']
    win_rates = alice_stats['win_rate_by_role']
    
    for role in ['liberal', 'fascist', 'hitler']:
        games_as_role = role_dist[role]
        wins_as_role = wins_by_role[role]
        win_rate_role = win_rates[role]
        
        if games_as_role > 0:
            print(f"  - As {role.capitalize()}: {games_as_role} games, {wins_as_role} wins ({win_rate_role:.1f}% win rate)")
        else:
            print(f"  - As {role.capitalize()}: 0 games")


def analyze_win_conditions(games_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze win conditions across all games."""
    
    win_stats = {
        'liberal_policies': 0,
        'fascist_policies': 0,
        'hitler_chancellor': 0,
        'hitler_killed': 0
    }
    
    winner_stats = {
        'liberals': 0,
        'fascists': 0,
        'unknown': 0
    }
    
    for game in games_data:
        win_condition = game['win_condition']
        winner = game['winner']
        
        win_stats[win_condition] += 1
        winner_stats[winner] += 1
    
    total_games = len(games_data)
    
    return {
        'total_games': total_games,
        'win_conditions': win_stats,
        'winners': winner_stats,
        'win_condition_percentages': {k: (v / total_games * 100) for k, v in win_stats.items()},
        'winner_percentages': {k: (v / total_games * 100) for k, v in winner_stats.items()}
    }


def print_win_analysis(analysis: Dict[str, Any]):
    """Print detailed win condition analysis."""
    print("\n" + "="*60)
    print("WIN CONDITION ANALYSIS")
    print("="*60)
    
    total = analysis['total_games']
    print(f"Total games analyzed: {total}")
    
    print(f"\nWin Conditions:")
    print(f"  1. Liberals win (5 liberal policies): {analysis['win_conditions']['liberal_policies']} games ({analysis['win_condition_percentages']['liberal_policies']:.1f}%)")
    print(f"  2. Liberals win (Hitler killed): {analysis['win_conditions']['hitler_killed']} games ({analysis['win_condition_percentages']['hitler_killed']:.1f}%)")
    print(f"  3. Fascists win (6 fascist policies): {analysis['win_conditions']['fascist_policies']} games ({analysis['win_condition_percentages']['fascist_policies']:.1f}%)")
    print(f"  4. Fascists win (Hitler as Chancellor): {analysis['win_conditions']['hitler_chancellor']} games ({analysis['win_condition_percentages']['hitler_chancellor']:.1f}%)")
    
    print(f"\nOverall Winners:")
    print(f"  - Liberals: {analysis['winners']['liberals']} games ({analysis['winner_percentages']['liberals']:.1f}%)")
    print(f"  - Fascists: {analysis['winners']['fascists']} games ({analysis['winner_percentages']['fascists']:.1f}%)")
    print(f"  - Unknown: {analysis['winners']['unknown']} games ({analysis['winner_percentages']['unknown']:.1f}%)")
    
    liberal_total = analysis['winners']['liberals']
    if liberal_total > 0:
        policies_pct = analysis['win_conditions']['liberal_policies'] / liberal_total * 100
        hitler_killed_pct = analysis['win_conditions']['hitler_killed'] / liberal_total * 100
        print(f"\nLiberal Win Breakdown:")
        print(f"  - Policy Victory: {analysis['win_conditions']['liberal_policies']}/{liberal_total} ({policies_pct:.1f}% of liberal wins)")
        print(f"  - Hitler Killed: {analysis['win_conditions']['hitler_killed']}/{liberal_total} ({hitler_killed_pct:.1f}% of liberal wins)")
    
    fascist_total = analysis['winners']['fascists']
    if fascist_total > 0:
        hitler_pct = analysis['win_conditions']['hitler_chancellor'] / fascist_total * 100
        policy_pct = analysis['win_conditions']['fascist_policies'] / fascist_total * 100
        print(f"\nFascist Win Breakdown:")
        print(f"  - Hitler Chancellor: {analysis['win_conditions']['hitler_chancellor']}/{fascist_total} ({hitler_pct:.1f}% of fascist wins)")
        print(f"  - Policy Victory: {analysis['win_conditions']['fascist_policies']}/{fascist_total} ({policy_pct:.1f}% of fascist wins)")


def analyze_game_length_distribution(games_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of game lengths (number of rounds)."""
    
    game_lengths = [game['total_rounds'] for game in games_data]
    
    if not game_lengths:
        return {'lengths': [], 'distribution': {}}
    
    # Create distribution histogram data
    min_length = min(game_lengths)
    max_length = max(game_lengths)
    distribution = defaultdict(int)
    
    for length in game_lengths:
        distribution[length] += 1
    
    return {
        'lengths': game_lengths,
        'distribution': dict(distribution),
        'min_length': min_length,
        'max_length': max_length,
        'mean_length': sum(game_lengths) / len(game_lengths),
        'total_games': len(game_lengths)
    }


def calculate_policy_counts_by_round(games_data: List[Dict[str, Any]], policy_type: str = 'liberal') -> Dict[str, Any]:
    """
    Calculate policy counts by round for a given policy type.
    Returns:
        Dictionary containing rounds, means, and statistics
    """
    # Find the maximum number of rounds across all games
    max_rounds = max(game['total_rounds'] for game in games_data) if games_data else 0
    total_games = len(games_data)
    
    # Initialize arrays to store policy counts by round
    policy_by_round = [[] for _ in range(max_rounds)]
    
    # Process each game
    for game in games_data:
        # Get logs from the enhanced parsed data
        logs = game.get('logs', [])
        policy_count = 0
        
        for round_idx, log in enumerate(logs):
            enacted_policy = log.get('enactedPolicy')
            if enacted_policy == policy_type:
                policy_count += 1
            
            # Store cumulative counts for this round
            if round_idx < max_rounds:
                policy_by_round[round_idx].append(policy_count)
    
    # Calculate statistics for each round and find cutoff point
    rounds_with_data = []
    policy_means = []
    cutoff_round = max_rounds  # Default to no cutoff
    
    for round_idx in range(max_rounds):
        if policy_by_round[round_idx]:  # If there's data for this round
            rounds_with_data.append(round_idx + 1)
            
            # Check if this round occurs in less than 10% of games
            games_reaching_round = len(policy_by_round[round_idx])
            percentage = (games_reaching_round / total_games) * 100
            
            # Find the first round that occurs in less than 10% of games
            if percentage < 10 and cutoff_round == max_rounds:
                cutoff_round = round_idx + 1
            
            # Policy statistics
            policy_data = np.array(policy_by_round[round_idx])
            policy_mean = np.mean(policy_data)
            policy_means.append(policy_mean)
    
    return {
        'rounds': rounds_with_data,
        'means': policy_means,
        'cutoff_round': cutoff_round,
        'max_rounds': max_rounds,
        'total_games': total_games,
        'policy_type': policy_type
    }


def analyze_game_length_distribution(games_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of game lengths (number of rounds)."""
    
    game_lengths = [game['total_rounds'] for game in games_data]
    
    if not game_lengths:
        return {'lengths': [], 'distribution': {}}
    
    # Create distribution histogram data
    min_length = min(game_lengths)
    max_length = max(game_lengths)
    distribution = defaultdict(int)
    
    for length in game_lengths:
        distribution[length] += 1
    
    return {
        'lengths': game_lengths,
        'distribution': dict(distribution),
        'min_length': min_length,
        'max_length': max_length,
        'mean_length': sum(game_lengths) / len(game_lengths),
        'total_games': len(game_lengths)
    }


def analyze_alice_game_state_impact(games_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Alice's impact on game state score when she's president or chancellor."""
    
    score_changes = []
    total_alice_actions = 0
    actions_by_role = {'liberal': [], 'fascist': [], 'hitler': []}
    
    for game in games_data:
        alice_role = game.get('alice_role')
        if alice_role is None:
            continue
            
        logs = game.get('logs', [])
        if len(logs) < 2:  # Need at least 2 logs to calculate change
            continue
        
        # Check each log where Alice is president or chancellor
        for i, log in enumerate(logs[:-1]):  # Skip last log since we need next log for comparison
            president_id = log.get('presidentId')
            chancellor_id = log.get('chancellorId')
            
            # Check if Alice (player ALICE_ID) is president or chancellor
            if president_id == ALICE_ID or chancellor_id == ALICE_ID:
                current_score = log.get('gameStateScore', 0)
                next_score = logs[i + 1].get('gameStateScore', 0)
                
                # Calculate score change
                score_change = next_score - current_score
                
                # If Alice is not liberal, negate the score change
                # (since gameStateScore is from liberal perspective)
                if alice_role != 'liberal':
                    score_change = -score_change
                
                score_changes.append(score_change)
                actions_by_role[alice_role].append(score_change)
                total_alice_actions += 1
    
    if not score_changes:
        return {
            'total_actions': 0,
            'average_impact': 0.0,
            'actions_by_role': {},
            'average_by_role': {}
        }
    
    # Calculate averages
    average_impact = sum(score_changes) / len(score_changes)
    
    average_by_role = {}
    for role, changes in actions_by_role.items():
        if changes:
            average_by_role[role] = sum(changes) / len(changes)
        else:
            average_by_role[role] = 0.0
    
    return {
        'total_actions': total_alice_actions,
        'average_impact': average_impact,
        'actions_by_role': {role: len(changes) for role, changes in actions_by_role.items()},
        'average_by_role': average_by_role,
        'all_score_changes': score_changes
    }

def analyze_alice_game_state_impact_by_elo(games_data: List[Dict[str, Any]], elo_threshold: float = 1650) -> Dict[str, Any]:
    """Analyze Alice's impact on game state score by ELO (high vs low)."""
    
    high_elo_changes = []
    low_elo_changes = []
    
    for game in games_data:
        alice_role = game.get('alice_role')
        if alice_role is None:
            continue
        
        # Get ELO data for this game
        lib_elo = game.get('libElo', {}).get('overall')
        fas_elo = game.get('fasElo', {}).get('overall')
        
        # Skip if no ELO data
        if lib_elo is None or fas_elo is None:
            continue
        
        # Determine which ELO to use based on Alice's role
        if alice_role == 'liberal':
            game_elo = lib_elo
        else:  # fascist or hitler
            game_elo = fas_elo
        
        logs = game.get('logs', [])
        if len(logs) < 2:
            continue
        
        # Check each log where Alice is president or chancellor
        for i, log in enumerate(logs[:-1]):
            president_id = log.get('presidentId')
            chancellor_id = log.get('chancellorId')
            
            if president_id == ALICE_ID or chancellor_id == ALICE_ID:
                current_score = log.get('gameStateScore', 0)
                next_score = logs[i + 1].get('gameStateScore', 0)
                
                score_change = next_score - current_score
                
                # If Alice is not liberal, negate the score change
                if alice_role != 'liberal':
                    score_change = -score_change
                
                # Categorize by ELO
                if game_elo >= elo_threshold:
                    high_elo_changes.append(score_change)
                else:
                    low_elo_changes.append(score_change)
    
    # Calculate statistics
    high_elo_avg = np.mean(high_elo_changes) if high_elo_changes else 0.0
    low_elo_avg = np.mean(low_elo_changes) if low_elo_changes else 0.0
    
    # Perform Mann-Whitney U test
    p_value = None
    u_statistic = None
    if len(high_elo_changes) > 0 and len(low_elo_changes) > 0:
        u_statistic, p_value = mannwhitneyu(high_elo_changes, low_elo_changes, alternative='two-sided')
    
    return {
        'elo_threshold': elo_threshold,
        'high_elo': {
            'n_actions': len(high_elo_changes),
            'average_impact': high_elo_avg,
            'std': np.std(high_elo_changes) if high_elo_changes else 0.0,
            'changes': high_elo_changes
        },
        'low_elo': {
            'n_actions': len(low_elo_changes),
            'average_impact': low_elo_avg,
            'std': np.std(low_elo_changes) if low_elo_changes else 0.0,
            'changes': low_elo_changes
        },
        'statistical_test': {
            'u_statistic': u_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05 if p_value is not None else False
        }
    }


def print_alice_game_state_impact_by_elo(analysis: Dict[str, Any]):
    """Print Alice's game state impact analysis by ELO."""
    print("\n" + "="*60)
    print("ALICE GAME STATE IMPACT BY ELO")
    print("="*60)
    
    threshold = analysis['elo_threshold']
    high = analysis['high_elo']
    low = analysis['low_elo']
    stats = analysis['statistical_test']
    
    print(f"ELO Threshold: {threshold}")
    print(f"\nHigh ELO (>= {threshold}):")
    print(f"  Number of actions: {high['n_actions']}")
    if high['n_actions'] > 0:
        print(f"  Average Game State Score Impact: {high['average_impact']:.4f}")
        print(f"  Standard Deviation: {high['std']:.4f}")
    else:
        print(f"  No actions found")
    
    print(f"\nLow ELO (< {threshold}):")
    print(f"  Number of actions: {low['n_actions']}")
    if low['n_actions'] > 0:
        print(f"  Average Game State Score Impact: {low['average_impact']:.4f}")
        print(f"  Standard Deviation: {low['std']:.4f}")
    else:
        print(f"  No actions found")
    
    if stats['p_value'] is not None:
        print(f"\nMann-Whitney U Test:")
        print(f"  U-statistic: {stats['u_statistic']:.2f}")
        print(f"  P-value: {stats['p_value']:.6f}")
        alpha = 0.05
        if stats['significant']:
            print(f"  Result: SIGNIFICANT (p < {alpha})")
            if high['average_impact'] > low['average_impact']:
                print(f"  Interpretation: High ELO games show significantly HIGHER impact")
            else:
                print(f"  Interpretation: Low ELO games show significantly HIGHER impact")
        else:
            print(f"  Result: NOT SIGNIFICANT (p >= {alpha})")
            print(f"  Interpretation: No significant difference between ELO groups")
    else:
        print(f"\nStatistical test not performed (insufficient data)")


def print_alice_game_state_impact(analysis: Dict[str, Any]):
    """Print Alice's game state impact analysis."""
    print("\n" + "="*60)
    print("ALICE GAME STATE IMPACT ANALYSIS")
    print("="*60)
    
    total_actions = analysis['total_actions']
    average_impact = analysis['average_impact']
    
    print(f"Total Alice actions (as President or Chancellor): {total_actions}")
    if total_actions > 0:
        print(f"Average Game State Score Impact: {average_impact:.4f}")
        print(f"  (Positive = beneficial for Alice's team, Negative = harmful)")
        
        print(f"\nImpact by Alice's Role:")
        actions_by_role = analysis['actions_by_role']
        average_by_role = analysis['average_by_role']
        
        for role in ['liberal', 'fascist', 'hitler']:
            actions = actions_by_role.get(role, 0)
            avg_impact = average_by_role.get(role, 0.0)
            if actions > 0:
                print(f"  - As {role.capitalize()}: {actions} actions, {avg_impact:.4f} average impact")
            else:
                print(f"  - As {role.capitalize()}: 0 actions")
    else:
        print("No Alice actions found in the data.")


def print_game_length_histogram(analysis: Dict[str, Any]):
    """Print a text histogram of game length distribution."""
    print("\n" + "="*60)
    print("GAME LENGTH DISTRIBUTION")
    print("="*60)
    
    distribution = analysis['distribution']
    total_games = analysis['total_games']
    mean_length = analysis['mean_length']
    min_length = analysis['min_length']
    max_length = analysis['max_length']
    
    print(f"Total games: {total_games}")
    print(f"Mean game length: {mean_length:.1f} rounds")
    print(f"Range: {min_length} - {max_length} rounds")
    print()
    
    if not distribution:
        print("No games to analyze.")
        return
    
    # Find the maximum count for scaling the histogram
    max_count = max(distribution.values())
    histogram_width = 50  # Maximum width of histogram bars
    
    print("Game Length Histogram:")
    print("Rounds | Count | Percentage | Histogram")
    print("-" * 65)
    
    # Sort by game length for ordered display
    for length in sorted(distribution.keys()):
        count = distribution[length]
        percentage = (count / total_games) * 100
        
        # Create the histogram bar
        bar_length = int((count / max_count) * histogram_width)
        bar = "█" * bar_length
        
        print(f"{length:6d} | {count:5d} | {percentage:8.1f}% | {bar}")
    
    print("-" * 65)


def create_policy_progression_plot(games_data: List[Dict[str, Any]], save_path: str = None):
    """Create a plot showing liberal and fascist policies by round."""
    
    # Calculate liberal and fascist policy progression using the refactored function
    liberal_data = calculate_policy_counts_by_round(games_data, policy_type='liberal')
    fascist_data = calculate_policy_counts_by_round(games_data, policy_type='fascist')
    
    # Use the cutoff from liberal data (typically more restrictive)
    cutoff_round = liberal_data['cutoff_round']
    
    # Print cutoff information
    if cutoff_round < liberal_data['max_rounds']:
        print(f"Cutting off plot at round {cutoff_round} (rounds beyond this occur in <10% of games)")
    
    # Create the plot
    plt.figure(figsize=(5.50, 3))
    
    # Plot liberal policies
    plt.plot(liberal_data['rounds'], liberal_data['means'], '-', marker='o', linewidth=2, markersize=6, 
             label='Liberal Policies (Mean)', color=ROLE_COLORS['liberal'])
    
    # Plot fascist policies
    plt.plot(fascist_data['rounds'], fascist_data['means'], '-', marker='s', linewidth=2, markersize=6, 
             label='Fascist Policies (Mean)', color=ROLE_COLORS['fascist'])
    
    plt.xlabel('Round')
    plt.ylabel('Mean Number of Policies Enacted')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis limits to cut off at the cutoff round
    rounds_with_data = liberal_data['rounds']
    if cutoff_round < liberal_data['max_rounds'] and rounds_with_data:
        plt.xlim(0.5, cutoff_round + 0.5)
        # Set integer ticks for rounds up to cutoff
        cutoff_ticks = [r for r in rounds_with_data if r <= cutoff_round]
        if cutoff_ticks:
            plt.xticks(cutoff_ticks)
    else:
        # Set integer ticks for all rounds if no cutoff needed
        if rounds_with_data:
            plt.xticks(rounds_with_data)
    
    # Add horizontal lines for win conditions
    plt.axhline(y=5, color=ROLE_COLORS['liberal'], linestyle='--', alpha=0.8, zorder=-1)
    plt.axhline(y=6, color=ROLE_COLORS['fascist'], linestyle='--', alpha=0.8, zorder=-1)
    plt.axhline(y=3, color=ROLE_COLORS['fascist'], linestyle=':', alpha=0.8, zorder=-1)
    
    model_name = extract_model_name(EVAL_DIR)
    imagebox = get_model_imagebox(model_name)
    if imagebox:
        legend2 = plt.legend(handles=[Line2D([0], [0], color='none', label=model_name)], 
                        loc='lower right', framealpha=1, handletextpad=-0.4)
        plt.gcf().add_artist(legend2)  # Add the new legend without removing the first one
        plt.gcf().canvas.draw()
        
        ab = AnnotationBbox(imagebox, (0.5, 0.5), xybox=(-3, 0), 
                        xycoords=legend2.legend_handles[0], boxcoords="offset points",
                        frameon=False, box_alignment=(0.5, 0.5), zorder=10)
        plt.gcf().add_artist(ab)
    
    
    plt.legend(loc='upper left', framealpha=1)
    plt.tight_layout()
    
    try:
        plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path + '.pdf'}")
    except Exception as e:
        print(f"Could not save plot: {e}")
        print("Continuing without saving...")

    # plt.show()
    
    # Return means for backward compatibility
    return liberal_data['means'], fascist_data['means']


def enhanced_parse_game_data(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced version that also stores the original logs and ELO data for plotting."""
    result = parse_game_data(game_data)
    result['logs'] = game_data.get('logs', [])
    result['libElo'] = game_data.get('libElo', {})
    result['fasElo'] = game_data.get('fasElo', {})
    return result


if __name__ == "__main__":
    if EVAL_DIR is None:
        print_usage()
    
    eval_files = load_eval_run_filenames()
    print(f"Found {len(eval_files)} files in eval/{EVAL_DIR}")
    
    # Load and parse each game file
    all_games_data = []
    skipped_avalon_games = 0
    
    for file_path in eval_files:
        if 'annotat' in file_path.lower():
            continue
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            
            # Skip games if gameSetting exists AND avalonSH is NOT NULL
            game_setting = game_data.get('gameSetting')
            if game_setting is not None and game_setting.get('avalonSH') is not None:
                skipped_avalon_games += 1
                continue
            
            parsed_data = enhanced_parse_game_data(game_data)
            all_games_data.append(parsed_data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
   
    # Filter out games with fewer than 4 rounds
    games_before_filter = len(all_games_data)
    all_games_data = [game for game in all_games_data if game['total_rounds'] >= 4]
    games_filtered_out = games_before_filter - len(all_games_data)
    
    print(f"\nTotal games processed: {len(all_games_data)}")
    if skipped_avalon_games > 0:
        print(f"Skipped {skipped_avalon_games} Avalon games (avalonSH setting not null)")
    if games_filtered_out > 0:
        print(f"Filtered out {games_filtered_out} games with fewer than 4 rounds")
    
    if len(all_games_data) == 0:
        print("No games found or processed successfully.")
        sys.exit(1)
    
    # Analyze win conditions
    analysis = analyze_win_conditions(all_games_data)
    
    # Analyze Alice's performance
    alice_analysis = analyze_alice_performance(all_games_data)
    
    # Analyze game length distribution
    length_analysis = analyze_game_length_distribution(all_games_data)
    
    # Analyze Alice's game state impact
    alice_impact_analysis = analyze_alice_game_state_impact(all_games_data)
    
    # Analyze Alice's game state impact by ELO (if ELO data exists)
    alice_impact_by_elo_analysis = analyze_alice_game_state_impact_by_elo(all_games_data)
    
    # Print results
    print_win_analysis(analysis)
    print_alice_analysis(alice_analysis)
    print_alice_game_state_impact(alice_impact_analysis)
    print_alice_game_state_impact_by_elo(alice_impact_by_elo_analysis)
    print_game_length_histogram(length_analysis)
    
    # Create policy progression plot
    # Clean up EVAL_DIR for filename use
    safe_eval_dir = EVAL_DIR.replace('/', '_').replace('\\', '_') if EVAL_DIR else "unknown"
    plot_filename = f"policy_progression_{safe_eval_dir}"
    create_policy_progression_plot(all_games_data, plot_filename)
