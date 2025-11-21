import os
import json
import glob
from collections import defaultdict


def load_summary_files():
    """Load all the summary JSON files from the summaries directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(f"{current_dir}/summaries/*.json")


def analyze_games():
    """Analyze all games and return separate lists for Avalon and regular games."""
    summary_files = load_summary_files()
    avalon_games = []
    regular_games = []

    print(f"Analyzing {len(summary_files)} summary files...")

    for file_path in summary_files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
                if data.get("gameSetting", {}).get("avalonSH") is not None:
                    avalon_games.append(file_path)
                else:
                    regular_games.append(file_path)
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {file_path}")

    # Print results
    print(f"\nResults:")
    print(f"Total games: {len(summary_files)}")
    print(f"Avalon SH games: {len(avalon_games)} ({len(avalon_games) / len(summary_files) * 100:.2f}%)")
    print(f"Regular SH games: {len(regular_games)} ({len(regular_games) / len(summary_files) * 100:.2f}%)")

    return avalon_games, regular_games


def analyze_player_counts(game_files):
    """Analyze player count distribution in games."""
    counts = defaultdict(int)

    print(f"Analyzing player counts for {len(game_files)} games...")

    for file_path in game_files:
        with open(file_path, "r") as f:
            players = json.load(f).get("players", [])
            counts[len(players)] += 1

    # Print distribution
    print("\nPlayer Count Distribution:")
    for count in sorted(counts.keys()):
        print(f"  {count} players: {counts[count]} games ({counts[count] / len(game_files) * 100:.2f}%)")

    return counts


def analyze_round_counts(game_files):
    """Analyze round length distribution in games."""
    counts = defaultdict(int)

    print(f"Analyzing round counts for {len(game_files)} games...")

    for file_path in game_files:
        with open(file_path, "r") as f:
            logs = json.load(f).get("logs", [])
            counts[len(logs)] += 1

    # Print distribution
    print("\nRound Count Distribution:")
    for count in sorted(counts.keys()):
        print(f"  {count} rounds: {counts[count]} games ({counts[count] / len(game_files) * 100:.2f}%)")

    return counts


def analyze_game_outcomes(game_files):
    """Analyze how games ended by counting enacted policies."""
    liberal_counts = defaultdict(int)
    fascist_counts = defaultdict(int)
    hitler_elected_count = 0
    hitler_executed_count = 0
    unknown_outcomes = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    replay_data_dir = os.path.join(current_dir, "replay_data")

    print(f"Analyzing policy outcomes for {len(game_files)} games...")

    for file_path in game_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            game_id = data.get("_id", "")
            logs = data.get("logs", [])

            # Count enacted policies
            liberal_policies = 0
            fascist_policies = 0

            for log in logs:
                enacted_policy = log.get("enactedPolicy")
                if enacted_policy == "liberal":
                    liberal_policies += 1
                elif enacted_policy == "fascist":
                    fascist_policies += 1

            # Record the counts
            liberal_counts[liberal_policies] += 1
            fascist_counts[fascist_policies] += 1

            # For games with unknown outcomes (not 5 lib or 6 fas), check xhr_data
            if liberal_policies < 5 and fascist_policies < 6:
                xhr_data_path = os.path.join(replay_data_dir, f"{game_id}_xhr_data.json")

                if os.path.exists(xhr_data_path):
                    try:
                        with open(xhr_data_path, "r") as xhr_f:
                            xhr_data = json.load(xhr_f)

                            hitler_elected = False
                            hitler_execute = False
                            if len(xhr_data) > 1 and isinstance(xhr_data[1], dict):
                                chats = xhr_data[1].get("chats", [])
                                for chat_entry in chats:
                                    if isinstance(chat_entry, dict) and "chat" in chat_entry and isinstance(chat_entry["chat"], list):
                                        for chat_item in chat_entry["chat"]:
                                            if isinstance(chat_item, dict) and "text" in chat_item:
                                                text = chat_item.get("text", "")
                                                if "has been elected chancellor after the 3rd fascist policy has been enacted" in text:
                                                    hitler_elected = True
                                                    break

                                        if len(chat_entry["chat"]) == 2:
                                            if ("Hitler" in chat_entry["chat"][0].get("text", "")) and ("has been executed." in chat_entry["chat"][1].get("text", "")):
                                                hitler_execute = True
                                                break

                            if hitler_elected:
                                hitler_elected_count += 1
                            elif hitler_execute:
                                hitler_executed_count += 1
                            else:
                                print(f"Unknown outcome for game {game_id}")
                                unknown_outcomes += 1
                    except (json.JSONDecodeError, IndexError) as e:
                        print(f"Error: Could not parse XHR data in {xhr_data_path}: {e}")
                        unknown_outcomes += 1

    # Calculate victory distribution
    liberal_wins = liberal_counts.get(5, 0)
    fascist_wins = fascist_counts.get(6, 0)

    # Total games with known outcomes
    total_games = len(game_files)
    assert liberal_wins + fascist_wins + hitler_elected_count + hitler_executed_count + unknown_outcomes == total_games

    # Calculate overall fascist win rate (policy + Hitler elected)
    total_fascist_wins = fascist_wins + hitler_elected_count
    total_liberal_wins = liberal_wins + hitler_executed_count
    total_known_outcome_games = total_liberal_wins + total_fascist_wins

    if total_known_outcome_games > 0:
        print(f"\nOverall Win Rates (excluding true unknowns):")
        print(f"  Liberal Team: {total_liberal_wins} games ({total_liberal_wins / total_known_outcome_games * 100:.2f}%)")
        print(f"    - Through policy: {liberal_wins} games ({liberal_wins / total_known_outcome_games * 100:.2f}%)")
        print(f"    - Through Hitler executed: {hitler_executed_count} games ({hitler_executed_count / total_known_outcome_games * 100:.2f}%)")
        print(f"  Fascist Team: {total_fascist_wins} games ({total_fascist_wins / total_known_outcome_games * 100:.2f}%)")
        print(f"    - Through policy: {fascist_wins} games ({fascist_wins / total_known_outcome_games * 100:.2f}%)")
        print(f"    - Through Hitler election: {hitler_elected_count} games ({hitler_elected_count / total_known_outcome_games * 100:.2f}%)")

    return liberal_counts, fascist_counts, hitler_executed_count, hitler_elected_count, unknown_outcomes


def analyze_elo_ratings(game_files):
    """Analyze ELO ratings across all games."""
    lib_elo_values = []
    fas_elo_values = []
    
    print(f"Analyzing ELO ratings for {len(game_files)} games...")
    
    for file_path in game_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            
            # Extract ELO ratings if they exist
            lib_elo = data.get("libElo", {}).get("overall")
            fas_elo = data.get("fasElo", {}).get("overall")
            
            if lib_elo is not None:
                lib_elo_values.append(lib_elo)
            if fas_elo is not None:
                fas_elo_values.append(fas_elo)
    
    # Calculate statistics
    lib_avg_elo = sum(lib_elo_values) / len(lib_elo_values) if lib_elo_values else 0
    fas_avg_elo = sum(fas_elo_values) / len(fas_elo_values) if fas_elo_values else 0
    
    # Find min and max values
    lib_min_elo = min(lib_elo_values) if lib_elo_values else 0
    lib_max_elo = max(lib_elo_values) if lib_elo_values else 0
    fas_min_elo = min(fas_elo_values) if fas_elo_values else 0
    fas_max_elo = max(fas_elo_values) if fas_elo_values else 0
    
    # Print results
    print("\nELO Rating Analysis:")
    print(f"  Liberal Team (from {len(lib_elo_values)} games):")
    print(f"    - Average ELO: {lib_avg_elo:.2f}")
    print(f"    - Min ELO: {lib_min_elo:.2f}")
    print(f"    - Max ELO: {lib_max_elo:.2f}")
    
    print(f"  Fascist Team (from {len(fas_elo_values)} games):")
    print(f"    - Average ELO: {fas_avg_elo:.2f}")
    print(f"    - Min ELO: {fas_min_elo:.2f}")
    print(f"    - Max ELO: {fas_max_elo:.2f}")
    
    # Calculate average ELO difference
    elo_diffs = []
    for i in range(min(len(lib_elo_values), len(fas_elo_values))):
        elo_diffs.append(lib_elo_values[i] - fas_elo_values[i])
    
    avg_elo_diff = sum(elo_diffs) / len(elo_diffs) if elo_diffs else 0
    print(f"\n  Average ELO difference (Liberal - Fascist): {avg_elo_diff:.2f}")
    
    return lib_avg_elo, fas_avg_elo, lib_elo_values, fas_elo_values


if __name__ == "__main__":
    print("Secret Hitler Game Mode Analysis")
    print("===============================")

    # Analyze games and get separate lists
    avalon_games, regular_games = analyze_games()

    print(f"\nReturned {len(regular_games)} regular game files for further processing")

    # Analyze player counts for regular games
    player_distribution = analyze_player_counts(regular_games)

    # Analyze round counts for regular games
    round_distribution = analyze_round_counts(regular_games)

    # Analyze game outcomes
    liberal_counts, fascist_counts, hitler_executed_count, hitler_elected_count, unknown_outcomes = analyze_game_outcomes(regular_games)
    
    # Analyze ELO ratings
    lib_avg_elo, fas_avg_elo, lib_elo_values, fas_elo_values = analyze_elo_ratings(regular_games)
