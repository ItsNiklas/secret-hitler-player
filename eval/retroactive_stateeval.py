#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "simulator"))
from metric.stateeval import evaluate_gamestate # noqa

PLAYERS_POWERS = {
    5: [None, None, "policy_peek", "execution", "execution", None],
    6: [None, None, "policy_peek", "execution", "execution", None], 
    7: [None, "investigate", "choose_president", "execution", "execution", None],
    8: [None, "investigate", "choose_president", "execution", "execution", None],
    9: ["investigate", "investigate", "choose_president", "execution", "execution", None],
    10: ["investigate", "investigate", "choose_president", "execution", "execution", None],
}


def extract_gamestate_data(game_data, round_idx):
    players = game_data.get("players", [])
    logs = game_data.get("logs", [])
    
    if round_idx >= len(logs):
        raise ValueError(f"Round index {round_idx} out of range (max: {len(logs) - 1})")
    
    liberal_policies = sum(1 for i in range(round_idx) if logs[i].get("enactedPolicy") == "liberal")
    fascist_policies = sum(1 for i in range(round_idx) if logs[i].get("enactedPolicy") == "fascist")
    
    deck_state = logs[round_idx].get("deckState", [])
    president_id = logs[round_idx].get("presidentId")
    president_role = "liberal"
    if president_id is not None and 0 <= president_id < len(players):
        president_role = players[president_id].get("role", "liberal")
    
    power_track = PLAYERS_POWERS.get(len(players), [])
    unlocked_powers = [p for p in power_track[:fascist_policies] if p]
    
    role_guesses_by_liberals = {}
    if round_idx > 0:
        role_guesses_by_liberals = extract_role_guesses(logs[round_idx - 1], players)
    
    return {
        "liberal_policies": liberal_policies,
        "fascist_policies": fascist_policies,
        "deck": {"L": sum(1 for p in deck_state if p == "liberal"), "F": sum(1 for p in deck_state if p == "fascist")},
        "president": president_id,
        "round": round_idx + 1,
        "unlocked_powers": unlocked_powers,
        "president_role": president_role,
        "num_players": len(players),
        "role_guesses_by_liberals": role_guesses_by_liberals
    }, {i: p.get("role", "liberal") for i, p in enumerate(players)}


def extract_role_guesses(log_entry, players):
    role_guesses = {}
    rapid_assessments = log_entry.get("rapidAssessments", {})
    name_to_idx = {p.get("username", ""): i for i, p in enumerate(players)}
    
    for player_idx_str, assessment_text in rapid_assessments.items():
        try:
            player_idx = int(player_idx_str)
        except ValueError:
            continue
            
        if player_idx >= len(players) or players[player_idx].get("role") != "liberal":
            continue
        
        player_guesses = {}
        if assessment_text:
            for line in assessment_text.strip().split('\n'):
                match = re.match(r'^([^:]+):\s*(Liberal|Fascist|Hitler|Unknown)', line.strip(), re.IGNORECASE)
                if match:
                    target_name = match.group(1).strip()
                    role_guess = match.group(2).strip().lower()
                    target_idx = name_to_idx.get(target_name)
                    if target_idx is not None and role_guess in ["liberal", "fascist", "hitler"]:
                        player_guesses[target_idx] = role_guess
        
        if player_guesses:
            role_guesses[player_idx] = player_guesses
    
    return role_guesses


def process_game_file(file_path, dry_run=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
        
        logs = game_data.get("logs", [])
        if not logs:
            return True, f"No logs found, skipping"
        
        if logs and "gameStateScore" in logs[0]:
            return True, f"Already has gameStateScore, skipping"
        
        scores_added = 0
        for round_idx in range(len(logs)):
            try:
                gamestate_data, true_roles = extract_gamestate_data(game_data, round_idx)
                score = evaluate_gamestate(gamestate_data, true_roles)
                if dry_run:
                    print(f"  [DRY RUN] Round {round_idx + 1}: Calculated gameStateScore = {score}")
                logs[round_idx]["gameStateScore"] = score
                scores_added += 1
            except Exception as e:
                return False, f"Error processing round {round_idx}: {e}"
        
        if scores_added and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(game_data, f, indent=4, ensure_ascii=False)
        
        status = "DRY RUN: Would add" if dry_run else "Added"
        return True, f"{status} {scores_added} gameStateScores"
    except Exception as e:
        return False, f"Error processing file: {e}"


def main():
    parser = argparse.ArgumentParser(description="Add gameStateScore to Secret Hitler game JSON files")
    parser.add_argument("folder", help="Folder containing game JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without making changes")
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)
    
    json_files = list(folder_path.glob("*_summary.json"))
    if not json_files:
        print(f"No *_summary.json files found in '{folder_path}'")
        sys.exit(1)
    
    print(f"Found {len(json_files)} game files to process")
    if args.dry_run:
        print("DRY RUN MODE: No files will be modified")
    print()
    
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing {file_path.name}...")
        success, message = process_game_file(file_path, dry_run=args.dry_run)
        
        if success:
            print(f"  ✓ {message}")
            success_count += 1
        else:
            print(f"  ✗ {message}")
            error_count += 1
    
    print()
    print(f"Processing complete:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Errors: {error_count}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()