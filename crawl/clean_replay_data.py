#!/usr/bin/env python3
"""
Clean replay data by:
1. Removing all chat messages before "The game begins."
2. Removing observer messages (messages from users not in the game)
"""

import json
import sys
from pathlib import Path
from typing import Set
from datetime import datetime
import traceback

from outcome import Value

# Set to True to enable debug logging
DEBUG = False


def get_player_usernames(data: dict) -> Set[str]:
    """Extract all player usernames from winning and losing players."""
    players = set()
    
    for player in data.get("winningPlayers", []):
        players.add(player["userName"])
    
    for player in data.get("losingPlayers", []):
        players.add(player["userName"])
    
    if DEBUG:
        print(f"  DEBUG: Found {len(players)} players: {players}")
    
    return players


def find_game_begins_timestamp(chats: list) -> str:
    """Find the timestamp of the 'The game begins.' message."""
    for chat in chats:
        if chat.get("gameChat") and isinstance(chat.get("chat"), list):
            for msg in chat["chat"]:
                if isinstance(msg, dict) and msg.get("text") == "The game begins.":
                    return chat.get("timestamp")
    return None


def clean_replay_data(file_path: Path) -> dict:
    """Clean a single replay data file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Data is in format ["replayGameData", {...actual data...}]
    if not isinstance(data, list) or len(data) < 2:
        print(f"Warning: Unexpected data format in {file_path}")
        return {"removed_pre_game": 0, "removed_observer": 0}
    
    game_data = data[1]
    chats = game_data.get("chats", [])
    
    if not chats:
        return {"removed_pre_game": 0, "removed_observer": 0}
    
    # Get player usernames
    players = get_player_usernames(game_data)
    
    # Find "The game begins." timestamp
    game_begins_timestamp = find_game_begins_timestamp(chats)
    
    if DEBUG:
        print(f"  DEBUG: Total chats: {len(chats)}")
        print(f"  DEBUG: Game begins timestamp: {game_begins_timestamp}")
    
    stats = {
        "removed_pre_game": 0,
        "removed_observer": 0
    }
    
    # Step 1: Remove pre-game messages (messages with timestamp before game begins)
    if game_begins_timestamp:
        chats_after_game_start = []
        for chat in chats:
            # Always keep game chat messages regardless of timestamp
            if chat.get("gameChat"):
                chats_after_game_start.append(chat)
                continue
            
            chat_timestamp = chat.get("timestamp")
            if DEBUG:
                print(f"  DEBUG: Chat timestamp: {chat_timestamp}")
            if chat_timestamp and chat_timestamp >= game_begins_timestamp:
                chats_after_game_start.append(chat)
            elif chat_timestamp:
                stats["removed_pre_game"] += 1
        chats = chats_after_game_start
    else:
        print(f"Warning: 'The game begins.' not found in {file_path}")
    
    # Step 2: Remove observer messages (messages from non-players)
    cleaned_chats = []
    for i, chat in enumerate(chats):
        if DEBUG and i < 5:  # Show first 5 messages in debug
            print(f"  DEBUG: Chat {i}: gameChat={chat.get('gameChat')}, " 
                  f"userName={chat.get('userName')}, "
                  f"chat_type={type(chat.get('chat')).__name__ if 'chat' in chat else 'N/A'}")
        
        # Game chat messages have gameChat: true and chat as array - always keep
        if chat.get("gameChat") and isinstance(chat.get("chat"), list):
            cleaned_chats.append(chat)
            if DEBUG and i < 5:
                print(f"    -> KEEPING (game chat)")
        # Player chat messages have userName and chat as string
        elif "userName" in chat and isinstance(chat.get("chat"), str):
            if chat["userName"] in players:
                cleaned_chats.append(chat)
                if DEBUG and i < 5:
                    print(f"    -> KEEPING (player message)")
            else:
                stats["removed_observer"] += 1
                if DEBUG and i < 5:
                    print(f"    -> REMOVING (observer)")
        else:
            # Keep any other format to be safe
            cleaned_chats.append(chat)
            if DEBUG and i < 5:
                print(f"    -> KEEPING (unknown format)")
    
    # Update the data
    game_data["chats"] = cleaned_chats
    data[1] = game_data
    
    if DEBUG:
        print(f"  DEBUG: Cleaned chats: {len(cleaned_chats)}")
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return stats


def main():
    global DEBUG
    
    # Check for debug flag
    if "--debug" in sys.argv:
        DEBUG = True
        print("DEBUG MODE ENABLED")
        print("=" * 60)
    
    # Get the replay_data directory
    script_dir = Path(__file__).parent
    replay_dir = script_dir / "replay_data"
    
    if not replay_dir.exists():
        print(f"Error: Directory {replay_dir} does not exist")
        sys.exit(1)
    
    # Find all JSON files
    json_files = list(replay_dir.glob("*_xhr_data.json"))
    
    if not json_files:
        print(f"No *_xhr_data.json files found in {replay_dir}")
        sys.exit(1)
    
    print(f"Found {len(json_files)} replay files to clean")
    print("-" * 60)
    
    total_stats = {
        "files_processed": 0,
        "total_removed_pre_game": 0,
        "total_removed_observer": 0
    }
    
    for json_file in sorted(json_files):
        try:
            stats = clean_replay_data(json_file)
            total_stats["files_processed"] += 1
            total_stats["total_removed_pre_game"] += stats["removed_pre_game"]
            total_stats["total_removed_observer"] += stats["removed_observer"]
            
            if stats["removed_pre_game"] > 0 or stats["removed_observer"] > 0:
                print(f"{json_file.name}:")
                print(f"  Removed {stats['removed_pre_game']} pre-game messages")
                print(f"  Removed {stats['removed_observer']} observer messages")
            else:
                pass
                # print(f"{json_file.name}: No messages removed")
        except ValueError as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Files processed: {total_stats['files_processed']}")
    print(f"  Total pre-game messages removed: {total_stats['total_removed_pre_game']}")
    print(f"  Total observer messages removed: {total_stats['total_removed_observer']}")


if __name__ == "__main__":
    main()
