import json
from pathlib import Path
from collections import defaultdict
import re

def _parse_rapid_assessment(text, player_name):
    # Find the last occurrence of this player name (whole-word, case-insensitive)
    matches = list(re.finditer(r"\b" + re.escape(player_name) + r"\b", text, re.IGNORECASE))
    if not matches:
        return "unknown"
    search_start = matches[-1].start()

    # Scan forward from that point for the first role keyword
    m = re.search(r"\b(liberal|fascist|hitler|unknown)", text[search_start:], re.IGNORECASE)
    if m is None:
        return "unknown"
    return m.group(1).lower()

def analyze_speed(folder_path):
    folder = Path(folder_path)
    games = list(folder.glob("*_summary.json"))
    
    rounds_to_id = []
    
    for game_file in games:
        with open(game_file, 'r') as f:
            game = json.load(f)
            
        roles = {p["username"]: p["role"].lower() for p in game.get("players", [])}
        
        # Find Alice (the model)
        alice_name = next((n for n in roles if n.startswith("Alice")), None)
        if not alice_name:
            continue
            
        alice_role = roles[alice_name]
        
        # Track for each player, when they first got it right
        for p in game.get("players", []):
            name = p["username"]
            if name == alice_name:
                continue
                
            first_correct_round = -1
            
            for round_idx, log in enumerate(game.get("logs", []), start=1):
                ra = log.get("rapidAssessments")
                if not ra:
                    continue
                    
                pid_str = str(p["_id"])
                if pid_str in ra:
                    assessment_text = ra[pid_str]
                    belief = _parse_rapid_assessment(assessment_text, alice_name)
                    if belief == alice_role:
                        first_correct_round = round_idx
                        break
            
            if first_correct_round != -1:
                rounds_to_id.append((first_correct_round, alice_role))
            else:
                rounds_to_id.append((-1, alice_role)) # never identified
                
    return rounds_to_id

folders_to_check = [
    "runs-human/runsH-KIMI25",
    "runs-human/runsH-GPT52",
    "runs-human/runsH-MISTRALSMALL",
    "runsF2-KIMIK25",
    "runsF2-GPT52",
    "runsF2-MISTRALSMALL"
]

for f in folders_to_check:
    res = analyze_speed(f)
    identified = [x[0] for x in res if x[0] != -1]
    never = len([x for x in res if x[0] == -1])
    
    if identified:
        avg = sum(identified) / len(identified)
        print(f"{f:30s} : Avg rounds to ID = {avg:.2f} (n={len(identified)}), Never ID'd = {never} ({(never/len(res))*100:.1f}%)")
    else:
        print(f"{f:30s} : No identifications (n={len(res)})")
