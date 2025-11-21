import re
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Process a string argument.")
parser.add_argument("output_file", type=str, help="The output file name.")

args = parser.parse_args()

log_path = rf"C:\Users\corin\PycharmProjects\secret-hitler-player\output_files\{args.output_file}.txt"

# Data structures
players = {}
votes = []
nominations = []
reflections = defaultdict(list)
policy_events = []
game_result = {}
new_categories = []


def extract_reasoning_category(refl: str) -> str:
    refl = refl.lower()

    # Loose matching patterns
    if re.search(r"\b(a)\b.*recent policy", refl):
        return "A: Recent policy"
    if re.search(r"\b(b)\b.*probability", refl):
        return "B: Probability"
    if re.search(r"\b(c)\b.*statement", refl) or re.search(r"\b(c)\b.*statements", refl):
        return "C: Statements"
    if re.search(r"\b(d)\b.*random", refl):
        return "D: Random"

    # Direct letter-based format (e.g. "B: ..." or "C: ...")
    match = re.search(r"\b([ABCD])\s*[:\-.]", refl)
    if match:
        letter = match.group(1).upper()
        return {
            "A": "A: Recent policy",
            "B": "B: Probability",
            "C": "C: Statements",
            "D": "D: Random"
        }.get(letter, "Other")

    # User-defined categories (e.g., "NONE: some explanation")
    if "none" in refl:
        try:
            new_cat = refl.split("none:", 1)[1].strip()
            if new_cat:
                return f"NONE:{new_cat}"
        except IndexError:
            pass

    return "Other"



# Regular expressions
role_pattern = re.compile(r"President (\w+) \((\w+)\) nominates (\w+) \((\w+)\)")
vote_pattern = re.compile(r"(\w+) votes (JA|NEIN)")
inner_thoughts_pattern = re.compile(r"(\w+)'s inner monologue \((\w+)\):")
reflection_pattern = re.compile(r"(\w+) reflection:")
policy_draw_pattern = re.compile(r"draws three policies: ([\w, ]+)")
policy_selection_pattern = re.compile(r"Policy Selection .*?:\nInner thoughts: (.*)")
game_end_pattern = re.compile(r"Game finished.*")
game_winner_pattern = re.compile(r"Game ended with (.*)")
final_votes_pattern = re.compile(r"Votes: (.+)")

# Helper
def clean_text(text):
    return text.replace('\u2013', '-').replace('\u2019', "'").strip()

# Parse the file
with open(log_path, "r", encoding="utf-8", errors="replace") as file:
    lines = file.readlines()

current_player = None
current_role = None
current_thoughts = []
in_thought_block = False

for line in lines:
    line = line.strip()

    # Roles and nominations
    match = role_pattern.search(line)
    if match:
        pres, pres_role, chanc, chanc_role = match.groups()
        players[pres] = pres_role
        players[chanc] = chanc_role
        nominations.append((pres, chanc))
        continue

    # Votes
    vote_match = vote_pattern.search(line)
    if vote_match:
        player, vote = vote_match.groups()
        votes.append((player, vote))
        continue

    # Inner thoughts block start
    thought_match = inner_thoughts_pattern.search(line)
    if thought_match:
        current_player, current_role = thought_match.groups()
        in_thought_block = True
        current_thoughts = []
        continue

    # Inner thoughts block end
    if in_thought_block:
        if "Category:" in line or line.endswith(")") or "reflection" in line:
            reflections[current_player].append(clean_text(" ".join(current_thoughts)))
            in_thought_block = False
            current_player = None
        else:
            current_thoughts.append(line)
        continue

    # Policy draw
    policy_match = policy_draw_pattern.search(line)
    if policy_match:
        policies = policy_match.group(1).split(", ")
        policy_events.append({"type": "draw", "policies": policies})
        continue

    # Game end
    if game_end_pattern.search(line):
        game_result["status"] = "Finished"
    if game_winner_pattern.search(line):
        game_result["winner"] = game_winner_pattern.search(line).group(1)
    if final_votes_pattern.search(line):
        vote_data = final_votes_pattern.search(line).group(1)
        vote_breakdown = dict(
            v.strip().split(": ")
            for v in vote_data.split(",")
            if ": " in v
        )
        game_result["final_votes"] = vote_breakdown

# === Summary Output ===
print("\n=== Game Summary ===")
print(f"Players and Roles:")
for player, role in players.items():
    print(f" - {player}: {role}")

print("\nNominations:")
for pres, chanc in nominations:
    print(f" - {pres} nominated {chanc} for Chancellor")

print("\nVotes:")
for player, vote in votes:
    print(f" - {player} voted {vote}")

print("\nPlayer Reflections and Reasoning:")
for player, thoughts in reflections.items():
    print(f"\n{player} ({players.get(player, 'Unknown')}):")
    for i, thought in enumerate(thoughts, 1):
        print(f"  [{i}] {thought}")

print("\nPolicy Events:")
for event in policy_events:
    print(f" - President drew: {', '.join(event['policies'])}")

print("\nGame Result:")
for key, val in game_result.items():
    print(f" - {key}: {val}")


# Example structure: list of rounds
data_table = []

import json

import datetime

# Extract timestamp for game_id
game_id = "unknown_game_id"
start_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Starting game with")

for line in lines:
    match = start_pattern.search(line)
    if match:
        raw_time = match.group(1)
        # Convert to a filesystem-friendly ID
        dt = datetime.datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
        game_id = dt.strftime("%Y-%m-%d_%H-%M-%S")
        break

# Identify Hitler and fascists
hitler_name = next((name for name, role in players.items() if role.lower() == "hitler"), None)
fascist_names = [name for name, role in players.items() if role.lower() == "fascist"]

# Determine max number of reflections (could vary per player)
max_rounds = max(len(thoughts) for thoughts in reflections.values()) if reflections else 0

# Loop through each round (by reflection index)
for round_number in range(max_rounds):
    round_entry = {
        "game_id": game_id,
        "hitler": hitler_name,
        "fascists": fascist_names,
        "round": round_number + 1,
        "suspicions": {},
        "responses": {},
        "reasoning_categories": {}
    }

    #category = extract_reasoning_category(reflections)

    # If it's a user-defined category (i.e., starts with 'NONE:'), store the new part
    #if category.startswith("NONE:"):
    #    new_cat = category.split("NONE:", 1)[1].strip()
    #    new_categories.append(new_cat)

    #round_entry["reasoning_categories"][player] = category

    for player in players:
        thoughts = reflections.get(player, [])
        if round_number < len(thoughts):
            reflection = thoughts[round_number]
            lower = reflection.lower()

            # Attempt basic extraction of suspects
            suspected_hitler = "unknown"
            suspected_fascist = "unknown"
            for name in players:
                if name.lower() in lower:
                    if "hitler" in lower and suspected_hitler == "unknown":
                        suspected_hitler = name
                    if "fascist" in lower and suspected_fascist == "unknown":
                        suspected_fascist = name

            category = extract_reasoning_category(reflection)

            if category.startswith("NONE:"):
                new_cat = category.split("NONE:", 1)[1].strip()
                new_categories.append(new_cat)

            round_entry["suspicions"][player] = {
                "suspected_hitler": suspected_hitler,
                "suspected_fascist": suspected_fascist
            }
            round_entry["responses"][player] = reflection
            round_entry["reasoning_categories"][player] = category

        else:
            round_entry["suspicions"][player] = {
                "suspected_hitler": "none",
                "suspected_fascist": "none"
            }
            round_entry["responses"][player] = "none"
            round_entry["reasoning_categories"][player] = "none"

    data_table.append(round_entry)
# Save to JSON file
path_crawl = r"C:\Users\corin\PycharmProjects\secret-hitler-player\crawl\belief_response_tables"
filename = rf"\belief_response_table_{game_id}.json"

with open(path_crawl + filename, "w", encoding="utf-8") as f:
    json.dump(data_table, f, ensure_ascii=False, indent=2)

# Save new user-defined categories if any
if new_categories:
    cat_filename = f"new_categories_{game_id}.txt"
    with open(cat_filename, "w", encoding="utf-8") as f:
        for cat in sorted(new_categories):
            f.write(cat + "\n")
    print(f"Saved new reasoning categories to {cat_filename}.")


print(f"\nSaved data to {filename}.")
