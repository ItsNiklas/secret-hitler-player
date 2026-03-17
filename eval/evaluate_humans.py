import json
from pathlib import Path
from deception_analysis import calc_deception_rates, load_games_from_folder, _get_alice_role

for folder_path in ["runs-human/runsH-KIMI25", "runs-human/runsH-GPT52", "runs-human/runsH-MISTRALSMALL"]:
    folder = Path(folder_path)
    games = load_games_from_folder(folder)
    rates = calc_deception_rates(games)
    print(f"\n{folder_path}:")
    for r, rate in sorted(rates.items()):
        print(f"  Round {r}: {rate:.1f}%")
