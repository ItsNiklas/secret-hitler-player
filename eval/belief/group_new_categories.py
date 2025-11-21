"""
Initialization phase: After some initialization rounds, gather all named reasoning categories, group them together and add the most relevant ones to the prompt.
"""

import os
import glob
import difflib
import json
from collections import Counter

# --- Config ---
CATEGORY_DIR = "/"  # Directory where new_categories_*.txt files are located
THRESHOLD = 0.7       # Similarity threshold for grouping
OUTPUT_FILENAME = "grouped_new_categories_summary.json"

# --- Step 1: Load raw categories from all files ---
raw_categories = []

for filepath in glob.glob(os.path.join(CATEGORY_DIR, "new_categories_*.txt")):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            category = line.strip()
            if category:
                raw_categories.append(category)

# --- Step 2: Count frequencies ---
category_counts = Counter(raw_categories)

# --- Step 3: Group similar categories ---
def group_similar_categories(categories, threshold=0.7):
    grouped = {}
    categories = sorted(categories, key=lambda x: -category_counts[x])  # Group high-freq first

    while categories:
        base = categories.pop(0)
        grouped[base] = [base]
        to_remove = []
        for other in categories:
            sim = difflib.SequenceMatcher(None, base.lower(), other.lower()).ratio()
            if sim >= threshold:
                grouped[base].append(other)
                to_remove.append(other)
        for rem in to_remove:
            categories.remove(rem)
    return grouped

grouped = group_similar_categories(list(category_counts.keys()), threshold=THRESHOLD)

# --- Step 4: Structure output ---
grouped_output = []
for group_leader, members in grouped.items():
    total_count = sum(category_counts[m] for m in members)
    grouped_output.append({
        "category_group": group_leader,
        "variations": members,
        "count": total_count
    })

# --- Step 5: Save to JSON ---
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    json.dump(sorted(grouped_output, key=lambda x: -x["count"]), f, indent=2)

print(f"Saved grouped categories summary to '{OUTPUT_FILENAME}'")
