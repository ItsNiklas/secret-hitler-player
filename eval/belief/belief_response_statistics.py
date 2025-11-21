# Cell 1: Imports
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set the folder path
path = r"C:/Users/corin/PycharmProjects/secret-hitler-player/crawl/belief_response_tables/"
files = [f for f in os.listdir(path) if f.startswith("belief_response_table_") and f.endswith(".json")]

# Cell 2: Load and aggregate all reasoning categories
all_categories = []

for filename in files:
    with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
        data = json.load(f)
        for round_data in data:
            reasoning_cats = round_data.get("reasoning_categories", {})
            all_categories.extend(reasoning_cats.values())

# Cell 3: Count frequency
category_counts = Counter(all_categories)

# Cell 4: Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()), palette="viridis")
plt.title("Frequency of Reasoning Categories (All Games)")
plt.xlabel("Reasoning Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

