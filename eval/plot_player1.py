from matplotlib.offsetbox import AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_config import get_model_imagebox, setup_plot_style, get_model_color

# Apply shared plotting configuration
setup_plot_style()
# plt.rcParams["legend.fontsize"] -= 1

data = {
    "Model": [
        "Random Agent",
        "Rule Agent",
        "Llama 3.1 8B",
        r"Qwen 3 32B \small non-thinking",
        "Llama 3.3 70B"
    ],
    "Games Finished": [95.6, None, 90.8, 92.4, 92.4],
    "Picked same chancellor": [8.3, 23.9, 14.9, 15.0, 12.9],
    "Picked same chancellor role": [37.9, 42.4, 36.8, 38.4, 40.5],
    "Picked same chancellor aff.": [48.8, 51.0, 45.2, 49.1, 53.5],
    "Voting agreement same-role govt.": [53.8, 84.6, 44.0, 60.7, 61.8],
    "Voting agreement same-aff. govt.": [53.8, 86.7, 45.6, 62.3, 59.7],
}

df = pd.DataFrame(data)

# Select the metrics to display based on the image
metrics_to_plot = [
    # "Games Finished",
    "Picked same chancellor role",
    "Picked same chancellor aff.",
    "Voting agreement same-role govt.",
    "Voting agreement same-aff. govt.",
]

# Set figure size
plt.figure(figsize=(5.50, 3.5))

# Define positions for the bars
x = np.arange(len(metrics_to_plot))
width = 0.18

# Create the bars for each model
for i, model in enumerate(df["Model"].unique()):
    model_data = df[df["Model"] == model]
    values = [model_data[metric].values[0] for metric in metrics_to_plot]
    
    # Get model name without LaTeX formatting for color lookup
    model_clean = model.replace(r"\small non-thinking", "").strip()
    color = get_model_color(model_clean)

    # Make Random model bars white and transparent
    plt.bar(x + (i - 1) * width, values, width, label=model, zorder=2, edgecolor="black", linewidth=1, color=color)

    # # Add value labels on top of bars
    # for j, val in enumerate(values):
    #     plt.text(x[j] + (i - 1) * width, val + 1, f"{val:.1f}%", ha="center", fontsize=6)

# Remove x-axis ticks and customize labels
plt.xticks([])
for i, label in enumerate(metrics_to_plot):
    plt.text(x[i] + 0.2, -1, label, rotation=8, ha="right", va="top")

# Set y-axis limits to match the image
plt.ylim(0, 100)
plt.ylabel("Accuracy")

# Format y-axis ticks to show percentage
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: rf'{int(y)}%'))

# Add gridlines to match the image
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add legend with specific formatting to match the image
legend = plt.gca().legend(loc="upper left", framealpha=1, 
                    handlelength=2, handletextpad=1.53)

# Add logos to legend - need to draw first to get positions
plt.gcf().canvas.draw()

# Add logos to legend
for model, legend_handle in zip(df["Model"], legend.legend_handles):
    imagebox = get_model_imagebox(model)
    if not imagebox:
        continue
    
    # Get the position of the legend handle (the line)
    # Position the logo at the start of each legend entry
    ab = AnnotationBbox(imagebox, (0.5, 0.5), 
                        xybox=(18, 0),  # Offset to the left of text
                        xycoords=legend_handle,
                        boxcoords="offset points",
                        frameon=False,
                        box_alignment=(0.5, 0.5),  # Center the imagebox
                        zorder=10)  # High zorder to appear in front
    plt.gcf().add_artist(ab)

# Show the plot
plt.savefig("eval1.pdf", bbox_inches="tight", dpi=300)
