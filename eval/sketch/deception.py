import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-muted")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Latin Modern Roman"]  # matches your lmodern
plt.rcParams["text.usetex"] = True  # use LaTeX for text
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern for math
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 12
plt.rcParams["lines.linewidth"] = 1.0
plt.rcParams["lines.markersize"] = 4
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.right"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def generate_deception_data():
    """Generate fake data showing deception performance over game rounds."""
    np.random.seed(42)
    
    rounds = np.arange(1, 15)  # 10 rounds of gameplay
    
    models = {
        "Llama-3-8B": {"color": "#e74c3c", "linestyle": "-"},
        "Llama-3-13B": {"color": "#f39c12", "linestyle": "-"},
        "Llama-3-70B": {"color": "#27ae60", "linestyle": "-"},
        "Qwen 3 32B": {"color": "#2980b9", "linestyle": "-"},
        "Qwen 3 120B": {"color": "#8e44ad", "linestyle": "-"}
    }
    
    data = {}
    
    for model_name in models.keys():
        if "8B" in model_name:
            # Small model: starts okay but deteriorates quickly
            base_performance = 0.65
            degradation = np.cumsum(np.random.normal(-0.08, 0.02, len(rounds)))
            noise = np.random.normal(0, 0.01, len(rounds))
            performance = base_performance + degradation + noise
            performance = np.clip(performance, 0.1, 1.0)
            
        elif "13B" in model_name:
            # Medium model: moderate degradation
            base_performance = 0.72
            degradation = np.cumsum(np.random.normal(-0.04, 0.02, len(rounds)))
            noise = np.random.normal(0, 0.04, len(rounds))
            performance = base_performance + degradation + noise
            performance = np.clip(performance, 0.2, 1.0)
            
        elif "70B" in model_name:
            # Large model: consistent performance with slight improvement
            base_performance = 0.75
            trend = np.cumsum(np.random.normal(-0.01, 0.015, len(rounds)))
            noise = np.random.normal(0, 0.02, len(rounds))
            performance = base_performance + trend + noise
            performance = np.clip(performance, 0.4, 1.0)
            
        elif "32" in model_name:
            # High-performance reasoning model: very consistent
            base_performance = 0.82
            trend = np.cumsum(np.random.normal(-0.02, 0.01, len(rounds)))
            noise = np.random.normal(0, 0.03, len(rounds))
            performance = base_performance + trend + noise
            performance = np.clip(performance, 0.5, 1.0)
            
        else:  # Claude-3-Opus
            # Another high-performance model: consistent with slight variation
            base_performance = 0.80
            trend = np.cumsum(np.random.normal(0, 0.01, len(rounds)))
            noise = np.random.normal(0, 0.02, len(rounds))
            performance = base_performance + trend + noise
            performance = np.clip(performance, 0.45, 1.0)
        
        data[model_name] = performance
    
    return rounds, data, models


def create_deception_plot():
    """Create a line plot showing deception consistency over game rounds."""
    rounds, data, models = generate_deception_data()
    
    fig = plt.figure(figsize=(5.50, 3))
    ax = fig.add_subplot(111)
    
    # Plot lines for each model
    for model_name, performance in data.items():
        model_config = models[model_name]
        ax.plot(rounds, performance, 
               label=model_name, 
               color=model_config["color"],
               linestyle=model_config["linestyle"],
               linewidth=2.0,
               marker='o' if model_config["linestyle"] == "-" else 's',
               markersize=5,
               alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel("Game Round", fontweight="bold")
    ax.set_ylabel("Deception Consistency Score", fontweight="bold")
    # ax.set_title("Deception Performance Across Game Rounds", fontweight="bold", pad=20)
    
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add simple legend
    ax.legend(loc='best')
    
    # Set axis limits and ticks
    ax.set_xticks(rounds)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    return fig





if __name__ == "__main__":
    print("Creating deception performance plot...")
    
    # Create line plot
    fig = create_deception_plot()
    fig.savefig("deception_consistency_over_rounds.pdf", dpi=300, bbox_inches="tight")
    print("Saved: deception_consistency_over_rounds.pdf")
    
    plt.show()