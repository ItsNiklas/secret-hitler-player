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
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


def generate_persuasion_data_cloud():
    """Generate cloud of data points (n=200 each) for each model using simple 2D normal distributions."""
    np.random.seed(456)
    
    # Same models as deception.py
    models = {
        "Llama-3-8B": {"color": "#e74c3c"},
        "Llama-3-13B": {"color": "#f39c12"},
        "Llama-3-70B": {"color": "#27ae60"},
        "Qwen 3 32B": {"color": "#2980b9"},
        "Qwen 3 120B": {"color": "#8e44ad"}
    }
    
    n_points = 200
    data = {}
    
    # Define means and covariance for each model (centered around 5, 0.5 with moderate variance)
    model_params = {
        "Llama-3-8B": {"mean": [3.5, 0.35], "cov": [[1.15, 0.06], [0.06, 0.025]]},
        "Llama-3-13B": {"mean": [4.2, 0.42], "cov": [[1.25, 0.065], [0.065, 0.028]]},
        "Llama-3-70B": {"mean": [5.0, 0.50], "cov": [[1.4, 0.075], [0.075, 0.032]]},
        "Qwen 3 32B": {"mean": [5.8, 0.58], "cov": [[1.35, 0.07], [0.07, 0.03]]},
        "Qwen 3 120B": {"mean": [6.5, 0.65], "cov": [[1.3, 0.065], [0.065, 0.028]]}
    }
    
    for model_name in models.keys():
        params = model_params[model_name]
        
        # Generate 200 points using multivariate normal with covariance (45-degree stretch)
        points = np.random.multivariate_normal(params["mean"], params["cov"], n_points)
        adaptabilities = points[:, 0]
        vote_sway_rates = points[:, 1]
        
        data[model_name] = {
            "vote_sway_rates": vote_sway_rates,
            "adaptabilities": adaptabilities
        }
    
    return data, models


def create_persuasion_scatter_cloud():
    """Create a scatter plot with 200 data points per model showing persuasion effectiveness."""
    data, models = generate_persuasion_data_cloud()
    
    fig = plt.figure(figsize=(6.46, 4))
    ax = fig.add_subplot(111)
    
    # Plot cloud of points for each model
    for model_name in data.keys():
        model_data = data[model_name]
        color = models[model_name]["color"]
        
        # Plot 200 tiny points for this model
        ax.scatter(model_data["adaptabilities"], model_data["vote_sway_rates"], 
                  c=color, s=10, alpha=0.6, 
                  label=model_name, edgecolors='none')
    
    # Customize the plot
    ax.set_xlabel("Unique Persuasion Techniques per Round", fontweight="bold")
    ax.set_ylabel("Vote Sway Success Rate", fontweight="bold")
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set limits with some padding
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.0, 1.0)
    
    # Legend for models
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Creating persuasion success cloud plot...")
    
    fig = create_persuasion_scatter_cloud()
    fig.savefig("persuasion_success_cloud.pdf", dpi=300, bbox_inches="tight")
    print("Saved: persuasion_success_cloud.pdf")
    
    plt.show()