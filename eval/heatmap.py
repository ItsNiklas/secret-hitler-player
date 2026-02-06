import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from collections import defaultdict
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
from plot_config import setup_plot_style, extract_model_name, perform_chi_square_test, load_techniques, get_model_imagebox, UNIBLAU

# Apply shared plotting configuration
setup_plot_style()


def get_f1_folders():
    """Get all F1 folder paths."""
    base_path = Path(__file__).parent
    f1_folders = [folder for folder in base_path.iterdir() if folder.is_dir() and folder.name.startswith("runsF1-")]
    return sorted(f1_folders)


def process_annotation_data(folder_path, model_name, techniques):
    """Process annotation data from a single folder."""
    annotation_counts = defaultdict(int)
    total_annotations = 0
    
    print(f"Processing {model_name}...")
    
    # Process all annotation files in this folder
    for filename in os.listdir(folder_path):
        if filename.endswith("-chat-annotated.json"):
            annotated_filepath = folder_path / filename
            
            try:
                with open(annotated_filepath, 'r') as f:
                    annotated_data = json.load(f)
                
                for item in annotated_data:
                    annotations = item.get("annotation", [])
                    
                    # Flatten list in case of nesting and get unique annotations
                    flat_annotations = []
                    for ann in annotations:
                        if isinstance(ann, list):
                            flat_annotations.extend(ann)
                        else:
                            flat_annotations.append(ann)
                    
                    # Filter annotations to only include valid techniques
                    unique_annotations = {ann for ann in flat_annotations if ann in techniques}
                    total_annotations += len(unique_annotations)
                    
                    for annotation in unique_annotations:
                        annotation_counts[annotation] += 1
                        
            except Exception as e:
                print(f"Error processing {annotated_filepath}: {e}")
                continue
    
    # Calculate relative frequencies within this data
    if total_annotations > 0:
        frequencies = np.array([annotation_counts[tech] / total_annotations for tech in techniques])
        counts = np.array([annotation_counts[tech] for tech in techniques])
        print(f"  {model_name}: {total_annotations} total annotations")
    else:
        frequencies = np.zeros(len(techniques))
        counts = np.zeros(len(techniques), dtype=int)
        print(f"  {model_name}: No annotations found")
        
    return frequencies, counts


def process_all_data(techniques):
    """Process all annotation data from F1 folders and human data."""
    model_data = {}
    model_counts = {}
    
    # Process F1 model folders
    f1_folders = get_f1_folders()
    for folder_path in f1_folders:
        model_name = extract_model_name(folder_path.name)
        if os.path.exists(folder_path / "annotationQwen2532B"):
            frequencies, counts = process_annotation_data(folder_path / "annotationQwen2532B", model_name, techniques)
            model_data[model_name] = frequencies
            model_counts[model_name] = counts
    
    # Process human data
    human_folder = Path(__file__).parent.parent / "crawl" / "replay_data" / "annotationQwen2532B"
    if human_folder.exists():
        human_frequencies, human_counts = process_annotation_data(human_folder, "Human", techniques)
        model_data["Human"] = human_frequencies
        model_counts["Human"] = human_counts
    else:
        print(f"Warning: Human data folder not found at {human_folder}")
        model_data["Human"] = np.zeros(len(techniques))
        model_counts["Human"] = np.zeros(len(techniques), dtype=int)
    
    return model_data, model_counts


def perform_human_vs_llm_statistical_test(model_counts, techniques):
    """Perform chi-square test comparing human vs LLM usage patterns.
    
    Args:
        model_counts (dict): Dictionary with model names as keys and count arrays as values
        techniques (list): List of technique names
    """
    print("\n" + "="*60)
    print("STATISTICAL TEST: HUMAN vs LLM USAGE PATTERNS")
    print("="*60)
    
    if "Human" not in model_counts:
        print("Warning: No human data found for statistical test")
        return
    
    # Aggregate all LLM counts
    llm_models = [model for model in model_counts.keys() if model != "Human"]
    if not llm_models:
        print("Warning: No LLM data found for statistical test")
        return
    
    human_counts = model_counts["Human"]
    llm_counts = np.zeros(len(techniques), dtype=int)
    
    for model in llm_models:
        llm_counts += model_counts[model]
    
    # Create contingency table
    contingency_data = np.array([human_counts, llm_counts])
    
    # Create DataFrame for better display
    contingency_df = pd.DataFrame(
        contingency_data,
        index=["Human", "LLM (All Models)"],
        columns=techniques
    )
    
    # Use shared chi-square test function
    try:
        perform_chi_square_test(
            contingency_df,
            test_name="Chi-Square Test: Human vs LLM Usage Patterns",
            group1_name="Human",
            group2_name="LLM (All Models)",
            remove_zero_columns=False,
            show_effect_size_interpretation=True
        )
    except ValueError as e:
        print(f"Chi-square test failed: {e}")


def create_heatmap():
    """Create and display the heatmap."""
    # Load persuasion techniques from parse_annotation.py
    techniques = load_techniques("persuasion_cialdini.jsonl")

    # Process all data (F1 models and human data)
    data, counts = process_all_data(techniques)
    
    # Perform statistical test comparing human vs LLM usage patterns
    perform_human_vs_llm_statistical_test(counts, techniques)
    
    # Extract model names
    models = list(data.keys())

    # Calculate total frequency for each technique to sort x-axis
    technique_totals = {}
    for technique in techniques:
        total = sum(data[model][techniques.index(technique)] for model in models)
        technique_totals[technique] = total

    # Sort techniques by total frequency
    # sorted_techniques = sorted(techniques, key=lambda x: technique_totals[x], reverse=True)

    # Reorder data matrix according to sorted techniques (top 15 only)
    technique_indices = [techniques.index(tech) for tech in techniques]
    matrix = np.array([data[model] for model in models])
    matrix = matrix[:, technique_indices]

    print(matrix)

    # Calculate figure size to make boxes square with more whitespace
    n_models = len(models)
    spacing = 0.5  # Reduced spacing between boxes
    fig_height = max(6, n_models * spacing)

    # Create the plot
    plt.figure(figsize=(5.50, 3))

    # Create custom colormap from 99% light white to UNIBLAU
    colors = ["#f1f1f1", UNIBLAU]  # 99% light white to UNIBLAU
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('white_uniblau', colors, N=n_bins)

    # Create heatmap with separation between cells and square boxes
    ax = sns.heatmap(
        matrix,
        xticklabels=techniques,
        yticklabels=models,
        cmap=cmap,
        annot=False,  # Show frequency values on the heatmap
        fmt=".3f",
        vmin=0,  # Set minimum value for colorscale
        vmax=0.75,  # Set maximum value for colorscale
        cbar_kws={"label": "Relative Frequency", "shrink": 0.4},  # Shrink colorbar to fit better
        linewidths=2.0,  # Increased separation between cells
        linecolor="white",
        square=True,  # Make boxes square
    )

    # Rotate x-axis labels for better readability with more padding
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    
    # Add padding to the left of y-axis labels to make room for icons
    ax.tick_params(axis='x', pad=-3)
    ax.tick_params(axis='y', pad=16)

    # Remove default grid
    ax.grid(False)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    # Add model icons to y-axis (before the model names)
    # Draw the figure to get proper coordinates
    plt.gcf().canvas.draw()
    
    # Get the y-tick positions and labels
    yticks = ax.get_yticks()
    
    # Add icons at each y-tick position
    for i, model_name in enumerate(models):
        imagebox = get_model_imagebox(model_name)
        if imagebox:
            # Position the icon to the left of the y-tick labels
            # Using axis coordinates for proper positioning
            ab = AnnotationBbox(
                imagebox,
                xy=(0, yticks[i]),  # Use the actual tick position
                xycoords=('axes fraction', 'data'),  # x in axes coords, y in data coords
                xybox=(-11, 0),  # Offset to the left of the axis
                boxcoords="offset points",
                frameon=False,
                box_alignment=(0.5, 0.5),  # Center the icon
                zorder=10
            )
            ax.add_artist(ab)

    return plt


if __name__ == "__main__":
    # Create heatmap
    print("Creating persuasion techniques heatmap...")
    plt1 = create_heatmap()
    plt1.savefig("persuasion_heatmap_full.pdf", dpi=300, bbox_inches="tight")
    # plt1.show()

    print("Heatmap created and saved as persuasion_heatmap_full.pdf!")
