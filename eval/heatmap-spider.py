import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
import os
from collections import defaultdict
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox
from plot_config import FAMILY_MARKERS, get_markerdata_for_model, setup_plot_style, extract_model_name, perform_chi_square_test, get_model_colors, get_model_imagebox, load_techniques

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


def create_spider_plot():
    """Create and display the spider/radar plot."""
    # Load persuasion techniques from parse_annotation.py
    techniques = load_techniques("persuasion_cialdini.jsonl")

    # Process all data (F1 models and human data)
    data, counts = process_all_data(techniques)
    print(data)
    
    # Perform statistical test comparing human vs LLM usage patterns
    perform_human_vs_llm_statistical_test(counts, techniques)
    
    # Extract model names
    models = list(data.keys())

    # Number of techniques (axes)
    num_vars = len(techniques)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # The plot is circular, so we need to "complete the loop" by appending the first value
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5.50, 4), subplot_kw=dict(projection='polar'))
    
    # Get colors for each model using the shared MODEL_COLORS
    colors = get_model_colors(models, warn_on_missing=True)
    
    # Plot each model
    plot_marker_sizes = {}  # Store original sizes for legend
    for idx, model in enumerate(models):
        values = data[model].tolist()
        values += values[:1]  # Complete the loop
        
        linewidth = 1.5
    
        # Select marker style based on model family
        marker, marker_size = get_markerdata_for_model(model)
        plot_marker_sizes[model] = marker_size  # Store original size
        
        ax.plot(angles, values, linewidth=linewidth, label=model, 
                color=colors[idx], linestyle='-', 
                marker=marker, markersize=6,
                markerfacecolor=colors[idx], markeredgecolor='white', markeredgewidth=1)
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # Set the labels for each axis with position-specific padding
    ax.set_xticks(angles[:-1])
    
    # Apply different padding based on label position
    # For 6 labels: adjust horizontal (left/right) more than vertical (top/bottom)
    for angle, technique in zip(angles[:-1], techniques):
        label = ax.text(angle, ax.get_ylim()[1] * 1.15, technique, 
                       size=11, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.05', facecolor='white', edgecolor='none', alpha=1))
        
        # Adjust horizontal alignment and position for left/right labels
        angle_deg = np.degrees(angle) % 360
        if 50 < angle_deg < 130 or 200 < angle_deg < 350:  # Near top or bottom
            label.set_position((angle, ax.get_ylim()[1] * 1.05))
        else:  # Near left or right - need more space
            label.set_position((angle, ax.get_ylim()[1] * 1.2))
        
    
    # Remove default tick labels since we're using custom text
    ax.set_xticklabels([])

    
    # Set y-axis limits and labels
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0.15, 0.30, 0.45, 0.60, 0.75])
    ax.set_yticklabels([r'15\%', r'30\%', r'45\%', r'60\%', r'75\%'], size=9,
                       bbox=dict(boxstyle='round,pad=0.05', facecolor='white', edgecolor='none', alpha=1))
    
    # Add grid
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Make the outer layer (polar spine) match the grid style instead of bold black
    ax.spines['polar'].set_linewidth(0.8)
    ax.spines['polar'].set_alpha(0.3)
    ax.spines['polar'].set_color('#888888')
    
    # Add legend with original marker sizes
    legend = ax.legend(framealpha=0, 
                      bbox_to_anchor=(0.5, -0), loc='upper center',
                      handlelength=2, handletextpad=1.4, ncol=3)
    
    # Update legend markers to use original sizes
    for idx, (model, legend_handle) in enumerate(zip(models, legend.legend_handles)):
        if hasattr(legend_handle, 'set_markersize'):
            legend_handle.set_markersize(plot_marker_sizes[model])
    
    # Add logos to legend - need to draw first to get positions
    fig.canvas.draw()
    
    # Add logos to legend
    for idx, (model, legend_handle) in enumerate(zip(models, legend.legend_handles)):
        imagebox = get_model_imagebox(model)
        if not imagebox:
            continue
        
        # Get the position of the legend handle (the line)
        # Position the logo at the start of each legend entry
        ab = AnnotationBbox(imagebox, (0.5, 0.5), 
                           xybox=(19, 0),  # Offset to the left of text
                           xycoords=legend_handle,
                           boxcoords="offset points",
                           frameon=False,
                           box_alignment=(0.5, 0.5),  # Center the imagebox
                           zorder=10)  # High zorder to appear in front
        fig.add_artist(ab)
    
    # Adjust layout to prevent clipping (?)
    plt.tight_layout()
    
    return plt


if __name__ == "__main__":
    # Create spider plot
    print("Creating persuasion techniques spider plot...")
    plt1 = create_spider_plot()
    plt1.savefig("persuasion_spider_plot.pdf", dpi=300, bbox_inches="tight")
    # plt1.show()

    print("Spider plot created and saved as persuasion_spider_plot.pdf!")
