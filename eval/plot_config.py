import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json
import os
from scipy.stats import chi2_contingency
from matplotlib.offsetbox import OffsetImage
from PIL import Image


def setup_plot_style(use_latex=True):
    """
    Configure matplotlib with consistent style settings.
    """
    # Apply matplotlib style
    plt.style.use("seaborn-v0_8-muted")

    # Font configuration - Latin Modern Roman with LaTeX support
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Latin Modern Roman"]
    plt.rcParams["text.usetex"] = use_latex
    plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern for math

    # Size configuration
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11

    # Line and marker configuration
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["lines.markersize"] = 6

    # Figure and export configuration
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Keep spines but make them gray and subtle
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.bottom"] = True
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.edgecolor"] = "0.85"  # Light gray
    plt.rcParams["axes.linewidth"] = 0.8
    
    # Remove ticks
    plt.rcParams["xtick.top"] = False
    plt.rcParams["xtick.bottom"] = False
    plt.rcParams["ytick.left"] = False
    plt.rcParams["ytick.right"] = False


# Color palettes for common use cases
ROLE_COLORS = {
    "liberal": "#607795",  # Slate Gray
    "fascist": "#C74E43",  # Jasper
    "hitler": "#8D322A",  # Burnt Umber
}

# ROLE_COLORS_PASTEL = {"liberal": "#98A6AC", "fascist": "#CA8E89", "hitler": "#B86961"}

# University/Logo color palette
UNIBLAU = "#153268"
LOGOBLAU = "#005f9b"
LOGOMITTELBLAU = "#0091c8"
LOGOHELLBLAU = "#50a5d2"
LOGOHELLBLAU30 = "#d2e6fa"

GAMMA = LOGOBLAU
ETA = LOGOHELLBLAU

# Categorical colormap for models
# Each model gets a unique, distinguishable color
# Colors based on Pastel6 from seaborn
MODEL_COLORS = {
    # Gemma models
    "Gemma 3 27B": "#e41a1c",  # Set 2
    "Gemma 3 12B": "#ED6668",  #
    # Qwen models
    "Qwen 3 32B": "#984ea3",  # Set2
    "Qwen 2.5 72B": "#BA89C2",  #
    "Qwen 2.5 32B": "#BA89C2",  #
    "Qwen 2.5 14B": "#BA89C2",  #
    "Qwen 2.5 7B": "#BA89C2",  #
    # Llama models
    "Llama 3.3 70B": "#377eb8",  # Set2
    "Llama 3.1 70B": "#7AA9D0",  #
    "Llama 3.1 8B": "#7AA9D0",  #
    # DeepSeek/R1 models
    "R1 Llama Distill 70B": "#4daf4a",  #
    "R1 Distill 70B": "#4daf4a",  # Alternate name
    # Rule-based agents
    "Rule Agent": "#A0A0A0",  # Gray
    "Random Agent": "#CFCFCF",  # Light Gray
    # Human players
    "Human": "#6B5E62",  # Dark Navy
}
# MODEL_COLORS = {
#     # Gemma models
#     "Gemma 3 27B": "#FF615C",  #
#     "Gemma 3 12B": "#FF8985",  #
#     # Qwen models
#     "Qwen 3 32B": "#A47AFF",  #
#     "Qwen 2.5 72B": "#C7ADFF",  #
#     "Qwen 2.5 32B": "#C7ADFF",  #
#     "Qwen 2.5 14B": "#C7ADFF",  #
#     "Qwen 2.5 7B": "#C7ADFF",  #
#     # Llama models
#     "Llama 3.3 70B": "#69A9ED",  #
#     "Llama 3.1 70B": "#91C0F2",  #
#     "Llama 3.1 8B": "#A4CBF4",  #
#     # DeepSeek/R1 models
#     "R1 Llama Distill 70B": "#61DB7E",  #
#     "R1 Distill 70B": "#61DB7E",  # Alternate name
#     # Rule-based agents
#     "Rule Agent": "#707070",  # Gray
#     "Random Agent": "#A0A0A0",  # Light Gray
#     # Human players
#     "Human": "#6B5E62",  # Dark Navy
# }



# Fallback color for undefined models - VERY OBVIOUS!
MODEL_COLOR_FALLBACK = "#FF00FF"  # Bright Magenta - impossible to miss!

# Define markers for different model families based on keywords
FAMILY_MARKERS = {
    'Human': ('h', 7),      # Circle
    'Gemma 3 12': ('^', 8),      # Square
    'Gemma 3 27': ('v', 8),      # Square
    'Qwen': ('s', 6),       # Pentagon
    'R1': ('o', 7),          # Plus (filled)
    'Llama': ('D', 6),      # Diamond
}

# Metric-based colors (for column-wise coloring)
METRIC_COLORS = {
    "Overall Impact": UNIBLAU,
    "Liberal Impact": ROLE_COLORS["liberal"],
    "Fascist Impact": ROLE_COLORS["fascist"],
    "Hitler Impact": ROLE_COLORS["hitler"],
    "Win Rate": UNIBLAU,
    "Liberal Win Rate": ROLE_COLORS["liberal"],
    "Fascist Win Rate": ROLE_COLORS["fascist"],
    "Hitler Win Rate": ROLE_COLORS["hitler"],
    "GSIR": UNIBLAU,
    r"GSIR\textsubscript{liberal}": ROLE_COLORS["liberal"],
    r"GSIR\textsubscript{fascist}": ROLE_COLORS["fascist"],
    r"GSIR\textsubscript{hitler}": ROLE_COLORS["hitler"],
}

def get_markerdata_for_model(model_name):
    """
    Get marker style based on model family keywords.
    """
    for keyword, marker_data in FAMILY_MARKERS.items():
        if keyword in model_name:
            return marker_data
    return "X"  # Default marker (circle) if no keyword matches


def get_metric_color(metric_name):
    return METRIC_COLORS.get(metric_name, MODEL_COLOR_FALLBACK)


def load_techniques(jsonl_path="persuasion_sh.jsonl"):
    """
    Load persuasion techniques from JSONL file.
    """
    techniques = []
    
    # Try to find the file in the current directory or parent directory
    if not os.path.exists(jsonl_path):
        parent_path = os.path.join("..", jsonl_path)
        if os.path.exists(parent_path):
            jsonl_path = parent_path
        else:
            raise FileNotFoundError(f"Could not find {jsonl_path} in current or parent directory")
    
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            techniques.append(data['ss_technique'])
    
    return techniques


def extract_model_name(folder_name_or_path):
    """
    Extract a clean model name from folder name or path.
    """
    # Convert to Path if it's a string
    if isinstance(folder_name_or_path, str):
        # Check if it looks like a path
        if "/" in folder_name_or_path or "\\" in folder_name_or_path:
            path = Path(folder_name_or_path)
            # Look for folder names that start with 'runs'
            for part in reversed(path.parts):
                if part.startswith("runs"):
                    folder_name = part
                    break
            else:
                # Fallback to using the name directly
                folder_name = path.name
        else:
            folder_name = folder_name_or_path
    else:
        folder_name = str(folder_name_or_path)

    # Remove "runsF1-" or "runs" prefix
    name = folder_name
    if "crawl" in str(folder_name_or_path):
        return "Human"
    if "runs13" in str(folder_name_or_path):
        return "Llama 3.3 70B"
    if name.startswith("runsF1-"):
        name = name.replace("runsF1-", "")
    elif name.startswith("runs"):
        # Extract part after 'runs' prefix (e.g., 'runs1-Qwen3' -> '1-Qwen3')
        name = name.split("-", 1)[-1] if "-" in name else name.replace("runs", "")

    # Clean up the name - apply replacements
    name = name.replace("F1", "").strip("-")
    name = name.replace("G3", "Gemma 3")
    name = name.replace("Q3", "Qwen 3 32B")
    name = name.replace("Llama33", "Llama 3.3")
    name = name.replace("R1Distill", "R1 Llama Distill")
    name = name.replace("-", " ")

    return name

def get_model_color(model_name, warn_on_missing=True):
    """
    Get the color for a specific model from the MODEL_COLORS dict.
    """
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    if warn_on_missing:
        print(f"⚠️  WARNING: No color defined for model '{model_name}'!")
        print(f"   Using fallback color {MODEL_COLOR_FALLBACK} (bright magenta)")
        print(f"   Please add '{model_name}' to MODEL_COLORS in plot_config.py")

    return MODEL_COLOR_FALLBACK


def get_model_colors(model_names, warn_on_missing=True):
    """
    Get colors for multiple models.
    """
    return [get_model_color(name, warn_on_missing) for name in model_names]


def get_model_imagebox(model_name):
    """
    Get an OffsetImage (imagebox) for a model's logo.
    """
    # Internal mapping for logo files - tuples of (width, height, zoom)
    # Some logos are taller, some are wider, adjust dimensions and zoom as needed
    LOGO_CONFIG = {
        "gemma.png": (64, 64, 1/7),
        "qwen.png": (64, 64, 1/7.5),
        "deepseek.png": (64, 64, 1/6),
        "llama.png": (64, 64, 1/6),
        "human.png": (64, 64, 1/8),
    }

    LOGO_MAPPING = {
        "Human": "human.png",
        "Gemma": "gemma.png",
        "Qwen": "qwen.png",
        "R1": "deepseek.png",
        "Llama": "llama.png",
    }

    logo_path = None
    for keyword, logo in LOGO_MAPPING.items():
        if keyword in model_name:
            logo_path = Path(__file__).parent / "logos" / logo
            if logo_path.exists():
                break
    
    if not logo_path:
        return None
    
    # Load the logo with PIL
    img_pil = Image.open(str(logo_path)).convert('RGBA')
    
    # Get configuration (size and zoom) based on logo filename
    width, height, zoom = LOGO_CONFIG.get(logo_path.name)  # Default config
    
    # Resize to thumbnail size while maintaining aspect ratio
    img_pil.thumbnail((width, height), Image.Resampling.LANCZOS)
    
    # Create and return OffsetImage with specified zoom
    imagebox = OffsetImage(np.array(img_pil), zoom=zoom)
    
    return imagebox


def perform_chi_square_test(contingency_table, test_name, group1_name, group2_name, alpha=0.05, remove_zero_columns=True, show_effect_size_interpretation=False):
    """
    Perform chi-square test for homogeneity on a contingency table.

    This is a general-purpose function for testing whether the distribution of
    categorical variables differs significantly between two or more groups.
    """
    # Convert to numpy array if needed
    if hasattr(contingency_table, "values"):  # pandas DataFrame
        data = contingency_table.copy()
        if remove_zero_columns:
            data = data.loc[:, (data != 0).any()]
        contingency_array = data.values
    else:
        contingency_array = np.array(contingency_table)
        if remove_zero_columns:
            # Remove columns that are all zeros
            contingency_array = contingency_array[:, (contingency_array != 0).any(axis=0)]

    print(f"\n--- {test_name} ---")
    if hasattr(contingency_table, "to_string"):
        print(f"Contingency table:")
        print(data.to_string() if remove_zero_columns else contingency_table.to_string())
    else:
        print(f"Contingency table shape: {contingency_array.shape}")

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_array)

    # Calculate Cramer's V (effect size)
    n = contingency_array.sum()  # Total sample size
    min_dim = min(contingency_array.shape[0], contingency_array.shape[1]) - 1
    cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0

    print(f"\nNull hypothesis: Distribution patterns are homogeneous across {group1_name} and {group2_name}")
    print(f"Alternative hypothesis: Distribution patterns differ significantly between groups")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")
    print(f"Cramer's V (effect size): {cramers_v:.4f}")

    if show_effect_size_interpretation:
        if cramers_v < 0.1:
            effect_interpretation = "negligible"
        elif cramers_v < 0.3:
            effect_interpretation = "small"
        elif cramers_v < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        print(f"Effect size interpretation: {effect_interpretation}")

    significant = p_value < alpha
    if significant:
        print(f"Result: SIGNIFICANT (p < {alpha}) - Distribution patterns differ significantly between groups")
    else:
        print(f"Result: NOT SIGNIFICANT (p >= {alpha}) - No significant difference in distribution patterns")

    return {"chi2_stat": chi2_stat, "p_value": p_value, "dof": dof, "cramers_v": cramers_v, "significant": significant}


# Additional color palettes can be added here in the future
# For example:
# PLAYER_COLORS = [...]
# TECHNIQUE_COLORS = {...}
# etc.
