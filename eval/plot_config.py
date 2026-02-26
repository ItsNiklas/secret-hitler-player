"""
Shared plot configuration for all eval scripts.

Provides consistent styling (setup_plot_style), color palettes, model name
extraction, logo imagebox helpers, chi-square tests, and plot path utilities.
Imported by every other script in eval/.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json
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
    plt.rcParams["font.serif"] = ["Palatino", "TeX Gyre Pagella", "Times"]

    plt.rcParams["text.usetex"] = use_latex
    plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern for math

    # Size configuration
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10

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

FIG_WIDTH = 5.50

# =============================================================================
# MODEL REGISTRY — single source of truth for every model / baseline
# =============================================================================
# Keys   : folder names exactly as they appear in eval/
# Values : dict with display "name", hex "color", "logo" filename (in
#          eval/logos/, PNG – create manually), and "marker" (style, size).
#
# To add a new model run, just add one row here.  All helper functions
# (extract_model_name, get_model_color, get_model_imagebox, …) read from
# this dict automatically.
# =============================================================================
MODEL_REGISTRY = {
    # ---- LLM models ----
    "runsF2-GEMMA": {
        "name": "Gemma 3 27B",
        "color": "#2E96FF",
        "logo": "gemma.png",
        "marker": ("v", 8),
    },
    "runsF2-AMORALGEMMA": {
        "name": "Amoral Gemma 3 27B",
        "color": "#2E96FF",
        "logo": "huggingface.png",
        "marker": ("^", 8),
        "abliterated": True,
    },
    "runsF2-GPTOSS120B": {
        "name": "GPT-OSS 120B",
        "color": "#000000",
        "logo": "openai.png",
        "marker": ("p", 7),
    },
    "runsF2-GPTOSS120B-DERESTRICTED": {
        "name": "GPT-OSS 120B Derestricted",
        "color": "#000000",
        "logo": "huggingface.png",
        "marker": ("P", 7),
        "abliterated": True,
    },
    "runsF2-GPTOSS20B": {
        "name": "GPT-OSS 20B",
        "color": "#000000",
        "logo": "openai.png",
        "marker": ("h", 7),
    },
    "runsF2-KIMIK25": {
        "name": "Kimi K2.5",
        "color": "#16191E",
        "logo": "moonshot.png",
        "marker": ("s", 6),
    },
    "runsF2-LLAMA3170B": {
        "name": "Llama 3.1 70B",
        "color": "#1D65C1",
        "logo": "llama.png",
        "marker": ("D", 6),
    },
    "runsF2-MISTRALSMALL": {
        "name": "Mistral Small 24B",
        "color": "#FA520F",
        "logo": "mistral.png",
        "marker": ("d", 7),
    },
    "runsF2-NOUSHERMES4": {
        "name": "Nous Hermes 4 70B",
        "color": "#2D6376",
        "logo": "nous.png",
        "marker": ("o", 7),
        "abliterated": True,
    },
    "runsF2-DOLPHINVENICE": {
        "name": "Dolphin Mistral 24B Venice",
        "color": "#6186DB",
        "logo": "dolphin.png",
        "marker": ("8", 7),
        "abliterated": True,
    },
    "runsF2-OLMO": {
        "name": "OLMo 3.1 32B",
        "color": "#F0529C",
        "logo": "allen.png",
        "marker": ("H", 7),
    },
    "runsF2-QWEN35": {
        "name": "Qwen 3.5 397B A17B",
        "color": "#615CED",
        "logo": "qwen.png",
        "marker": ("H", 7),
    },
    # ---- Baselines ----
    "runsF2Base-Cpu": {
        "name": "CPU Baseline",
        "color": "#A0A0A0",
        "logo": "robot.png",
        "marker": ("X", 7),
    },
    "runsF2Base-Random": {
        "name": "Random Agent",
        "color": "#CFCFCF",
        "logo": "robot.png",
        "marker": ("*", 7),
    },
    # ---- Special / non-folder entries ----
    "crawl": {"name": "Human", "color": "#6B5E62", "logo": "human.png", "marker": ("h", 7)},
}

# Derived lookups (auto-generated from MODEL_REGISTRY)
MODEL_COLORS = {v["name"]: v["color"] for v in MODEL_REGISTRY.values()}
MODEL_COLOR_FALLBACK = "#FF00FF"  # Bright Magenta – impossible to miss!

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
    Get (marker_style, marker_size) for *model_name* (display name).
    Looks up the MODEL_REGISTRY by display name.
    """
    for entry in MODEL_REGISTRY.values():
        if entry["name"] == model_name:
            return entry.get("marker", ("X", 7))
    return ("X", 7)  # Default marker if no match


def get_metric_color(metric_name):
    return METRIC_COLORS.get(metric_name, MODEL_COLOR_FALLBACK)


def _resolve_folder_key(folder_name_or_path):
    """
    Resolve a folder name or path to a MODEL_REGISTRY key.
    Returns the key string (e.g. 'runsF2-GEMMA') or the raw folder name
    if no registry entry matches.
    """
    raw = str(folder_name_or_path)

    # If it looks like a path, walk parts to find a 'runs*' or 'crawl' folder
    if "/" in raw or "\\" in raw:
        path = Path(raw)
        for part in reversed(path.parts):
            if part.startswith("runs") or part == "crawl":
                raw = part
                break
        else:
            raw = path.name
    # Strip trailing slash
    raw = raw.rstrip("/")

    # Also check for 'crawl' anywhere in the original string
    if "crawl" in str(folder_name_or_path):
        raw = "crawl"

    return raw


def extract_model_name(folder_name_or_path):
    """
    Extract a clean display name from a folder name or path.

    Looks up MODEL_REGISTRY first; falls back to a simple split-on-dash
    heuristic so new folders still produce *something* readable.
    """
    key = _resolve_folder_key(folder_name_or_path)

    # Registry lookup (exact match)
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]["name"]

    # Fallback: strip common prefixes and replace dashes with spaces
    name = key
    for prefix in ("runsF2-", "runsF1-", "runsF2Base-", "runs"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("-", " ").strip()


def get_model_color(model_name, warn_on_missing=True):
    """
    Get the color for a model display name.
    Checks the derived MODEL_COLORS dict (built from MODEL_REGISTRY).
    """
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    if warn_on_missing:
        print(f"⚠️  WARNING: No color defined for model '{model_name}'!")
        print(f"   Using fallback color {MODEL_COLOR_FALLBACK} (bright magenta)")
        print(f"   Add an entry to MODEL_REGISTRY in plot_config.py")

    return MODEL_COLOR_FALLBACK


def get_model_colors(model_names, warn_on_missing=True):
    """
    Get colors for multiple models.
    """
    return [get_model_color(name, warn_on_missing) for name in model_names]


def get_model_imagebox(model_name):
    """
    Get an OffsetImage (imagebox) for a model's logo.

    Resolves the logo filename from MODEL_REGISTRY (by display name).
    """
    # Per-file sizing overrides  (width, height, zoom)
    LOGO_CONFIG = {
        "gemma.png": (64, 64, 1 / 7),
        "qwen.png": (64, 64, 1 / 8),
        "deepseek.png": (64, 64, 1 / 6),
        "llama.png": (64, 64, 1 / 7),
        "human.png": (64, 64, 1 / 8),
        "openai.png": (64, 64, 1 / 8.5),
        "moonshot.png": (64, 64, 1 / 8),
        "huggingface.png": (64, 64, 1 / 8),
        "nous.png": (64, 64, 1 / 7.5),
        "gemma.png": (64, 64, 1 / 6.5),
        "olmo.png": (64, 64, 1 / 7.5),
    }
    LOGO_DEFAULT = (64, 64, 1 / 7)  # sensible default for new logos

    # Find the logo filename from MODEL_REGISTRY by display name
    logo_file = None
    for entry in MODEL_REGISTRY.values():
        if entry["name"] == model_name:
            logo_file = entry.get("logo")
            break

    if not logo_file:
        return None

    logo_path = Path(__file__).parent / "logos" / logo_file
    if not logo_path.exists():
        return None

    # Load the logo with PIL
    img_pil = Image.open(str(logo_path)).convert("RGBA")

    # Get configuration (size and zoom) based on logo filename
    width, height, zoom = LOGO_CONFIG.get(logo_path.name, LOGO_DEFAULT)

    # Resize to thumbnail size while maintaining aspect ratio
    img_pil.thumbnail((width, height), Image.Resampling.LANCZOS)

    # Create and return OffsetImage with specified zoom
    return OffsetImage(np.array(img_pil), zoom=zoom)


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


PLOTS_DIR = Path(__file__).parent / "plots"


def get_plot_path(filename):
    """Get the full path for saving a plot to the plots/ directory.

    Ensures the plots/ directory exists and returns the full path.
    All plotting scripts should use this to save their output.
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    return PLOTS_DIR / filename


def load_summary_file(file_path):
    """Load and parse a JSON summary file.

    Returns the parsed JSON dict, or None on failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None
