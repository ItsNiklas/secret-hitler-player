#!/usr/bin/env python3
"""
Model comparison chart from a LaTeX results table.

Reads a LaTeX table, picks a baseline model, and draws horizontal bars
showing each model's relative difference to the baseline.

Usage: python model_comparison.py <latex_file> [options]
  latex_file      Path to the LaTeX table file
  --metrics       Metric columns to plot (default: all)
  --baseline      Row index of baseline model (default: auto-detect Human)
  --baseline-name Name to match for baseline (default: Human)
  --color-mode    'model' or 'metric' colouring (default: model)
  -o, --output    Output PDF path (default: plots/model_comparison.pdf)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import re
from matplotlib.offsetbox import AnnotationBbox
from plot_config import (
    setup_plot_style,
    get_model_color,
    extract_model_name,
    get_model_imagebox,
    get_metric_color,
    get_plot_path,
)

BAR_HEIGHT = 0.45
FMT = '.3f'
FMT_X = '.1f'
USE_PERCENTAGE = False  # Set to False for non-percentage metrics


def clean_latex_text(text):
    """
    Remove LaTeX formatting commands from text.
    
    Args:
        text: String with LaTeX commands
        
    Returns:
        Cleaned string with LaTeX commands removed
    """
    # Remove complete \makebox construct: \makebox[...]{...}
    # This handles cases like: \makebox[1em][c]{\includegraphics[...]{...}}
    # We need to match the whole construct including nested braces
    while r'\makebox' in text:
        # Find \makebox
        start = text.find(r'\makebox')
        if start == -1:
            break
        
        # Skip past the optional arguments [...][...]
        pos = start + len(r'\makebox')
        while pos < len(text) and text[pos:pos+1] == '[':
            # Find matching ]
            depth = 1
            pos += 1
            while pos < len(text) and depth > 0:
                if text[pos] == '[':
                    depth += 1
                elif text[pos] == ']':
                    depth -= 1
                pos += 1
        
        # Now we should be at the {content} part
        if pos < len(text) and text[pos] == '{':
            # Find matching closing brace
            depth = 1
            pos += 1
            brace_start = pos - 1
            while pos < len(text) and depth > 0:
                if text[pos] == '{':
                    depth += 1
                elif text[pos] == '}':
                    depth -= 1
                pos += 1
            
            # Remove the entire \makebox[...][...]{...} construct
            text = text[:start] + text[pos:]
        else:
            # Malformed, just remove "\makebox"
            text = text[:start] + text[start + len(r'\makebox'):]
    
    # Remove \includegraphics commands (should be gone with makebox, but just in case)
    text = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', '', text)
    
    # Remove \textbf{...} but keep content, preserving any nested commands like \textsubscript
    # Use a more sophisticated regex that captures everything including nested commands
    max_iterations = 10
    iteration = 0
    while r'\textbf{' in text and iteration < max_iterations:
        # Match \textbf{...} where ... can contain other LaTeX commands
        # Use a non-greedy match to handle nested braces properly
        new_text = re.sub(r'\\textbf\{((?:[^{}]|(?:\{[^{}]*\}))*)\}', r'\1', text)
        if new_text == text:  # No change, avoid infinite loop
            break
        text = new_text
        iteration += 1
    
    # Remove other common formatting commands while keeping content
    text = re.sub(r'\\text(it|tt|sf|rm)\{(.*?)\}', r'\2', text)
    text = re.sub(r'\\emph\{(.*?)\}', r'\1', text)
    
    # Remove percentage backslash escape
    text = text.replace(r'\%', '%')
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def parse_latex_table(latex_file):
    """
    Parse a LaTeX table to extract model data.
    
    Expected format:
    - First data row is the baseline model
    - Subsequent rows are comparison models
    - Columns contain various metrics
    
    Returns:
        pd.DataFrame: Table with model names and metrics
    """
    with open(latex_file, 'r') as f:
        content = f.read()
    
    # Extract table rows (between \begin{tabular} and \end{tabular})
    table_match = re.search(r'\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}', content, re.DOTALL)
    if not table_match:
        raise ValueError("Could not find tabular environment in LaTeX file")
    
    table_content = table_match.group(1)
    
    # Split into rows and parse
    rows = []
    for line in table_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('%') or '\\hline' in line or '\\toprule' in line or '\\midrule' in line or '\\bottomrule' in line:
            continue
        
        # Remove trailing \\ and split by &
        if '\\\\' in line:
            line = line.replace('\\\\', '')
        
        # Remove LaTeX comments (% and everything after)
        if ' %' in line:
            line = line.split(' %')[0]
        
        cells = [cell.strip() for cell in line.split('&')]
        
        # Clean LaTeX formatting from each cell
        cells = [clean_latex_text(cell) for cell in cells]

        print(cells)
        
        if len(cells) > 1 and any(cells):  # Valid data row with content
            rows.append(cells)
    
    if not rows:
        raise ValueError("No data rows found in LaTeX table")
    
    # First row is header, rest are data
    headers = rows[0]
    data = rows[1:]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    # Convert numeric columns
    for col in df.columns[1:]:  # Skip model name column
        # Remove % signs and convert to numeric
        df[col] = pd.to_numeric(df[col].str.replace('%', '').str.replace('\\%', ''), errors='coerce')
    
    return df



def create_comparison_chart(df, metric_columns=None, output_path=None, baseline_idx=None, baseline_name_match=None, color_mode='model'):
    """
    Create a comparison chart with baseline in top row and deltas in bottom rows.
    
    Args:
        df: DataFrame with model data (first column is model names)
        metric_columns: List of column names to plot (default: all numeric columns)
        output_path: Where to save the figure (optional)
        baseline_idx: Index of baseline model row (default: None, will search for 'Human')
        baseline_name_match: String to match in model name for baseline (default: 'Human')
        color_mode: 'model' (default) for model-based colors, 'metric' for column-based colors
    """
    setup_plot_style(use_latex=True)
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    
    # Get model names and metrics
    model_col = df.columns[0]
    models = df[model_col].tolist()
    
    # Find baseline index if not specified
    if baseline_idx is None:
        if baseline_name_match is None:
            baseline_name_match = 'Human'
        
        # Search for baseline by name
        for i, model in enumerate(models):
            if baseline_name_match.lower() in model.lower():
                baseline_idx = i
                break
        
        if baseline_idx is None:
            print(f"Warning: Could not find '{baseline_name_match}' in models, using first model as baseline")
            baseline_idx = 0
    
    if metric_columns is None:
        # Use all numeric columns
        metric_columns = [col for col in df.columns[1:] if pd.api.types.is_numeric_dtype(df[col])]
    
    n_metrics = len(metric_columns)
    baseline_model = models[baseline_idx]
    comparison_models = [m for i, m in enumerate(models) if i != baseline_idx]
    comparison_indices = [i for i in range(len(models)) if i != baseline_idx]
    n_comparisons = len(comparison_models)
    
    # Reverse the order of comparison models
    comparison_models = list(reversed(comparison_models))
    comparison_indices = list(reversed(comparison_indices))
    
    # Extract clean model names
    baseline_name = extract_model_name(baseline_model)
    comparison_names = [extract_model_name(m) for m in comparison_models]
    
    # Calculate figure dimensions
    fig_width = 5.50  # Standard text width
    fig_height = 2.3 #(1 + n_comparisons) * row_height + 0.8  # rows + spacing
    
    # Create figure with height ratios (1 for baseline, n_comparisons for others)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, n_metrics, height_ratios=[1, n_comparisons], 
                         hspace=0.55)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_metrics)] for i in range(2)])
    
    # Ensure axes is 2D array even for single column
    if n_metrics == 1:
        axes = axes.reshape(2, 1)
    
    # Get baseline values
    baseline_values = df.iloc[baseline_idx][metric_columns].values
    baseline_color = get_model_color(baseline_name)
    
    # Calculate global min/max for consistent axis scaling
    all_values = df[metric_columns].values.flatten()
    all_values = all_values[~np.isnan(all_values)]  # Remove NaN values
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    # For top plot: if all values are non-negative, fix minimum at 0
    if global_min >= 0:
        xlim_min = 0
        # Add padding to max (30% of range)
        value_range = global_max - global_min
        padding = value_range * 0.3
        xlim_max = global_max + padding
    else:
        # Add some padding (30% on each side)
        value_range = global_max - global_min
        padding = value_range * 0.3
        xlim_min = global_min - padding
        xlim_max = global_max + padding
    
    # For delta plots, calculate max absolute delta
    max_abs_delta = 0
    for col_idx, metric in enumerate(metric_columns):
        baseline_val = baseline_values[col_idx]
        for i in range(len(models)):
            if i != baseline_idx:
                delta = abs(df.iloc[i][metric] - baseline_val)
                max_abs_delta = max(max_abs_delta, delta)
    
    # Add padding for delta range
    delta_padding = max_abs_delta * 0.2
    delta_lim = max_abs_delta + delta_padding
    
    # Top row: Baseline model values
    for col_idx, (metric, value) in enumerate(zip(metric_columns, baseline_values)):
        ax = axes[0, col_idx]
        
        # Choose color based on color mode
        if color_mode == 'metric':
            bar_color = get_metric_color(metric)
        else:
            bar_color = baseline_color
        
        # Create horizontal bar
        bar = ax.barh([0], [value], color=bar_color, height=BAR_HEIGHT)
        
        # Add value label - position based on available space
        label_offset = (xlim_max - xlim_min) * 0.02  # 2% of range
        pct_suffix = '\\%' if USE_PERCENTAGE else ''
        if not USE_PERCENTAGE:
            label_offset = (xlim_max - xlim_min) * 0.02  # 2% of range
            pct_suffix = '\\%' if USE_PERCENTAGE else ''
            
            # Format label with + for positive values, $-$ (math mode) for negative
            if value > 0:
                label_text = f'+{value:{FMT}}{pct_suffix}'
            elif value < 0:
                label_text = f'$-${abs(value):{FMT}}{pct_suffix}'
            else:
                label_text = f'{value:{FMT}}{pct_suffix}'
            
            ax.text((value if value > 0 else 0) + label_offset, 0, label_text, ha='left', va='center', 
                   fontweight='bold', color='black', fontsize=8)
        else:
            ax.text((value if value > 0 else 0) + label_offset, 0, f'{value:{FMT}}{pct_suffix}', ha='left', va='center', 
               fontweight='bold', color='black', fontsize=8)
        
        # Configure axis - dynamic range based on data
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        # Format x tick labels with +/- for non-percentage mode
        if USE_PERCENTAGE:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:{FMT_X}}{pct_suffix}'))
        else:
            def format_baseline_tick(x, pos):
                if abs(x) < 1e-12:
                    return '0'
                elif x > 0:
                    return f'+{x:{FMT_X}}{pct_suffix}'
                else:
                    return f'$-${abs(x):{FMT_X}}{pct_suffix}'
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_baseline_tick))
        ax.tick_params(axis='x', color='0.85', labelcolor='0')
        ax.tick_params(axis='y', color='0.85', labelcolor='0')
        
        if col_idx == 0:
            ax.set_yticklabels([baseline_name])
            
            # Add model icon to the right of the label
            imagebox = get_model_imagebox(baseline_name)
            if imagebox is not None:
                ax.tick_params(axis='y', pad=16)  # Add padding for icons
                ab = AnnotationBbox(
                    imagebox,
                    xy=(0, 0),  # y-tick position
                    xycoords=('axes fraction', 'data'),
                    xybox=(-11, 0),  # Offset to the right of the y-axis
                    boxcoords="offset points",
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                    zorder=10
                )
                ax.add_artist(ab)
        else:
            ax.set_yticklabels([])
        
        ax.tick_params(axis='x', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(metric, fontsize=9)

    # Bottom rows: Comparison model deltas
    for col_idx, metric in enumerate(metric_columns):
        ax = axes[1, col_idx]
        
        baseline_val = baseline_values[col_idx]
        
        # Calculate deltas for each comparison model (in reversed order)
        deltas = []
        colors = []
        for model in comparison_models:
            # Find the model's index in the original list
            orig_idx = models.index(model)
            
            model_val = df.iloc[orig_idx][metric]
            delta = model_val - baseline_val
            deltas.append(delta)
            
            # Choose color based on color mode
            if color_mode == 'metric':
                color = get_metric_color(metric)
            else:
                # Get color for this model
                model_name = extract_model_name(model)
                color = get_model_color(model_name)
            
            colors.append(color)
        
        # Plot bars
        y_pos = np.arange(len(deltas))
        bars = ax.barh(y_pos, deltas, color=colors, height=BAR_HEIGHT)
        
        # Add value labels
        label_offset = delta_lim * 0.03  # 3% of range
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            # Skip very small values but keep exactly 0.0
            if abs(delta) < 0.001 and abs(delta) > 1e-12:
                continue
            
            width = bar.get_width()
            
            # Position label on the right for positive, left for negative (and zero)
            if delta >= 0:
                label_x = width + label_offset
                ha = 'left'
            else:
                label_x = 0 + label_offset
                ha = 'left'
            
            pct_suffix = '\\%' if USE_PERCENTAGE else ''
            
            # Format label with + for positive values, $-$ (math mode) for negative
            if delta > 0:
                label_text = f'+{delta:{FMT}}{pct_suffix}'
            elif delta < 0:
                label_text = f'$-${abs(delta):{FMT}}{pct_suffix}'
            else:
                label_text = f'{delta:{FMT}}{pct_suffix}'
            
            ax.text(label_x, bar.get_y() + bar.get_height()/2, label_text, 
                   ha=ha, va='center', color='black', fontweight='bold', fontsize=8)
        
        # Configure axis - dynamic range based on deltas
        ax.set_xlim(-delta_lim, delta_lim)
        # Format x tick labels: show exact zero as '0' (fixed), others with + for positive and $-$ for negative
        pct_suffix = '\\%' if USE_PERCENTAGE else ''
        def format_delta_tick(x, pos):
            if abs(x) < 1e-12:
                return '0'
            elif x > 0:
                return f'+{x:{FMT_X}}{pct_suffix}'
            else:
                return f'$-${abs(x):{FMT_X}}{pct_suffix}'
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_delta_tick))
        ax.set_ylim(-0.5, len(deltas) - 0.5)
        ax.set_yticks(y_pos)
        ax.tick_params(axis='x', color='0.85', labelcolor='0')
        ax.tick_params(axis='y', color='0.85', labelcolor='0')

        if col_idx == 0:
            ax.set_yticklabels(comparison_names)
            
            # Check if any model has an icon - if so, add padding for all
            has_any_icon = any(get_model_imagebox(name) is not None for name in comparison_names)
            if has_any_icon:
                ax.tick_params(axis='y', pad=16)  # Add padding for icons
            
            # Add model icons to the right of each label
            for i, model_name in enumerate(comparison_names):
                imagebox = get_model_imagebox(model_name)
                if imagebox is not None:
                    ab = AnnotationBbox(
                        imagebox,
                        xy=(0, y_pos[i]),  # Use the actual tick position
                        xycoords=('axes fraction', 'data'),
                        xybox=(-11, 0),  # Offset to the right of the y-axis
                        boxcoords="offset points",
                        frameon=False,
                        box_alignment=(0.5, 0.5),
                        zorder=10
                    )
                    ax.add_artist(ab)
        else:
            ax.set_yticklabels([])
        
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel(f'$\\Delta$ {metric}', fontsize=9)
        ax.axvline(0, color='0.85', linewidth=0.8, linestyle='-')
        
        # Add light grid with dashed lines
        # ax.grid(True, axis='x', alpha=0.3, linestyle='--', zorder=0)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    else:
        default_path = get_plot_path("model_comparison.pdf")
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {default_path}")
    
    plt.close()
    return fig


def main():
    """
    Main function to generate the comparison chart from LaTeX table.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create model comparison chart from LaTeX table')
    parser.add_argument('latex_file', help='Path to LaTeX table file')
    parser.add_argument('--metrics', nargs='+', help='Metric columns to plot (default: all)')
    parser.add_argument('--baseline', type=int, default=None, 
                       help='Index of baseline model row (default: auto-detect Human)')
    parser.add_argument('--baseline-name', type=str, default='Human',
                       help='Name to match for baseline model (default: Human)')
    parser.add_argument('--color-mode', type=str, default='model', choices=['model', 'metric'],
                       help='Color mode: "model" for model-based colors, "metric" for column-based colors (default: model)')
    parser.add_argument('--output', '-o', help='Output file path (default: plots/model_comparison.pdf)')
    
    args = parser.parse_args()
    
    # Parse LaTeX table
    print(f"Parsing LaTeX table from: {args.latex_file}")
    df = parse_latex_table(args.latex_file)
    print(f"\nParsed {len(df)} models:")
    print(df.to_string())
    
    # Generate the chart
    fig = create_comparison_chart(
        df, 
        metric_columns=args.metrics,
        output_path=args.output,
        baseline_idx=args.baseline,
        baseline_name_match=args.baseline_name,
        color_mode=args.color_mode
    )


if __name__ == '__main__':
    main()
