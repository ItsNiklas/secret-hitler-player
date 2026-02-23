#!/usr/bin/env python3
"""
Single-model comparison chart from a LaTeX results table.

Like model_comparison.py but uses absolute values instead of relative
differences; suited for comparing a single model's metrics.

Usage: python model_comparison-single.py <latex_file> [options]
  latex_file    Path to the LaTeX table file
  --metrics     Metric columns to plot (default: all)
  --color-mode  'model' or 'metric' colouring (default: model)
  -o, --output  Output PDF path (default: plots/model_comparison_single.pdf)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox
from plot_config import (
    setup_plot_style,
    get_model_color,
    extract_model_name,
    get_model_imagebox,
    get_metric_color,
    get_plot_path,
)
from model_comparison import clean_latex_text, parse_latex_table

BAR_HEIGHT = 0.45


def create_comparison_chart(df, metric_columns=None, output_path=None, color_mode='model'):
    """
    Create a simple bar chart showing all models and their metric values.
    
    Args:
        df: DataFrame with model data (first column is model names)
        metric_columns: List of column names to plot (default: all numeric columns)
        output_path: Where to save the figure (optional)
        color_mode: 'model' (default) for model-based colors, 'metric' for column-based colors
    """
    setup_plot_style(use_latex=True)
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    
    # Get model names and metrics
    model_col = df.columns[0]
    models = df[model_col].tolist()
    
    if metric_columns is None:
        # Use all numeric columns
        metric_columns = [col for col in df.columns[1:] if pd.api.types.is_numeric_dtype(df[col])]
    
    n_metrics = len(metric_columns)
    n_models = len(models)
    
    # Reverse the order of models for bottom-to-top display
    models = list(reversed(models))
    
    # Extract clean model names
    model_names = [extract_model_name(m) for m in models]
    
    # Calculate figure dimensions
    fig_width = 5.50  # Standard text width
    fig_height = 1 + n_models * 0.15  # Dynamic height based on number of models
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height), sharey=True)
    
    # Ensure axes is always an array
    if n_metrics == 1:
        axes = [axes]
    
    # Calculate global min/max for consistent axis scaling
    all_values = df[metric_columns].values.flatten()
    all_values = all_values[~np.isnan(all_values)]  # Remove NaN values
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    # Add some padding (30% on each side)
    value_range = global_max - global_min
    padding = value_range * 0.2
    xlim_min = -0.12#global_min - padding
    xlim_max = 0.12#global_max + padding
    
    # Plot each metric
    for col_idx, metric in enumerate(metric_columns):
        ax = axes[col_idx]
        
        # Get values for this metric (in reversed order)
        values = []
        colors = []
        for model in models:
            orig_idx = df[model_col].tolist().index(model)
            value = df.iloc[orig_idx][metric]
            values.append(value)
            
            # Choose color based on color mode
            if color_mode == 'metric':
                color = get_metric_color(metric)
            else:
                model_name = extract_model_name(model)
                color = get_model_color(model_name)
            
            colors.append(color)
        
        # Create horizontal bars
        y_pos = np.arange(len(values))
        bars = ax.barh(y_pos, values, color=colors, height=BAR_HEIGHT, zorder = 5)
        
        # Add value labels
        label_offset = (xlim_max - xlim_min) * 0.02  # 2% of range
        for i, (bar, value) in enumerate(zip(bars, values)):
            if np.isnan(value):
                continue
            
            # Format label with + for positive values, $-$ (math mode) for negative
            if value > 0:
                label_text = f'+{value:.3f}'
            elif value < 0:
                label_text = f'$-${abs(value):.3f}'
            else:
                label_text = f'{value:.3f}'
            
            # Position label to the right of the bar
            ax.text((value if value > 0 else 0) + label_offset, bar.get_y() + bar.get_height()/2, 
                   label_text, ha='left', va='center', 
                   fontweight='bold', color='black', fontsize=8)
        
        # Configure axis
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(-0.4, len(values) - 0.4)
        ax.set_yticks(y_pos)
        
        # Format x-axis tick labels with +/- formatting
        import matplotlib.ticker as mticker
        def format_tick(x, pos):
            if abs(x) < 1e-12:
                return ''
            elif x > 0:
                return f'+{x:.1f}'
            else:
                return f'$-${abs(x):.1f}'
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_tick))
        
        ax.tick_params(axis='x', color='0.85', labelcolor='0')
        ax.tick_params(axis='y', color='0.85', labelcolor='0')
        
        # Only show y-axis labels on the first subplot
        if col_idx == 0:
            ax.set_yticklabels(model_names)
            
            # Check if any model has an icon - if so, add padding for all
            has_any_icon = any(get_model_imagebox(name) is not None for name in model_names)
            if has_any_icon:
                ax.tick_params(axis='y', pad=17)  # Add padding for icons
            
            # Add model icons to the right of each label
            for i, model_name in enumerate(model_names):
                imagebox = get_model_imagebox(model_name)
                if imagebox is not None:
                    ab = AnnotationBbox(
                        imagebox,
                        xy=(0, y_pos[i]),
                        xycoords=('axes fraction', 'data'),
                        xybox=(-12, 0),
                        boxcoords="offset points",
                        frameon=False,
                        box_alignment=(0.5, 0.5),
                        zorder=10
                    )
                    ax.add_artist(ab)
        
        # Add vertical line at 0.0
        ax.axvline(0, color='0.85', linewidth=0.8, linestyle='-', zorder=10)
        
        # Add light grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', zorder=0)
        
        ax.set_xlabel(metric)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    else:
        default_path = get_plot_path("model_comparison_single.pdf")
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
    parser.add_argument('--color-mode', type=str, default='model', choices=['model', 'metric'],
                       help='Color mode: "model" for model-based colors, "metric" for column-based colors (default: model)')
    parser.add_argument('--output', '-o', help='Output file path (default: plots/model_comparison_single.pdf)')
    
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
        color_mode=args.color_mode
    )


if __name__ == '__main__':
    main()
