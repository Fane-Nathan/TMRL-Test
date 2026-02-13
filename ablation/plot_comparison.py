"""
Ablation Study: Comparison Plotting Script
============================================
Generates thesis-quality charts comparing different architecture variants.

Usage:
    python ablation/plot_comparison.py

Reads CSV files from ~/TmrlData/ablation/ directory.
Each CSV corresponds to one training run (set via TMRL_RUN_NAME env var).
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# --- Configuration ---
ABLATION_DIR = Path.home() / "TmrlData" / "ablation"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colors for each variant
COLORS = {
    'contextual_film': '#2196F3',   # Blue
    'gru_only': '#FF9800',     # Orange
    'baseline': '#9E9E9E',     # Gray
}

LABELS = {
    'contextual_film': 'Contextual FiLM (Transformer+GRU+FiLM)',
    'gru_only': 'GRU-Only Context (Pre-Thesis)',
    'baseline': 'Reactive DroQ Baseline (No Context)',
}


def load_runs():
    """Load all CSV files from the ablation directory."""
    runs = {}
    if not ABLATION_DIR.exists():
        print(f"Error: Ablation directory not found: {ABLATION_DIR}")
        print("Run training with TMRL_RUN_NAME env var to generate CSVs.")
        sys.exit(1)

    for csv_path in sorted(ABLATION_DIR.glob("*.csv")):
        name = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
            runs[name] = df
            print(f"Loaded: {name} ({len(df)} rounds)")
        except Exception as e:
            print(f"Warning: Failed to load {csv_path}: {e}")

    if not runs:
        print("No CSV files found. Run experiments first.")
        sys.exit(1)

    return runs


def smooth(values, window=3):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def plot_return_vs_time(runs):
    """Plot Return vs Wall Clock Time (the key thesis chart)."""
    fig, ax = plt.subplots()

    for name, df in runs.items():
        if 'wall_clock_seconds' not in df.columns or 'return_train' not in df.columns:
            print(f"Skipping {name}: missing columns")
            continue

        time_minutes = df['wall_clock_seconds'].values / 60
        returns = df['return_train'].values
        smoothed = smooth(returns, window=5)

        color = COLORS.get(name, '#333333')
        label = LABELS.get(name, name)

        ax.plot(time_minutes, smoothed, color=color, linewidth=2, label=label)
        ax.fill_between(time_minutes, returns, alpha=0.15, color=color)

    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Average Return')
    ax.set_title('Learning Speed: Return vs Training Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    path = OUTPUT_DIR / "return_vs_time.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_episode_length(runs):
    """Plot Episode Length vs Training Time."""
    fig, ax = plt.subplots()

    for name, df in runs.items():
        if 'wall_clock_seconds' not in df.columns or 'episode_length_train' not in df.columns:
            continue

        time_minutes = df['wall_clock_seconds'].values / 60
        ep_len = df['episode_length_train'].values
        smoothed = smooth(ep_len, window=5)

        color = COLORS.get(name, '#333333')
        label = LABELS.get(name, name)

        ax.plot(time_minutes, smoothed, color=color, linewidth=2, label=label)
        ax.fill_between(time_minutes, ep_len, alpha=0.15, color=color)

    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Driving Proficiency: Episode Length vs Training Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "episode_length_vs_time.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_critic_loss(runs):
    """Plot Critic Loss vs Training Time (stability metric)."""
    fig, ax = plt.subplots()

    for name, df in runs.items():
        if 'wall_clock_seconds' not in df.columns or 'loss_critic' not in df.columns:
            continue

        time_minutes = df['wall_clock_seconds'].values / 60
        loss = df['loss_critic'].values
        smoothed = smooth(loss, window=5)

        color = COLORS.get(name, '#333333')
        label = LABELS.get(name, name)

        ax.plot(time_minutes, smoothed, color=color, linewidth=2, label=label)

    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Training Stability: Critic Loss vs Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    path = OUTPUT_DIR / "critic_loss_vs_time.png"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_summary_table(runs):
    """Generate a summary comparison table."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<35} {'Peak Return':>12} {'Max EpLen':>10} {'Time to R>5':>12}")
    print("-" * 70)

    for name, df in runs.items():
        label = LABELS.get(name, name)
        peak_return = df['return_train'].max() if 'return_train' in df.columns else 0
        max_ep_len = df['episode_length_train'].max() if 'episode_length_train' in df.columns else 0

        # Time to reach return > 5
        above_5 = df[df['return_train'] > 5] if 'return_train' in df.columns else pd.DataFrame()
        if len(above_5) > 0:
            time_to_5 = above_5.iloc[0]['wall_clock_seconds'] / 60
            time_str = f"{time_to_5:.1f} min"
        else:
            time_str = "Never"

        print(f"{label:<35} {peak_return:>12.2f} {max_ep_len:>10.1f} {time_str:>12}")

    print("=" * 70)


def main():
    print("=" * 50)
    print("Ablation Study: Architecture Comparison")
    print("=" * 50)

    runs = load_runs()

    plot_return_vs_time(runs)
    plot_episode_length(runs)
    plot_critic_loss(runs)
    plot_summary_table(runs)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Use these in your thesis!")


if __name__ == "__main__":
    main()
