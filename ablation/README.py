# Ablation Study: Architecture Comparison
# ========================================
# This script runs training with a specific architecture variant.
#
# Usage:
#   $env:TMRL_RUN_NAME="contextual_film"; python -m tmrl --trainer
#   $env:TMRL_RUN_NAME="gru_only"; python -m tmrl --trainer
#   $env:TMRL_RUN_NAME="baseline"; python -m tmrl --trainer
#
# After all runs, plot:
#   python ablation/plot_comparison.py

# Ablation Study Framework
# ========================
#
# This folder contains tools for comparing architecture variants.
#
# ## Quick Start
#
# ### Step 1: Run the "Contextual FiLM" Architecture (Current)
# ```powershell
# $env:TMRL_RUN_NAME="contextual_film"
# python -m tmrl --trainer
# ```
# (Let it run for 2+ hours, then Ctrl+C)
#
# ### Step 2: Run GRU-Only (Pre-Thesis Baseline)
# ```powershell
# python ablation/run_experiment.py gru_only
# ```
#
# ### Step 3: Run Reactive Baseline (No Context, DroQ loop unchanged)
# ```powershell
# python ablation/run_experiment.py baseline
# ```
#
# ### Step 4: Generate Comparison Charts
# ```powershell
# python ablation/plot_comparison.py
# ```
#
# Charts are saved to ablation/figures/
