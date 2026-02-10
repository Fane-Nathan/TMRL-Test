# Ablation Study: Architecture Comparison
# ========================================
# This script runs training with a specific architecture variant.
#
# Usage:
#   $env:TMRL_RUN_NAME="everything"; python -m tmrl --trainer
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
# ### Step 1: Run the "Everything" Architecture (Current)
# ```powershell
# $env:TMRL_RUN_NAME="everything"
# python -m tmrl --trainer
# ```
# (Let it run for 2+ hours, then Ctrl+C)
#
# ### Step 2: Switch to GRU-Only (Pre-Thesis Baseline)
# In config_objects.py line 51, change:
#   TRAIN_MODEL = ContextualDroQHybridActorCritic
# to:
#   TRAIN_MODEL = GRUOnlyDroQHybridActorCritic
#
# Delete old weights:
# ```powershell
# Remove-Item ~/TmrlData/weights/* -Force
# Remove-Item ~/TmrlData/checkpoints/* -Force
# $env:TMRL_RUN_NAME="gru_only"
# python -m tmrl --trainer
# ```
#
# ### Step 3: Switch to Vanilla Baseline (No Context)
# In config_objects.py line 51, change to:
#   TRAIN_MODEL = VanillaDroQHybridActorCritic
#
# Delete old weights and run:
# ```powershell
# Remove-Item ~/TmrlData/weights/* -Force
# Remove-Item ~/TmrlData/checkpoints/* -Force
# $env:TMRL_RUN_NAME="baseline"
# python -m tmrl --trainer
# ```
#
# ### Step 4: Generate Comparison Charts
# ```powershell
# python ablation/plot_comparison.py
# ```
#
# Charts are saved to ablation/figures/
