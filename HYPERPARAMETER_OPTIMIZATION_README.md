# Hyperparameter Optimization for Stable SAC Training

This module provides a comprehensive hyperparameter optimization system for SAC (Soft Actor-Critic) training, specifically designed to address **training stability issues** like those observed in `Run_Test_13_Hybrid_Sensor`.

## Problem Analysis

Your test run showed these issues:
- **Policy collapse**: Deterministic returns stuck at baseline (-0.405)
- **Entropy coefficient crash**: 0.198 → 0.120 (too deterministic)
- **Loss oscillations**: Actor loss swinging ±3.0+
- **Early performance degradation**: Best return 8.93 → final -0.33

## Solution

Bayesian optimization using **Optuna** with **WandB** integration, focusing on:
1. **Loss smoothness**: Minimize variance in actor/critic losses
2. **Return consistency**: Stable performance across epochs
3. **Entropy stability**: Maintain healthy exploration (0.08-0.25 range)
4. **Performance trend**: Reward improving trajectories

## Quick Start

### 1. Install Dependencies

```bash
pip install optuna wandb plotly kaleido
```

### 2. Run Hyperparameter Sweep

```bash
# Test mode (5 trials, 3 epochs each)
python hyperparameter_sweep_optuna.py --test-mode

# Full sweep (100 trials, 10 epochs each)
python hyperparameter_sweep_optuna.py --n-trials 100

# Resume interrupted sweep
python hyperparameter_sweep_optuna.py --resume ~/TmrlData/hyperparameter_results/study_database.db
```

### 3. Visualize Results

```bash
python visualize_sweep.py
```

Opens interactive HTML dashboards in `~/TmrlData/hyperparameter_results/`:
- `dashboard_optimization_history.html` - Score over trials
- `dashboard_parameter_importance.html` - Which params matter most
- `dashboard_parallel_coordinates.html` - Param combinations
- `dashboard_score_distribution.html` - Score distribution
- `dashboard_top_trials.html` - Best configurations
- `summary_report.txt` - Text summary

### 4. Validate Best Config

```bash
# Validate with 50 epochs
python validate_best_config.py --epochs 50

# Compare with baseline
python validate_best_config.py --compare-baseline --epochs 50
```

### 5. Apply Best Config

```bash
# Preview changes
python apply_best_config.py --dry-run

# Apply best config
python apply_best_config.py

# Restore previous config
python apply_best_config.py --restore-backup
```

## Files

| File | Purpose |
|------|---------|
| `stability_metrics.py` | Stability scoring and early stopping logic |
| `hyperparameter_sweep_optuna.py` | Main optimization script |
| `visualize_sweep.py` | Create visualization dashboards |
| `validate_best_config.py` | Long-run validation of best config |
| `apply_best_config.py` | Apply best config to production |

## Hyperparameter Search Space

```python
# Learning rates (current were too conservative at 5e-5)
lr_actor:     1e-5 to 1e-3 (log scale)
lr_critic:    1e-5 to 1e-3 (log scale)
lr_entropy:   1e-5 to 1e-2 (log scale)

# Entropy control (prevent collapse)
alpha:          0.05 to 0.5
target_entropy: -3.0 to -0.5
alpha_floor:    0.01 to 0.2  # Critical for preventing collapse!

# Stability parameters
gamma:    0.95 to 0.999
polyak:   0.990 to 0.999
batch_size: [128, 256, 512]

# Regularization (was missing!)
l2_actor:   1e-6 to 1e-3
l2_critic:  1e-6 to 1e-3

# Optimizer settings
betas: [(0.9, 0.999), (0.95, 0.999), (0.85, 0.995)]

# REDQ ensemble
redq_n: [2, 4, 8]
redq_m: [1, 2]
```

## Stability Scoring

The composite stability score (0-1, higher is better):

```
score = 0.25 * loss_smoothness      # Low variance in losses
      + 0.25 * return_consistency   # Low variance in returns
      + 0.20 * entropy_health       # Entropy in healthy range (0.08-0.25)
      + 0.15 * trend_score          # Positive performance drift
      + 0.15 * performance          # Absolute return value
```

## Early Stopping

Trials are stopped early if:
- Stability score < 0.15 for 3+ epochs
- Entropy coefficient < 0.05 (policy collapse)
- Returns < -0.35 consistently
- Loss divergence detected

## Output Structure

```
~/TmrlData/
├── hyperparameter_results/
│   ├── study_database.db          # Optuna SQLite database
│   ├── optimization_results.json  # All trial results
│   ├── best_hyperparameters.json  # Best config only
│   ├── trial_0_config.json        # Per-trial configs
│   ├── trial_1_config.json
│   ├── ...
│   ├── dashboard_*.html           # Visualization files
│   └── summary_report.txt
├── validation_results/
│   ├── validation_best_*.json     # Validation run results
│   └── validation_comparison.json
└── config_backups/
    └── config_backup_*.json       # Config backups
```

## Parallel Execution

For multi-GPU setups:

```bash
# Terminal 1
python hyperparameter_sweep_optuna.py --n-trials 100 --n-jobs 4
```

This runs 4 trials in parallel.

## Tips

1. **Start with test mode** to verify everything works
2. **Monitor WandB dashboard** during sweeps
3. **Validate top 3 configs** before final deployment
4. **Keep config backups** - apply_best_config.py creates them automatically
5. **Check entropy coefficient** - if it drops below 0.1, policy may be collapsing

## Troubleshooting

**"No module named 'optuna'"**
```bash
pip install optuna
```

**WandB connection issues**
- Check your WANDB_KEY in config.json
- Verify internet connection

**Trials failing early**
- Check GPU memory (reduce batch_size)
- Verify environment is working
- Check server/worker connectivity

**Results not improving**
- Increase n_trials (Bayesian optimization needs ~30+ trials)
- Widen search space
- Check if stability metrics are being calculated correctly
