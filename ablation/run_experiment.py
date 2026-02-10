"""
Ablation Study Runner
=====================
Run this script to execute one architecture variant.
Set the variant via command-line argument.

Usage:
  python ablation/run_experiment.py everything
  python ablation/run_experiment.py gru_only
  python ablation/run_experiment.py baseline

Each run:
  1. Sets TMRL_CONTEXT_MODE and TMRL_RUN_NAME env vars
  2. Deletes old weights/checkpoints (fresh start)
  3. Launches the trainer

Results are saved to ~/TmrlData/ablation/{variant_name}.csv
After all runs, plot with: python ablation/plot_comparison.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

VALID_MODES = ["everything", "gru_only", "baseline"]
TMRL_DATA = Path.home() / "TmrlData"


def clean_weights():
    """Delete old weights and checkpoints for fresh training."""
    weights_dir = TMRL_DATA / "weights"
    checkpoints_dir = TMRL_DATA / "checkpoints"
    for d in [weights_dir, checkpoints_dir]:
        if d.exists():
            for f in d.iterdir():
                f.unlink()
                print(f"  Deleted: {f}")


def run_experiment(mode: str):
    print("=" * 60)
    print(f"  ABLATION EXPERIMENT: {mode.upper()}")
    print("=" * 60)

    # 1. Clean old weights
    print("\n[1/3] Cleaning old weights and checkpoints...")
    clean_weights()

    # 2. Set environment variables
    print(f"[2/3] Setting TMRL_CONTEXT_MODE={mode}, TMRL_RUN_NAME={mode}")
    env = os.environ.copy()
    env["TMRL_CONTEXT_MODE"] = mode
    env["TMRL_RUN_NAME"] = mode

    # 3. Launch trainer
    print("[3/3] Launching trainer (Ctrl+C to stop)...")
    print("-" * 60)

    try:
        subprocess.run(
            [sys.executable, "-m", "tmrl", "--trainer"],
            env=env,
            cwd=str(Path(__file__).parent.parent),  # project root
        )
    except KeyboardInterrupt:
        print(f"\n\nExperiment '{mode}' stopped by user.")
        print(f"Results saved to: {TMRL_DATA / 'ablation' / f'{mode}.csv'}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
        print(f"Usage: python {sys.argv[0]} <{'|'.join(VALID_MODES)}>")
        print(f"\nValid modes:")
        for m in VALID_MODES:
            desc = {
                "everything": "Full Transformer+GRU+FiLM (current best)",
                "gru_only": "GRU-only context, no Transformer (pre-thesis)",
                "baseline": "No context, plain MLP (vanilla SAC)",
            }
            print(f"  {m:15s} - {desc[m]}")
        sys.exit(1)

    mode = sys.argv[1]
    run_experiment(mode)


if __name__ == "__main__":
    main()
