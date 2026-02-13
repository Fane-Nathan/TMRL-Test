"""
Ablation Study Runner
=====================
Run this script to execute one architecture variant.
Set the variant via command-line argument.

Usage:
  python ablation/run_experiment.py contextual_film
  python ablation/run_experiment.py gru_only
  python ablation/run_experiment.py baseline

Each run:
  1. Sets TMRL_CONTEXT_MODE and TMRL_RUN_NAME env vars
  2. Deletes old artifacts for this mode only (fresh start)
  3. Launches the trainer

Results are saved to ~/TmrlData/ablation/{variant_name}.csv
After all runs, plot with: python ablation/plot_comparison.py
"""
import os
import sys
import subprocess
from pathlib import Path

VALID_MODES = ["contextual_film", "gru_only", "baseline"]
TMRL_DATA = Path.home() / "TmrlData"


def clean_mode_artifacts(mode: str):
    """Delete only mode-specific artifacts for a fresh run of that mode."""
    weights_dir = TMRL_DATA / "weights"
    checkpoints_dir = TMRL_DATA / "checkpoints"
    ablation_dir = TMRL_DATA / "ablation"
    targets = [
        weights_dir / f"{mode}.tmod",
        weights_dir / f"{mode}_t.tmod",
        checkpoints_dir / f"{mode}_t.tcpt",
        checkpoints_dir / f"{mode}.tmod",
        checkpoints_dir / f"{mode}_t.tmod",
        ablation_dir / f"{mode}.csv",
    ]

    for path in targets:
        if path.exists():
            path.unlink()
            print(f"  Deleted: {path}")

    if weights_dir.exists():
        for history_file in weights_dir.glob(f"{mode}_*.tmod"):
            if history_file.exists():
                history_file.unlink()
                print(f"  Deleted: {history_file}")


def run_experiment(mode: str):
    print("=" * 60)
    print(f"  ABLATION EXPERIMENT: {mode.upper()}")
    print("=" * 60)

    # 1. Clean old mode-specific artifacts
    print("\n[1/3] Cleaning mode-specific weights/checkpoints/CSV...")
    clean_mode_artifacts(mode)

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
                "contextual_film": "Full Transformer+GRU+FiLM (default, thesis architecture)",
                "gru_only": "GRU-only context, no Transformer (pre-thesis)",
                "baseline": "No context, plain MLP (reactive DroQ baseline)",
            }
            print(f"  {m:20s} - {desc[m]}")
        sys.exit(1)

    mode = sys.argv[1]
    run_experiment(mode)


if __name__ == "__main__":
    main()

