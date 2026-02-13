"""
Apply Best Hyperparameters to Production Config

Updates the production config.json with the best found hyperparameters.
Creates a backup of the current config before modification.

Usage:
    # Apply best config
    python apply_best_config.py

    # Apply specific trial
    python apply_best_config.py --trial-number 42

    # Dry run (preview changes without applying)
    python apply_best_config.py --dry-run

    # Restore from backup
    python apply_best_config.py --restore-backup
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = Path.home() / "TmrlData" / "hyperparameter_results"
CONFIG_DIR = Path.home() / "TmrlData" / "config"
BACKUP_DIR = Path.home() / "TmrlData" / "config_backups"


def load_best_params(results_path: Optional[Path] = None) -> Dict:
    """Load best hyperparameters from optimization results."""
    if results_path is None:
        results_path = RESULTS_DIR / "best_hyperparameters.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Best hyperparameters not found at {results_path}\n"
            "Run hyperparameter_sweep_optuna.py first."
        )

    with open(results_path, "r") as f:
        data = json.load(f)

    return data["best_params"], data.get("best_stability_score", None)


def load_trial_params(trial_number: int, results_dir: Optional[Path] = None) -> Dict:
    """Load hyperparameters from a specific trial."""
    if results_dir is None:
        results_dir = RESULTS_DIR

    results_path = results_dir / "optimization_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path, "r") as f:
        data = json.load(f)

    trial = next((t for t in data["all_trials"] if t["number"] == trial_number), None)

    if trial is None:
        raise ValueError(f"Trial {trial_number} not found")

    return trial["params"], trial["value"]


def backup_config(config_path: Path) -> Path:
    """Create backup of current config."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"config_backup_{timestamp}.json"

    shutil.copy(config_path, backup_path)

    print(f"Backup created: {backup_path}")

    return backup_path


def list_backups() -> list:
    """List available config backups."""
    if not BACKUP_DIR.exists():
        return []

    backups = sorted(BACKUP_DIR.glob("config_backup_*.json"), reverse=True)
    return backups


def restore_backup(backup_path: Optional[Path] = None) -> bool:
    """Restore config from backup."""
    config_path = CONFIG_DIR / "config.json"

    if backup_path is None:
        backups = list_backups()
        if not backups:
            print("No backups found!")
            return False
        backup_path = backups[0]
        print(f"Restoring from most recent backup: {backup_path}")

    if not backup_path.exists():
        print(f"Backup not found: {backup_path}")
        return False

    shutil.copy(backup_path, config_path)
    print(f"Config restored from: {backup_path}")

    return True


def apply_hyperparams_to_config(
    params: Dict, config: Dict, stability_score: Optional[float] = None
) -> Dict:
    """Apply hyperparameters to config."""
    new_config = config.copy()
    new_config["ALG"] = config["ALG"].copy()

    param_mapping = {
        "lr_actor": "LR_ACTOR",
        "lr_critic": "LR_CRITIC",
        "lr_entropy": "LR_ENTROPY",
        "alpha": "ALPHA",
        "target_entropy": "TARGET_ENTROPY",
        "alpha_floor": "ALPHA_FLOOR",
        "gamma": "GAMMA",
        "polyak": "POLYAK",
        "batch_size": "BATCH_SIZE",
        "l2_actor": "L2_ACTOR",
        "l2_critic": "L2_CRITIC",
        "redq_n": "REDQ_N",
        "redq_m": "REDQ_M",
    }

    for param_key, config_key in param_mapping.items():
        if param_key in params:
            new_config["ALG"][config_key] = params[param_key]

    if "betas_actor" in params:
        new_config["ALG"]["BETAS_ACTOR"] = list(params["betas_actor"])
    if "betas_critic" in params:
        new_config["ALG"]["BETAS_CRITIC"] = list(params["betas_critic"])

    new_config["RESET_TRAINING"] = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_config["RUN_NAME"] = f"Optimized_{timestamp}"

    if stability_score is not None:
        if "OPTIMIZATION_METADATA" not in new_config:
            new_config["OPTIMIZATION_METADATA"] = {}
        new_config["OPTIMIZATION_METADATA"]["stability_score"] = stability_score
        new_config["OPTIMIZATION_METADATA"]["applied_at"] = timestamp

    return new_config


def print_config_diff(old_config: Dict, new_config: Dict):
    """Print differences between old and new config."""
    print("\n" + "=" * 70)
    print("CONFIGURATION CHANGES")
    print("=" * 70)

    alg_old = old_config.get("ALG", {})
    alg_new = new_config.get("ALG", {})

    all_keys = set(alg_old.keys()) | set(alg_new.keys())

    changes = []
    for key in sorted(all_keys):
        old_val = alg_old.get(key, "N/A")
        new_val = alg_new.get(key, "N/A")

        if old_val != new_val:
            changes.append((key, old_val, new_val))

    if not changes:
        print("No changes detected.")
        return

    print(f"\n{'Parameter':<25} {'Old Value':>20} {'New Value':>20}")
    print("-" * 65)

    for key, old_val, new_val in changes:
        if isinstance(old_val, float):
            old_str = f"{old_val:.6e}"
        elif isinstance(old_val, list):
            old_str = str(old_val)
        else:
            old_str = str(old_val)

        if isinstance(new_val, float):
            new_str = f"{new_val:.6e}"
        elif isinstance(new_val, list):
            new_str = str(new_val)
        else:
            new_str = str(new_val)

        print(f"{key:<25} {old_str:>20} {new_str:>20}")

    print("-" * 65)
    print(f"Total changes: {len(changes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply best hyperparameters to production config"
    )
    parser.add_argument(
        "--trial-number",
        type=int,
        default=None,
        help="Apply specific trial number (default: best trial)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing optimization results",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    parser.add_argument(
        "--restore-backup",
        action="store_true",
        help="Restore config from most recent backup",
    )
    parser.add_argument(
        "--backup-path", type=str, default=None, help="Specific backup to restore from"
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Custom run name for new config"
    )
    parser.add_argument(
        "--no-reset", action="store_true", help="Do not set RESET_TRAINING flag"
    )

    args = parser.parse_args()

    if args.restore_backup:
        backup_path = Path(args.backup_path) if args.backup_path else None
        success = restore_backup(backup_path)
        return 0 if success else 1

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    config_path = CONFIG_DIR / "config.json"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    with open(config_path, "r") as f:
        current_config = json.load(f)

    print("Current config loaded")
    print(f"  RUN_NAME: {current_config.get('RUN_NAME', 'N/A')}")
    print(f"  MAX_EPOCHS: {current_config.get('MAX_EPOCHS', 'N/A')}")

    try:
        if args.trial_number is not None:
            print(f"\nLoading hyperparameters from trial #{args.trial_number}")
            params, score = load_trial_params(args.trial_number, results_dir)
        else:
            print("\nLoading best hyperparameters")
            params, score = load_best_params(results_dir / "best_hyperparameters.json")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(
            "Run hyperparameter_sweep_optuna.py first to find optimal hyperparameters."
        )
        return 1

    print(f"  Stability score: {score:.4f}" if score else "  Stability score: N/A")

    new_config = apply_hyperparams_to_config(params, current_config, score)

    if args.run_name:
        new_config["RUN_NAME"] = args.run_name

    if args.no_reset:
        new_config["RESET_TRAINING"] = False

    print_config_diff(current_config, new_config)

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - No changes applied")
        print("=" * 70)
        print("\nTo apply these changes, run without --dry-run")
        return 0

    print("\n" + "=" * 70)
    print("APPLYING CHANGES")
    print("=" * 70)

    backup_path = backup_config(config_path)

    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2)

    print(f"\nNew config saved to: {config_path}")
    print(f"Backup saved to: {backup_path}")

    print("\n" + "=" * 70)
    print("CONFIG UPDATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nNew RUN_NAME: {new_config['RUN_NAME']}")
    print(f"RESET_TRAINING: {new_config['RESET_TRAINING']}")
    print("\nYour next training run will use the optimized hyperparameters.")
    print("To restore the previous config, run:")
    print(f"  python apply_best_config.py --restore-backup")

    return 0


if __name__ == "__main__":
    sys.exit(main())
