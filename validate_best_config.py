"""
Validate Best Hyperparameter Configuration

Runs longer training with the best found hyperparameters to verify stability.
Compares results against baseline configuration.

Usage:
    # Validate best config with 50 epochs
    python validate_best_config.py --epochs 50

    # Validate specific trial
    python validate_best_config.py --trial-number 42 --epochs 50

    # Compare with baseline
    python validate_best_config.py --compare-baseline --epochs 50
"""

import os
import sys
import json
import argparse
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stability_metrics import StabilityTracker
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.networking import Trainer


RESULTS_DIR = Path.home() / "TmrlData" / "hyperparameter_results"
VALIDATION_DIR = Path.home() / "TmrlData" / "validation_results"


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

    return data["best_params"]


def load_trial_config(trial_number: int, results_dir: Optional[Path] = None) -> Dict:
    """Load configuration from a specific trial."""
    if results_dir is None:
        results_dir = RESULTS_DIR

    config_path = results_dir / f"trial_{trial_number}_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Trial {trial_number} config not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def create_validation_config(
    params: Dict, base_config: Dict, run_name: str, epochs: int
) -> Dict:
    """Create validation configuration from hyperparameters."""

    def _normalize_betas(value):
        if isinstance(value, str):
            return [float(x.strip()) for x in value.split(",")]
        if isinstance(value, (list, tuple)):
            return [float(value[0]), float(value[1])]
        raise ValueError(f"Unsupported beta format: {value!r}")

    config = base_config.copy()
    config["ALG"] = base_config["ALG"].copy()

    config["RUN_NAME"] = run_name
    config["RESET_TRAINING"] = True
    config["MAX_EPOCHS"] = epochs

    config["ALG"]["LR_ACTOR"] = params["lr_actor"]
    config["ALG"]["LR_CRITIC"] = params["lr_critic"]
    config["ALG"]["LR_ENTROPY"] = params["lr_entropy"]
    config["ALG"]["ALPHA"] = params["alpha"]
    config["ALG"]["TARGET_ENTROPY"] = params["target_entropy"]
    config["ALG"]["ALPHA_FLOOR"] = params["alpha_floor"]
    config["ALG"]["GAMMA"] = params["gamma"]
    config["ALG"]["POLYAK"] = params["polyak"]
    config["BATCH_SIZE"] = params["batch_size"]
    if "max_training_steps_per_env_step" in params:
        config["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"] = float(
            params["max_training_steps_per_env_step"]
        )
    config["ALG"]["L2_ACTOR"] = params["l2_actor"]
    config["ALG"]["L2_CRITIC"] = params["l2_critic"]
    if "betas_actor" in params:
        config["ALG"]["BETAS_ACTOR"] = _normalize_betas(params["betas_actor"])
    else:
        config["ALG"]["BETAS_ACTOR"] = [
            float(params["beta1_actor"]),
            float(params["beta2_actor"]),
        ]
    if "betas_critic" in params:
        config["ALG"]["BETAS_CRITIC"] = _normalize_betas(params["betas_critic"])
    else:
        config["ALG"]["BETAS_CRITIC"] = [
            float(params["beta1_critic"]),
            float(params["beta2_critic"]),
        ]
    config["ALG"]["REDQ_N"] = params["redq_n"]
    config["ALG"]["REDQ_M"] = params["redq_m"]

    return config


def run_validation(config: Dict, run_name: str, output_dir: Path) -> Dict:
    """
    Run validation training with given configuration.

    Returns:
        Dict: Validation metrics and stability scores
    """
    import wandb

    temp_config_path = Path.home() / "TmrlData" / "config" / "config.json"
    backup_path = output_dir / "config_backup.json"

    if temp_config_path.exists():
        shutil.copy(temp_config_path, backup_path)

    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    import importlib

    importlib.reload(cfg)

    validation_results = {
        "run_name": run_name,
        "config": config,
        "start_time": datetime.now().isoformat(),
        "epochs_completed": 0,
        "metrics_history": [],
        "stability_scores": [],
    }

    stability_tracker = StabilityTracker(window_size=10)

    wandb_run = None
    try:
        wandb_run = wandb.init(
            project="tmrl-validation",
            name=run_name,
            config=config,
            reinit=True,
            allow_val_change=True,
        )

        trainer = Trainer(
            training_cls=cfg_obj.TRAINER,
            server_ip=cfg.SERVER_IP_FOR_TRAINER,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            security=cfg.SECURITY,
        )

        max_epochs = config.get("MAX_EPOCHS", 50)

        print(f"\nStarting validation: {run_name}")
        print(f"Max epochs: {max_epochs}")
        print("=" * 60)

        from tmrl.networking import iterate_epochs, TrainerInterface

        interface = TrainerInterface(
            server_ip=cfg.SERVER_IP_FOR_TRAINER,
            server_port=cfg.PORT,
            password=cfg.PASSWORD,
            local_com_port=cfg.LOCAL_PORT_TRAINER,
            header_size=cfg.HEADER_SIZE,
            max_buf_len=cfg.BUFFER_SIZE,
            security=cfg.SECURITY,
            keys_dir=cfg.CREDENTIALS_DIRECTORY,
            hostname=cfg.HOSTNAME,
            model_path=cfg.MODEL_PATH_TRAINER,
        )

        checkpoint_path = str(output_dir / f"{run_name}_checkpoint.pkl")

        for stats_list in iterate_epochs(
            run_cls=cfg_obj.TRAINER,
            interface=interface,
            checkpoint_path=checkpoint_path,
        ):
            if stats_list:
                last_stats = stats_list[-1]

                epoch_metrics = {
                    "loss_actor": float(last_stats.get("loss_actor", 0.0)),
                    "loss_critic": float(last_stats.get("loss_critic", 0.0)),
                    "return_test": float(last_stats.get("return_test", 0.0)),
                    "return_train": float(last_stats.get("return_train", 0.0)),
                    "entropy_coef": float(last_stats.get("entropy_coef", 0.2)),
                    "epoch": validation_results["epochs_completed"],
                }

                validation_results["metrics_history"].append(epoch_metrics)

                stability_tracker.update(
                    loss_actor=epoch_metrics["loss_actor"],
                    loss_critic=epoch_metrics["loss_critic"],
                    return_test=epoch_metrics["return_test"],
                    return_train=epoch_metrics["return_train"],
                    entropy_coef=epoch_metrics["entropy_coef"],
                    epoch=epoch_metrics["epoch"],
                )

                if (
                    validation_results["epochs_completed"]
                    >= stability_tracker.window_size
                ):
                    stability_score = stability_tracker.compute_stability_score()
                    diagnostics = stability_tracker.get_diagnostics()

                    validation_results["stability_scores"].append(stability_score)

                    wandb.log(
                        {
                            "epoch": validation_results["epochs_completed"],
                            "stability_score": stability_score,
                            **diagnostics,
                            **epoch_metrics,
                        }
                    )

                    print(
                        f"Epoch {validation_results['epochs_completed']:3d}: "
                        f"score={stability_score:.4f}, "
                        f"return={epoch_metrics['return_test']:.4f}, "
                        f"entropy={epoch_metrics['entropy_coef']:.4f}"
                    )

                validation_results["epochs_completed"] += 1

        validation_results["end_time"] = datetime.now().isoformat()
        validation_results["final_stability_score"] = (
            stability_tracker.compute_stability_score()
        )
        validation_results["final_diagnostics"] = stability_tracker.get_diagnostics()

        print("=" * 60)
        print(f"Validation completed!")
        print(
            f"Final stability score: {validation_results['final_stability_score']:.4f}"
        )

    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback

        traceback.print_exc()
        validation_results["error"] = str(e)

    finally:
        if backup_path.exists():
            shutil.copy(backup_path, temp_config_path)

        if wandb_run is not None:
            wandb.finish()

    return validation_results


def compare_validations(results_list: list, output_dir: Path):
    """Compare multiple validation runs."""
    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON")
    print("=" * 70)

    print(f"\n{'Run Name':<40} {'Stability':>12} {'Final Return':>15} {'Entropy':>12}")
    print("-" * 79)

    for result in results_list:
        name = result["run_name"][:38]
        stability = result.get("final_stability_score", 0.0)

        if result["metrics_history"]:
            final_return = result["metrics_history"][-1]["return_test"]
            final_entropy = result["metrics_history"][-1]["entropy_coef"]
        else:
            final_return = 0.0
            final_entropy = 0.0

        print(
            f"{name:<40} {stability:>12.4f} {final_return:>15.4f} {final_entropy:>12.4f}"
        )

    print("-" * 79)

    comparison_path = output_dir / "validation_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "comparisons": results_list},
            f,
            indent=2,
        )

    print(f"\nComparison saved to: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate best hyperparameter configuration"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for validation (default: 50)",
    )
    parser.add_argument(
        "--trial-number",
        type=int,
        default=None,
        help="Validate specific trial number (default: best trial)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing optimization results",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also validate baseline config for comparison",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (default: auto-generated)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    base_config_path = Path.home() / "TmrlData" / "config" / "config.json"
    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    results_list = []

    if args.trial_number is not None:
        print(f"Validating trial #{args.trial_number}")
        trial_config = load_trial_config(args.trial_number, results_dir)
        params_path = results_dir / "optimization_results.json"
        with open(params_path, "r") as f:
            opt_results = json.load(f)
        trial_data = [
            t for t in opt_results["all_trials"] if t["number"] == args.trial_number
        ][0]
        params = trial_data["params"]
    else:
        print("Validating best configuration")
        params = load_best_params(results_dir / "best_hyperparameters.json")

    run_name = (
        args.run_name or f"validation_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    validation_config = create_validation_config(
        params, base_config, run_name, args.epochs
    )

    validation_results = run_validation(
        config=validation_config, run_name=run_name, output_dir=VALIDATION_DIR
    )

    results_list.append(validation_results)

    validation_output_path = VALIDATION_DIR / f"{run_name}_results.json"
    with open(validation_output_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"\nResults saved to: {validation_output_path}")

    if args.compare_baseline:
        print("\n" + "=" * 60)
        print("Validating BASELINE configuration for comparison")
        print("=" * 60)

        baseline_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        baseline_config = base_config.copy()
        baseline_config["RUN_NAME"] = baseline_name
        baseline_config["RESET_TRAINING"] = True
        baseline_config["MAX_EPOCHS"] = args.epochs

        baseline_results = run_validation(
            config=baseline_config, run_name=baseline_name, output_dir=VALIDATION_DIR
        )

        results_list.append(baseline_results)

        baseline_output_path = VALIDATION_DIR / f"{baseline_name}_results.json"
        with open(baseline_output_path, "w") as f:
            json.dump(baseline_results, f, indent=2)

    if len(results_list) > 1:
        compare_validations(results_list, VALIDATION_DIR)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
