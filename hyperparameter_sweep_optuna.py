"""
Hyperparameter Optimization with Optuna and WandB

Stability-focused Bayesian optimization for SAC hyperparameters.
Addresses issues from Run_Test_13_Hybrid_Sensor:
- Policy collapse (entropy too low)
- Loss oscillations
- Performance degradation

Usage:
    # Quick test (5 trials)
    python hyperparameter_sweep_optuna.py --n-trials 5 --test-mode

    # Full sweep (100 trials)
    python hyperparameter_sweep_optuna.py --n-trials 100

    # Resume interrupted sweep
    python hyperparameter_sweep_optuna.py --resume study_database.db
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
import subprocess
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stability_metrics import StabilityTracker, EarlyStoppingManager
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.networking import TrainerInterface, iterate_epochs
from tmrl.util import partial_to_dict, load, dump


STUDY_NAME = "DroQ_SAC_stability_optimization"
OUTPUT_DIR = Path.home() / "TmrlData" / "hyperparameter_results"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", cfg.WANDB_ENTITY)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", cfg.WANDB_PROJECT)

# Global list to track subprocesses for cleanup
_subprocesses: List[subprocess.Popen] = []


def _ensure_wandb_auth() -> None:
    """Ensure W&B has an API key, loading from config.json when needed."""
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if api_key:
        return

    config_path = Path.home() / "TmrlData" / "config" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg_json = json.load(f)
            config_key = str(cfg_json.get("WANDB_KEY", "")).strip()
            if config_key:
                print(f"Setting WANDB_API_KEY from {config_path}")
                os.environ["WANDB_API_KEY"] = config_key
                wandb.login(key=config_key)
                return
        except Exception as e:
            print(f"Warning: Failed to load WANDB_KEY from config: {e}")

    raise EnvironmentError(
        "WANDB_API_KEY is not set and no valid WANDB_KEY was found in "
        f"{config_path}."
    )


def _cleanup_subprocesses():
    """Clean up all spawned subprocesses on exit."""
    global _subprocesses
    for proc in _subprocesses:
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _subprocesses = []


    _subprocesses = []


def _kill_zombie_python_processes():
    """Aggressively kill any lingering python processes running tmrl."""
    if sys.platform == "win32":
        try:
            # Kill processes by window title or command line if possible
            # But simplest is to trust atexit. 
            # If we want to be aggressive:
            subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq tmrl_worker*"], 
                         capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq tmrl_server*"], 
                         capture_output=True)
        except Exception:
            pass


def start_server(tmrl_path: str, port: int) -> subprocess.Popen:
    """Start the tmrl server in a subprocess."""
    print(f"Starting server on port {port}...")
    
    # We need to pass the port via environment or command line
    # TMRL server typically reads from config, but we can override via env vars 
    # if the underlying code supports it. 
    # However, tmrl config is loaded on import. 
    # Best way is to modify the temporary config file BEFORE starting server.
    
    env = os.environ.copy()
    env["TMRL_PORT"] = str(port)
    
    # NOTE: The server process loads config.json on startup.
    # We must ensure the config.json on disk HAS THE CORRECT PORT before starting.
    # This is handled in the main loop.
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "tmrl", "--server"],
        cwd=tmrl_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32"
        else 0,
    )
    _subprocesses.append(proc)
    time.sleep(5)  # Wait for server to start
    print(f"Server started (PID: {proc.pid})")
    return proc


def start_worker(tmrl_path: str, port: int) -> subprocess.Popen:
    """Start the tmrl worker in a subprocess."""
    print(f"Starting worker connecting to port {port}...")
    
    env = os.environ.copy()
    env["TMRL_PORT"] = str(port)
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "tmrl", "--worker"],
        cwd=tmrl_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32"
        else 0,
    )
    _subprocesses.append(proc)
    time.sleep(10)  # Wait for worker to connect (longer wait)
    print(f"Worker started (PID: {proc.pid})")
    return proc


    return None, None


def create_search_space(trial: optuna.Trial) -> Dict:
    """
    Define hyperparameter search space for stability optimization.

    Based on analysis of Run_Test_13 issues:
    - Learning rates were too conservative (5e-5)
    - Missing L2 regularization
    - Entropy floor too low (0.05)

    Returns:
        Dict: Sampled hyperparameters
    """
    params = {}

    # Learning rates - wider range to find optimal
    # Current: 5e-5, likely too conservative
    params["lr_actor"] = trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True)
    params["lr_critic"] = trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True)
    params["lr_entropy"] = trial.suggest_float("lr_entropy", 1e-5, 1e-2, log=True)

    # Entropy control - prevent collapse
    params["alpha"] = trial.suggest_float("alpha", 0.05, 0.5, log=True)
    params["target_entropy"] = trial.suggest_float("target_entropy", -3.0, -0.5)
    params["alpha_floor"] = trial.suggest_float("alpha_floor", 0.01, 0.2, log=True)

    # Stability parameters
    params["gamma"] = trial.suggest_float("gamma", 0.95, 0.999)
    params["polyak"] = trial.suggest_float("polyak", 0.990, 0.999)
    params["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])

    # Regularization - crucial for stability
    params["l2_actor"] = trial.suggest_float("l2_actor", 1e-6, 1e-3, log=True)
    params["l2_critic"] = trial.suggest_float("l2_critic", 1e-6, 1e-3, log=True)

    # Optimizer betas (use strings since Optuna doesn't support tuples)
    beta_choices = ["0.9,0.999", "0.95,0.999", "0.85,0.995"]
    params["betas_actor_str"] = trial.suggest_categorical("betas_actor", beta_choices)
    params["betas_critic_str"] = trial.suggest_categorical("betas_critic", beta_choices)

    # REDQ ensemble parameters
    params["redq_n"] = trial.suggest_categorical("redq_n", [2, 4, 8])
    params["redq_m"] = trial.suggest_categorical("redq_m", [1, 2])

    return params


def apply_hyperparameters_to_config(params: Dict, base_config: Dict) -> Dict:
    """
    Apply sampled hyperparameters to base config.

    Args:
        params: Sampled hyperparameters from Optuna
        base_config: Base configuration dictionary

    Returns:
        Dict: Updated configuration
    """
    config = base_config.copy()
    config["ALG"] = base_config["ALG"].copy()

    # Map parameters to config structure
    config["ALG"]["LR_ACTOR"] = params["lr_actor"]
    config["ALG"]["LR_CRITIC"] = params["lr_critic"]
    config["ALG"]["LR_ENTROPY"] = params["lr_entropy"]
    config["ALG"]["ALPHA"] = params["alpha"]
    config["ALG"]["TARGET_ENTROPY"] = params["target_entropy"]
    config["ALG"]["ALPHA_FLOOR"] = params["alpha_floor"]
    config["ALG"]["GAMMA"] = params["gamma"]
    config["ALG"]["POLYAK"] = params["polyak"]
    config["BATCH_SIZE"] = params["batch_size"]
    config["ALG"]["L2_ACTOR"] = params["l2_actor"]
    config["ALG"]["L2_CRITIC"] = params["l2_critic"]
    # Parse betas from strings
    config["ALG"]["BETAS_ACTOR"] = [
        float(x) for x in params["betas_actor_str"].split(",")
    ]
    config["ALG"]["BETAS_CRITIC"] = [
        float(x) for x in params["betas_critic_str"].split(",")
    ]
    config["ALG"]["REDQ_N"] = params["redq_n"]
    config["ALG"]["REDQ_M"] = params["redq_m"]

    return config


def run_training_epoch_with_stability(
    trial: optuna.Trial,
    config: Dict,
    run_cls,
    interface: TrainerInterface,
    checkpoint_path: str,
    max_epochs: int = 10,
    stability_window: int = 5,
) -> Tuple[float, Dict]:
    """
    Run training with stability monitoring and Optuna pruning.

    This is a modified version of iterate_epochs that:
    1. Tracks stability metrics
    2. Reports to Optuna for pruning
    3. Supports early stopping

    Args:
        trial: Optuna trial object
        config: Training configuration
        run_cls: Training class
        interface: Trainer interface
        checkpoint_path: Path for checkpoints
        max_epochs: Maximum epochs to run
        stability_window: Window size for stability calculations

    Returns:
        Tuple[float, Dict]: (final_stability_score, final_metrics)
    """
    from tmrl.networking import load_run_instance, dump_run_instance

    stability_tracker = StabilityTracker(window_size=stability_window)
    early_stopping = EarlyStoppingManager(
        patience_epochs=3,
        stability_threshold=0.15,
        return_threshold=-0.35,
        entropy_threshold=0.05,
        min_epochs=2,
        max_epochs=max_epochs,
    )

    run_instance = None
    all_metrics = []

    try:
        if not os.path.exists(checkpoint_path):
            run_instance = run_cls()
            dump_run_instance(run_instance, checkpoint_path)
        else:
            run_instance = load_run_instance(checkpoint_path)

        epoch_metrics = {}

        while run_instance.epoch < max_epochs:
            stats_list = run_instance.run_epoch(interface=interface)

            if stats_list:
                last_stats = stats_list[-1]

                loss_actor = float(last_stats.get("loss_actor", 0.0))
                loss_critic = float(last_stats.get("loss_critic", 0.0))
                return_test = float(last_stats.get("return_test", 0.0))
                return_train = float(last_stats.get("return_train", 0.0))
                entropy_coef = float(last_stats.get("entropy_coef", 0.2))

                epoch_metrics = {
                    "loss_actor": loss_actor,
                    "loss_critic": loss_critic,
                    "return_test": return_test,
                    "return_train": return_train,
                    "entropy_coef": entropy_coef,
                    "epoch": run_instance.epoch,
                }
                all_metrics.append(epoch_metrics)

                stability_tracker.update(
                    loss_actor=loss_actor,
                    loss_critic=loss_critic,
                    return_test=return_test,
                    return_train=return_train,
                    entropy_coef=entropy_coef,
                    epoch=run_instance.epoch,
                )

                stability_score = stability_tracker.compute_stability_score()
                wandb_payload = {
                    "epoch": run_instance.epoch,
                    "trial_number": trial.number,
                    "loss_actor": loss_actor,
                    "loss_critic": loss_critic,
                    "return_test": return_test,
                    "return_train": return_train,
                    "entropy_coef": entropy_coef,
                    "stability_score": stability_score,
                    "trial_state": "running",
                }
                for key in (
                    "debug_alpha_steer",
                    "debug_alpha_gas",
                    "debug_alpha_brake",
                    "debug_log_std_mean",
                    "debug_log_std_min",
                ):
                    if key in last_stats:
                        wandb_payload[key] = float(last_stats[key])
                wandb.log(wandb_payload)

                if run_instance.epoch >= stability_window:
                    diagnostics = stability_tracker.get_diagnostics()

                    trial.report(stability_score, run_instance.epoch)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    should_stop, stop_reason = early_stopping.update(epoch_metrics)

                    if should_stop:
                        break

        dump_run_instance(run_instance, checkpoint_path)

        final_score = stability_tracker.compute_stability_score()
        final_diagnostics = stability_tracker.get_diagnostics()

        return final_score, final_diagnostics

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 0.0, {}


class StabilityObjective:
    """
    Optuna objective function for stability-focused hyperparameter optimization.
    """

    def __init__(
        self,
        base_config_path: str,
        output_dir: Path,
        max_epochs: int = 10,
        test_mode: bool = False,
        auto_start: bool = True,
    ):
        """
        Args:
            base_config_path: Path to base config.json
            output_dir: Directory to save results
            max_epochs: Maximum epochs per trial
            test_mode: If True, run minimal training for testing
            auto_start: If True, automatically start server and worker per trial
        """
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.test_mode = test_mode
        self.auto_start = auto_start

        with open(base_config_path, "r") as f:
            self.base_config = json.load(f)

        self.original_run_name = self.base_config.get("RUN_NAME", "sweep")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Execute single Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            float: Stability score to maximize
        """
        params = create_search_space(trial)
        config = apply_hyperparameters_to_config(params, self.base_config)

        trial_run_name = f"{self.original_run_name}_trial_{trial.number}"
        config["RUN_NAME"] = trial_run_name
        config["RUN_NAME"] = trial_run_name
        config["RESET_TRAINING"] = True
        config["MAX_EPOCHS"] = self.max_epochs
        
        # Dynamic Port Allocation to avoid "Address already in use"
        # Base port 55555, increment by 10 for each trial number to avoid conflict
        # Use modulo to stay within valid range (e.g. 50 parallel slots)
        # Port for Server (Incoming) and Trainer (Outgoing to Server)
        current_port_server = 55555 + ((trial.number * 10) % 1000)
        current_port_server_local = 55556 + ((trial.number * 10) % 1000)
        
        config["PORT"] = current_port_server
        config["LOCAL_PORT_SERVER"] = current_port_server_local
        
        # Increase timeouts for stability
        config["RECV_TIMEOUT_WORKER_FROM_SERVER"] = 600.0  # 10 mins
        config["WAIT_BEFORE_RECONNECTION"] = 20.0

        trial_config_path = self.output_dir / f"trial_{trial.number}_config.json"
        with open(trial_config_path, "w") as f:
            json.dump(config, f, indent=2)

        wandb_run = None
        temp_config_path = Path.home() / "TmrlData" / "config" / "config.json"
        backup_config_path = self.output_dir / f"backup_config_trial_{trial.number}.json"

        try:
            wandb_run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=f"hpsweep_trial_{trial.number}",
                config={
                    "trial_number": trial.number,
                    "hyperparameters": params,
                    "max_epochs": self.max_epochs,
                    "test_mode": self.test_mode,
                },
                reinit=True,
                allow_val_change=True,
            )

            if temp_config_path.exists():
                shutil.copy(temp_config_path, backup_config_path)

            with open(temp_config_path, "w") as f:
                json.dump(config, f, indent=2)

            import importlib

            importlib.reload(cfg)
            importlib.reload(cfg_obj)

            training_cls = cfg_obj.TRAINER

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
            
            # Start Server and Worker for this specific trial using the trial config
            if self.auto_start:
                tmrl_path = os.path.dirname(os.path.abspath(__file__))
                server_proc = start_server(tmrl_path, current_port_server)
                worker_proc = start_worker(tmrl_path, current_port_server)

            checkpoint_path = str(
                self.output_dir / f"trial_{trial.number}_checkpoint.pkl"
            )

            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            stability_score, diagnostics = run_training_epoch_with_stability(
                trial=trial,
                config=config,
                run_cls=training_cls,
                interface=interface,
                checkpoint_path=checkpoint_path,
                max_epochs=self.max_epochs,
                stability_window=5,
            )

            wandb.log(
                {
                    "final_stability_score": stability_score,
                    "trial_number": trial.number,
                    "trial_state": "completed",
                    **diagnostics,
                }
            )
            
            # Flush stdout
            print(
                f"Trial {trial.number} completed with stability score: {stability_score:.4f}"
            )
            
            # Explicitly cleanup subprocesses AFTER EACH TRIAL to free ports
            # The next trial will start new ones with new ports
            if self.auto_start:
                _cleanup_subprocesses()
                time.sleep(5) # Give OS time to release ports

            return stability_score

        except optuna.TrialPruned:
            if wandb_run is not None:
                wandb.log(
                    {
                        "pruned": True,
                        "trial_number": trial.number,
                        "trial_state": "pruned",
                    }
                )
            print(f"Trial {trial.number} pruned")
            raise

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback

            traceback.print_exc()
            if wandb_run is not None:
                wandb.log(
                    {
                        "error": str(e),
                        "trial_number": trial.number,
                        "trial_state": "failed",
                    }
                )
            return 0.0

        finally:
            if backup_config_path.exists():
                shutil.copy(backup_config_path, temp_config_path)
            if wandb_run is not None:
                wandb.finish()
            
            # Final cleanup ensures no zombies
            if self.auto_start:
                _cleanup_subprocesses()


def run_hyperparameter_search(
    n_trials: int = 100,
    max_epochs: int = 10,
    n_jobs: int = 1,
    test_mode: bool = False,
    resume_study_path: Optional[str] = None,
    auto_start: bool = True,
) -> optuna.Study:
    """
    Run hyperparameter optimization using Optuna.

    Args:
        n_trials: Number of trials to run
        max_epochs: Maximum epochs per trial
        n_jobs: Number of parallel jobs (1 = sequential)
        test_mode: If True, run minimal test
        resume_study_path: Path to existing study to resume
        auto_start: If True, automatically start server and worker

    Returns:
        optuna.Study: Completed study
    """
    _ensure_wandb_auth()
    
    # We rely on StabilityObjective to start/stop processes per trial
    # so we don't need to start them globally here.

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        base_config_path = Path.home() / "TmrlData" / "config" / "config.json"

        if not base_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {base_config_path}")

        if resume_study_path:
            study = optuna.load_study(
                study_name=STUDY_NAME, storage=f"sqlite:///{resume_study_path}"
            )
            print(f"Resuming study from {resume_study_path}")
            print(f"Previous trials: {len(study.trials)}")
        else:
            sampler = TPESampler(seed=42, n_startup_trials=5, multivariate=True)

            pruner = MedianPruner(
                n_startup_trials=5, n_warmup_steps=3, interval_steps=1
            )

            storage_path = OUTPUT_DIR / "study_database.db"
            storage = f"sqlite:///{storage_path}"

            study = optuna.create_study(
                study_name=STUDY_NAME,
                storage=storage,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )

            print(f"Created new study: {STUDY_NAME}")
            print(f"Database: {storage_path}")

        wandb_callback = WeightsAndBiasesCallback(
            metric_name="stability_score",
            wandb_kwargs={"entity": WANDB_ENTITY, "project": WANDB_PROJECT},
        )

        objective = StabilityObjective(
            base_config_path=str(base_config_path),
            output_dir=OUTPUT_DIR,
            max_epochs=max_epochs,
            test_mode=test_mode,
            auto_start=auto_start,
        )

        print(f"\n{'=' * 70}")
        print(f"Starting hyperparameter optimization")
        print(f"{'=' * 70}")
        print(f"Trials: {n_trials}")
        print(f"Max epochs per trial: {max_epochs}")
        print(f"Test mode: {test_mode}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"{'=' * 70}\n")

        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[wandb_callback],
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        print(f"\n{'=' * 70}")
        print(f"Optimization completed!")
        print(f"{'=' * 70}")

        best_trial = study.best_trial
        print(f"\nBest trial: #{best_trial.number}")
        print(f"Best stability score: {best_trial.value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        results = {
            "study_name": STUDY_NAME,
            "timestamp": datetime.now().isoformat(),
            "n_trials": len(study.trials),
            "best_trial_number": best_trial.number,
            "best_stability_score": best_trial.value,
            "best_params": best_trial.params,
            "all_trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }

        results_path = OUTPUT_DIR / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        best_config_path = OUTPUT_DIR / "best_hyperparameters.json"
        with open(best_config_path, "w") as f:
            json.dump(
                {
                    "best_params": best_trial.params,
                    "best_stability_score": best_trial.value,
                    "trial_number": best_trial.number,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to:")
        print(f"  - {results_path}")
        print(f"  - {best_config_path}")

        try:
            import optuna.visualization as vis

            fig_history = vis.plot_optimization_history(study)
            fig_history.write_html(str(OUTPUT_DIR / "optimization_history.html"))

            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html(str(OUTPUT_DIR / "param_importances.html"))

            fig_parallel = vis.plot_parallel_coordinate(study)
            fig_parallel.write_html(str(OUTPUT_DIR / "parallel_coordinate.html"))

            fig_slice = vis.plot_slice(study)
            fig_slice.write_html(str(OUTPUT_DIR / "slice_plot.html"))

            print(f"\nVisualizations saved to {OUTPUT_DIR}/")
            print(f"  - optimization_history.html")
            print(f"  - param_importances.html")
            print(f"  - parallel_coordinate.html")
            print(f"  - slice_plot.html")

        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

        return study

    finally:
        _cleanup_subprocesses()
        print("Cleaned up server and worker processes")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for SAC with stability focus"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run (default: 100)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum epochs per trial (default: 10)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, sequential)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with minimal settings",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing study database to resume",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start server and worker (start them manually)",
    )

    args = parser.parse_args()

    if args.test_mode:
        args.n_trials = min(args.n_trials, 5)
        args.max_epochs = min(args.max_epochs, 3)
        print("Running in TEST MODE with reduced trials and epochs")

    study = run_hyperparameter_search(
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        n_jobs=args.n_jobs,
        test_mode=args.test_mode,
        resume_study_path=args.resume,
        auto_start=not args.no_auto_start,
    )

    return study


if __name__ == "__main__":
    study = main()
