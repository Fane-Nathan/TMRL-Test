import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna

import hyperparameter_sweep_optuna as sweep


class _FakeRun:
    def __init__(self):
        self.epoch = 0

    def run_epoch(self, interface=None):
        self.epoch += 1
        return [
            {
                "loss_actor": 1.0 / self.epoch,
                "loss_critic": 0.5 / self.epoch,
                "return_test": float(self.epoch),
                "return_train": float(self.epoch) * 1.1,
                "entropy_coef": 0.2 - 0.01 * self.epoch,
                "debug_alpha_steer": 0.2,
                "debug_alpha_gas": 0.18,
                "debug_alpha_brake": 0.16,
                "debug_log_std_mean": -1.5,
                "debug_log_std_min": -2.0,
            }
        ]


class _FakeTrial:
    def __init__(self, number=0):
        self.number = number
        self.reports = []

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return False


class _PruningTrial(_FakeTrial):
    def should_prune(self):
        return len(self.reports) >= 1


class TestSweepEpochLogging(unittest.TestCase):
    def test_ensure_wandb_auth_raises_without_env_or_config(self):
        previous = os.environ.pop("WANDB_API_KEY", None)
        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td)
            with patch("hyperparameter_sweep_optuna.Path.home", return_value=fake_home):
                with self.assertRaises(EnvironmentError):
                    sweep._ensure_wandb_auth()
        if previous is not None:
            os.environ["WANDB_API_KEY"] = previous

    def test_ensure_wandb_auth_loads_key_from_config(self):
        previous = os.environ.pop("WANDB_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as td:
                fake_home = Path(td)
                config_dir = fake_home / "TmrlData" / "config"
                config_dir.mkdir(parents=True, exist_ok=True)
                config_path = config_dir / "config.json"
                config_path.write_text(
                    json.dumps({"WANDB_KEY": "wandb_test_key_from_config"}),
                    encoding="utf-8",
                )
                with patch("hyperparameter_sweep_optuna.Path.home", return_value=fake_home):
                    with patch("hyperparameter_sweep_optuna.wandb.login") as mock_login:
                        sweep._ensure_wandb_auth()
                        self.assertEqual(
                            os.environ.get("WANDB_API_KEY"), "wandb_test_key_from_config"
                        )
                        mock_login.assert_called_once_with(
                            key="wandb_test_key_from_config"
                        )
        finally:
            os.environ.pop("WANDB_API_KEY", None)
            if previous is not None:
                os.environ["WANDB_API_KEY"] = previous

    def test_epoch_metrics_are_logged_each_epoch(self):
        trial = _FakeTrial(number=7)
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = os.path.join(td, "run.tcpt")
            with patch("hyperparameter_sweep_optuna.wandb.log") as mock_log:
                score, diagnostics = sweep.run_training_epoch_with_stability(
                    trial=trial,
                    config={},
                    run_cls=_FakeRun,
                    interface=object(),
                    checkpoint_path=checkpoint_path,
                    max_epochs=3,
                    stability_window=2,
                )

        self.assertGreaterEqual(mock_log.call_count, 3)
        for call in mock_log.call_args_list[:3]:
            payload = call.args[0]
            for key in (
                "epoch",
                "trial_number",
                "loss_actor",
                "loss_critic",
                "return_test",
                "return_train",
                "entropy_coef",
                "stability_score",
            ):
                self.assertIn(key, payload)
        self.assertIsInstance(score, float)
        self.assertIsInstance(diagnostics, dict)

    def test_prune_still_logs_epoch_before_exception(self):
        trial = _PruningTrial(number=11)
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = os.path.join(td, "run.tcpt")
            with patch("hyperparameter_sweep_optuna.wandb.log") as mock_log:
                with self.assertRaises(optuna.TrialPruned):
                    sweep.run_training_epoch_with_stability(
                        trial=trial,
                        config={},
                        run_cls=_FakeRun,
                        interface=object(),
                        checkpoint_path=checkpoint_path,
                        max_epochs=3,
                        stability_window=1,
                    )

        self.assertGreaterEqual(mock_log.call_count, 1)
        first_payload = mock_log.call_args_list[0].args[0]
        self.assertEqual(first_payload["trial_number"], 11)
        self.assertEqual(first_payload["epoch"], 1)
