import unittest
from pathlib import Path

import tmrl.config.config_constants as cfg
from tmrl.training_offline import compute_stall_state, compute_stall_dashboard


class TestStallMetrics(unittest.TestCase):
    def test_stall_detector_flags_flat_eval_series(self):
        eval_returns = [10.0] * 25
        train_returns = [12.0] * 25
        state = compute_stall_state(eval_returns, train_returns, best_eval_ma10=None, stall_epochs=0)
        self.assertTrue(state["stalled"])
        self.assertGreaterEqual(state["stall_epochs"], 10)

    def test_stall_detector_keeps_improving_series_unstalled(self):
        eval_returns = [1.0 * (1.2 ** i) for i in range(20)]
        train_returns = [2.0 * (1.2 ** i) for i in range(20)]
        best = None
        stall_epochs = 0
        state = None
        for i in range(len(eval_returns)):
            state = compute_stall_state(
                eval_returns[: i + 1],
                train_returns[: i + 1],
                best_eval_ma10=best,
                stall_epochs=stall_epochs,
            )
            best = state["best_eval_ma10"]
            stall_epochs = state["stall_epochs"]
        self.assertIsNotNone(state)
        self.assertFalse(state["stalled"])

    def test_dashboard_status_levels(self):
        healthy = compute_stall_dashboard({"stall_epochs": 2, "patience": 10}, warning_epochs=5)
        warning = compute_stall_dashboard({"stall_epochs": 6, "patience": 10}, warning_epochs=5)
        stalled = compute_stall_dashboard({"stall_epochs": 10, "patience": 10}, warning_epochs=5)
        self.assertEqual(healthy["status"], "healthy")
        self.assertEqual(warning["status"], "warning")
        self.assertEqual(stalled["status"], "stalled")


class TestEvalIntervalWiring(unittest.TestCase):
    def test_test_episode_interval_constant_exists(self):
        self.assertTrue(hasattr(cfg, "TEST_EPISODE_INTERVAL"))
        self.assertGreaterEqual(int(cfg.TEST_EPISODE_INTERVAL), 0)

    def test_worker_run_uses_configured_test_episode_interval(self):
        source = Path("tmrl/tmrl/__main__.py").read_text(encoding="utf-8")
        self.assertIn("rw.run(test_episode_interval=cfg.TEST_EPISODE_INTERVAL)", source)
