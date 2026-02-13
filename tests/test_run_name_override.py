import importlib
import os
import unittest

import tmrl.config.config_constants as cfg_constants


class TestRunNameOverride(unittest.TestCase):
    def test_tmrl_run_name_env_override(self):
        run_name = "pytest_run_override"
        previous = os.environ.get("TMRL_RUN_NAME")
        os.environ["TMRL_RUN_NAME"] = run_name

        cfg = importlib.reload(cfg_constants)
        self.assertEqual(cfg.RUN_NAME, run_name)
        self.assertTrue(cfg.MODEL_PATH_WORKER.endswith(f"{run_name}.tmod"))
        self.assertTrue(cfg.MODEL_PATH_TRAINER.endswith(f"{run_name}_t.tmod"))
        self.assertTrue(cfg.CHECKPOINT_PATH.endswith(f"{run_name}_t.tcpt"))

        if previous is None:
            os.environ.pop("TMRL_RUN_NAME", None)
        else:
            os.environ["TMRL_RUN_NAME"] = previous
        importlib.reload(cfg)
