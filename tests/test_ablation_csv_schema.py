import pandas as pd
import numpy as np
import os
import tempfile
import unittest
from gymnasium.spaces import Box

from tmrl.training_offline import TrainingOffline


class DummyEnv:
    def __enter__(self):
        self.observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DummyMemory:
    def __init__(self, nb_steps, device):
        self.nb_steps = nb_steps
        self.device = device

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())


class DummyAgent:
    def __init__(self, observation_space, action_space, device):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device


def _append_csv_row(training_obj, epoch, rnd, return_train, loss_critic):
    row_data = pd.Series({"return_train": return_train, "loss_critic": loss_critic})
    if not training_obj._csv_header_written:
        header = ["wall_clock_seconds", "epoch", "round"] + list(row_data.index)
        training_obj._csv_writer.writerow(header)
        training_obj._csv_header_written = True

    values = [
        "1.0",
        epoch,
        rnd,
        f"{return_train:.6f}",
        f"{loss_critic:.6f}",
    ]
    training_obj._csv_writer.writerow(values)
    training_obj._csv_file.flush()


class TestAblationCsvSchema(unittest.TestCase):
    def test_ablation_csv_parseable_after_resume(self):
        previous_ablation_dir = os.environ.get("TMRL_ABLATION_DIR")
        previous_run_name = os.environ.get("TMRL_RUN_NAME")
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["TMRL_ABLATION_DIR"] = tmp_dir
            os.environ["TMRL_RUN_NAME"] = "csv_resume_test"

            first = TrainingOffline(env_cls=DummyEnv, memory_cls=DummyMemory, training_agent_cls=DummyAgent)
            _append_csv_row(first, epoch=0, rnd=0, return_train=1.0, loss_critic=0.5)
            first._csv_file.close()

            second = TrainingOffline(env_cls=DummyEnv, memory_cls=DummyMemory, training_agent_cls=DummyAgent)
            self.assertTrue(second._csv_header_written)
            _append_csv_row(second, epoch=0, rnd=1, return_train=1.2, loss_critic=0.4)
            second._csv_file.close()

            csv_path = os.path.join(tmp_dir, "csv_resume_test.csv")
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 2)
            self.assertTrue({"wall_clock_seconds", "epoch", "round", "return_train", "loss_critic"}.issubset(df.columns))
            pd.to_numeric(df["wall_clock_seconds"], errors="raise")
            pd.to_numeric(df["return_train"], errors="raise")
            pd.to_numeric(df["loss_critic"], errors="raise")

        if previous_ablation_dir is None:
            os.environ.pop("TMRL_ABLATION_DIR", None)
        else:
            os.environ["TMRL_ABLATION_DIR"] = previous_ablation_dir

        if previous_run_name is None:
            os.environ.pop("TMRL_RUN_NAME", None)
        else:
            os.environ["TMRL_RUN_NAME"] = previous_run_name
