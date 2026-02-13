import numpy as np
import unittest
from gymnasium.spaces import Box

from ablation.model_variants import (
    GRUOnlySharedBackboneHybridActor,
    VanillaSharedBackboneHybridActor,
)
from tmrl.actor import TorchActorModule
from tmrl.custom.custom_models import ContextualSharedBackboneHybridActor


def _make_spaces():
    observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return observation_space, action_space


class TestAblationActorInterface(unittest.TestCase):
    def test_ablation_policies_inherit_torch_actor_module(self):
        self.assertTrue(issubclass(ContextualSharedBackboneHybridActor, TorchActorModule))
        self.assertTrue(issubclass(GRUOnlySharedBackboneHybridActor, TorchActorModule))
        self.assertTrue(issubclass(VanillaSharedBackboneHybridActor, TorchActorModule))

    def test_ablation_policies_support_rollout_worker_device_api(self):
        observation_space, action_space = _make_spaces()
        for policy_cls in (
            ContextualSharedBackboneHybridActor,
            GRUOnlySharedBackboneHybridActor,
            VanillaSharedBackboneHybridActor,
        ):
            policy = policy_cls(observation_space, action_space)
            moved_policy = policy.to_device("cpu")
            self.assertIs(moved_policy, policy)
            self.assertTrue(hasattr(policy, "load"))
