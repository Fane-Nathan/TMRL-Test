import unittest

import numpy as np
import torch
from gymnasium.spaces import Box

from tmrl.custom.custom_models import (
    DroQHybridActorCritic,
    SharedBackboneHybridActorCritic,
)


def _make_spaces():
    observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return observation_space, action_space


def _make_obs(batch_size=2):
    return (
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.zeros(batch_size, 4, 64, 64),
        torch.zeros(batch_size, 19),
        torch.zeros(batch_size, 3),
        torch.zeros(batch_size, 3),
    )


class TestNonContextInterfaceContract(unittest.TestCase):
    def _assert_contract(self, model):
        obs = _make_obs()
        features, film, z = model.forward_features(obs)
        self.assertIsNone(film)
        self.assertIsNone(z)
        self.assertEqual(features.ndim, 2)

        action, logp_total, logp_per_dim = model.actor_from_features(
            features, film, with_logprob=True
        )
        self.assertEqual(action.shape[-1], 3)
        self.assertEqual(logp_total.ndim, 1)
        self.assertEqual(logp_per_dim.shape[-1], 3)

    def test_shared_backbone_hybrid_actor_critic_contract(self):
        observation_space, action_space = _make_spaces()
        model = SharedBackboneHybridActorCritic(observation_space, action_space, n=2)
        self._assert_contract(model)

    def test_droq_hybrid_actor_critic_contract(self):
        observation_space, action_space = _make_spaces()
        model = DroQHybridActorCritic(observation_space, action_space, dropout_rate=0.01)
        self._assert_contract(model)
