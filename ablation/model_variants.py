"""
Ablation Study: Architecture Variants
======================================
GRU-Only and Vanilla (No Context) baselines for comparison
against the full "Everything" architecture.

Usage: Set TMRL_CONTEXT_MODE env var before starting trainer:
  - "contextual_film" (default): Full Transformer+GRU+FiLM
  - "gru_only": GRU context encoder, no Transformer
  - "baseline": No context encoder, plain MLP
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import ModuleList
from torch.nn.utils import spectral_norm

# Import shared constants and components from custom_models
from tmrl.custom.custom_models import (
    CONTEXT_INPUT_DIM, CONTEXT_Z_DIM, CONTEXT_WINDOW_SIZE, FUSED_DIM,
    FILM_HIDDEN, FILM_N_LAYERS, LOG_STD_MIN, LOG_STD_MAX,
    _compute_log_std_smooth, _squashed_gaussian_logprob,
    FiLMGenerator, FiLMActorMLP, FiLMQHead,
    EffNetV2, NANO_EFFNET_CFG, mlp, FusionGate,
)
from tmrl.actor import TorchActorModule


# ==============================================================
# Variant B: GRU-Only Context Encoder (Pre-Thesis Proposal)
# ==============================================================

class GRUOnlyContextEncoder(nn.Module):
    """
    Simplified context encoder using ONLY GRU (no Transformers).
    This matches the pre-thesis proposal architecture.

    Pipeline: context → deltas → GRU → z
    No dual-stream, no cross-attention, no compression.
    """
    REWARD_HORIZONS = [1, 3, 5]

    def __init__(self, input_dim=CONTEXT_INPUT_DIM, z_dim=CONTEXT_Z_DIM,
                 fused_dim=FUSED_DIM, max_len=CONTEXT_WINDOW_SIZE):
        super().__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        enriched_dim = input_dim * 2  # raw + deltas = 48

        # Simple projection → GRU (no transformer)
        self.input_proj = nn.Linear(enriched_dim, z_dim)
        self.gru = nn.GRU(z_dim, z_dim, num_layers=1, batch_first=True)

        # Output normalization
        self.norm = nn.LayerNorm(z_dim)

        # Reward predictors (same as Everything for fair comparison)
        self.reward_predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, 1))
            for _ in self.REWARD_HORIZONS
        ])

        self._prev_step = None

    def _compute_deltas(self, context_seq):
        deltas = context_seq[:, 1:, :] - context_seq[:, :-1, :]
        deltas = F.pad(deltas, (0, 0, 1, 0))
        return torch.cat([context_seq, deltas], dim=-1)

    def predict_reward(self, z):
        return [head(z).squeeze(-1) for head in self.reward_predictors]

    def forward(self, context_seq, fused_obs=None):
        enriched = self._compute_deltas(context_seq)  # (B, K, 48)
        projected = self.input_proj(enriched)  # (B, K, z_dim)
        self.gru.flatten_parameters()
        gru_out, h_n = self.gru(projected)  # h_n: (1, B, z_dim)
        z = self.norm(h_n.squeeze(0))  # (B, z_dim)
        return z

    def get_initial_hidden(self, batch_size=1, device='cpu'):
        return torch.zeros(1, batch_size, self.z_dim, device=device)

    def reset_online_state(self):
        self._prev_step = None

    def step(self, context_step, hidden, fused_obs=None):
        if self._prev_step is not None:
            delta = context_step - self._prev_step
        else:
            delta = torch.zeros_like(context_step)
        self._prev_step = context_step.detach()
        enriched = torch.cat([context_step, delta], dim=-1)
        projected = self.input_proj(enriched).unsqueeze(1)
        self.gru.flatten_parameters()
        gru_out, h_n = self.gru(projected, hidden)
        z = self.norm(h_n.squeeze(0))
        return z, h_n


# ==============================================================
# Variant B: GRU-Only Actor (uses GRUOnlyContextEncoder + FiLM)
# ==============================================================

class GRUOnlySharedBackboneHybridActor(TorchActorModule):
    """Actor using GRU-only context (no Transformer)."""
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        act_space = action_space
        act_dim = act_space.shape[0]
        self.act_limit = act_space.high[0]

        # Same perception as Everything
        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.float_mlp = mlp([28, 64, 64], nn.ReLU, nn.ReLU)
        self.fusion_gate = FusionGate(img_dim=128, float_dim=64, fused_dim=FUSED_DIM)

        # GRU-only context (THIS IS THE DIFFERENCE)
        self.context_encoder = GRUOnlyContextEncoder()
        self.film_generator = FiLMGenerator(z_dim=CONTEXT_Z_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS)

        self.net = FiLMActorMLP(input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS)
        self.mu_layer = nn.Linear(FILM_HIDDEN, act_dim)
        self.log_std_layer = nn.Linear(FILM_HIDDEN, act_dim)
        self.std_net = nn.Sequential(
            nn.Linear(FUSED_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, act_dim)
        )
        self._gru_hidden = None
        self._online_prev_reward = 0.0

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, lidar, act1, act2 = obs
        img_embed = self.cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)
        fused = self.fusion_gate(img_embed, float_embed)

        # Build context step for online GRU
        prev_reward = torch.full_like(speed, float(self._online_prev_reward))
        context_step = torch.cat((speed, lidar, act1, prev_reward), dim=-1)
        if self._gru_hidden is None:
            self._gru_hidden = self.context_encoder.get_initial_hidden(
                batch_size=fused.shape[0], device=fused.device)
        z, self._gru_hidden = self.context_encoder.step(context_step, self._gru_hidden, fused_obs=fused)

        film_params = self.film_generator(z)
        net_out = self.net(fused, film_params)
        mu = self.mu_layer(net_out)
        log_std_raw = self.std_net(fused)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi, _ = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()

    def reset_context(self):
        self._gru_hidden = None
        self._online_prev_reward = 0.0
        self.context_encoder.reset_online_state()

    def set_online_transition(self, reward=0.0, terminated=False, truncated=False, train_mode=True):
        if terminated or truncated:
            self._online_prev_reward = 0.0
        else:
            self._online_prev_reward = float(reward)


class GRUOnlyDroQHybridActorCritic(nn.Module):
    """DroQ Actor-Critic with GRU-only context (no Transformer)."""
    def __init__(self, observation_space, action_space, dropout_rate=0.01):
        super().__init__()
        self.n = 2

        self.actor = GRUOnlySharedBackboneHybridActor(observation_space, action_space)
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_gate = self.actor.fusion_gate

        self.context_encoder = self.actor.context_encoder
        self.film_generator = self.actor.film_generator

        self.qs = ModuleList([
            FiLMQHead(action_space, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN,
                      n_layers=FILM_N_LAYERS, dropout_rate=dropout_rate)
            for _ in range(2)
        ])

    def forward_features(self, obs, context=None):
        speed, gear, rpm, images, lidar, act1, act2 = obs
        img_embed = self._actor_cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(floats)
        fused = self._actor_fusion_gate(img_embed, float_embed)

        if context is not None:
            z = self.context_encoder(context, fused_obs=fused)
        else:
            z = torch.zeros(fused.shape[0], CONTEXT_Z_DIM, device=fused.device)

        film_params = self.film_generator(z)
        return fused, film_params, z

    def actor_from_features(self, fused, film_params, test=False, with_logprob=True):
        net_out = self.actor.net(fused, film_params)
        mu = self.actor.mu_layer(net_out)
        log_std_raw = self.actor.std_net(fused)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_total, logp_per_dim = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_total, logp_per_dim = None, None

        pi_action = torch.tanh(pi_action) * self.actor.act_limit
        return pi_action, logp_total, logp_per_dim

    def q_from_features(self, fused, act, film_params, q_idx=None):
        if q_idx is not None:
            return self.qs[q_idx](fused, act, film_params)
        return [q(fused, act, film_params) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


# ==============================================================
# Variant C: Vanilla Baseline (No Context, No FiLM)
# ==============================================================

class VanillaActorMLP(nn.Module):
    """Standard MLP actor without FiLM modulation."""
    def __init__(self, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x, film_params=None):
        # film_params ignored — this is the ablation point
        return self.net(x)


class VanillaQHead(nn.Module):
    """Standard Q-head without FiLM modulation."""
    def __init__(self, action_space, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN,
                 n_layers=FILM_N_LAYERS, dropout_rate=0.01):
        super().__init__()
        act_dim = action_space.shape[0]
        layers = []
        in_d = input_dim + act_dim
        for _ in range(n_layers):
            layers.append(spectral_norm(nn.Linear(in_d, hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, fused, act, film_params=None):
        # film_params ignored — no FiLM conditioning
        x = torch.cat([fused, act], dim=-1)
        return self.net(x).squeeze(-1)


class VanillaSharedBackboneHybridActor(TorchActorModule):
    """Actor with NO context encoder, NO FiLM. Pure reactive policy."""
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        self.cnn = EffNetV2(cfgs=NANO_EFFNET_CFG, nb_channels_in=4, dim_output=128)
        self.float_mlp = mlp([28, 64, 64], nn.ReLU, nn.ReLU)
        self.fusion_gate = FusionGate(img_dim=128, float_dim=64, fused_dim=FUSED_DIM)

        # Plain MLP, no FiLM
        self.net = VanillaActorMLP(input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN, n_layers=FILM_N_LAYERS)
        self.mu_layer = nn.Linear(FILM_HIDDEN, act_dim)
        self.log_std_layer = nn.Linear(FILM_HIDDEN, act_dim)
        self.std_net = nn.Sequential(
            nn.Linear(FUSED_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, lidar, act1, act2 = obs
        img_embed = self.cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self.float_mlp(floats)
        fused = self.fusion_gate(img_embed, float_embed)

        net_out = self.net(fused)
        mu = self.mu_layer(net_out)
        log_std_raw = self.std_net(fused)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi, _ = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.squeeze().cpu().numpy()


class VanillaDroQHybridActorCritic(nn.Module):
    """DroQ Actor-Critic with NO context, NO FiLM. Pure reactive baseline."""
    def __init__(self, observation_space, action_space, dropout_rate=0.01):
        super().__init__()
        self.n = 2

        self.actor = VanillaSharedBackboneHybridActor(observation_space, action_space)
        self._actor_cnn = self.actor.cnn
        self._actor_float_mlp = self.actor.float_mlp
        self._actor_fusion_gate = self.actor.fusion_gate

        # No context encoder, no FiLM generator
        self.context_encoder = None
        self.film_generator = None

        self.qs = ModuleList([
            VanillaQHead(action_space, input_dim=FUSED_DIM, hidden_dim=FILM_HIDDEN,
                         n_layers=FILM_N_LAYERS, dropout_rate=dropout_rate)
            for _ in range(2)
        ])

    def forward_features(self, obs, context=None):
        speed, gear, rpm, images, lidar, act1, act2 = obs
        img_embed = self._actor_cnn(images.float())
        floats = torch.cat((speed, gear, rpm, lidar, act1, act2), dim=-1)
        float_embed = self._actor_float_mlp(floats)
        fused = self._actor_fusion_gate(img_embed, float_embed)

        # No context → dummy z and film_params
        z = torch.zeros(fused.shape[0], CONTEXT_Z_DIM, device=fused.device)
        film_params = None
        return fused, film_params, z

    def actor_from_features(self, fused, film_params, test=False, with_logprob=True):
        net_out = self.actor.net(fused)
        mu = self.actor.mu_layer(net_out)
        log_std_raw = self.actor.std_net(fused)
        log_std = _compute_log_std_smooth(log_std_raw)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_total, logp_per_dim = _squashed_gaussian_logprob(pi_distribution, pi_action)
        else:
            logp_total, logp_per_dim = None, None

        pi_action = torch.tanh(pi_action) * self.actor.act_limit
        return pi_action, logp_total, logp_per_dim

    def q_from_features(self, fused, act, film_params, q_idx=None):
        if q_idx is not None:
            return self.qs[q_idx](fused, act, film_params)
        return [q(fused, act, film_params) for q in self.qs]

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()
