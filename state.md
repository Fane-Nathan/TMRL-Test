STATUS: APPROVED

# 🛠️ TMRL Architecture & Stability Remediation Plan

**Objective:** Prevent late-stage policy collapse, stabilize Q-value targets, and fix the deterministic policy (`return_test_det`) in the TrackMania DroQ/SAC custom architecture.

## Phase 1: Mathematical & Normalization Fixes

_File to edit: `tmrl/custom/custom_models.py`_

### [x] 1. Fix the `LOG_STD_MIN` Bottleneck

**Why:** A minimum log standard deviation of `-2` forces the car to inject a minimum of ~13.5% random noise into its steering, making it impossible to drive straight at high speeds.

- **Find:** `LOG_STD_MIN = -2` (around line 34)
- **Change to:**

```python
LOG_STD_MIN = -20.0
```

### [ ] 2. Remove `BatchNorm2d` from the CNN

**Why:** Batch Normalization tracks a running mean/variance. In off-policy RL, the replay buffer distribution constantly shifts, causing the running stats to corrupt the network during `eval()` mode (which is why your deterministic policy scored `-0.405`).

- **Find:** `conv_3x3_bn`, `conv_1x1_bn`, and `MBConv` functions/classes.
- **Change:** Replace every instance of `nn.BatchNorm2d(...)` with `nn.GroupNorm(num_groups=1, num_channels=...)`.

```python
# Example fix for conv_3x3_bn:
def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.GroupNorm(num_groups=1, num_channels=oup), # <--- Replaced BatchNorm2d
        SiLU()
    )
```

_(Make sure to do this for `conv_1x1_bn` and the `MBConv` class as well)._

### [ ] 3. Disable Dropout in the Context Encoder

**Why:** DroQ relies on dropout _only_ in the Q-heads to estimate uncertainty. Having dropout in the Transformer creates noisy, moving targets for the Bellman equation, preventing the Critic from converging.

- **Find:** `ContextEncoder.__init__`
- **Change:** Set `dropout=0.0` in both Transformer layers (`state_layer` and `action_layer`).

```python
state_layer = nn.TransformerEncoderLayer(
    d_model=enriched_dim, nhead=4, dim_feedforward=128,
    dropout=0.0, batch_first=True, norm_first=True  # <--- Changed from 0.1 to 0.0
)
```

---

## Phase 2: Fixing Gradient Interference (The Shared Backbone)

_File to edit: `tmrl/custom/custom_models.py`_

**Why:** Because the Actor and Critic share the CNN and Context Encoder, calling `.backward()` on the Actor destroys the representation learned by the Critic. We must block the Actor's gradients from flowing into the shared features.

### [ ] 4. Detach features in `SharedBackboneHybridActorCritic`

- **Find:** `def actor_from_features(self, features, film=None, test=False, with_logprob=True):`
- **Change:** Add `.detach()` to the features.

```python
def actor_from_features(self, features, film=None, test=False, with_logprob=True):
    del film
    features = features.detach() # <--- ADD THIS LINE
    net_out = self.actor.net(features)
    # ... rest of function remains the same
```

### [ ] 5. Detach features & FiLM params in `ContextualDroQHybridActorCritic`

- **Find:** `def actor_from_features(self, fused, film_params, test=False, with_logprob=True):`
- **Change:** Add `.detach()` to both inputs.

```python
def actor_from_features(self, fused, film_params, test=False, with_logprob=True):
    fused = fused.detach() # <--- ADD THIS LINE

    if film_params is not None:
        # <--- ADD THIS BLOCK --->
        # Detach the tuples of (gamma, beta) to protect the FiLM generator
        film_params = [(g.detach(), b.detach()) for g, b in film_params]

    net_out = self.actor.net(fused, film_params)
    # ... rest of function remains the same
```

---

## Phase 3: Fixing the Hyperparameter Search Space

_File to edit: `hyperparameter_sweep_optuna.py`_

**Why:** Optuna is currently allowed to select combinations that are mathematically guaranteed to fail in non-stationary, high-UTD reinforcement learning. You must restrict the search space.

### [ ] 6. Restrict `GAMMA` (Discount Factor)

**Why:** TrackMania requires looking far ahead. A gamma of `0.95` only looks ~20 steps ahead.

- **Fix:** Change your Optuna `suggest_` limits for gamma to force far-sightedness:

```python
gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
```

### [ ] 7. Restrict Adam `beta2`

**Why:** A `beta2` of `0.999` retains too much gradient history, causing explosive update steps when the car hits a new part of the track.

- **Fix:** Change your Optuna space for both betas to avoid 0.999:

```python
beta2_critic = trial.suggest_categorical("beta2_critic", [0.99, 0.995])
beta2_actor = trial.suggest_categorical("beta2_actor", [0.99, 0.995])
```

### [ ] 8. Cap the UTD (Update-To-Data) Ratio

**Why:** Updating the network 20 times per environment step (`20.0`) on a shared backbone will cause severe overfitting to the most recent batch.

- **Fix:** Lower the maximum training steps per environment step to a safer range (e.g., 2 to 8).

```python
max_training_steps_per_env_step = trial.suggest_int("max_training_steps_per_env_step", 2, 8)
```

---

## Phase 4: Restart Protocol

Once these code changes are implemented, your previous replay buffers and Optuna trials are mathematically poisoned with data from the old, unstable architecture.

- [ ] **Wipe Optuna Database:** Delete your `study_database.db` (or whatever your Optuna sqlite file is named) so the hyperparameter search doesn't try to learn from the collapsed runs.
- [ ] **Clear Replay Buffers:** Empty your TMRL replay memory folders (delete `.pkl` files). The old data contains thousands of high-speed wobbles/crashes caused by the `LOG_STD_MIN = -2` bug.
- [ ] **Restart Training:** Launch your sweep again.

With these fixes, your deterministic policy will actually start driving, your actor loss will remain flat/stable, and the agent will be able to sustain training for 50+ epochs without catastrophic collapse!
