# STATUS: APPROVED

## Mission: Long-Term Entropy Stability — Structural Architecture Hardening

**Date:** 2026-02-12
**Architect:** Antigravity (Lead System Architect)
**Executor:** GPT-5.3-Codex

---

## Problem Statement

The current actor architecture is structurally prone to entropy collapse over long training because:

1. **Shared bottleneck:** `mu_layer` and `log_std_layer` consume the **same** `net_out` tensor. When the network learns confident actions (low loss → sharp `mu`), the same features drive `log_std` down.
2. **Hard clamp dead zone:** `torch.clamp(log_std, MIN, MAX)` kills gradients at boundaries — once `log_std` hits the floor, there is no gradient signal to push it back.
3. **Scalar alpha:** One entropy coefficient governs 3 action dimensions with vastly different exploration needs (steering vs gas vs brake).

## Design: Three Structural Fixes

### Fix A — Separate `log_std` Branch

Give `log_std` its own MLP from raw fused features, completely decoupled from the policy `net_out`:

```
Current:   features → net(features) → [mu_layer, log_std_layer]  (COUPLED)
Proposed:  features → net(features) → mu_layer                   (policy confidence)
           features → std_net(features) → log_std                (independent uncertainty)
```

`std_net` is a tiny 2-layer MLP: `Linear(FUSED_DIM, 64) → SiLU → Linear(64, dim_act)`.

### Fix B — Smooth Tanh Parametrization (Replace Clamp)

Replace the hard clamp with a smooth tanh mapping that preserves gradients everywhere:

```python
# BEFORE (dead gradient at boundaries):
log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

# AFTER (smooth, always has gradient):
log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std_raw) + 1)
```

This maps `(-∞, +∞) → [LOG_STD_MIN, LOG_STD_MAX]` smoothly. Used by Stable Baselines 3 — battle-tested.

### Fix C — Per-Dimension Entropy Coefficient

Replace scalar `log_alpha` with a vector of size `dim_act = 3`:

```python
# BEFORE: one alpha for all dims
self.log_alpha = torch.log(torch.ones(1, ...) * self.alpha)

# AFTER: one alpha per action dimension
self.log_alpha = torch.log(torch.ones(dim_act, ...) * self.alpha)
self.target_entropy = torch.full((dim_act,), target_entropy / dim_act, ...)
```

This lets steering keep exploring while gas/brake can independently become more deterministic. The alpha update and policy loss use **per-dimension** log probabilities (before `.sum()`).

---

## Executor Checklist

> Execute steps **in order**. Each step is atomic.

---

### Step 1: Add `std_net` to `SharedBackboneHybridActor`

- **File:** `tmrl/custom/custom_models.py`
- **Class:** `SharedBackboneHybridActor.__init__` (line ~906)
- **After** `self.log_std_layer = nn.Linear(256, dim_act)` (line 920), add:

```python
self.std_net = nn.Sequential(
    nn.Linear(128 + 64, 64),
    nn.SiLU(),
    nn.Linear(64, dim_act)
)
```

- **Note:** Keep `self.log_std_layer` in place for now (backward compat with checkpoints). The new `std_net` will be used instead, but `log_std_layer` stays as dead weight until old checkpoints are no longer needed.

---

### Step 2: Add `std_net` to `ContextualSharedBackboneHybridActor`

- **File:** `tmrl/custom/custom_models.py`
- **Class:** `ContextualSharedBackboneHybridActor.__init__` (line ~1456)
- **After** `self.log_std_layer = nn.Linear(FILM_HIDDEN, dim_act)` (line 1479), add:

```python
self.std_net = nn.Sequential(
    nn.Linear(FUSED_DIM, 64),
    nn.SiLU(),
    nn.Linear(64, dim_act)
)
```

---

### Step 3: Create a shared helper for smooth log_std + logprob computation

- **File:** `tmrl/custom/custom_models.py`
- **Location:** After the `LOG_STD_MIN` / `LOG_STD_MAX` constants (line ~49), add:

```python
def _compute_log_std_smooth(log_std_raw):
    """Tanh-based smooth log_std parametrization. Always has gradient."""
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std_raw) + 1)

def _squashed_gaussian_logprob(pi_distribution, pi_action):
    """Compute per-dim and total log prob with tanh squashing correction."""
    logp_per_dim = pi_distribution.log_prob(pi_action)
    tanh_correction = 2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
    logp_per_dim = logp_per_dim - tanh_correction  # (B, dim_act)
    logp_total = logp_per_dim.sum(dim=-1)           # (B,)
    return logp_total, logp_per_dim
```

---

### Step 4: Update `SharedBackboneHybridActor.forward` (line ~923)

- **Replace the log_std + logprob block** (lines 936–950) with:

```python
# Policy head — DECOUPLED mu/std
mu = self.mu_layer(net_out)
log_std_raw = self.std_net(features)  # from raw fused features, NOT net_out
log_std = _compute_log_std_smooth(log_std_raw)
std = torch.exp(log_std)

pi_distribution = Normal(mu, std)
if test:
    pi_action = mu
else:
    pi_action = pi_distribution.rsample()

if with_logprob:
    logp_pi, _ = _squashed_gaussian_logprob(pi_distribution, pi_action)
else:
    logp_pi = None
```

---

### Step 5: Update `ContextualSharedBackboneHybridActor.forward` (line ~1504)

- **Replace the log_std + logprob block** (lines 1528–1543) with the same pattern as Step 4, but using `fused` instead of `features`:

```python
mu = self.mu_layer(net_out)
log_std_raw = self.std_net(fused)  # from fused features, NOT net_out
log_std = _compute_log_std_smooth(log_std_raw)
std = torch.exp(log_std)

pi_distribution = Normal(mu, std)
if test:
    pi_action = mu
else:
    pi_action = pi_distribution.rsample()

if with_logprob:
    logp_pi, _ = _squashed_gaussian_logprob(pi_distribution, pi_action)
else:
    logp_pi = None
```

---

### Step 6: Update ALL `actor_from_features` methods to return per-dim logprob

There are 4 `actor_from_features` implementations. Each must:

1. Use `self.actor.std_net(features)` instead of `self.actor.log_std_layer(net_out)`
2. Use `_compute_log_std_smooth` instead of `torch.clamp`
3. Return 3 values: `(pi_action, logp_total, logp_per_dim)`

**Locations:**

#### 6a: `SharedBackboneHybridActorCritic.actor_from_features` (line ~996)

- Input: `features`
- Change: `log_std_raw = self.actor.std_net(features)`

#### 6b: `DroQHybridActorCritic.actor_from_features` (line ~1108)

- Input: `features`
- Change: `log_std_raw = self.actor.std_net(features)`

#### 6c: `ContextualDroQHybridActorCritic.actor_from_features` (line ~1624)

- Input: `fused`
- Change: `log_std_raw = self.actor.std_net(fused)`

#### 6d: `SharedBackboneHybridActorCritic.actor_from_features` — (already covered in 6a, same class)

**Template for all 4:**

```python
def actor_from_features(self, features, *args, test=False, with_logprob=True):
    net_out = self.actor.net(features, *args)  # args = film_params for contextual
    mu = self.actor.mu_layer(net_out)
    log_std_raw = self.actor.std_net(features)  # DECOUPLED from net_out
    log_std = _compute_log_std_smooth(log_std_raw)
    std = torch.exp(log_std)

    pi_distribution = Normal(mu, std)
    if test:
        pi_action = mu
    else:
        pi_action = pi_distribution.rsample()

    if with_logprob:
        logp_total, logp_per_dim = _squashed_gaussian_logprob(pi_distribution, pi_action)
    else:
        logp_total, logp_per_dim = None, None

    pi_action = torch.tanh(pi_action)
    pi_action = self.actor.act_limit * pi_action
    return pi_action, logp_total, logp_per_dim
```

> **IMPORTANT:** All callers of `actor_from_features` that unpack `(action, logp)` must now unpack `(action, logp, logp_per_dim)`. Callers that don't need per-dim can use `_`.
> Search for all call sites in `custom_algorithms.py` and update them.

---

### Step 7: Convert `DroQSACAgent` to per-dimension alpha

- **File:** `tmrl/custom/custom_algorithms.py`

#### 7a: `__init__` (line ~621)

**Replace** (line 682–691):

```python
if self.target_entropy is None:
    self.target_entropy = -np.prod(action_space.shape)
else:
    self.target_entropy = float(self.target_entropy)

if self.learn_entropy_coef:
    self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
    self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
else:
    self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)
```

**With:**

```python
dim_act = action_space.shape[0]
if self.target_entropy is None:
    self.target_entropy = torch.full((dim_act,), -1.0, device=self.device)
else:
    per_dim = float(self.target_entropy) / dim_act
    self.target_entropy = torch.full((dim_act,), per_dim, device=self.device)

if self.learn_entropy_coef:
    self.log_alpha = torch.log(torch.ones(dim_act, device=self.device) * self.alpha).requires_grad_(True)
    self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
else:
    self.alpha_t = torch.full((dim_act,), float(self.alpha), device=self.device)
```

#### 7b: `train()` — alpha usage (throughout the method)

All places where `alpha_t` is used must handle the vector form:

**Target Q backup (line ~746):**

```python
# BEFORE:
backup = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * (min_q - alpha_t * logp_a2.unsqueeze(-1))

# AFTER: logp_a2 is scalar (total), alpha_t is scalar (mean of per-dim)
alpha_scalar = alpha_t.mean()
backup = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * (min_q - alpha_scalar * logp_a2.unsqueeze(-1))
```

**Policy loss (line ~800):**

```python
# BEFORE:
loss_pi = (alpha_t * logp_pi.unsqueeze(-1) - min_q_pi).mean()

# AFTER: use per-dim logp and per-dim alpha
entropy_cost = (alpha_t * logp_per_dim).sum(dim=-1)  # (B,)
loss_pi = (entropy_cost.unsqueeze(-1) - min_q_pi).mean()
```

**Alpha loss (line ~818):**

```python
# BEFORE:
loss_alpha = -(self.log_alpha * (logp_pi_alpha + self.target_entropy).detach()).mean()

# AFTER: per-dim alpha update
_, logp_per_dim_alpha = self.model.actor_from_features(fused_alpha, film_alpha)[1:]
loss_alpha = -(self.log_alpha * (logp_per_dim_alpha.detach().mean(dim=0) + self.target_entropy)).sum()
```

**Alpha floor (line ~826):**

```python
# Same logic but per-dim
with torch.no_grad():
    self.log_alpha.clamp_(min=np.log(self.alpha_floor))
```

#### 7c: Update all `actor_from_features` call sites in `train()`

Every call to `actor_from_features` now returns 3 values. Update unpacking:

```python
# BEFORE:
a2, logp_a2 = self.model_target.actor_from_features(...)
pi, logp_pi = self.model.actor_from_features(...)

# AFTER:
a2, logp_a2, _ = self.model_target.actor_from_features(...)
pi, logp_pi, logp_per_dim = self.model.actor_from_features(...)
```

---

### Step 8: Update diagnostics in `ret_dict`

- **File:** `tmrl/custom/custom_algorithms.py`
- **In `DroQSACAgent.train()`**, update the diagnostics section to log per-dim alpha:

```python
ret_dict["debug_alpha_steer"] = alpha_t[0].item()
ret_dict["debug_alpha_gas"] = alpha_t[1].item()
ret_dict["debug_alpha_brake"] = alpha_t[2].item()
```

---

### Step 9: Update ablation model variants for compatibility

- **File:** `ablation/model_variants.py`
- The ablation variants import `FILM_HIDDEN`, `FILM_N_LAYERS` from `custom_models.py` (already fixed).
- The ablation actors (`GRUOnlySharedBackboneHybridActor`, `VanillaSharedBackboneHybridActor`) must also add `std_net` and use the new smooth parametrization.
- Their corresponding `DroQHybridActorCritic` variants must also return 3 values from `actor_from_features`.

> **Rule:** Any class with `mu_layer` + `log_std_layer` must get `std_net` + smooth parametrization.

---

## Interface Contract Changes

| Method                     | Old Return             | New Return                             |
| -------------------------- | ---------------------- | -------------------------------------- |
| `actor_from_features(...)` | `(action, logp_total)` | `(action, logp_total, logp_per_dim)`   |
| `*.forward(obs, ...)`      | `(action, logp_total)` | `(action, logp_total)` — **no change** |

> **Standalone `forward()` methods** (used by worker for inference) **keep the 2-tuple** return. Only `actor_from_features` (used by training loop) returns 3 values, since per-dim logp is only needed for training.

---

## Verification Plan

### Automated: Import Smoke Test

```powershell
cd c:\Users\felix\OneDrive\Documents\Data Mining\tmrl
python -c "from tmrl.custom.custom_models import DroQHybridActorCritic, ContextualDroQHybridActorCritic; print('OK')"
```

### Automated: Existing Actor Interface Test

```powershell
cd c:\Users\felix\OneDrive\Documents\Data Mining\tmrl
python -m pytest tests/test_ablation_actor_interface.py -v
```

### Manual: Training Launch

1. Start server: `python -m tmrl --server`
2. Start trainer: `python -m tmrl --trainer`
3. Start worker: `python -m tmrl --worker`
4. Verify no crash within first 30 seconds
5. After ~500 steps, check logs for `debug_alpha_steer`, `debug_alpha_gas`, `debug_alpha_brake` — they should diverge from each other (proving per-dim is working)
6. Check `debug_log_std_mean` stays in `[-3, 0]` range

---

## Risk Assessment

| Fix               | Impact                     | Risk                               | Checkpoint Compat                                                       |
| ----------------- | -------------------------- | ---------------------------------- | ----------------------------------------------------------------------- |
| A (std_net)       | High — breaks coupling     | Low — additive                     | ✅ Old `log_std_layer` kept as dead weight                              |
| B (tanh param)    | Medium — removes dead zone | Very Low — drop-in                 | ✅ No weight shape change                                               |
| C (per-dim alpha) | High — per-dim regulation  | Medium — changes training dynamics | ⚠️ Old scalar checkpoints incompatible, requires `RESET_TRAINING: true` |

---

## Phase 2: Enhanced Telemetry & Safety (Active)

**Date:** 2026-02-12
**Mission:** Enable real-time visibility into training collapse dynamics.

### Step 10: Enable Epoch-Level WandB Logging

- **File:** `tmrl/hyperparameter_sweep_optuna.py`
- **Function:** `run_training_epoch_with_stability` (line ~220)
- **Goal:** Log metrics to WandB _inside_ the epoch loop so we can see the trajectory of collapsed trials.

**Instructions:**

1.  Import `wandb` inside the function (or at top level).
2.  In the `while run_instance.epoch < max_epochs:` loop, after `epochs_metrics` and `stability_tracker.update()` are computed:
    - Call `wandb.log({...})`.
    - Log:
      - `loss_actor`, `loss_critic`
      - `return_test`, `return_train`
      - `entropy_coef` (CRITICAL for safe tuning)
      - `stability_score` (from `stability_tracker.compute_stability_score()`)
      - `epoch`: `run_instance.epoch`
      - `trial_number`: `trial.number`
3.  Ensure this logging happens _before_ any potential `optuna.TrialPruned` exception, so we capture the data point that caused the pruning.

**Snippet:**

```python
# After stability_tracker.update(...)

stability_score = stability_tracker.compute_stability_score()
wandb.log({
    "epoch": run_instance.epoch,
    "trial_number": trial.number,
    "return_test": return_test,
    "return_train": return_train,
    "entropy_coef": entropy_coef,
    "loss_actor": loss_actor,
    "loss_critic": loss_critic,
    "stability_score": stability_score,
})
```

### Step 11: Finalize Handoff

- Once implemented, change `state.md` status to `APPROVED`.

### Step 12: Fix Interface Mismatch in `custom_models.py`

**Problem:** `DroQSACAgent` (in `custom_algorithms.py`) assumes all actors are contextual (unpacks 3 values from `forward_features`, passes 2 args to `actor_from_features`). The vanilla `DroQHybridActorCritic` currently only returns 1 value and takes 1 arg, causing crashes (`ValueError: not enough values to unpack`).

**Instructions:**

- **File:** `tmrl/custom/custom_models.py`

1.  **Update `SharedBackboneHybridActorCritic.forward_features`** (line ~1005):

    ```python
    def forward_features(self, obs):
        # ... existing extraction ...
        features = self._actor_fusion_norm(combined)
        return features, None, None  # Return dummy film/z for compatibility
    ```

2.  **Update `SharedBackboneHybridActorCritic.actor_from_features`** (line ~1016):

    ```python
    def actor_from_features(self, features, film=None, test=False, with_logprob=True):
        # Added film=None arg. Ignore it.
        # ... rest of function ...
    ```

3.  **Update `DroQHybridActorCritic.forward_features`** (line ~1116):

    ```python
    def forward_features(self, obs):
        # ... existing extraction ...
        features = self._actor_fusion_norm(combined)
        return features, None, None
    ```

4.  **Update `DroQHybridActorCritic.actor_from_features`** (line ~1127):

### Step 13: Fix WandB Authentication

**Problem:** `hyperparameter_sweep_optuna.py` crashes with `OSError: WANDB_API_KEY is not set` because `tmrl` ignores the config key by default, and the environment variable is missing.

**Instructions:**

- **File:** `tmrl/hyperparameter_sweep_optuna.py`
- **Function:** `run_hyperparameter_search` (line ~485)

1.  **Add Authentication Logic:** Before initializing `WeightsAndBiasesCallback`, add code to load the key from `config.json`:
    ```python
    # Ensure WandB is authenticated
    if "WANDB_API_KEY" not in os.environ:
        config_path = Path.home() / "TmrlData" / "config" / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg_json = json.load(f)
                    if "WANDB_KEY" in cfg_json:
                        print(f"Setting WANDB_API_KEY from {config_path}")
                        os.environ["WANDB_API_KEY"] = cfg_json["WANDB_KEY"]
                        wandb.login(key=cfg_json["WANDB_KEY"])
            except Exception as e:
                print(f"Warning: Failed to load WANDB_KEY from config: {e}")
    ```

**Final Handoff:**

- Review all steps (10-13) and ensure they are implemented.
- Run `python hyperparameter_sweep_optuna.py --test-mode` to verify.
