"""
Stability Metrics for Hyperparameter Optimization

Tracks multiple stability metrics during training to evaluate hyperparameter configurations.
Addresses the issues observed in Run_Test_13_Hybrid_Sensor:
- Policy collapse (entropy too low)
- Loss oscillations
- Performance degradation
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class StabilityTracker:
    """
    Tracks multiple stability metrics during training.

    Metrics tracked:
    1. Loss smoothness: Low variance in loss values
    2. Return consistency: Low variance in episode returns
    3. Entropy health: Maintains exploration (not too low or high)
    4. Trend score: Positive performance drift

    Usage:
        tracker = StabilityTracker(window_size=10)

        for epoch in range(max_epochs):
            metrics = train_epoch()
            tracker.update(
                loss_actor=metrics['loss_actor'],
                return_test=metrics['return_test'],
                entropy_coef=metrics['entropy_coef']
            )

            if epoch >= tracker.window_size:
                score = tracker.compute_stability_score()
                if tracker.should_early_stop():
                    break
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Number of recent epochs to consider for stability calculations
        """
        self.window_size = window_size

        self.loss_actor_history = deque(maxlen=window_size * 2)
        self.loss_critic_history = deque(maxlen=window_size * 2)
        self.return_test_history = deque(maxlen=window_size * 2)
        self.return_train_history = deque(maxlen=window_size * 2)
        self.entropy_coef_history = deque(maxlen=window_size * 2)
        self.epoch_history = []

        # Stability thresholds
        self.min_entropy = 0.05
        self.max_entropy = 0.5
        self.optimal_entropy = 0.15
        self.collapse_threshold = -0.35
        self.divergence_threshold = 5.0

    def update(
        self,
        loss_actor: float,
        loss_critic: float,
        return_test: float,
        return_train: float,
        entropy_coef: float,
        epoch: Optional[int] = None,
    ):
        """
        Update tracker with latest metrics.

        Args:
            loss_actor: Actor loss value
            loss_critic: Critic loss value
            return_test: Test episode return
            return_train: Train episode return
            entropy_coef: Entropy coefficient (alpha)
            epoch: Current epoch number
        """
        self.loss_actor_history.append(loss_actor)
        self.loss_critic_history.append(loss_critic)
        self.return_test_history.append(return_test)
        self.return_train_history.append(return_train)
        self.entropy_coef_history.append(entropy_coef)

        if epoch is not None:
            self.epoch_history.append(epoch)

    def compute_loss_smoothness(self) -> float:
        """
        Compute loss smoothness score.

        Lower variance in loss = higher smoothness = better stability.
        Uses normalized variance to handle different scales.

        Returns:
            float: Score between 0 and 1 (higher is better)
        """
        if len(self.loss_actor_history) < self.window_size:
            return 0.5

        recent_losses = list(self.loss_actor_history)[-self.window_size :]

        mean_loss = np.mean(np.abs(recent_losses))
        var_loss = np.var(recent_losses)

        if mean_loss < 1e-6:
            return 1.0

        normalized_var = var_loss / (mean_loss**2)
        smoothness = 1.0 / (1.0 + normalized_var)

        return float(np.clip(smoothness, 0.0, 1.0))

    def compute_return_consistency(self) -> float:
        """
        Compute return consistency score.

        Lower variance in returns = higher consistency = better stability.

        Returns:
            float: Score between 0 and 1 (higher is better)
        """
        if len(self.return_test_history) < self.window_size:
            return 0.5

        recent_returns = list(self.return_test_history)[-self.window_size :]

        mean_return = np.mean(recent_returns)
        var_return = np.var(recent_returns)

        if abs(mean_return) < 1e-6:
            normalized_var = var_return
        else:
            normalized_var = var_return / (mean_return**2)

        consistency = 1.0 / (1.0 + normalized_var)

        return float(np.clip(consistency, 0.0, 1.0))

    def compute_entropy_health(self) -> float:
        """
        Compute entropy coefficient health score.

        Healthy range: 0.08 to 0.25 (optimal around 0.15)
        Penalizes if too low (collapse) or too high (instability).

        Returns:
            float: Score between 0 and 1 (higher is better)
        """
        if len(self.entropy_coef_history) < self.window_size:
            return 0.5

        recent_entropy = list(self.entropy_coef_history)[-self.window_size :]
        avg_entropy = np.mean(recent_entropy)

        if avg_entropy < self.min_entropy:
            health = 0.0
        elif avg_entropy > self.max_entropy:
            health = 0.5
        else:
            distance_from_optimal = abs(avg_entropy - self.optimal_entropy)
            max_distance = max(
                self.optimal_entropy - self.min_entropy,
                self.max_entropy - self.optimal_entropy,
            )
            health = 1.0 - (distance_from_optimal / max_distance)

        return float(np.clip(health, 0.0, 1.0))

    def compute_trend_score(self) -> float:
        """
        Compute performance trend score.

        Rewards positive drift in returns (improvement over time).

        Returns:
            float: Score between 0 and 1 (higher is better)
        """
        if len(self.return_test_history) < 2 * self.window_size:
            return 0.5

        recent = np.mean(list(self.return_test_history)[-self.window_size :])
        previous = np.mean(
            list(self.return_test_history)[-2 * self.window_size : -self.window_size]
        )

        if abs(previous) < 1e-6:
            trend = 0.0
        else:
            trend = (recent - previous) / (abs(previous) + 1e-6)

        trend_score = 0.5 + 0.5 * np.tanh(trend)

        return float(np.clip(trend_score, 0.0, 1.0))

    def compute_performance_score(self) -> float:
        """
        Compute absolute performance score.

        Based on average return value (higher is better).

        Returns:
            float: Score between 0 and 1 (higher is better)
        """
        if len(self.return_test_history) < self.window_size:
            return 0.5

        recent_returns = list(self.return_test_history)[-self.window_size :]
        avg_return = np.mean(recent_returns)

        # Normalize: assume returns range from -1 to 10
        min_return = -1.0
        max_return = 10.0

        normalized = (avg_return - min_return) / (max_return - min_return)

        return float(np.clip(normalized, 0.0, 1.0))

    def compute_stability_score(
        self, weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute composite stability score.

        Weighted combination of all stability metrics.

        Args:
            weights: Custom weights for each metric component.
                    Default weights prioritize loss smoothness and consistency.

        Returns:
            float: Composite score between 0 and 1 (higher is better)
        """
        if weights is None:
            weights = {
                "loss_smoothness": 0.25,
                "return_consistency": 0.25,
                "entropy_health": 0.20,
                "trend_score": 0.15,
                "performance": 0.15,
            }

        components = {
            "loss_smoothness": self.compute_loss_smoothness(),
            "return_consistency": self.compute_return_consistency(),
            "entropy_health": self.compute_entropy_health(),
            "trend_score": self.compute_trend_score(),
            "performance": self.compute_performance_score(),
        }

        composite_score = sum(weights[key] * components[key] for key in weights.keys())

        return float(np.clip(composite_score, 0.0, 1.0))

    def should_early_stop(
        self,
        patience: int = 5,
        stability_threshold: float = 0.15,
        return_threshold: float = -0.35,
        entropy_threshold: float = 0.05,
    ) -> Tuple[bool, str]:
        """
        Determine if training should early stop based on stability metrics.

        Args:
            patience: Number of epochs to wait before stopping
            stability_threshold: Minimum acceptable stability score
            return_threshold: Minimum acceptable average return
            entropy_threshold: Minimum acceptable entropy coefficient

        Returns:
            Tuple[bool, str]: (should_stop, reason)
        """
        if len(self.return_test_history) < patience:
            return False, ""

        recent_returns = list(self.return_test_history)[-patience:]
        recent_entropy = list(self.entropy_coef_history)[-patience:]

        avg_return = np.mean(recent_returns)
        avg_entropy = np.mean(recent_entropy)
        stability_score = self.compute_stability_score()

        # Check for policy collapse (entropy too low)
        if avg_entropy < entropy_threshold:
            return (
                True,
                f"Policy collapse: entropy {avg_entropy:.4f} < {entropy_threshold}",
            )

        # Check for performance collapse
        if avg_return < return_threshold:
            return (
                True,
                f"Performance collapse: return {avg_return:.4f} < {return_threshold}",
            )

        # Check for instability
        if stability_score < stability_threshold:
            return (
                True,
                f"Training unstable: stability score {stability_score:.4f} < {stability_threshold}",
            )

        # Check for severe loss divergence
        if len(self.loss_actor_history) >= patience:
            recent_losses = list(self.loss_actor_history)[-patience:]
            if any(abs(loss) > self.divergence_threshold for loss in recent_losses):
                return True, f"Loss divergence detected"

        return False, ""

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Get comprehensive diagnostics for logging.

        Returns:
            Dict with all metrics and component scores
        """
        diagnostics = {
            # Raw metrics
            "loss_actor_mean": np.mean(list(self.loss_actor_history))
            if self.loss_actor_history
            else 0.0,
            "loss_actor_var": np.var(list(self.loss_actor_history))
            if len(self.loss_actor_history) > 1
            else 0.0,
            "return_test_mean": np.mean(list(self.return_test_history))
            if self.return_test_history
            else 0.0,
            "return_test_var": np.var(list(self.return_test_history))
            if len(self.return_test_history) > 1
            else 0.0,
            "entropy_coef_mean": np.mean(list(self.entropy_coef_history))
            if self.entropy_coef_history
            else 0.0,
            # Component scores
            "loss_smoothness": self.compute_loss_smoothness(),
            "return_consistency": self.compute_return_consistency(),
            "entropy_health": self.compute_entropy_health(),
            "trend_score": self.compute_trend_score(),
            "performance_score": self.compute_performance_score(),
            # Composite score
            "stability_score": self.compute_stability_score(),
        }

        return diagnostics

    def reset(self):
        """Reset all history."""
        self.loss_actor_history.clear()
        self.loss_critic_history.clear()
        self.return_test_history.clear()
        self.return_train_history.clear()
        self.entropy_coef_history.clear()
        self.epoch_history.clear()


class EarlyStoppingManager:
    """
    Manages early stopping logic for hyperparameter trials.

    Combines multiple stopping criteria:
    - Stability-based stopping
    - Patience-based stopping
    - Resource limit stopping
    """

    def __init__(
        self,
        patience_epochs: int = 5,
        stability_threshold: float = 0.15,
        return_threshold: float = -0.35,
        entropy_threshold: float = 0.05,
        min_epochs: int = 3,
        max_epochs: int = 100,
    ):
        """
        Args:
            patience_epochs: Epochs to wait before early stopping
            stability_threshold: Minimum stability score
            return_threshold: Minimum average return
            entropy_threshold: Minimum entropy coefficient
            min_epochs: Minimum epochs before allowing early stop
            max_epochs: Maximum epochs to run
        """
        self.patience_epochs = patience_epochs
        self.stability_threshold = stability_threshold
        self.return_threshold = return_threshold
        self.entropy_threshold = entropy_threshold
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

        self.tracker = StabilityTracker(window_size=patience_epochs)
        self.current_epoch = 0
        self.best_score = 0.0
        self.epochs_without_improvement = 0

    def update(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Update with new epoch metrics and check if should stop.

        Args:
            metrics: Dictionary containing loss_actor, return_test, entropy_coef

        Returns:
            Tuple[bool, str]: (should_stop, reason)
        """
        self.current_epoch += 1

        self.tracker.update(
            loss_actor=metrics.get("loss_actor", 0.0),
            loss_critic=metrics.get("loss_critic", 0.0),
            return_test=metrics.get("return_test", 0.0),
            return_train=metrics.get("return_train", 0.0),
            entropy_coef=metrics.get("entropy_coef", 0.2),
            epoch=self.current_epoch,
        )

        if self.current_epoch >= self.tracker.window_size:
            current_score = self.tracker.compute_stability_score()

            if current_score > self.best_score:
                self.best_score = current_score
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        if self.current_epoch >= self.max_epochs:
            return True, f"Reached max epochs ({self.max_epochs})"

        if self.current_epoch < self.min_epochs:
            return False, ""

        should_stop, reason = self.tracker.should_early_stop(
            patience=self.patience_epochs,
            stability_threshold=self.stability_threshold,
            return_threshold=self.return_threshold,
            entropy_threshold=self.entropy_threshold,
        )

        if should_stop:
            return True, reason

        if self.epochs_without_improvement >= self.patience_epochs * 2:
            return True, f"No improvement for {self.epochs_without_improvement} epochs"

        return False, ""

    def get_final_score(self) -> float:
        """Get final stability score for this run."""
        return self.tracker.compute_stability_score()

    def get_diagnostics(self) -> Dict[str, float]:
        """Get current diagnostics."""
        return self.tracker.get_diagnostics()


if __name__ == "__main__":
    print("Testing StabilityTracker...")

    tracker = StabilityTracker(window_size=5)

    test_metrics = [
        {
            "loss_actor": 1.0,
            "loss_critic": 0.5,
            "return_test": -0.4,
            "return_train": -0.4,
            "entropy_coef": 0.2,
        },
        {
            "loss_actor": 0.8,
            "loss_critic": 0.4,
            "return_test": 0.5,
            "return_train": 0.3,
            "entropy_coef": 0.19,
        },
        {
            "loss_actor": 0.6,
            "loss_critic": 0.3,
            "return_test": 1.2,
            "return_train": 1.0,
            "entropy_coef": 0.18,
        },
        {
            "loss_actor": 0.5,
            "loss_critic": 0.25,
            "return_test": 2.0,
            "return_train": 1.8,
            "entropy_coef": 0.17,
        },
        {
            "loss_actor": 0.4,
            "loss_critic": 0.2,
            "return_test": 3.0,
            "return_train": 2.5,
            "entropy_coef": 0.16,
        },
        {
            "loss_actor": 0.3,
            "loss_critic": 0.15,
            "return_test": 4.0,
            "return_train": 3.5,
            "entropy_coef": 0.15,
        },
        {
            "loss_actor": 0.25,
            "loss_critic": 0.12,
            "return_test": 5.0,
            "return_train": 4.5,
            "entropy_coef": 0.14,
        },
    ]

    for i, metrics in enumerate(test_metrics):
        tracker.update(
            loss_actor=metrics["loss_actor"],
            loss_critic=metrics["loss_critic"],
            return_test=metrics["return_test"],
            return_train=metrics["return_train"],
            entropy_coef=metrics["entropy_coef"],
            epoch=i,
        )

        if i >= tracker.window_size:
            score = tracker.compute_stability_score()
            should_stop, reason = tracker.should_early_stop()

            print(f"\nEpoch {i}:")
            print(f"  Stability Score: {score:.4f}")
            print(f"  Should Stop: {should_stop}")
            if reason:
                print(f"  Reason: {reason}")

            diagnostics = tracker.get_diagnostics()
            print(f"  Diagnostics:")
            for key, value in diagnostics.items():
                print(f"    {key}: {value:.4f}")

    print("\n✓ StabilityTracker test completed")
