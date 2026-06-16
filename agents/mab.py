"""
Epsilon-Greedy Multi-Armed Bandit for bid multiplier optimization.

Arms represent bid multipliers applied to base bids.
Reward signal: ROAS = revenue / cost (higher is better).

State is in-memory — stateless across restarts is fine for MVP.
"""

import random
from typing import Dict, List, Optional


class EpsilonGreedyMAB:
    """Epsilon-Greedy MAB for campaign bid multiplier selection.

    Balances exploration of new campaigns with exploitation of
    well-performing ones. Each campaign has its own set of arm stats.

    Arms are bid multipliers: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    """

    DEFAULT_ARMS: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    def __init__(self, epsilon: float = 0.1, arms: Optional[List[float]] = None):
        """Initialize MAB with exploration rate and arm set.

        Args:
            epsilon: Probability of exploration (random arm selection).
                    0.0 = pure exploitation, 1.0 = pure exploration.
            arms: List of bid multiplier arms. Defaults to DEFAULT_ARMS.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon
        self.arms = arms if arms is not None else list(self.DEFAULT_ARMS)
        # Internal: {campaign_id: {arm: {"count": int, "mean": float}}}
        self._stats: Dict[str, Dict[float, Dict[str, float]]] = {}

    def _init_campaign(self, campaign_id: str) -> None:
        """Initialize stat tracking for a new campaign."""
        if campaign_id not in self._stats:
            self._stats[campaign_id] = {}
            for arm in self.arms:
                self._stats[campaign_id][arm] = {"count": 0, "mean": 0.0}

    def select_arm(self, campaign_id: str) -> float:
        """Select a bid multiplier arm using epsilon-greedy strategy.

        With probability epsilon, picks a random arm (exploration).
        Otherwise, picks the arm with highest mean reward (exploitation).

        Args:
            campaign_id: Unique campaign identifier.

        Returns:
            Selected bid multiplier.
        """
        self._init_campaign(campaign_id)

        if random.random() < self.epsilon:
            # Exploration: pick a random arm
            return random.choice(self.arms)
        else:
            # Exploitation: pick arm with highest mean reward
            stats = self._stats[campaign_id]
            best_arm = self.arms[0]
            best_mean = stats[self.arms[0]]["mean"]

            for arm in self.arms[1:]:
                arm_mean = stats[arm]["mean"]
                # Break ties in favor of higher reward
                if arm_mean > best_mean:
                    best_mean = arm_mean
                    best_arm = arm

            return best_arm

    def update(self, campaign_id: str, arm: float, reward: float) -> None:
        """Update arm stats using incremental mean.

        Args:
            campaign_id: Unique campaign identifier.
            arm: The bid multiplier that was selected.
            reward: Observed reward (ROAS).
        """
        self._init_campaign(campaign_id)

        if arm not in self._stats[campaign_id]:
            raise ValueError(
                f"Unknown arm {arm}. Valid arms: {self.arms}"
            )

        entry = self._stats[campaign_id][arm]
        # Incremental mean update:
        # new_mean = old_mean + (reward - old_mean) / (count + 1)
        entry["mean"] = entry["mean"] + (reward - entry["mean"]) / (entry["count"] + 1)
        entry["count"] += 1

    def get_stats(self, campaign_id: str) -> Dict[float, Dict[str, float]]:
        """Get arm statistics for a campaign (for explainability).

        Args:
            campaign_id: Unique campaign identifier.

        Returns:
            Dict mapping arm -> {"count": int, "mean": float}.
            Returns empty dict if campaign has no data.
        """
        if campaign_id not in self._stats:
            return {}
        # Return a copy to prevent external mutation
        return {
            arm: dict(info) for arm, info in self._stats[campaign_id].items()
        }

    @property
    def campaign_count(self) -> int:
        """Number of campaigns with any data."""
        return len(self._stats)
