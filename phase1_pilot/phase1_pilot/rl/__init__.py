"""SSL-as-RL rewards, token filters, and Dr. GRPO + DAPO Clip-Higher trainer."""

from phase1_pilot.rl.filters import (
    compute_cross_scale_gap,
    compute_predictive_entropy,
    compute_self_variance_scores,
    select_thought_entropy_mask,
    select_tokens,
)
from phase1_pilot.rl.grpo import (
    SSLRLTrainer,
    compute_dapo_pg_loss,
    compute_dr_grpo_advantages,
)
from phase1_pilot.rl.rewards import compute_brier_score, compute_reward

__all__ = [
    "SSLRLTrainer",
    "compute_brier_score",
    "compute_cross_scale_gap",
    "compute_dapo_pg_loss",
    "compute_dr_grpo_advantages",
    "compute_predictive_entropy",
    "compute_reward",
    "compute_self_variance_scores",
    "select_thought_entropy_mask",
    "select_tokens",
]
