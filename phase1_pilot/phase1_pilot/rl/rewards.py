"""Corpus-native reward functions for SSL-as-RL with frozen-reference anchoring.

Implements Track 1 (01-ssl-as-rl-latent-cot.md §5.4):
- Exact O(|V|) bounded strictly proper Brier score in [-2, 0]
- R1_exact: binary exact match averaged over m future tokens
- R2_log: debiased, clipped multi-token log-likelihood gain against frozen ref
- R3_brier: debiased multi-token Brier-score gain against frozen ref

CRITICAL ANTI-SABOTAGE INVARIANT:
All outputs of ``compute_reward`` are strictly detached (computed under
``torch.no_grad()``) and compare the thought-conditioned policy against a
frozen / EMA reference no-thought anchor ``p_{theta_ref}`` so gradients never
flow through the reward scalar or sabotage the live no-thought baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_REWARD_ALIASES = {
    "R1": "R1_exact",
    "R1_exact": "R1_exact",
    "R2": "R2_log",
    "R2_log": "R2_log",
    "R3": "R3_brier",
    "R3_brier": "R3_brier",
}


def compute_brier_score(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Compute exact bounded strictly proper Brier score in O(|V|).

    .. math::
        \\text{BS}(q, v^*) = 2 q(v^*) - \\sum_{v \\in \\mathcal{V}} q(v)^2 - 1 \\in [-2, 0]

    Equivalently:
    ``-(1 - q(v*))^2 - sum_{v != v*} q(v)^2``.
    Its expectation under ``v* ~ p_true`` is uniquely maximized at ``q = p_true``.

    Args:
        logits: Unnormalized log-probabilities of shape ``[..., V]``.
        targets: Integer ground-truth token IDs of shape ``[...]`` matching
            ``logits.shape[:-1]``. Positions equal to ``ignore_index`` receive 0.0.
        ignore_index: Target index to ignore (default ``-100``).

    Returns:
        Tensor of shape ``[...]`` in ``[-2.0, 0.0]`` (float32).
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits batch shape {tuple(logits.shape[:-1])} must match "
            f"targets shape {tuple(targets.shape)}"
        )

    probs = torch.softmax(logits.float(), dim=-1)
    valid_mask = targets != ignore_index
    safe_targets = torch.where(valid_mask, targets.long(), torch.zeros_like(targets, dtype=torch.long))
    target_probs = probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    sum_sq = (probs * probs).sum(dim=-1)
    brier = 2.0 * target_probs - sum_sq - 1.0
    brier = brier.clamp(min=-2.0, max=0.0)

    if not torch.all(valid_mask):
        brier = torch.where(valid_mask, brier, torch.zeros_like(brier))
    return brier


def _broadcast_for_reward(
    logits_thought: Tensor,
    logits_ref_nothought: Tensor | None,
    targets: Tensor,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor | None]:
    """Align shapes so ``[B, G, m, V]`` broadcasts cleanly against ``[B, m, V]`` / ``[B, m]``."""
    target_batch_shape = logits_thought.shape[:-1]

    if logits_ref_nothought is not None:
        if logits_ref_nothought.ndim == logits_thought.ndim - 1 and logits_thought.ndim == 4:
            # [B, m, V] -> [B, 1, m, V]
            logits_ref_nothought = logits_ref_nothought.unsqueeze(1)
        logits_ref_nothought = logits_ref_nothought.expand_as(logits_thought)

    if targets.ndim == len(target_batch_shape) - 1 and len(target_batch_shape) == 3:
        # [B, m] -> [B, 1, m]
        targets = targets.unsqueeze(1)
    targets = targets.expand(target_batch_shape)

    if mask is not None:
        if mask.ndim == len(target_batch_shape) - 1 and len(target_batch_shape) == 3:
            mask = mask.unsqueeze(1)
        mask = mask.expand(target_batch_shape)

    return logits_thought, logits_ref_nothought, targets, mask


def _reduce_horizon(
    per_token: Tensor,
    valid_mask: Tensor,
    has_horizon_dim: bool,
) -> Tensor:
    """Average per-token rewards over the m-token continuation horizon."""
    if not has_horizon_dim:
        return torch.where(valid_mask, per_token, torch.zeros_like(per_token))

    weights = valid_mask.float()
    denom = weights.sum(dim=-1).clamp(min=1.0)
    return (per_token * weights).sum(dim=-1) / denom


@torch.no_grad()
def compute_reward(
    logits_thought: Tensor,
    logits_ref_nothought: Tensor | None,
    targets: Tensor,
    reward_type: str = "R3_brier",
    tau: float = 5.0,
    mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Compute detached SSL-as-RL reward for thought-conditioned predictions.

    Supports input shapes:
    - ``[B, V]`` with ``targets: [B]`` (single token, returns ``[B]``)
    - ``[B, m, V]`` with ``targets: [B, m]`` (averages over ``m``, returns ``[B]``)
    - ``[B, G, m, V]`` with ``targets: [B, G, m]`` or ``[B, m]`` (averages over ``m``,
      returns ``[B, G]``)

    Args:
        logits_thought: Logits from thought-conditioned forward
            ``p_theta(. | x_{<t}, z_t, x_{t:t+k-1})``.
        logits_ref_nothought: Logits from the **frozen/EMA reference anchor**
            ``p_{theta_ref}(. | x_{<t}, x_{t:t+k-1})``. Required for ``R2_log``
            and ``R3_brier``; unused for ``R1_exact``.
        targets: Ground-truth continuation token IDs.
        reward_type: One of ``"R1_exact"``, ``"R2_log"``, ``"R3_brier"`` (or
            short aliases ``"R1"``, ``"R2"``, ``"R3"``).
        tau: Symmetric clipping threshold for ``R2_log`` (default ``5.0``).
        mask: Optional boolean/float mask of valid target positions.
        ignore_index: Target token ID to exclude from averaging (default ``-100``).

    Returns:
        Detached float32 reward tensor (``requires_grad == False``).
    """
    if reward_type not in _REWARD_ALIASES:
        raise ValueError(
            f"Unknown reward_type={reward_type!r}. "
            f"Expected one of {sorted(set(_REWARD_ALIASES.values()))}."
        )
    canonical = _REWARD_ALIASES[reward_type]

    # Explicit stop-gradient on all inputs to enforce the anti-sabotage invariant.
    lt = logits_thought.detach()
    lr = logits_ref_nothought.detach() if logits_ref_nothought is not None else None
    tgt = targets.detach()
    msk = mask.detach() if mask is not None else None

    if canonical in ("R2_log", "R3_brier") and lr is None:
        raise ValueError(
            f"reward_type={canonical!r} requires frozen-reference logits "
            "`logits_ref_nothought` for anti-sabotage debiasing."
        )

    lt, lr, tgt, msk = _broadcast_for_reward(lt, lr, tgt, msk)

    valid_mask = tgt != ignore_index
    if msk is not None:
        valid_mask = valid_mask & msk.bool()

    safe_tgt = torch.where(valid_mask, tgt.long(), torch.zeros_like(tgt, dtype=torch.long))
    has_horizon_dim = lt.ndim >= 3

    if canonical == "R1_exact":
        preds = lt.argmax(dim=-1)
        per_token = (preds == safe_tgt).float()
    elif canonical == "R2_log":
        assert lr is not None
        logp_thought = F.log_softmax(lt.float(), dim=-1).gather(
            dim=-1, index=safe_tgt.unsqueeze(-1)
        ).squeeze(-1)
        logp_ref = F.log_softmax(lr.float(), dim=-1).gather(
            dim=-1, index=safe_tgt.unsqueeze(-1)
        ).squeeze(-1)
        per_token = torch.clamp(logp_thought - logp_ref, min=-float(tau), max=float(tau))
    elif canonical == "R3_brier":
        assert lr is not None
        bs_thought = compute_brier_score(lt, tgt, ignore_index=ignore_index)
        bs_ref = compute_brier_score(lr, tgt, ignore_index=ignore_index)
        per_token = bs_thought - bs_ref
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unhandled reward_type={canonical!r}")

    reward = _reduce_horizon(per_token, valid_mask, has_horizon_dim=has_horizon_dim)
    return reward.detach()
