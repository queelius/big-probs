"""Token-triggering filters and intra-thought high-entropy masking.

Implements all 4 Track 1 (§5.4) token filters:
- ``T0_all``: all valid prediction positions
- ``T1_entropy``: top-p highest predictive entropy positions under p_{theta_ref}
- ``T2_self``: strictly teacher-free online filter selecting top-p positions by
  within-group reward std (sigma_g > 0) across G sampled thoughts
- ``T2_scale``: upper-bound cross-scale probability gap
  ``Delta^scale_t = log p_{ref-large}(x_t | x_{<t}) - log p_{small}(x_t | x_{<t})``

Also provides intra-thought top-p high-entropy token gradient masking
(Wang et al., arXiv:2506.01939).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor

_FILTER_ALIASES = {
    "T0": "T0_all",
    "T0_all": "T0_all",
    "T1": "T1_entropy",
    "T1_entropy": "T1_entropy",
    "T2-self": "T2_self",
    "T2_self": "T2_self",
    "T2-scale": "T2_scale",
    "T2_scale": "T2_scale",
}


@torch.no_grad()
def compute_predictive_entropy(logits: Tensor) -> Tensor:
    """Compute Shannon entropy H(p(. | x_{<t})) in nats from logits ``[..., V]``."""
    log_probs = F.log_softmax(logits.detach().float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.clamp(min=0.0)


@torch.no_grad()
def compute_self_variance_scores(group_rewards: Tensor) -> Tensor:
    """Compute strictly teacher-free within-group reward std ``sigma_g`` over dim -1.

    Args:
        group_rewards: Tensor of shape ``[..., G]`` containing rewards for ``G``
            sampled thoughts per position.

    Returns:
        Tensor of shape ``[...]`` containing population std across the ``G`` thoughts.
    """
    if group_rewards.ndim < 1 or group_rewards.shape[-1] < 2:
        raise ValueError(
            f"group_rewards must have group dim >= 2 at last axis, got {tuple(group_rewards.shape)}"
        )
    return group_rewards.detach().float().std(dim=-1, unbiased=False)


@torch.no_grad()
def compute_cross_scale_gap(
    logits_ref_large: Tensor,
    logits_small: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Compute cross-scale log-probability gap ``Delta^scale_t``.

    .. math::
        \\Delta^{\\text{scale}}_t = \\log p_{\\text{ref-large}}(x_t \\mid x_{<t})
                                    - \\log p_{\\text{small}}(x_t \\mid x_{<t})

    Args:
        logits_ref_large: Logits from a frozen larger reference model ``[..., V]``.
        logits_small: Logits from the small target model ``[..., V]``.
        targets: Ground-truth token IDs ``[...]``.
        ignore_index: Token ID to ignore (assigned ``-inf`` gap).

    Returns:
        Float32 tensor of shape ``[...]``.
    """
    if logits_ref_large.shape != logits_small.shape:
        raise ValueError(
            f"logits_ref_large shape {tuple(logits_ref_large.shape)} must match "
            f"logits_small shape {tuple(logits_small.shape)}"
        )
    if logits_small.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits_small.shape[:-1])} must match "
            f"targets shape {tuple(targets.shape)}"
        )

    valid_mask = targets != ignore_index
    safe_tgt = torch.where(valid_mask, targets.long(), torch.zeros_like(targets, dtype=torch.long))
    logp_large = F.log_softmax(logits_ref_large.detach().float(), dim=-1).gather(
        dim=-1, index=safe_tgt.unsqueeze(-1)
    ).squeeze(-1)
    logp_small = F.log_softmax(logits_small.detach().float(), dim=-1).gather(
        dim=-1, index=safe_tgt.unsqueeze(-1)
    ).squeeze(-1)
    gap = logp_large - logp_small
    if not torch.all(valid_mask):
        gap = torch.where(valid_mask, gap, torch.full_like(gap, float("-inf")))
    return gap


def _select_top_p(
    scores: Tensor,
    valid_mask: Tensor,
    top_p: float,
    min_score: float | None = None,
    per_sequence: bool = False,
) -> Tensor:
    """Select top-p fraction of valid positions by score (optionally requiring score > min_score)."""
    eligible = valid_mask.bool()
    if min_score is not None:
        eligible = eligible & (scores > min_score)

    selected = torch.zeros_like(eligible, dtype=torch.bool)
    if top_p <= 0.0:
        return selected

    if per_sequence and scores.ndim >= 2:
        flat_scores = scores.reshape(-1, scores.shape[-1])
        flat_eligible = eligible.reshape(-1, scores.shape[-1])
        flat_valid = valid_mask.bool().reshape(-1, scores.shape[-1])
        flat_selected = torch.zeros_like(flat_eligible, dtype=torch.bool)
        for row in range(flat_scores.shape[0]):
            n_elig = int(flat_eligible[row].sum().item())
            if n_elig == 0:
                continue
            n_base = int(flat_valid[row].sum().item())
            k = n_elig if top_p >= 1.0 else min(n_elig, max(1, math.ceil(top_p * n_base)))
            row_scores = torch.where(
                flat_eligible[row],
                flat_scores[row].float(),
                torch.full_like(flat_scores[row].float(), float("-inf")),
            )
            top_idx = torch.topk(row_scores, k=k, largest=True).indices
            flat_selected[row, top_idx] = True
        return flat_selected.reshape_as(eligible)

    n_eligible = int(eligible.sum().item())
    if n_eligible == 0:
        return selected

    n_valid = int(valid_mask.bool().sum().item())
    k = n_eligible if top_p >= 1.0 else min(n_eligible, max(1, math.ceil(top_p * n_valid)))

    flat_scores = torch.where(
        eligible.reshape(-1),
        scores.reshape(-1).float(),
        torch.full((scores.numel(),), float("-inf"), device=scores.device, dtype=torch.float32),
    )
    top_idx = torch.topk(flat_scores, k=k, largest=True).indices
    flat_selected = torch.zeros(scores.numel(), device=scores.device, dtype=torch.bool)
    flat_selected[top_idx] = True
    return flat_selected.reshape_as(eligible)


@torch.no_grad()
def select_tokens(
    filter_type: str,
    *,
    top_p_tokens: float = 0.2,
    valid_mask: Tensor | None = None,
    logits_ref: Tensor | None = None,
    group_rewards: Tensor | None = None,
    logits_ref_large: Tensor | None = None,
    logits_small: Tensor | None = None,
    targets: Tensor | None = None,
    std_eps: float = 1e-8,
    per_sequence: bool = False,
) -> Tensor:
    """Select prediction positions for SSL-as-RL updates under Track 1 filters.

    Args:
        filter_type: One of ``"T0_all"``, ``"T1_entropy"``, ``"T2_self"``,
            ``"T2_scale"`` (aliases ``"T0"``, ``"T1"``, ``"T2-self"``, ``"T2-scale"``).
        top_p_tokens: Fraction of valid positions to retain (default ``0.2``).
        valid_mask: Optional boolean mask of valid prediction positions.
        logits_ref: Reference no-thought logits ``[..., V]`` (for ``T1_entropy``).
        group_rewards: Within-group rewards ``[..., G]`` (for ``T2_self``).
            Strictly teacher-free: does not accept or use external teacher logits.
        logits_ref_large: Larger reference model logits ``[..., V]`` (for ``T2_scale``).
        logits_small: Small model logits ``[..., V]`` (for ``T2_scale``).
        targets: Ground-truth token IDs ``[...]`` (for ``T2_scale``).
        std_eps: Minimum within-group std threshold for ``T2_self`` (default ``1e-8``).
        per_sequence: Whether to compute top-p independently per sequence row.

    Returns:
        Boolean tensor (``torch.bool``) indicating selected positions.
    """
    if filter_type not in _FILTER_ALIASES:
        raise ValueError(
            f"Unknown filter_type={filter_type!r}. "
            f"Expected one of {sorted(set(_FILTER_ALIASES.values()))}."
        )
    canonical = _FILTER_ALIASES[filter_type]

    if canonical == "T0_all":
        if valid_mask is not None:
            return valid_mask.bool().clone()
        if logits_ref is not None:
            return torch.ones(logits_ref.shape[:-1], dtype=torch.bool, device=logits_ref.device)
        if group_rewards is not None:
            return torch.ones(group_rewards.shape[:-1], dtype=torch.bool, device=group_rewards.device)
        if targets is not None:
            return torch.ones_like(targets, dtype=torch.bool)
        raise ValueError("T0_all requires at least one of valid_mask, logits_ref, group_rewards, or targets.")

    if canonical == "T1_entropy":
        if logits_ref is None:
            raise ValueError("T1_entropy requires `logits_ref` from p_{theta_ref}(. | x_{<t}).")
        scores = compute_predictive_entropy(logits_ref)
        vm = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
        return _select_top_p(scores, vm, top_p=top_p_tokens, per_sequence=per_sequence)

    if canonical == "T2_self":
        if logits_ref_large is not None:
            raise ValueError(
                "T2_self is strictly teacher-free and must not receive `logits_ref_large`."
            )
        if group_rewards is None:
            raise ValueError("T2_self requires `group_rewards` of shape [..., G].")
        scores = compute_self_variance_scores(group_rewards)
        vm = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
        return _select_top_p(
            scores,
            vm,
            top_p=top_p_tokens,
            min_score=std_eps,
            per_sequence=per_sequence,
        )

    if canonical == "T2_scale":
        if logits_ref_large is None or logits_small is None or targets is None:
            raise ValueError(
                "T2_scale requires `logits_ref_large`, `logits_small`, and `targets`."
            )
        scores = compute_cross_scale_gap(logits_ref_large, logits_small, targets)
        vm = valid_mask.bool() if valid_mask is not None else (targets != -100)
        vm = vm & torch.isfinite(scores)
        return _select_top_p(scores, vm, top_p=top_p_tokens, per_sequence=per_sequence)

    raise ValueError(f"Unhandled filter_type={canonical!r}")  # pragma: no cover


@torch.no_grad()
def select_thought_entropy_mask(
    token_entropies: Tensor,
    top_p: float = 0.2,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Select top-p highest-entropy thought tokens per trajectory (Wang et al. 2506.01939).

    Args:
        token_entropies: Per-token entropy inside each thought ``[..., L]``.
        top_p: Fraction of valid thought tokens per trajectory to unmask for PG
            gradients (default ``0.2``). If ``>= 1.0``, selects all valid tokens.
        valid_mask: Optional boolean mask of valid (non-padded) thought tokens ``[..., L]``.

    Returns:
        Boolean mask of shape ``[..., L]``.
    """
    vm = valid_mask.bool() if valid_mask is not None else torch.ones_like(token_entropies, dtype=torch.bool)
    if top_p <= 0.0:
        return torch.zeros_like(vm, dtype=torch.bool)
    if top_p >= 1.0:
        return vm.clone()

    return _select_top_p(
        token_entropies.detach().float(),
        vm,
        top_p=top_p,
        per_sequence=True,
    )
