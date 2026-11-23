"""Dr. GRPO + DAPO Clip-Higher trainer for SSL-as-RL (Track 1 §5.4).

Implements:
- Deep-copied, strictly frozen reference anchor ``p_{theta_ref}`` (optional slow
  EMA only when ``ref_ema_decay < 1.0``; default ``1.0`` = strictly frozen).
- Dr. GRPO group-centered advantages ``A_i = R_i - mean(R)`` WITHOUT dividing by
  ``sigma_R`` and normalized by fixed ``max_thought_len`` (eliminating difficulty
  and length-inflation biases; Liu et al., arXiv:2503.20783).
- DAPO Clip-Higher asymmetric PPO clipping (``clip_low=0.2, clip_high=0.28``;
  Yu et al., arXiv:2503.14476).
- Dynamic ``sigma = 0`` group filtering (skip zero-variance groups and report
  ``zero_var_frac``).
- Intra-thought top-p high-entropy token gradient masking (Wang et al.,
  arXiv:2506.01939).
- Explicit auxiliary standard no-thought NTP loss ``lambda_ntp * L_NTP`` plus
  configurable coupled continuation cross-entropy loss on target tokens.
- Native integration with ``DiscreteThoughtHelper`` (``I1_discrete``) and
  ``ContinuousLatentWrapper`` HRPO-style gated rollouts (``I2_hybrid``).
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from phase1_pilot.config import RLConfig
from phase1_pilot.models import ContinuousLatentWrapper, DiscreteThoughtHelper
from phase1_pilot.rl.filters import (
    compute_self_variance_scores,
    select_thought_entropy_mask,
    select_tokens,
)
from phase1_pilot.rl.rewards import compute_reward


def compute_dr_grpo_advantages(rewards: Tensor) -> Tensor:
    """Compute Dr. GRPO group-centered advantages WITHOUT std-dev division.

    .. math::
        A_{b, g} = R_{b, g} - \\frac{1}{G} \\sum_{g'=1}^{G} R_{b, g'}

    Args:
        rewards: Detached reward tensor of shape ``[..., G]``.

    Returns:
        Detached advantage tensor of shape ``[..., G]``.
    """
    r = rewards.detach().float()
    return r - r.mean(dim=-1, keepdim=True)


def compute_dapo_pg_loss(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    token_mask: Tensor,
    max_thought_len: int,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    n_active_groups: int | None = None,
) -> tuple[Tensor, float]:
    """Compute Dr. GRPO length-debiased loss with DAPO Clip-Higher.

    Normalizes summed per-token surrogate objectives by the fixed constant
    ``n_active_groups * G * max_thought_len`` rather than per-sequence realized
    length so trajectories cannot inflate length to dilute negative advantages.

    Args:
        new_logprobs: Current policy log-probs ``[B, G, L]`` (requires_grad=True).
        old_logprobs: Behavior policy log-probs ``[B, G, L]`` (detached).
        advantages: Group-centered advantages ``[B, G]`` (detached).
        token_mask: Boolean mask ``[B, G, L]`` of unmasked active thought tokens.
        max_thought_len: Fixed max thought length ``L_max`` for length-debiased norm.
        clip_low: Lower PPO clip epsilon (default ``0.2``).
        clip_high: Asymmetric upper PPO clip epsilon (default ``0.28``).
        n_active_groups: Number of active groups (defaults to count of rows with
            any active token in ``token_mask``).

    Returns:
        Tuple ``(loss_pg, clip_frac)``.
    """
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"new_logprobs shape {tuple(new_logprobs.shape)} != "
            f"old_logprobs shape {tuple(old_logprobs.shape)}"
        )

    mask_bool = token_mask.bool()
    mask_f = mask_bool.float()
    active_tokens = mask_f.sum()

    if int(active_tokens.item()) == 0:
        zero = new_logprobs.sum() * 0.0
        return zero, 0.0

    log_ratio = (new_logprobs.float() - old_logprobs.detach().float()).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    adv = advantages.detach().float().unsqueeze(-1)

    clipped_ratio = torch.clamp(ratio, min=1.0 - float(clip_low), max=1.0 + float(clip_high))
    surr1 = ratio * adv
    surr2 = clipped_ratio * adv
    pg_obj = torch.minimum(surr1, surr2)

    group_size = new_logprobs.shape[1]
    if n_active_groups is None:
        active_groups_mask = mask_bool.any(dim=(1, 2))
        n_active_groups = max(1, int(active_groups_mask.sum().item()))
    else:
        n_active_groups = max(1, int(n_active_groups))

    fixed_denom = float(n_active_groups * group_size * max(1, int(max_thought_len)))
    loss_pg = -(pg_obj * mask_f).sum() / fixed_denom

    is_clipped = ((ratio < 1.0 - float(clip_low)) | (ratio > 1.0 + float(clip_high))) & mask_bool
    clip_frac = float((is_clipped.float().sum() / active_tokens.clamp(min=1.0)).item())
    return loss_pg, clip_frac


def _all_reduce_model_grads(model: nn.Module) -> bool:
    """Average parameter gradients across ranks when torch.distributed is active.

    Uses ``SUM`` + divide by ``world_size`` for portability across both NCCL
    (CUDA) and Gloo (CPU) backends, and materializes zero-grads for any
    trainable parameter without a local gradient to prevent cross-rank deadlock
    when one rank has zero active RL groups.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return False
    world_size = dist.get_world_size()
    if world_size <= 1:
        return False
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(float(world_size))
    return True


def _broadcast_model_state(model: nn.Module, src: int = 0) -> None:
    """Broadcast parameters and buffers from ``src`` rank when distributed is active."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src)
        for buf in model.buffers():
            dist.broadcast(buf.data, src=src)


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _unwrap_base_lm(model: nn.Module) -> nn.Module:
    m = _unwrap_module(model)
    return m.base_lm if hasattr(m, "base_lm") else m


def _extract_logits(output: Any) -> Tensor:
    """Extract logits tensor from LMOutput dataclass, tuple, or raw Tensor."""
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)}")


class SSLRLTrainer:
    """Dr. GRPO + DAPO Clip-Higher trainer for SSL-as-RL with frozen anchor.

    Supports:
    - ``R1_exact``, ``R2_log``, ``R3_brier`` rewards anchored to frozen ``ref_model``
    - ``T0_all``, ``T1_entropy``, ``T2_self`` (teacher-free), ``T2_scale`` filters
    - ``I1_discrete`` (via ``DiscreteThoughtHelper``) and ``I2_hybrid`` (via
      ``ContinuousLatentWrapper`` HRPO-style gated rollouts)
    """

    def __init__(
        self,
        model: nn.Module,
        rl_cfg: RLConfig,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        ref_model: nn.Module | None = None,
        teacher_model: nn.Module | None = None,
        pad_id: int = 0,
        think_start_id: int = 3,
        think_end_id: int = 4,
        grad_clip: float = 1.0,
        temperature: float = 1.0,
        continuation_sft_weight: float = 0.1,
        ppo_epochs: int = 1,
        bf16: bool = False,
        device: torch.device | str | None = None,
        std_eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.rl_cfg = rl_cfg
        self.pad_id = int(pad_id)
        self.think_start_id = int(think_start_id)
        self.think_end_id = int(think_end_id)
        self.grad_clip = float(grad_clip)
        self.temperature = float(temperature)
        self.continuation_sft_weight = float(continuation_sft_weight)
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.std_eps = float(std_eps)

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            self.model.to(self.device)

        self.bf16 = bool(bf16) and self.device.type == "cuda"

        self.thought_helper = DiscreteThoughtHelper(
            think_start_id=self.think_start_id,
            think_end_id=self.think_end_id,
            pad_id=self.pad_id,
        )

        # Ensure identical starting weights across distributed ranks before freezing ref_model
        _broadcast_model_state(self.model, src=0)

        # Deep-copy and strictly freeze the reference anchor p_{theta_ref}
        src_ref = _unwrap_module(self.model)
        self.ref_model = copy.deepcopy(src_ref if ref_model is None else ref_model).to(self.device)
        if ref_model is not None:
            _broadcast_model_state(self.ref_model, src=0)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Optional frozen larger teacher (ONLY for T2_scale upper-bound ablation)
        self.teacher_model: nn.Module | None = None
        if teacher_model is not None:
            if rl_cfg.filter_type == "T2_self":
                raise ValueError(
                    "T2_self is strictly teacher-free and must not be initialized with teacher_model."
                )
            self.teacher_model = teacher_model.to(self.device)
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=rl_cfg.lr,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer

        self.step_count: int = 0

    @property
    def _is_hybrid_interface(self) -> bool:
        return (
            self.rl_cfg.interface == "I2_hybrid"
            and isinstance(_unwrap_module(self.model), ContinuousLatentWrapper)
        )

    def _autocast_ctx(self):
        if self.bf16:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def _maybe_update_ref(self) -> None:
        """Slowly EMA-update ref_model ONLY if ``ref_ema_decay < 1.0``.

        Default ``ref_ema_decay == 1.0`` keeps ``ref_model`` strictly frozen to
        the pre-RL checkpoint, preventing baseline sabotage.
        """
        decay = float(self.rl_cfg.ref_ema_decay)
        if decay >= 1.0:
            return
        src = _unwrap_module(self.model)
        for ref_p, live_p in zip(self.ref_model.parameters(), src.parameters()):
            ref_p.data.mul_(decay).add_(live_p.data, alpha=1.0 - decay)
            ref_p.requires_grad_(False)

    def _parse_batch(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """Normalize incoming batch dict into tensors on ``self.device``.

        Returns:
            ``(prefix_ids, prefix_mask, target_ids, target_mask, full_ids, full_labels, full_attn_mask)``
        """
        m = int(self.rl_cfg.n_ahead)

        if "prefix_ids" in batch and "target_ids" in batch:
            prefix_ids = batch["prefix_ids"].to(self.device).long()
            prefix_mask = (
                batch["prefix_mask"].to(self.device).long()
                if "prefix_mask" in batch
                else (prefix_ids != self.pad_id).long()
            )
            target_ids = batch["target_ids"].to(self.device).long()
            if target_ids.ndim == 1:
                target_ids = target_ids.unsqueeze(-1)
            target_ids = target_ids[:, :m]
            target_mask = (target_ids != -100) & (target_ids != self.pad_id)
            if "input_ids" in batch:
                full_ids = batch["input_ids"].to(self.device).long()
            else:
                safe_t = torch.where(
                    target_mask, target_ids, torch.full_like(target_ids, self.pad_id)
                )
                full_ids = torch.cat([prefix_ids, safe_t], dim=1)
            if "labels" in batch:
                full_labels = batch["labels"].to(self.device).long()
            else:
                full_labels = full_ids.clone()
                full_labels[full_labels == self.pad_id] = -100
            full_attn = (
                batch["attention_mask"].to(self.device).long()
                if "attention_mask" in batch
                else (full_ids != self.pad_id).long()
            )
            return prefix_ids, prefix_mask, target_ids, target_mask, full_ids, full_labels, full_attn

        input_ids = batch["input_ids"].to(self.device).long()
        B, T = input_ids.shape
        full_attn = (
            batch["attention_mask"].to(self.device).long()
            if "attention_mask" in batch
            else (input_ids != self.pad_id).long()
        )
        if "labels" in batch:
            full_labels = batch["labels"].to(self.device).long()
        else:
            full_labels = input_ids.clone()
            full_labels[full_labels == self.pad_id] = -100

        prompt_lens_raw = batch.get("prompt_lens", batch.get("prompt_len", None))
        if prompt_lens_raw is None:
            split = max(1, T - m)
            prompt_lens = torch.full((B,), split, device=self.device, dtype=torch.long)
        elif isinstance(prompt_lens_raw, Tensor):
            prompt_lens = prompt_lens_raw.to(self.device).long()
        else:
            prompt_lens = torch.tensor(prompt_lens_raw, device=self.device, dtype=torch.long)

        # Left-pad prefixes so every prompt ends cleanly at column max_p - 1
        max_p = max(1, min(int(prompt_lens.max().item()), T - 1))
        prefix_ids = torch.full((B, max_p), self.pad_id, device=self.device, dtype=torch.long)
        prefix_mask = torch.zeros((B, max_p), device=self.device, dtype=torch.long)
        target_ids = torch.full((B, m), -100, device=self.device, dtype=torch.long)

        for b in range(B):
            pl = max(1, min(int(prompt_lens[b].item()), T - 1))
            prefix_ids[b, max_p - pl :] = input_ids[b, :pl]
            prefix_mask[b, max_p - pl :] = full_attn[b, :pl]
            avail = min(m, T - pl)
            if avail > 0:
                tgt_slice = input_ids[b, pl : pl + avail]
                tgt_slice = torch.where(
                    tgt_slice == self.pad_id,
                    torch.full_like(tgt_slice, -100),
                    tgt_slice,
                )
                target_ids[b, :avail] = tgt_slice

        target_mask = target_ids != -100
        return prefix_ids, prefix_mask, target_ids, target_mask, input_ids, full_labels, full_attn

    def _compute_thought_valid_mask(self, thought_ids: Tensor) -> Tensor:
        """Mask tokens strictly after the first ``</think>`` (if emitted inside thought)."""
        B, G, L = thought_ids.shape
        is_end = thought_ids == self.think_end_id
        # Include the </think> token itself if generated, mask subsequent tokens
        seen_end_before = torch.cumsum(is_end.long(), dim=-1) > is_end.long()
        return ~seen_end_before

    @torch.no_grad()
    def _sample_thoughts(
        self,
        prefix_ids: Tensor,
        prefix_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample G thoughts per prefix using DiscreteThoughtHelper or ContinuousLatentWrapper."""
        was_training = self.model.training
        self.model.eval()
        try:
            if self._is_hybrid_interface:
                wrapper: ContinuousLatentWrapper = _unwrap_module(self.model)  # type: ignore[assignment]
                out = wrapper.rollout_hybrid(
                    prefix_ids=prefix_ids,
                    group_size=self.rl_cfg.group_size,
                    n_steps=self.rl_cfg.thought_len,
                    temperature=self.temperature,
                    prefix_mask=prefix_mask,
                )
                valid_mask = torch.ones_like(out.thought_ids, dtype=torch.bool)
                return out.thought_ids, out.old_logprobs, out.token_entropies, valid_mask

            rollout = self.thought_helper.sample_thoughts(
                model=self.model,
                prefix_ids=prefix_ids,
                thought_len=self.rl_cfg.thought_len,
                group_size=self.rl_cfg.group_size,
                temperature=self.temperature,
                prefix_mask=prefix_mask,
            )
            valid_mask = self._compute_thought_valid_mask(rollout.thought_ids)
            return rollout.thought_ids, rollout.old_logprobs, rollout.token_entropies, valid_mask
        finally:
            if was_training:
                self.model.train()

    def _recompute_thought_and_continuation(
        self,
        prefix_ids: Tensor,
        prefix_mask: Tensor,
        thought_ids: Tensor,
        safe_targets: Tensor,
        need_continuation: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Differentiably compute ``(new_logprobs [B, G, L], cont_logits [B, G, m, V] | None)``."""
        if self._is_hybrid_interface:
            wrapper: ContinuousLatentWrapper = _unwrap_module(self.model)  # type: ignore[assignment]
            new_lp, _new_ent, cont_logits = wrapper.recompute_hybrid_logprobs_and_continuation(
                prefix_ids=prefix_ids,
                thought_ids=thought_ids,
                target_ids=safe_targets,
                temperature=self.temperature,
                prefix_mask=prefix_mask,
            )
            return new_lp, cont_logits

        new_lp, _new_ent = self.thought_helper.recompute_thought_logprobs(
            model=self.model,
            prefix_ids=prefix_ids,
            thought_ids=thought_ids,
            temperature=self.temperature,
            prefix_mask=prefix_mask,
        )
        if not need_continuation:
            return new_lp, None
        cont = self.thought_helper.evaluate_continuation_logits(
            model=self.model,
            prefix_ids=prefix_ids,
            thought_ids=thought_ids,
            target_ids=safe_targets,
            prefix_mask=prefix_mask,
            ref_model=self.ref_model,
        )
        return new_lp, cont.with_thought

    def _compute_ntp_loss(
        self,
        full_ids: Tensor,
        full_labels: Tensor,
        full_attn: Tensor | None,
    ) -> Tensor:
        """Compute explicit auxiliary standard no-thought NTP cross-entropy loss."""
        base_lm = _unwrap_base_lm(self.model)
        out = base_lm(input_ids=full_ids, attention_mask=full_attn, labels=full_labels)
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        logits = _extract_logits(out)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = full_labels[:, 1:].contiguous()
        if not torch.any(shift_labels != -100):
            return shift_logits.sum() * 0.0
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _compute_target_nll(logits: Tensor, target_ids: Tensor) -> float:
        """Compute mean NLL (nats) on valid continuation target tokens."""
        valid = target_ids != -100
        if not torch.any(valid):
            return 0.0
        loss = F.cross_entropy(
            logits.detach().float().reshape(-1, logits.shape[-1]),
            target_ids.detach().reshape(-1),
            ignore_index=-100,
        )
        return float(loss.item())

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute one full Dr. GRPO + DAPO Clip-Higher SSL-as-RL optimization step.

        Returns stability telemetry dict containing:
            ``loss_total``, ``loss_pg``, ``loss_ntp``, ``loss_cont``,
            ``mean_reward``, ``reward_std``, ``policy_entropy``,
            ``zero_var_frac``, ``active_group_frac``, ``nothought_nll``,
            ``ref_nothought_nll``, ``nothought_nll_drift``,
            ``mean_thought_len``, ``clip_frac``.
        """
        self.model.train()
        (
            prefix_ids,
            prefix_mask,
            target_ids,
            target_mask,
            full_ids,
            full_labels,
            full_attn,
        ) = self._parse_batch(batch)

        B = prefix_ids.shape[0]
        G = int(self.rl_cfg.group_size)
        L = int(self.rl_cfg.thought_len)
        safe_targets = torch.where(
            target_ids >= 0, target_ids, torch.full_like(target_ids, self.pad_id)
        )

        # 1. Sample G thoughts per prefix under current policy (detached rollout)
        thought_ids, old_logprobs, token_entropies, thought_valid_mask = self._sample_thoughts(
            prefix_ids, prefix_mask
        )

        # 2. Compute frozen-reference no-thought anchor & detached thought continuation logits
        with torch.no_grad():
            with self._autocast_ctx():
                ref_nothought_logits = self.thought_helper.evaluate_nothought_logits(
                    self.ref_model, prefix_ids, safe_targets, prefix_mask=prefix_mask
                )
                live_nothought_logits = self.thought_helper.evaluate_nothought_logits(
                    self.model, prefix_ids, safe_targets, prefix_mask=prefix_mask
                )
                _, thought_cont_logits_detached = self._recompute_thought_and_continuation(
                    prefix_ids, prefix_mask, thought_ids, safe_targets
                )

            # Strictly detached reward anchored to frozen p_{theta_ref}
            rewards = compute_reward(
                logits_thought=thought_cont_logits_detached,
                logits_ref_nothought=ref_nothought_logits,
                targets=target_ids,
                reward_type=self.rl_cfg.reward_type,
                tau=self.rl_cfg.reward_clip_tau,
                mask=target_mask,
            )  # [B, G]

            # 3. Dr. GRPO advantages: group-centered WITHOUT std-dev division
            advantages = compute_dr_grpo_advantages(rewards)

            # 4. Dynamic sigma=0 group filtering & Track 1 token filter selection
            group_std = compute_self_variance_scores(rewards)  # [B]
            nonzero_var_mask = group_std > self.std_eps
            zero_var_frac = float(1.0 - nonzero_var_mask.float().mean().item())

            valid_items = target_mask.any(dim=-1)
            filter_type = self.rl_cfg.filter_type
            if filter_type == "T0_all":
                selected_groups = select_tokens("T0_all", valid_mask=valid_items)
            elif filter_type == "T1_entropy":
                selected_groups = select_tokens(
                    "T1_entropy",
                    top_p_tokens=self.rl_cfg.top_p_tokens,
                    valid_mask=valid_items,
                    logits_ref=ref_nothought_logits[:, 0, :],
                )
            elif filter_type == "T2_self":
                selected_groups = select_tokens(
                    "T2_self",
                    top_p_tokens=self.rl_cfg.top_p_tokens,
                    valid_mask=valid_items,
                    group_rewards=rewards,
                    std_eps=self.std_eps,
                )
            elif filter_type == "T2_scale":
                if self.teacher_model is None:
                    raise ValueError(
                        "filter_type='T2_scale' requires `teacher_model` to be passed to SSLRLTrainer."
                    )
                teacher_logits = self.thought_helper.evaluate_nothought_logits(
                    self.teacher_model, prefix_ids, safe_targets, prefix_mask=prefix_mask
                )
                selected_groups = select_tokens(
                    "T2_scale",
                    top_p_tokens=self.rl_cfg.top_p_tokens,
                    valid_mask=valid_items,
                    logits_ref_large=teacher_logits[:, 0, :],
                    logits_small=live_nothought_logits[:, 0, :],
                    targets=target_ids[:, 0],
                )
            else:
                raise ValueError(f"Unsupported filter_type={filter_type!r}")

            active_groups = (
                selected_groups & nonzero_var_mask
                if self.rl_cfg.filter_zero_var
                else selected_groups
            )
            n_active_groups = int(active_groups.sum().item())
            active_group_frac = float(n_active_groups / max(1, B))

            # 5. Top-p high-entropy token gradient masking inside thought z_t
            thought_entropy_mask = select_thought_entropy_mask(
                token_entropies,
                top_p=self.rl_cfg.top_p_thought_entropy,
                valid_mask=thought_valid_mask,
            )
            pg_token_mask = thought_entropy_mask & active_groups[:, None, None]

            ref_nothought_nll = self._compute_target_nll(ref_nothought_logits, target_ids)
            live_nothought_nll = self._compute_target_nll(live_nothought_logits, target_ids)
            nothought_nll_drift = live_nothought_nll - ref_nothought_nll

            valid_ent = token_entropies[thought_valid_mask]
            policy_entropy = float(valid_ent.mean().item()) if valid_ent.numel() > 0 else 0.0
            mean_thought_len = float(thought_valid_mask.float().sum(dim=-1).mean().item())

        # 6. Dr. GRPO + DAPO PG loss + explicit auxiliary NTP loss + coupled continuation CE
        last_loss_pg = 0.0
        last_loss_ntp = 0.0
        last_loss_cont = 0.0
        last_loss_total = 0.0
        last_clip_frac = 0.0

        need_cont_in_pg = self.continuation_sft_weight > 0.0 and bool(torch.any(target_mask).item())
        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_ctx():
                new_logprobs, cont_logits = self._recompute_thought_and_continuation(
                    prefix_ids,
                    prefix_mask,
                    thought_ids,
                    safe_targets,
                    need_continuation=need_cont_in_pg,
                )
                loss_pg, clip_frac = compute_dapo_pg_loss(
                    new_logprobs=new_logprobs,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    token_mask=pg_token_mask,
                    max_thought_len=L,
                    clip_low=self.rl_cfg.clip_low,
                    clip_high=self.rl_cfg.clip_high,
                    n_active_groups=n_active_groups if n_active_groups > 0 else 1,
                )

                if self.rl_cfg.lambda_ntp > 0.0:
                    loss_ntp = self._compute_ntp_loss(full_ids, full_labels, full_attn)
                else:
                    loss_ntp = new_logprobs.sum() * 0.0

                if need_cont_in_pg and cont_logits is not None:
                    exp_targets = target_ids.unsqueeze(1).expand(B, G, -1)
                    loss_cont = F.cross_entropy(
                        cont_logits.reshape(-1, cont_logits.shape[-1]).float(),
                        exp_targets.reshape(-1),
                        ignore_index=-100,
                    )
                else:
                    loss_cont = new_logprobs.sum() * 0.0

                total_loss = (
                    loss_pg
                    + float(self.rl_cfg.lambda_ntp) * loss_ntp
                    + float(self.continuation_sft_weight) * loss_cont
                )

            if total_loss.requires_grad:
                total_loss.backward()
            synced = _all_reduce_model_grads(self.model)
            if total_loss.requires_grad or synced:
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            last_loss_pg = float(loss_pg.detach().item())
            last_loss_ntp = float(loss_ntp.detach().item())
            last_loss_cont = float(loss_cont.detach().item())
            last_loss_total = float(total_loss.detach().item())
            last_clip_frac = float(clip_frac)

        # Slow EMA update ONLY when ref_ema_decay < 1.0 (default 1.0 = strictly frozen)
        self._maybe_update_ref()
        self.step_count += 1

        return {
            "step": float(self.step_count),
            "loss_total": last_loss_total,
            "loss_pg": last_loss_pg,
            "loss_ntp": last_loss_ntp,
            "loss_cont": last_loss_cont,
            "mean_reward": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "policy_entropy": policy_entropy,
            "zero_var_frac": zero_var_frac,
            "active_group_frac": active_group_frac,
            "nothought_nll": live_nothought_nll,
            "ref_nothought_nll": ref_nothought_nll,
            "nothought_nll_drift": nothought_nll_drift,
            "mean_thought_len": mean_thought_len,
            "clip_frac": last_clip_frac,
        }
