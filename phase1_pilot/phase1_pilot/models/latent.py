"""Latent reasoning interfaces & baselines (B3 Pause, B4 Continuous Latent, I1/I2 RL)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_pilot.config import ModelConfig
from phase1_pilot.models.llama import LMOutput, LlamaLM, RMSNorm


def _unwrap_base_lm(model: nn.Module) -> LlamaLM:
    if isinstance(model, LlamaLM):
        return model
    if hasattr(model, "base_lm") and isinstance(model.base_lm, LlamaLM):
        return model.base_lm
    raise TypeError(f"Expected LlamaLM or wrapper around LlamaLM, got {type(model).__name__}")


def _left_align_prefix(
    prefix_ids: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
    pad_id: int = 0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Left-pad masked prefixes so the final real token always sits at column -1."""
    if prefix_mask is None:
        return prefix_ids, None
    mask = prefix_mask.long()
    if (mask[:, -1] == 1).all():
        return prefix_ids, mask
    bsz, t_len = prefix_ids.shape
    aligned_ids = torch.full_like(prefix_ids, pad_id)
    aligned_mask = torch.zeros_like(mask)
    for b in range(bsz):
        valid = prefix_ids[b][mask[b].bool()]
        n_valid = valid.numel()
        if n_valid > 0:
            aligned_ids[b, t_len - n_valid :] = valid
            aligned_mask[b, t_len - n_valid :] = 1
    return aligned_ids, aligned_mask


def _resolve_prompt_lens(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    prompt_lens: Optional[Union[torch.Tensor, Sequence[int], int]] = None,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> list[int]:
    """Resolve per-example split index where pause/latent tokens are inserted."""
    if prompt_lens is not None:
        if isinstance(prompt_lens, int):
            vals = [prompt_lens] * batch_size
        elif isinstance(prompt_lens, torch.Tensor):
            vals = [int(v) for v in prompt_lens.detach().cpu().tolist()]
        else:
            vals = [int(v) for v in prompt_lens]
        return [max(1, min(v, seq_len)) for v in vals]

    if labels is not None:
        vals = []
        for b in range(batch_size):
            active = (labels[b] != ignore_index).nonzero(as_tuple=False).view(-1)
            if active.numel() > 0:
                first_target = int(active[0].item())
                if first_target > 0:
                    vals.append(min(first_target, seq_len))
                else:
                    # Pure full-sequence LM labels: insert at midpoint
                    vals.append(max(1, seq_len // 2))
            else:
                vals.append(seq_len)
        return vals

    if attention_mask is not None and attention_mask.ndim == 2:
        lengths = attention_mask.long().sum(dim=1).cpu().tolist()
        return [max(1, min(int(v), seq_len)) for v in lengths]

    return [seq_len] * batch_size


class PauseTokenWrapper(nn.Module):
    """B3 Baseline: inserts `n_pause_tokens` learnable `<pause>` embeddings between prompt and answer.

    Returns `LMOutput` with `logits` aligned to `[B, T, V]` so `logits[b, p_b - 1]` is the
    prediction AFTER all `n_pause_tokens` pause passes, and pause positions are masked out of
    cross-entropy loss.
    """

    def __init__(
        self,
        base_lm: LlamaLM,
        n_pause_tokens: int = 4,
        pause_token_id: int = 5,
        learnable_pause_embeds: bool = True,
    ) -> None:
        super().__init__()
        self.base_lm = base_lm
        self.config = base_lm.config
        self.n_pause_tokens = n_pause_tokens
        self.pause_token_id = pause_token_id
        self.learnable_pause_embeds = learnable_pause_embeds

        d_model = base_lm.config.d_model
        if n_pause_tokens > 0 and learnable_pause_embeds:
            init = base_lm.tok_embeddings.weight[pause_token_id].detach().clone()
            pause_init = init.unsqueeze(0).repeat(n_pause_tokens, 1)
            pause_init = pause_init + 0.01 * torch.randn_like(pause_init)
            self.pause_embeddings = nn.Parameter(pause_init)
        else:
            self.register_parameter("pause_embeddings", None)

    def _get_pause_embeds(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.pause_embeddings is not None:
            return self.pause_embeddings.to(device=device, dtype=dtype)
        ids = torch.full(
            (self.n_pause_tokens,), self.pause_token_id, device=device, dtype=torch.long
        )
        return self.base_lm.tok_embeddings(ids).to(dtype=dtype)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_lens: Optional[Union[torch.Tensor, Sequence[int], int]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        ignore_index: int = -100,
    ) -> LMOutput:
        if self.n_pause_tokens <= 0:
            return self.base_lm(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=use_cache,
                return_hidden=return_hidden,
                labels=labels,
                ignore_index=ignore_index,
            )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is not None:
            base_embeds = inputs_embeds
        else:
            assert input_ids is not None
            base_embeds = self.base_lm.tok_embeddings(input_ids)

        bsz, seq_len, d_model = base_embeds.shape
        device = base_embeds.device
        dtype = base_embeds.dtype
        n_p = self.n_pause_tokens

        splits = _resolve_prompt_lens(
            batch_size=bsz,
            seq_len=seq_len,
            device=device,
            prompt_lens=prompt_lens,
            labels=labels,
            attention_mask=attention_mask,
            ignore_index=ignore_index,
        )

        pause_emb = self._get_pause_embeds(device=device, dtype=dtype)  # [P, d]

        expanded_embeds_list: list[torch.Tensor] = []
        expanded_mask_list: list[torch.Tensor] = []
        expanded_labels_list: list[torch.Tensor] = []

        if attention_mask is None:
            base_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)
        else:
            base_mask = attention_mask.long()

        ones_p = torch.ones((n_p,), device=device, dtype=torch.long)
        ignore_p = torch.full((n_p,), ignore_index, device=device, dtype=torch.long)

        for b in range(bsz):
            p = splits[b]
            emb_b = torch.cat(
                (base_embeds[b, :p, :], pause_emb, base_embeds[b, p:, :]), dim=0
            )
            expanded_embeds_list.append(emb_b)

            mask_b = torch.cat((base_mask[b, :p], ones_p, base_mask[b, p:]), dim=0)
            expanded_mask_list.append(mask_b)

            if labels is not None:
                lbl_b = torch.cat((labels[b, :p], ignore_p, labels[b, p:]), dim=0)
                expanded_labels_list.append(lbl_b)

        expanded_embeds = torch.stack(expanded_embeds_list, dim=0)  # [B, T + P, d]
        expanded_mask = torch.stack(expanded_mask_list, dim=0)  # [B, T + P]
        expanded_labels = (
            torch.stack(expanded_labels_list, dim=0) if labels is not None else None
        )

        out = self.base_lm(
            inputs_embeds=expanded_embeds,
            attention_mask=expanded_mask,
            use_cache=use_cache,
            return_hidden=return_hidden,
            labels=expanded_labels,
            ignore_index=ignore_index,
        )

        # Gather logits aligned back to original [B, T, V]:
        # [0..p-2] -> prompt predictions, [p + P - 1] -> post-pause prediction for token p,
        # [p + P .. T + P - 1] -> remaining target predictions.
        aligned_logits_list: list[torch.Tensor] = []
        aligned_hidden_list: list[torch.Tensor] = []
        for b in range(bsz):
            p = splits[b]
            idx_before = torch.arange(0, max(p - 1, 0), device=device)
            idx_after = torch.arange(p + n_p - 1, seq_len + n_p, device=device)
            gather_idx = torch.cat((idx_before, idx_after), dim=0)
            aligned_logits_list.append(out.logits[b, gather_idx, :])
            if return_hidden and out.hidden_states is not None:
                aligned_hidden_list.append(out.hidden_states[b, gather_idx, :])

        aligned_logits = torch.stack(aligned_logits_list, dim=0)
        aligned_hidden = (
            torch.stack(aligned_hidden_list, dim=0)
            if (return_hidden and out.hidden_states is not None)
            else None
        )

        return LMOutput(
            logits=aligned_logits,
            hidden_states=aligned_hidden,
            past_kvs=out.past_kvs,
            loss=out.loss,
        )


@dataclass
class HybridRolloutOutput:
    """Outputs of HRPO-style hybrid continuous+discrete latent rollout (I2)."""

    thought_ids: torch.Tensor  # [B, G, L]
    old_logprobs: torch.Tensor  # [B, G, L]
    token_entropies: torch.Tensor  # [B, G, L]
    latent_embeds: torch.Tensor  # [B, G, L, d_model]


class ContinuousLatentWrapper(nn.Module):
    """B4 (Continuous Latent BPTT) and I2 (HRPO Hybrid Gated Latent) Wrapper.

    Executes `n_latent_steps` autoregressive hidden-state feed-back loops using a learned
    residual adapter (`latent_norm` + `latent_proj`) with full BPTT gradient flow across
    every latent step, then decodes target tokens conditioned on the latent trajectory.

    When `hybrid_gate=True`, adds an HRPO-style learnable sigmoid gate blending continuous
    latent projections with sampled discrete token embeddings for hybrid RL (`I2`).
    """

    def __init__(
        self,
        base_lm: LlamaLM,
        n_latent_steps: int = 4,
        hybrid_gate: bool = False,
    ) -> None:
        super().__init__()
        self.base_lm = base_lm
        self.config = base_lm.config
        self.n_latent_steps = n_latent_steps
        self.hybrid_gate = hybrid_gate

        d_model = base_lm.config.d_model
        self.latent_norm = RMSNorm(d_model, eps=base_lm.config.norm_eps)
        self.latent_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.latent_proj.weight, mean=0.0, std=0.02)

        if hybrid_gate:
            self.gate_proj: Optional[nn.Linear] = nn.Linear(2 * d_model, d_model, bias=True)
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.gate_proj.bias)
        else:
            self.gate_proj = None

    def project_latent(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project transformer hidden state back into input embedding space via residual adapter."""
        h_norm = self.latent_norm(hidden_state)
        return h_norm + self.latent_proj(h_norm)

    def blend_hybrid(
        self,
        hidden_state: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """HRPO-style gated residual blending of continuous latent state & discrete token embed."""
        latent_emb = self.project_latent(hidden_state)
        if not self.hybrid_gate or self.gate_proj is None:
            return latent_emb
        tok_emb = self.base_lm.tok_embeddings(token_ids)
        gate = torch.sigmoid(self.gate_proj(torch.cat((latent_emb, tok_emb), dim=-1)))
        return gate * latent_emb + (1.0 - gate) * tok_emb

    def _run_prefix_and_latents(
        self,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        prefix_pos_ids: torch.Tensor,
        n_steps: int,
        latent_override: Optional[torch.Tensor] = None,
        latent_perturb_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> tuple[
        torch.Tensor,  # prefix_logits [B, p_max, V]
        torch.Tensor,  # prefix_hidden [B, p_max, d]
        torch.Tensor,  # last_latent_logits [B, 1, V]
        torch.Tensor,  # last_latent_hidden [B, 1, d]
        torch.Tensor,  # latent_vectors [B, n_steps, d]
        list[tuple[torch.Tensor, torch.Tensor]],  # past_kvs
        torch.Tensor,  # cur_mask [B, p_max + n_steps]
        torch.Tensor,  # next_pos [B, 1]
    ]:
        bsz = prefix_embeds.shape[0]
        device = prefix_embeds.device

        prefix_out = self.base_lm(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            position_ids=prefix_pos_ids,
            use_cache=True,
            return_hidden=True,
        )
        assert prefix_out.hidden_states is not None and prefix_out.past_kvs is not None

        past_kvs = prefix_out.past_kvs
        cur_mask = prefix_mask
        h_cur = prefix_out.hidden_states[:, -1:, :]  # [B, 1, d]
        last_logits = prefix_out.logits[:, -1:, :]  # [B, 1, V]
        next_pos = prefix_pos_ids[:, -1:] + 1  # [B, 1]

        latent_list: list[torch.Tensor] = []

        if latent_override is not None:
            # Feed externally supplied/swapped/perturbed latent sequence directly
            z_seq = latent_override
            if z_seq.ndim != 3 or z_seq.shape[0] != bsz:
                raise ValueError(
                    f"latent_override must have shape [B={bsz}, L, d], got {tuple(z_seq.shape)}"
                )
            n_ov = z_seq.shape[1]
            step_offsets = torch.arange(n_ov, device=device).unsqueeze(0)
            ov_pos = next_pos + step_offsets  # [B, n_ov]
            cur_mask = torch.cat(
                (cur_mask, torch.ones((bsz, n_ov), device=device, dtype=cur_mask.dtype)),
                dim=1,
            )
            ov_out = self.base_lm(
                inputs_embeds=z_seq,
                attention_mask=cur_mask,
                past_kvs=past_kvs,
                position_ids=ov_pos,
                use_cache=True,
                return_hidden=True,
            )
            assert ov_out.hidden_states is not None and ov_out.past_kvs is not None
            past_kvs = ov_out.past_kvs
            h_cur = ov_out.hidden_states[:, -1:, :]
            last_logits = ov_out.logits[:, -1:, :]
            next_pos = next_pos + n_ov
            latent_vectors = z_seq
        else:
            ones_col = torch.ones((bsz, 1), device=device, dtype=cur_mask.dtype)
            for _ in range(n_steps):
                z_step = self.project_latent(h_cur)  # [B, 1, d]
                if latent_perturb_fn is not None:
                    z_step = latent_perturb_fn(z_step)
                latent_list.append(z_step)
                cur_mask = torch.cat((cur_mask, ones_col), dim=1)
                step_out = self.base_lm(
                    inputs_embeds=z_step,
                    attention_mask=cur_mask,
                    past_kvs=past_kvs,
                    position_ids=next_pos,
                    use_cache=True,
                    return_hidden=True,
                )
                assert step_out.hidden_states is not None and step_out.past_kvs is not None
                past_kvs = step_out.past_kvs
                h_cur = step_out.hidden_states
                last_logits = step_out.logits
                next_pos = next_pos + 1

            latent_vectors = (
                torch.cat(latent_list, dim=1)
                if latent_list
                else prefix_embeds.new_empty((bsz, 0, prefix_embeds.shape[-1]))
            )

        return (
            prefix_out.logits,
            prefix_out.hidden_states,
            last_logits,
            h_cur,
            latent_vectors,
            past_kvs,
            cur_mask,
            next_pos,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_lens: Optional[Union[torch.Tensor, Sequence[int], int]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        return_latents: bool = False,
        latent_override: Optional[torch.Tensor] = None,
        latent_perturb_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ignore_index: int = -100,
    ) -> LMOutput:
        if self.n_latent_steps <= 0 and latent_override is None:
            return self.base_lm(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=use_cache,
                return_hidden=return_hidden,
                labels=labels,
                ignore_index=ignore_index,
            )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is not None:
            base_embeds = inputs_embeds
        else:
            assert input_ids is not None
            base_embeds = self.base_lm.tok_embeddings(input_ids)

        bsz, seq_len, d_model = base_embeds.shape
        device = base_embeds.device
        dtype = base_embeds.dtype

        splits = _resolve_prompt_lens(
            batch_size=bsz,
            seq_len=seq_len,
            device=device,
            prompt_lens=prompt_lens,
            labels=labels,
            attention_mask=attention_mask,
            ignore_index=ignore_index,
        )

        if attention_mask is None:
            base_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)
        else:
            base_mask = attention_mask.long()

        # Left-pad prompts so every example's last prompt token aligns at column p_max - 1
        p_max = max(splits)
        prefix_embeds = torch.zeros((bsz, p_max, d_model), device=device, dtype=dtype)
        prefix_mask = torch.zeros((bsz, p_max), device=device, dtype=torch.long)
        prefix_pos_ids = torch.zeros((bsz, p_max), device=device, dtype=torch.long)

        for b in range(bsz):
            p = splits[b]
            pad_left = p_max - p
            prefix_embeds[b, pad_left:, :] = base_embeds[b, :p, :]
            prefix_mask[b, pad_left:] = base_mask[b, :p]
            prefix_pos_ids[b, pad_left:] = torch.arange(p, device=device, dtype=torch.long)

        (
            prefix_logits,
            prefix_hidden,
            last_latent_logits,
            last_latent_hidden,
            latent_vectors,
            past_kvs,
            cur_mask,
            next_pos,
        ) = self._run_prefix_and_latents(
            prefix_embeds=prefix_embeds,
            prefix_mask=prefix_mask,
            prefix_pos_ids=prefix_pos_ids,
            n_steps=self.n_latent_steps,
            latent_override=latent_override,
            latent_perturb_fn=latent_perturb_fn,
        )

        # Decode remaining suffix/target tokens conditioned on the continuous latent states
        rem_lens = [seq_len - p for p in splits]
        t_max = max(rem_lens)

        target_logits: Optional[torch.Tensor] = None
        target_hidden: Optional[torch.Tensor] = None
        final_past_kvs = past_kvs

        if t_max > 0:
            target_embeds = torch.zeros((bsz, t_max, d_model), device=device, dtype=dtype)
            target_mask = torch.zeros((bsz, t_max), device=device, dtype=torch.long)
            for b in range(bsz):
                p = splits[b]
                rem = rem_lens[b]
                if rem > 0:
                    target_embeds[b, :rem, :] = base_embeds[b, p : p + rem, :]
                    target_mask[b, :rem] = base_mask[b, p : p + rem]

            full_mask = torch.cat((cur_mask, target_mask), dim=1)
            target_pos_ids = next_pos + torch.arange(t_max, device=device).unsqueeze(0)

            target_out = self.base_lm(
                inputs_embeds=target_embeds,
                attention_mask=full_mask,
                past_kvs=past_kvs,
                position_ids=target_pos_ids,
                use_cache=use_cache,
                return_hidden=return_hidden,
            )
            target_logits = target_out.logits
            target_hidden = target_out.hidden_states
            if use_cache and target_out.past_kvs is not None:
                final_past_kvs = target_out.past_kvs

        # Reassemble aligned [B, T, V] logits matching LlamaLM causal shift convention
        aligned_logits_list: list[torch.Tensor] = []
        aligned_hidden_list: list[torch.Tensor] = []

        for b in range(bsz):
            p = splits[b]
            rem = rem_lens[b]
            pad_left = p_max - p

            parts_l = []
            if p > 1:
                parts_l.append(prefix_logits[b, pad_left : pad_left + p - 1, :])
            parts_l.append(last_latent_logits[b, :, :])  # slot p - 1 predicts token p
            if rem > 0 and target_logits is not None:
                parts_l.append(target_logits[b, :rem, :])
            aligned_logits_list.append(torch.cat(parts_l, dim=0))

            if return_hidden:
                parts_h = []
                if p > 1:
                    parts_h.append(prefix_hidden[b, pad_left : pad_left + p - 1, :])
                parts_h.append(last_latent_hidden[b, :, :])
                if rem > 0 and target_hidden is not None:
                    parts_h.append(target_hidden[b, :rem, :])
                aligned_hidden_list.append(torch.cat(parts_h, dim=0))

        aligned_logits = torch.stack(aligned_logits_list, dim=0)  # [B, T, V]
        aligned_hidden = (
            torch.stack(aligned_hidden_list, dim=0) if return_hidden else None
        )

        loss: Optional[torch.Tensor] = None
        if labels is not None and seq_len > 1:
            shift_logits = aligned_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if (shift_labels != ignore_index).any():
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=ignore_index,
                )
            else:
                loss = shift_logits.sum() * 0.0

        return LMOutput(
            logits=aligned_logits,
            hidden_states=aligned_hidden,
            past_kvs=final_past_kvs if use_cache else None,
            loss=loss,
            latent_states=latent_vectors if return_latents else None,
        )

    def generate_latents(
        self,
        prefix_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run `n_steps` continuous latent steps after `prefix_ids`.

        Returns `(latent_vectors [B, L, d], last_logits [B, V], past_kvs)`.
        """
        steps = self.n_latent_steps if n_steps is None else n_steps
        prefix_ids, attention_mask = _left_align_prefix(prefix_ids, attention_mask)
        bsz, t_len = prefix_ids.shape
        device = prefix_ids.device
        prefix_embeds = self.base_lm.tok_embeddings(prefix_ids)
        if attention_mask is None:
            mask = torch.ones((bsz, t_len), device=device, dtype=torch.long)
            pos_ids = torch.arange(t_len, device=device).unsqueeze(0).expand(bsz, t_len)
        else:
            mask = attention_mask.long()
            pos_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)

        _, _, last_logits, _, latent_vectors, past_kvs, _, _ = self._run_prefix_and_latents(
            prefix_embeds=prefix_embeds,
            prefix_mask=mask,
            prefix_pos_ids=pos_ids,
            n_steps=steps,
        )
        return latent_vectors, last_logits.squeeze(1), past_kvs

    def evaluate_with_latents(
        self,
        prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        latent_vectors: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute exact continuation logits `[B, m, V]` for `target_ids` given supplied
        `latent_vectors [B, L, d]` (enabling swap-sensitivity & perturbation faithfulness probes)."""
        prefix_ids, prefix_mask = _left_align_prefix(prefix_ids, prefix_mask)
        bsz, t_len = prefix_ids.shape
        m = target_ids.shape[1]
        l_steps = latent_vectors.shape[1]
        device = prefix_ids.device

        prefix_embeds = self.base_lm.tok_embeddings(prefix_ids)
        if m > 1:
            target_in_embeds = self.base_lm.tok_embeddings(target_ids[:, :-1])
            full_embeds = torch.cat((prefix_embeds, latent_vectors, target_in_embeds), dim=1)
        else:
            full_embeds = torch.cat((prefix_embeds, latent_vectors), dim=1)

        if prefix_mask is not None:
            extra_ones = torch.ones(
                (bsz, l_steps + max(m - 1, 0)), device=device, dtype=prefix_mask.dtype
            )
            full_mask: Optional[torch.Tensor] = torch.cat((prefix_mask, extra_ones), dim=1)
            pos_ids: Optional[torch.Tensor] = (full_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        else:
            full_mask = None
            pos_ids = None

        out = self.base_lm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            position_ids=pos_ids,
        )
        # Last latent vector at index (t_len + l_steps - 1) predicts target_ids[:, 0]
        start = t_len + l_steps - 1
        return out.logits[:, start : start + m, :]

    @torch.no_grad()
    def rollout_hybrid(
        self,
        prefix_ids: torch.Tensor,
        group_size: int = 8,
        n_steps: Optional[int] = None,
        temperature: float = 1.0,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> HybridRolloutOutput:
        """Sample `G` stochastic HRPO-style hybrid continuous+discrete latent trajectories."""
        steps = self.n_latent_steps if n_steps is None else n_steps
        prefix_ids, prefix_mask = _left_align_prefix(prefix_ids, prefix_mask)
        bsz, t_len = prefix_ids.shape
        device = prefix_ids.device

        if prefix_mask is not None:
            mask = prefix_mask.long()
            pos_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
        else:
            mask = torch.ones((bsz, t_len), device=device, dtype=torch.long)
            pos_ids = torch.arange(t_len, device=device).unsqueeze(0).expand(bsz, t_len)

        prefix_out = self.base_lm(
            input_ids=prefix_ids,
            attention_mask=mask if prefix_mask is not None else None,
            position_ids=pos_ids if prefix_mask is not None else None,
            use_cache=True,
            return_hidden=True,
        )
        assert prefix_out.hidden_states is not None and prefix_out.past_kvs is not None

        h_cur = prefix_out.hidden_states[:, -1:, :].repeat_interleave(group_size, dim=0)
        logits_cur = prefix_out.logits[:, -1, :].repeat_interleave(group_size, dim=0)
        past_kvs = [
            (
                k.repeat_interleave(group_size, dim=0),
                v.repeat_interleave(group_size, dim=0),
            )
            for (k, v) in prefix_out.past_kvs
        ]
        cur_mask = mask.repeat_interleave(group_size, dim=0) if prefix_mask is not None else None
        next_pos = (pos_ids[:, -1:] + 1).repeat_interleave(group_size, dim=0)

        tok_list: list[torch.Tensor] = []
        lp_list: list[torch.Tensor] = []
        ent_list: list[torch.Tensor] = []
        emb_list: list[torch.Tensor] = []

        temp = max(float(temperature), 1e-6)
        ones_col = torch.ones((bsz * group_size, 1), device=device, dtype=torch.long)

        for step in range(steps):
            scaled = logits_cur / temp
            log_probs = F.log_softmax(scaled, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)

            if temperature <= 0.0:
                tok = scaled.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(probs, num_samples=1)

            tok_lp = log_probs.gather(-1, tok).squeeze(-1)
            blended_emb = self.blend_hybrid(h_cur, tok)  # [B*G, 1, d]

            tok_list.append(tok.squeeze(-1))
            lp_list.append(tok_lp)
            ent_list.append(entropy)
            emb_list.append(blended_emb)

            if cur_mask is not None:
                cur_mask = torch.cat((cur_mask, ones_col), dim=1)

            step_out = self.base_lm(
                inputs_embeds=blended_emb,
                attention_mask=cur_mask,
                past_kvs=past_kvs,
                position_ids=next_pos if prefix_mask is not None else None,
                use_cache=True,
                return_hidden=True,
            )
            assert step_out.hidden_states is not None and step_out.past_kvs is not None
            past_kvs = step_out.past_kvs
            h_cur = step_out.hidden_states
            logits_cur = step_out.logits[:, -1, :]
            next_pos = next_pos + 1

        thought_ids = torch.stack(tok_list, dim=1).view(bsz, group_size, steps)
        old_logprobs = torch.stack(lp_list, dim=1).view(bsz, group_size, steps)
        token_entropies = torch.stack(ent_list, dim=1).view(bsz, group_size, steps)
        latent_embeds = torch.cat(emb_list, dim=1).view(bsz, group_size, steps, -1)

        return HybridRolloutOutput(
            thought_ids=thought_ids,
            old_logprobs=old_logprobs,
            token_entropies=token_entropies,
            latent_embeds=latent_embeds,
        )

    def recompute_hybrid_logprobs_and_continuation(
        self,
        prefix_ids: torch.Tensor,
        thought_ids: torch.Tensor,
        target_ids: torch.Tensor,
        temperature: float = 1.0,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable recomputation of HRPO hybrid thought logprobs, entropies, and
        target continuation logits `[B, G, m, V]` with full gradient flow through `base_lm`,
        `latent_proj`, and `gate_proj`."""
        prefix_ids, prefix_mask = _left_align_prefix(prefix_ids, prefix_mask)
        bsz, t_len = prefix_ids.shape
        if thought_ids.ndim == 2:
            thought_ids = thought_ids.unsqueeze(1)
            squeeze_g = True
        else:
            squeeze_g = False
        _, group_size, steps = thought_ids.shape
        m = target_ids.shape[1]
        device = prefix_ids.device

        if prefix_mask is not None:
            mask = prefix_mask.long()
            pos_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
        else:
            mask = torch.ones((bsz, t_len), device=device, dtype=torch.long)
            pos_ids = torch.arange(t_len, device=device).unsqueeze(0).expand(bsz, t_len)

        prefix_out = self.base_lm(
            input_ids=prefix_ids,
            attention_mask=mask if prefix_mask is not None else None,
            position_ids=pos_ids if prefix_mask is not None else None,
            use_cache=True,
            return_hidden=True,
        )
        assert prefix_out.hidden_states is not None and prefix_out.past_kvs is not None

        h_cur = prefix_out.hidden_states[:, -1:, :].repeat_interleave(group_size, dim=0)
        logits_cur = prefix_out.logits[:, -1, :].repeat_interleave(group_size, dim=0)
        past_kvs = [
            (
                k.repeat_interleave(group_size, dim=0),
                v.repeat_interleave(group_size, dim=0),
            )
            for (k, v) in prefix_out.past_kvs
        ]
        cur_mask = mask.repeat_interleave(group_size, dim=0) if prefix_mask is not None else None
        next_pos = (pos_ids[:, -1:] + 1).repeat_interleave(group_size, dim=0)
        ones_col = torch.ones((bsz * group_size, 1), device=device, dtype=torch.long)

        thought_flat = thought_ids.reshape(bsz * group_size, steps)
        temp = max(float(temperature), 1e-6)

        lp_list: list[torch.Tensor] = []
        ent_list: list[torch.Tensor] = []

        for step in range(steps):
            scaled = logits_cur / temp
            log_probs = F.log_softmax(scaled, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)

            tok = thought_flat[:, step : step + 1]
            tok_lp = log_probs.gather(-1, tok).squeeze(-1)
            lp_list.append(tok_lp)
            ent_list.append(entropy)

            blended_emb = self.blend_hybrid(h_cur, tok)
            if cur_mask is not None:
                cur_mask = torch.cat((cur_mask, ones_col), dim=1)

            step_out = self.base_lm(
                inputs_embeds=blended_emb,
                attention_mask=cur_mask,
                past_kvs=past_kvs,
                position_ids=next_pos if prefix_mask is not None else None,
                use_cache=True,
                return_hidden=True,
            )
            assert step_out.hidden_states is not None and step_out.past_kvs is not None
            past_kvs = step_out.past_kvs
            h_cur = step_out.hidden_states
            logits_cur = step_out.logits[:, -1, :]
            next_pos = next_pos + 1

        # logits_cur now predicts target_ids[:, 0]
        first_target_logits = logits_cur.unsqueeze(1)  # [B*G, 1, V]
        if m > 1:
            target_in = target_ids[:, :-1].repeat_interleave(group_size, dim=0)
            if cur_mask is not None:
                t_ones = torch.ones((bsz * group_size, m - 1), device=device, dtype=torch.long)
                cur_mask = torch.cat((cur_mask, t_ones), dim=1)
            t_pos = (
                next_pos + torch.arange(m - 1, device=device).unsqueeze(0)
                if prefix_mask is not None
                else None
            )
            t_out = self.base_lm(
                input_ids=target_in,
                attention_mask=cur_mask,
                past_kvs=past_kvs,
                position_ids=t_pos,
            )
            cont_logits = torch.cat((first_target_logits, t_out.logits), dim=1)
        else:
            cont_logits = first_target_logits

        new_logprobs = torch.stack(lp_list, dim=1).view(bsz, group_size, steps)
        new_entropies = torch.stack(ent_list, dim=1).view(bsz, group_size, steps)
        cont_logits = cont_logits.view(bsz, group_size, m, -1)

        if squeeze_g:
            return new_logprobs.squeeze(1), new_entropies.squeeze(1), cont_logits.squeeze(1)
        return new_logprobs, new_entropies, cont_logits


class ThoughtRollout(NamedTuple):
    """Vectorized discrete thought rollout outputs."""

    thought_ids: torch.Tensor  # [B, G, L]
    old_logprobs: torch.Tensor  # [B, G, L]
    token_entropies: torch.Tensor  # [B, G, L]


class ContinuationLogits(NamedTuple):
    """Target continuation logits with thought z_t and without thought."""

    with_thought: torch.Tensor  # [B, G, m, V] or [B, m, V]
    without_thought: torch.Tensor  # [B, m, V]


class DiscreteThoughtHelper:
    """KV-cached discrete thought rollout & continuation evaluator for I1/I2 RL."""

    def __init__(
        self,
        think_start_id: int = 3,
        think_end_id: int = 4,
        pad_id: int = 0,
    ) -> None:
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.pad_id = pad_id

    @torch.no_grad()
    def sample_thoughts(
        self,
        model: nn.Module,
        prefix_ids: torch.Tensor,
        thought_len: int,
        group_size: int = 8,
        temperature: float = 1.0,
        think_start_id: Optional[int] = None,
        think_end_id: Optional[int] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> ThoughtRollout:
        """Sample `G` discrete thoughts `z_t` (`[B, G, L]`) with KV-cache.

        Prefix + `<think>` is prefilled once per batch element and broadcast across the
        `G` group rollouts before autoregressive sampling.
        """
        del think_end_id  # Used during continuation evaluation after the L internal tokens
        base_lm = _unwrap_base_lm(model)
        prefix_ids, prefix_mask = _left_align_prefix(prefix_ids, prefix_mask, self.pad_id)
        start_id = self.think_start_id if think_start_id is None else think_start_id
        bsz, t_len = prefix_ids.shape
        device = prefix_ids.device

        start_col = torch.full((bsz, 1), start_id, device=device, dtype=torch.long)
        prefill_ids = torch.cat((prefix_ids, start_col), dim=1)  # [B, T + 1]

        if prefix_mask is not None:
            prefill_mask = torch.cat(
                (prefix_mask.long(), torch.ones((bsz, 1), device=device, dtype=torch.long)),
                dim=1,
            )
            pos_ids = (prefill_mask.cumsum(dim=1) - 1).clamp(min=0)
        else:
            prefill_mask = None
            pos_ids = None

        prefill_out = base_lm(
            input_ids=prefill_ids,
            attention_mask=prefill_mask,
            position_ids=pos_ids,
            use_cache=True,
        )
        assert prefill_out.past_kvs is not None

        logits_cur = prefill_out.logits[:, -1, :].repeat_interleave(group_size, dim=0)
        past_kvs = [
            (
                k.repeat_interleave(group_size, dim=0),
                v.repeat_interleave(group_size, dim=0),
            )
            for (k, v) in prefill_out.past_kvs
        ]

        cur_mask = (
            prefill_mask.repeat_interleave(group_size, dim=0)
            if prefill_mask is not None
            else None
        )
        next_pos = (
            (pos_ids[:, -1:] + 1).repeat_interleave(group_size, dim=0)
            if pos_ids is not None
            else None
        )
        ones_col = torch.ones((bsz * group_size, 1), device=device, dtype=torch.long)

        thought_cols: list[torch.Tensor] = []
        logprob_cols: list[torch.Tensor] = []
        entropy_cols: list[torch.Tensor] = []

        temp = max(float(temperature), 1e-6)

        for step in range(thought_len):
            scaled = logits_cur / temp
            log_probs = F.log_softmax(scaled, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)

            if temperature <= 0.0:
                tok = scaled.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(probs, num_samples=1)

            tok_lp = log_probs.gather(-1, tok).squeeze(-1)

            thought_cols.append(tok.squeeze(-1))
            logprob_cols.append(tok_lp)
            entropy_cols.append(entropy)

            if step < thought_len - 1:
                if cur_mask is not None:
                    cur_mask = torch.cat((cur_mask, ones_col), dim=1)
                step_out = base_lm(
                    input_ids=tok,
                    attention_mask=cur_mask,
                    past_kvs=past_kvs,
                    position_ids=next_pos,
                    use_cache=True,
                )
                assert step_out.past_kvs is not None
                past_kvs = step_out.past_kvs
                logits_cur = step_out.logits[:, -1, :]
                if next_pos is not None:
                    next_pos = next_pos + 1

        if thought_len == 0:
            empty_ids = torch.empty((bsz, group_size, 0), device=device, dtype=torch.long)
            empty_flt = torch.empty((bsz, group_size, 0), device=device, dtype=logits_cur.dtype)
            return ThoughtRollout(empty_ids, empty_flt, empty_flt)

        thought_ids = torch.stack(thought_cols, dim=1).view(bsz, group_size, thought_len)
        old_logprobs = torch.stack(logprob_cols, dim=1).view(bsz, group_size, thought_len)
        token_entropies = torch.stack(entropy_cols, dim=1).view(bsz, group_size, thought_len)

        return ThoughtRollout(
            thought_ids=thought_ids,
            old_logprobs=old_logprobs,
            token_entropies=token_entropies,
        )

    # Alias for convenience
    rollout = sample_thoughts

    def recompute_thought_logprobs(
        self,
        model: nn.Module,
        prefix_ids: torch.Tensor,
        thought_ids: torch.Tensor,
        temperature: float = 1.0,
        think_start_id: Optional[int] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced differentiable recomputation of thought logprobs and entropies.

        Matches `sample_thoughts` logprobs exactly when evaluated on identical weights.
        """
        base_lm = _unwrap_base_lm(model)
        prefix_ids, prefix_mask = _left_align_prefix(prefix_ids, prefix_mask, self.pad_id)
        start_id = self.think_start_id if think_start_id is None else think_start_id
        bsz, t_len = prefix_ids.shape
        device = prefix_ids.device

        if thought_ids.ndim == 2:
            thought_3d = thought_ids.unsqueeze(1)
            squeeze_g = True
        else:
            thought_3d = thought_ids
            squeeze_g = False

        _, group_size, thought_len = thought_3d.shape
        if thought_len == 0:
            empty = torch.empty((bsz, group_size, 0), device=device)
            return (empty.squeeze(1), empty.squeeze(1)) if squeeze_g else (empty, empty)

        prefix_exp = prefix_ids.repeat_interleave(group_size, dim=0)
        start_col = torch.full((bsz * group_size, 1), start_id, device=device, dtype=torch.long)
        thought_flat = thought_3d.reshape(bsz * group_size, thought_len)

        # Input sequence: [prefix, <think>, z_0, ..., z_{L-2}] -> logits at [T : T + L] predict z_0..z_{L-1}
        if thought_len > 1:
            seq = torch.cat((prefix_exp, start_col, thought_flat[:, :-1]), dim=1)
        else:
            seq = torch.cat((prefix_exp, start_col), dim=1)

        if prefix_mask is not None:
            mask_exp = prefix_mask.long().repeat_interleave(group_size, dim=0)
            extra = torch.ones((bsz * group_size, thought_len), device=device, dtype=torch.long)
            full_mask: Optional[torch.Tensor] = torch.cat((mask_exp, extra), dim=1)
            pos_ids: Optional[torch.Tensor] = (full_mask.cumsum(dim=1) - 1).clamp(min=0)
        else:
            full_mask = None
            pos_ids = None

        out = base_lm(input_ids=seq, attention_mask=full_mask, position_ids=pos_ids)
        thought_logits = out.logits[:, t_len : t_len + thought_len, :]

        temp = max(float(temperature), 1e-6)
        log_probs = F.log_softmax(thought_logits / temp, dim=-1)
        probs = log_probs.exp()
        entropies = -(probs * log_probs).sum(dim=-1)
        token_lps = log_probs.gather(-1, thought_flat.unsqueeze(-1)).squeeze(-1)

        token_lps = token_lps.view(bsz, group_size, thought_len)
        entropies = entropies.view(bsz, group_size, thought_len)

        if squeeze_g:
            return token_lps.squeeze(1), entropies.squeeze(1)
        return token_lps, entropies

    def evaluate_nothought_logits(
        self,
        model: nn.Module,
        prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute exact continuation logits `p(x_{t+k} | x_{<t}, x_{t:t+k-1})` WITHOUT thought.

        Returns tensor of shape `[B, m, V]`.
        """
        base_lm = _unwrap_base_lm(model)
        prefix_ids, prefix_mask = _left_align_prefix(prefix_ids, prefix_mask, self.pad_id)
        bsz, t_len = prefix_ids.shape
        m = target_ids.shape[1]
        device = prefix_ids.device

        if m > 1:
            seq = torch.cat((prefix_ids, target_ids[:, :-1]), dim=1)
        else:
            seq = prefix_ids

        if prefix_mask is not None:
            if m > 1:
                t_ones = torch.ones((bsz, m - 1), device=device, dtype=prefix_mask.dtype)
                full_mask: Optional[torch.Tensor] = torch.cat((prefix_mask, t_ones), dim=1)
            else:
                full_mask = prefix_mask
            pos_ids: Optional[torch.Tensor] = (full_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        else:
            full_mask = None
            pos_ids = None

        out = base_lm(input_ids=seq, attention_mask=full_mask, position_ids=pos_ids)
        return out.logits[:, t_len - 1 : t_len - 1 + m, :]

    def evaluate_continuation_logits(
        self,
        model: nn.Module,
        prefix_ids: torch.Tensor,
        thought_ids: torch.Tensor,
        target_ids: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        ref_model: Optional[nn.Module] = None,
        think_start_id: Optional[int] = None,
        think_end_id: Optional[int] = None,
        include_delimiters: bool = True,
    ) -> ContinuationLogits:
        """Compute exact target continuation logits `p(x_{t+k} | x_{<t}, z_t, x_{t:t+k-1})`
        for `k = 0 .. m - 1`, BOTH with thought `z_t` and without thought (`no_thought`).

        Args:
            model: Policy model (`LlamaLM` or wrapper).
            prefix_ids: `[B, T]` prefix token IDs $x_{<t}$.
            thought_ids: `[B, G, L]` or `[B, L]` discrete thought token IDs $z_t$.
            target_ids: `[B, m]` future target token IDs $x_{t:t+m-1}$.
            prefix_mask: Optional `[B, T]` mask (1=valid, 0=pad).
            ref_model: Optional frozen reference model for `without_thought` baseline.
                When provided, `without_thought` is evaluated under `torch.no_grad()`.
            include_delimiters: Wrap `thought_ids` with `<think>` and `</think>`.

        Returns:
            `ContinuationLogits(with_thought, without_thought)` where
            `with_thought` has shape `[B, G, m, V]` (or `[B, m, V]` if `thought_ids` is 2D)
            and `without_thought` has shape `[B, m, V]`.
        """
        base_lm = _unwrap_base_lm(model)
        prefix_ids, prefix_mask = _left_align_prefix(prefix_ids, prefix_mask, self.pad_id)
        start_id = self.think_start_id if think_start_id is None else think_start_id
        end_id = self.think_end_id if think_end_id is None else think_end_id

        bsz, t_len = prefix_ids.shape
        m = target_ids.shape[1]
        device = prefix_ids.device

        if thought_ids.ndim == 2:
            thought_3d = thought_ids.unsqueeze(1)
            squeeze_g = True
        elif thought_ids.ndim == 3:
            thought_3d = thought_ids
            squeeze_g = False
        else:
            raise ValueError(f"thought_ids must be 2D or 3D, got shape {tuple(thought_ids.shape)}")

        _, group_size, thought_len = thought_3d.shape
        bg = bsz * group_size

        prefix_exp = prefix_ids.repeat_interleave(group_size, dim=0)
        thought_flat = thought_3d.reshape(bg, thought_len)

        seq_parts = [prefix_exp]
        if include_delimiters:
            seq_parts.append(torch.full((bg, 1), start_id, device=device, dtype=torch.long))
        seq_parts.append(thought_flat)
        if include_delimiters:
            seq_parts.append(torch.full((bg, 1), end_id, device=device, dtype=torch.long))
        if m > 1:
            seq_parts.append(target_ids[:, :-1].repeat_interleave(group_size, dim=0))

        seq_with = torch.cat(seq_parts, dim=1)
        thought_total = thought_len + (2 if include_delimiters else 0)
        predict_start = t_len + thought_total - 1

        if prefix_mask is not None:
            mask_exp = prefix_mask.long().repeat_interleave(group_size, dim=0)
            extra_len = thought_total + max(m - 1, 0)
            full_mask: Optional[torch.Tensor] = torch.cat(
                (mask_exp, torch.ones((bg, extra_len), device=device, dtype=torch.long)),
                dim=1,
            )
            pos_ids: Optional[torch.Tensor] = (full_mask.cumsum(dim=1) - 1).clamp(min=0)
        else:
            full_mask = None
            pos_ids = None

        out_with = base_lm(input_ids=seq_with, attention_mask=full_mask, position_ids=pos_ids)
        logits_with = out_with.logits[:, predict_start : predict_start + m, :]
        logits_with = logits_with.view(bsz, group_size, m, -1)
        if squeeze_g:
            logits_with = logits_with.squeeze(1)

        if ref_model is not None:
            with torch.no_grad():
                logits_without = self.evaluate_nothought_logits(
                    ref_model, prefix_ids, target_ids, prefix_mask=prefix_mask
                )
        else:
            logits_without = self.evaluate_nothought_logits(
                base_lm, prefix_ids, target_ids, prefix_mask=prefix_mask
            )

        return ContinuationLogits(with_thought=logits_with, without_thought=logits_without)


def build_model(config: ModelConfig, pause_token_id: int = 5) -> nn.Module:
    """Factory constructing the configured LlamaLM or latent/pause wrapper."""
    base_lm = LlamaLM(config)
    variant = config.variant.strip().lower()
    if variant == "ntp":
        return base_lm
    if variant == "pause":
        return PauseTokenWrapper(
            base_lm=base_lm,
            n_pause_tokens=config.n_pause_tokens,
            pause_token_id=pause_token_id,
        )
    if variant == "continuous_latent":
        return ContinuousLatentWrapper(
            base_lm=base_lm,
            n_latent_steps=config.n_latent_steps,
            hybrid_gate=False,
        )
    if variant == "hybrid_latent":
        return ContinuousLatentWrapper(
            base_lm=base_lm,
            n_latent_steps=config.n_latent_steps,
            hybrid_gate=True,
        )
    raise ValueError(
        f"Unsupported model variant {config.variant!r}; expected one of "
        "('ntp', 'pause', 'continuous_latent', 'hybrid_latent')"
    )
