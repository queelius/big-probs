"""Llama-style decoder-only Transformer backbone and exact param/FLOPs accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_pilot.config import ModelConfig


@dataclass
class LMOutput:
    """Unified output container for LlamaLM and latent/pause wrappers."""

    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
    loss: Optional[torch.Tensor] = None
    latent_states: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-style)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normed.to(dtype=x.dtype)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: [B, n_heads, T, head_dim]
        k: [B, n_kv_heads, T, head_dim]
        cos: broadcastable to [B, 1, T, head_dim]
        sin: broadcastable to [B, 1, T, head_dim]
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embeddings supporting offset & custom positions."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached: int = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        self.max_seq_len_cached = max(seq_len, 1)
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=torch.float32,
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, head_dim]
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def get_cos_sin(
        self,
        seq_len: int,
        pos_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) broadcastable to [B, 1, T, head_dim]."""
        if position_ids is not None:
            needed = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else 1
        else:
            needed = pos_offset + seq_len

        if needed > self.max_seq_len_cached:
            self._build_cache(max(needed, self.max_seq_len_cached * 2))

        if device is not None and self.cos_cached.device != device:
            self.cos_cached = self.cos_cached.to(device=device)
            self.sin_cached = self.sin_cached.to(device=device)

        if position_ids is not None:
            pos = position_ids.clamp(min=0)
            cos = self.cos_cached[pos]  # [B, T, D] or [T, D]
            sin = self.sin_cached[pos]
            if cos.ndim == 2:
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)  # [B, 1, T, D]
                sin = sin.unsqueeze(1)
        else:
            cos = self.cos_cached[pos_offset : pos_offset + seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[pos_offset : pos_offset + seq_len].unsqueeze(0).unsqueeze(0)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class GroupedQueryAttention(nn.Module):
    """Multi-head / Grouped-Query Self-Attention with causal masking & KV caching."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bsz, n_kv, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, n_kv, self.n_rep, seq_len, head_dim)
            .reshape(bsz, n_kv * self.n_rep, seq_len, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.shape

        q = self.q_proj(x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_kv = (k, v) if use_cache else None

        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)
        kv_len = k_rep.shape[2]

        dropout_p = self.dropout if self.training else 0.0

        if attention_mask is None:
            if past_len == 0 and q_len > 1:
                attn_out = F.scaled_dot_product_attention(
                    q, k_rep, v_rep, attn_mask=None, dropout_p=dropout_p, is_causal=True
                )
            elif q_len == 1:
                attn_out = F.scaled_dot_product_attention(
                    q, k_rep, v_rep, attn_mask=None, dropout_p=dropout_p, is_causal=False
                )
            else:
                causal = torch.tril(
                    torch.ones((q_len, kv_len), device=x.device, dtype=torch.bool),
                    diagonal=past_len,
                )
                attn_bias = torch.zeros((1, 1, q_len, kv_len), device=x.device, dtype=q.dtype)
                attn_bias.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn_out = F.scaled_dot_product_attention(
                    q, k_rep, v_rep, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
                )
        else:
            causal = torch.tril(
                torch.ones((q_len, kv_len), device=x.device, dtype=torch.bool),
                diagonal=past_len,
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

            if attention_mask.ndim == 2:
                # [B, kv_len] where 1/True = attend, 0/False = pad
                key_valid = attention_mask[:, :kv_len].bool().unsqueeze(1).unsqueeze(2)
                allow = causal & key_valid
            elif attention_mask.ndim == 3:
                allow = causal & attention_mask.bool().unsqueeze(1)
            elif attention_mask.ndim == 4:
                allow = causal & attention_mask.bool()
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

            # Prevent NaN softmax/backward on fully masked pad query rows (e.g. left-padding)
            empty_rows = ~allow.any(dim=-1, keepdim=True)
            if empty_rows.any():
                fallback = torch.zeros_like(allow)
                fallback[..., 0] = True
                allow = torch.where(empty_rows, fallback, allow)

            attn_bias = torch.zeros((bsz, 1, q_len, kv_len), device=x.device, dtype=q.dtype)
            attn_bias.masked_fill_(~allow, float("-inf"))
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.o_proj(attn_out), present_kv


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (Llama-style)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Llama Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present_kv = self.attn(
            self.attn_norm(x),
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        h = x + attn_out
        out = h + self.mlp(self.ffn_norm(h))
        return out, present_kv


class LlamaLM(nn.Module):
    """Decoder-only Llama-style Language Model supporting token & continuous inputs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        residual_scale = 1.0 / math.sqrt(2.0 * max(self.config.n_layers, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=0.02 * residual_scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=0.02 * residual_scale)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> LMOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            assert input_ids is not None
            h = self.tok_embeddings(input_ids)

        seq_len = h.shape[1]
        past_len = 0
        if past_kvs is not None and len(past_kvs) > 0 and past_kvs[0] is not None:
            past_len = past_kvs[0][0].shape[2]

        cos, sin = self.rope.get_cos_sin(
            seq_len=seq_len,
            pos_offset=past_len,
            position_ids=position_ids,
            device=h.device,
            dtype=h.dtype,
        )

        new_past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )

        for idx, layer in enumerate(self.layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            h, present_kv = layer(
                h,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                past_kv=layer_past,
                use_cache=use_cache,
            )
            if use_cache and new_past_kvs is not None:
                assert present_kv is not None
                new_past_kvs.append(present_kv)

        hidden_states = self.norm(h)
        logits = self.lm_head(hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None and seq_len > 1:
            shift_logits = logits[:, :-1, :].contiguous()
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
            logits=logits,
            hidden_states=hidden_states if return_hidden else None,
            past_kvs=new_past_kvs,
            loss=loss,
        )


def count_parameters_breakdown(model: nn.Module) -> dict[str, int]:
    """Return exact parameter counts broken down into total, embedding, and non-embedding."""
    total = 0
    embedding = 0
    seen: set[int] = set()

    for module in model.modules():
        is_emb = isinstance(module, nn.Embedding)
        for param in module.parameters(recurse=False):
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            n = param.numel()
            total += n
            if is_emb:
                embedding += n

    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": total - embedding,
    }


def count_parameters(model: nn.Module, non_embedding: bool = False) -> int:
    """Count total (or non-embedding) parameters of a model."""
    breakdown = count_parameters_breakdown(model)
    return breakdown["non_embedding"] if non_embedding else breakdown["total"]


def estimate_parameter_count(config: ModelConfig) -> dict[str, int]:
    """Analytically compute exact parameter counts for a LlamaLM (and wrapper overhead)
    directly from ModelConfig without instantiating weights (safe for 160M/410M/1B+)."""
    d = config.d_model
    d_kv = config.n_kv_heads * config.head_dim
    d_ff = config.d_ff
    n_layers = config.n_layers
    vocab = config.vocab_size

    attn_per_layer = d * (d + 2 * d_kv + d)
    mlp_per_layer = 3 * d * d_ff
    norms_per_layer = 2 * d
    transformer_blocks = n_layers * (attn_per_layer + mlp_per_layer + norms_per_layer)
    final_norm = d
    embedding = vocab * d
    lm_head = d * vocab

    extra_wrapper = 0
    if config.variant == "pause":
        extra_wrapper = config.n_pause_tokens * d
    elif config.variant == "continuous_latent":
        extra_wrapper = d + d * d  # latent_norm + latent_proj
    elif config.variant == "hybrid_latent":
        extra_wrapper = d + d * d + (2 * d * d + d)  # + gate_proj (weight + bias)

    non_embedding = transformer_blocks + final_norm + lm_head + extra_wrapper
    total = non_embedding + embedding

    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": non_embedding,
        "transformer_blocks": transformer_blocks,
        "lm_head": lm_head,
        "wrapper_extra": extra_wrapper,
    }


def default_extra_steps_for_config(config: ModelConfig) -> int:
    """Return the default extra inference steps per target prediction implied by config.variant."""
    if config.variant == "pause":
        return config.n_pause_tokens
    if config.variant in ("continuous_latent", "hybrid_latent"):
        return config.n_latent_steps
    return 0


def estimate_inference_flops_per_token(
    config: ModelConfig,
    seq_len: int,
    extra_steps: float = 0,
) -> float:
    """Estimate exact autoregressive inference FLOPs per predicted target token.

    Counts 2 FLOPs per multiply-accumulate (MAC) across:
      - Q, K, V, O linear projections per layer: 2 * d * (2*d + 2*d_kv)
      - Causal attention QK^T and Attn*V over active KV cache length S: 4 * d * S
      - SwiGLU MLP (gate, up, down) per layer: 6 * d * d_ff
      - Output LM head projection: 2 * d * vocab_size

    When `extra_steps > 0` (pause tokens, continuous latent steps, or discrete thought
    tokens—integer or amortized float—per predicted target token), evaluates the exact
    closed-form sum over all `1 + extra_steps` forward passes including KV-cache growth
    `S = seq_len + s` for `s = 0 .. extra_steps`:
      sum_{s=0}^{E} (seq_len + s) = (E + 1) * (seq_len + E / 2).
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if extra_steps < 0:
        raise ValueError(f"extra_steps must be >= 0, got {extra_steps}")

    d = float(config.d_model)
    d_kv = float(config.n_kv_heads * config.head_dim)
    d_ff = float(config.d_ff)
    n_layers = float(config.n_layers)
    vocab = float(config.vocab_size)
    e = float(extra_steps)

    proj_per_layer = 2.0 * d * (2.0 * d + 2.0 * d_kv)
    mlp_per_layer = 6.0 * d * d_ff
    lm_head_flops = 2.0 * d * vocab

    adapter_flops = 0.0
    if config.variant == "continuous_latent":
        adapter_flops = 2.0 * d * d
    elif config.variant == "hybrid_latent":
        adapter_flops = 2.0 * d * d + 4.0 * d * d  # latent_proj + gate_proj(2d -> d)

    n_passes = e + 1.0
    base_non_attn_per_pass = n_layers * (proj_per_layer + mlp_per_layer) + lm_head_flops
    sum_ctx = n_passes * (float(seq_len) + 0.5 * e)
    attn_total = n_layers * 4.0 * d * sum_ctx

    return n_passes * base_non_attn_per_pass + attn_total + e * adapter_flops
