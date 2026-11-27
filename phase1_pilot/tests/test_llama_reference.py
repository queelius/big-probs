"""`LlamaLM` must compute what a reference Llama computes with the same weights.

`EXP-006` found that micro models never leave the prompt-blind floor on Regime 1. This test
rules out the backbone as the cause and guards it for every larger preset: copied into
`transformers`' `LlamaForCausalLM`, our weights give the same logits. Skipped when
`transformers` is not installed (run the suite with `--with transformers==5.24.0`).
"""

from __future__ import annotations

import pytest
import torch

from phase1_pilot.config import ModelConfig
from phase1_pilot.models.llama import LlamaLM

transformers = pytest.importorskip("transformers")


def _reference_copy(ours: LlamaLM, cfg: ModelConfig, rope_theta: float | None = None):
    hf_cfg = transformers.LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps, rope_theta=cfg.rope_theta if rope_theta is None else rope_theta,
        tie_word_embeddings=False, attention_bias=False, mlp_bias=False, hidden_act="silu",
    )
    hf = transformers.LlamaForCausalLM(hf_cfg).eval()
    sd = {
        "model.embed_tokens.weight": ours.tok_embeddings.weight,
        "model.norm.weight": ours.norm.weight,
        "lm_head.weight": ours.lm_head.weight,
    }
    for i, layer in enumerate(ours.layers):
        p = f"model.layers.{i}."
        for a in ("q", "k", "v", "o"):
            sd[p + f"self_attn.{a}_proj.weight"] = getattr(layer.attn, f"{a}_proj").weight
        for m in ("gate", "up", "down"):
            sd[p + f"mlp.{m}_proj.weight"] = getattr(layer.mlp, f"{m}_proj").weight
        sd[p + "input_layernorm.weight"] = layer.attn_norm.weight
        sd[p + "post_attention_layernorm.weight"] = layer.ffn_norm.weight
    missing, unexpected = hf.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m]
    return hf


def _model(n_kv_heads: int) -> tuple[LlamaLM, ModelConfig]:
    torch.manual_seed(0)
    cfg = ModelConfig.preset("micro-small", vocab_size=512, max_seq_len=128, variant="ntp", n_kv_heads=n_kv_heads)
    ours = LlamaLM(cfg).eval()
    with torch.no_grad():  # non-unit norms, so a swapped or dropped norm weight shows
        for name, p in ours.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.1 * torch.randn_like(p))
    return ours, cfg


@pytest.mark.parametrize("n_kv_heads", [4, 2])  # multi-head and grouped-query attention
def test_logits_match_reference_llama(n_kv_heads):
    ours, cfg = _model(n_kv_heads)
    hf = _reference_copy(ours, cfg)
    ids = torch.randint(8, cfg.vocab_size, (3, 90))
    mask = torch.ones_like(ids)
    mask[1, 50:] = 0
    mask[2, 20:] = 0
    with torch.no_grad():
        assert torch.allclose(ours(ids).logits, hf(ids).logits, atol=1e-5)
        a, b = ours(ids, attention_mask=mask).logits, hf(ids, attention_mask=mask).logits
    valid = mask.bool()
    assert torch.allclose(a[valid], b[valid], atol=1e-5)


def test_parity_check_detects_a_rope_mismatch():
    ours, cfg = _model(4)
    hf = _reference_copy(ours, cfg, rope_theta=500000.0)
    ids = torch.randint(8, cfg.vocab_size, (2, 90))
    with torch.no_grad():
        assert (ours(ids).logits - hf(ids).logits).abs().max() > 1e-3
