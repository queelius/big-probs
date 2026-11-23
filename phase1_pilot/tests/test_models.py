"""Unit tests for LlamaLM, FLOPs/parameter accounting, B3 pause, B4/I2 latent, and I1 rollouts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phase1_pilot.config import ModelConfig
from phase1_pilot.models import (
    ContinuousLatentWrapper,
    DiscreteThoughtHelper,
    LlamaLM,
    PauseTokenWrapper,
    build_model,
    count_parameters,
    count_parameters_breakdown,
    estimate_inference_flops_per_token,
    estimate_parameter_count,
)


def test_parameter_counts_micro_and_analytical_large_presets() -> None:
    """Verify instantiated micro-small/micro-large counts and analytical 160m/410m/1b counts."""
    cfg_small = ModelConfig.preset("micro-small")
    cfg_large = ModelConfig.preset("micro-large")

    m_small = LlamaLM(cfg_small)
    m_large = LlamaLM(cfg_large)

    est_small = estimate_parameter_count(cfg_small)
    est_large = estimate_parameter_count(cfg_large)

    assert count_parameters(m_small) == est_small["total"]
    assert count_parameters(m_small, non_embedding=True) == est_small["non_embedding"]
    assert count_parameters(m_large) == est_large["total"]
    assert count_parameters(m_large, non_embedding=True) == est_large["non_embedding"]

    bd_small = count_parameters_breakdown(m_small)
    assert bd_small["total"] == bd_small["embedding"] + bd_small["non_embedding"]

    # micro-small ~0.8M-1.5M, micro-large ~2.5x-3.0x larger
    assert 800_000 <= est_small["total"] <= 1_500_000
    ratio_micro = est_large["total"] / est_small["total"]
    assert 2.2 <= ratio_micro <= 3.2

    # Analytical checks for 160m / 410m / 1b without instantiating billion-param weights
    est_160m = estimate_parameter_count(ModelConfig.preset("160m"))
    est_410m = estimate_parameter_count(ModelConfig.preset("410m"))
    est_1b = estimate_parameter_count(ModelConfig.preset("1b"))

    assert 100_000_000 <= est_160m["total"] <= 200_000_000
    assert 300_000_000 <= est_410m["total"] <= 460_000_000
    assert 900_000_000 <= est_1b["total"] <= 1_200_000_000
    assert est_160m["total"] < est_410m["total"] < est_1b["total"]
    assert est_160m["non_embedding"] < est_410m["non_embedding"] < est_1b["non_embedding"]

    # Wrapper parameter counts match analytical estimate
    cfg_pause = ModelConfig.preset("micro-small", variant="pause", n_pause_tokens=4)
    assert count_parameters(build_model(cfg_pause)) == estimate_parameter_count(cfg_pause)["total"]

    cfg_cont = ModelConfig.preset("micro-small", variant="continuous_latent", n_latent_steps=4)
    assert count_parameters(build_model(cfg_cont)) == estimate_parameter_count(cfg_cont)["total"]

    cfg_hyb = ModelConfig.preset("micro-small", variant="hybrid_latent", n_latent_steps=4)
    assert count_parameters(build_model(cfg_hyb)) == estimate_parameter_count(cfg_hyb)["total"]


def test_kv_cache_equivalence_and_gqa() -> None:
    """Verify exact equivalence (max_abs_diff < 1e-4) between cached & uncached decoding."""
    torch.manual_seed(0)
    for n_kv_heads in (4, 2):  # Test both MHA and GQA
        cfg = ModelConfig(
            vocab_size=64,
            d_model=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=n_kv_heads,
            d_ff=128,
            max_seq_len=64,
        )
        model = LlamaLM(cfg).eval()
        input_ids = torch.randint(1, 64, (2, 16))

        with torch.no_grad():
            full_out = model(input_ids=input_ids, return_hidden=True)

            # Prefill first 10 tokens, then decode remaining 6 tokens one by one
            prefill = model(input_ids=input_ids[:, :10], use_cache=True, return_hidden=True)
            past_kvs = prefill.past_kvs
            step_logits = [prefill.logits]
            for t in range(10, 16):
                step_out = model(
                    input_ids=input_ids[:, t : t + 1],
                    past_kvs=past_kvs,
                    use_cache=True,
                )
                past_kvs = step_out.past_kvs
                step_logits.append(step_out.logits)
            cached_logits = torch.cat(step_logits, dim=1)

            # Also test multi-token chunk continuation with cache (10 + 6)
            chunk_out = model(
                input_ids=input_ids[:, 10:],
                past_kvs=prefill.past_kvs,
                use_cache=True,
            )
            chunk_logits = torch.cat([prefill.logits, chunk_out.logits], dim=1)

        diff_step = (full_out.logits - cached_logits).abs().max().item()
        diff_chunk = (full_out.logits - chunk_logits).abs().max().item()
        assert diff_step < 1e-4, f"Step KV cache mismatch: {diff_step}"
        assert diff_chunk < 1e-4, f"Chunk KV cache mismatch: {diff_chunk}"

        # Verify inputs_embeds forward matches input_ids forward identically
        with torch.no_grad():
            emb_out = model(inputs_embeds=model.tok_embeddings(input_ids))
        assert (full_out.logits - emb_out.logits).abs().max().item() < 1e-6


def test_pause_token_wrapper_loss_masking_and_gradients() -> None:
    """Verify B3 PauseTokenWrapper masks pause tokens from loss and trains pause embeddings."""
    torch.manual_seed(1)
    cfg = ModelConfig.preset("micro-small", variant="pause", n_pause_tokens=4)
    model = build_model(cfg)
    assert isinstance(model, PauseTokenWrapper)

    bsz, seq_len = 3, 20
    input_ids = torch.randint(8, cfg.vocab_size, (bsz, seq_len))
    prompt_lens = torch.tensor([10, 12, 14])

    # Answer-only labels (-100 on prompt)
    labels = torch.full_like(input_ids, -100)
    for b in range(bsz):
        p = int(prompt_lens[b].item())
        labels[b, p:] = input_ids[b, p:]

    out = model(input_ids=input_ids, prompt_lens=prompt_lens, labels=labels)
    assert out.logits.shape == (bsz, seq_len, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)

    # Aligned logits CE loss must match internal masked-pause loss exactly
    manual_loss = F.cross_entropy(
        out.logits[:, :-1, :].reshape(-1, cfg.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out.loss, manual_loss, atol=1e-5)

    out.loss.backward()
    assert model.pause_embeddings is not None
    assert model.pause_embeddings.grad is not None
    assert torch.isfinite(model.pause_embeddings.grad).all()
    assert model.pause_embeddings.grad.abs().sum().item() > 0.0
    # Every individual pause slot should receive non-zero gradient
    for slot in range(cfg.n_pause_tokens):
        assert model.pause_embeddings.grad[slot].abs().sum().item() > 0.0


def test_continuous_latent_wrapper_bptt_through_all_steps() -> None:
    """Verify B4 ContinuousLatentWrapper backpropagates through EVERY latent step."""
    torch.manual_seed(2)
    n_steps = 4
    cfg = ModelConfig.preset("micro-small", variant="continuous_latent", n_latent_steps=n_steps)
    model = build_model(cfg)
    assert isinstance(model, ContinuousLatentWrapper)

    bsz, seq_len = 3, 18
    input_ids = torch.randint(8, cfg.vocab_size, (bsz, seq_len))
    prompt_lens = torch.tensor([8, 10, 12])
    labels = torch.full_like(input_ids, -100)
    for b in range(bsz):
        p = int(prompt_lens[b].item())
        labels[b, p:] = input_ids[b, p:]

    # Capture intermediate latent vectors via hook to verify grad reaches ALL n_steps
    captured_latents: list[torch.Tensor] = []

    def capture_hook(z: torch.Tensor) -> torch.Tensor:
        z.retain_grad()
        captured_latents.append(z)
        return z

    out = model(
        input_ids=input_ids,
        prompt_lens=prompt_lens,
        labels=labels,
        return_latents=True,
        latent_perturb_fn=capture_hook,
    )
    assert out.logits.shape == (bsz, seq_len, cfg.vocab_size)
    assert out.latent_states is not None and out.latent_states.shape == (bsz, n_steps, cfg.d_model)
    assert out.loss is not None and torch.isfinite(out.loss)

    out.loss.backward()

    assert len(captured_latents) == n_steps
    for step_idx, z in enumerate(captured_latents):
        assert z.grad is not None, f"Missing grad at latent step {step_idx}"
        assert torch.isfinite(z.grad).all(), f"Non-finite grad at latent step {step_idx}"
        assert z.grad.abs().sum().item() > 0.0, f"Zero grad at latent step {step_idx}"

    assert model.latent_proj.weight.grad is not None
    assert model.latent_proj.weight.grad.abs().sum().item() > 0.0
    assert model.base_lm.tok_embeddings.weight.grad is not None
    assert torch.isfinite(model.base_lm.tok_embeddings.weight.grad).all()

    # Verify generate_latents + evaluate_with_latents matches forward target logits
    model.eval()
    uniform_prefix = input_ids[:, :10]
    uniform_target = input_ids[:, 10:14]
    with torch.no_grad():
        latents, _, _ = model.generate_latents(uniform_prefix)
        eval_logits = model.evaluate_with_latents(uniform_prefix, uniform_target, latents)
        full_out = model(input_ids=input_ids[:, :14], prompt_lens=10)
        fwd_target_logits = full_out.logits[:, 9:13, :]
    assert eval_logits.shape == (bsz, 4, cfg.vocab_size)
    assert (eval_logits - fwd_target_logits).abs().max().item() < 1e-4


def test_hybrid_latent_wrapper_hrpo_rollout_and_recompute() -> None:
    """Verify I2 hybrid_gate HRPO rollout matches differentiable recomputation."""
    torch.manual_seed(3)
    cfg = ModelConfig.preset("micro-small", variant="hybrid_latent", n_latent_steps=4)
    model = build_model(cfg)
    assert isinstance(model, ContinuousLatentWrapper) and model.hybrid_gate
    model.eval()

    bsz, t_len, m, group_size = 2, 10, 4, 4
    prefix_ids = torch.randint(8, cfg.vocab_size, (bsz, t_len))
    target_ids = torch.randint(8, cfg.vocab_size, (bsz, m))

    roll = model.rollout_hybrid(prefix_ids, group_size=group_size, temperature=0.9)
    assert roll.thought_ids.shape == (bsz, group_size, 4)
    assert roll.old_logprobs.shape == (bsz, group_size, 4)
    assert roll.token_entropies.shape == (bsz, group_size, 4)
    assert roll.latent_embeds.shape == (bsz, group_size, 4, cfg.d_model)
    assert (roll.old_logprobs <= 1e-6).all()
    assert (roll.token_entropies >= -1e-6).all()

    new_lps, new_ents, cont_logits = model.recompute_hybrid_logprobs_and_continuation(
        prefix_ids, roll.thought_ids, target_ids, temperature=0.9
    )
    assert cont_logits.shape == (bsz, group_size, m, cfg.vocab_size)
    assert (new_lps - roll.old_logprobs).abs().max().item() < 1e-4
    assert (new_ents - roll.token_entropies).abs().max().item() < 1e-4

    # Verify gradients flow into gate_proj, latent_proj, and base_lm
    loss = -new_lps.mean() + cont_logits.pow(2).mean() * 1e-3
    loss.backward()
    assert model.gate_proj is not None and model.gate_proj.weight.grad is not None
    assert model.gate_proj.weight.grad.abs().sum().item() > 0.0
    assert model.latent_proj.weight.grad is not None
    assert model.latent_proj.weight.grad.abs().sum().item() > 0.0


def test_discrete_thought_helper_rollout_recompute_and_continuation() -> None:
    """Verify DiscreteThoughtHelper rollout shapes, logprob agreement, and continuation logits."""
    torch.manual_seed(4)
    cfg = ModelConfig.preset("micro-small")
    model = LlamaLM(cfg).eval()
    helper = DiscreteThoughtHelper(think_start_id=3, think_end_id=4, pad_id=0)

    bsz, t_len, thought_len, group_size, m = 2, 12, 6, 4, 4
    prefix_ids = torch.randint(8, cfg.vocab_size, (bsz, t_len))
    target_ids = torch.randint(8, cfg.vocab_size, (bsz, m))

    thought_ids, old_logprobs, token_entropies = helper.sample_thoughts(
        model=model,
        prefix_ids=prefix_ids,
        thought_len=thought_len,
        group_size=group_size,
        temperature=0.8,
    )
    assert thought_ids.shape == (bsz, group_size, thought_len)
    assert old_logprobs.shape == (bsz, group_size, thought_len)
    assert token_entropies.shape == (bsz, group_size, thought_len)
    assert torch.isfinite(old_logprobs).all() and (old_logprobs <= 1e-6).all()
    assert torch.isfinite(token_entropies).all() and (token_entropies >= -1e-6).all()

    # Cached sampling logprobs must exactly match teacher-forced recomputation
    recomp_lps, recomp_ents = helper.recompute_thought_logprobs(
        model=model,
        prefix_ids=prefix_ids,
        thought_ids=thought_ids,
        temperature=0.8,
    )
    assert (recomp_lps - old_logprobs).abs().max().item() < 1e-4
    assert (recomp_ents - token_entropies).abs().max().item() < 1e-4

    # Evaluate continuation logits WITH and WITHOUT thought
    logits_with, logits_without = helper.evaluate_continuation_logits(
        model=model,
        prefix_ids=prefix_ids,
        thought_ids=thought_ids,
        target_ids=target_ids,
    )
    assert logits_with.shape == (bsz, group_size, m, cfg.vocab_size)
    assert logits_without.shape == (bsz, m, cfg.vocab_size)

    # Cross-check against explicit single-sequence forward pass for b=0, g=1
    with torch.no_grad():
        seq_manual = torch.cat(
            [
                prefix_ids[0:1],
                torch.tensor([[3]]),
                thought_ids[0:1, 1, :],
                torch.tensor([[4]]),
                target_ids[0:1, :-1],
            ],
            dim=1,
        )
        manual_with = model(input_ids=seq_manual).logits[:, -m:, :]
        seq_nt = torch.cat([prefix_ids[0:1], target_ids[0:1, :-1]], dim=1)
        manual_nt = model(input_ids=seq_nt).logits[:, t_len - 1 : t_len - 1 + m, :]

    assert (logits_with[0:1, 1] - manual_with).abs().max().item() < 1e-4
    assert (logits_without[0:1] - manual_nt).abs().max().item() < 1e-4

    # Verify right-padded prefix_mask matches unpadded single-example evaluation
    padded_prefix = torch.zeros((2, t_len + 4), dtype=torch.long)
    padded_mask = torch.zeros((2, t_len + 4), dtype=torch.long)
    padded_prefix[0, :t_len] = prefix_ids[0]
    padded_mask[0, :t_len] = 1
    padded_prefix[1, : t_len - 2] = prefix_ids[1, : t_len - 2]
    padded_mask[1, : t_len - 2] = 1
    with torch.no_grad():
        pad_with, pad_nt = helper.evaluate_continuation_logits(
            model, padded_prefix, thought_ids, target_ids, prefix_mask=padded_mask
        )
    assert (pad_nt[0:1] - manual_nt).abs().max().item() < 1e-4
    assert (pad_with[0:1, 1] - manual_with).abs().max().item() < 1e-4

    # Verify float amortized extra_steps works cleanly in FLOPs estimator
    f_frac = estimate_inference_flops_per_token(cfg, seq_len=64, extra_steps=1.6)
    assert (
        estimate_inference_flops_per_token(cfg, seq_len=64, extra_steps=1)
        < f_frac
        < estimate_inference_flops_per_token(cfg, seq_len=64, extra_steps=2)
    )


def test_flops_accounting_monotonicity_and_scale_comparisons() -> None:
    """Verify exact FLOPs accounting monotonicity across extra_steps, seq_len, and scale."""
    cfg_small = ModelConfig.preset("micro-small")
    cfg_large = ModelConfig.preset("micro-large")
    cfg_160m = ModelConfig.preset("160m")
    cfg_410m = ModelConfig.preset("410m")
    cfg_1b = ModelConfig.preset("1b")

    # Monotonic in extra_steps
    f0 = estimate_inference_flops_per_token(cfg_small, seq_len=64, extra_steps=0)
    f4 = estimate_inference_flops_per_token(cfg_small, seq_len=64, extra_steps=4)
    f8 = estimate_inference_flops_per_token(cfg_small, seq_len=64, extra_steps=8)
    assert 0.0 < f0 < f4 < f8
    # 4 extra steps = 5 total forward passes -> strictly > 5x single pass due to KV growth
    assert f4 > 5.0 * f0

    # Monotonic in seq_len
    f_short = estimate_inference_flops_per_token(cfg_small, seq_len=32, extra_steps=2)
    f_long = estimate_inference_flops_per_token(cfg_small, seq_len=128, extra_steps=2)
    assert f_short < f_long

    # Monotonic in model scale
    assert (
        estimate_inference_flops_per_token(cfg_small, seq_len=64)
        < estimate_inference_flops_per_token(cfg_large, seq_len=64)
        < estimate_inference_flops_per_token(cfg_160m, seq_len=64)
        < estimate_inference_flops_per_token(cfg_410m, seq_len=64)
        < estimate_inference_flops_per_token(cfg_1b, seq_len=64)
    )

    # FLOPs-matched sanity check: 160M with ~2 extra steps approaches 410M 1-step FLOPs
    f160_base = estimate_inference_flops_per_token(cfg_160m, seq_len=256, extra_steps=0)
    f160_2step = estimate_inference_flops_per_token(cfg_160m, seq_len=256, extra_steps=2)
    f410_base = estimate_inference_flops_per_token(cfg_410m, seq_len=256, extra_steps=0)
    assert f160_base < f410_base < f160_2step * 1.5

    with pytest.raises(ValueError):
        estimate_inference_flops_per_token(cfg_small, seq_len=0)
    with pytest.raises(ValueError):
        estimate_inference_flops_per_token(cfg_small, seq_len=32, extra_steps=-1)
