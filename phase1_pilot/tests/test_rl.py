"""Comprehensive unit & integration tests for SSL-as-RL rewards, filters, Dr. GRPO + DAPO, and train.py."""

from __future__ import annotations

import math

import pytest
import torch

from phase1_pilot.config import ModelConfig, RLConfig, TrainConfig
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
from phase1_pilot.train import build_arg_parser, train_ssl_rl, train_supervised


# ---------------------------------------------------------------------------
# 1. Exact Brier Score: Boundedness in [-2, 0] & Strict Propriety
# ---------------------------------------------------------------------------


def test_brier_score_bounds_and_exact_formula():
    torch.manual_seed(0)
    V = 32
    logits = torch.randn(4, 5, V) * 3.0
    targets = torch.randint(0, V, (4, 5))

    bs = compute_brier_score(logits, targets)
    assert bs.shape == (4, 5)
    assert torch.all(bs >= -2.0) and torch.all(bs <= 0.0)

    # Verify exact algebraic equivalence to -(1 - q(v*))^2 - sum_{v != v*} q(v)^2
    probs = torch.softmax(logits.float(), dim=-1)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=V).float()
    manual_bs = -((probs - one_hot) ** 2).sum(dim=-1)
    assert torch.allclose(bs, manual_bs, atol=1e-6)

    # Extreme corners: perfect confidence on target -> 0.0; perfect confidence on wrong token -> -2.0
    perfect_logits = torch.full((1, 4), -100.0)
    perfect_logits[0, 2] = 100.0
    assert math.isclose(compute_brier_score(perfect_logits, torch.tensor([2])).item(), 0.0, abs_tol=1e-6)
    assert math.isclose(compute_brier_score(perfect_logits, torch.tensor([0])).item(), -2.0, abs_tol=1e-6)


def test_brier_score_strictly_proper_numerically():
    """Verify E_{v* ~ p_true}[BS(q, v*)] is uniquely maximized at q = p_true."""
    torch.manual_seed(123)
    V = 12
    p_true = torch.softmax(torch.randn(V) * 1.5, dim=-1)

    def expected_brier(q_probs: torch.Tensor) -> float:
        logits = torch.log(q_probs.clamp(min=1e-12))
        all_targets = torch.arange(V)
        scores = compute_brier_score(logits.unsqueeze(0).expand(V, V), all_targets)
        return float((p_true * scores).sum().item())

    opt_score = expected_brier(p_true)

    # 1. Random simplex perturbations must be strictly worse than p_true
    for _ in range(100):
        noise = torch.randn(V) * 0.25
        q_pert = torch.softmax(torch.log(p_true) + noise, dim=-1)
        if torch.norm(q_pert - p_true).item() > 1e-4:
            assert expected_brier(q_pert) < opt_score

    # 2. Gradient ascent on expected Brier over free logits recovers p_true
    q_logits = torch.zeros(V, requires_grad=True)
    opt = torch.optim.Adam([q_logits], lr=0.1)
    all_targets = torch.arange(V)
    for _ in range(400):
        opt.zero_grad()
        scores = compute_brier_score(q_logits.unsqueeze(0).expand(V, V), all_targets)
        loss = -(p_true * scores).sum()
        loss.backward()
        opt.step()

    recovered = torch.softmax(q_logits.detach(), dim=-1)
    assert torch.allclose(recovered, p_true, atol=1e-3)


# ---------------------------------------------------------------------------
# 2. Rewards R1 / R2 / R3 & Anti-Sabotage Stop-Gradient Invariant
# ---------------------------------------------------------------------------


def test_compute_reward_modes_and_stop_gradient():
    torch.manual_seed(7)
    B, G, m, V = 3, 4, 4, 16
    logits_thought = torch.randn(B, G, m, V, requires_grad=True)
    logits_ref = torch.randn(B, m, V, requires_grad=True)
    targets = torch.randint(0, V, (B, m))

    r1 = compute_reward(logits_thought, None, targets, reward_type="R1_exact")
    r2 = compute_reward(logits_thought, logits_ref, targets, reward_type="R2_log", tau=5.0)
    r3 = compute_reward(logits_thought, logits_ref, targets, reward_type="R3_brier")

    assert r1.shape == (B, G)
    assert r2.shape == (B, G)
    assert r3.shape == (B, G)

    # Anti-sabotage invariant: rewards MUST be strictly detached
    for r in (r1, r2, r3):
        assert not r.requires_grad
        assert r.grad_fn is None

    # Bounds: R1 in [0, 1], R2 in [-tau, +tau], R3 in [-2, +2]
    assert torch.all((r1 >= 0.0) & (r1 <= 1.0))
    assert torch.all((r2 >= -5.0) & (r2 <= 5.0))
    assert torch.all((r3 >= -2.0) & (r3 <= 2.0))

    # Identical thought and ref logits -> exactly 0.0 gain for R2 and R3
    r2_zero = compute_reward(logits_ref.unsqueeze(1).expand(B, G, m, V), logits_ref, targets, "R2_log")
    r3_zero = compute_reward(logits_ref.unsqueeze(1).expand(B, G, m, V), logits_ref, targets, "R3_brier")
    assert torch.allclose(r2_zero, torch.zeros_like(r2_zero), atol=1e-6)
    assert torch.allclose(r3_zero, torch.zeros_like(r3_zero), atol=1e-6)

    # R2 clipping check with extreme logits
    extreme_good = torch.full((1, 1, 2, 4), -50.0)
    extreme_good[..., 0] = 50.0
    extreme_bad = torch.full((1, 2, 4), -50.0)
    extreme_bad[..., 1] = 50.0
    r2_clipped = compute_reward(extreme_good, extreme_bad, torch.zeros(1, 2, dtype=torch.long), "R2_log", tau=5.0)
    assert math.isclose(r2_clipped.item(), 5.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 3. Token Filters (T0 / T1 / T2-self teacher-free / T2-scale) & Thought Mask
# ---------------------------------------------------------------------------


def test_filters_t0_t1_t2_self_t2_scale():
    torch.manual_seed(11)
    B, V = 10, 16

    # T0_all selects every valid position
    valid = torch.tensor([True] * 8 + [False, False])
    m0 = select_tokens("T0_all", valid_mask=valid)
    assert torch.equal(m0, valid)

    # T1_entropy selects top-20% highest entropy positions
    logits_ref = torch.zeros(B, V)
    # Make positions 0..7 peaky (low entropy) and 8, 9 uniform (max entropy)
    logits_ref[:8, 0] = 20.0
    m1 = select_tokens("T1_entropy", top_p_tokens=0.2, logits_ref=logits_ref)
    assert m1.sum().item() == 2
    assert m1[8].item() and m1[9].item()
    assert torch.all(compute_predictive_entropy(logits_ref[8:]) > compute_predictive_entropy(logits_ref[:8]).max())

    # T2_self: strictly teacher-free and drops sigma=0 groups cleanly
    group_rewards = torch.zeros(B, 8)
    # Only positions 2 and 5 have non-zero variance; position 5 has larger variance
    group_rewards[2] = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
    group_rewards[5] = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    m2_self = select_tokens("T2_self", top_p_tokens=0.2, group_rewards=group_rewards)
    assert m2_self.sum().item() == 2
    assert m2_self[2].item() and m2_self[5].item()
    # All-zero variance -> selects ZERO groups (never leaks sigma=0 groups)
    m2_zero = select_tokens("T2_self", top_p_tokens=0.5, group_rewards=torch.ones(B, 8) * 0.42)
    assert m2_zero.sum().item() == 0
    # Passing teacher logits to T2_self raises ValueError (zero teacher leakage)
    with pytest.raises(ValueError, match="teacher-free"):
        select_tokens("T2_self", group_rewards=group_rewards, logits_ref_large=logits_ref)

    # T2_scale selects positions with largest cross-scale log-prob gap
    targets = torch.zeros(B, dtype=torch.long)
    logits_small = torch.zeros(B, V)
    logits_large = torch.zeros(B, V)
    logits_large[3, 0] = 10.0
    logits_large[7, 0] = 8.0
    gap = compute_cross_scale_gap(logits_large, logits_small, targets)
    assert gap[3] > gap[7] > gap[0]
    m2_scale = select_tokens(
        "T2_scale",
        top_p_tokens=0.2,
        logits_ref_large=logits_large,
        logits_small=logits_small,
        targets=targets,
    )
    assert m2_scale.sum().item() == 2
    assert m2_scale[3].item() and m2_scale[7].item()


def test_thought_entropy_mask():
    ent = torch.tensor([[[0.1, 0.2, 0.9, 0.3, 1.5, 0.0, 0.4, 0.8, 0.2, 0.1]]])  # [1, 1, 10]
    mask = select_thought_entropy_mask(ent, top_p=0.2)
    assert mask.shape == (1, 1, 10)
    assert mask.sum().item() == 2
    # Highest entropies are at index 4 (1.5) and index 2 (0.9)
    assert mask[0, 0, 4].item() and mask[0, 0, 2].item()


# ---------------------------------------------------------------------------
# 4. Dr. GRPO Length-Debiased Advantage & DAPO Asymmetric Clip-Higher
# ---------------------------------------------------------------------------


def test_dr_grpo_advantage_no_std_division_and_length_debiasing():
    # Scaling rewards by 10x must scale Dr. GRPO advantages by 10x (proving NO std division!)
    r_small = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    r_large = r_small * 10.0
    adv_small = compute_dr_grpo_advantages(r_small)
    adv_large = compute_dr_grpo_advantages(r_large)
    assert torch.allclose(adv_small.mean(dim=-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(adv_large, adv_small * 10.0, atol=1e-5)

    # Length-debiased normalization: short (len=2) vs long (len=8) trajectories both
    # divide by fixed max_thought_len=8 so per-token gradient magnitude is IDENTICAL!
    old_lp = torch.zeros(1, 2, 8)
    new_lp = torch.zeros(1, 2, 8, requires_grad=True)
    adv = torch.tensor([[1.0, -1.0]])
    mask = torch.zeros(1, 2, 8, dtype=torch.bool)
    mask[0, 0, :2] = True  # short trajectory (2 tokens)
    mask[0, 1, :8] = True  # long trajectory (8 tokens)

    loss, _ = compute_dapo_pg_loss(
        new_logprobs=new_lp,
        old_logprobs=old_lp,
        advantages=adv,
        token_mask=mask,
        max_thought_len=8,
        clip_low=0.2,
        clip_high=0.28,
    )
    loss.backward()
    grad_short_tok = abs(new_lp.grad[0, 0, 0].item())
    grad_long_tok = abs(new_lp.grad[0, 1, 0].item())
    # Equal per-token gradient weight (1 / (1 * 2 * 8) = 1/16), NOT 4x larger on short seq!
    assert math.isclose(grad_short_tok, grad_long_tok, rel_tol=1e-5)
    assert math.isclose(grad_short_tok, 1.0 / 16.0, rel_tol=1e-5)


def test_dapo_asymmetric_clip_higher():
    """Verify clip_low=0.2 vs clip_high=0.28 behaves asymmetrically."""
    old_lp = torch.zeros(1, 1, 1)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)

    # Positive advantage with ratio = 1.25:
    # Under symmetric 0.2 clipping it WOULD clip; under DAPO clip_high=0.28 it does NOT clip!
    ratio_125 = torch.tensor([[[math.log(1.25)]]], requires_grad=True)
    loss_125, clip_125 = compute_dapo_pg_loss(
        ratio_125, old_lp, torch.tensor([[1.0]]), mask, max_thought_len=1, clip_low=0.2, clip_high=0.28
    )
    assert clip_125 == 0.0
    assert math.isclose(loss_125.item(), -1.25, rel_tol=1e-5)

    # Positive advantage with ratio = 1.35 (> 1.28) DOES clip at 1.28!
    ratio_135 = torch.tensor([[[math.log(1.35)]]], requires_grad=True)
    loss_135, clip_135 = compute_dapo_pg_loss(
        ratio_135, old_lp, torch.tensor([[1.0]]), mask, max_thought_len=1, clip_low=0.2, clip_high=0.28
    )
    assert clip_135 == 1.0
    assert math.isclose(loss_135.item(), -1.28, rel_tol=1e-5)

    # Negative advantage with ratio = 0.75 (< 0.80) DOES clip at 0.80!
    ratio_075 = torch.tensor([[[math.log(0.75)]]], requires_grad=True)
    loss_075, clip_075 = compute_dapo_pg_loss(
        ratio_075, old_lp, torch.tensor([[-1.0]]), mask, max_thought_len=1, clip_low=0.2, clip_high=0.28
    )
    assert clip_075 == 1.0
    assert math.isclose(loss_075.item(), 0.80, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 5. End-to-End SSLRLTrainer & train.py Smoke Tests on CPU (micro-small)
# ---------------------------------------------------------------------------


def _make_synthetic_batch(batch_size: int = 4, seq_len: int = 24, prompt_len: int = 16, vocab_size: int = 512):
    input_ids = torch.randint(8, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    prompt_lens = torch.full((batch_size,), prompt_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_lens": prompt_lens,
    }


def test_ssl_rl_trainer_anti_sabotage_and_smoke_micro_small():
    """End-to-end 3-step SSLRLTrainer smoke test on CPU with micro-small + anti-sabotage check."""
    from phase1_pilot.models import build_model

    torch.manual_seed(42)
    mcfg = ModelConfig.preset("micro-small", variant="ntp")
    model = build_model(mcfg)
    rl_cfg = RLConfig(
        reward_type="R3_brier",
        filter_type="T2_self",
        interface="I1_discrete",
        group_size=4,
        n_ahead=4,
        thought_len=6,
        lambda_ntp=0.2,
        top_p_tokens=0.5,
        top_p_thought_entropy=0.25,
        ref_ema_decay=1.0,
    )
    trainer = SSLRLTrainer(model=model, rl_cfg=rl_cfg, device="cpu")

    # Verify ref_model is strictly frozen
    assert all(not p.requires_grad for p in trainer.ref_model.parameters())
    ref_weights_before = [p.detach().clone() for p in trainer.ref_model.parameters()]

    batch = _make_synthetic_batch(batch_size=4, seq_len=24, prompt_len=16, vocab_size=mcfg.vocab_size)
    required_keys = {
        "loss_pg",
        "loss_ntp",
        "mean_reward",
        "policy_entropy",
        "zero_var_frac",
        "nothought_nll_drift",
        "mean_thought_len",
        "clip_frac",
    }

    for step in range(3):
        metrics = trainer.train_step(batch)
        assert required_keys.issubset(metrics.keys())
        for k, v in metrics.items():
            assert math.isfinite(v), f"Non-finite metric {k}={v} at step {step}"
        # With lambda_ntp > 0, no-thought NLL must stay bounded and not explode
        assert metrics["nothought_nll"] < 20.0
        assert abs(metrics["nothought_nll_drift"]) < 10.0

    # Anti-sabotage: ref_model weights must be bit-for-bit unchanged and have zero grads
    for p_before, p_after in zip(ref_weights_before, trainer.ref_model.parameters()):
        assert p_after.grad is None
        assert torch.equal(p_before, p_after)


def test_ssl_rl_trainer_i2_hybrid_smoke():
    """Verify I2_hybrid continuous latent + HRPO-style gated RL executes cleanly on CPU."""
    from phase1_pilot.models import build_model

    torch.manual_seed(99)
    mcfg = ModelConfig.preset("micro-small", variant="hybrid_latent", n_latent_steps=2)
    model = build_model(mcfg)
    rl_cfg = RLConfig(
        reward_type="R2_log",
        filter_type="T2_self",
        interface="I2_hybrid",
        group_size=4,
        n_ahead=3,
        thought_len=4,
        lambda_ntp=0.2,
        top_p_tokens=0.5,
    )
    trainer = SSLRLTrainer(model=model, rl_cfg=rl_cfg, device="cpu")
    batch = _make_synthetic_batch(batch_size=2, seq_len=20, prompt_len=14, vocab_size=mcfg.vocab_size)
    metrics = trainer.train_step(batch)
    assert math.isfinite(metrics["loss_total"])
    assert math.isfinite(metrics["mean_reward"])


def test_train_supervised_and_rl_helpers_and_cli_parser(tmp_path):
    """Verify train_supervised, train_ssl_rl, checkpoint saving, and CLI parser flags."""
    from phase1_pilot.models import build_model

    torch.manual_seed(7)
    mcfg = ModelConfig.preset("micro-small", variant="ntp")
    model = build_model(mcfg)
    batches = [_make_synthetic_batch(batch_size=4, seq_len=20, prompt_len=12, vocab_size=mcfg.vocab_size) for _ in range(3)]

    sup_dir = tmp_path / "sup"
    sup_cfg = TrainConfig(
        preset="micro-small",
        batch_size=4,
        max_steps=3,
        lr=1e-3,
        device="cpu",
        log_every=1,
        output_dir=str(sup_dir),
    )
    sup_res = train_supervised(model, batches, batches, sup_cfg)
    assert sup_res["steps"] == 3
    assert math.isfinite(sup_res["final_train_loss"])
    assert math.isfinite(sup_res["final_val_loss"])
    assert (sup_dir / "checkpoint.pt").exists()
    assert (sup_dir / "metrics.jsonl").exists()

    rl_dir = tmp_path / "rl"
    rl_train_cfg = TrainConfig(
        preset="micro-small",
        batch_size=4,
        max_steps=2,
        device="cpu",
        log_every=1,
        output_dir=str(rl_dir),
    )
    rl_cfg = RLConfig(
        reward_type="R3_brier",
        filter_type="T2_self",
        interface="I1_discrete",
        group_size=4,
        n_ahead=4,
        thought_len=4,
    )
    rl_res = train_ssl_rl(model, batches, batches, rl_train_cfg, rl_cfg)
    assert rl_res["steps"] == 2
    assert math.isfinite(rl_res["final_metrics"]["loss_total"])
    assert (rl_dir / "checkpoint.pt").exists()
    assert (rl_dir / "metrics.jsonl").exists()

    # CLI parser smoke check
    parser = build_arg_parser()
    parsed = parser.parse_args(
        ["--mode", "rl", "--preset", "160m", "--reward-type", "R3_brier", "--filter-type", "T2_self", "--bf16"]
    )
    assert parsed.mode == "rl"
    assert parsed.preset == "160m"
    assert parsed.bf16 is True


def test_real_collated_dataset_integration_all_variants_and_filters(tmp_path):
    """Verify real PilotDataset + collate_batch with variable prompt_lens across B3/B4 and RL filters."""
    from functools import partial

    from torch.utils.data import DataLoader

    from phase1_pilot.config import DataConfig
    from phase1_pilot.data import PilotTokenizer, build_stratified_datasets, collate_batch
    from phase1_pilot.models import build_model
    from phase1_pilot.train import main as train_cli_main

    torch.manual_seed(17)
    tokenizer = PilotTokenizer()
    dcfg = DataConfig(seed=17, n_train_samples=32, n_eval_samples=16, max_seq_len=160)
    ds = build_stratified_datasets(dcfg, tokenizer)
    loader = DataLoader(
        ds["train"],
        batch_size=4,
        shuffle=False,
        collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id),
    )
    real_batch = next(iter(loader))

    # B3 (Pause) and B4 (Continuous Latent) supervised smoke test on real variable-length batch
    for variant in ("pause", "continuous_latent"):
        mcfg = ModelConfig.preset("micro-small", variant=variant, n_pause_tokens=2, n_latent_steps=2)
        m = build_model(mcfg)
        res = train_supervised(
            m,
            [real_batch],
            [real_batch],
            TrainConfig(max_steps=2, device="cpu", output_dir="", log_every=2),
            pad_id=tokenizer.pad_id,
        )
        assert math.isfinite(res["final_train_loss"])
        assert math.isfinite(res["final_val_loss"])

    # Exercise all 4 filters & 3 rewards on real collated batch (including T2_scale with frozen teacher)
    base_cfg = ModelConfig.preset("micro-small", variant="ntp")
    teacher = build_model(ModelConfig.preset("micro-large", variant="ntp"))

    for ftype, rtype, use_teacher in (
        ("T0_all", "R1_exact", False),
        ("T1_entropy", "R2_log", False),
        ("T2_self", "R3_brier", False),
        ("T2_scale", "R3_brier", True),
    ):
        model = build_model(base_cfg)
        rl_cfg = RLConfig(
            reward_type=rtype,
            filter_type=ftype,
            interface="I1_discrete",
            group_size=4,
            n_ahead=3,
            thought_len=4,
            top_p_tokens=0.5,
            ref_ema_decay=0.95 if ftype == "T0_all" else 1.0,
        )
        trainer = SSLRLTrainer(
            model=model,
            rl_cfg=rl_cfg,
            teacher_model=teacher if use_teacher else None,
            pad_id=tokenizer.pad_id,
            think_start_id=tokenizer.think_start_id,
            think_end_id=tokenizer.think_end_id,
            continuation_sft_weight=0.0 if ftype == "T1_entropy" else 0.1,
            device="cpu",
        )
        metrics = trainer.train_step(real_batch)
        assert math.isfinite(metrics["loss_total"])
        assert math.isfinite(metrics["nothought_nll_drift"])
        assert all(not p.requires_grad for p in trainer.ref_model.parameters())

    # End-to-end CLI smoke test for --mode rl
    cli_out = tmp_path / "cli_rl"
    cli_res = train_cli_main(
        [
            "--mode", "rl",
            "--preset", "micro-small",
            "--reward-type", "R3_brier",
            "--filter-type", "T2_self",
            "--group-size", "2",
            "--thought-len", "3",
            "--batch-size", "4",
            "--max-steps", "2",
            "--n-train-samples", "24",
            "--n-eval-samples", "12",
            "--device", "cpu",
            "--output-dir", str(cli_out),
            "--log-every", "1",
        ]
    )
    assert cli_res["steps"] == 2
    assert (cli_out / "checkpoint.pt").exists()
    assert (cli_out / "metrics.jsonl").exists()


def test_rl_package_exports():
    import phase1_pilot.rl as rl_pkg

    for name in (
        "SSLRLTrainer",
        "compute_brier_score",
        "compute_reward",
        "select_tokens",
        "select_thought_entropy_mask",
        "compute_dr_grpo_advantages",
        "compute_dapo_pg_loss",
    ):
        assert hasattr(rl_pkg, name), f"Missing export {name} in phase1_pilot.rl"
        assert name in rl_pkg.__all__


def _ddp_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import os
    from pathlib import Path

    import torch.distributed as dist

    from phase1_pilot.models import build_model

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Intentionally seed model init differently per rank to prove rank-0 broadcast works!
        torch.manual_seed(100 + rank)
        mcfg = ModelConfig.preset("micro-small", variant="ntp")
        model = build_model(mcfg)

        # Distinct batches per rank to verify gradient all-reduce produces identical weights!
        torch.manual_seed(200 + rank)
        sup_batch = _make_synthetic_batch(batch_size=2, seq_len=16, prompt_len=10, vocab_size=mcfg.vocab_size)
        train_supervised(
            model,
            [sup_batch],
            None,
            TrainConfig(max_steps=2, lr=1e-3, device="cpu", output_dir="", seed=42),
        )
        sup_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        rl_batch = _make_synthetic_batch(batch_size=2, seq_len=16, prompt_len=10, vocab_size=mcfg.vocab_size)
        rl_cfg = RLConfig(
            reward_type="R3_brier",
            filter_type="T2_self",
            interface="I1_discrete",
            group_size=2,
            n_ahead=3,
            thought_len=3,
            top_p_tokens=0.5,
        )
        trainer = SSLRLTrainer(model=model, rl_cfg=rl_cfg, device="cpu")
        trainer.train_step(rl_batch)
        rl_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ref_weights = {k: v.detach().cpu().clone() for k, v in trainer.ref_model.state_dict().items()}

        torch.save(
            {"sup": sup_weights, "rl": rl_weights, "ref": ref_weights},
            Path(out_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_torch_distributed_sync_supervised_and_rl(tmp_path):
    """Verify 2-rank Gloo distributed broadcast + gradient all-reduce keeps ranks bit-synchronized."""
    import torch.multiprocessing as mp

    init_file = str(tmp_path / "dist_init")
    out_dir = str(tmp_path / "dist_out")
    (tmp_path / "dist_out").mkdir()

    mp.spawn(_ddp_worker, args=(2, init_file, out_dir), nprocs=2, join=True)

    r0 = torch.load(tmp_path / "dist_out" / "rank_0.pt", map_location="cpu", weights_only=False)
    r1 = torch.load(tmp_path / "dist_out" / "rank_1.pt", map_location="cpu", weights_only=False)

    for stage in ("sup", "rl", "ref"):
        for k in r0[stage]:
            assert torch.allclose(r0[stage][k], r1[stage][k], atol=1e-6), (
                f"Rank mismatch after {stage} step on parameter {k}"
            )
