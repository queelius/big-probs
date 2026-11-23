"""Unit tests for the 5 mandatory diagnostics and Track 1 §5.5 verdict engine."""

from __future__ import annotations

import math

import pytest
import torch

from phase1_pilot.eval.diagnostics import (
    ArmEvalBundle,
    CalibrationResult,
    ParetoPoint,
    PassAtKResult,
    ThoughtFaithfulnessResult,
    check_monotonic_depth_scaling,
    compute_ece_and_brier,
    compute_faithfulness_summary,
    compute_pass_at_k_from_counts,
    compute_pilot_verdict,
    corrupt_discrete_thought_format_matched,
    evaluate_thought_faithfulness,
    perturb_continuous_latents_norm_matched,
    unbiased_pass_at_k,
)


# ---------------------------------------------------------------------------
# Diagnostic 2: Unbiased pass@k edge cases & depth aggregation
# ---------------------------------------------------------------------------


class TestPassAtK:
    def test_edge_cases(self) -> None:
        # c = 0 -> always 0.0
        assert unbiased_pass_at_k(64, 0, 1) == 0.0
        assert unbiased_pass_at_k(64, 0, 16) == 0.0
        assert unbiased_pass_at_k(64, 0, 64) == 0.0

        # c = n -> always 1.0
        assert unbiased_pass_at_k(64, 64, 1) == 1.0
        assert unbiased_pass_at_k(64, 64, 64) == 1.0

        # n - c < k -> guaranteed hit -> 1.0
        assert unbiased_pass_at_k(10, 4, 7) == 1.0
        assert unbiased_pass_at_k(10, 4, 10) == 1.0

        # k = 1 -> exact c / n
        assert math.isclose(unbiased_pass_at_k(64, 16, 1), 0.25, rel_tol=1e-12)
        assert math.isclose(unbiased_pass_at_k(64, 1, 1), 1.0 / 64.0, rel_tol=1e-12)

        # k = n -> 1.0 iff c >= 1 else 0.0
        assert unbiased_pass_at_k(64, 1, 64) == 1.0

        # Exact combinatorial check against math.comb for intermediate values
        expected = 1.0 - math.comb(64 - 5, 16) / math.comb(64, 16)
        assert math.isclose(unbiased_pass_at_k(64, 5, 16), expected, rel_tol=1e-12)

    def test_monotonicity_in_k_and_c(self) -> None:
        vals_k = [unbiased_pass_at_k(64, 3, k) for k in (1, 4, 16, 64)]
        assert vals_k[0] < vals_k[1] < vals_k[2] < vals_k[3] == 1.0

        vals_c = [unbiased_pass_at_k(16, c, 4) for c in range(0, 17)]
        assert all(vals_c[i] <= vals_c[i + 1] for i in range(len(vals_c) - 1))

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            unbiased_pass_at_k(0, 0, 1)
        with pytest.raises(ValueError):
            unbiased_pass_at_k(10, -1, 1)
        with pytest.raises(ValueError):
            unbiased_pass_at_k(10, 11, 1)
        with pytest.raises(ValueError):
            unbiased_pass_at_k(10, 2, 0)
        with pytest.raises(ValueError):
            unbiased_pass_at_k(10, 2, 11)

    def test_depth_breakdown_and_ood_aggregation(self) -> None:
        counts = {
            2: [(16, 16), (16, 12)],
            4: [(16, 8), (16, 8)],
            6: [(16, 2), (16, 0)],
            8: [(16, 1), (16, 0)],
        }
        res = compute_pass_at_k_from_counts(counts, ks=(1, 4, 16, 64), ood_depths=(6, 8), arm_name="test")
        # k=64 filtered out since n=16
        assert res.ks == [1, 4, 16]
        assert res.n_samples_per_prompt == 16
        assert res.by_depth[2][16] == 1.0
        # On OOD depths (6, 8): 2 prompts have c>=1 and 2 have c==0 => pass@16 = 0.5
        assert math.isclose(res.ood_pass_at_max_k, 0.5, rel_tol=1e-12)
        assert res.by_depth[2][1] > res.by_depth[8][1]


# ---------------------------------------------------------------------------
# Diagnostic 4: ECE and Brier score calibration bounds
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_perfect_prediction_bounds(self) -> None:
        # Near-one-hot logits on the true class -> ECE ~ 0, Brier ~ 0, acc = 1
        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.full((4, 8), -30.0)
        logits[torch.arange(4), targets] = 30.0
        res = compute_ece_and_brier(logits, targets, n_bins=10)
        assert res["accuracy"] == 1.0
        assert res["ece"] < 1e-5
        assert -1e-5 <= res["mean_brier_score"] <= 0.0
        assert 0.0 <= res["mean_brier_loss"] <= 1e-5
        assert abs(res["overconfidence_gap"]) < 1e-5

    def test_confidently_wrong_hits_worst_brier_and_high_ece(self) -> None:
        # Predict wrong class with probability ~1.0 -> BS = -2.0, ECE = 1.0
        targets = torch.tensor([0, 1, 2, 3])
        wrong = torch.tensor([1, 2, 3, 0])
        logits = torch.full((4, 8), -30.0)
        logits[torch.arange(4), wrong] = 30.0
        res = compute_ece_and_brier(logits, targets, n_bins=10)
        assert res["accuracy"] == 0.0
        assert res["ece"] > 0.999
        assert math.isclose(res["mean_brier_score"], -2.0, abs_tol=1e-4)
        assert math.isclose(res["mean_brier_loss"], 2.0, abs_tol=1e-4)
        assert res["overconfidence_gap"] > 0.999

    def test_aleatoric_uniform_distribution_well_calibrated(self) -> None:
        # Uniform distribution over V=4 classes with uniform targets -> acc=0.25, conf=0.25, ECE=0
        v = 4
        targets = torch.tensor([0, 1, 2, 3] * 100)
        logits = torch.zeros((400, v))
        res = compute_ece_and_brier(logits, targets, n_bins=10)
        assert math.isclose(res["mean_confidence"], 0.25, abs_tol=1e-6)
        assert math.isclose(res["accuracy"], 0.25, abs_tol=1e-6)
        assert res["ece"] < 1e-6
        # BS for uniform q=1/V: 2*(1/V) - V*(1/V^2) - 1 = 1/V - 1 = -0.75
        assert math.isclose(res["mean_brier_score"], -0.75, abs_tol=1e-5)

    def test_general_bounds_and_ignore_index(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(100, 16)
        targets = torch.randint(0, 16, (100,))
        targets[:20] = -100
        res = compute_ece_and_brier(logits, targets, n_bins=10, ignore_index=-100)
        assert res["n_tokens"] == 80
        assert 0.0 <= res["ece"] <= 1.0
        assert -2.0 <= res["mean_brier_score"] <= 0.0
        assert math.isclose(res["mean_brier_loss"], -res["mean_brier_score"], rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Diagnostic 3: Causal thought faithfulness (swap, format-matched trunc, depth)
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    think_start_id = 3
    think_end_id = 4
    pause_id = 5
    sep_id = 6
    ans_id = 7
    vocab_size = 64

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


class TestThoughtFaithfulness:
    def test_format_matched_discrete_corruption_preserves_delimiters_and_length(self) -> None:
        gen = torch.Generator().manual_seed(123)
        thought = torch.tensor([3, 12, 18, 25, 31, 40, 4], dtype=torch.long)
        corrupted = corrupt_discrete_thought_format_matched(
            thought, think_start_id=3, think_end_id=4, vocab_size=64, generator=gen
        )
        assert corrupted.shape == thought.shape
        assert int(corrupted[0].item()) == 3
        assert int(corrupted[-1].item()) == 4
        # Every internal token must be corrupted and non-special
        for pos in range(1, len(thought) - 1):
            assert int(corrupted[pos].item()) != int(thought[pos].item())
            assert int(corrupted[pos].item()) not in {0, 1, 2, 3, 4, 5, 6, 7}

    def test_continuous_norm_matched_orthogonal_perturbation(self) -> None:
        gen = torch.Generator().manual_seed(42)
        latents = torch.randn(4, 32)
        perturbed = perturb_continuous_latents_norm_matched(latents, generator=gen)
        assert perturbed.shape == latents.shape
        orig_norms = latents.norm(dim=-1)
        pert_norms = perturbed.norm(dim=-1)
        assert torch.allclose(orig_norms, pert_norms, atol=1e-5)
        # Dot product between each original and perturbed step vector should be ~0
        dots = (latents * perturbed).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-4)

    def test_dummy_content_free_thought_has_zero_swap_and_trunc(self) -> None:
        tok = _DummyTokenizer()
        samples = [
            {"prompt_ids": [1, 10 + i], "target_ids": [20 + (i % 4)], "depth": d}
            for i, d in enumerate([2, 2, 4, 4, 6, 6, 8, 8])
        ]

        # Content-free decoder ignores thought z completely (depends only on prompt)
        def dummy_nll(prompt_ids: list[int], thought: torch.Tensor, target_ids: list[int]) -> float:
            del thought
            return 1.0 + 0.05 * (prompt_ids[-1] % 5) + 0.01 * target_ids[0]

        def thought_provider(sample: dict) -> torch.Tensor:
            val = sample["target_ids"][0]
            return torch.tensor([3, val, val, val, 4], dtype=torch.long)

        res = evaluate_thought_faithfulness(
            model=torch.nn.Identity(),
            samples=samples,
            tokenizer=tok,
            arm_name="collapsed",
            mode="discrete",
            nll_evaluator=dummy_nll,
            thought_provider=thought_provider,
        )
        assert abs(res.delta_nll_swap) < 1e-9
        assert abs(res.delta_nll_trunc) < 1e-9
        assert not res.monotonic_depth_scaling

    def test_predictive_thought_yields_positive_swap_trunc_and_monotonic_depth(self) -> None:
        tok = _DummyTokenizer()
        samples = [
            {"prompt_ids": [1, 10 + i], "target_ids": [20 + i], "depth": d}
            for i, d in enumerate([2, 2, 4, 4, 6, 6, 8, 8])
        ]

        # Predictive decoder: low NLL when internal thought encodes target, high when mismatched/corrupted;
        # benefit scales with hop depth (harder problems rely more on thought).
        depth_by_target = {20 + i: d for i, d in enumerate([2, 2, 4, 4, 6, 6, 8, 8])}

        def predictive_nll(prompt_ids: list[int], thought: torch.Tensor, target_ids: list[int]) -> float:
            del prompt_ids
            tgt = target_ids[0]
            d = depth_by_target[tgt]
            internal = thought[1:-1]
            # Thought matches target iff internal tokens equal target
            if int((internal == tgt).sum().item()) == int(internal.numel()):
                return 0.1
            return 0.1 + 0.15 * d  # Swap/trunc penalty grows monotonically with depth d

        def thought_provider(sample: dict) -> torch.Tensor:
            tgt = sample["target_ids"][0]
            return torch.tensor([3, tgt, tgt, tgt, 4], dtype=torch.long)

        res = evaluate_thought_faithfulness(
            model=torch.nn.Identity(),
            samples=samples,
            tokenizer=tok,
            arm_name="faithful_rl",
            mode="discrete",
            nll_evaluator=predictive_nll,
            thought_provider=thought_provider,
        )
        assert res.delta_nll_swap > 0.5
        assert res.delta_nll_trunc > 0.5
        assert res.monotonic_swap_depth_scaling is True
        assert res.monotonic_depth_scaling is True
        assert res.delta_nll_swap_by_depth[8] > res.delta_nll_swap_by_depth[2]

    def test_monotonic_depth_helper_rejects_flat_or_decreasing(self) -> None:
        assert check_monotonic_depth_scaling({2: 0.1, 4: 0.2, 6: 0.35, 8: 0.5}) is True
        assert check_monotonic_depth_scaling({2: 0.2, 4: 0.2, 6: 0.2, 8: 0.2}) is False  # flat
        assert check_monotonic_depth_scaling({2: 0.3, 4: 0.2, 6: 0.4, 8: 0.5}) is False  # dip
        assert check_monotonic_depth_scaling({2: 0.3}) is False  # too few depths


# ---------------------------------------------------------------------------
# Diagnostic 5: Track 1 §5.5 Verdict Engine (Confirmed / Partial / Falsified)
# ---------------------------------------------------------------------------


def _make_bundle(
    name: str,
    acc: float,
    flops: float,
    ood_pass_k: float,
    d_swap: float = 0.0,
    d_trunc: float = 0.0,
    mono: bool = False,
    ece: float = 0.05,
    brier: float = -0.75,
    is_rl: bool = False,
    is_teacher_free: bool = True,
    is_cont_or_hybrid: bool = False,
    is_warm_start: bool = False,
    drift: float | None = 0.02,
    mode: str = "discrete",
) -> ArmEvalBundle:
    return ArmEvalBundle(
        arm_name=name,
        pareto=ParetoPoint(
            arm_name=name,
            variant=mode,
            total_params=1_000_000,
            non_embedding_params=800_000,
            exact_accuracy=acc,
            token_accuracy=acc,
            mean_nll=0.5,
            hard_bpb=0.7,
            inference_flops_per_token=flops,
            extra_steps_per_target=0.0 if mode == "ntp" else 4.0,
            n_samples=100,
        ),
        pass_at_k=PassAtKResult(
            arm_name=name,
            n_samples_per_prompt=16,
            ks=[1, 4, 16],
            overall={1: acc, 4: min(1.0, acc + 0.1), 16: min(1.0, ood_pass_k + 0.1)},
            by_depth={6: {16: ood_pass_k}, 8: {16: ood_pass_k}},
            ood_depths=[6, 8],
            ood_pass_at_max_k=ood_pass_k,
        ),
        faithfulness=ThoughtFaithfulnessResult(
            arm_name=name,
            mode=mode,
            nll_with_thought=0.4,
            nll_swapped=0.4 + d_swap,
            delta_nll_swap=d_swap,
            nll_truncated=0.4 + d_trunc,
            delta_nll_trunc=d_trunc,
            monotonic_swap_depth_scaling=mono,
            monotonic_accuracy_lift_scaling=mono,
            monotonic_depth_scaling=mono,
            n_samples=100,
        ),
        calibration_aleatoric=CalibrationResult(
            arm_name=name,
            regime="regime3_aleatoric",
            ece=ece,
            mean_brier_score=brier,
            mean_brier_loss=-brier,
            mean_confidence=0.25 + ece,
            accuracy=0.25,
            overconfidence_gap=ece,
            n_tokens=100,
        ),
        is_rl_arm=is_rl,
        is_teacher_free=is_teacher_free,
        is_continuous_or_hybrid=is_cont_or_hybrid,
        is_warm_start=is_warm_start,
        nothought_nll_drift=drift,
    )


class TestVerdictEngine:
    def test_strong_confirmation(self) -> None:
        b1 = _make_bundle("B1", acc=0.40, flops=1e6, ood_pass_k=0.30, mode="ntp")
        b2 = _make_bundle("B2", acc=0.60, flops=2.5e6, ood_pass_k=0.50, mode="ntp")
        b3 = _make_bundle("B3", acc=0.48, flops=2.0e6, ood_pass_k=0.35, mode="pause")
        rl_r3 = _make_bundle(
            "RL-R3",
            acc=0.63,
            flops=2.2e6,
            ood_pass_k=0.45,
            d_swap=0.35,
            d_trunc=0.22,
            mono=True,
            ece=0.06,
            is_rl=True,
            is_teacher_free=True,
            drift=0.03,
        )
        verdict = compute_pilot_verdict(b1, b2, b3, candidate_arms=[rl_r3])
        assert verdict.status == "CONFIRMED"
        assert verdict.best_rl_arm == "RL-R3"
        assert all(verdict.checks.values())
        assert "| `RL-R3` |" in verdict.markdown_table

    def test_partial_confirmation_when_pass_at_k_shrinks(self) -> None:
        b1 = _make_bundle("B1", acc=0.40, flops=1e6, ood_pass_k=0.40, mode="ntp")
        b2 = _make_bundle("B2", acc=0.60, flops=2.5e6, ood_pass_k=0.55, mode="ntp")
        b3 = _make_bundle("B3", acc=0.48, flops=2.0e6, ood_pass_k=0.42, mode="pause")
        # Beats B2 & B3 with faithful thoughts, but shrinks OOD pass@k below B1 (pure sharpening)
        rl_sharp = _make_bundle(
            "RL-R3",
            acc=0.62,
            flops=2.2e6,
            ood_pass_k=0.35,  # < B1's 0.40
            d_swap=0.30,
            d_trunc=0.18,
            mono=True,
            is_rl=True,
            is_teacher_free=True,
        )
        verdict = compute_pilot_verdict(b1, b2, b3, candidate_arms=[rl_sharp])
        assert verdict.status == "PARTIAL_CONFIRMATION"
        assert verdict.checks["ood_pass_at_k_expansion"] is False

    def test_partial_confirmation_when_only_b4_or_i2_wins(self) -> None:
        b1 = _make_bundle("B1", acc=0.40, flops=1e6, ood_pass_k=0.30, mode="ntp")
        b2 = _make_bundle("B2", acc=0.60, flops=2.5e6, ood_pass_k=0.50, mode="ntp")
        b3 = _make_bundle("B3", acc=0.48, flops=2.0e6, ood_pass_k=0.35, mode="pause")
        rl_cold = _make_bundle("RL-R1", acc=0.44, flops=2.3e6, ood_pass_k=0.28, is_rl=True)
        b4 = _make_bundle(
            "B4",
            acc=0.58,
            flops=2.0e6,
            ood_pass_k=0.42,
            d_swap=0.25,
            d_trunc=0.12,
            mono=False,
            is_rl=False,
            is_cont_or_hybrid=True,
            mode="continuous",
        )
        verdict = compute_pilot_verdict(b1, b2, b3, candidate_arms=[rl_cold], b4=b4)
        assert verdict.status == "PARTIAL_CONFIRMATION"
        assert verdict.best_rl_arm == "B4"

    def test_falsified_when_no_arm_beats_b3_and_b2(self) -> None:
        b1 = _make_bundle("B1", acc=0.40, flops=1e6, ood_pass_k=0.30, mode="ntp")
        b2 = _make_bundle("B2", acc=0.65, flops=2.5e6, ood_pass_k=0.55, mode="ntp")
        b3 = _make_bundle("B3", acc=0.52, flops=2.0e6, ood_pass_k=0.38, mode="pause")
        rl_weak = _make_bundle(
            "RL-R3",
            acc=0.49,  # Fails to beat B3 (0.52) and B2 (0.65)
            flops=2.4e6,
            ood_pass_k=0.31,
            d_swap=0.02,
            d_trunc=0.0,
            mono=False,
            is_rl=True,
        )
        verdict = compute_pilot_verdict(b1, b2, b3, candidate_arms=[rl_weak])
        assert verdict.status == "FALSIFIED"
        assert verdict.checks["beats_b3_strict"] is False
        assert verdict.checks["b2_accuracy_parity"] is False

    def test_flops_inversion_prevents_confirmation(self) -> None:
        b1 = _make_bundle("B1", acc=0.40, flops=1e6, ood_pass_k=0.30, mode="ntp")
        b2 = _make_bundle("B2", acc=0.60, flops=2.5e6, ood_pass_k=0.50, mode="ntp")
        b3 = _make_bundle("B3", acc=0.48, flops=2.0e6, ood_pass_k=0.35, mode="pause")
        # High accuracy, but spends 4x more FLOPs than B2!
        rl_bloated = _make_bundle(
            "RL-Bloated",
            acc=0.70,
            flops=10.0e6,
            ood_pass_k=0.60,
            d_swap=0.40,
            d_trunc=0.25,
            mono=True,
            is_rl=True,
        )
        verdict = compute_pilot_verdict(b1, b2, b3, candidate_arms=[rl_bloated])
        assert verdict.status == "FALSIFIED"
        assert verdict.checks["b2_flops_matched"] is False


# ---------------------------------------------------------------------------
# End-to-end live-model integration tests across LlamaLM / Pause / Continuous
# ---------------------------------------------------------------------------


class TestLiveModelDiagnostics:
    def test_all_probes_on_real_models_and_datasets(self) -> None:
        from phase1_pilot.config import DataConfig, ModelConfig
        from phase1_pilot.data import PilotTokenizer, build_stratified_datasets
        from phase1_pilot.eval.diagnostics import (
            evaluate_calibration,
            evaluate_pareto_point,
            evaluate_pass_at_k,
        )
        from phase1_pilot.models import build_model

        torch.manual_seed(0)
        tok = PilotTokenizer()
        ds = build_stratified_datasets(
            DataConfig(seed=7, n_train_samples=40, n_eval_samples=16, max_seq_len=192),
            tok,
        )
        r1_samples = ds["eval_id"].tokenized[:4] + ds["eval_ood"].tokenized[:4]
        r3_samples = ds["eval_aleatoric"].tokenized[:6]

        cfg_ntp = ModelConfig(
            vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, d_ff=64, max_seq_len=192, variant="ntp"
        )
        cfg_pause = ModelConfig(
            vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, d_ff=64, max_seq_len=192, variant="pause", n_pause_tokens=3
        )
        cfg_cont = ModelConfig(
            vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, d_ff=64, max_seq_len=192, variant="continuous_latent", n_latent_steps=3
        )

        m_ntp = build_model(cfg_ntp)
        m_pause = build_model(cfg_pause, pause_token_id=tok.pause_id)
        m_cont = build_model(cfg_cont)

        # Diagnostic 1: FLOPs accounting strictly increases with pause/latent/thought steps
        p_ntp = evaluate_pareto_point(m_ntp, r1_samples, tok, arm_name="B1", use_thought=False)
        p_pause = evaluate_pareto_point(m_pause, r1_samples, tok, arm_name="B3", use_thought=False)
        p_cont = evaluate_pareto_point(m_cont, r1_samples, tok, arm_name="B4", use_thought=False)
        p_disc = evaluate_pareto_point(m_ntp, r1_samples, tok, arm_name="RL", use_thought=True, thought_len=4)
        assert p_pause.inference_flops_per_token > p_ntp.inference_flops_per_token
        assert p_cont.inference_flops_per_token > p_ntp.inference_flops_per_token
        assert p_disc.inference_flops_per_token > p_ntp.inference_flops_per_token
        assert p_ntp.hard_bpb > 0.0 and not math.isnan(p_ntp.hard_bpb)

        # Diagnostic 2: pass@k runs across ID and OOD depths
        pk = evaluate_pass_at_k(m_ntp, r1_samples, tok, arm_name="RL", ks=(1, 4), n_samples=4, use_thought=True, thought_len=3)
        assert pk.ks == [1, 4]
        assert set(pk.by_depth.keys()).issubset({2, 4, 6, 8})
        assert 0.0 <= pk.ood_pass_at_max_k <= 1.0

        # Diagnostic 3: Faithfulness runs in discrete, continuous, and pause modes
        f_disc = evaluate_thought_faithfulness(m_ntp, r1_samples, tok, arm_name="RL", mode="discrete", thought_len=3)
        f_cont = evaluate_thought_faithfulness(m_cont, r1_samples, tok, arm_name="B4", mode="continuous")
        f_pause = evaluate_thought_faithfulness(m_pause, r1_samples, tok, arm_name="B3", mode="pause")
        assert f_disc.n_samples == len(r1_samples)
        assert f_cont.n_samples == len(r1_samples)
        assert abs(f_pause.delta_nll_swap) < 1e-9
        assert abs(f_pause.delta_nll_trunc) < 1e-9

        # Diagnostic 4: Calibration on Regime 3 Aleatoric Negative Control
        cal_r3 = evaluate_calibration(m_ntp, r3_samples, tok, arm_name="B1", regime="regime3_aleatoric")
        assert 0.0 <= cal_r3.ece <= 1.0
        assert -2.0 <= cal_r3.mean_brier_score <= 0.0
        assert cal_r3.n_tokens > 0

    def test_hybrid_latent_eval_matches_recompute_hybrid_parity(self) -> None:
        """Verify diagnostics.py I2-Hybrid eval matches ContinuousLatentWrapper.recompute_hybrid_logprobs_and_continuation (< 1e-4)."""
        from phase1_pilot.config import ModelConfig
        from phase1_pilot.data import PilotTokenizer
        from phase1_pilot.eval.diagnostics import (
            _build_prefix_embeds_for_model,
            _forward_target_logits_with_prefix_embeds,
            _generate_hybrid_thought_and_embeds,
            evaluate_pareto_point,
            evaluate_thought_faithfulness,
        )
        from phase1_pilot.models import build_model

        torch.manual_seed(123)
        tok = PilotTokenizer()
        thought_len = 6
        cfg_i2 = ModelConfig(
            vocab_size=tok.vocab_size,
            d_model=32,
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=192,
            variant="hybrid_latent",
            n_latent_steps=4,
        )
        model = build_model(cfg_i2)
        model.eval()

        prompt_ids = torch.tensor([[1, 12, 18, 25, 31, 7]], dtype=torch.long)
        target_ids = torch.tensor([[19, 24, 30]], dtype=torch.long)

        # Eval path in diagnostics.py
        thought_ids, blended_embeds = _generate_hybrid_thought_and_embeds(
            model, prompt_ids, thought_len=thought_len, temperature=0.0
        )
        assert thought_ids.shape == (1, thought_len)
        assert blended_embeds.shape == (1, thought_len, 32)

        prefix_embeds, extra_steps = _build_prefix_embeds_for_model(
            model, prompt_ids, tok, use_thought=True, thought_len=thought_len, temperature=0.0
        )
        assert extra_steps == thought_len  # Exact match: no extra prepended latents or <think>/</think>
        eval_logits = _forward_target_logits_with_prefix_embeds(model.base_lm, prefix_embeds, target_ids)

        # Native training path in ContinuousLatentWrapper
        with torch.no_grad():
            _, _, native_logits = model.recompute_hybrid_logprobs_and_continuation(
                prefix_ids=prompt_ids,
                thought_ids=thought_ids,
                target_ids=target_ids,
                temperature=1.0,
            )

        max_diff = float((eval_logits - native_logits).abs().max().item())
        assert max_diff < 1e-4, f"Hybrid eval vs training continuation mismatch: max_diff={max_diff}"

        # Also verify Pareto extra_steps_per_target and faithfulness mode="hybrid"
        samples = [
            {"prompt_ids": [1, 12, 18, 25, 31, 7], "target_ids": [19, 24, 30], "depth": 2},
            {"prompt_ids": [1, 14, 20, 27, 33, 7], "target_ids": [21, 26, 32], "depth": 4},
        ]
        pareto = evaluate_pareto_point(
            model, samples, tok, arm_name="I2-Hybrid", use_thought=True, thought_len=thought_len
        )
        assert math.isclose(pareto.extra_steps_per_target, thought_len / 3.0, rel_tol=1e-6)
        faith = evaluate_thought_faithfulness(
            model, samples, tok, arm_name="I2-Hybrid", mode="hybrid", thought_len=thought_len
        )
        assert faith.mode == "hybrid"
        assert faith.n_samples == 2


