"""Tests for scripts/aggregate_seeds.py: combining per-seed micro-pilot metrics."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "aggregate_seeds.py"
_spec = importlib.util.spec_from_file_location("aggregate_seeds", _SCRIPT)
agg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(agg)


def _arm(acc: dict[int, float], swap: float = 0.0, ece: float = 0.05, ood_pass: float = 0.3) -> dict:
    return {
        "pareto": {
            "accuracy_by_depth": {str(d): v for d, v in acc.items()},
            "hard_bpb": 2.0,
            "inference_flops_per_token": 1.0e6,
            "total_params": 1000,
        },
        "pass_at_k": {"ood_pass_at_max_k": ood_pass},
        "faithfulness": {"delta_nll_swap": swap, "delta_nll_trunc": 0.5},
        "calibration_aleatoric": {"ece": ece},
        "nothought_nll_drift": None,
    }


def _payload(seed: int, arms: dict[str, dict]) -> dict:
    return {"config": {"seed": seed}, "arms": arms, "verdict": {"status": "FALSIFIED"}}


def _write(tmp_path: Path, payloads: list[dict]) -> list[Path]:
    paths = []
    for p in payloads:
        path = tmp_path / f"seed{p['config']['seed']}.json"
        path.write_text(json.dumps(p))
        paths.append(path)
    return paths


def test_id_and_ood_accuracy_are_means_over_their_depths():
    arm = _arm({2: 0.6, 4: 0.4, 6: 0.2, 8: 0.0})
    m = agg.arm_metrics(arm)
    assert m["id_acc"] == pytest.approx(0.5)
    assert m["ood_acc"] == pytest.approx(0.1)
    assert m["delta_nll_swap"] == 0.0
    assert m["reg3_ece"] == 0.05


def test_summary_reports_mean_and_sample_std_across_seeds(tmp_path):
    payloads = [
        _payload(s, {"B1": _arm({2: a, 4: a, 6: 0.0, 8: 0.0})})
        for s, a in ((1, 0.2), (2, 0.4), (3, 0.6))
    ]
    out = agg.aggregate(agg.load_runs(_write(tmp_path, payloads)))
    stat = out["arms"]["B1"]["id_acc"]
    assert stat["n"] == 3
    assert stat["mean"] == pytest.approx(0.4)
    assert stat["std"] == pytest.approx(0.2)  # ddof=1
    assert stat["values"] == pytest.approx([0.2, 0.4, 0.6])
    assert out["seeds"] == [1, 2, 3]


def test_paired_difference_against_a_control_keeps_seed_pairing(tmp_path):
    # The candidate beats the control by +5, +5, -1 pp: positive mean, not every seed.
    rows = ((1, 0.10, 0.15), (2, 0.30, 0.35), (3, 0.50, 0.49))
    payloads = [
        _payload(s, {"B3": _arm({2: c, 4: c}), "I2-Hybrid": _arm({2: x, 4: x})}) for s, c, x in rows
    ]
    out = agg.aggregate(agg.load_runs(_write(tmp_path, payloads)))
    diff = agg.paired_difference(out, "I2-Hybrid", "B3", "id_acc")
    assert diff["values"] == pytest.approx([0.05, 0.05, -0.01])
    assert diff["mean"] == pytest.approx(0.03)
    assert diff["all_positive"] is False


def test_missing_metric_is_none_and_skipped_in_stats(tmp_path):
    payloads = [_payload(s, {"B1": _arm({2: 0.1, 4: 0.1})}) for s in (1, 2)]
    out = agg.aggregate(agg.load_runs(_write(tmp_path, payloads)))
    stat = out["arms"]["B1"]["nothought_nll_drift"]
    assert stat["n"] == 0 and stat["mean"] is None and stat["std"] is None


def test_single_seed_has_no_std(tmp_path):
    out = agg.aggregate(agg.load_runs(_write(tmp_path, [_payload(7, {"B1": _arm({2: 0.3, 4: 0.3})})])))
    assert out["arms"]["B1"]["id_acc"]["std"] is None


def test_duplicate_seed_is_rejected(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_payload(1, {"B1": _arm({2: 0.1, 4: 0.1})})))
    b.write_text(json.dumps(_payload(1, {"B1": _arm({2: 0.2, 4: 0.2})})))
    with pytest.raises(ValueError, match="seed 1"):
        agg.load_runs([a, b])


def test_mismatched_arm_sets_are_rejected(tmp_path):
    payloads = [
        _payload(1, {"B1": _arm({2: 0.1, 4: 0.1})}),
        _payload(2, {"B1": _arm({2: 0.1, 4: 0.1}), "B2": _arm({2: 0.1, 4: 0.1})}),
    ]
    with pytest.raises(ValueError, match="arms"):
        agg.aggregate(agg.load_runs(_write(tmp_path, payloads)))


def test_nll_below_prompt_blind_uses_each_seeds_own_reference(tmp_path):
    payloads = []
    for seed, ref, arm_nll in ((1, 3.0, 2.5), (2, 3.2, 3.4)):
        arm = _arm({2: 0.1, 4: 0.1})
        arm["pareto"]["mean_nll"] = arm_nll
        p = _payload(seed, {"B1": arm})
        p["prompt_blind_reference"] = {"mean_nll": ref}
        payloads.append(p)
    out = agg.aggregate(agg.load_runs(_write(tmp_path, payloads)))
    stat = out["arms"]["B1"]["nll_below_prompt_blind"]
    assert stat["values"] == pytest.approx([0.5, -0.2])


def test_runs_without_a_reference_have_no_nll_below_prompt_blind(tmp_path):
    out = agg.aggregate(agg.load_runs(_write(tmp_path, [_payload(1, {"B1": _arm({2: 0.1, 4: 0.1})})])))
    assert out["arms"]["B1"]["nll_below_prompt_blind"]["mean"] is None


def test_cli_writes_json_and_prints_table(tmp_path, capsys):
    payloads = [
        _payload(s, {"B1": _arm({2: 0.2, 4: 0.2}), "B3": _arm({2: 0.1, 4: 0.1})}) for s in (1, 2)
    ]
    paths = _write(tmp_path, payloads)
    out_path = tmp_path / "agg.json"
    agg.main([*map(str, paths), "--output", str(out_path)])
    data = json.loads(out_path.read_text())
    assert set(data["arms"]) == {"B1", "B3"}
    assert math.isclose(data["arms"]["B1"]["id_acc"]["mean"], 0.2)
    printed = capsys.readouterr().out
    assert "B1" in printed and "id_acc" in printed
