"""Tests for scripts/trainer_check_mqar.py: Zoology-style MQAR data and answer accuracy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "trainer_check_mqar.py"
_spec = importlib.util.spec_from_file_location("trainer_check_mqar", _SCRIPT)
mqar = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mqar)


def _bindings(row: torch.Tensor, k: int) -> dict[int, int]:
    ctx = row[: 2 * k].tolist()
    return dict(zip(ctx[0::2], ctx[1::2]))


@pytest.mark.parametrize("k,supervised", [(4, None), (4, 1), (3, None)])
def test_each_label_is_the_context_value_of_the_key_just_before_it(k, supervised):
    x, y = mqar.generate_mqar(200, num_kv_pairs=k, supervised_queries=supervised, seed=1)
    assert x.shape == y.shape == (200, 64)
    for row, lab in zip(x, y):
        binding = _bindings(row, k)
        positions = (lab != -100).nonzero().flatten().tolist()
        assert len(positions) == (k if supervised is None else supervised)
        for t in positions:
            assert t > 2 * k  # labels sit only in the query region
            assert int(lab[t]) == binding[int(row[t - 1])]


def test_every_key_is_queried_even_when_only_one_query_is_supervised():
    """Single-query rows keep the multi-query inputs; only the label count changes."""
    x_all, _ = mqar.generate_mqar(50, supervised_queries=None, seed=5)
    x_one, _ = mqar.generate_mqar(50, supervised_queries=1, seed=5)
    assert torch.equal(x_all, x_one)
    for row in x_one:
        binding = _bindings(row, 4)
        assert set(binding) <= set(row[8:].tolist())


def test_key_value_and_filler_ranges_avoid_protocol_ids_and_do_not_overlap():
    x, y = mqar.generate_mqar(300, n_keys=127, n_values=128, seed=2)
    keys, values = x[:, 0:8:2], x[:, 1:8:2]
    assert keys.min() >= 8 + 1 and keys.max() <= 8 + 127
    assert values.min() >= 8 + 128 and values.max() <= 8 + 255
    assert x.min() >= 8 and x.max() <= 8 + 255
    for row in keys:
        assert len(set(row.tolist())) == 4  # keys without replacement
    assert y[y != -100].min() >= 8 + 128


def test_small_vocab_variant_matches_the_pointer_task_sizes():
    x, _ = mqar.generate_mqar(100, n_keys=16, n_values=19, num_kv_pairs=3, seed=3)
    keys, values = x[:, 0:6:2], x[:, 1:6:2]
    assert keys.min() >= 9 and keys.max() <= 8 + 16
    assert values.min() >= 8 + 17 and values.max() <= 8 + 16 + 19


def test_generation_is_deterministic_per_seed():
    a = mqar.generate_mqar(20, seed=7)
    b = mqar.generate_mqar(20, seed=7)
    c = mqar.generate_mqar(20, seed=8)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not torch.equal(a[0], c[0])


class _RuleModel(torch.nn.Module):
    """At each position, puts all mass on `rule(row, t)`; stands in for a trained model."""

    def __init__(self, rule, vocab: int = 512):
        super().__init__()
        self.rule, self.vocab = rule, vocab

    def forward(self, input_ids):
        logits = torch.full((*input_ids.shape, self.vocab), -1e4)
        for b, row in enumerate(input_ids):
            for t in range(row.shape[0]):
                logits[b, t, self.rule(row, t)] = 0.0
        return SimpleNamespace(logits=logits)


def test_answer_accuracy_scores_an_oracle_at_one_and_a_value_copier_near_chance():
    x, y = mqar.generate_mqar(120, num_kv_pairs=4, seed=4)

    def oracle(row, t):
        return _bindings(row, 4).get(int(row[t]), 0)

    def first_value(row, t):
        return int(row[1])

    assert mqar.answer_accuracy(_RuleModel(oracle), x, y) == 1.0
    assert 0.15 < mqar.answer_accuracy(_RuleModel(first_value), x, y) < 0.35


def test_non_finite_train_loss_aborts_the_run(tmp_path, monkeypatch):
    def fake_train(model, train_loader, val_loader, cfg, *, step_callback):
        step_callback(1, {"train_loss": float("nan")})

    monkeypatch.setattr(mqar, "train_supervised", fake_train)
    with pytest.raises(FloatingPointError, match="step 1"):
        mqar.main(["--steps", "2", "--n-train", "32", "--n-test", "16", "--threads", "1",
                   "--output", str(tmp_path / "r.json")])


def test_cli_smoke_writes_a_report(tmp_path):
    out = tmp_path / "r.json"
    report = mqar.main(["--steps", "4", "--eval-every", "2", "--n-train", "64", "--n-test", "32",
                        "--threads", "1", "--output", str(out)])
    assert out.exists()
    assert [p["step"] for p in report["curve"]] == [2, 4]
    assert 0.0 <= report["final_test_acc"] <= 1.0
    assert report["config"]["num_kv_pairs"] == 4
