"""Tests for the prompt-blind reference: the best a model can do without reading the prompt."""

from __future__ import annotations

import math

import pytest

from phase1_pilot.data.records import TokenizedSample
from phase1_pilot.eval.reference import prompt_blind_reference


def _s(task: str, target: list[int], depth: int = 2) -> TokenizedSample:
    return TokenizedSample(
        input_ids=[1, 9, 9, 7, *target], prompt_len=4, target_ids=target,
        regime="algorithmic", task=task, depth=depth,
    )


def test_nll_is_the_smoothed_answer_marginal_of_each_task():
    # Task "a": answer 10 seen 3x, answer 11 seen 1x. Add-one over {10, 11, unseen}: denominator 4 + 3.
    train = [_s("a", [10]), _s("a", [10]), _s("a", [10]), _s("a", [11])]
    ref = prompt_blind_reference(train, [_s("a", [10]), _s("a", [11])])
    expected = (-math.log(4 / 7) - math.log(2 / 7)) / 2
    assert ref.mean_nll == pytest.approx(expected)
    assert ref.n_samples == 2 and ref.n_target_tokens == 2


def test_unseen_answer_gets_the_unseen_bucket():
    train = [_s("a", [10])]
    ref = prompt_blind_reference(train, [_s("a", [99])])
    assert ref.mean_nll == pytest.approx(-math.log(1 / 3))


def test_task_absent_from_train_is_rejected():
    with pytest.raises(ValueError, match="'b'"):
        prompt_blind_reference([_s("a", [10])], [_s("b", [10])])


def test_accuracy_predicts_each_tasks_most_common_answer():
    train = [_s("a", [10]), _s("a", [10]), _s("a", [11]), _s("b", [5]), _s("b", [6]), _s("b", [6])]
    ev = [_s("a", [10]), _s("a", [11]), _s("b", [6]), _s("b", [6])]
    ref = prompt_blind_reference(train, ev)
    assert ref.accuracy == pytest.approx(3 / 4)


def test_tie_for_most_common_answer_breaks_to_the_smallest_answer():
    train = [_s("a", [11]), _s("a", [10])]
    ref = prompt_blind_reference(train, [_s("a", [10])])
    assert ref.accuracy == 1.0


def test_multi_token_answer_spreads_its_nll_over_its_tokens():
    # One answer, seen twice; add-one over {answer, unseen}: p = 3/4, spread over 2 tokens.
    train = [_s("a", [10, 11]), _s("a", [10, 11])]
    ref = prompt_blind_reference(train, [_s("a", [10, 11])])
    assert ref.mean_nll == pytest.approx(-math.log(3 / 4) / 2)
    assert ref.n_target_tokens == 2


def test_breakdown_by_task_and_depth():
    train = [_s("a", [10]), _s("b", [5])]
    ev = [_s("a", [10], depth=2), _s("a", [10], depth=4), _s("b", [5], depth=2)]
    ref = prompt_blind_reference(train, ev)
    assert set(ref.mean_nll_by_task) == {"a", "b"}
    assert set(ref.mean_nll_by_depth) == {2, 4}
    assert ref.mean_nll_by_depth[4] == pytest.approx(-math.log(2 / 3))


def test_to_dict_is_json_ready():
    import json

    ref = prompt_blind_reference([_s("a", [10])], [_s("a", [10])])
    json.dumps(ref.to_dict())
