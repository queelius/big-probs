"""Tests for scripts/probe_learnability.py: the pointer-chain task and answer-only labels."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from phase1_pilot.data import PilotTokenizer
from phase1_pilot.data.generators import tokenize_sample

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "probe_learnability.py"
_spec = importlib.util.spec_from_file_location("probe_learnability", _SCRIPT)
probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(probe)


def _resolve(prompt: str) -> tuple[str, int]:
    """Follow the query through the bindings; return (answer, hops taken)."""
    body = prompt.removeprefix("Ptr: ").removesuffix("<ans>")
    eqs, query = body.rsplit("; Query: ", 1)
    binding = dict(e.split("=") for e in eqs.split("; "))
    node, hops = query, 0
    while not binding[node].isdigit():
        node, hops = binding[node], hops + 1
    return binding[node], hops


@pytest.mark.parametrize("depth", [0, 1, 2, 4, 8])
def test_pointer_target_is_reached_by_exactly_depth_hops(depth):
    for s in probe.generate_pointer_chains(50, (depth,), seed=3, pool_size=32):
        answer, hops = _resolve(s.prompt)
        assert answer == s.target
        assert hops == depth and s.depth == depth


def test_pointer_prompt_has_three_chains_with_distinct_literals():
    for s in probe.generate_pointer_chains(50, (3,), seed=5):
        literals = re.findall(r"=(\d+)", s.prompt)
        assert len(literals) == probe.POINTER_CHAINS
        assert len(set(literals)) == probe.POINTER_CHAINS
        names = re.findall(r"(v\d+)=", s.prompt)
        assert len(names) == len(set(names)) == probe.POINTER_CHAINS * 4


def test_default_pool_names_are_single_pilot_tokens():
    tok = PilotTokenizer()
    for s in probe.generate_pointer_chains(200, (0, 1, 2, 4), seed=11):
        for name in re.findall(r"v\d+", s.prompt):
            assert len(tok.encode(name, add_bos=False, add_eos=False)) == 1, name


def test_pool_too_small_for_depth_is_rejected():
    with pytest.raises(ValueError, match="needs 18 names"):
        probe.generate_pointer_chains(1, (5,), seed=0, pool_size=16)


def test_pointer_generator_is_deterministic_and_cycles_depths():
    a = probe.generate_pointer_chains(6, (1, 2, 3), seed=9)
    b = probe.generate_pointer_chains(6, (1, 2, 3), seed=9)
    assert [s.prompt for s in a] == [s.prompt for s in b]
    assert [s.depth for s in a] == [1, 2, 3, 1, 2, 3]


def test_answer_only_collate_supervises_only_target_tokens():
    tok = PilotTokenizer()
    samples = [tokenize_sample(s, tok, 192) for s in probe.generate_pointer_chains(3, (1, 2, 3), seed=1)]
    batch = probe.answer_only_collate(samples, pad_id=tok.pad_id)
    for i, s in enumerate(samples):
        row = batch["labels"][i]
        kept = (row != -100).nonzero().flatten().tolist()
        assert kept == list(range(s.prompt_len, len(s.input_ids)))
        # The target and the `<eos>` that `tokenize_sample` appends after it.
        assert row[kept].tolist() == s.target_ids + [tok.eos_id]


def test_overlapping_depths_are_rejected():
    with pytest.raises(SystemExit):
        probe.parse_args(["--train-depths", "1,2", "--ood-depths", "2,3", "--output", "x.json"])
