"""Offline parts of scripts/compare_tokenizers.py (the Hub parts are not exercised)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from phase1_pilot.data.tokenizer import PilotTokenizer

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_tokenizers.py"
_spec = importlib.util.spec_from_file_location("compare_tokenizers", _SCRIPT)
ct = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ct)


def test_pilot_targets_are_all_single_tokens():
    """The pilot tokenizer's premise: every Regime 1 and Regime 3 target is one token."""
    stats = ct.target_stats(PilotTokenizer(), ct.synthetic_samples(n=40))
    assert set(stats) == {"aleatoric_cloze", "dag_planning", "modular_varbind", "python_trace"}
    assert all(s["single_token_target_frac"] == 1.0 for s in stats.values())


def test_target_stats_counts_multi_token_targets(hf_tokenizer):
    """The stand-in HF tokenizer splits answers such as `N27`, so the share drops below 1."""
    stats = ct.target_stats(hf_tokenizer, ct.synthetic_samples(n=40))
    assert stats["dag_planning"]["single_token_target_frac"] < 1.0


def test_bytes_per_token_is_bytes_over_tokens():
    tok = PilotTokenizer()
    docs = ["abc def", "é"]
    n_tok = sum(len(tok.encode_corpus(d)) for d in docs)
    assert ct.bytes_per_token(tok, docs) == pytest.approx((7 + 2) / n_tok)


def test_pins_from_a_report_replay_its_recorded_revisions():
    report = {
        "sources": {"codeparrot": {"dataset": "codeparrot/codeparrot-clean", "revision": "s1"}},
        "tokenizers": {"org/a": {"revision": "t1"}, "org/b": {"revision": "t2"}},
    }
    assert ct.pins_from_report(report) == {
        "tokenizers": {"org/a": "t1", "org/b": "t2"},
        "sources": {"codeparrot": "s1"},
    }


def test_committed_comparison_pins_every_candidate_and_source():
    committed = Path(__file__).resolve().parent.parent / "results" / "bl11_tokenizer_comparison.json"
    pins = ct.pins_from_report(json.loads(committed.read_text()))
    assert set(pins["tokenizers"]) == set(ct.CANDIDATES)
    assert set(pins["sources"]) == {label for label, *_ in ct.SOURCES}
    assert all(len(r) == 40 for group in pins.values() for r in group.values())


def test_preset_params_at_49160_match_the_preset_names():
    params = ct.preset_params(49_160)
    assert params == {"160m": 160_463_616, "410m": 409_011_200, "1b": 1_112_131_328}
