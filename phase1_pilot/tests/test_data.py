"""Unit tests for Phase 1 config contract, PilotTokenizer, and 3-regime dataset generators."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import re

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from phase1_pilot.config import DataConfig, ModelConfig, RLConfig, TrainConfig
from phase1_pilot.data.generators import (
    NATURAL_TEMPLATES,
    TRAIN_REGIME_MIX,
    PilotDataset,
    Sample,
    TokenizedSample,
    _natural_signature,
    _split_counts,
    build_stratified_datasets,
    collate_batch,
    generate_aleatoric_cloze,
    generate_dag_planning,
    generate_modular_varbind,
    generate_natural_stream,
    generate_python_trace,
    load_h100_web_mixture,
    tokenize_sample,
)
from phase1_pilot.data.tokenizer import SPECIAL_TOKENS, HFTokenizerAdapter, PilotTokenizer


# ---------------------------------------------------------------------------
# 1. Shared Config Contract Tests
# ---------------------------------------------------------------------------


def test_config_defaults_and_presets() -> None:
    mc = ModelConfig()
    assert mc.vocab_size == 512
    assert mc.d_model == 128
    assert mc.n_layers == 4
    assert mc.n_heads == 4
    assert mc.n_kv_heads == 4
    assert mc.d_ff == 384
    assert mc.max_seq_len == 256
    assert mc.variant == "ntp"
    assert mc.n_pause_tokens == 4
    assert mc.n_latent_steps == 4
    assert mc.max_thought_len == 8
    assert mc.head_dim == 32

    for name, expected_layers, expected_d, expected_vocab in (
        ("micro-small", 4, 128, 512),
        ("micro-large", 6, 192, 512),
        ("160m", 12, 768, 32000),
        ("410m", 24, 1024, 32000),
        ("1b", 24, 1792, 32000),
    ):
        cfg = ModelConfig.preset(name)
        assert cfg.n_layers == expected_layers
        assert cfg.d_model == expected_d
        assert cfg.vocab_size == expected_vocab

    paused = ModelConfig.preset("micro-small", variant="pause", n_pause_tokens=6)
    assert paused.variant == "pause"
    assert paused.n_pause_tokens == 6

    with pytest.raises(ValueError):
        ModelConfig.preset("nonexistent-scale")

    rl = RLConfig()
    assert rl.reward_type == "R3_brier"
    assert rl.filter_type == "T2_self"
    assert rl.interface == "I1_discrete"
    assert rl.group_size == 8
    assert rl.n_ahead == 4
    assert rl.clip_low == pytest.approx(0.2)
    assert rl.clip_high == pytest.approx(0.28)
    assert rl.reward_clip_tau == pytest.approx(5.0)
    assert rl.lambda_ntp == pytest.approx(0.2)
    assert rl.ref_ema_decay == pytest.approx(1.0)
    assert rl.filter_zero_var is True

    dc = DataConfig()
    assert dc.train_depths == (2, 4)
    assert dc.eval_ood_depths == (6, 8)
    assert dc.max_seq_len == 256

    tc = TrainConfig()
    assert tc.preset == "micro-small"
    assert tc.batch_size == 16
    assert tc.seed == 42


# ---------------------------------------------------------------------------
# 2. Tokenizer Tests
# ---------------------------------------------------------------------------


def test_tokenizer_special_ids_and_roundtrip() -> None:
    tok = PilotTokenizer()
    assert tok.vocab_size == 512
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.think_start_id == 3
    assert tok.think_end_id == 4
    assert tok.pause_id == 5
    assert tok.sep_id == 6
    assert tok.ans_id == 7

    samples = [
        "",
        "DAG: Edges: N0->N1, N1->N12; Start: N0; Cand: N12, N5, N9; Reach:<ans>N12",
        "Trace: a=3; b=5; if a<b: a=(a+b)%17; print(a)<ans>8",
        "<think>step 1: N0->N1</think><pause><sep><ans>42",
        "Unicode UTF-8 round-trip: α + β = γ 🚀",
        "def foo(x, y):\n    return (x * 2 + y) % 19",
    ]
    for text in samples:
        ids = tok.encode(text)
        assert all(0 <= i < tok.vocab_size for i in ids)
        assert tok.decode(ids, skip_special_tokens=False) == text

        ids_be = tok.encode(text, add_bos=True, add_eos=True)
        assert ids_be[0] == tok.bos_id
        assert ids_be[-1] == tok.eos_id
        assert tok.decode(ids_be, skip_special_tokens=False) == f"<bos>{text}<eos>"

    # Byte length helper must match exact UTF-8 byte length on non-special strings.
    plain = "Compute total = 12 * 4 + 7 -> 55!"
    plain_ids = tok.encode(plain)
    assert sum(tok.token_byte_length(i) for i in plain_ids) == len(plain.encode("utf-8"))


def test_tokenizer_ans_boundary_invariance_and_collate() -> None:
    tok = PilotTokenizer()
    prompt = "Mod: v1=3; v2=(v1+4)%19; Query: v2<ans>"
    target = "7"
    assert tok.encode(prompt) + tok.encode(target) == tok.encode(prompt + target)

    batch = tok.encode_batch(["abc", "DAG: N0->N1<ans>N1"], add_bos=True, add_eos=True)
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["lengths"].tolist()[0] < batch["lengths"].tolist()[1]
    pad_positions = batch["attention_mask"] == 0
    assert torch.all(batch["input_ids"][pad_positions] == tok.pad_id)


def test_hf_adapter_keeps_protocol_tokens_distinct_when_native_ones_are_shared(make_fake_hf) -> None:
    """GPT-2-style tokenizers use one token for pad, bos and eos; the protocol must not."""
    native = make_fake_hf(native_pad_is_eos=True)
    assert native.pad_token_id == native.bos_token_id == native.eos_token_id

    adapter = HFTokenizerAdapter(native)

    ids = [
        adapter.pad_id, adapter.bos_id, adapter.eos_id, adapter.think_start_id,
        adapter.think_end_id, adapter.pause_id, adapter.sep_id, adapter.ans_id,
    ]
    assert len(set(ids)) == 8
    assert adapter.special_ids == frozenset(ids)
    assert max(ids) < adapter.vocab_size
    encoded = adapter.encode("x", add_bos=True, add_eos=True)
    assert encoded == [adapter.bos_id, *native.encode("x"), adapter.eos_id]
    collated = adapter.collate([encoded, encoded[:2]])
    assert collated["input_ids"].tolist() == [encoded, encoded[:2] + [adapter.pad_id]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_hf_adapter_rejects_a_tokenizer_that_aliases_protocol_tokens() -> None:
    class _Aliasing:
        unk_token_id = None

        def get_vocab(self) -> dict[str, int]:
            return {}

        def add_special_tokens(self, mapping: dict[str, list[str]]) -> int:
            return 0

        def convert_tokens_to_ids(self, token: str) -> int:
            return 7

    with pytest.raises(ValueError, match="non-distinct"):
        HFTokenizerAdapter(_Aliasing())


def test_hf_adapter_rejects_a_tokenizer_that_does_not_keep_protocol_tokens_whole() -> None:
    """A vocabulary can hold `<ans>` as an ordinary entry that the tokenizer never emits."""

    class _NeverEmits:
        unk_token_id = None

        def __init__(self) -> None:
            self._vocab = {t: 300 + i for i, t in enumerate(SPECIAL_TOKENS)}

        def get_vocab(self) -> dict[str, int]:
            return dict(self._vocab)

        def add_special_tokens(self, mapping: dict[str, list[str]]) -> int:
            return 0

        def convert_tokens_to_ids(self, token: str) -> int:
            return self._vocab[token]

        def __len__(self) -> int:
            return 308

        def encode(self, text: str, **kwargs) -> list[int]:
            return list(text.encode("utf-8"))

    with pytest.raises(ValueError, match="single token"):
        HFTokenizerAdapter(_NeverEmits())


def test_hf_adapter_vocab_size_covers_every_id(make_fake_hf) -> None:
    class _Sparse(make_fake_hf):
        def __len__(self) -> int:  # reports fewer entries than its highest ID
            return 10

    adapter = HFTokenizerAdapter(_Sparse())

    assert adapter.vocab_size == max(adapter.protocol_ids().values()) + 1


def test_encode_corpus_keeps_marker_strings_as_text(any_tokenizer) -> None:
    text = "pad with '<pad>' then <think>"

    ids = any_tokenizer.encode_corpus(text)

    assert not set(ids) & any_tokenizer.special_ids
    assert any_tokenizer.decode(ids) == text
    assert set(any_tokenizer.encode(text)) & any_tokenizer.special_ids, "plain encode still tokenizes markers"


def test_pilot_encode_corpus_matches_encode_on_marker_free_text() -> None:
    tok = PilotTokenizer()
    text = "def f(x):\n    return (a+b) % 17 == N3 -> the series"
    assert tok.encode_corpus(text) == tok.encode(text)


# ---------------------------------------------------------------------------
# 3. Regime 1 Algorithmic Exactness Tests
# ---------------------------------------------------------------------------


def test_dag_planning_exact_bfs_and_acyclic() -> None:
    samples = generate_dag_planning(n=48, depths=(2, 4, 6, 8), seed=123)
    assert len(samples) == 48
    seen_depths = collections.Counter(s.depth for s in samples)
    assert set(seen_depths.keys()) == {2, 4, 6, 8}

    for s in samples:
        assert s.regime == "algorithmic"
        assert s.task == "dag_planning"
        assert s.full_text == s.prompt + s.target
        assert s.answer_span_start == len(s.prompt)

        m = re.match(
            r"^DAG: Edges: ([^;]+); Start: (N\d+); Cand: ([^;]+); Reach:<ans>$",
            s.prompt,
        )
        assert m is not None, f"Unexpected prompt format: {s.prompt}"
        edges_raw, start, cands_raw = m.group(1), m.group(2), m.group(3)
        edges = [tuple(e.strip().split("->")) for e in edges_raw.split(",")]
        cands = [c.strip() for c in cands_raw.split(",")]
        assert len(cands) == 3

        # Independent cycle check (Kahn's algorithm) -> must be a strict DAG.
        indeg: dict[str, int] = collections.defaultdict(int)
        adj: dict[str, list[str]] = collections.defaultdict(list)
        nodes: set[str] = set()
        for u, v in edges:
            adj[u].append(v)
            indeg[v] += 1
            nodes.add(u)
            nodes.add(v)
        q = collections.deque([u for u in nodes if indeg[u] == 0])
        visited = 0
        while q:
            cur = q.popleft()
            visited += 1
            for nxt in adj[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        assert visited == len(nodes), "Graph contains a cycle!"

        # Independent BFS reachability and exact hop distance check.
        dist = {start: 0}
        bq = collections.deque([start])
        while bq:
            cur = bq.popleft()
            for nxt in adj[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    bq.append(nxt)
        reachable = [c for c in cands if c in dist]
        assert reachable == [s.target]
        assert dist[s.target] == s.depth


def test_python_trace_exact_execution() -> None:
    samples = generate_python_trace(n=48, depths=(2, 4, 6, 8), seed=456)
    assert len(samples) == 48

    for s in samples:
        assert s.regime == "algorithmic"
        assert s.task == "python_trace"
        assert s.prompt.startswith("Trace: ")
        assert s.prompt.endswith("<ans>")

        body = s.prompt[len("Trace: ") : -len("<ans>")]
        stmts = [part.strip() for part in body.split(";")]
        # 3 init lines (a,b,c) + `depth` transition lines + 1 print line.
        assert len(stmts) == 3 + s.depth + 1
        assert stmts[-1].startswith("print(")

        # Independently execute the exact Python program and capture stdout.
        buf = io.StringIO()
        program = "\n".join(stmts)
        with contextlib.redirect_stdout(buf):
            exec(compile(program, "<verify>", "exec"), {})
        printed = buf.getvalue().strip()
        assert printed == s.target


def test_modular_varbind_exact_solution_and_depth() -> None:
    samples = generate_modular_varbind(n=48, depths=(2, 4, 6, 8), seed=789)
    assert len(samples) == 48

    for s in samples:
        assert s.regime == "algorithmic"
        assert s.task == "modular_varbind"
        m = re.match(r"^Mod: (.+); Query: (v\d+)<ans>$", s.prompt)
        assert m is not None
        eqs_raw, query_var = m.group(1), m.group(2)
        eqs = [e.strip() for e in eqs_raw.split(";")]
        assert len(eqs) == s.depth + 1

        # Parse bindings and resolve iteratively while tracking exact dependency depth.
        exprs: dict[str, str] = {}
        for eq in eqs:
            lhs, rhs = eq.split("=", 1)
            exprs[lhs.strip()] = rhs.strip()

        vals: dict[str, int] = {}
        var_depth: dict[str, int] = {}
        while len(vals) < len(exprs):
            progressed = False
            for var, rhs in exprs.items():
                if var in vals:
                    continue
                deps = re.findall(r"v\d+", rhs)
                if all(dep in vals for dep in deps):
                    vals[var] = int(eval(rhs, {"__builtins__": {}}, vals))
                    var_depth[var] = 0 if not deps else 1 + max(var_depth[d] for d in deps)
                    progressed = True
            assert progressed, "Cyclic or unresolvable variable binding!"

        assert str(vals[query_var]) == s.target
        assert var_depth[query_var] == s.depth


# ---------------------------------------------------------------------------
# 4. Regime 2 Natural Stream & H100 Loader Tests
# ---------------------------------------------------------------------------


def test_natural_stream_and_h100_mixture_loader() -> None:
    tok = PilotTokenizer()
    samples = generate_natural_stream(n=30, seed=99)
    assert len(samples) == 30
    for s in samples:
        assert s.regime == "natural"
        assert s.full_text == s.prompt + s.target
        # Multi-token target for non-myopic n_ahead=4 RL evaluation.
        assert len(tok.encode(s.target)) >= 3

    calls: list[tuple[str, str | None]] = []

    def _fake_loader(name: str, subset: str | None = None, split: str = "train", streaming: bool = True):
        calls.append((name, subset))
        return {"name": name, "subset": subset, "split": split, "streaming": streaming}

    mix = load_h100_web_mixture(seed=7, dataset_loader=_fake_loader)
    assert mix["weights"] == [0.40, 0.40, 0.20]
    assert len(calls) == 3
    assert "open-web-math" in calls[0][0]
    assert "the-stack-v2" in calls[1][0]
    assert "fineweb-edu" in calls[2][0]

    with pytest.raises(ValueError):
        load_h100_web_mixture(weights=(0.5, 0.5, 0.5), dataset_loader=_fake_loader)


def _sig(sample: Sample) -> str:
    return _natural_signature(sample.prompt, sample.target)


def test_natural_templates_original_domains_render_distinct_samples() -> None:
    # Exhaustion tracking and capacity() both assume one distinct sample per combination.
    for template in NATURAL_TEMPLATES:
        assert len(template.original_signatures()) == template.capacity(widened=False), template.name
        for p in template.params:
            assert set(p.original.values()) <= set(p.widened.values()), (template.name, p.name)


def test_natural_stream_widens_only_after_original_domain_is_exhausted() -> None:
    k = len(NATURAL_TEMPLATES)
    t_idx = min(range(k), key=lambda i: NATURAL_TEMPLATES[i].capacity(widened=False))
    tight = NATURAL_TEMPLATES[t_idx]
    n_orig = tight.capacity(widened=False)
    # Largest request the pre-widening generator could fill: the tightest template's original
    # domain fully used, plus the samples of templates ahead of it in round-robin order.
    at_cap = n_orig * k + t_idx

    small = generate_natural_stream(at_cap, seed=5)
    large = generate_natural_stream(3 * at_cap, seed=5)

    assert len(large) == 3 * at_cap
    assert large[:at_cap] == small  # widening never changes samples generated before it
    assert len({_sig(s) for s in large}) == len(large)
    tight_sigs = [_sig(s) for s in large[t_idx::k]]
    original = tight.original_signatures()
    assert set(tight_sigs[:n_orig]) == original
    assert original.isdisjoint(tight_sigs[n_orig:])


def _samples_digest(samples: list[Sample]) -> str:
    payload = json.dumps([dataclasses.asdict(s) for s in samples], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_natural_stream_matches_pre_widening_fixture() -> None:
    # Digests computed with the pre-widening generator (commit 8c3e18e). Train uses 300 of the
    # 360 arithmetic-series samples; eval needs exactly the remaining 60 under exclusions.
    train = generate_natural_stream(900, seed=11)
    assert _samples_digest(train) == "4e80c2cda1365d0dbfab34ebe4a4c5a81051a6a4798eae63556dd2b9767dfc76"
    excluded = {_sig(s) for s in train}
    evals = generate_natural_stream(180, seed=12, exclude_signatures=set(excluded))
    assert _samples_digest(evals) == "ce28f4e1981d92fdd8a848e3d0e48dd3d0d30a4aeb295718cdb2bf910e695bed"

    # Past that point the same call widens instead of failing, and keeps the fixture as a prefix.
    k = len(NATURAL_TEMPLATES)
    t_idx = next(i for i, t in enumerate(NATURAL_TEMPLATES) if t.name == "arithmetic_series")
    longer = generate_natural_stream(360, seed=12, exclude_signatures=set(excluded))
    assert longer[:180] == evals
    original = NATURAL_TEMPLATES[t_idx].original_signatures()
    assert original.isdisjoint(_sig(s) for s in longer[180:][(t_idx - 180) % k :: k])
    assert excluded.isdisjoint(_sig(s) for s in longer)


def test_build_stratified_datasets_past_legacy_natural_cap() -> None:
    # 300 train + 900 eval natural samples exceed the ~1,080 the pre-widening generator allowed.
    cfg = DataConfig(seed=42, n_train_samples=2000, n_eval_samples=900, max_seq_len=256)
    splits = build_stratified_datasets(cfg, PilotTokenizer())
    train_nat = [_sig(s) for s in splits["train"].samples if s.regime == "natural"]
    eval_nat = [_sig(s) for s in splits["eval_natural"].samples]
    assert len(train_nat) == _split_counts(2000, TRAIN_REGIME_MIX)[1]
    assert len(eval_nat) == 900
    assert len(set(train_nat)) == len(train_nat)
    assert len(set(eval_nat)) == len(eval_nat)
    assert set(train_nat).isdisjoint(eval_nat)


def test_natural_stream_meets_h100_launcher_default_demand() -> None:
    # scripts/launch_h100_stage1.sh defaults: N_TRAIN_SAMPLES=200000, N_EVAL_SAMPLES=2000 (BL-11).
    n_train_nat = _split_counts(200_000, TRAIN_REGIME_MIX)[1]
    train = generate_natural_stream(n_train_nat, seed=1)
    train_sigs = {_sig(s) for s in train}
    evals = generate_natural_stream(2000, seed=2, exclude_signatures=train_sigs)
    eval_sigs = {_sig(s) for s in evals}
    assert len(train_sigs) == n_train_nat
    assert len(evals) == len(eval_sigs) == 2000
    assert train_sigs.isdisjoint(eval_sigs)


def test_natural_stream_rejects_requests_beyond_capacity() -> None:
    k = len(NATURAL_TEMPLATES)
    t_idx = min(range(k), key=lambda i: NATURAL_TEMPLATES[i].capacity(widened=True))
    n = NATURAL_TEMPLATES[t_idx].capacity(widened=True) * k + t_idx + 1
    with pytest.raises(ValueError, match="can produce at most"):
        generate_natural_stream(n, seed=0)


# sha256 over every split's samples and tokenized records. These pin the data behind EXP-001
# (research/experiments.jsonl) and the published HF splits (scripts/publish_hf.py exports
# `micro_pilot` with the EXP-001 config and `stage1_benchmark`). If a digest changes, those
# artifacts no longer reproduce from HEAD: record it per research/PROCEDURE.md §4.4.
_GOLDEN_DATASET_DIGESTS: dict[str, tuple[DataConfig, str]] = {
    "exp001_micro_pilot": (
        DataConfig(seed=42, train_depths=(2, 4), eval_ood_depths=(6, 8),
                   n_train_samples=600, n_eval_samples=48, max_seq_len=192),
        "2097a4ab9569b6a57d48a9e88c98abaeddcb674657cd1956aa0b4057137fd122",
    ),
    "hf_stage1_benchmark": (
        DataConfig(seed=42, train_depths=(2, 4), eval_ood_depths=(6, 8),
                   n_train_samples=2000, n_eval_samples=300, max_seq_len=256),
        "b67c63f3174c5c133158722fb2de6d6a121772582324e2a6b32451eb892c1202",
    ),
}


@pytest.mark.parametrize("name", sorted(_GOLDEN_DATASET_DIGESTS))
def test_generated_datasets_match_committed_artifacts(name: str) -> None:
    cfg, expected = _GOLDEN_DATASET_DIGESTS[name]
    splits = build_stratified_datasets(cfg, PilotTokenizer())
    payload = {
        split: {
            "samples": [dataclasses.asdict(s) for s in ds.samples],
            "tokenized": [dataclasses.asdict(t) for t in ds.tokenized],
        }
        for split, ds in splits.items()
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert digest == expected


# ---------------------------------------------------------------------------
# 5. Regime 3 Aleatoric Uniform Entropy Tests
# ---------------------------------------------------------------------------


def test_aleatoric_cloze_uniform_entropy_and_independence() -> None:
    num_classes = 16
    n = 4000
    samples = generate_aleatoric_cloze(n=n, seed=2026, num_classes=num_classes, namespace="test")
    assert len(samples) == n

    # All entity prompts must be strictly unique.
    prompts = [s.prompt for s in samples]
    assert len(set(prompts)) == n

    # Empirical Shannon entropy must be very close to theoretical max log2(16) = 4.0 bits.
    counts = collections.Counter(s.target for s in samples)
    assert len(counts) == num_classes
    probs = np.array([counts[f"C{i:02d}"] / n for i in range(num_classes)], dtype=np.float64)
    entropy_bits = float(-np.sum(probs * np.log2(probs)))
    max_entropy = math.log2(num_classes)
    assert entropy_bits > max_entropy - 0.05, f"Entropy too low: {entropy_bits:.4f} vs {max_entropy:.4f}"

    # Pearson chi-square statistic for uniform distribution (df=15; p=0.001 critical ~37.7).
    expected = n / num_classes
    chi2 = float(np.sum((np.array(list(counts.values())) - expected) ** 2 / expected))
    assert chi2 < 40.0, f"Aleatoric class distribution failed uniformity chi2={chi2:.2f}"

    # Empirical mutual information between attribute name and target must be near 0.
    joint = collections.Counter()
    attr_counts = collections.Counter()
    for s in samples:
        attr = s.prompt.split("Attr: ")[1].split(" |")[0]
        joint[(attr, s.target)] += 1
        attr_counts[attr] += 1
    mi = 0.0
    for (attr, tgt), c_xy in joint.items():
        p_xy = c_xy / n
        p_x = attr_counts[attr] / n
        p_y = counts[tgt] / n
        mi += p_xy * math.log2(p_xy / (p_x * p_y))
    assert mi < 0.03, f"Spurious mutual information between prompt attr and target: {mi:.4f}"


# ---------------------------------------------------------------------------
# 6. Stratified Builder, Zero Leakage, & Collate Tests
# ---------------------------------------------------------------------------


def test_build_stratified_datasets_zero_leakage_and_collate() -> None:
    tok = PilotTokenizer()
    cfg = DataConfig(
        seed=42,
        train_depths=(2, 4),
        eval_ood_depths=(6, 8),
        n_train_samples=240,
        n_eval_samples=90,
        max_seq_len=256,
    )
    splits = build_stratified_datasets(cfg, tok)
    assert set(splits.keys()) == {
        "train",
        "eval_id",
        "eval_ood",
        "eval_natural",
        "eval_aleatoric",
    }
    assert len(splits["train"]) == 240
    for k in ("eval_id", "eval_ood", "eval_natural", "eval_aleatoric"):
        assert len(splits[k]) == 90

    # Strict depth separation: train & eval_id use {2,4}; eval_ood uses {6,8}.
    train_alg_depths = {s.depth for s in splits["train"].samples if s.regime == "algorithmic"}
    eval_id_depths = {s.depth for s in splits["eval_id"].samples}
    eval_ood_depths = {s.depth for s in splits["eval_ood"].samples}
    assert train_alg_depths == {2, 4}
    assert eval_id_depths == {2, 4}
    assert eval_ood_depths == {6, 8}
    assert train_alg_depths.isdisjoint(eval_ood_depths)

    # Zero exact-prompt overlap between train and EVERY eval split.
    train_prompts = {s.prompt for s in splits["train"].samples}
    for k in ("eval_id", "eval_ood", "eval_natural", "eval_aleatoric"):
        eval_prompts = {s.prompt for s in splits[k].samples}
        assert train_prompts.isdisjoint(eval_prompts), f"Leakage detected between train and {k}!"

    # All tokenized samples fit strictly within max_seq_len=256 and preserve exact target slice.
    for name, ds in splits.items():
        for item in ds.tokenized:
            assert len(item.input_ids) <= cfg.max_seq_len, f"Overflow in {name}: {len(item.input_ids)}"
            assert item.input_ids[0] == tok.bos_id
            assert item.input_ids[-1] == tok.eos_id
            assert item.input_ids[item.prompt_len : item.prompt_len + len(item.target_ids)] == item.target_ids

    # Collation & PyTorch DataLoader compatibility.
    loader = DataLoader(splits["eval_ood"], batch_size=16, shuffle=False, collate_fn=collate_batch)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 16
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["target_mask"].shape == batch["input_ids"].shape
    assert set(batch["depths"].tolist()).issubset({6, 8})
    # Verify pad positions are masked to -100 in labels and False in target_mask.
    pad_mask = batch["attention_mask"] == 0
    if pad_mask.any():
        assert torch.all(batch["labels"][pad_mask] == -100)
        assert torch.all(~batch["target_mask"][pad_mask])
    # Verify target_mask selects exact target tokens.
    for i in range(16):
        p_len = int(batch["prompt_lens"][i])
        t_len = int(batch["target_lens"][i])
        assert int(batch["target_mask"][i].sum()) == t_len
        assert batch["input_ids"][i, p_len : p_len + t_len].tolist() == splits["eval_ood"][i].target_ids
