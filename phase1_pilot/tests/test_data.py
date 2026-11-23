"""Unit tests for Phase 1 config contract, PilotTokenizer, and 3-regime dataset generators."""

from __future__ import annotations

import collections
import contextlib
import io
import math
import re

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from phase1_pilot.config import DataConfig, ModelConfig, RLConfig, TrainConfig
from phase1_pilot.data.generators import (
    PilotDataset,
    Sample,
    TokenizedSample,
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
from phase1_pilot.data.tokenizer import HFTokenizerAdapter, PilotTokenizer


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


def test_hf_tokenizer_adapter_mock() -> None:
    class _MockHF:
        def __init__(self) -> None:
            self._vocab: dict[str, int] = {
                "<pad>": 0, "<bos>": 1, "<eos>": 2,
                "hi": 10, "there": 11,
            }
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.unk_token_id = -1
            self.pad_token = "<pad>"

        def get_vocab(self) -> dict[str, int]:
            return dict(self._vocab)

        def add_special_tokens(self, d: dict[str, list[str]]) -> int:
            for t in d.get("additional_special_tokens", []):
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab) + 100
            return len(d.get("additional_special_tokens", []))

        def convert_tokens_to_ids(self, tok: str) -> int:
            return self._vocab[tok]

        def __len__(self) -> int:
            return len(self._vocab)

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [self._vocab.get(w, 10) for w in text.split()]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            inv = {v: k for k, v in self._vocab.items()}
            return " ".join(inv.get(i, "?") for i in ids)

    adapter = HFTokenizerAdapter(_MockHF())
    assert adapter.pad_id == 0
    assert adapter.bos_id == 1
    assert adapter.eos_id == 2
    assert adapter.think_start_id >= 100
    assert adapter.think_end_id >= 100
    assert adapter.pause_id >= 100
    assert adapter.sep_id >= 100
    assert adapter.ans_id >= 100
    encoded = adapter.encode("hi there", add_bos=True, add_eos=True)
    assert encoded == [1, 10, 11, 2]
    collated = adapter.collate([encoded, [1, 10, 2]])
    assert collated["input_ids"].shape == (2, 4)


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
