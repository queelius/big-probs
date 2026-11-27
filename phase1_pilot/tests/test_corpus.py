"""Tests for the real-text Regime 2 path: corpus documents to one-boundary samples."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest

from phase1_pilot.config import CorpusConfig, DataConfig
from phase1_pilot.data.corpus import (
    Document,
    build_corpus_splits,
    corpus_split,
    document_windows,
    read_corpus_jsonl,
)
from phase1_pilot.data.generators import (
    build_stratified_datasets,
    collate_batch,
    natural_train_count,
)
from phase1_pilot.data.tokenizer import HFTokenizerAdapter, PilotTokenizer


CFG = CorpusConfig(min_prompt_tokens=6, min_target_tokens=4, eval_fraction=0.25, headroom_tokens=0)
MAX_SEQ_LEN = 48
MARKER_TEXT = (
    "vocab = ['<pad>', '<bos>', '<eos>', '<sep>'] and the model emits <think> tokens "
    "</think> before <ans> or after a <pause> marker, as the theorem of padding shows"
)


def _windows(doc: Document, tokenizer, seed: int = 0, cfg: CorpusConfig = CFG, max_seq_len: int = MAX_SEQ_LEN):
    return document_windows(doc, tokenizer, max_seq_len=max_seq_len, cfg=cfg, boundary_seed=seed)


# ---------------------------------------------------------------------------
# document_windows
# ---------------------------------------------------------------------------


def test_window_is_the_joint_tokenization_cut_at_one_boundary(any_tokenizer, make_documents) -> None:
    doc = make_documents(1, seed=3, min_words=12, max_words=12)[0]
    ids = any_tokenizer.encode_corpus(doc.text)
    assert len(ids) < MAX_SEQ_LEN - 1, "fixture must fit one window"

    (window,) = _windows(doc, any_tokenizer)

    assert window.input_ids == [any_tokenizer.bos_id] + ids
    assert window.target_ids == window.input_ids[window.prompt_len :]
    n_prompt = window.prompt_len - 1
    assert CFG.min_prompt_tokens <= n_prompt <= len(ids) - CFG.min_target_tokens
    assert (window.regime, window.task, window.depth) == ("natural", f"corpus:{doc.source}", 0)


def test_long_document_windows_tile_its_tokens(any_tokenizer, make_documents) -> None:
    doc = make_documents(1, seed=5, min_words=300, max_words=300)[0]
    ids = any_tokenizer.encode_corpus(doc.text)

    windows = _windows(doc, any_tokenizer)

    assert len(windows) >= 3
    assert all(len(w.input_ids) == MAX_SEQ_LEN for w in windows[:-1])
    assert len(windows[-1].input_ids) <= MAX_SEQ_LEN
    tiled = [t for w in windows for t in w.input_ids[1:]]
    assert tiled == ids[: len(tiled)]
    assert len(ids) - len(tiled) < CFG.min_prompt_tokens + CFG.min_target_tokens


def test_headroom_leaves_room_for_inserted_positions(any_tokenizer, make_documents) -> None:
    doc = make_documents(1, seed=5, min_words=300, max_words=300)[0]
    cfg = dataclasses.replace(CFG, headroom_tokens=10)

    windows = _windows(doc, any_tokenizer, cfg=cfg)

    assert max(len(w.input_ids) for w in windows) == MAX_SEQ_LEN - 10


def test_document_too_short_for_one_boundary_yields_no_window(any_tokenizer) -> None:
    doc = Document(key="tiny", text="x = sum", source="codeparrot")
    assert len(any_tokenizer.encode_corpus(doc.text)) < CFG.min_prompt_tokens + CFG.min_target_tokens

    assert _windows(doc, any_tokenizer) == []


def test_window_too_small_for_one_boundary_is_an_error(any_tokenizer, make_documents) -> None:
    doc = make_documents(1, seed=5)[0]
    with pytest.raises(ValueError, match="min_prompt_tokens"):
        _windows(doc, any_tokenizer, max_seq_len=CFG.min_prompt_tokens + CFG.min_target_tokens)


def test_marker_strings_in_text_stay_text(any_tokenizer) -> None:
    doc = Document(key="llm-docs", text=MARKER_TEXT, source="fineweb-edu")
    assert any_tokenizer.pad_id in any_tokenizer.encode(MARKER_TEXT), "fixture must hit the hazard"

    windows = _windows(doc, any_tokenizer)

    body = [t for w in windows for t in w.input_ids[1:]]
    assert not set(body) & any_tokenizer.special_ids
    assert all(w.input_ids[0] == any_tokenizer.bos_id for w in windows)
    assert any_tokenizer.decode(body) == MARKER_TEXT[: len(any_tokenizer.decode(body))]
    assert "'<pad>', '<bos>'" in any_tokenizer.decode(body)


def test_document_is_rejected_when_the_tokenizer_cannot_keep_markers_literal(
    make_documents, make_fake_hf
) -> None:
    tokenizer = HFTokenizerAdapter(make_fake_hf(honors_split_special_tokens=False))
    marked = Document(key="llm-docs", text=MARKER_TEXT, source="fineweb-edu")
    clean = make_documents(1, seed=3)[0]

    assert _windows(marked, tokenizer) == []
    assert _windows(clean, tokenizer) != []


def test_boundary_depends_only_on_seed_text_and_window_index(any_tokenizer, make_documents) -> None:
    doc = make_documents(1, seed=9, min_words=300, max_words=300)[0]
    renamed = Document(key="mirror-of-" + doc.key, text=doc.text, source=doc.source)

    first = _windows(doc, any_tokenizer, seed=1)

    assert _windows(doc, any_tokenizer, seed=1) == first
    assert _windows(renamed, any_tokenizer, seed=1) == first
    other_seed = _windows(doc, any_tokenizer, seed=2)
    assert [w.prompt_len for w in other_seed] != [w.prompt_len for w in first]
    assert [w.input_ids for w in other_seed] == [w.input_ids for w in first]


def test_boundaries_cover_the_allowed_range(any_tokenizer, make_documents) -> None:
    lo, hi = CFG.min_prompt_tokens, MAX_SEQ_LEN - 1 - CFG.min_target_tokens
    seen: set[int] = set()
    for doc in make_documents(40, seed=11, min_words=300, max_words=300):
        for w in _windows(doc, any_tokenizer):
            if len(w.input_ids) == MAX_SEQ_LEN:
                seen.add(w.prompt_len - 1)

    assert seen == set(range(lo, hi + 1))


def test_corpus_config_rejects_degenerate_settings() -> None:
    for bad in (
        dict(min_prompt_tokens=0),
        dict(min_target_tokens=0),
        dict(headroom_tokens=-1),
        dict(eval_fraction=0.0),
        dict(eval_fraction=1.0),
    ):
        with pytest.raises(ValueError):
            CorpusConfig(**bad)


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def test_split_is_a_fixed_function_of_document_text() -> None:
    texts = [f"document number {i} about the theorem" for i in range(400)]

    labels = [corpus_split(t, eval_fraction=0.25) for t in texts]

    assert set(labels) == {"train", "eval"}
    assert 0.18 < labels.count("eval") / len(labels) < 0.32
    # Pinned: a change to the hash recipe would silently move documents across the split.
    assert [i for i, s in enumerate(labels[:40]) if s == "eval"] == [10, 18, 19, 21, 22, 32, 37, 39]
    # Raising the fraction only ever moves documents from train to eval.
    wider = [corpus_split(t, eval_fraction=0.5) for t in texts]
    assert all(w == "eval" for w, s in zip(wider, labels) if s == "eval")


# ---------------------------------------------------------------------------
# build_corpus_splits
# ---------------------------------------------------------------------------


def _reference_splits(docs, tokenizer, seed: int):
    """Windows per split built document by document, with no quota."""
    out: dict[str, list] = {"train": [], "eval": []}
    for doc in docs:
        split = corpus_split(doc.text, CFG.eval_fraction)
        boundary_seed = seed if split == "train" else 0
        out[split].extend(_windows(doc, tokenizer, seed=boundary_seed))
    return out


def _build(docs, tokenizer, n_train: int, n_eval: int, seed: int = 7):
    return build_corpus_splits(
        docs, tokenizer, max_seq_len=MAX_SEQ_LEN, n_train=n_train, n_eval=n_eval, seed=seed, cfg=CFG
    )


def test_splits_take_windows_in_document_order_and_never_share_a_document(
    any_tokenizer, make_documents
) -> None:
    docs = make_documents(60, seed=1)
    ref = _reference_splits(docs, any_tokenizer, seed=7)

    splits = _build(docs, any_tokenizer, n_train=50, n_eval=20)

    assert splits.train == ref["train"][:50]
    assert splits.eval == ref["eval"][:20]


def test_smaller_request_is_a_prefix_of_a_larger_one(any_tokenizer, make_documents) -> None:
    docs = make_documents(60, seed=1)

    small = _build(docs, any_tokenizer, n_train=20, n_eval=8)
    large = _build(docs, any_tokenizer, n_train=50, n_eval=20)

    assert large.train[:20] == small.train
    assert large.eval[:8] == small.eval


def test_eval_windows_do_not_depend_on_run_seed_or_train_size(any_tokenizer, make_documents) -> None:
    docs = make_documents(60, seed=1)

    a = _build(docs, any_tokenizer, n_train=50, n_eval=20, seed=7)
    b = _build(docs, any_tokenizer, n_train=10, n_eval=20, seed=8)

    assert a.eval == b.eval
    assert a.train[:10] != b.train


def test_repeated_text_is_used_once(any_tokenizer, make_documents) -> None:
    docs = make_documents(30, seed=1)
    mirrored = [d for doc in docs for d in (doc, Document("mirror/" + doc.key, doc.text, doc.source))]

    once = _build(docs, any_tokenizer, n_train=20, n_eval=8)
    twice = _build(mirrored, any_tokenizer, n_train=20, n_eval=8)

    assert (twice.train, twice.eval) == (once.train, once.eval)
    assert twice.stats.skipped_duplicate > 0


def test_corpus_too_small_for_the_request_is_an_error(any_tokenizer, make_documents) -> None:
    docs = make_documents(5, seed=1)
    with pytest.raises(ValueError, match="corpus ran out"):
        _build(docs, any_tokenizer, n_train=10_000, n_eval=5)


def test_build_stops_reading_once_the_quotas_are_full(any_tokenizer, make_documents) -> None:
    """A bad record after the last needed document must not fail a complete build."""
    consumed = 0

    def stream():
        nonlocal consumed
        for doc in make_documents(60, seed=1):
            consumed += 1
            yield doc

    splits = _build(stream(), any_tokenizer, n_train=20, n_eval=8)

    assert consumed == splits.stats.documents_read < 60


def test_stats_report_the_realized_mixture(any_tokenizer, make_documents) -> None:
    docs = [Document("tiny", "x = sum", "codeparrot")] + make_documents(60, seed=1)

    splits = _build(docs, any_tokenizer, n_train=50, n_eval=20)
    stats = splits.stats

    for name, windows in (("train", splits.train), ("eval", splits.eval)):
        by_source: dict[str, list[int]] = {}
        for w in windows:
            entry = by_source.setdefault(w.task.removeprefix("corpus:"), [0, 0])
            entry[0] += 1
            entry[1] += len(w.input_ids) - 1
        assert stats.windows[name] == {s: v[0] for s, v in by_source.items()}
        assert stats.tokens[name] == {s: v[1] for s, v in by_source.items()}
    assert set(stats.windows["train"]) == {"open-web-math", "codeparrot", "fineweb-edu"}
    assert stats.skipped_short >= 1
    assert stats.documents_read <= len(docs)


def test_corpus_windows_are_pinned() -> None:
    """Golden digest: the corpus path is data, and data is pinned (see CLAUDE.md)."""
    rng = random.Random(2026)
    words = ("theorem", "of", "the", "integral", "def ", "return ", "x", " = ", "sum", "range")
    docs = [
        Document(f"pin-{i}", " ".join(rng.choice(words) for _ in range(rng.randint(30, 200))), "open-web-math")
        for i in range(40)
    ]

    splits = _build(docs, PilotTokenizer(), n_train=30, n_eval=10)

    payload = [[dataclasses.asdict(w) for w in part] for part in (splits.train, splits.eval)]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert digest == "e75eefc1eaf1ee7346647717e570391c46f5342c0bac081505b2a01b0a104b67"


# ---------------------------------------------------------------------------
# read_corpus_jsonl
# ---------------------------------------------------------------------------


def test_read_corpus_jsonl_round_trips_documents(tmp_path, make_documents) -> None:
    docs = make_documents(5, seed=2) + [Document("unicode", "naïve ∑ proof\nline two", "open-web-math")]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(dataclasses.asdict(d)) + "\n" for d in docs), encoding="utf-8")

    assert list(read_corpus_jsonl(path)) == docs


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"key": "a", "text": "t"}', "source"),
        ('{"key": "a", "text": 3, "source": "s"}', "text"),
        ("not json", "line 2"),
        ('["a", "t", "s"]', "line 2"),
    ],
)
def test_read_corpus_jsonl_names_the_bad_line(tmp_path, line: str, message: str) -> None:
    path = tmp_path / "corpus.jsonl"
    good = json.dumps({"key": "ok", "text": "fine", "source": "s"})
    path.write_text(f"{good}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        list(read_corpus_jsonl(path))


# ---------------------------------------------------------------------------
# build_stratified_datasets with a corpus
# ---------------------------------------------------------------------------

DATA_CFG = DataConfig(seed=11, n_train_samples=200, n_eval_samples=24, max_seq_len=128)


def _natural(docs, tokenizer, n_train: int | None = None):
    return build_corpus_splits(
        docs,
        tokenizer,
        max_seq_len=DATA_CFG.max_seq_len,
        n_train=natural_train_count(DATA_CFG.n_train_samples) if n_train is None else n_train,
        n_eval=DATA_CFG.n_eval_samples,
        seed=DATA_CFG.seed,
        cfg=CFG,
    )


def test_stratified_datasets_take_regime_2_from_the_corpus(any_tokenizer, make_documents) -> None:
    natural = _natural(make_documents(60, seed=1), any_tokenizer)

    real = build_stratified_datasets(DATA_CFG, any_tokenizer, natural=natural)
    synthetic = build_stratified_datasets(DATA_CFG, any_tokenizer)

    train = real["train"].tokenized
    assert len(train) == DATA_CFG.n_train_samples
    assert sorted((t for t in train if t.regime == "natural"), key=repr) == sorted(natural.train, key=repr)
    assert real["eval_natural"].tokenized == natural.eval
    # Regimes 1 and 3 are exactly what the synthetic build produces.
    for name in ("eval_id", "eval_ood", "eval_aleatoric"):
        assert real[name].tokenized == synthetic[name].tokenized

    def others(ds):
        return sorted((t for t in ds.tokenized if t.regime != "natural"), key=repr)

    assert others(real["train"]) == others(synthetic["train"])
    # The corpus windows are shuffled into the split, not appended to it.
    at = [i for i, t in enumerate(train) if t.regime == "natural"]
    assert at != list(range(at[0], at[0] + len(at)))
    assert build_stratified_datasets(DATA_CFG, any_tokenizer, natural=natural)["train"].tokenized == train


def test_stratified_datasets_reject_a_corpus_build_of_the_wrong_size(any_tokenizer, make_documents) -> None:
    wrong = _natural(make_documents(60, seed=1), any_tokenizer, n_train=7)
    with pytest.raises(ValueError, match="natural"):
        build_stratified_datasets(DATA_CFG, any_tokenizer, natural=wrong)


def test_corpus_rows_collate_with_exact_target_alignment(any_tokenizer, make_documents) -> None:
    natural = _natural(make_documents(60, seed=1), any_tokenizer)
    train = build_stratified_datasets(DATA_CFG, any_tokenizer, natural=natural)["train"].tokenized
    rows = [t for t in train if t.regime == "natural"][:3] + [t for t in train if t.regime != "natural"][:3]

    batch = collate_batch(rows, pad_id=any_tokenizer.pad_id)

    for i, row in enumerate(rows):
        n = len(row.input_ids)
        assert batch["input_ids"][i, :n].tolist() == row.input_ids
        assert batch["labels"][i, :n].tolist() == row.input_ids
        assert (batch["labels"][i, n:] == -100).all()
        assert batch["attention_mask"][i].tolist() == [1] * n + [0] * (batch["input_ids"].shape[1] - n)
    for i, row in enumerate(rows[:3]):
        on = batch["target_mask"][i].nonzero().flatten().tolist()
        assert on == list(range(row.prompt_len, len(row.input_ids)))
