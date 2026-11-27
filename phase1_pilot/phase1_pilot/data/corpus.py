"""Real-text Regime 2: corpus documents to one-boundary samples.

Track 1 §5.2 trains Regime 2 by plain next-token prediction on web text. The rest of
the pilot spends its extra compute at one thinking boundary per sample, so each
document is cut into token windows and one boundary is drawn inside each window.
Everything here works on token IDs: a document is encoded once, and the prompt and
the target are slices of that one encoding.

A document's identity is its text. The train/eval split, duplicate detection, and the
boundary draw all hash the text, so a mirrored copy under another URL behaves exactly
like the original.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from phase1_pilot.config import CorpusConfig
from phase1_pilot.data.records import TokenizedSample


# Eval windows use a fixed boundary seed, so every run scores the same prefixes.
EVAL_BOUNDARY_SEED = 0

_SPLITS = ("train", "eval")


@dataclass(frozen=True)
class Document:
    """One raw corpus document. `key` records where it came from (a URL or repo path)."""

    key: str
    text: str
    source: str


def read_corpus_jsonl(path: str | Path) -> Iterator[Document]:
    """Read documents from a JSON Lines file with string fields `key`, `text`, `source`."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {lineno} is not valid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {lineno} must be a JSON object")
            fields = {}
            for name in ("key", "text", "source"):
                value = obj.get(name)
                if not isinstance(value, str):
                    raise ValueError(f"{path}: line {lineno}: field {name!r} must be a string")
                fields[name] = value
            yield Document(**fields)


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_hash(*parts: Any) -> int:
    """A 64-bit hash of structured parts, identical across processes and runs."""
    payload = json.dumps(parts, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _split_of(digest: str, eval_fraction: float) -> str:
    return "eval" if _stable_hash("corpus-split-v1", digest) / 2**64 < eval_fraction else "train"


def corpus_split(text: str, eval_fraction: float) -> str:
    """`"train"` or `"eval"` for a document, as a fixed function of its text.

    The run seed plays no part. RL arms start from a supervised checkpoint, and a
    seed-dependent split could hand them eval documents that checkpoint trained on.
    """
    return _split_of(_text_digest(text), eval_fraction)


def _cut_windows(
    ids: list[int],
    *,
    digest: str,
    source: str,
    bos_id: int,
    max_seq_len: int,
    cfg: CorpusConfig,
    boundary_seed: int,
) -> list[TokenizedSample]:
    body_len = max_seq_len - 1 - cfg.headroom_tokens
    min_len = cfg.min_prompt_tokens + cfg.min_target_tokens
    if body_len < min_len:
        raise ValueError(
            f"max_seq_len={max_seq_len} with headroom_tokens={cfg.headroom_tokens} leaves "
            f"{body_len} corpus tokens per window, fewer than "
            f"min_prompt_tokens + min_target_tokens = {min_len}"
        )

    windows: list[TokenizedSample] = []
    for index, start in enumerate(range(0, len(ids), body_len)):
        body = ids[start : start + body_len]
        if len(body) < min_len:
            break
        rng = random.Random(_stable_hash("corpus-boundary-v1", boundary_seed, digest, index))
        boundary = rng.randint(cfg.min_prompt_tokens, len(body) - cfg.min_target_tokens)
        windows.append(
            TokenizedSample(
                input_ids=[bos_id] + body,
                prompt_len=1 + boundary,
                target_ids=body[boundary:],
                regime="natural",
                task=f"corpus:{source}",
                depth=0,
            )
        )
    return windows


def _encode_document(text: str, tokenizer: Any) -> list[int] | None:
    """Token IDs of a document, or `None` if it would put a protocol token in the stream."""
    ids = tokenizer.encode_corpus(text)
    if not tokenizer.special_ids.isdisjoint(ids):
        return None
    return ids


def document_windows(
    doc: Document,
    tokenizer: Any,
    max_seq_len: int,
    cfg: CorpusConfig,
    boundary_seed: int,
) -> list[TokenizedSample]:
    """Cut one document into consecutive token windows with one boundary each.

    A window is `<bos>` plus up to `max_seq_len - 1 - cfg.headroom_tokens` corpus
    tokens. No `<ans>` or `<eos>` is added: the text is plain corpus continuation and
    a window usually ends mid-document. A trailing piece too short to hold a boundary
    is dropped. The boundary of window `i` is drawn from an RNG seeded by
    `(boundary_seed, text, i)`, so it does not depend on where the document sits in a
    stream.

    A document is rejected (no windows) if the tokenizer turns part of its text into
    a protocol token, because a `<pad>` inside real text would be masked out of loss
    and attention.
    """
    ids = _encode_document(doc.text, tokenizer)
    if ids is None:
        return []
    return _cut_windows(
        ids,
        digest=_text_digest(doc.text),
        source=doc.source,
        bos_id=tokenizer.bos_id,
        max_seq_len=max_seq_len,
        cfg=cfg,
        boundary_seed=boundary_seed,
    )


@dataclass
class CorpusStats:
    """What a corpus build actually used. `windows` and `tokens` map split to source to count."""

    documents_read: int = 0
    skipped_duplicate: int = 0
    skipped_protocol_token: int = 0
    skipped_short: int = 0
    windows: dict[str, dict[str, int]] = field(default_factory=lambda: {s: {} for s in _SPLITS})
    tokens: dict[str, dict[str, int]] = field(default_factory=lambda: {s: {} for s in _SPLITS})


@dataclass(frozen=True)
class CorpusSplits:
    train: list[TokenizedSample]
    eval: list[TokenizedSample]
    stats: CorpusStats


def build_corpus_splits(
    documents: Iterable[Document],
    tokenizer: Any,
    *,
    max_seq_len: int,
    n_train: int,
    n_eval: int,
    seed: int,
    cfg: CorpusConfig | None = None,
) -> CorpusSplits:
    """Fill a train and an eval quota of windows from a document stream.

    Documents are taken in stream order, so a smaller request returns a prefix of a
    larger one and the source mixture is whatever the stream interleaves. Each
    document goes wholly to one split (`corpus_split`). A document whose text was
    already seen is skipped. Train boundaries are drawn with `seed`; eval boundaries
    use `EVAL_BOUNDARY_SEED`.
    """
    cfg = cfg or CorpusConfig()
    quotas = {"train": n_train, "eval": n_eval}
    out: dict[str, list[TokenizedSample]] = {s: [] for s in _SPLITS}
    stats = CorpusStats()
    seen: set[str] = set()

    stream = iter(documents)
    while any(len(out[s]) < quotas[s] for s in _SPLITS):
        # Checked before each read, so nothing past the last needed document is touched.
        doc = next(stream, None)
        if doc is None:
            break
        stats.documents_read += 1
        digest = _text_digest(doc.text)
        if digest in seen:
            stats.skipped_duplicate += 1
            continue
        seen.add(digest)

        split = _split_of(digest, cfg.eval_fraction)
        room = quotas[split] - len(out[split])
        if room <= 0:
            continue
        ids = _encode_document(doc.text, tokenizer)
        if ids is None:
            stats.skipped_protocol_token += 1
            continue
        windows = _cut_windows(
            ids,
            digest=digest,
            source=doc.source,
            bos_id=tokenizer.bos_id,
            max_seq_len=max_seq_len,
            cfg=cfg,
            boundary_seed=seed if split == "train" else EVAL_BOUNDARY_SEED,
        )
        if not windows:
            stats.skipped_short += 1
            continue
        taken = windows[:room]
        out[split].extend(taken)
        stats.windows[split][doc.source] = stats.windows[split].get(doc.source, 0) + len(taken)
        stats.tokens[split][doc.source] = stats.tokens[split].get(doc.source, 0) + sum(
            len(w.input_ids) - 1 for w in taken
        )

    missing = {s: quotas[s] - len(out[s]) for s in _SPLITS if len(out[s]) < quotas[s]}
    if missing:
        raise ValueError(
            f"corpus ran out after {stats.documents_read} documents: still need "
            + ", ".join(f"{n} {s} windows" for s, n in missing.items())
            + f" (have {len(out['train'])} train, {len(out['eval'])} eval)"
        )
    return CorpusSplits(train=out["train"], eval=out["eval"], stats=stats)
