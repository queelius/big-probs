"""Deterministic byte+symbol tokenizer (vocab_size=512) and optional HF adapter."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Sequence

import torch


SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",    # 0
    "<bos>",    # 1
    "<eos>",    # 2
    "<think>",  # 3
    "</think>", # 4
    "<pause>",  # 5
    "<sep>",    # 6
    "<ans>",    # 7
)

# Common algorithmic, code, math, and prose multi-character tokens (IDs 264..511).
# Kept strictly disjoint from SPECIAL_TOKENS and guaranteed lossless via UTF-8 byte
# fallback on IDs 8..263.
_BASE_MULTI_TOKENS: tuple[str, ...] = (
    # Operators & punctuation clusters
    "->", "==", "!=", "<=", ">=", "+=", "-=", "*=", "//", "**", ":=", "=>",
    " = ", " + ", " - ", " * ", " % ", " < ", " > ", " | ", ", ", "; ", ": ",
    "    ", "  ", "\n    ", "()", "():", "[]", "{}",
    # Python & code keywords
    "def ", "return ", "if ", "else:", "elif ", "for ", "while ", " in ",
    "range(", "print(", "True", "False", "None", " and ", " or ", " not ",
    "len(", "sum(", "min(", "max(", "int(", "str(", "abs(", "mod ",
    # Algorithmic task headers & labels
    "DAG:", "Edges:", "Start:", "Cand:", "Hops:", "Query:", "Trace:",
    "Mod:", "Vars:", "Bind:", "Entity:", "Attr:", "Code:", "Math:",
    "Step ", "Result:", "Output:", "Target:", "Reach:", "Path:",
    "Given ", "Then ", "Compute ", "Therefore ", "Thus ", "Since ",
    "Note:", "Proof:", "Rule:", "Value:", "State:", "Final:",
    # Common domain words
    " the ", " of ", " to ", " is ", " for ", " with ", " that ", " from ",
    " by ", " as ", " be ", " at ", " on ", " an ", " are ", " was ",
    "equals ", "modulo ", "graph ", "node ", "edge ", "value ", "total ",
    "count ", "index ", "result", "answer", "number", "square", "double",
    "triple", "offset", "remain", "factor", "series", "vector", "matrix",
    "detective", "badge", "vault", "label", "token", "class", "group",
    # Node IDs N0..N31
    *(f"N{i}" for i in range(32)),
    # Variable IDs v0..v15
    *(f"v{i}" for i in range(16)),
    # Aleatoric class IDs C00..C15
    *(f"C{i:02d}" for i in range(16)),
    # Small two-digit numbers 10..49
    *(f"{i}" for i in range(10, 50)),
    # Common Python variable/expression patterns
    "a=", "b=", "c=", "d=", "x=", "y=", "z=", "n=", "k=", "s=",
    "(a+b)", "(b+c)", "(a+c)", "(a-b)", "(b-c)", "(c-a)", "(a*2", "(b*2", "(c*2",
    "%17", "%19", "%13", "%11", "%23", "%29",
    "if a<b:", "if b<c:", "if c<a:", "if a>b:", "if b>c:",
    "if (a+b)%2==0:", "if (b+c)%2==0:", "if (a+c)%2==0:",
    "print(a)", "print(b)", "print(c)",
)


# Text whose encoding separates tokenizers that share a vocabulary but differ in
# normalization, prefix-space handling, or their treatment of marker strings.
_FINGERPRINT_PROBE = (
    "The theorem: for x in range(10), f(x) = x**2 + 1.\n"
    "    return [a, b]  # Tabs\tand  double  spaces\n"
    " leading space, UPPER lower, naïve café ∑ π, 3.14159, <pad> <think> </think> <ans>"
)


def _fingerprint(tokenizer: Any, id_token_pairs: Iterable[tuple[int, str]]) -> str:
    """Digest of a tokenizer's ID-to-token table and of how it encodes a fixed probe text.

    The table alone would miss two tokenizers with one vocabulary and different
    merge rules or normalization. Hashing behaviour instead of serialized state
    keeps the value stable across library versions.
    """
    payload = json.dumps(
        [
            sorted(id_token_pairs),
            tokenizer.encode(_FINGERPRINT_PROBE),
            tokenizer.encode_corpus(_FINGERPRINT_PROBE),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PilotTokenizer:
    """Deterministic lossless byte+symbol tokenizer with exact vocab_size=512.

    Token ID layout:
      - 0..7:     Special tokens (<pad>, <bos>, <eos>, <think>, </think>, <pause>, <sep>, <ans>)
      - 8..263:   Raw UTF-8 bytes 0x00..0xFF (guarantees lossless encoding of any string)
      - 264..511: Curated algorithmic/code/math multi-character tokens (and reserved slots)
    """

    BYTE_OFFSET = 8
    MULTI_OFFSET = 264
    DEFAULT_VOCAB_SIZE = 512

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        if vocab_size < self.MULTI_OFFSET:
            raise ValueError(
                f"vocab_size must be >= {self.MULTI_OFFSET}, got {vocab_size}"
            )
        self._vocab_size = vocab_size
        self._special_tokens = SPECIAL_TOKENS
        self._special_to_id: dict[str, int] = {
            tok: idx for idx, tok in enumerate(self._special_tokens)
        }

        # Deduplicate multi-char tokens while preserving deterministic order.
        multi_capacity = vocab_size - self.MULTI_OFFSET
        seen: set[str] = set(self._special_tokens)
        multi_tokens: list[str] = []
        for tok in _BASE_MULTI_TOKENS:
            if tok not in seen and len(tok.encode("utf-8")) >= 2:
                seen.add(tok)
                multi_tokens.append(tok)
                if len(multi_tokens) == multi_capacity:
                    break

        self._multi_tokens: tuple[str, ...] = tuple(multi_tokens)
        self._multi_to_id: dict[str, int] = {
            tok: self.MULTI_OFFSET + idx for idx, tok in enumerate(self._multi_tokens)
        }

        # Build complete id -> token string table (for inspection / debugging).
        self._id_to_str: list[str] = list(self._special_tokens)
        self._id_to_str.extend(f"<0x{b:02X}>" for b in range(256))
        self._id_to_str.extend(self._multi_tokens)
        while len(self._id_to_str) < self._vocab_size:
            self._id_to_str.append(f"<rsv_{len(self._id_to_str)}>")

        # Byte lengths for exact Bits-Per-Byte (BPB) accounting.
        self._byte_lengths: list[int] = [0] * len(self._special_tokens)
        self._byte_lengths.extend([1] * 256)
        self._byte_lengths.extend(len(t.encode("utf-8")) for t in self._multi_tokens)
        while len(self._byte_lengths) < self._vocab_size:
            self._byte_lengths.append(0)

        # Compile greedy longest-match regex: special tokens take priority, then multi tokens.
        special_alts = sorted(self._special_tokens, key=len, reverse=True)
        multi_alts = sorted(self._multi_tokens, key=len, reverse=True)
        pattern_parts = [re.escape(t) for t in special_alts] + [
            re.escape(t) for t in multi_alts
        ]
        self._pattern = re.compile("|".join(pattern_parts))
        # Corpus text is encoded without the special tokens, so a marker string in a
        # document stays ordinary text.
        self._text_pattern = (
            re.compile("|".join(re.escape(t) for t in multi_alts)) if multi_alts else None
        )

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def think_start_id(self) -> int:
        return 3

    @property
    def think_end_id(self) -> int:
        return 4

    @property
    def pause_id(self) -> int:
        return 5

    @property
    def sep_id(self) -> int:
        return 6

    @property
    def ans_id(self) -> int:
        return 7

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_ids(self) -> frozenset[int]:
        return frozenset(range(len(self._special_tokens)))

    def is_special(self, token_id: int) -> bool:
        return 0 <= token_id < len(self._special_tokens)

    def protocol_ids(self) -> dict[str, int]:
        """Each protocol token and its ID."""
        return dict(self._special_to_id)

    def fingerprint(self) -> str:
        return _fingerprint(self, enumerate(self._id_to_str))

    def token_to_id(self, token: str) -> int:
        if token in self._special_to_id:
            return self._special_to_id[token]
        if token in self._multi_to_id:
            return self._multi_to_id[token]
        raw = token.encode("utf-8")
        if len(raw) == 1:
            return self.BYTE_OFFSET + raw[0]
        raise KeyError(f"Token not in vocabulary: {token!r}")

    def id_to_token(self, token_id: int) -> str:
        if not (0 <= token_id < self._vocab_size):
            raise IndexError(f"token_id {token_id} out of range [0, {self._vocab_size})")
        return self._id_to_str[token_id]

    def token_byte_length(self, token_id: int) -> int:
        if not (0 <= token_id < self._vocab_size):
            raise IndexError(f"token_id {token_id} out of range [0, {self._vocab_size})")
        return self._byte_lengths[token_id]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode string to token IDs losslessly."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        self._encode_into(ids, text, self._pattern)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_corpus(self, text: str) -> list[int]:
        """Encode raw corpus text. Marker strings such as `<pad>` stay ordinary bytes."""
        ids: list[int] = []
        self._encode_into(ids, text, self._text_pattern)
        return ids

    def _encode_into(self, ids: list[int], text: str, pattern: re.Pattern[str] | None) -> None:
        pos = 0
        n = len(text)
        for match in pattern.finditer(text) if pattern is not None else ():
            start, end = match.span()
            if start > pos:
                for b in text[pos:start].encode("utf-8"):
                    ids.append(self.BYTE_OFFSET + b)
            tok = match.group(0)
            if tok in self._special_to_id:
                ids.append(self._special_to_id[tok])
            else:
                ids.append(self._multi_to_id[tok])
            pos = end

        if pos < n:
            for b in text[pos:].encode("utf-8"):
                ids.append(self.BYTE_OFFSET + b)

    def decode(
        self,
        ids: Sequence[int] | torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs back to string."""
        if isinstance(ids, torch.Tensor):
            id_list = ids.detach().cpu().tolist()
        else:
            id_list = list(ids)

        pieces: list[str] = []
        byte_buf = bytearray()

        def _flush_bytes() -> None:
            if byte_buf:
                pieces.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        multi_end = self.MULTI_OFFSET + len(self._multi_tokens)
        for idx in id_list:
            idx_int = int(idx)
            if self.BYTE_OFFSET <= idx_int < self.MULTI_OFFSET:
                byte_buf.append(idx_int - self.BYTE_OFFSET)
            elif self.MULTI_OFFSET <= idx_int < multi_end:
                _flush_bytes()
                pieces.append(self._multi_tokens[idx_int - self.MULTI_OFFSET])
            elif 0 <= idx_int < len(self._special_tokens):
                _flush_bytes()
                if not skip_special_tokens:
                    pieces.append(self._special_tokens[idx_int])
            elif multi_end <= idx_int < self._vocab_size:
                _flush_bytes()
                if not skip_special_tokens:
                    pieces.append(self._id_to_str[idx_int])
            else:
                raise IndexError(
                    f"token_id {idx_int} out of range [0, {self._vocab_size})"
                )

        _flush_bytes()
        return "".join(pieces)

    def collate(
        self,
        sequences: Sequence[Sequence[int]],
        pad_to: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Right-pad a batch of token ID sequences to a uniform tensor."""
        if not sequences:
            raise ValueError("Cannot collate an empty batch")
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if pad_to is None else max(pad_to, max(lengths))
        batch_size = len(sequences)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len > 0:
                input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :seq_len] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }

    def encode_batch(
        self,
        texts: Sequence[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode and right-pad a batch of strings."""
        seqs = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]
        if max_len is not None:
            seqs = [s[:max_len] for s in seqs]
        return self.collate(seqs)


class HFTokenizerAdapter:
    """Optional HuggingFace tokenizer adapter for H100 Stage 1/2 runs.

    Exposes the exact same property/method interface as `PilotTokenizer` while
    lazily importing `transformers` only when instantiated from a pretrained model.
    """

    def __init__(self, hf_tokenizer: Any) -> None:
        self._tok = hf_tokenizer
        # Register every protocol token. The wrapped tokenizer's own pad/bos/eos are
        # not used: many tokenizers ship them as one shared token, and a `<bos>` that
        # equals `<pad>` would be masked out of loss and attention.
        # All eight are registered, including any the vocabulary already holds as an
        # ordinary entry, so that each is matched whole wherever it appears.
        self._tok.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
        vocab = self._tok.get_vocab()
        self._ids: dict[str, int] = {t: self._id_of(t) for t in SPECIAL_TOKENS}
        if len(set(self._ids.values())) != len(SPECIAL_TOKENS):
            raise ValueError(
                f"HF tokenizer maps protocol tokens to non-distinct IDs: {self._ids}"
            )
        # Embedding rows needed. `len()` is right for dense IDs; the maximum covers
        # tokenizers whose IDs have gaps.
        self._vocab_size = max(len(self._tok), max((int(i) for i in vocab.values()), default=-1) + 1)
        for token, idx in self._ids.items():
            if self._tok.encode(f"a{token}b", add_special_tokens=False).count(idx) != 1:
                raise ValueError(
                    f"HF tokenizer does not encode {token!r} as a single token inside text, "
                    f"so prompts could lose the marker"
                )

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> HFTokenizerAdapter:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HFTokenizerAdapter.from_pretrained requires optional [h100] dependencies "
                "(install via `uv sync --extra h100`)."
            ) from exc
        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))

    def _id_of(self, token: str, fallback: int | None = None) -> int:
        idx = self._tok.convert_tokens_to_ids(token)
        if idx is None or idx == getattr(self._tok, "unk_token_id", -1):
            if fallback is not None:
                return int(fallback)
            raise KeyError(f"Special token {token!r} missing from HF tokenizer")
        return int(idx)

    @property
    def pad_id(self) -> int:
        return self._ids["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._ids["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._ids["<eos>"]

    @property
    def think_start_id(self) -> int:
        return self._ids["<think>"]

    @property
    def think_end_id(self) -> int:
        return self._ids["</think>"]

    @property
    def pause_id(self) -> int:
        return self._ids["<pause>"]

    @property
    def sep_id(self) -> int:
        return self._ids["<sep>"]

    @property
    def ans_id(self) -> int:
        return self._ids["<ans>"]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def special_ids(self) -> frozenset[int]:
        """The IDs of the protocol tokens (the counterpart of `PilotTokenizer.special_ids`)."""
        return frozenset(self._ids.values())

    def protocol_ids(self) -> dict[str, int]:
        """Each protocol token and its ID."""
        return dict(self._ids)

    def fingerprint(self) -> str:
        return _fingerprint(self, ((int(i), t) for t, i in self._tok.get_vocab().items()))

    def encode_corpus(self, text: str) -> list[int]:
        """Encode raw corpus text, asking the backend to keep marker strings as ordinary text.

        A backend that ignores `split_special_tokens` still returns protocol IDs for
        such strings; `data.corpus` rejects those documents.
        """
        return list(self._tok.encode(text, add_special_tokens=False, split_special_tokens=True))

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = list(self._tok.encode(text, add_special_tokens=False))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(
        self,
        ids: Sequence[int] | torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return str(self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens))

    def collate(
        self,
        sequences: Sequence[Sequence[int]],
        pad_to: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if not sequences:
            raise ValueError("Cannot collate an empty batch")
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if pad_to is None else max(pad_to, max(lengths))
        batch_size = len(sequences)
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            if seq_len > 0:
                input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :seq_len] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }


PILOT_TOKENIZER_SPEC = "pilot"


def build_tokenizer(spec: str = PILOT_TOKENIZER_SPEC) -> PilotTokenizer | HFTokenizerAdapter:
    """`"pilot"` gives the built-in `PilotTokenizer`; anything else is a HuggingFace name or path."""
    if spec == PILOT_TOKENIZER_SPEC:
        return PilotTokenizer()
    return HFTokenizerAdapter.from_pretrained(spec)
