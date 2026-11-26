"""Shared test stand-ins: a HuggingFace-style tokenizer and synthetic corpus documents.

Neither `transformers` nor a network is available in the CPU dev environment, so the
real-text data path is exercised with these stand-ins.
"""

from __future__ import annotations

import random
from typing import Callable

import pytest

from phase1_pilot.data.corpus import Document
from phase1_pilot.data.tokenizer import HFTokenizerAdapter, PilotTokenizer


class FakeHFTokenizer:
    """The slice of the HuggingFace tokenizer API that `HFTokenizerAdapter` uses.

    IDs 0..255 are bytes and 256.. are multi-character merges, so the same text
    tokenizes differently when it is cut before encoding. The native end-of-text
    token and every token added later sit above the merges, which puts the
    project's protocol tokens at IDs far from the pilot's 0..7.

    Bytes are rotated so that a space is ID 0 and `#`, `$`, `%` are 3, 4, 5. Code
    that falls back to the pilot's literal IDs then mistakes ordinary text for
    `<pad>`, `<think>`, `</think>` or `<pause>`, and the tests can see it.
    """

    _BYTE_SHIFT = 32

    _MERGES: tuple[str, ...] = (
        " the ", "the", "ing", "tion", " = ", "def ", "return ", "(x)", " of ", "\n    ",
    )
    _NATIVE_EOS = "<|endoftext|>"

    def __init__(self, native_pad_is_eos: bool = False, honors_split_special_tokens: bool = True) -> None:
        """`native_pad_is_eos` mimics GPT-2-style tokenizers where pad, bos and eos are one token.

        `honors_split_special_tokens=False` mimics a backend that ignores that option.
        """
        self._vocab: dict[str, int] = {}
        for i, merge in enumerate(self._MERGES):
            self._vocab[merge] = 256 + i
        self._specials: list[str] = []
        self._honors_split = honors_split_special_tokens
        self._add(self._NATIVE_EOS)
        native = self._vocab[self._NATIVE_EOS]
        self.eos_token_id: int | None = native
        self.bos_token_id: int | None = native if native_pad_is_eos else None
        self.unk_token_id: int | None = None
        self.pad_token_id: int | None = native if native_pad_is_eos else None
        self._pad_token: str | None = self._NATIVE_EOS if native_pad_is_eos else None

    def _add(self, token: str) -> None:
        if token not in self._vocab:
            self._vocab[token] = 256 + len(self._vocab)
            self._specials.append(token)

    # -- HuggingFace surface -------------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def add_special_tokens(self, mapping: dict[str, list[str]]) -> int:
        added = mapping.get("additional_special_tokens", [])
        for token in added:
            self._add(token)
        return len(added)

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, token: str) -> None:
        self._add(token)
        self._pad_token = token
        self.pad_token_id = self._vocab[token]

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._vocab.get(token, self.unk_token_id)

    def __len__(self) -> int:
        return 256 + len(self._vocab)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ) -> list[int]:
        # Like a real HF tokenizer, special-token strings inside raw text match
        # unless the caller asks for them to be split into ordinary pieces.
        literal = split_special_tokens and self._honors_split
        candidates = self._MERGES if literal else tuple(self._vocab)
        pieces = sorted(candidates, key=len, reverse=True)
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            match = next((p for p in pieces if text.startswith(p, pos)), None)
            if match is not None:
                ids.append(self._vocab[match])
                pos += len(match)
            else:
                ids.extend((b - self._BYTE_SHIFT) % 256 for b in text[pos].encode("utf-8"))
                pos += 1
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        inverse = {v: k for k, v in self._vocab.items()}
        out: list[str] = []
        buf = bytearray()
        for i in ids:
            if i < 256:
                buf.append((i + self._BYTE_SHIFT) % 256)
                continue
            out.append(buf.decode("utf-8", errors="replace"))
            buf.clear()
            if not (skip_special_tokens and inverse[i] in self._specials):
                out.append(inverse[i])
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


@pytest.fixture
def make_fake_hf() -> type[FakeHFTokenizer]:
    """The raw stand-in class, for tests that need a non-default configuration."""
    return FakeHFTokenizer


@pytest.fixture
def hf_tokenizer() -> HFTokenizerAdapter:
    return HFTokenizerAdapter(FakeHFTokenizer())


def _real_hf_tokenizer():
    """A tiny byte-level BPE built with the real `transformers`/`tokenizers` libraries.

    It is trained locally on the stand-in vocabulary, so no Hub access is needed. Like
    GPT-2, it ships one native token for pad, bos and eos. Skips when the libraries
    are not installed (they are the optional `h100` extra).
    """
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.BPE())
    tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator([" ".join(_WORDS)] * 200, trainer)
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )


@pytest.fixture
def real_hf_tokenizer() -> HFTokenizerAdapter:
    return HFTokenizerAdapter(_real_hf_tokenizer())


@pytest.fixture(params=["pilot", "hf", "hf-real"])
def any_tokenizer(request: pytest.FixtureRequest) -> PilotTokenizer | HFTokenizerAdapter:
    """Run a test once per tokenizer family the data path must support."""
    if request.param == "pilot":
        return PilotTokenizer()
    if request.param == "hf":
        return HFTokenizerAdapter(FakeHFTokenizer())
    return HFTokenizerAdapter(_real_hf_tokenizer())


_WORDS: tuple[str, ...] = (
    "the", "theorem", "of", "returning", "def ", "integral", "function", "x", "=",
    "proof", "matrix", "for", "in", "range", "lemma", "equation", "(x)", "sum", "if",
    "value", "\n    ", "result", "series", "note", "where", "then", "is", "and",
)
_SOURCES: tuple[str, ...] = ("open-web-math", "codeparrot", "fineweb-edu")


@pytest.fixture
def make_documents() -> Callable[..., list[Document]]:
    """Factory for deterministic stand-in documents, interleaved across three sources."""

    def _make(n: int, seed: int = 0, min_words: int = 40, max_words: int = 400) -> list[Document]:
        rng = random.Random(seed)
        docs: list[Document] = []
        for i in range(n):
            n_words = rng.randint(min_words, max_words)
            text = " ".join(rng.choice(_WORDS) for _ in range(n_words))
            docs.append(Document(key=f"doc-{seed}-{i:05d}", text=text, source=_SOURCES[i % 3]))
        return docs

    return _make
