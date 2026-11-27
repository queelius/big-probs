"""Record types shared by the synthetic generators and the real-text corpus path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """Raw text sample across all 3 regimes."""

    prompt: str
    target: str
    full_text: str
    regime: str
    task: str
    depth: int
    answer_span_start: int


@dataclass(frozen=True)
class TokenizedSample:
    """Tokenized sample aligned for supervised NTP, latent BPTT, and SSL-as-RL."""

    input_ids: list[int]
    prompt_len: int
    target_ids: list[int]
    regime: str
    task: str
    depth: int
