"""Prompt-blind reference: what a predictor scores without reading the prompt.

A model that has learned only which answers each task uses, and not how the prompt
selects among them, predicts each task's answer marginal. Its NLL is the floor that any
claim of reasoning must beat. `EXP-001` arms scored worse than this on Regime 1, so a
comparison between them measured how close each came to the marginal, not reasoning.

The marginal is estimated from the training samples, the same data the arms saw, with
add-one smoothing over the answers seen per task plus one bucket for unseen answers. A
whole target sequence is one answer; its NLL is spread over its tokens so the result is
per target token, like `ParetoPoint.mean_nll`.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from phase1_pilot.data.records import TokenizedSample


@dataclass
class PromptBlindReference:
    mean_nll: float
    accuracy: float
    n_samples: int
    n_target_tokens: int
    mean_nll_by_task: dict[str, float] = field(default_factory=dict)
    mean_nll_by_depth: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _per_token_mean(groups: dict[Any, list[tuple[float, int]]]) -> dict[Any, float]:
    return {k: sum(n for n, _ in v) / sum(t for _, t in v) for k, v in sorted(groups.items())}


def prompt_blind_reference(
    train: Iterable[TokenizedSample],
    eval_samples: Sequence[TokenizedSample],
) -> PromptBlindReference:
    """Score `eval_samples` with each task's smoothed answer marginal from `train`."""
    counts: dict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    for s in train:
        counts[s.task][tuple(s.target_ids)] += 1

    by_task: dict[str, list[tuple[float, int]]] = defaultdict(list)
    by_depth: dict[int, list[tuple[float, int]]] = defaultdict(list)
    correct = 0
    for s in eval_samples:
        if s.task not in counts:
            raise ValueError(f"task {s.task!r} has no training samples")
        c = counts[s.task]
        answer = tuple(s.target_ids)
        denom = sum(c.values()) + len(c) + 1
        nll = -math.log((c[answer] + 1) / denom)  # unseen answers share the extra bucket
        n_tok = len(answer)
        by_task[s.task].append((nll, n_tok))
        by_depth[s.depth].append((nll, n_tok))
        mode = min(c, key=lambda a: (-c[a], a))  # most common; ties go to the smallest answer
        correct += int(mode == answer)

    rows = [r for v in by_task.values() for r in v]
    n_tok_total = sum(t for _, t in rows)
    return PromptBlindReference(
        mean_nll=sum(n for n, _ in rows) / max(1, n_tok_total),
        accuracy=correct / max(1, len(eval_samples)),
        n_samples=len(eval_samples),
        n_target_tokens=n_tok_total,
        mean_nll_by_task=_per_token_mean(by_task),
        mean_nll_by_depth=_per_token_mean(by_depth),
    )
