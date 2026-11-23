"""Five mandatory diagnostic probes and falsification verdict engine (Track 1 §5.5).

Implements:
1. Diagnostic 1 — FLOPs-Matched Pareto Evaluator (`evaluate_pareto_point`)
2. Diagnostic 2 — Support-Boundary Probe (`evaluate_pass_at_k`, `unbiased_pass_at_k`)
3. Diagnostic 3 — Causal Thought Faithfulness (`evaluate_thought_faithfulness`,
   swap-sensitivity, format-matched truncation/perturbation, monotonic OOD depth scaling)
4. Diagnostic 4 — Distributional Calibration (`evaluate_calibration`, `compute_ece_and_brier`)
5. Diagnostic 5 — Falsification Verdict Engine (`compute_pilot_verdict`)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# JSON-serializable result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    """Diagnostic 1: FLOPs-matched accuracy, Hard-BPB, and inference FLOPs/token."""

    arm_name: str
    variant: str
    total_params: int
    non_embedding_params: int
    exact_accuracy: float
    token_accuracy: float
    mean_nll: float
    hard_bpb: float
    inference_flops_per_token: float
    extra_steps_per_target: float
    accuracy_by_depth: dict[int, float] = field(default_factory=dict)
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PassAtKResult:
    """Diagnostic 2: Unbiased pass@k support-boundary probe broken down by hop depth."""

    arm_name: str
    n_samples_per_prompt: int
    ks: list[int]
    overall: dict[int, float]
    by_depth: dict[int, dict[int, float]]
    ood_depths: list[int]
    ood_pass_at_max_k: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThoughtFaithfulnessResult:
    """Diagnostic 3: Causal thought faithfulness (swap, format-matched trunc, OOD depth)."""

    arm_name: str
    mode: str  # "discrete", "continuous", "pause", "ntp"
    nll_with_thought: float
    nll_swapped: float
    delta_nll_swap: float
    nll_truncated: float
    delta_nll_trunc: float
    delta_nll_swap_by_depth: dict[int, float] = field(default_factory=dict)
    delta_nll_trunc_by_depth: dict[int, float] = field(default_factory=dict)
    accuracy_lift_by_depth: dict[int, float] = field(default_factory=dict)
    monotonic_swap_depth_scaling: bool = False
    monotonic_accuracy_lift_scaling: bool = False
    monotonic_depth_scaling: bool = False
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationResult:
    """Diagnostic 4: Distributional calibration (ECE, proper Brier score, overconfidence)."""

    arm_name: str
    regime: str
    ece: float
    mean_brier_score: float  # Strictly proper score in [-2, 0] (higher is better)
    mean_brier_loss: float  # Equivalent positive loss = -mean_brier_score in [0, 2]
    mean_confidence: float
    accuracy: float
    overconfidence_gap: float  # mean_confidence - accuracy (>0 => overconfident)
    bin_stats: list[dict[str, float]] = field(default_factory=list)
    n_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArmEvalBundle:
    """Complete evaluation bundle for one experimental arm fed into the verdict engine."""

    arm_name: str
    pareto: ParetoPoint
    pass_at_k: PassAtKResult | None = None
    faithfulness: ThoughtFaithfulnessResult | None = None
    calibration_primary: CalibrationResult | None = None
    calibration_aleatoric: CalibrationResult | None = None
    is_rl_arm: bool = False
    is_teacher_free: bool = True
    is_continuous_or_hybrid: bool = False
    is_warm_start: bool = False
    nothought_nll_drift: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PilotVerdict:
    """Diagnostic 5: Structured Track 1 §5.5 falsification verdict and summary table."""

    status: str  # "CONFIRMED", "PARTIAL_CONFIRMATION", "FALSIFIED"
    best_rl_arm: str | None
    checks: dict[str, bool]
    arm_checks: dict[str, dict[str, bool]]
    metrics_summary: dict[str, Any]
    rationale: str
    markdown_table: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure mathematical core helpers (directly unit-testable)
# ---------------------------------------------------------------------------


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n - c, k) / C(n, k).

    Follows Chen et al. (2021) and Yue et al. (2025, arXiv:2504.13837).
    Computed via exact product ratio to prevent combinatorial overflow.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if c < 0 or c > n:
        raise ValueError(f"c must satisfy 0 <= c <= n, got c={c}, n={n}")
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    ratio = math.prod((n - c - i) / (n - i) for i in range(k))
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def compute_pass_at_k_from_counts(
    counts_by_depth: dict[int, Sequence[tuple[int, int]]],
    ks: Sequence[int] = (1, 4, 16, 64),
    ood_depths: Sequence[int] = (6, 8),
    arm_name: str = "",
) -> PassAtKResult:
    """Aggregate unbiased pass@k from per-depth lists of (n, c) counts."""
    all_pairs: list[tuple[int, int]] = []
    for pairs in counts_by_depth.values():
        all_pairs.extend(pairs)
    if not all_pairs:
        return PassAtKResult(
            arm_name=arm_name,
            n_samples_per_prompt=0,
            ks=[],
            overall={},
            by_depth={},
            ood_depths=list(ood_depths),
            ood_pass_at_max_k=0.0,
        )

    min_n = min(n for n, _ in all_pairs)
    valid_ks = sorted({int(k) for k in ks if 1 <= int(k) <= min_n})
    if not valid_ks:
        valid_ks = [min_n]

    overall: dict[int, float] = {}
    for k in valid_ks:
        vals = [unbiased_pass_at_k(n, c, k) for n, c in all_pairs]
        overall[k] = float(sum(vals) / len(vals))

    by_depth: dict[int, dict[int, float]] = {}
    for depth in sorted(counts_by_depth.keys()):
        pairs = counts_by_depth[depth]
        if not pairs:
            continue
        by_depth[int(depth)] = {
            k: float(sum(unbiased_pass_at_k(n, c, k) for n, c in pairs) / len(pairs))
            for k in valid_ks
        }

    max_k = max(valid_ks)
    ood_pairs: list[tuple[int, int]] = []
    present_ood: list[int] = []
    for d in ood_depths:
        if d in counts_by_depth and counts_by_depth[d]:
            ood_pairs.extend(counts_by_depth[d])
            present_ood.append(int(d))

    if ood_pairs:
        ood_pass_max = float(
            sum(unbiased_pass_at_k(n, c, max_k) for n, c in ood_pairs) / len(ood_pairs)
        )
    else:
        ood_pass_max = overall[max_k]

    return PassAtKResult(
        arm_name=arm_name,
        n_samples_per_prompt=min_n,
        ks=valid_ks,
        overall=overall,
        by_depth=by_depth,
        ood_depths=present_ood if present_ood else list(ood_depths),
        ood_pass_at_max_k=ood_pass_max,
    )


def corrupt_discrete_thought_format_matched(
    thought_ids: Tensor,
    think_start_id: int,
    think_end_id: int,
    vocab_size: int,
    generator: torch.Generator | None = None,
    special_ids: Sequence[int] | None = None,
) -> Tensor:
    """Corrupt internal tokens strictly between <think> and </think>.

    Preserves:
    - Exact sequence length
    - Leading `think_start_id` and trailing `think_end_id` delimiters
    Guarantees every internal token between the delimiters is replaced with a
    different non-special token ID.
    """
    if thought_ids.ndim != 1:
        raise ValueError(f"Expected 1D thought_ids tensor, got shape {tuple(thought_ids.shape)}")

    out = thought_ids.clone()
    length = int(out.numel())
    if length == 0:
        return out

    # Identify inner span strictly inside delimiters
    start_idx = 1 if int(out[0].item()) == think_start_id else 0
    end_idx = length - 1 if (length > start_idx and int(out[-1].item()) == think_end_id) else length
    if end_idx <= start_idx:
        return out

    excluded = set(special_ids) if special_ids is not None else {0, 1, 2, 3, 4, 5, 6, 7}
    excluded.update({think_start_id, think_end_id})
    candidates = [tok for tok in range(vocab_size) if tok not in excluded]
    if len(candidates) < 2:
        candidates = [tok for tok in range(vocab_size) if tok not in {think_start_id, think_end_id}]
    if not candidates:
        return out

    cand_tensor = torch.tensor(candidates, dtype=out.dtype, device=out.device)
    n_cand = len(candidates)
    for pos in range(start_idx, end_idx):
        orig = int(out[pos].item())
        idx = int(torch.randint(0, n_cand, (1,), generator=generator).item())
        new_tok = int(cand_tensor[idx].item())
        if new_tok == orig and n_cand > 1:
            new_tok = int(cand_tensor[(idx + 1) % n_cand].item())
        out[pos] = new_tok

    return out


def perturb_continuous_latents_norm_matched(
    latents: Tensor,
    generator: torch.Generator | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Apply norm-matched orthogonal perturbation to continuous latent vectors.

    For each latent step vector z in R^D, generates a random vector r, projects out
    its component along z (Gram-Schmidt orthogonalization), and rescales the
    orthogonal vector to match ||z||_2 exactly. Preserves step count and per-step
    L2 norm while destroying directional semantic content.
    """
    orig_dtype = latents.dtype
    z = latents.float()
    noise = torch.randn(z.shape, generator=generator, device="cpu", dtype=torch.float32).to(z.device)
    z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
    z_norm = torch.sqrt(z_norm_sq.clamp(min=0.0))

    # Gram-Schmidt orthogonal component
    proj = ((noise * z).sum(dim=-1, keepdim=True) / (z_norm_sq + eps)) * z
    orth = noise - proj
    orth_norm = torch.sqrt((orth * orth).sum(dim=-1, keepdim=True).clamp(min=eps))
    perturbed = orth * (z_norm / orth_norm)

    # Handle exact zero-norm latent vectors gracefully
    perturbed = torch.where(z_norm > eps, perturbed, torch.zeros_like(z))
    return perturbed.to(orig_dtype)


def check_monotonic_depth_scaling(
    values_by_depth: dict[int, float],
    tol: float = 1e-6,
) -> bool:
    """Return True iff values grow monotonically across sorted hop depths (non-flat)."""
    if len(values_by_depth) < 2:
        return False
    sorted_depths = sorted(values_by_depth.keys())
    vals = [float(values_by_depth[d]) for d in sorted_depths]
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return False
    non_decreasing = all(vals[i + 1] >= vals[i] - tol for i in range(len(vals) - 1))
    strictly_higher_end = vals[-1] > vals[0] + tol
    return bool(non_decreasing and strictly_higher_end)


def compute_faithfulness_summary(
    records: Sequence[dict[str, float | int]],
    arm_name: str,
    mode: str,
) -> ThoughtFaithfulnessResult:
    """Aggregate per-sample faithfulness records into a `ThoughtFaithfulnessResult`.

    Each record must contain:
      - `depth`: int
      - `nll_thought`: float
      - `nll_swap`: float
      - `nll_trunc`: float
    Optional keys:
      - `acc_thought`: float (0 or 1)
      - `acc_nothought`: float (0 or 1)
    """
    if not records:
        return ThoughtFaithfulnessResult(
            arm_name=arm_name,
            mode=mode,
            nll_with_thought=0.0,
            nll_swapped=0.0,
            delta_nll_swap=0.0,
            nll_truncated=0.0,
            delta_nll_trunc=0.0,
            n_samples=0,
        )

    n = len(records)
    mean_nll = sum(float(r["nll_thought"]) for r in records) / n
    mean_swap = sum(float(r["nll_swap"]) for r in records) / n
    mean_trunc = sum(float(r["nll_trunc"]) for r in records) / n

    by_depth: dict[int, list[dict[str, float | int]]] = {}
    for r in records:
        d = int(r["depth"])
        by_depth.setdefault(d, []).append(r)

    swap_by_depth: dict[int, float] = {}
    trunc_by_depth: dict[int, float] = {}
    acc_lift_by_depth: dict[int, float] = {}
    has_acc_lift = all(("acc_thought" in r and "acc_nothought" in r) for r in records)

    for d in sorted(by_depth.keys()):
        group = by_depth[d]
        m = len(group)
        swap_by_depth[d] = sum(float(r["nll_swap"]) - float(r["nll_thought"]) for r in group) / m
        trunc_by_depth[d] = sum(float(r["nll_trunc"]) - float(r["nll_thought"]) for r in group) / m
        if has_acc_lift:
            acc_lift_by_depth[d] = (
                sum(float(r["acc_thought"]) - float(r["acc_nothought"]) for r in group) / m
            )

    mono_swap = check_monotonic_depth_scaling(swap_by_depth)
    mono_acc = check_monotonic_depth_scaling(acc_lift_by_depth) if has_acc_lift else True
    mono_both = bool(mono_swap and (mono_acc if has_acc_lift else True))

    return ThoughtFaithfulnessResult(
        arm_name=arm_name,
        mode=mode,
        nll_with_thought=float(mean_nll),
        nll_swapped=float(mean_swap),
        delta_nll_swap=float(mean_swap - mean_nll),
        nll_truncated=float(mean_trunc),
        delta_nll_trunc=float(mean_trunc - mean_nll),
        delta_nll_swap_by_depth=swap_by_depth,
        delta_nll_trunc_by_depth=trunc_by_depth,
        accuracy_lift_by_depth=acc_lift_by_depth,
        monotonic_swap_depth_scaling=mono_swap,
        monotonic_accuracy_lift_scaling=mono_acc if has_acc_lift else False,
        monotonic_depth_scaling=mono_both,
        n_samples=n,
    )


def compute_ece_and_brier(
    logits: Tensor,
    targets: Tensor,
    n_bins: int = 10,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """Compute 10-bin Expected Calibration Error (ECE) and exact proper Brier score.

    Args:
        logits: Tensor of shape `[..., V]`.
        targets: Integer tensor of shape `[...]`.
        n_bins: Number of equal-width confidence bins on `[0, 1]` (default 10).
        ignore_index: Target ID to ignore.

    Returns:
        Dictionary with `ece`, `mean_brier_score` (in `[-2, 0]`), `mean_brier_loss`
        (in `[0, 2]`), `mean_confidence`, `accuracy`, `overconfidence_gap`,
        `bin_stats`, and `n_tokens`.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits batch shape {tuple(logits.shape[:-1])} != targets shape {tuple(targets.shape)}"
        )

    flat_logits = logits.detach().float().reshape(-1, logits.shape[-1])
    flat_targets = targets.detach().long().reshape(-1)
    valid = flat_targets != ignore_index
    flat_logits = flat_logits[valid]
    flat_targets = flat_targets[valid]

    n_tokens = int(flat_targets.numel())
    if n_tokens == 0:
        return {
            "ece": 0.0,
            "mean_brier_score": 0.0,
            "mean_brier_loss": 0.0,
            "mean_confidence": 0.0,
            "accuracy": 0.0,
            "overconfidence_gap": 0.0,
            "bin_stats": [],
            "n_tokens": 0,
        }

    probs = torch.softmax(flat_logits, dim=-1)
    confidences, preds = probs.max(dim=-1)
    correct = (preds == flat_targets).float()

    # Exact O(|V|) strictly proper Brier score: 2*q(v*) - sum_v q(v)^2 - 1 in [-2, 0]
    target_probs = probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
    sum_sq = (probs * probs).sum(dim=-1)
    brier_scores = (2.0 * target_probs - sum_sq - 1.0).clamp(min=-2.0, max=0.0)
    mean_brier_score = float(brier_scores.mean().item())
    mean_brier_loss = float((-brier_scores).mean().item())

    # 10 equal-width bins on [0, 1]
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0
    bin_stats: list[dict[str, float]] = []
    for b in range(n_bins):
        lo = float(bin_boundaries[b].item())
        hi = float(bin_boundaries[b + 1].item())
        if b == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum().item())
        if count > 0:
            avg_conf = float(confidences[in_bin].mean().item())
            avg_acc = float(correct[in_bin].mean().item())
            gap = abs(avg_acc - avg_conf)
            ece += (count / n_tokens) * gap
        else:
            avg_conf = 0.0
            avg_acc = 0.0
            gap = 0.0
        bin_stats.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "count": float(count),
                "avg_conf": avg_conf,
                "avg_acc": avg_acc,
                "gap": gap,
            }
        )

    mean_conf = float(confidences.mean().item())
    acc = float(correct.mean().item())
    return {
        "ece": float(max(0.0, min(1.0, ece))),
        "mean_brier_score": mean_brier_score,
        "mean_brier_loss": mean_brier_loss,
        "mean_confidence": mean_conf,
        "accuracy": acc,
        "overconfidence_gap": float(mean_conf - acc),
        "bin_stats": bin_stats,
        "n_tokens": n_tokens,
    }


# ---------------------------------------------------------------------------
# Internal model-inspection and forward-pass adapters
# ---------------------------------------------------------------------------


def _infer_device(model: nn.Module, device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_base_lm(model: nn.Module) -> nn.Module:
    """Return underlying LlamaLM whether `model` is raw or wrapped (B3/B4/I2)."""
    if hasattr(model, "base_lm"):
        return model.base_lm
    return model


def _get_model_config(model: nn.Module) -> Any:
    if hasattr(model, "config"):
        return model.config
    base = _get_base_lm(model)
    if hasattr(base, "config"):
        return base.config
    raise AttributeError("Model has no `.config` or `.base_lm.config` attribute.")


def _get_variant_name(model: nn.Module) -> str:
    if hasattr(model, "n_pause_tokens") and hasattr(model, "base_lm"):
        return "pause"
    if hasattr(model, "n_latent_steps") and hasattr(model, "base_lm"):
        if getattr(model, "hybrid_gate", False):
            return "hybrid_latent"
        return "continuous_latent"
    cfg = _get_model_config(model)
    return str(getattr(cfg, "variant", "ntp"))


def _extract_sample_fields(sample: Any, tokenizer: Any) -> tuple[list[int], list[int], int, str]:
    """Normalize a Sample / TokenizedSample / dict into (prompt_ids, target_ids, depth, target_str)."""
    if isinstance(sample, dict):
        if "prompt_ids" in sample and "target_ids" in sample:
            p_ids = list(sample["prompt_ids"])
            t_ids = list(sample["target_ids"])
        elif "input_ids" in sample and "prompt_len" in sample:
            ids = sample["input_ids"]
            ids_list = ids.tolist() if isinstance(ids, Tensor) else list(ids)
            plen = int(sample["prompt_len"])
            p_ids = ids_list[:plen]
            if "target_ids" in sample:
                t_raw = sample["target_ids"]
                t_ids = t_raw.tolist() if isinstance(t_raw, Tensor) else list(t_raw)
            else:
                t_ids = ids_list[plen:]
        else:
            p_ids = tokenizer.encode(sample["prompt"], add_bos=True, add_eos=False)
            t_ids = tokenizer.encode(sample["target"], add_bos=False, add_eos=False)
        depth = int(sample.get("depth", 0))
        target_str = str(sample.get("target", ""))
    elif hasattr(sample, "prompt_len") and hasattr(sample, "input_ids"):
        ids = sample.input_ids
        ids_list = ids.tolist() if isinstance(ids, Tensor) else list(ids)
        plen = int(sample.prompt_len)
        p_ids = ids_list[:plen]
        t_raw = getattr(sample, "target_ids", ids_list[plen:])
        t_ids = t_raw.tolist() if isinstance(t_raw, Tensor) else list(t_raw)
        depth = int(getattr(sample, "depth", 0))
        target_str = str(getattr(sample, "target", ""))
    elif hasattr(sample, "prompt") and hasattr(sample, "target"):
        p_ids = tokenizer.encode(sample.prompt, add_bos=True, add_eos=False)
        t_ids = tokenizer.encode(sample.target, add_bos=False, add_eos=False)
        depth = int(getattr(sample, "depth", 0))
        target_str = str(sample.target)
    else:
        raise TypeError(f"Unsupported sample object of type {type(sample)!r}")

    pad_id = getattr(tokenizer, "pad_id", 0)
    eos_id = getattr(tokenizer, "eos_id", 2)
    t_ids = [int(t) for t in t_ids if int(t) not in (pad_id, eos_id, -100)]
    p_ids = [int(t) for t in p_ids if int(t) != pad_id]
    if not target_str and t_ids and hasattr(tokenizer, "decode"):
        target_str = tokenizer.decode(t_ids)
    return p_ids, t_ids, depth, target_str


def _strip_special_for_match(token_ids: Sequence[int], tokenizer: Any) -> list[int]:
    pad_id = getattr(tokenizer, "pad_id", 0)
    eos_id = getattr(tokenizer, "eos_id", 2)
    out: list[int] = []
    for t in token_ids:
        ti = int(t)
        if ti == eos_id:
            break
        if ti == pad_id:
            continue
        out.append(ti)
    return out


def _embed_tokens(base_lm: nn.Module, token_ids: Tensor) -> Tensor:
    """Embed token IDs using LlamaLM's `tok_embeddings` (or compatible attribute)."""
    for attr in ("tok_embeddings", "embed_tokens", "tok_emb", "token_emb", "wte"):
        layer = getattr(base_lm, attr, None)
        if isinstance(layer, nn.Embedding):
            return layer(token_ids)
    raise AttributeError(f"Could not find token embedding layer on {type(base_lm).__name__}")


def _compute_continuous_latent_embeds(
    model: nn.Module,
    prompt_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run B4/I2 ContinuousLatentWrapper latent loop and return (prompt_embeds, latent_embeds)."""
    base_lm = _get_base_lm(model)
    n_steps = int(getattr(model, "n_latent_steps", 0))
    prompt_embeds = _embed_tokens(base_lm, prompt_ids)  # [B, T_p, D]

    if n_steps <= 0:
        return prompt_embeds, prompt_embeds[:, :0, :]

    # Prefer ContinuousLatentWrapper's native KV-cached latent generator
    if hasattr(model, "generate_latents"):
        latent_embeds, _, _ = model.generate_latents(prompt_ids, n_steps=n_steps)
        return prompt_embeds, latent_embeds

    curr_embeds = prompt_embeds
    latent_list: list[Tensor] = []
    for _ in range(n_steps):
        out = base_lm(inputs_embeds=curr_embeds, return_hidden=True)
        last_hidden = out.hidden_states[:, -1:, :]
        if hasattr(model, "project_latent"):
            next_emb = model.project_latent(last_hidden)
        elif hasattr(model, "latent_proj") and hasattr(model, "latent_norm"):
            h_norm = model.latent_norm(last_hidden)
            next_emb = h_norm + model.latent_proj(h_norm)
        elif hasattr(model, "latent_proj"):
            next_emb = model.latent_proj(last_hidden)
        else:
            next_emb = last_hidden
        latent_list.append(next_emb)
        curr_embeds = torch.cat([curr_embeds, next_emb], dim=1)

    latent_embeds = torch.cat(latent_list, dim=1)
    return prompt_embeds, latent_embeds


def _get_pause_embeds(model: nn.Module, batch_size: int, device: torch.device) -> Tensor:
    """Retrieve `[B, n_pause_tokens, D]` pause embeddings from a PauseTokenWrapper."""
    base_lm = _get_base_lm(model)
    n_pause = int(getattr(model, "n_pause_tokens", 0))
    if hasattr(model, "_get_pause_embeds"):
        dtype = base_lm.tok_embeddings.weight.dtype if hasattr(base_lm, "tok_embeddings") else torch.float32
        pe = model._get_pause_embeds(device=device, dtype=dtype)  # [P, D]
        return pe.unsqueeze(0).expand(batch_size, -1, -1)
    if getattr(model, "pause_embeddings", None) is not None:
        pe = model.pause_embeddings
        if pe.ndim == 2:
            pe = pe.unsqueeze(0)
        return pe.expand(batch_size, -1, -1).to(device)
    pause_id = int(getattr(model, "pause_token_id", 5))
    pause_ids = torch.full((batch_size, n_pause), pause_id, dtype=torch.long, device=device)
    return _embed_tokens(base_lm, pause_ids)


def _forward_target_logits_with_prefix_embeds(
    base_lm: nn.Module,
    prefix_embeds: Tensor,
    target_ids: Tensor,
) -> Tensor:
    """Compute teacher-forced logits `[B, M, V]` predicting each token in `target_ids` `[B, M]`."""
    m = target_ids.shape[1]
    if m == 0:
        vocab = base_lm.config.vocab_size
        return torch.empty((prefix_embeds.shape[0], 0, vocab), device=prefix_embeds.device)
    if m > 1:
        tgt_in_embeds = _embed_tokens(base_lm, target_ids[:, :-1])
        full_embeds = torch.cat([prefix_embeds, tgt_in_embeds], dim=1)
    else:
        full_embeds = prefix_embeds
    max_len = int(getattr(base_lm.config, "max_seq_len", 256))
    if full_embeds.shape[1] > max_len:
        full_embeds = full_embeds[:, -max_len:, :]
    out = base_lm(inputs_embeds=full_embeds)
    return out.logits[:, -m:, :]


def _sample_next_token(
    logits: Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample or greedily decode next token from `[B, V]` logits."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    scaled = logits.float() / float(temperature)
    probs = torch.softmax(scaled, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        remove = cumsum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        if generator is not None:
            sampled_sorted = torch.multinomial(sorted_probs.cpu(), num_samples=1, generator=generator).to(
                probs.device
            )
        else:
            sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(dim=-1, index=sampled_sorted).squeeze(-1)
    if generator is not None:
        return torch.multinomial(probs.cpu(), num_samples=1, generator=generator).squeeze(-1).to(
            probs.device
        )
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def _embed_hybrid_thought_sequence(
    model: nn.Module,
    prompt_ids: Tensor,
    thought_ids: Tensor,
) -> Tensor:
    """Compute HRPO-blended embeddings `[B, L, D]` matching `recompute_hybrid_logprobs_and_continuation`.

    Runs `L` HRPO-gated steps directly after `prompt_ids` with NO prepended pure continuous
    latents and NO `<think>`/`</think>` delimiters, matching `ContinuousLatentWrapper.rollout_hybrid`.
    """
    base_lm = _get_base_lm(model)
    max_len = int(getattr(base_lm.config, "max_seq_len", 256))
    steps = int(thought_ids.shape[1])
    if steps == 0:
        d = int(base_lm.config.d_model)
        return torch.empty((prompt_ids.shape[0], 0, d), device=prompt_ids.device)
    if not hasattr(model, "blend_hybrid"):
        return _embed_tokens(base_lm, thought_ids)

    p_ids = prompt_ids[:, -max_len:] if prompt_ids.shape[1] > max_len else prompt_ids
    prefix_out = base_lm(input_ids=p_ids, use_cache=True, return_hidden=True)
    past_kvs = prefix_out.past_kvs
    h_cur = prefix_out.hidden_states[:, -1:, :]
    cur_len = p_ids.shape[1]

    step_embeds: list[Tensor] = []
    for step in range(steps):
        tok = thought_ids[:, step : step + 1]
        blended = model.blend_hybrid(h_cur, tok)
        step_embeds.append(blended)
        if step + 1 < steps:
            if past_kvs is not None and cur_len + step + 1 < max_len:
                out = base_lm(
                    inputs_embeds=blended, past_kvs=past_kvs, use_cache=True, return_hidden=True
                )
                past_kvs = out.past_kvs
                h_cur = out.hidden_states
            else:
                curr = torch.cat([_embed_tokens(base_lm, p_ids), *step_embeds], dim=1)[:, -max_len:, :]
                out = base_lm(inputs_embeds=curr, use_cache=False, return_hidden=True)
                h_cur = out.hidden_states[:, -1:, :]
    return torch.cat(step_embeds, dim=1)


@torch.no_grad()
def _generate_hybrid_thought_and_embeds(
    model: nn.Module,
    prompt_ids: Tensor,
    thought_len: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate `(thought_ids [B, L], blended_embeds [B, L, D])` matching `rollout_hybrid`."""
    base_lm = _get_base_lm(model)
    bsz = prompt_ids.shape[0]
    device = prompt_ids.device
    max_len = int(getattr(base_lm.config, "max_seq_len", 256))

    if thought_len <= 0:
        d = int(base_lm.config.d_model)
        return (
            torch.empty((bsz, 0), dtype=torch.long, device=device),
            torch.empty((bsz, 0, d), device=device),
        )

    p_ids = prompt_ids[:, -max_len:] if prompt_ids.shape[1] > max_len else prompt_ids
    prefix_out = base_lm(input_ids=p_ids, use_cache=True, return_hidden=True)
    past_kvs = prefix_out.past_kvs
    h_cur = prefix_out.hidden_states[:, -1:, :]
    logits_cur = prefix_out.logits[:, -1, :]
    cur_len = p_ids.shape[1]

    tok_list: list[Tensor] = []
    emb_list: list[Tensor] = []
    for step in range(thought_len):
        nxt = _sample_next_token(logits_cur, temperature, top_p, generator).unsqueeze(-1)
        if hasattr(model, "blend_hybrid"):
            blended = model.blend_hybrid(h_cur, nxt)
        else:
            blended = _embed_tokens(base_lm, nxt)
        tok_list.append(nxt)
        emb_list.append(blended)
        if step + 1 < thought_len:
            if past_kvs is not None and cur_len + step + 1 < max_len:
                step_out = base_lm(
                    inputs_embeds=blended, past_kvs=past_kvs, use_cache=True, return_hidden=True
                )
                past_kvs = step_out.past_kvs
                h_cur = step_out.hidden_states
                logits_cur = step_out.logits[:, -1, :]
            else:
                curr = torch.cat([_embed_tokens(base_lm, p_ids), *emb_list], dim=1)[:, -max_len:, :]
                step_out = base_lm(inputs_embeds=curr, use_cache=False, return_hidden=True)
                h_cur = step_out.hidden_states[:, -1:, :]
                logits_cur = step_out.logits[:, -1, :]

    return torch.cat(tok_list, dim=1), torch.cat(emb_list, dim=1)


@torch.no_grad()
def _generate_discrete_thought(
    model: nn.Module,
    prompt_ids: Tensor,
    tokenizer: Any,
    thought_len: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate discrete thought sequence (`[B, thought_len + 2]` for I1, `[B, thought_len]` for I2)."""
    if _get_variant_name(model) == "hybrid_latent":
        thought_ids, _ = _generate_hybrid_thought_and_embeds(
            model,
            prompt_ids,
            thought_len=thought_len,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        return thought_ids

    base_lm = _get_base_lm(model)
    bsz = prompt_ids.shape[0]
    device = prompt_ids.device
    think_start_id = int(getattr(tokenizer, "think_start_id", 3))
    think_end_id = int(getattr(tokenizer, "think_end_id", 4))
    max_len = int(getattr(base_lm.config, "max_seq_len", 256))

    start_tok = torch.full((bsz, 1), think_start_id, dtype=torch.long, device=device)
    seq = torch.cat([prompt_ids, start_tok], dim=1)
    if seq.shape[1] > max_len:
        seq = seq[:, -max_len:]

    # Use KV-cache if supported
    out = base_lm(input_ids=seq, use_cache=True)
    past_kvs = out.past_kvs
    next_logits = out.logits[:, -1, :]
    tokens = [start_tok]
    for step in range(thought_len):
        nxt = _sample_next_token(next_logits, temperature, top_p, generator).unsqueeze(-1)
        tokens.append(nxt)
        if step + 1 < thought_len:
            if past_kvs is not None and seq.shape[1] + step + 1 < max_len:
                out = base_lm(input_ids=nxt, past_kvs=past_kvs, use_cache=True)
                past_kvs = out.past_kvs
                next_logits = out.logits[:, -1, :]
            else:
                seq = torch.cat([seq, nxt], dim=1)[:, -max_len:]
                out = base_lm(input_ids=seq, use_cache=False)
                next_logits = out.logits[:, -1, :]

    end_tok = torch.full((bsz, 1), think_end_id, dtype=torch.long, device=device)
    tokens.append(end_tok)
    return torch.cat(tokens, dim=1)


@torch.no_grad()
def _build_prefix_embeds_for_model(
    model: nn.Module,
    prompt_ids: Tensor,
    tokenizer: Any,
    use_thought: bool = False,
    thought_len: int = 8,
    temperature: float = 0.0,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int]:
    """Construct prefix embeddings `[B, T_prefix, D]` and return `(prefix_embeds, extra_steps)`."""
    base_lm = _get_base_lm(model)
    variant = _get_variant_name(model)
    bsz = prompt_ids.shape[0]
    device = prompt_ids.device

    if variant == "pause":
        prompt_embeds = _embed_tokens(base_lm, prompt_ids)
        pause_embeds = _get_pause_embeds(model, bsz, device)
        return torch.cat([prompt_embeds, pause_embeds], dim=1), int(pause_embeds.shape[1])

    if variant == "continuous_latent":
        prompt_embeds, latent_embeds = _compute_continuous_latent_embeds(model, prompt_ids)
        return torch.cat([prompt_embeds, latent_embeds], dim=1), int(latent_embeds.shape[1])

    if variant == "hybrid_latent":
        if use_thought and thought_len > 0:
            # Match ContinuousLatentWrapper.rollout_hybrid & recompute_hybrid_logprobs_and_continuation:
            # `thought_len` HRPO-blended steps directly after prompt_ids (no extra prepended latents,
            # no <think>/</think> delimiters, exact extra_steps == thought_len).
            prompt_embeds = _embed_tokens(base_lm, prompt_ids)
            _, blended_embeds = _generate_hybrid_thought_and_embeds(
                model,
                prompt_ids,
                thought_len=thought_len,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            return torch.cat([prompt_embeds, blended_embeds], dim=1), int(blended_embeds.shape[1])
        prompt_embeds, latent_embeds = _compute_continuous_latent_embeds(model, prompt_ids)
        return torch.cat([prompt_embeds, latent_embeds], dim=1), int(latent_embeds.shape[1])

    # Standard NTP or discrete-thought RL model
    prompt_embeds = _embed_tokens(base_lm, prompt_ids)
    if use_thought and thought_len > 0:
        thought_ids = _generate_discrete_thought(
            model,
            prompt_ids,
            tokenizer,
            thought_len=thought_len,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        thought_embeds = _embed_tokens(base_lm, thought_ids)
        return torch.cat([prompt_embeds, thought_embeds], dim=1), int(thought_ids.shape[1])

    return prompt_embeds, 0


@torch.no_grad()
def _decode_target_from_prefix_embeds(
    base_lm: nn.Module,
    prefix_embeds: Tensor,
    n_target_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressively decode `n_target_tokens` given `prefix_embeds` `[B, T, D]`."""
    max_len = int(getattr(base_lm.config, "max_seq_len", 256))
    if prefix_embeds.shape[1] > max_len:
        prefix_embeds = prefix_embeds[:, -max_len:, :]

    out = base_lm(inputs_embeds=prefix_embeds, use_cache=True)
    past_kvs = out.past_kvs
    next_logits = out.logits[:, -1, :]
    preds: list[Tensor] = []
    cur_len = prefix_embeds.shape[1]

    for step in range(n_target_tokens):
        nxt = _sample_next_token(next_logits, temperature, top_p, generator).unsqueeze(-1)
        preds.append(nxt)
        if step + 1 < n_target_tokens:
            if past_kvs is not None and cur_len + step + 1 < max_len:
                out = base_lm(input_ids=nxt, past_kvs=past_kvs, use_cache=True)
                past_kvs = out.past_kvs
                next_logits = out.logits[:, -1, :]
            else:
                prefix_embeds = torch.cat([prefix_embeds, _embed_tokens(base_lm, nxt)], dim=1)[
                    :, -max_len:, :
                ]
                out = base_lm(inputs_embeds=prefix_embeds, use_cache=False)
                next_logits = out.logits[:, -1, :]

    return torch.cat(preds, dim=1) if preds else torch.empty((prefix_embeds.shape[0], 0), dtype=torch.long)


def _estimate_flops(model_config: Any, seq_len: int, extra_steps: float) -> float:
    """Call `model-eng`'s `estimate_inference_flops_per_token` (supports float extra_steps)."""
    from phase1_pilot.models.llama import estimate_inference_flops_per_token

    return float(
        estimate_inference_flops_per_token(
            model_config, seq_len=max(1, int(seq_len)), extra_steps=float(extra_steps)
        )
    )


def _count_params(model: nn.Module) -> tuple[int, int]:
    """Return `(total_params, non_embedding_params)` via `count_parameters_breakdown`."""
    from phase1_pilot.models.llama import count_parameters_breakdown

    bd = count_parameters_breakdown(model)
    return int(bd["total"]), int(bd["non_embedding"])


# ---------------------------------------------------------------------------
# Diagnostic 1 — FLOPs-Matched Pareto Evaluator
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_pareto_point(
    model: nn.Module,
    samples: Sequence[Any],
    tokenizer: Any,
    arm_name: str,
    use_thought: bool = False,
    thought_len: int = 8,
    device: torch.device | str | None = None,
) -> ParetoPoint:
    """Diagnostic 1: Exact accuracy, Hard-BPB, and inference FLOPs/token."""
    model.eval()
    dev = _infer_device(model, device)
    base_lm = _get_base_lm(model)
    cfg = _get_model_config(model)
    variant = _get_variant_name(model)
    total_params, non_emb_params = _count_params(model)

    n_exact = 0
    n_tok_correct = 0
    n_tok_total = 0
    total_nll = 0.0
    total_bytes = 0
    total_seq_len = 0
    total_extra_steps = 0
    depth_hits: dict[int, list[int]] = {}
    n_eval = 0

    for sample in samples:
        p_ids, t_ids, depth, target_str = _extract_sample_fields(sample, tokenizer)
        if not p_ids or not t_ids:
            continue
        prompt_t = torch.tensor([p_ids], dtype=torch.long, device=dev)
        target_t = torch.tensor([t_ids], dtype=torch.long, device=dev)

        prefix_embeds, extra_steps = _build_prefix_embeds_for_model(
            model,
            prompt_t,
            tokenizer,
            use_thought=use_thought,
            thought_len=thought_len,
            temperature=0.0,
        )
        logits = _forward_target_logits_with_prefix_embeds(base_lm, prefix_embeds, target_t)
        # Teacher-forced NLL and token accuracy
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            target_t.reshape(-1),
            reduction="sum",
        ).item()
        tok_preds = logits.argmax(dim=-1)[0]
        tok_corr = int((tok_preds == target_t[0]).sum().item())

        # Greedy autoregressive exact match on target span
        if len(t_ids) == 1:
            gen_ids = [int(tok_preds[0].item())]
        else:
            gen = _decode_target_from_prefix_embeds(
                base_lm, prefix_embeds, n_target_tokens=len(t_ids), temperature=0.0
            )
            gen_ids = _strip_special_for_match(gen[0].tolist(), tokenizer)

        exact = int(gen_ids == t_ids)
        n_exact += exact
        n_tok_correct += tok_corr
        n_tok_total += len(t_ids)
        total_nll += float(nll)
        n_bytes = len(target_str.encode("utf-8")) if target_str else len(t_ids)
        total_bytes += max(1, n_bytes)
        total_seq_len += len(p_ids) + extra_steps + len(t_ids)
        total_extra_steps += extra_steps
        depth_hits.setdefault(depth, []).append(exact)
        n_eval += 1

    mean_nll = total_nll / max(1, n_tok_total)
    hard_bpb = (total_nll / math.log(2.0)) / max(1, total_bytes)
    mean_seq_len = int(round(total_seq_len / max(1, n_eval))) if n_eval else 64
    extra_per_target = float(total_extra_steps / max(1, n_tok_total))
    flops_per_tok = _estimate_flops(cfg, seq_len=max(1, mean_seq_len), extra_steps=extra_per_target)

    acc_by_depth = {
        int(d): float(sum(hits) / len(hits)) for d, hits in sorted(depth_hits.items()) if hits
    }

    return ParetoPoint(
        arm_name=arm_name,
        variant=variant if not use_thought else f"{variant}+discrete_thought",
        total_params=total_params,
        non_embedding_params=non_emb_params,
        exact_accuracy=float(n_exact / max(1, n_eval)),
        token_accuracy=float(n_tok_correct / max(1, n_tok_total)),
        mean_nll=float(mean_nll),
        hard_bpb=float(hard_bpb),
        inference_flops_per_token=float(flops_per_tok),
        extra_steps_per_target=extra_per_target,
        accuracy_by_depth=acc_by_depth,
        n_samples=n_eval,
    )


# ---------------------------------------------------------------------------
# Diagnostic 2 — Support-Boundary Probe (pass@k)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_pass_at_k(
    model: nn.Module,
    samples: Sequence[Any],
    tokenizer: Any,
    arm_name: str,
    ks: Sequence[int] = (1, 4, 16, 64),
    n_samples: int = 16,
    temperature: float = 0.8,
    top_p: float = 0.95,
    use_thought: bool = False,
    thought_len: int = 8,
    ood_depths: Sequence[int] = (6, 8),
    device: torch.device | str | None = None,
    seed: int = 42,
) -> PassAtKResult:
    """Diagnostic 2: Unbiased pass@k across hop depths to detect support expansion vs sharpening."""
    model.eval()
    dev = _infer_device(model, device)
    base_lm = _get_base_lm(model)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    counts_by_depth: dict[int, list[tuple[int, int]]] = {}
    for sample in samples:
        p_ids, t_ids, depth, _ = _extract_sample_fields(sample, tokenizer)
        if not p_ids or not t_ids:
            continue
        prompt_t = torch.tensor([p_ids], dtype=torch.long, device=dev)
        prompt_batch = prompt_t.expand(n_samples, -1)

        if use_thought:
            # Draw `n_samples` stochastic thoughts + stochastic answer rollouts in one batch
            prefix_embeds, _ = _build_prefix_embeds_for_model(
                model,
                prompt_batch,
                tokenizer,
                use_thought=True,
                thought_len=thought_len,
                temperature=temperature,
                top_p=top_p,
                generator=gen,
            )
        else:
            # Deterministic prefix (B1/B2/B3/B4) shared across `n_samples` stochastic answer samples
            prefix_single, _ = _build_prefix_embeds_for_model(
                model,
                prompt_t,
                tokenizer,
                use_thought=False,
                thought_len=0,
                temperature=0.0,
            )
            prefix_embeds = prefix_single.expand(n_samples, -1, -1)

        decoded = _decode_target_from_prefix_embeds(
            base_lm,
            prefix_embeds,
            n_target_tokens=len(t_ids),
            temperature=temperature,
            top_p=top_p,
            generator=gen,
        )
        c = 0
        for row in decoded.tolist():
            if _strip_special_for_match(row, tokenizer) == t_ids:
                c += 1
        counts_by_depth.setdefault(depth, []).append((n_samples, c))

    return compute_pass_at_k_from_counts(
        counts_by_depth=counts_by_depth,
        ks=ks,
        ood_depths=ood_depths,
        arm_name=arm_name,
    )


# ---------------------------------------------------------------------------
# Diagnostic 3 — Causal Thought Faithfulness
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_thought_faithfulness(
    model: nn.Module,
    samples: Sequence[Any],
    tokenizer: Any,
    arm_name: str,
    mode: str = "auto",
    thought_len: int = 8,
    device: torch.device | str | None = None,
    seed: int = 42,
    nll_evaluator: Callable[[list[int], Tensor, list[int]], float] | None = None,
    thought_provider: Callable[[Any], Tensor] | None = None,
) -> ThoughtFaithfulnessResult:
    """Diagnostic 3: Swap-sensitivity, format-matched truncation/perturbation, OOD depth scaling.

    Args:
        model: Evaluated model (`LlamaLM`, `PauseTokenWrapper`, or `ContinuousLatentWrapper`).
        samples: Evaluation samples spanning multiple hop depths (e.g. d in {2, 4, 6, 8}).
        tokenizer: PilotTokenizer instance.
        arm_name: Human-readable arm label.
        mode: `"auto"`, `"discrete"`, `"continuous"`, `"pause"`, or `"ntp"`.
        thought_len: Internal thought length for discrete rollouts.
        device: Computation device.
        seed: Deterministic RNG seed for format-matched corruption / orthogonal perturbation.
        nll_evaluator: Optional pure callable `(prompt_ids, thought_tensor, target_ids) -> nll`
            for unit testing custom decoders without constructing a full Transformer.
        thought_provider: Optional callable `(sample) -> thought_tensor` paired with `nll_evaluator`.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Pure hook path (primarily for analytical unit tests)
    if nll_evaluator is not None and thought_provider is not None:
        resolved_mode = mode if mode != "auto" else "discrete"
        parsed: list[tuple[list[int], list[int], int, Tensor]] = []
        for s in samples:
            p_ids, t_ids, depth, _ = _extract_sample_fields(s, tokenizer)
            if not t_ids:
                continue
            parsed.append((p_ids, t_ids, depth, thought_provider(s)))
        if not parsed:
            return compute_faithfulness_summary([], arm_name=arm_name, mode=resolved_mode)

        records: list[dict[str, float | int]] = []
        n = len(parsed)
        think_start = int(getattr(tokenizer, "think_start_id", 3))
        think_end = int(getattr(tokenizer, "think_end_id", 4))
        vocab_size = int(getattr(tokenizer, "vocab_size", 512))
        for i, (p_ids, t_ids, depth, z_i) in enumerate(parsed):
            z_swap = parsed[(i + 1) % n][3] if n > 1 else z_i
            if resolved_mode == "discrete":
                # Match length if needed while keeping delimiters
                if z_swap.numel() != z_i.numel():
                    z_swap = z_i.clone()
                z_trunc = corrupt_discrete_thought_format_matched(
                    z_i, think_start, think_end, vocab_size, generator=gen
                )
            else:
                z_trunc = perturb_continuous_latents_norm_matched(z_i, generator=gen)
            nll_orig = float(nll_evaluator(p_ids, z_i, t_ids))
            nll_sw = float(nll_evaluator(p_ids, z_swap, t_ids))
            nll_tr = float(nll_evaluator(p_ids, z_trunc, t_ids))
            records.append(
                {
                    "depth": depth,
                    "nll_thought": nll_orig,
                    "nll_swap": nll_sw,
                    "nll_trunc": nll_tr,
                }
            )
        return compute_faithfulness_summary(records, arm_name=arm_name, mode=resolved_mode)

    model.eval()
    dev = _infer_device(model, device)
    base_lm = _get_base_lm(model)
    variant = _get_variant_name(model)
    resolved_mode = mode
    if resolved_mode == "auto":
        if variant == "continuous_latent":
            resolved_mode = "continuous"
        elif variant == "hybrid_latent":
            resolved_mode = "hybrid" if thought_len > 0 else "continuous"
        elif variant == "pause":
            resolved_mode = "pause"
        else:
            resolved_mode = "discrete"
    elif variant == "hybrid_latent" and resolved_mode == "discrete":
        # Align hybrid_latent with its trained rollout_hybrid interface (no <think>/</think>)
        resolved_mode = "hybrid"

    if resolved_mode in ("ntp", "pause"):
        # No sample-specific thought: swap and truncation deltas are identically 0.0
        records_zero: list[dict[str, float | int]] = []
        for s in samples:
            p_ids, t_ids, depth, _ = _extract_sample_fields(s, tokenizer)
            if not p_ids or not t_ids:
                continue
            prompt_t = torch.tensor([p_ids], dtype=torch.long, device=dev)
            target_t = torch.tensor([t_ids], dtype=torch.long, device=dev)
            prefix_embeds, _ = _build_prefix_embeds_for_model(
                model, prompt_t, tokenizer, use_thought=False
            )
            logits = _forward_target_logits_with_prefix_embeds(base_lm, prefix_embeds, target_t)
            nll = float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    target_t.reshape(-1),
                    reduction="mean",
                ).item()
            )
            records_zero.append(
                {
                    "depth": depth,
                    "nll_thought": nll,
                    "nll_swap": nll,
                    "nll_trunc": nll,
                    "acc_thought": 0.0,
                    "acc_nothought": 0.0,
                }
            )
        return compute_faithfulness_summary(records_zero, arm_name=arm_name, mode=resolved_mode)

    think_start_id = int(getattr(tokenizer, "think_start_id", 3))
    think_end_id = int(getattr(tokenizer, "think_end_id", 4))
    vocab_size = int(getattr(base_lm.config, "vocab_size", getattr(tokenizer, "vocab_size", 512)))

    # Pass 1: Collect prompts, targets, and model-generated thoughts/latents
    items: list[dict[str, Any]] = []
    for s in samples:
        p_ids, t_ids, depth, _ = _extract_sample_fields(s, tokenizer)
        if not p_ids or not t_ids:
            continue
        prompt_t = torch.tensor([p_ids], dtype=torch.long, device=dev)
        target_t = torch.tensor([t_ids], dtype=torch.long, device=dev)
        prompt_embeds = _embed_tokens(base_lm, prompt_t)

        # No-thought baseline accuracy (for accuracy lift by depth)
        logits_no = _forward_target_logits_with_prefix_embeds(base_lm, prompt_embeds, target_t)
        acc_no = float(int((logits_no.argmax(dim=-1)[0] == target_t[0]).all().item()))

        if resolved_mode == "continuous":
            _, latent_embeds = _compute_continuous_latent_embeds(model, prompt_t)  # [1, L, D]
            items.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "target_t": target_t,
                    "depth": depth,
                    "thought": latent_embeds.detach(),
                    "acc_nothought": acc_no,
                }
            )
        elif resolved_mode == "hybrid":
            # Match rollout_hybrid / recompute_hybrid_logprobs_and_continuation directly
            _, blended_embeds = _generate_hybrid_thought_and_embeds(
                model, prompt_t, thought_len=thought_len, temperature=0.0
            )  # [1, L, D]
            items.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "target_t": target_t,
                    "depth": depth,
                    "thought": blended_embeds.detach(),
                    "acc_nothought": acc_no,
                }
            )
        else:
            thought_ids = _generate_discrete_thought(
                model,
                prompt_t,
                tokenizer,
                thought_len=thought_len,
                temperature=0.0,
            )[0]  # [L + 2]
            items.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "target_t": target_t,
                    "depth": depth,
                    "thought": thought_ids.detach(),
                    "acc_nothought": acc_no,
                }
            )

    if not items:
        return compute_faithfulness_summary([], arm_name=arm_name, mode=resolved_mode)

    # Pass 2: Evaluate original, length-matched swapped, and format-matched truncated/perturbed
    records_out: list[dict[str, float | int]] = []
    n_items = len(items)
    for i, item in enumerate(items):
        # Choose a mismatched prompt j != i (cyclic shift across dataset)
        j = (i + max(1, n_items // 2)) % n_items if n_items > 1 else i
        if j == i and n_items > 1:
            j = (i + 1) % n_items

        prompt_embeds = item["prompt_embeds"]
        target_t = item["target_t"]
        thought_orig = item["thought"]
        thought_swap = items[j]["thought"]

        if resolved_mode in ("continuous", "hybrid"):
            thought_trunc = perturb_continuous_latents_norm_matched(thought_orig, generator=gen)
            emb_orig = torch.cat([prompt_embeds, thought_orig], dim=1)
            emb_swap = torch.cat([prompt_embeds, thought_swap], dim=1)
            emb_trunc = torch.cat([prompt_embeds, thought_trunc], dim=1)
        else:
            # Ensure length match for swap while preserving delimiters
            if thought_swap.numel() != thought_orig.numel():
                target_len = int(thought_orig.numel())
                if thought_swap.numel() > target_len:
                    inner = thought_swap[1 : target_len - 1]
                else:
                    pad_needed = target_len - int(thought_swap.numel())
                    filler = thought_swap[1:-1] if thought_swap.numel() > 2 else thought_swap[:1]
                    inner = torch.cat([thought_swap[1:-1], filler.repeat((pad_needed // max(1, filler.numel())) + 1)[:pad_needed]])
                thought_swap = torch.cat(
                    [
                        thought_orig[:1],
                        inner[: max(0, target_len - 2)],
                        thought_orig[-1:],
                    ]
                )
            thought_trunc = corrupt_discrete_thought_format_matched(
                thought_orig,
                think_start_id=think_start_id,
                think_end_id=think_end_id,
                vocab_size=vocab_size,
                generator=gen,
            )
            emb_orig = torch.cat([prompt_embeds, _embed_tokens(base_lm, thought_orig.unsqueeze(0))], dim=1)
            emb_swap = torch.cat([prompt_embeds, _embed_tokens(base_lm, thought_swap.unsqueeze(0))], dim=1)
            emb_trunc = torch.cat([prompt_embeds, _embed_tokens(base_lm, thought_trunc.unsqueeze(0))], dim=1)

        logits_orig = _forward_target_logits_with_prefix_embeds(base_lm, emb_orig, target_t)
        logits_swap = _forward_target_logits_with_prefix_embeds(base_lm, emb_swap, target_t)
        logits_trunc = _forward_target_logits_with_prefix_embeds(base_lm, emb_trunc, target_t)

        nll_orig = float(
            F.cross_entropy(
                logits_orig.reshape(-1, logits_orig.shape[-1]).float(),
                target_t.reshape(-1),
                reduction="mean",
            ).item()
        )
        nll_sw = float(
            F.cross_entropy(
                logits_swap.reshape(-1, logits_swap.shape[-1]).float(),
                target_t.reshape(-1),
                reduction="mean",
            ).item()
        )
        nll_tr = float(
            F.cross_entropy(
                logits_trunc.reshape(-1, logits_trunc.shape[-1]).float(),
                target_t.reshape(-1),
                reduction="mean",
            ).item()
        )
        acc_th = float(int((logits_orig.argmax(dim=-1)[0] == target_t[0]).all().item()))

        records_out.append(
            {
                "depth": item["depth"],
                "nll_thought": nll_orig,
                "nll_swap": nll_sw,
                "nll_trunc": nll_tr,
                "acc_thought": acc_th,
                "acc_nothought": item["acc_nothought"],
            }
        )

    return compute_faithfulness_summary(records_out, arm_name=arm_name, mode=resolved_mode)


# ---------------------------------------------------------------------------
# Diagnostic 4 — Distributional Calibration
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_calibration(
    model: nn.Module,
    samples: Sequence[Any],
    tokenizer: Any,
    arm_name: str,
    regime: str = "regime3_aleatoric",
    n_bins: int = 10,
    use_thought: bool = False,
    thought_len: int = 8,
    device: torch.device | str | None = None,
) -> CalibrationResult:
    """Diagnostic 4: 10-bin ECE and proper token Brier score on Regimes 1, 2, or 3."""
    model.eval()
    dev = _infer_device(model, device)
    base_lm = _get_base_lm(model)

    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []

    for sample in samples:
        p_ids, t_ids, _, _ = _extract_sample_fields(sample, tokenizer)
        if not p_ids or not t_ids:
            continue
        prompt_t = torch.tensor([p_ids], dtype=torch.long, device=dev)
        target_t = torch.tensor([t_ids], dtype=torch.long, device=dev)

        prefix_embeds, _ = _build_prefix_embeds_for_model(
            model,
            prompt_t,
            tokenizer,
            use_thought=use_thought,
            thought_len=thought_len,
            temperature=0.0,
        )
        logits = _forward_target_logits_with_prefix_embeds(base_lm, prefix_embeds, target_t)
        all_logits.append(logits[0].cpu())
        all_targets.append(target_t[0].cpu())

    if not all_logits:
        return CalibrationResult(
            arm_name=arm_name,
            regime=regime,
            ece=0.0,
            mean_brier_score=0.0,
            mean_brier_loss=0.0,
            mean_confidence=0.0,
            accuracy=0.0,
            overconfidence_gap=0.0,
            bin_stats=[],
            n_tokens=0,
        )

    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    stats = compute_ece_and_brier(cat_logits, cat_targets, n_bins=n_bins)
    return CalibrationResult(
        arm_name=arm_name,
        regime=regime,
        ece=stats["ece"],
        mean_brier_score=stats["mean_brier_score"],
        mean_brier_loss=stats["mean_brier_loss"],
        mean_confidence=stats["mean_confidence"],
        accuracy=stats["accuracy"],
        overconfidence_gap=stats["overconfidence_gap"],
        bin_stats=stats["bin_stats"],
        n_tokens=stats["n_tokens"],
    )


# ---------------------------------------------------------------------------
# Diagnostic 5 — Track 1 §5.5 Falsification Verdict Engine
# ---------------------------------------------------------------------------


def _evaluate_single_arm_criteria(
    arm: ArmEvalBundle,
    b1: ArmEvalBundle,
    b2: ArmEvalBundle,
    b3: ArmEvalBundle,
    swap_threshold: float,
    b3_margin: float,
    ece_tolerance: float,
    drift_tolerance: float,
) -> dict[str, bool]:
    """Evaluate all 8 Track 1 §5.5 confirmation criteria for one candidate arm."""
    b2_acc_parity = arm.pareto.exact_accuracy >= b2.pareto.exact_accuracy - 1e-9
    b2_flops_ok = arm.pareto.inference_flops_per_token <= (1.0 + 1e-9) * b2.pareto.inference_flops_per_token
    beats_b3_margin = (arm.pareto.exact_accuracy - b3.pareto.exact_accuracy) >= b3_margin - 1e-9
    beats_b3_strict = arm.pareto.exact_accuracy > b3.pareto.exact_accuracy + 1e-9

    if arm.faithfulness is not None:
        swap_pass = arm.faithfulness.delta_nll_swap > swap_threshold
        trunc_pass = arm.faithfulness.delta_nll_trunc > 0.0
        mono_depth = bool(arm.faithfulness.monotonic_depth_scaling)
    else:
        swap_pass = False
        trunc_pass = False
        mono_depth = False

    if arm.pass_at_k is not None and b1.pass_at_k is not None:
        ood_pass_expansion = arm.pass_at_k.ood_pass_at_max_k > b1.pass_at_k.ood_pass_at_max_k + 1e-9
    else:
        ood_pass_expansion = False

    # Calibration stability (Regime 3 aleatoric if available, else primary) + no-thought drift
    cal_arm = arm.calibration_aleatoric or arm.calibration_primary
    cal_b1 = b1.calibration_aleatoric or b1.calibration_primary
    if cal_arm is not None and cal_b1 is not None:
        ece_ok = cal_arm.ece <= cal_b1.ece + ece_tolerance
    else:
        ece_ok = True
    drift_ok = (arm.nothought_nll_drift is None) or (arm.nothought_nll_drift <= drift_tolerance)
    calibration_stable = bool(ece_ok and drift_ok)

    return {
        "teacher_free": bool(arm.is_teacher_free),
        "b2_accuracy_parity": bool(b2_acc_parity),
        "b2_flops_matched": bool(b2_flops_ok),
        "beats_b3_by_3pp": bool(beats_b3_margin),
        "beats_b3_strict": bool(beats_b3_strict),
        "swap_sensitivity_gt_0_15": bool(swap_pass),
        "trunc_faithfulness_gt_0": bool(trunc_pass),
        "monotonic_ood_depth_scaling": bool(mono_depth),
        "ood_pass_at_k_expansion": bool(ood_pass_expansion),
        "calibration_stable": bool(calibration_stable),
    }


def _format_markdown_table(
    all_arms: Sequence[ArmEvalBundle],
    b2_flops: float,
) -> str:
    """Render a crisp GitHub-flavored Markdown comparison table across all arms."""
    header = (
        "| Arm | Params | FLOPs/tok (vs B2) | Exact Acc | Hard-BPB | OOD pass@max_k "
        "| ΔNLL_swap | ΔNLL_trunc | Mono Depth | Reg3 ECE | Reg3 Brier |"
    )
    sep = (
        "| :--- | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | :---: | ---: | ---: |"
    )
    rows = [header, sep]
    for arm in all_arms:
        p = arm.pareto
        flops_ratio = p.inference_flops_per_token / max(b2_flops, 1.0)
        ood_pk = (
            f"{arm.pass_at_k.ood_pass_at_max_k * 100:.1f}%"
            if arm.pass_at_k is not None
            else "—"
        )
        if arm.faithfulness is not None and arm.faithfulness.mode not in ("ntp", "pause"):
            d_swap = f"{arm.faithfulness.delta_nll_swap:+.3f}"
            d_trunc = f"{arm.faithfulness.delta_nll_trunc:+.3f}"
            mono = "Yes" if arm.faithfulness.monotonic_depth_scaling else "No"
        else:
            d_swap, d_trunc, mono = "—", "—", "—"

        cal = arm.calibration_aleatoric or arm.calibration_primary
        ece_str = f"{cal.ece:.3f}" if cal is not None else "—"
        brier_str = f"{cal.mean_brier_score:.3f}" if cal is not None else "—"
        rows.append(
            f"| `{arm.arm_name}` | {p.total_params:,} | {flops_ratio:.2f}× | "
            f"{p.exact_accuracy * 100:.1f}% | {p.hard_bpb:.3f} | {ood_pk} | "
            f"{d_swap} | {d_trunc} | {mono} | {ece_str} | {brier_str} |"
        )
    return "\n".join(rows)


def compute_pilot_verdict(
    b1: ArmEvalBundle,
    b2: ArmEvalBundle,
    b3: ArmEvalBundle,
    candidate_arms: Sequence[ArmEvalBundle],
    b4: ArmEvalBundle | None = None,
    swap_threshold: float = 0.15,
    b3_margin: float = 0.03,
    ece_tolerance: float = 0.05,
    drift_tolerance: float = 0.15,
) -> PilotVerdict:
    """Diagnostic 5: Evaluate Track 1 §5.5 Confirmation / Partial / Falsification criteria.

    Decision logic (exact fidelity to `01-ssl-as-rl-latent-cot.md` §5.5):
    - **CONFIRMED (Strong Form):** At least one strictly teacher-free latent-CoT RL arm
      (`T2-self` + `R2`/`R3` or `I2`) satisfies ALL criteria:
      (1) `exact_accuracy >= B2`, (2) `inference_flops_per_token <= 1.0 * B2`,
      (3) `exact_accuracy >= B3 + 3 pp`, (4) `delta_nll_swap > 0.15`,
      (5) `delta_nll_trunc > 0.0`, (6) monotonic OOD depth scaling,
      (7) strictly higher OOD `pass@max_k` than `B1`, and (8) stable calibration.
    - **PARTIAL_CONFIRMATION:** Strong confirmation is not met, BUT at least one latent
      or RL arm (e.g., `B4` continuous latent, `I2` hybrid, warm-start RL, teacher-filtered
      `T2-scale`, or an RL arm that sharpens `pass@1` without expanding `pass@k`) beats
      `B3` AND achieves FLOPs-matched `B2` parity (or continuous/hybrid/warm-start beats
      `B3` by >= 3 pp with causal faithfulness while cold-start discrete RL fails).
    - **FALSIFIED:** No small-model latent-CoT / RL variant beats BOTH `B3` (pause control)
      and FLOPs-matched `B2` (2.5x larger NTP baseline).
    """
    eval_pool: list[ArmEvalBundle] = list(candidate_arms)
    if b4 is not None and all(a.arm_name != b4.arm_name for a in eval_pool):
        eval_pool.append(b4)

    arm_checks: dict[str, dict[str, bool]] = {}
    strong_keys = (
        "teacher_free",
        "b2_accuracy_parity",
        "b2_flops_matched",
        "beats_b3_by_3pp",
        "swap_sensitivity_gt_0_15",
        "trunc_faithfulness_gt_0",
        "monotonic_ood_depth_scaling",
        "ood_pass_at_k_expansion",
        "calibration_stable",
    )

    confirmed_arms: list[ArmEvalBundle] = []
    partial_arms: list[ArmEvalBundle] = []

    for arm in eval_pool:
        chk = _evaluate_single_arm_criteria(
            arm=arm,
            b1=b1,
            b2=b2,
            b3=b3,
            swap_threshold=swap_threshold,
            b3_margin=b3_margin,
            ece_tolerance=ece_tolerance,
            drift_tolerance=drift_tolerance,
        )
        arm_checks[arm.arm_name] = chk

        if arm.is_rl_arm and all(chk[k] for k in strong_keys):
            confirmed_arms.append(arm)
        else:
            # Partial confirmation if beats B3 AND matches FLOPs-matched B2,
            # OR conditional arm (B4/I2/warm-start/teacher-filtered) beats B3 by >=3pp
            # with genuine causal faithfulness and matched FLOPs.
            beats_b3_and_b2 = (
                chk["beats_b3_strict"]
                and chk["b2_accuracy_parity"]
                and chk["b2_flops_matched"]
            )
            conditional_causal_win = (
                (arm.is_continuous_or_hybrid or arm.is_warm_start or not arm.is_teacher_free)
                and chk["beats_b3_by_3pp"]
                and chk["b2_flops_matched"]
                and chk["swap_sensitivity_gt_0_15"]
                and chk["trunc_faithfulness_gt_0"]
            )
            if beats_b3_and_b2 or conditional_causal_win:
                partial_arms.append(arm)

    # Pick best arm for headline reporting
    def _rank_key(a: ArmEvalBundle) -> tuple[int, float]:
        score = sum(1 for k in strong_keys if arm_checks[a.arm_name].get(k, False))
        return (score, a.pareto.exact_accuracy)

    if confirmed_arms:
        best = max(confirmed_arms, key=_rank_key)
        status = "CONFIRMED"
        rationale = (
            f"Strong Confirmation: teacher-free RL arm `{best.arm_name}` matched/exceeded "
            f"B2 ({best.pareto.exact_accuracy*100:.1f}% vs {b2.pareto.exact_accuracy*100:.1f}%) "
            f"at {best.pareto.inference_flops_per_token / max(b2.pareto.inference_flops_per_token, 1.0):.2f}x B2 FLOPs, "
            f"beat pause control B3 ({b3.pareto.exact_accuracy*100:.1f}%) by "
            f"{(best.pareto.exact_accuracy - b3.pareto.exact_accuracy)*100:+.1f} pp, "
            f"passed causal thought faithfulness (ΔNLL_swap={best.faithfulness.delta_nll_swap:+.3f}, "
            f"ΔNLL_trunc={best.faithfulness.delta_nll_trunc:+.3f}, monotonic OOD depth scaling), "
            f"expanded OOD pass@k over B1, and maintained stable calibration."
        )
    elif partial_arms:
        best = max(partial_arms, key=_rank_key)
        status = "PARTIAL_CONFIRMATION"
        failed = [k for k in strong_keys if not arm_checks[best.arm_name].get(k, False)]
        rationale = (
            f"Partial Confirmation: arm `{best.arm_name}` achieved conditional gains "
            f"(acc={best.pareto.exact_accuracy*100:.1f}% vs B3={b3.pareto.exact_accuracy*100:.1f}%, "
            f"B2={b2.pareto.exact_accuracy*100:.1f}%), but strong-form teacher-free RL confirmation "
            f"was not met (unmet criteria on `{best.arm_name}`: {', '.join(failed)})."
        )
    else:
        best = max(eval_pool, key=_rank_key) if eval_pool else None
        status = "FALSIFIED"
        best_desc = (
            f"Best candidate `{best.arm_name}` reached {best.pareto.exact_accuracy*100:.1f}% "
            f"at {best.pareto.inference_flops_per_token / max(b2.pareto.inference_flops_per_token, 1.0):.2f}x B2 FLOPs"
            if best is not None
            else "No candidate arms evaluated"
        )
        rationale = (
            f"Falsified: no small-model latent-CoT or RL arm simultaneously beat pause control "
            f"B3 ({b3.pareto.exact_accuracy*100:.1f}%) and FLOPs-matched B2 "
            f"({b2.pareto.exact_accuracy*100:.1f}%). {best_desc}."
        )

    headline_checks = (
        dict(arm_checks[best.arm_name])
        if best is not None
        else {k: False for k in strong_keys}
    )

    all_for_table: list[ArmEvalBundle] = [b1, b2, b3]
    if b4 is not None:
        all_for_table.append(b4)
    for a in candidate_arms:
        if all(existing.arm_name != a.arm_name for existing in all_for_table):
            all_for_table.append(a)

    md_table = _format_markdown_table(all_for_table, b2_flops=b2.pareto.inference_flops_per_token)
    summary = {
        "b1_exact_accuracy": b1.pareto.exact_accuracy,
        "b2_exact_accuracy": b2.pareto.exact_accuracy,
        "b2_flops_per_token": b2.pareto.inference_flops_per_token,
        "b3_exact_accuracy": b3.pareto.exact_accuracy,
        "best_arm": best.arm_name if best is not None else None,
        "best_arm_exact_accuracy": best.pareto.exact_accuracy if best is not None else None,
        "best_arm_flops_ratio_vs_b2": (
            best.pareto.inference_flops_per_token / max(b2.pareto.inference_flops_per_token, 1.0)
            if best is not None
            else None
        ),
    }

    return PilotVerdict(
        status=status,
        best_rl_arm=best.arm_name if best is not None else None,
        checks=headline_checks,
        arm_checks=arm_checks,
        metrics_summary=summary,
        rationale=rationale,
        markdown_table=md_table,
    )
