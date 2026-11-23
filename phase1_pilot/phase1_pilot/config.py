"""Shared configuration dataclasses for the Phase 1 Latent-CoT RL Pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VALID_VARIANTS = ("ntp", "pause", "continuous_latent", "hybrid_latent")
VALID_REWARDS = ("R1_exact", "R2_log", "R3_brier")
VALID_FILTERS = ("T0_all", "T1_entropy", "T2_self", "T2_scale")
VALID_INTERFACES = ("I1_discrete", "I2_hybrid")


@dataclass
class ModelConfig:
    """Decoder-only Llama-style Transformer & latent-CoT configuration."""

    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 384
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    variant: str = "ntp"
    n_pause_tokens: int = 4
    n_latent_steps: int = 4
    max_thought_len: int = 8

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; expected one of {VALID_VARIANTS}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> ModelConfig:
        """Construct a named model scale preset with optional field overrides."""
        key = name.strip().lower()
        presets: dict[str, dict[str, Any]] = {
            "micro-small": dict(
                vocab_size=512,
                d_model=128,
                n_layers=4,
                n_heads=4,
                n_kv_heads=4,
                d_ff=384,
                max_seq_len=256,
            ),
            "micro-large": dict(
                vocab_size=512,
                d_model=192,
                n_layers=6,
                n_heads=6,
                n_kv_heads=6,
                d_ff=512,
                max_seq_len=256,
            ),
            "160m": dict(
                vocab_size=32000,
                d_model=768,
                n_layers=12,
                n_heads=12,
                n_kv_heads=12,
                d_ff=2048,
                max_seq_len=1024,
            ),
            "410m": dict(
                vocab_size=32000,
                d_model=1024,
                n_layers=24,
                n_heads=16,
                n_kv_heads=16,
                d_ff=2816,
                max_seq_len=1024,
            ),
            "1b": dict(
                vocab_size=32000,
                d_model=1792,
                n_layers=24,
                n_heads=16,
                n_kv_heads=16,
                d_ff=4864,
                max_seq_len=1024,
            ),
        }
        if key not in presets:
            raise ValueError(
                f"Unknown preset {name!r}; expected one of {tuple(presets.keys())}"
            )
        kwargs = {**presets[key], **overrides}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RLConfig:
    """SSL-as-RL (Dr. GRPO + DAPO Clip-Higher + Frozen-Anchor Rewards) configuration."""

    reward_type: str = "R3_brier"
    filter_type: str = "T2_self"
    interface: str = "I1_discrete"
    group_size: int = 8
    n_ahead: int = 4
    thought_len: int = 8
    clip_low: float = 0.2
    clip_high: float = 0.28
    reward_clip_tau: float = 5.0
    lambda_ntp: float = 0.2
    top_p_tokens: float = 0.2
    top_p_thought_entropy: float = 0.2
    lr: float = 3e-4
    ref_ema_decay: float = 1.0
    filter_zero_var: bool = True

    def __post_init__(self) -> None:
        if self.reward_type not in VALID_REWARDS:
            raise ValueError(
                f"Unknown reward_type {self.reward_type!r}; expected one of {VALID_REWARDS}"
            )
        if self.filter_type not in VALID_FILTERS:
            raise ValueError(
                f"Unknown filter_type {self.filter_type!r}; expected one of {VALID_FILTERS}"
            )
        if self.interface not in VALID_INTERFACES:
            raise ValueError(
                f"Unknown interface {self.interface!r}; expected one of {VALID_INTERFACES}"
            )
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {self.group_size}")
        if not (0.0 <= self.ref_ema_decay <= 1.0):
            raise ValueError(
                f"ref_ema_decay must be in [0, 1], got {self.ref_ema_decay}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    """3-regime stratified dataset generation configuration."""

    seed: int = 42
    train_depths: tuple[int, ...] = (2, 4)
    eval_ood_depths: tuple[int, ...] = (6, 8)
    n_train_samples: int = 2000
    n_eval_samples: int = 300
    max_seq_len: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """Supervised / RL training loop configuration."""

    preset: str = "micro-small"
    batch_size: int = 16
    max_steps: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = False
    device: str = "auto"
    log_every: int = 20
    output_dir: str = "runs/default"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
