"""Model architectures, latent reasoning interfaces, and parameter/FLOPs accounting."""

from phase1_pilot.models.latent import (
    ContinuationLogits,
    ContinuousLatentWrapper,
    DiscreteThoughtHelper,
    HybridRolloutOutput,
    PauseTokenWrapper,
    ThoughtRollout,
    build_model,
)
from phase1_pilot.models.llama import (
    GroupedQueryAttention,
    LlamaLM,
    LMOutput,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    count_parameters,
    count_parameters_breakdown,
    default_extra_steps_for_config,
    estimate_inference_flops_per_token,
    estimate_parameter_count,
)

__all__ = [
    "LMOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "TransformerBlock",
    "LlamaLM",
    "count_parameters",
    "count_parameters_breakdown",
    "estimate_parameter_count",
    "estimate_inference_flops_per_token",
    "default_extra_steps_for_config",
    "PauseTokenWrapper",
    "ContinuousLatentWrapper",
    "HybridRolloutOutput",
    "ThoughtRollout",
    "ContinuationLogits",
    "DiscreteThoughtHelper",
    "build_model",
]
