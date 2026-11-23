"""Data package: tokenizer and 3-regime stratified dataset generators."""

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

__all__ = [
    "HFTokenizerAdapter",
    "PilotDataset",
    "PilotTokenizer",
    "Sample",
    "TokenizedSample",
    "build_stratified_datasets",
    "collate_batch",
    "generate_aleatoric_cloze",
    "generate_dag_planning",
    "generate_modular_varbind",
    "generate_natural_stream",
    "generate_python_trace",
    "load_h100_web_mixture",
    "tokenize_sample",
]
