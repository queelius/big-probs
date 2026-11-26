"""Data package: tokenizers, 3-regime synthetic generators, and the real-text corpus path."""

from phase1_pilot.data.corpus import (
    CorpusSplits,
    CorpusStats,
    Document,
    build_corpus_splits,
    corpus_split,
    document_windows,
    read_corpus_jsonl,
)
from phase1_pilot.data.generators import (
    PilotDataset,
    TokenizedDataset,
    build_stratified_datasets,
    collate_batch,
    generate_aleatoric_cloze,
    generate_dag_planning,
    generate_modular_varbind,
    generate_natural_stream,
    generate_python_trace,
    load_h100_web_mixture,
    natural_train_count,
    tokenize_sample,
)
from phase1_pilot.data.records import Sample, TokenizedSample
from phase1_pilot.data.tokenizer import HFTokenizerAdapter, PilotTokenizer, build_tokenizer

__all__ = [
    "CorpusSplits",
    "CorpusStats",
    "Document",
    "HFTokenizerAdapter",
    "PilotDataset",
    "PilotTokenizer",
    "Sample",
    "TokenizedDataset",
    "TokenizedSample",
    "build_corpus_splits",
    "build_stratified_datasets",
    "build_tokenizer",
    "collate_batch",
    "generate_aleatoric_cloze",
    "generate_dag_planning",
    "generate_modular_varbind",
    "generate_natural_stream",
    "generate_python_trace",
    "corpus_split",
    "document_windows",
    "load_h100_web_mixture",
    "natural_train_count",
    "read_corpus_jsonl",
    "tokenize_sample",
]
