"""Compare Hub tokenizers for Stage 1 (`BL-11`): compression, preset sizes, answer tokenization.

For each candidate, pinned to its current Hub commit:
- bytes per token on the first `--n-docs` documents of each Regime 2 source (pinned revisions),
- `160m` / `410m` / `1b` parameter counts at the adapter's vocabulary (protocol tokens included),
- the fraction of Regime 1 and Regime 3 targets that encode as one token, and mean sample length,
- how a few probe strings split (digits, indentation).

Needs the Hub and `transformers`, `tokenizers`, `datasets`:

    uv run --with "transformers==5.24.0" --with "tokenizers==0.23.3" --with datasets \
        python scripts/compare_tokenizers.py --output runs/tokenizers.json \
        --pin-from results/bl11_tokenizer_comparison.json   # replay the committed comparison
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from phase1_pilot.config import ModelConfig
from phase1_pilot.data.generators import (
    generate_aleatoric_cloze,
    generate_dag_planning,
    generate_modular_varbind,
    generate_python_trace,
)
from phase1_pilot.data.records import Sample
from phase1_pilot.models.llama import estimate_parameter_count

CANDIDATES = (
    "HuggingFaceTB/SmolLM2-135M",
    "bigcode/starcoder2-3b",
    "EleutherAI/pythia-160m",
    "openai-community/gpt2",
)
SOURCES = (  # label, dataset, config, text field: the Stage 1 Regime 2 mixture (BL-11)
    ("openwebmath", "open-web-math/open-web-math", None, "text"),
    ("codeparrot", "codeparrot/codeparrot-clean", None, "content"),
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
)
PRESETS = ("160m", "410m", "1b")
PROBES = ("123456789", "x = 4096", "        return x", "v3 = v7 ; v7 = 12", "3.14159", "  \n\n    def f():")


def bytes_per_token(tokenizer: Any, docs: Iterable[str]) -> float:
    docs = list(docs)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    return n_bytes / sum(len(tokenizer.encode_corpus(d)) for d in docs)


def target_stats(tokenizer: Any, samples: Sequence[Sample]) -> dict[str, dict[str, float]]:
    """Per task: share of targets that are one token, and mean tokenized sample length."""
    single, length = defaultdict(list), defaultdict(list)
    for s in samples:
        t = tokenizer.encode(s.target)
        single[s.task].append(len(t) == 1)
        length[s.task].append(len(tokenizer.encode(s.prompt)) + len(t) + 2)  # + <bos>, <eos>
    return {
        task: {"single_token_target_frac": round(statistics.mean(single[task]), 3),
               "mean_sample_tokens": round(statistics.mean(length[task]), 1)}
        for task in sorted(single)
    }


def preset_params(vocab_size: int) -> dict[str, int]:
    return {p: estimate_parameter_count(ModelConfig.preset(p, vocab_size=vocab_size))["total"] for p in PRESETS}


def synthetic_samples(seed: int = 1, n: int = 300) -> list[Sample]:
    depths = (2, 4, 6, 8)
    return [
        *generate_dag_planning(n, depths=depths, seed=seed),
        *generate_python_trace(n, depths=depths, seed=seed),
        *generate_modular_varbind(n, depths=depths, seed=seed),
        *generate_aleatoric_cloze(n, seed=seed),
    ]


def pins_from_report(report: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The tokenizer and dataset commits a previous report used, to replay it exactly."""
    return {
        "tokenizers": {name: t["revision"] for name, t in report["tokenizers"].items()},
        "sources": {label: src["revision"] for label, src in report["sources"].items()},
    }


def sample_documents(n_docs: int, max_chars: int, revisions: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """First `n_docs` documents of each source, at `revisions[label]` or the current Hub head."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api, out = HfApi(), {}
    for label, name, config, field in SOURCES:
        revision = (revisions or {}).get(label) or api.dataset_info(name).sha
        rows = load_dataset(name, config, split="train", streaming=True, revision=revision)
        docs = [row[field][:max_chars] for row in itertools.islice(rows, n_docs)]
        out[label] = {"dataset": name, "config": config, "revision": revision, "docs": docs}
    return out


def describe(
    name: str, docs: dict[str, dict[str, Any]], samples: Sequence[Sample], revision: str | None = None
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    from phase1_pilot.data.tokenizer import build_tokenizer

    revision = revision or HfApi().model_info(name).sha
    tok = build_tokenizer(name, revision=revision)
    params = preset_params(tok.vocab_size)
    return {
        "revision": revision,
        "adapter_vocab_size": tok.vocab_size,
        "params": params,
        "ratios_vs_160m": {p: round(params[p] / params["160m"], 3) for p in PRESETS[1:]},
        "bytes_per_token": {label: round(bytes_per_token(tok, d["docs"]), 3) for label, d in docs.items()},
        "targets": target_stats(tok, samples),
        "probes": {s: tok._tok.convert_ids_to_tokens(tok.encode(s)) for s in PROBES},
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True)
    p.add_argument("--n-docs", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=20_000, help="truncate each document to this many characters")
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    p.add_argument("--pin-from", default=None,
                   help="earlier report JSON whose tokenizer and dataset commits to reuse (default: current Hub heads)")
    args = p.parse_args(argv)

    from phase1_pilot.data.tokenizer import PilotTokenizer

    pins = {"tokenizers": {}, "sources": {}}
    if args.pin_from:
        with open(args.pin_from, encoding="utf-8") as f:
            pins = pins_from_report(json.load(f))
    docs = sample_documents(args.n_docs, args.max_chars, pins["sources"])
    samples = synthetic_samples()
    report = {
        "config": vars(args),
        "sources": {k: {kk: v[kk] for kk in ("dataset", "config", "revision")} | {"n_docs": len(v["docs"])}
                    for k, v in docs.items()},
        "pilot_targets": target_stats(PilotTokenizer(), samples),
        "tokenizers": {name: describe(name, docs, samples, pins["tokenizers"].get(name)) for name in args.candidates},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
