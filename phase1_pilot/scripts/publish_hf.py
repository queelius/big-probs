"""Reusable HuggingFace Hub publisher for Phase 1 Latent-CoT RL models & datasets.

Publishes trained checkpoints, telemetry, evaluation reports, and deterministic
3-regime stratified datasets to HuggingFace Hub using ``huggingface_hub``
(and ``pyarrow`` when available for HuggingFace Dataset Viewer Parquet splits).

Designed to publish both the CPU micro-scale proof-of-life run (`micro-pilot-v1`)
and future large H100 Stage 1 (`160m` / `410m` / `1b`) runs seamlessly via `--tag`
and `--run-dir`.

Examples:
    # Publish CPU micro-pilot models + stratified datasets (with Parquet):
    uv run --with huggingface_hub --with pyarrow python scripts/publish_hf.py

    # Publish a future H100 Stage 1 run under a new tag (models only):
    uv run --with huggingface_hub python scripts/publish_hf.py \\
        --tag h100-stage1 \\
        --run-dir runs/h100_stage1 \\
        --report results/h100_stage1_report.md \\
        --metrics results/h100_stage1_metrics.json \\
        --skip-datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GITHUB_REPO_URL = "https://github.com/queelius/big-probs"
SPLIT_ORDER: tuple[str, ...] = (
    "train",
    "eval_id",
    "eval_ood",
    "eval_natural",
    "eval_aleatoric",
)


def _resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() or p.exists():
        return p.resolve()
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate.resolve()
    return p.resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.2f} GB"


def _normalize_arm_key(name: str) -> str:
    return name.lower().replace("-", "_").strip()


def _infer_variant(
    arm_dir_name: str,
    ckpt: dict[str, Any] | None,
    arm_metrics: dict[str, Any] | None,
) -> str:
    from phase1_pilot.config import VALID_VARIANTS

    if arm_metrics and isinstance(arm_metrics.get("pareto"), dict):
        raw_variant = str(arm_metrics["pareto"].get("variant", "")).split("+", 1)[0].strip()
        if raw_variant in VALID_VARIANTS:
            return raw_variant
    if ckpt and isinstance(ckpt.get("rl_config"), dict):
        iface = ckpt["rl_config"].get("interface")
        if iface == "I2_hybrid":
            return "hybrid_latent"
    if ckpt and isinstance(ckpt.get("model_state_dict"), dict):
        keys = ckpt["model_state_dict"].keys()
        if any("pause" in k for k in keys):
            return "pause"
        if any("gate" in k for k in keys) and any("latent" in k for k in keys):
            return "hybrid_latent"
        if any("latent" in k for k in keys):
            return "continuous_latent"
    lowered = arm_dir_name.lower()
    if "hybrid" in lowered or lowered.startswith("i2"):
        return "hybrid_latent"
    if "pause" in lowered or lowered == "b3":
        return "pause"
    if "continuous" in lowered or lowered == "b4":
        return "continuous_latent"
    return "ntp"


def _build_arm_config_json(
    arm_dir: Path,
    arm_metrics: dict[str, Any] | None,
    global_run_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    """Synthesize a self-describing config.json for an arm directory."""
    import torch

    from phase1_pilot.config import ModelConfig

    ckpt_path = arm_dir / "checkpoint.pt"
    ckpt: dict[str, Any] | None = None
    total_params: int | None = None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), dict):
            total_params = sum(int(v.numel()) for v in ckpt["model_state_dict"].values())

    train_cfg = ckpt.get("train_config", {}) if isinstance(ckpt, dict) else {}
    rl_cfg = ckpt.get("rl_config") if isinstance(ckpt, dict) else None
    preset = str(train_cfg.get("preset", "micro-small"))
    variant = _infer_variant(arm_dir.name, ckpt, arm_metrics)

    max_seq_len = int((global_run_cfg or {}).get("max_seq_len", 192 if preset.startswith("micro") else 1024))
    n_latent_steps = int((global_run_cfg or {}).get("n_latent_steps", 4))
    thought_len = int(
        (rl_cfg or {}).get("thought_len", (global_run_cfg or {}).get("thought_len", 8))
    )

    model_overrides: dict[str, Any] = {
        "variant": variant,
        "max_seq_len": max_seq_len,
        "n_pause_tokens": n_latent_steps,
        "n_latent_steps": n_latent_steps,
        "max_thought_len": thought_len,
    }
    if preset.startswith("micro"):
        model_overrides["vocab_size"] = 512

    saved_model_cfg = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    if isinstance(saved_model_cfg, dict):
        # `train.py` records the exact model it built. The reconstruction below is only
        # for checkpoints written before it did (and for `run_micro_pilot.py` arms).
        model_cfg_dict = dict(saved_model_cfg)
    else:
        try:
            model_cfg_dict = ModelConfig.preset(preset, **model_overrides).to_dict()
        except ValueError:
            model_cfg_dict = {"preset": preset, **model_overrides}

    arm_display_name = (
        arm_metrics.get("arm_name")
        if isinstance(arm_metrics, dict) and arm_metrics.get("arm_name")
        else arm_dir.name
    )

    config_payload: dict[str, Any] = {
        "arm_name": arm_display_name,
        "arm_dir": arm_dir.name,
        "preset": preset,
        "variant": variant,
        "total_params": total_params,
        "model_config": model_cfg_dict,
        "train_config": train_cfg,
        "steps": ckpt.get("steps") if isinstance(ckpt, dict) else None,
        "final_val_loss": ckpt.get("final_val_loss") if isinstance(ckpt, dict) else None,
    }
    if isinstance(ckpt, dict) and "final_train_loss" in ckpt:
        config_payload["final_train_loss"] = ckpt["final_train_loss"]
    if isinstance(ckpt, dict) and "data_protocol" in ckpt:
        config_payload["data_protocol"] = ckpt["data_protocol"]
    if rl_cfg is not None:
        config_payload["rl_config"] = rl_cfg
    if isinstance(ckpt, dict) and "final_metrics" in ckpt:
        config_payload["final_rl_metrics"] = ckpt["final_metrics"]
    return config_payload


def _render_model_card(
    *,
    models_repo_id: str,
    datasets_repo_id: str,
    tag: str,
    arm_configs: list[dict[str, Any]],
    report_text: str | None,
    metrics_data: dict[str, Any] | None,
) -> str:
    """Build a crisp HuggingFace Model Card (README.md) with YAML frontmatter."""
    verdict = (metrics_data or {}).get("verdict", {})
    scope = (metrics_data or {}).get(
        "scope",
        f"Phase 1 Small-Model Latent-CoT RL Pilot checkpoints ({tag})",
    )
    run_cfg = (metrics_data or {}).get("config", {})
    summary_table = verdict.get("markdown_table", "")
    verdict_status = verdict.get("status")
    verdict_rationale = verdict.get("rationale")

    lines: list[str] = [
        "---",
        "license: mit",
        "language:",
        "  - en",
        "library_name: pytorch",
        "tags:",
        "  - reinforcement-learning",
        "  - rlvr",
        "  - latent-cot",
        "  - grpo",
        "  - dapo",
        "  - self-supervised-learning",
        "  - pytorch",
        "datasets:",
        f"  - {datasets_repo_id}",
        "---",
        "",
        "# Big-Probs Phase 1: Small-Model Latent-CoT RL Pilot Checkpoints",
        "",
        f"Trained PyTorch checkpoints, training telemetry, and evaluation artifacts for the "
        f"[Big-Probs Phase 1 Latent-CoT RL Pilot]({GITHUB_REPO_URL}), testing whether "
        f"teacher-free self-supervised RL (Dr. GRPO + DAPO Clip-Higher + frozen-anchor proper "
        f"scoring rewards) and continuous/hybrid latent thought interfaces enable small "
        f"decoder-only Transformers to outperform pause-token controls and FLOPs-matched baselines.",
        "",
        f"- **Code & Methodology:** [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})",
        f"- **Stratified Datasets:** [https://huggingface.co/datasets/{datasets_repo_id}](https://huggingface.co/datasets/{datasets_repo_id})",
        f"- **Latest Published Tag:** `{tag}`",
        "",
        f"> **Scope Notice:** {scope}.",
        "",
        "## Published Arms (`" + tag + "/`)",
        "",
        "| Directory | Arm | Preset | Variant | Parameters | Steps | Interface / Reward |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | :--- |",
    ]

    for ac in arm_configs:
        rl_cfg = ac.get("rl_config")
        if isinstance(rl_cfg, dict):
            iface_str = f"`{rl_cfg.get('interface')}` / `{rl_cfg.get('reward_type')}` (`{rl_cfg.get('filter_type')}`)"
        else:
            iface_str = "Supervised NTP"
        params_str = f"{ac['total_params']:,}" if ac.get("total_params") is not None else "—"
        steps_str = str(ac["steps"]) if ac.get("steps") is not None else "—"
        lines.append(
            f"| `{tag}/{ac['arm_dir']}/` | `{ac['arm_name']}` | `{ac['preset']}` | "
            f"`{ac['variant']}` | {params_str} | {steps_str} | {iface_str} |"
        )

    lines.extend(["", "## Evaluation Summary"])
    if verdict_status:
        lines.append("")
        lines.append(f"- **Track 1 §5.5 Falsification Verdict (`{tag}`):** `{verdict_status}`")
        if verdict_rationale:
            lines.append(f"- **Rationale:** {verdict_rationale}")
    if run_cfg:
        lines.append(
            f"- **Run Config:** `seed={run_cfg.get('seed')}`, "
            f"`n_train={run_cfg.get('n_train')}`, "
            f"`sup_steps={run_cfg.get('sup_steps')}`, "
            f"`rl_steps={run_cfg.get('rl_steps')}`, "
            f"`group_size={run_cfg.get('group_size')}`, "
            f"`thought_len={run_cfg.get('thought_len')}`, "
            f"`n_ahead={run_cfg.get('n_ahead')}`"
        )

    if summary_table:
        lines.extend(["", "### Summary Comparison Table (All 7 Arms)", "", summary_table])

    if report_text:
        # Include Diagnostics 1-5 verbatim from the report (everything after the first ---)
        body_sections = report_text.split("\n---\n", 1)
        if len(body_sections) == 2:
            # Drop duplicate section 1 header if summary table already shown, or include sections 2-5
            rest = body_sections[1]
            sec2_idx = rest.find("## 2.")
            if sec2_idx != -1:
                lines.extend(["", rest[sec2_idx:].strip()])

    lines.extend(
        [
            "",
            "## Repository Structure",
            "",
            "```text",
            f"{tag}/",
        ]
    )
    for ac in arm_configs:
        lines.append(f"  {ac['arm_dir']}/")
        lines.append("    checkpoint.pt      # PyTorch state_dict + train_config (+ rl_config)")
        lines.append("    config.json        # ModelConfig + TrainConfig (+ RLConfig) metadata")
        lines.append("    metrics.jsonl      # Step-by-step training/validation telemetry")
    lines.extend(
        [
            "  micro_pilot_report.md   # Full 5-diagnostic Markdown evaluation report",
            "  micro_pilot_metrics.json # Machine-readable diagnostic metrics payload",
            "  manifest.json           # SHA256 checksums and file sizes",
            "```",
            "",
            "Future H100 Stage 1 runs (`160m` / `410m` / `1b`) are published under sibling tags "
            "(e.g., `h100-stage1/`) using the same `scripts/publish_hf.py` publisher without "
            "overwriting prior tags.",
            "",
            "## Loading a Checkpoint in PyTorch",
            "",
            "```python",
            "import json",
            "import torch",
            "from huggingface_hub import hf_hub_download",
            "from phase1_pilot.config import ModelConfig",
            "from phase1_pilot.data import build_tokenizer",
            "from phase1_pilot.models import build_model",
            "",
            f'REPO_ID = "{models_repo_id}"',
            f'ARM_PATH = "{tag}/i2_hybrid"  # or b1, b2, b3, b4, rl_r1, rl_r3',
            "",
            'cfg_file = hf_hub_download(repo_id=REPO_ID, filename=f"{ARM_PATH}/config.json")',
            'ckpt_file = hf_hub_download(repo_id=REPO_ID, filename=f"{ARM_PATH}/checkpoint.pt")',
            "",
            'meta = json.loads(open(cfg_file, encoding="utf-8").read())',
            'model_cfg = ModelConfig(**meta["model_config"])',
            'tokenizer = build_tokenizer(meta.get("data_protocol", {}).get("tokenizer", "pilot"))',
            'kwargs = {"pause_token_id": tokenizer.pause_id} if model_cfg.variant == "pause" else {}',
            "model = build_model(model_cfg, **kwargs)",
            "",
            'ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)',
            'model.load_state_dict(ckpt["model_state_dict"], strict=True)',
            "model.eval()",
            "```",
            "",
            "## License",
            "",
            "Released under the MIT License. See the [Big-Probs repository](https://github.com/queelius/big-probs) for full project documentation.",
            "",
        ]
    )
    return "\n".join(lines)


def _export_dataset_config(
    *,
    config_name: str,
    data_cfg: Any,
    tokenizer: Any,
    staging_root: Path,
    use_parquet: bool,
) -> dict[str, Any]:
    """Generate deterministic stratified splits, verify zero leakage, and write Parquet + JSONL."""
    from phase1_pilot.data import build_stratified_datasets

    datasets = build_stratified_datasets(data_cfg, tokenizer)

    # Strict zero-leakage assertions across train vs eval splits
    train_alg_prompts = {
        s.prompt for s in datasets["train"].samples if s.regime == "algorithmic"
    }
    train_alg_depths = {
        s.depth for s in datasets["train"].samples if s.regime == "algorithmic"
    }
    train_nat_prompts = {
        s.prompt for s in datasets["train"].samples if s.regime == "natural"
    }
    train_alea_prompts = {
        s.prompt for s in datasets["train"].samples if s.regime == "aleatoric"
    }

    eval_id_prompts = {s.prompt for s in datasets["eval_id"].samples}
    eval_ood_depths = {s.depth for s in datasets["eval_ood"].samples}
    eval_nat_prompts = {s.prompt for s in datasets["eval_natural"].samples}
    eval_alea_prompts = {s.prompt for s in datasets["eval_aleatoric"].samples}

    if train_alg_prompts & eval_id_prompts:
        raise RuntimeError(f"[{config_name}] Leakage detected between train and eval_id!")
    if train_alg_depths & eval_ood_depths:
        raise RuntimeError(f"[{config_name}] Depth overlap between train and eval_ood!")
    if train_nat_prompts & eval_nat_prompts:
        raise RuntimeError(f"[{config_name}] Leakage detected between train and eval_natural!")
    if train_alea_prompts & eval_alea_prompts:
        raise RuntimeError(f"[{config_name}] Leakage detected between train and eval_aleatoric!")

    pa = None
    pq = None
    if use_parquet:
        import pyarrow as pa  # type: ignore[no-redef]
        import pyarrow.parquet as pq  # type: ignore[no-redef]

    data_dir = staging_root / "data" / config_name
    jsonl_dir = staging_root / "jsonl" / config_name
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    splits_summary: dict[str, Any] = {}
    for split_name in SPLIT_ORDER:
        ds = datasets[split_name]
        rows: list[dict[str, Any]] = []
        regime_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        depth_counts: Counter[int] = Counter()

        for idx, (sample, tok) in enumerate(zip(ds.samples, ds.tokenized)):
            regime_counts[sample.regime] += 1
            task_counts[sample.task] += 1
            depth_counts[sample.depth] += 1
            rows.append(
                {
                    "id": f"{config_name}_{split_name}_{idx:06d}",
                    "prompt": sample.prompt,
                    "target": sample.target,
                    "full_text": sample.full_text,
                    "regime": sample.regime,
                    "task": sample.task,
                    "depth": int(sample.depth),
                    "answer_span_start": int(sample.answer_span_start),
                    "prompt_len_tokens": int(tok.prompt_len),
                    "target_len_tokens": int(len(tok.target_ids)),
                    "seq_len_tokens": int(len(tok.input_ids)),
                    "input_ids": [int(x) for x in tok.input_ids],
                    "target_ids": [int(x) for x in tok.target_ids],
                }
            )

        # Always write clean JSONL mirror
        jsonl_path = jsonl_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if use_parquet and pa is not None and pq is not None:
            schema = pa.schema(
                [
                    ("id", pa.string()),
                    ("prompt", pa.string()),
                    ("target", pa.string()),
                    ("full_text", pa.string()),
                    ("regime", pa.string()),
                    ("task", pa.string()),
                    ("depth", pa.int32()),
                    ("answer_span_start", pa.int32()),
                    ("prompt_len_tokens", pa.int32()),
                    ("target_len_tokens", pa.int32()),
                    ("seq_len_tokens", pa.int32()),
                    ("input_ids", pa.list_(pa.int32())),
                    ("target_ids", pa.list_(pa.int32())),
                ]
            )
            table = pa.Table.from_pylist(rows, schema=schema)
            primary_path = data_dir / f"{split_name}.parquet"
            pq.write_table(table, primary_path, compression="snappy")
            primary_rel = f"data/{config_name}/{split_name}.parquet"
        else:
            primary_path = data_dir / f"{split_name}.jsonl"
            shutil.copy2(jsonl_path, primary_path)
            primary_rel = f"data/{config_name}/{split_name}.jsonl"

        splits_summary[split_name] = {
            "num_rows": len(rows),
            "primary_file": primary_rel,
            "primary_bytes": primary_path.stat().st_size,
            "primary_sha256": _sha256_file(primary_path),
            "jsonl_file": f"jsonl/{config_name}/{split_name}.jsonl",
            "jsonl_bytes": jsonl_path.stat().st_size,
            "regimes": dict(sorted(regime_counts.items())),
            "tasks": dict(sorted(task_counts.items())),
            "depths": {str(k): v for k, v in sorted(depth_counts.items())},
        }

    return {
        "config_name": config_name,
        "data_config": data_cfg.to_dict(),
        "total_rows": sum(s["num_rows"] for s in splits_summary.values()),
        "zero_leakage_verified": True,
        "splits": splits_summary,
    }


def _render_dataset_card(
    *,
    datasets_repo_id: str,
    models_repo_id: str,
    configs_meta: list[dict[str, Any]],
    use_parquet: bool,
) -> str:
    """Build a HuggingFace Dataset Card with YAML configs/data_files for Dataset Viewer."""
    total_rows_all = sum(c["total_rows"] for c in configs_meta)
    size_cat = "1K<n<10K" if total_rows_all < 10_000 else "10K<n<100K"
    ext = "parquet" if use_parquet else "jsonl"

    lines: list[str] = [
        "---",
        "license: mit",
        "language:",
        "  - en",
        "task_categories:",
        "  - text-generation",
        "pretty_name: Big-Probs Phase 1 Stratified Latent-CoT RL Datasets",
        "tags:",
        "  - synthetic",
        "  - algorithmic-reasoning",
        "  - reinforcement-learning",
        "  - rlvr",
        "  - latent-cot",
        "  - calibration",
        "  - aleatoric-uncertainty",
        "size_categories:",
        f"  - {size_cat}",
        "configs:",
    ]

    for idx, cfg in enumerate(configs_meta):
        cname = cfg["config_name"]
        lines.append(f"  - config_name: {cname}")
        if idx == 0:
            lines.append("    default: true")
        lines.append("    data_files:")
        for split_name in SPLIT_ORDER:
            rel_path = cfg["splits"][split_name]["primary_file"]
            lines.append(f"      - split: {split_name}")
            lines.append(f"        path: {rel_path}")

    lines.extend(
        [
            "---",
            "",
            "# Big-Probs Phase 1 Stratified Latent-CoT RL Datasets",
            "",
            f"Deterministic 3-regime stratified datasets for the "
            f"[Big-Probs Phase 1 Small-Model Latent-CoT RL Pilot]({GITHUB_REPO_URL}), designed to "
            f"probe multi-hop compositional reasoning, OOD depth generalization, and distributional "
            f"calibration under irreducible aleatoric uncertainty.",
            "",
            f"- **Code & Generators:** [{GITHUB_REPO_URL}]({GITHUB_REPO_URL}) (`phase1_pilot/data/generators.py`)",
            f"- **Trained Checkpoints:** [https://huggingface.co/{models_repo_id}](https://huggingface.co/{models_repo_id})",
            f"- **Format:** Primary splits stored as `{ext.upper()}` under `data/<config>/` (with JSONL mirrors under `jsonl/<config>/`).",
            "",
            "## Three-Regime Design (Track 1 §5.2)",
            "",
            "1. **Regime 1 — Algorithmic & Compositional (`algorithmic`):**",
            "   - `dag_planning`: ProsQA-style directed acyclic graph reachability queries with one true chain of exact shortest-path hop depth `d`, two distractor chains, and a branching decoy twig (verified by exact BFS).",
            "   - `python_trace`: Deterministic stateful Python register programs (`a, b, c` modulo 17) with `d` sequential arithmetic/conditional transitions (verified by isolated execution).",
            "   - `modular_varbind`: Shuffled multi-step modular arithmetic variable-binding chains (`v0..v15` modulo 19) of exact dependency depth `d`.",
            "   - **Depth Stratification:** In-distribution (ID) `d in {2, 4}` for `train` and `eval_id`; strictly disjoint out-of-distribution (OOD) `d in {6, 8}` for `eval_ood`.",
            "2. **Regime 2 — Structured Natural / Code Stream (`natural`):**",
            "   - `natural_code_math`: Multi-token structured math-prose derivations, Python function evaluations, and arithmetic sequences suitable for non-myopic `n_ahead=4` RL rewards.",
            "   - *(Note: For full H100 Stage 1 runs, `load_h100_web_mixture()` streams 40% OpenWebMath / 40% The Stack v2 / 20% FineWeb-Edu.)*",
            "3. **Regime 3 — Aleatoric Negative Control (`aleatoric`):**",
            "   - `aleatoric_cloze`: Unique entity-attribute cloze queries assigned i.i.d. uniform targets across 16 classes (`C00..C15`), giving mathematically irreducible conditional entropy of `log2(16) = 4.0` bits to detect RL-induced overconfidence.",
            "",
            "## Zero Train / Eval Leakage Guarantee",
            "",
            "Every configuration is deterministically generated and verified at export time to guarantee:",
            "",
            "- **Zero prompt overlap** between `train` (Regime 1) and `eval_id`.",
            "- **Strictly disjoint hop depths** between `train` (`d in {2, 4}`) and `eval_ood` (`d in {6, 8}`).",
            "- **Zero prompt overlap** between `train` (Regime 2) and `eval_natural`.",
            "- **Strictly disjoint entity namespaces** (`tr` vs `ev`) between `train` (Regime 3) and `eval_aleatoric`.",
            "",
            "## Configurations & Exact Split Row Counts",
            "",
        ]
    )

    for cfg in configs_meta:
        cname = cfg["config_name"]
        dcfg = cfg["data_config"]
        lines.append(
            f"### `{cname}` (`seed={dcfg['seed']}`, `max_seq_len={dcfg['max_seq_len']}`, "
            f"`train_depths={tuple(dcfg['train_depths'])}`, `eval_ood_depths={tuple(dcfg['eval_ood_depths'])}`)"
        )
        lines.append("")
        lines.append(
            "| Split | Rows | Regimes | Tasks | Depths | File Size |"
        )
        lines.append("| :--- | ---: | :--- | :--- | :--- | ---: |")
        for split_name in SPLIT_ORDER:
            s = cfg["splits"][split_name]
            reg_str = ", ".join(f"{k}: {v}" for k, v in s["regimes"].items())
            task_str = ", ".join(f"{k}: {v}" for k, v in s["tasks"].items())
            depth_str = ", ".join(f"d={k}: {v}" for k, v in s["depths"].items())
            lines.append(
                f"| `{split_name}` | {s['num_rows']:,} | {reg_str} | {task_str} | "
                f"{depth_str} | {_format_bytes(s['primary_bytes'])} |"
            )
        lines.append(f"\nTotal rows in `{cname}`: **{cfg['total_rows']:,}**.\n")

    lines.extend(
        [
            "## Schema",
            "",
            "| Column | Type | Description |",
            "| :--- | :--- | :--- |",
            "| `id` | `string` | Deterministic record identifier (`<config>_<split>_<index>`) |",
            "| `prompt` | `string` | Input prompt terminating in `<ans>` |",
            "| `target` | `string` | Ground-truth answer span |",
            "| `full_text` | `string` | Concatenated `prompt + target` |",
            "| `regime` | `string` | `algorithmic`, `natural`, or `aleatoric` |",
            "| `task` | `string` | `dag_planning`, `python_trace`, `modular_varbind`, `natural_code_math`, or `aleatoric_cloze` |",
            "| `depth` | `int32` | Exact compositional hop depth (`2, 4, 6, 8` for Regime 1; `0` for Regimes 2 & 3) |",
            "| `answer_span_start` | `int32` | Character offset where `target` begins in `full_text` |",
            "| `prompt_len_tokens` | `int32` | Prompt token length under `PilotTokenizer` (including `<bos>`) |",
            "| `target_len_tokens` | `int32` | Target token length under `PilotTokenizer` |",
            "| `seq_len_tokens` | `int32` | Total tokenized length (`<bos> + prompt + target + <eos>`) |",
            "| `input_ids` | `list<int32>` | Exact lossless `PilotTokenizer` (`vocab_size=512`) token IDs |",
            "| `target_ids` | `list<int32>` | Exact `PilotTokenizer` target token IDs |",
            "",
            "## Usage",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            f'# Load exact micro-pilot splits (default config: "{configs_meta[0]["config_name"]}")',
            f'ds = load_dataset("{datasets_repo_id}")',
            'print(ds["eval_ood"][0])',
            "",
            "# Or load the larger canonical Stage 1 benchmark config:",
            f'ds_bench = load_dataset("{datasets_repo_id}", "stage1_benchmark")',
            "```",
            "",
            "## License",
            "",
            "Released under the MIT License.",
            "",
        ]
    )
    return "\n".join(lines)


def publish_models(
    *,
    api: Any,
    models_repo_id: str,
    datasets_repo_id: str,
    run_dir: Path,
    report_path: Path | None,
    metrics_path: Path | None,
    tag: str,
    private: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Stage and upload all arm directories, report, metrics, and Model Card."""
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    arm_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    if not arm_dirs:
        raise RuntimeError(f"No trained arm subdirectories found under {run_dir}")

    metrics_data: dict[str, Any] | None = None
    if metrics_path and metrics_path.exists():
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    report_text: str | None = None
    if report_path and report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")

    # Map normalized arm name -> metrics bundle
    arm_metrics_lookup: dict[str, dict[str, Any]] = {}
    if metrics_data and isinstance(metrics_data.get("arms"), dict):
        for k, v in metrics_data["arms"].items():
            arm_metrics_lookup[_normalize_arm_key(k)] = v

    global_run_cfg = (metrics_data or {}).get("config", {})

    with tempfile.TemporaryDirectory(prefix="big_probs_hf_models_") as tmp:
        stage_root = Path(tmp)
        tag_stage = stage_root / tag
        tag_stage.mkdir(parents=True, exist_ok=True)

        arm_configs: list[dict[str, Any]] = []
        manifest_files: list[dict[str, Any]] = []

        for arm_dir in arm_dirs:
            dst_arm = tag_stage / arm_dir.name
            dst_arm.mkdir(parents=True, exist_ok=True)
            for item in sorted(arm_dir.iterdir()):
                if item.is_file():
                    shutil.copy2(item, dst_arm / item.name)

            arm_metrics = arm_metrics_lookup.get(_normalize_arm_key(arm_dir.name))
            cfg_dst = dst_arm / "config.json"
            if cfg_dst.exists():
                arm_cfg = json.loads(cfg_dst.read_text(encoding="utf-8"))
            else:
                arm_cfg = _build_arm_config_json(arm_dir, arm_metrics, global_run_cfg)
                cfg_dst.write_text(json.dumps(arm_cfg, indent=2) + "\n", encoding="utf-8")
            arm_configs.append(arm_cfg)

        if report_path and report_path.exists():
            shutil.copy2(report_path, tag_stage / report_path.name)
        if metrics_path and metrics_path.exists():
            shutil.copy2(metrics_path, tag_stage / metrics_path.name)

        # Build per-tag manifest
        for fpath in sorted(tag_stage.rglob("*")):
            if fpath.is_file():
                rel = fpath.relative_to(stage_root).as_posix()
                manifest_files.append(
                    {
                        "path": rel,
                        "bytes": fpath.stat().st_size,
                        "sha256": _sha256_file(fpath),
                    }
                )
        manifest = {
            "tag": tag,
            "repo_id": models_repo_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "num_arms": len(arm_configs),
            "arms": [a["arm_dir"] for a in arm_configs],
            "total_bytes": sum(f["bytes"] for f in manifest_files),
            "files": manifest_files,
        }
        (tag_stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        # Generate Model Card README.md at repo root
        readme_text = _render_model_card(
            models_repo_id=models_repo_id,
            datasets_repo_id=datasets_repo_id,
            tag=tag,
            arm_configs=arm_configs,
            report_text=report_text,
            metrics_data=metrics_data,
        )
        readme_path = stage_root / "README.md"
        readme_path.write_text(readme_text, encoding="utf-8")

        if dry_run:
            staged_files = [
                p.relative_to(stage_root).as_posix()
                for p in sorted(stage_root.rglob("*"))
                if p.is_file()
            ]
            print(f"[dry-run] Would upload {len(staged_files)} files to model repo {models_repo_id}")
            return {"repo_id": models_repo_id, "dry_run": True, "files": staged_files}

        api.create_repo(
            repo_id=models_repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(tag_stage),
            path_in_repo=tag,
            repo_id=models_repo_id,
            repo_type="model",
            commit_message=f"Publish Phase 1 pilot models ({tag}: {len(arm_configs)} arms)",
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=models_repo_id,
            repo_type="model",
            commit_message=f"Update Model Card for {tag}",
        )

        remote_files = api.list_repo_files(repo_id=models_repo_id, repo_type="model")
        return {
            "repo_id": models_repo_id,
            "tag": tag,
            "num_arms": len(arm_configs),
            "uploaded_bytes": manifest["total_bytes"],
            "remote_files": remote_files,
        }


def publish_datasets(
    *,
    api: Any,
    datasets_repo_id: str,
    models_repo_id: str,
    metrics_path: Path | None,
    seed: int,
    n_train: int | None,
    n_eval: int | None,
    max_seq_len: int | None,
    benchmark_n_train: int,
    benchmark_n_eval: int,
    include_benchmark: bool,
    private: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Export deterministic 3-regime datasets and upload to HuggingFace Hub."""
    from phase1_pilot.config import DataConfig
    from phase1_pilot.data import PilotTokenizer

    # Detect pyarrow for Parquet support (ideal for HF Dataset Viewer)
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401

        use_parquet = True
    except ImportError:
        use_parquet = False

    # Align micro_pilot config with micro_pilot_metrics.json when available
    run_cfg: dict[str, Any] = {}
    if metrics_path and metrics_path.exists():
        run_cfg = json.loads(metrics_path.read_text(encoding="utf-8")).get("config", {})

    mp_seed = int(run_cfg.get("seed", seed))
    mp_n_train = int(n_train if n_train is not None else run_cfg.get("n_train", 600))
    if n_eval is not None:
        mp_n_eval = int(n_eval)
    else:
        eval_per_depth = int(run_cfg.get("eval_per_depth", 16))
        eval_aleatoric = int(run_cfg.get("eval_aleatoric", 48))
        mp_n_eval = max(eval_per_depth * 3, eval_aleatoric)
    mp_max_seq_len = int(
        max_seq_len if max_seq_len is not None else run_cfg.get("max_seq_len", 192)
    )

    tokenizer = PilotTokenizer()

    export_specs: list[tuple[str, DataConfig]] = [
        (
            "micro_pilot",
            DataConfig(
                seed=mp_seed,
                train_depths=(2, 4),
                eval_ood_depths=(6, 8),
                n_train_samples=mp_n_train,
                n_eval_samples=mp_n_eval,
                max_seq_len=mp_max_seq_len,
            ),
        ),
    ]
    if include_benchmark:
        export_specs.append(
            (
                "stage1_benchmark",
                DataConfig(
                    seed=mp_seed,
                    train_depths=(2, 4),
                    eval_ood_depths=(6, 8),
                    n_train_samples=benchmark_n_train,
                    n_eval_samples=benchmark_n_eval,
                    max_seq_len=256,
                ),
            )
        )

    with tempfile.TemporaryDirectory(prefix="big_probs_hf_data_") as tmp:
        stage_root = Path(tmp)
        configs_meta: list[dict[str, Any]] = []

        for config_name, dcfg in export_specs:
            meta = _export_dataset_config(
                config_name=config_name,
                data_cfg=dcfg,
                tokenizer=tokenizer,
                staging_root=stage_root,
                use_parquet=use_parquet,
            )
            configs_meta.append(meta)

        manifest = {
            "repo_id": datasets_repo_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "format": "parquet+jsonl" if use_parquet else "jsonl",
            "tokenizer": {
                "class": "PilotTokenizer",
                "vocab_size": tokenizer.vocab_size,
            },
            "configs": configs_meta,
        }
        (stage_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        readme_text = _render_dataset_card(
            datasets_repo_id=datasets_repo_id,
            models_repo_id=models_repo_id,
            configs_meta=configs_meta,
            use_parquet=use_parquet,
        )
        (stage_root / "README.md").write_text(readme_text, encoding="utf-8")

        if dry_run:
            staged_files = [
                p.relative_to(stage_root).as_posix()
                for p in sorted(stage_root.rglob("*"))
                if p.is_file()
            ]
            print(
                f"[dry-run] Would upload {len(staged_files)} files to dataset repo {datasets_repo_id}"
            )
            return {
                "repo_id": datasets_repo_id,
                "dry_run": True,
                "configs": configs_meta,
                "files": staged_files,
            }

        api.create_repo(
            repo_id=datasets_repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(stage_root),
            repo_id=datasets_repo_id,
            repo_type="dataset",
            commit_message=(
                f"Publish Phase 1 stratified datasets "
                f"({', '.join(c['config_name'] for c in configs_meta)}; "
                f"{'Parquet + JSONL' if use_parquet else 'JSONL'})"
            ),
        )

        remote_files = api.list_repo_files(repo_id=datasets_repo_id, repo_type="dataset")
        return {
            "repo_id": datasets_repo_id,
            "format": "parquet+jsonl" if use_parquet else "jsonl",
            "configs": configs_meta,
            "remote_files": remote_files,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish Big-Probs Phase 1 pilot models and stratified datasets to HuggingFace Hub."
    )
    p.add_argument("--namespace", type=str, default="queelius", help="HuggingFace account/org namespace.")
    p.add_argument("--models-repo", type=str, default="big-probs-phase1-models", help="Model repository name.")
    p.add_argument("--datasets-repo", type=str, default="big-probs-phase1-data", help="Dataset repository name.")
    p.add_argument("--run-dir", type=str, default="results/micro_runs", help="Directory containing trained arm subdirs.")
    p.add_argument("--report", type=str, default="results/micro_pilot_report.md", help="Markdown evaluation report path.")
    p.add_argument("--metrics", type=str, default="results/micro_pilot_metrics.json", help="JSON metrics payload path.")
    p.add_argument("--tag", type=str, default="micro-pilot-v1", help="Subdirectory tag in model repo (e.g. h100-stage1).")
    p.add_argument("--skip-models", action="store_true", help="Skip model upload.")
    p.add_argument("--skip-datasets", action="store_true", help="Skip dataset export and upload.")
    p.add_argument("--private", action="store_true", help="Create private HF repos (default: public).")
    p.add_argument("--seed", type=int, default=42, help="Dataset generation seed (default: 42).")
    p.add_argument("--n-train", type=int, default=None, help="Override micro_pilot n_train_samples.")
    p.add_argument("--n-eval", type=int, default=None, help="Override micro_pilot n_eval_samples.")
    p.add_argument("--max-seq-len", type=int, default=None, help="Override micro_pilot max_seq_len.")
    p.add_argument("--benchmark-n-train", type=int, default=2000, help="stage1_benchmark n_train_samples.")
    p.add_argument("--benchmark-n-eval", type=int, default=300, help="stage1_benchmark n_eval_samples.")
    p.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Export only micro_pilot dataset config without stage1_benchmark.",
    )
    p.add_argument("--dry-run", action="store_true", help="Stage artifacts locally without uploading to HF Hub.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.skip_models and args.skip_datasets:
        print("Both --skip-models and --skip-datasets were specified; nothing to do.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Run via: "
            "`uv run --with huggingface_hub --with pyarrow python scripts/publish_hf.py`"
        ) from exc

    api = HfApi()
    if not args.dry_run:
        # Validate auth without ever printing or logging any token.
        who = api.whoami()
        print(f"Authenticated on HuggingFace Hub as: {who.get('name', 'unknown')}")

    models_repo_id = f"{args.namespace}/{args.models_repo}"
    datasets_repo_id = f"{args.namespace}/{args.datasets_repo}"
    run_dir = _resolve_path(args.run_dir)
    report_path = _resolve_path(args.report)
    metrics_path = _resolve_path(args.metrics)

    if not args.skip_models:
        assert run_dir is not None
        print(f"\n[1/2] Publishing models from {run_dir} -> https://huggingface.co/{models_repo_id} (tag={args.tag})")
        m_res = publish_models(
            api=api,
            models_repo_id=models_repo_id,
            datasets_repo_id=datasets_repo_id,
            run_dir=run_dir,
            report_path=report_path,
            metrics_path=metrics_path,
            tag=args.tag,
            private=args.private,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(
                f"  Uploaded {m_res['num_arms']} arms ({_format_bytes(m_res['uploaded_bytes'])}); "
                f"repo now contains {len(m_res['remote_files'])} files."
            )

    if not args.skip_datasets:
        print(f"\n[2/2] Publishing datasets -> https://huggingface.co/datasets/{datasets_repo_id}")
        d_res = publish_datasets(
            api=api,
            datasets_repo_id=datasets_repo_id,
            models_repo_id=models_repo_id,
            metrics_path=metrics_path,
            seed=args.seed,
            n_train=args.n_train,
            n_eval=args.n_eval,
            max_seq_len=args.max_seq_len,
            benchmark_n_train=args.benchmark_n_train,
            benchmark_n_eval=args.benchmark_n_eval,
            include_benchmark=not args.no_benchmark,
            private=args.private,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"  Format: {d_res['format']} | repo now contains {len(d_res['remote_files'])} files.")
            for cfg in d_res["configs"]:
                counts = ", ".join(
                    f"{sp}={cfg['splits'][sp]['num_rows']}" for sp in SPLIT_ORDER
                )
                print(f"  Config `{cfg['config_name']}` ({cfg['total_rows']} rows): {counts}")

    print("\nPublish complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
