"""`train.py` on real-text Regime 2: tokenizer-derived protocol, corpus data, checkpoints."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from phase1_pilot.data.generators import natural_train_count
from phase1_pilot.data.tokenizer import HFTokenizerAdapter, PilotTokenizer, build_tokenizer
from phase1_pilot.train import (
    build_arg_parser,
    build_dataloaders,
    main,
    resolve_model_config,
    run_training,
)


@pytest.fixture
def corpus_path(tmp_path, make_documents) -> Path:
    path = tmp_path / "corpus.jsonl"
    docs = make_documents(80, seed=4)
    path.write_text("".join(json.dumps(dataclasses.asdict(d)) + "\n" for d in docs), encoding="utf-8")
    return path


def _argv(out_dir: Path, corpus: Path | None, *extra: str) -> list[str]:
    argv = [
        "--preset", "micro-small", "--max-steps", "2", "--batch-size", "8", "--log-every", "1",
        "--n-train-samples", "80", "--n-eval-samples", "16", "--max-seq-len", "128",
        "--device", "cpu", "--output-dir", str(out_dir),
    ]
    if corpus is not None:
        argv += ["--corpus", str(corpus), "--corpus-headroom-tokens", "16", "--corpus-eval-fraction", "0.25"]
    return [*argv, *extra]


def _args(out_dir: Path, corpus: Path | None, *extra: str):
    return build_arg_parser().parse_args(_argv(out_dir, corpus, *extra))


def test_supervised_pause_arm_takes_vocab_and_pause_id_from_the_tokenizer(
    tmp_path, corpus_path, hf_tokenizer
) -> None:
    args = _args(tmp_path / "b3", corpus_path, "--variant", "pause", "--regime", "regime2")

    res = run_training(args, hf_tokenizer)

    model = res["model"]
    assert model.pause_token_id == hf_tokenizer.pause_id != PilotTokenizer().pause_id
    assert model.base_lm.config.vocab_size == hf_tokenizer.vocab_size != PilotTokenizer().vocab_size
    assert math.isfinite(res["final_train_loss"]) and math.isfinite(res["final_val_loss"])
    stats = res["corpus_stats"]
    assert sum(stats.windows["train"].values()) == natural_train_count(80)
    assert sum(stats.windows["eval"].values()) == 16


def test_rl_arm_takes_ids_from_the_tokenizer_and_scores_the_full_horizon(
    tmp_path, corpus_path, hf_tokenizer
) -> None:
    args = _args(tmp_path / "rl", corpus_path, "--mode", "rl", "--group-size", "2", "--thought-len", "4")

    res = run_training(args, hf_tokenizer)

    trainer = res["trainer"]
    assert (trainer.pad_id, trainer.think_start_id, trainer.think_end_id) == (
        hf_tokenizer.pad_id, hf_tokenizer.think_start_id, hf_tokenizer.think_end_id,
    )
    assert math.isfinite(res["final_metrics"]["loss_ntp"])

    # On corpus rows RL sees the whole prefix and all n_ahead target tokens. The
    # stand-in tokenizer encodes a space as ID 0, so a fallback to pad_id=0 would
    # mask every space out of the prefix.
    train_loader, _, _ = build_dataloaders(args, hf_tokenizer)
    checked = 0
    for batch in train_loader:
        prefix_ids, prefix_mask, target_ids, target_mask, *_ = trainer._parse_batch(batch)
        for row, regime in enumerate(batch["regimes"]):
            pl = int(batch["prompt_lens"][row])
            # Every regime: the horizon stops where the row stops, never in the padding.
            row_len = int(batch["attention_mask"][row].sum())
            assert int(target_mask[row].sum()) == min(args.n_ahead, row_len - pl)
            if regime != "natural":
                continue
            assert int(prefix_mask[row].sum()) == pl
            assert prefix_ids[row, -pl:].tolist() == batch["input_ids"][row, :pl].tolist()
            assert target_mask[row].all()
            assert target_ids[row].tolist() == batch["input_ids"][row, pl : pl + args.n_ahead].tolist()
            assert 0 in batch["input_ids"][row, :pl].tolist(), "fixture: prefix must contain ordinary ID 0"
            checked += 1
    assert checked == natural_train_count(80)


def test_model_vocab_follows_the_tokenizer_for_every_preset(hf_tokenizer) -> None:
    parser = build_arg_parser()
    for preset in ("micro-small", "160m", "1b"):
        args = parser.parse_args(["--preset", preset])
        assert resolve_model_config(args, PilotTokenizer()).vocab_size == 512
        assert resolve_model_config(args, hf_tokenizer).vocab_size == hf_tokenizer.vocab_size


def test_checkpoint_records_the_model_config_and_data_protocol(tmp_path, corpus_path, hf_tokenizer) -> None:
    args = _args(tmp_path / "b1", corpus_path)

    res = run_training(args, hf_tokenizer)

    ckpt = torch.load(res["checkpoint_path"], map_location="cpu", weights_only=True)
    assert ckpt["model_config"] == resolve_model_config(args, hf_tokenizer).to_dict()
    protocol = ckpt["data_protocol"]
    assert protocol["tokenizer"] == "pilot"  # the --tokenizer spec the run was launched with
    assert protocol["vocab_size"] == hf_tokenizer.vocab_size
    assert protocol["special_token_ids"]["<pause>"] == hf_tokenizer.pause_id
    assert protocol["tokenizer_fingerprint"] == hf_tokenizer.fingerprint()
    assert protocol["corpus"] == {
        "eval_fraction": 0.25,
        "headroom_tokens": 16,
        "min_prompt_tokens": 16,
        "min_target_tokens": 8,
        "max_seq_len": 128,
    }


def test_init_checkpoint_from_a_different_tokenizer_is_rejected(tmp_path, corpus_path, hf_tokenizer) -> None:
    b1 = run_training(_args(tmp_path / "b1", corpus_path), hf_tokenizer)
    rl = ("--mode", "rl", "--group-size", "2", "--thought-len", "4", "--init-checkpoint", b1["checkpoint_path"])

    same = run_training(_args(tmp_path / "rl", corpus_path, *rl), hf_tokenizer)
    assert math.isfinite(same["final_metrics"]["loss_ntp"])

    with pytest.raises(ValueError, match="tokenizer"):
        run_training(_args(tmp_path / "rl2", corpus_path, *rl), PilotTokenizer())


@pytest.mark.parametrize(
    "flag, value, field",
    [
        ("--corpus-eval-fraction", "0.4", "eval_fraction"),
        ("--corpus-headroom-tokens", "32", "headroom_tokens"),
        ("--max-seq-len", "96", "max_seq_len"),
    ],
)
def test_init_checkpoint_cut_from_other_corpus_windows_is_rejected(
    tmp_path, corpus_path, hf_tokenizer, flag: str, value: str, field: str
) -> None:
    """All arms must train on the same windows, and none may eval on another's train documents."""
    b1 = run_training(_args(tmp_path / "b1", corpus_path), hf_tokenizer)
    argv = _argv(tmp_path / "b1b", corpus_path, "--init-checkpoint", b1["checkpoint_path"], flag, value)

    with pytest.raises(ValueError, match=field):
        run_training(build_arg_parser().parse_args(argv), hf_tokenizer)


@pytest.mark.parametrize(
    "extra",
    [
        ("--variant", "pause", "--n-pause-tokens", "17"),
        ("--variant", "continuous_latent", "--n-latent-steps", "17"),
        ("--mode", "rl", "--group-size", "2", "--thought-len", "15"),
    ],
)
def test_arm_that_inserts_more_positions_than_the_headroom_is_rejected(
    tmp_path, corpus_path, hf_tokenizer, extra: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="headroom"):
        run_training(_args(tmp_path / "arm", corpus_path, *extra), hf_tokenizer)


def test_checkpoint_without_a_protocol_is_treated_as_pilot(tmp_path, corpus_path, hf_tokenizer) -> None:
    b1 = run_training(_args(tmp_path / "b1", None), PilotTokenizer())
    ckpt = torch.load(b1["checkpoint_path"], map_location="cpu", weights_only=True)
    legacy = tmp_path / "legacy.pt"
    torch.save({k: v for k, v in ckpt.items() if k not in ("data_protocol", "model_config")}, legacy)

    ok = run_training(_args(tmp_path / "b1c", None, "--init-checkpoint", str(legacy)), PilotTokenizer())
    assert math.isfinite(ok["final_train_loss"])

    with pytest.raises(ValueError, match="tokenizer"):
        run_training(_args(tmp_path / "b1d", corpus_path, "--init-checkpoint", str(legacy)), hf_tokenizer)


def test_rl_rejects_a_horizon_longer_than_the_corpus_target_floor(tmp_path, corpus_path, hf_tokenizer) -> None:
    args = _args(tmp_path / "rl", corpus_path, "--mode", "rl", "--group-size", "2", "--n-ahead", "12")
    with pytest.raises(ValueError, match="n_ahead"):
        run_training(args, hf_tokenizer)


def test_cli_trains_on_a_corpus_with_the_default_tokenizer(tmp_path, corpus_path, capsys) -> None:
    res = main(_argv(tmp_path / "cli", corpus_path, "--regime", "regime2"))

    assert math.isfinite(res["final_val_loss"])
    out = capsys.readouterr().out
    assert "[supervised]" in out
    assert "[corpus]" in out and "codeparrot" in out


def test_build_tokenizer_routes_pilot_and_hf_specs(monkeypatch, hf_tokenizer) -> None:
    assert isinstance(build_tokenizer("pilot"), PilotTokenizer)

    requested: list[str] = []
    monkeypatch.setattr(
        HFTokenizerAdapter,
        "from_pretrained",
        classmethod(lambda cls, name, **kw: requested.append(name) or hf_tokenizer),
    )
    assert build_tokenizer("some-org/some-tokenizer") is hf_tokenizer
    assert requested == ["some-org/some-tokenizer"]


def test_tokenizer_fingerprint_identifies_vocabulary_and_behaviour(hf_tokenizer, make_fake_hf) -> None:
    assert PilotTokenizer().fingerprint() == PilotTokenizer().fingerprint()
    assert hf_tokenizer.fingerprint() == HFTokenizerAdapter(make_fake_hf()).fingerprint()
    assert hf_tokenizer.fingerprint() != PilotTokenizer().fingerprint()
    assert PilotTokenizer(vocab_size=300).fingerprint() != PilotTokenizer().fingerprint()
    # Same ID-to-token table, different encoding of the same text.
    other = HFTokenizerAdapter(make_fake_hf(honors_split_special_tokens=False))
    assert other.protocol_ids() == hf_tokenizer.protocol_ids()
    assert other.fingerprint() != hf_tokenizer.fingerprint()


def test_published_arm_config_uses_the_checkpoint_model_config(tmp_path, corpus_path, hf_tokenizer) -> None:
    """`publish_hf.py` must describe the model that was trained, not the preset's defaults."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "publish_hf", Path(__file__).resolve().parents[1] / "scripts" / "publish_hf.py"
    )
    publish_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publish_hf)

    args = _args(tmp_path / "B3", corpus_path, "--variant", "pause", "--n-pause-tokens", "3")
    run_training(args, hf_tokenizer)

    payload = publish_hf._build_arm_config_json(tmp_path / "B3", None, None)

    assert payload["model_config"] == resolve_model_config(args, hf_tokenizer).to_dict()
    assert payload["model_config"]["vocab_size"] == hf_tokenizer.vocab_size
    assert payload["data_protocol"]["tokenizer_fingerprint"] == hf_tokenizer.fingerprint()

    card = publish_hf._render_model_card(
        models_repo_id="org/models", datasets_repo_id="org/data", tag="t",
        arm_configs=[payload], report_text=None, metrics_data=None,
    )
    assert 'build_tokenizer(meta.get("data_protocol", {}).get("tokenizer", "pilot"))' in card
    assert "PilotTokenizer()" not in card


def test_real_transformers_tokenizer_trains_supervised_and_rl(tmp_path, corpus_path, real_hf_tokenizer) -> None:
    """The same path on the real library (skipped unless `transformers` is installed)."""
    tok = real_hf_tokenizer
    assert len({tok.pad_id, tok.bos_id, tok.eos_id}) == 3

    b1 = run_training(_args(tmp_path / "b1", corpus_path, "--regime", "regime2"), tok)
    rl = run_training(
        _args(tmp_path / "rl", corpus_path, "--mode", "rl", "--group-size", "2", "--thought-len", "4",
              "--init-checkpoint", b1["checkpoint_path"]),
        tok,
    )

    assert b1["model"].config.vocab_size == tok.vocab_size
    assert math.isfinite(b1["final_val_loss"]) and math.isfinite(rl["final_metrics"]["loss_ntp"])
    assert rl["trainer"].think_start_id == tok.think_start_id
