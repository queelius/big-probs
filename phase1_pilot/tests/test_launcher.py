"""`scripts/launch_h100_stage1.sh`: every job parses with train.py and shares one tokenizer.

The launcher runs against a stub `python` that records its arguments, so nothing trains.
Each recorded command is parsed with `train.py`'s own argument parser.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from phase1_pilot.train import build_arg_parser

_LAUNCHER = Path(__file__).resolve().parent.parent / "scripts" / "launch_h100_stage1.sh"
_STUB = """#!/usr/bin/env bash
# Stand-in for `python -m phase1_pilot.train ...`: write argv (after `-m module`) NUL-separated, one file per call.
printf '%s\\0' "${@:3}" >> "$STUB_LOG.$(cat "$STUB_LOG.count" 2>/dev/null || echo 0)"
echo $(( $(cat "$STUB_LOG.count" 2>/dev/null || echo 0) + 1 )) > "$STUB_LOG.count"
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run_launcher(tmp_path: Path, **env: str) -> list[list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    stub = bin_dir / "python"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    log = tmp_path / "argv"
    inherited = {k: v for k, v in os.environ.items() if k not in ("TOKENIZER", "TOKENIZER_REVISION")}
    full_env = {**inherited, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "STUB_LOG": str(log), "NPROC_PER_NODE": "1", "OUT_ROOT": str(tmp_path / "out"), **env}
    subprocess.run(["bash", str(_LAUNCHER)], env=full_env, check=True, capture_output=True, text=True)
    n = int((tmp_path / "argv.count").read_text())
    return [Path(f"{log}.{i}").read_text().split("\0")[:-1] for i in range(n)]


def test_every_job_parses_and_uses_the_pinned_default_tokenizer(tmp_path) -> None:
    jobs = [build_arg_parser().parse_args(argv) for argv in _run_launcher(tmp_path)]

    assert len(jobs) == 8
    specs = {(a.tokenizer, a.tokenizer_revision) for a in jobs}
    assert len(specs) == 1, f"jobs disagree on the tokenizer: {specs}"
    (spec, revision), = specs
    assert spec != "pilot"
    assert revision is not None and len(revision) == 40  # a full commit hash, not a branch name
    # RL arms start from supervised checkpoints, so they must share the tokenizer (DE-002 chain).
    assert {a.init_checkpoint is not None for a in jobs} == {True, False}


def test_tokenizer_can_be_overridden_and_pilot_takes_no_revision(tmp_path) -> None:
    argvs = _run_launcher(tmp_path, TOKENIZER="pilot")
    jobs = [build_arg_parser().parse_args(argv) for argv in argvs]

    assert len(jobs) == 8
    assert all(argv[argv.index("--tokenizer") + 1] == "pilot" for argv in argvs)  # passed, not defaulted
    assert {(a.tokenizer, a.tokenizer_revision) for a in jobs} == {("pilot", None)}


def test_a_custom_hub_tokenizer_needs_its_own_revision(tmp_path) -> None:
    """The default commit belongs to SmolLM2; reusing it for another repository would be wrong."""
    with pytest.raises(subprocess.CalledProcessError) as err:
        _run_launcher(tmp_path / "unpinned", TOKENIZER="org/other")
    assert "TOKENIZER_REVISION" in err.value.stderr
    assert not (tmp_path / "unpinned" / "argv.count").exists()  # failed before any job started

    jobs = [build_arg_parser().parse_args(argv)
            for argv in _run_launcher(tmp_path / "pinned", TOKENIZER="org/other", TOKENIZER_REVISION="f" * 40)]
    assert {(a.tokenizer, a.tokenizer_revision) for a in jobs} == {("org/other", "f" * 40)}
