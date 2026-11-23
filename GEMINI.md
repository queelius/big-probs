# Gemini Cold-Restart Entrypoint (`big-probs`)

> **Canonical Instructions (Single Source of Truth):** Read [`AGENTS.md`](AGENTS.md), [`research/STATE.md`](research/STATE.md), and [`research/PROCEDURE.md`](research/PROCEDURE.md).

## Cold-Restart Boot Sequence (Run Immediately on Turn 1)

1. Read [`research/STATE.md`](research/STATE.md) for the current frontier, hardware constraints, verified state, and top-3 next decisions.
2. Read [`research/PROCEDURE.md`](research/PROCEDURE.md) for the 6-stage lifecycle, isoFLOPs Pareto rules, and anti-patterns.
3. Consult [`research/QUESTIONS.md`](research/QUESTIONS.md), [`research/EXPERIMENTS.md`](research/EXPERIMENTS.md) (`research/experiments.jsonl`), and [`research/BACKLOG.md`](research/BACKLOG.md) (including Dead-Ends Graveyard `DE-001..DE-006`) before proposing or running any experiment.
4. Check repo/test state (`git log -n 5 --oneline && git status`; `cd phase1_pilot && uv run pytest -q`).

## Non-Negotiable Rules (Full Details in [`AGENTS.md`](AGENTS.md))

- **Preregister & normalize:** Preregister non-trivial runs via `research/templates/EXP_PREREG.md`; normalize Pareto claims by inference FLOPs/token & params (`B1`/`B2`/`B3`/`B4`); never fabricate metrics or citations.
- **Verify code:** Run `cd phase1_pilot && uv run pytest -v` after code edits and preserve `< 1e-4` train-vs-eval parity.
- **Artifact hygiene & handoff:** Never commit `.venv/`, `*.pt`, `*.parquet`, or large run dirs to GitHub (publish to HuggingFace via `phase1_pilot/scripts/publish_hf.py`); log runs via `phase1_pilot/scripts/log_experiment.py`, update `research/{STATE,QUESTIONS,EXPERIMENTS,BACKLOG}.md`, and leave `git status` clean.
