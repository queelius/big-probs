# `big-probs` — Canonical Agent Cold-Restart Instructions (`AGENTS.md`)

> **Single Source of Truth:** This file governs every autonomous agent and coding assistant working in `queelius/big-probs`. All project methodology, artifact boundaries, and handoff rules derive from [`research/STATE.md`](research/STATE.md) and [`research/PROCEDURE.md`](research/PROCEDURE.md).

---

## 1. Mandatory <2-Minute Cold-Restart Boot Sequence

Before writing code, proposing an experiment, or running compute from a fresh context, execute these 4 steps in order:

1. **Read [`research/STATE.md`](research/STATE.md)** — current phase, Pareto frontier snapshot, local hardware constraints, verified components, and top-3 resume decisions.
2. **Read [`research/PROCEDURE.md`](research/PROCEDURE.md)** — 6-stage experiment lifecycle, isoFLOPs Pareto normalization rules (§3), cold-restart protocol (§4), anti-patterns (§5), and artifact discipline (§6).
3. **Consult the living registries before proposing or running any experiment:**
   - [`research/QUESTIONS.md`](research/QUESTIONS.md) — active hypotheses (`H1`–`H3`, `H1a`–`H1c`) and Research Questions (`RQ-xxx`).
   - [`research/EXPERIMENTS.md`](research/EXPERIMENTS.md) & [`research/experiments.jsonl`](research/experiments.jsonl) — append-only ledger of completed and planned runs (`EXP-001`, ...).
   - [`research/BACKLOG.md`](research/BACKLOG.md) — VoI-ranked queue (`BL-01`..`BL-10`), open gaps, and the **Dead-Ends Graveyard (`DE-001`..`DE-006`)** so you never repeat an already-falsified mechanism or known trap.
4. **Verify workspace & test baseline:**
   ```bash
   git log -n 5 --oneline && git status
   cd phase1_pilot && uv run pytest -q
   ```

---

## 2. Repository Map

- `00-unified-framework.md`, `01..03-*.md` — Theoretical synthesis, 4-Tier Verification Stack, and falsifiable hypotheses `H1`–`H3`.
- `research/` — Cold-restart operating memory (`STATE.md`, `PROCEDURE.md`, `QUESTIONS.md`, `EXPERIMENTS.md`, `experiments.jsonl`, `BACKLOG.md`, `templates/EXP_PREREG.md`).
- `phase1_pilot/` — Track 1 Latent-CoT RL codebase (`phase1_pilot/`, `tests/`, `scripts/{run_micro_pilot.py,launch_h100_stage1.sh,log_experiment.py,publish_hf.py}`, `results/micro_pilot_{report.md,metrics.json}`).

---

## 3. Hard Research Discipline Rules

- **Preregister Before Compute:** For every non-trivial experiment (`EXP-xxx`), instantiate [`research/templates/EXP_PREREG.md`](research/templates/EXP_PREREG.md) with competing hypotheses, frozen evaluation axes, and explicit kill criteria (stability, `sigma=0` signal collapse, anti-sabotage drift, dominance stop-loss), and commit to Git before launching training.
- **Climb the Ladder of Scale (`L0 -> L1 -> L2 -> L3`):** Never spend GPU budgets to discover bugs catchable on CPU. Pass `L0` unit invariants (`<10s`), then `L1` CPU micro-pilot (`1–5 min`), before `L2`/`L3` H100 runs. Never present `L1` proof-of-life numbers as `L2`/`L3` capability proofs.
- **Normalize Every Pareto Claim:** Always report **inference FLOPs/token** and **parameter count** alongside accuracy/Hard-BPB against the required controls (`B1` small base, `B2` iso-FLOPs parameter-scaled NTP, `B3` iso-FLOPs pause control, `B4` continuous latent control). Never compare latent-CoT to raw `B1` alone. Pair accuracy with causal faithfulness (`ΔNLL_swap`, format-matched `ΔNLL_trunc`) and Regime 3 calibration (`ECE`).
- **Zero Fabricated Metrics or Citations:** Every metric, verdict, or empirical claim must cite an exact Git commit SHA, raw JSON artifact (e.g., `phase1_pilot/results/micro_pilot_metrics.json`, `research/experiments.jsonl`), or HuggingFace URI. Generate Markdown tables programmatically via `log_experiment.py`; never transcribe unverified numbers.
- **Mandatory Code & Parity Verification:** After any code change in `phase1_pilot/`, run:
  ```bash
  cd phase1_pilot && uv run pytest -v
  ```
  Every latent/thought interface change must maintain `< 1e-4` train-vs-eval log-prob parity (`DE-003`), frozen reference-anchor anti-sabotage (`DE-002`), and 2-rank distributed sync tests (`DE-004`).

---

## 4. Git vs. HuggingFace Artifact Hygiene

| Storage | Canonical Repo | Allowed Contents | Forbidden |
| :--- | :--- | :--- | :--- |
| **GitHub** | `queelius/big-probs` | Code, theory docs (`00..03-*.md`), `research/`, tests, scripts, lightweight reports & metrics JSONs (`phase1_pilot/results/micro_pilot_report.md`, `phase1_pilot/results/micro_pilot_metrics.json`) | `.venv/`, `*.pt`, `*.safetensors`, `*.parquet`, secrets, large run dirs (`results/micro_runs/`, `runs/`) |
| **HuggingFace Hub** | `queelius/big-probs-phase1-models` & `queelius/big-probs-phase1-data` | Model checkpoints, weights, large Parquet datasets, Model/Dataset Cards | Unverified local scratch dumps |

Publish promoted checkpoints and datasets using:
```bash
cd phase1_pilot
uv run --with huggingface_hub --with pyarrow python scripts/publish_hf.py \
  --namespace queelius --tag <run-tag> --run-dir <run-dir>
```

---

## 5. Mandatory End-of-Session Handoff Checklist

Never end a session with undocumented empirical state or a dirty working tree. Before finishing:

- [ ] **Log every executed run** (including failed, killed, or null runs) into append-only `research/experiments.jsonl` via `phase1_pilot/scripts/log_experiment.py` (never rewrite prior JSONL rows; use `"supersedes"` if correcting a buggy run):
  ```bash
  cd phase1_pilot && uv run python scripts/log_experiment.py --metrics <path/to/metrics.json> --exp-id <EXP-xxx> --title "<title>"
  ```
- [ ] **Refresh research memory:** Update `research/EXPERIMENTS.md` (post-mortem), `research/QUESTIONS.md` (hypothesis statuses), `research/BACKLOG.md` (queue + Dead-Ends Graveyard for any newly discovered failure mode), and `research/STATE.md` (ground truth + top next actions).
- [ ] **Verify tests & clean Git state:** Confirm `cd phase1_pilot && uv run pytest -v` passes, heavy artifacts are offloaded to HuggingFace Hub (or `.gitignore`d), and `git status` is clean.
