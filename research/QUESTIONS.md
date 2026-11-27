# Living Research Question & Hypothesis Registry

> **Purpose:** Tracks every core thesis question (`RQ-xx`) and falsifiable hypothesis (`H1`–`H3`, `H1a`–`H1c`) in the `big-probs` program with explicit falsification criteria, provenance-linked evidence, and honest gap analysis. Update this file whenever an experiment closes in [`research/EXPERIMENTS.md`](EXPERIMENTS.md).

### Status Legend
- `open` — Defined with explicit falsification criterion; not yet tested at a decisive scale.
- `micro-validated` — Core mechanism/invariant verified in code and/or CPU micro-pilot (`EXP-001`), but full target scale remains untested.
- `partially-confirmed` — Confirmed under conditional constraints (e.g., continuous pretraining or warm-start required; see Track 1 §5.5).
- `confirmed` — Passed all preregistered confirmation criteria across decisive scale & multi-seed replicates.
- `falsified` — Failed decisive kill criterion at target scale; pivot or archive in [`research/BACKLOG.md`](BACKLOG.md) Graveyard.

---

## Quick Status Dashboard

| ID | Title | Track / Tier | Status | Linked Experiments |
| :--- | :--- | :--- | :---: | :--- |
| `RQ-01` | SSL-as-RL & Small-Model Weight-Class Punching | Track 1 / Tier 0A | `micro-validated` | `EXP-001`, `EXP-002`, `EXP-003` |
| `RQ-02` | Structural Verification Asymmetry & Grounded Inversion | Track 2 / Tier 0B | `open` | `EXP-004` |
| `RQ-03` | Compressing Slow Verification & Quarantining Epistemic Remainder | Track 3 / Tiers 1–3 | `open` | `EXP-004`, `EXP-005` |
| `H1` | Cold-Start Support vs. On-Policy Transfer (`I2-Hybrid` Bridge) | Cross-Track (1 & 3) | `micro-validated` | `EXP-001`, `EXP-002`, `EXP-003` |
| `H1a` | Strictly Proper Brier (`R3`) vs. Binary Exact-Match (`R1`) Starvation | Track 1 | `micro-validated` | `EXP-001`, `EXP-002` |
| `H1b` | Parametric De-Congestion (`160M` + Latent CoT vs. `410M` NTP) | Track 1 | `open` | `EXP-001`, `EXP-002` |
| `H1c` | Teacher-Free Compute-Elastic Filter (`T2-self`) vs. `T2-scale` / `T1` / `T0` | Track 1 | `open` | `EXP-001`, `EXP-002`, `EXP-003` |
| `H2` | Cross-Modal Stealth-Goodhart Defense Against Correlated Priors | Cross-Track (2 & 3) | `open` | `EXP-004` |
| `H3` | Quasi-Experimental Replay + Co-Adaptive Sim Causal Latency Floor | Cross-Track (2 & 3) | `open` | `EXP-005` |

---

## Part A: Core Thesis Questions

### `RQ-01` — SSL-as-RL & Small-Model Weight-Class Punching
- **Status:** `micro-validated` *(pipeline & anti-sabotage invariants verified in `EXP-001`; 160M/410M/1B H100 scale still `open`)*
- **Question:** Can framing self-supervised next-token prediction as RL over unsupervised latent chain-of-thought (`SSL-as-RL`) enable small models ($N$ parameters) to match or beat $2.5\times$–$6\times$ larger dense NTP models at $\le 1.0\times$ matched inference FLOPs without teacher distillation?
- **Falsification Criterion:** ([`01-ssl-as-rl-latent-cot.md §5.5`](../01-ssl-as-rl-latent-cot.md#L308-L313)) Falsified if, at controlled from-scratch (`160M` vs. `410M`/`1.0B` on 2B tokens) and continued-pretraining (`Qwen2.5-0.5B/1.5B`) scales on Regime 1 algorithmic tasks, no small-model latent-CoT RL arm simultaneously beats (a) FLOPs-matched pause-token control `B3` by $\ge 3$ pp and (b) FLOPs-matched larger NTP baseline `B2` (`410M`) at $\le 1.0\times$ `B2` inference FLOPs, while passing causal thought faithfulness ($\Delta\text{NLL}_{\text{swap}} > 0.15$ nats, $\Delta\text{NLL}_{\text{trunc}} > 0$, monotonic OOD depth scaling) and expanding OOD `pass@64`.
- **Evidence For:**
  - *Literature:* Circuit complexity bounds (Feng et al. `2305.15408`; Merrill & Sabharwal `2310.07923`), `PonderLM-2-1.4B` beating `Pythia-2.8B` (`2509.23184`), `Huginn-3.5B` (`2502.05171`), Snell et al. (`2408.03314`), `HRPO` (`2505.18454`) — see [`00-unified-framework.md §2.1`](../00-unified-framework.md#L82-L84).
  - *Local (`EXP-001`):* `I2-Hybrid` (1.03M) matched `B2` (2.85M) Hard-BPB (`2.271` vs. `2.266`; `B1 = 2.318`), beat `B1`/`B2`/`B3` on OOD `pass@16` (`37.5%` vs. `31.2%`), and beat `B3` (`0.0%`) by `+3.1` pp with `ΔNLL_trunc = +2.732` nats. **Withdrawn as evidence `2026-11-26` (`EXP-006`, `DE-007`):** every `EXP-001` arm scored worse on Regime 1 than a predictor that never reads the prompt (NLL `3.362`–`3.452` vs. `3.150`), so these differences do not measure reasoning.
- **Evidence Against:**
  - *Literature:* `DeepSeek-R1-Zero-Qwen-32B` trails distilled `R1-Distill-Qwen-32B` (`2501.12948`); Yue et al. (`2504.13837`) show RLVR often shrinks `pass@k` at large $k$.
  - *Local (`EXP-001`):* `FALSIFIED` at 0.98M CPU micro-scale: `I2-Hybrid` exact acc (`3.1%`) trailed `B2` (`6.2%`) at `2.51×` `B2` FLOPs, and $\Delta\text{NLL}_{\text{swap}} \approx 0$ across all arms. *Reinterpreted `2026-11-26` (`EXP-006`, `DE-007`):* all arms were below the prompt-blind floor, so this is not evidence against `RQ-01` either.
- **What We're Missing:** `160M`/`410M`/`1.0B` training on 2B tokens (`EXP-002`), real Regime 2 web text evaluation, adaptive/sparse inference triggering to hit $\le 1.0\times$ `B2` FLOPs, and multi-seed confidence intervals.

### `RQ-02` — Structural Verification Asymmetry & Grounded Inversion
- **Status:** `open`
- **Question:** Outside math and software unit tests, can the 8 structural verification mechanisms (grounded real-artifact inversion, NP witnesses/autoformalization, conservation/PDE residuals, primal–dual gaps, grounded round-trips, sandboxed execution, mechanical-leaf adversaries, MDL compression) provide zero-false-positive Tier 0B gates without collapsing into cold-start $\sigma=0$ or Hadamard ill-posedness?
- **Falsification Criterion:** Falsified if, in a non-math/code synthesis domain (`EXP-004`: Scientific PDE solver synthesis or Executable Contract/Tax PoV auditing), lexicographically gated Tier 0B structural verifiers fail to sustain rising held-out ground-truth accuracy over $10^3+$ RL steps or collapse via unphysical/loophole false positives despite physical/formal priors $\Omega_{\text{prior}}(y)$.
- **Evidence For:**
  - *Literature:* `SWE-smith` (`2504.21798`), `AZR` (`2505.03335`), `RLVP` PDE residuals (`2607.10474`), `IPT` (`2604.15149`), `Law & Order` (`2610.02792`), `AlphaEvolve` (`2506.13131`), `RTRL` (`2510.01527`) — see [`02-verification-asymmetry-and-inversion.md §2–§4`](../02-verification-asymmetry-and-inversion.md#L19-L323).
- **Evidence Against:**
  - *Literature:* Cai et al. (`2510.00915`) and Zhu & Kang (`2603.16140`) prove asymmetric FP noise breaks RLVR; Hadamard ill-posedness and steganographic round-trip collusion occur when forward maps co-adapt without frozen anchors or priors.
- **What We're Missing:** Any runnable Tier 0B verification harness in this repository (`EXP-004`), empirical ablation of lexicographic staged gating vs. multiplicative hard gating (`sigma=0` rate), and domain benchmark setup.

### `RQ-03` — Compressing Slow/Delayed Verification & Quarantining the Epistemic Remainder
- **Status:** `open`
- **Question:** For domains with verification latency $\tau \in [10^5, 10^8]\,\text{s}$ (medicine, macroeconomics, materials, novels), can a recalibrated Tier 1–3 funnel (chronological/quasi-experimental replay + recalibrated surrogates + Multi-Fidelity GFlowNets + CCMV auto-decaying milestones + Debate/Peer Prediction) yield compounding real-world lift while exposing irreducible normative/causal trade-offs cleanly to humans?
- **Falsification Criterion:** Falsified if, across $\ge 3$ slow recalibration epochs in a realistic simulator or live anchor (`EXP-005`), the multi-tier funnel fails to (a) outperform naive top-$K$ proxy promotion in real hit-rate per dollar-day, or (b) automatically detect and decay a Goodharted proxy milestone via CCMV / Anchor-and-Drift before terminal objective turns negative.
- **Evidence For:**
  - *Literature:* Chronological forecasting RL (`Turtel et al.` `2505.17989`, `OpenForecaster` `2512.25070`), `rbio1`/RLEMF, `RaR` (`2507.17746`), `RcRL` (`2611.06244`), `PAVs` (`2410.08146`), Multi-Fidelity GFlowNets (`2306.11715`), AI Debate (`2402.06782`), Peer Prediction inverse scaling (`2601.20299`) — see [`03-slow-and-unknown-verification.md §3–§4`](../03-slow-and-unknown-verification.md#L73-L291).
- **Evidence Against:**
  - *Literature:* A-Lab characterization FPs (*PRX Energy* 2024); Gao et al. (`2210.10760`) Goodhart overoptimization curves; Skalse et al. (`2209.13085`); Lucas Critique & Arrow's Impossibility Theorem.
- **What We're Missing:** Runnable causal simulator & CCMV auto-decay prototype (`BL-04` / `EXP-005`), empirical calibration of surrogate TTL / KL-gating thresholds, and a live Tier 3 anchor (e.g., `ForecastBench` or lab partner).

---

## Part B: Deep Cross-Track Hypotheses & Sub-Hypotheses

### `H1` — Cold-Start Support vs. On-Policy Transfer (Continuous-Pretraining $\to$ Hybrid RL Bridge)
- **Status:** `micro-validated` *(I2-Hybrid pipeline micro-validated in `EXP-001`; the OOD `pass@16` lift is withdrawn as evidence by `EXP-006`; transfer geometry vs. `B5` SFT untested)*
- **Claim ([`00-unified-framework.md §4 Tension 1`](../00-unified-framework.md#L164-L168)):** Pretraining a small model with differentiable continuous latent computation (`B4` / `PonderLM-2` / `Huginn` style) followed by strictly on-policy hybrid RL (`I2-Hybrid` / `HRPO`) on teacher-free compute-elastic tokens (`T2-self`) simultaneously breaks through the discrete RL cold-start support wall AND preserves cross-domain transfer geometry destroyed by off-policy teacher SFT distillation (`B5`).
- **Falsification Criterion:** On matched small backbones (`160M` from-scratch; `Qwen2.5-0.5B/1.5B-Base`), compare (a) cold-start discrete RL (`I1`), (b) off-policy 7B-distilled SFT + RL (`B5`), and (c) continuous latent pretraining + on-policy hybrid RL (`I2-Hybrid`). **Confirm `H1`** iff (c) strictly outperforms (a) on OOD algorithmic `pass@64` ($d=6,8$) AND strictly outperforms (b) on non-target transfer domains (coding, general QA, PCA latent preservation) at matched inference FLOPs. **Falsify `H1`** if (c) fails either inequality.
- **Evidence For:**
  - *Local (`EXP-001`):* `I2-Hybrid` achieved the highest OOD `pass@16` of any arm (`37.5%` vs. `RL-R1` `31.2%`, `RL-R3` `25.0%`, `B4` `18.8%`, `B1`/`B2` `31.2%`), lowest small-model Hard-BPB (`2.271` vs. `RL-R3` `2.309` / `RL-R1` `2.326`), `sigma=0` frac `0.00`, and `ΔNLL_trunc = +2.732` nats. **Withdrawn as evidence `2026-11-26` (`EXP-006`, `DE-007`):** all arms were below the prompt-blind floor, and `37.5%` OOD `pass@16` (16 DAG items) is under the `39.8%` that uniform guessing over the 32 node IDs gives.
  - *Literature:* `HRPO` (`2505.18454`) + Huan et al. (`2507.00432`) showing on-policy RL preserves transfer geometry while off-policy SFT causes latent drift.
- **Evidence Against:**
  - *Local (`EXP-001`):* At 0.98M / 20 RL steps, `I2-Hybrid` had `ΔNLL_swap = -0.003` nats (no prompt-specific thought swap sensitivity yet) and `2.51×` `B2` FLOPs (above the `1.0×` isoFLOP target).
- **What We're Missing:** `B5` off-policy teacher-distilled SFT arm, non-target transfer benchmarks + PCA latent-geometry drift metric, `Qwen2.5-0.5B/1.5B` runs (`EXP-003`), and `160M` H100 run (`EXP-002`).

### `H1a` — Strictly Proper Brier Reward (`R3`) vs. Binary Exact-Match (`R1`) Variance Starvation
- **Status:** `micro-validated`
- **Claim ([`01-ssl-as-rl-latent-cot.md §1.3, §5.4`](../01-ssl-as-rl-latent-cot.md#L284-L291)):** On small models with low base accuracy, binary exact-match reward `R1` suffers catastrophic zero-variance group collapse ($\sigma = 0$) and calibration degradation, whereas bounded strictly proper token-level Brier-score gain `R3` ($\text{BS}(q,v^*)=2q(v^*)-\sum_v q(v)^2-1\in[-2,0]$) anchored to frozen $p_{\theta_{\text{ref}}}$ eliminates $\sigma=0$ starvation, avoids $-\infty$ log-likelihood tail variance, and preserves Regime 3 aleatoric calibration.
- **Falsification Criterion:** Falsified if `R3` fails to reduce `sigma=0` group fraction relative to `R1` on small models, yields higher gradient variance / worse Regime 3 ECE than `R1`/`R2`, or underperforms `R2` (clipped log-gain) on both Hard-BPB and Regime 1 accuracy across `EXP-002`.
- **Evidence For:**
  - *Local (`EXP-001` & `tests/test_rl.py`):* Unit tests numerically verified strict propriety and exact $O(\lvert\mathcal{V}\rvert)$ bounds `[-2, 0]`. In live `EXP-001`, **`RL-R1` hit `sigma=0` fraction = `1.00`** (`active_group_frac = 0.0`, `reward_std = 0.0`), whereas **`RL-R3` and `I2-Hybrid` hit `sigma=0` fraction = `0.00`** (`active_group_frac = 0.25`, `reward_std = 0.095` and `0.099`), with `RL-R3` improving Hard-BPB over `RL-R1` (`2.309` vs. `2.326`) and maintaining low Regime 3 ECE (`0.040` and `0.032`). *Scope (`2026-11-26`, Codex review in `EXP-006`):* `sigma=0 = 0.00` under `R3` shows that rewards varied within groups, which any continuous reward does when thoughts perturb logits. It does not show the variation was useful: the policy was at the prompt-blind floor (`DE-007`). Usefulness needs a held-out gain over a matched irrelevant-thought control.
- **Evidence Against:**
  - *Theoretical caveat ([`00-unified-framework.md §3.1`](../00-unified-framework.md#L114-L116)):* Brier score has a flatter gradient slope near $p \approx 0$ than clipped log-likelihood `R2`, which could slow early cold-start learning on large vocabularies (`|V| = 32k–150k`).
  - *Local (`EXP-001`):* `RL-R3` greedy exact acc (`1.6%`) was 1 sample lower than `RL-R1` (`3.1%`, i.e., 1/64 vs. 2/64 — within single-sample noise).
- **What We're Missing:** Direct head-to-head comparison of `R3_brier` vs. `R2_log` (clipped $\tau=5.0$) at both CPU (`BL-02`) and H100 `160M` large-vocab scale (`EXP-002`).

### `H1b` — Parametric De-Congestion (`160M` + Latent Compute vs. `410M` NTP)
- **Status:** `open` *(falsified at 0.98M micro-scale `EXP-001`; untested at 160M target scale `EXP-002`)*
- **Claim ([`00-unified-framework.md §2.1`](../00-unified-framework.md#L83)):** Decomposing parameter capacity into parametric memory ($N_{\text{mem}}$) and serial algorithmic depth ($N_{\text{comp}}$), offloading serial computation into temporal latent steps during from-scratch pretraining (`B4`/`I2`) frees weights for factual knowledge and allows a `160M` model to match/beat a `410M` (`2.5×`) standard NTP baseline at $\le 1.0\times$ inference FLOPs on algorithmic tasks while preserving Regime 3 calibration.
- **Falsification Criterion:** Falsified if in `EXP-002` (`160M` vs. `410M` on 2B tokens), neither `B4` nor `I2-Hybrid` nor `RL-R2/R3` matches `410M` (`B2a`) accuracy/Hard-BPB at $\le 1.0\times$ `410M` inference FLOPs on Regime 1, or if `B4`/`I2` degrades Regime 3 factual/aleatoric recall relative to `B1` (`160M` NTP).
- **Evidence For:**
  - *Literature:* `PonderLM-2-1.4B` > `Pythia-2.8B` (`2509.23184`); `Huginn-3.5B` (`2502.05171`).
  - *Local (`EXP-001`):* Both `B4` (`2.272`) and `I2-Hybrid` (`2.271`) closed ~90% of the Hard-BPB gap between `B1` (`2.318`) and `B2` (`2.266`) using `~1.0M` vs. `2.85M` params. **Withdrawn as evidence `2026-11-26` (`EXP-006`, `DE-007`):** the whole gap lies above the prompt-blind floor (NLL `3.150`, about `2.12` Hard-BPB on the same samples), so closing it measures distance to the answer marginal, not reasoning.
- **Evidence Against:**
  - *Local (`EXP-001`):* At 0.98M / 80 steps, exact accuracy (`3.1%`) lagged `B2` (`6.2%`) and FLOPs (`1.74×` and `2.51×` `B2`) exceeded the `1.0×` isoFLOPs target because extra steps ran on every target token.
- **What We're Missing:** `EXP-002` (`160M` / `410M` / `1.0B`, 2B tokens on H100) + adaptive/sparse inference FLOPs control so small models stay $\le 1.0\times$ `B2` FLOPs.

### `H1c` — Teacher-Free Compute-Elastic Filtering (`T2-self`) vs. Cross-Scale (`T2-scale`), Entropy (`T1`), & All-Token (`T0`)
- **Status:** `open` *(code implemented & `T2-self` verified non-zero PG grads in `EXP-001`; comparative ablation unrun)*
- **Claim ([`01-ssl-as-rl-latent-cot.md §4.1, §5.4`](../01-ssl-as-rl-latent-cot.md#L279-L283)):** Selecting top-20% tokens via strictly teacher-free within-group predictive-gain standard deviation (`T2-self`, $\sigma > 0$) filters out both predictable syntax (~80% of tokens) and irreducible aleatoric noise (Regime 3), capturing the majority of cross-scale teacher-filter lift (`T2-scale`) while strictly beating raw entropy filtering (`T1`) and all-token updates (`T0`) per training FLOP.
- **Falsification Criterion:** Falsified if (a) `T2-self` fails to beat `T0` and `T1` at matched training FLOPs on Regimes 1 & 2, or (b) `T2-scale` outperforms `T2-self` by $>5$ pp while `T2-self` fails to beat `B3` (showing external teacher token selection is required).
- **Evidence For:**
  - *Literature:* Wang et al. (`2506.01939`) showing top-20% forking tokens drive RLVR gains.
  - *Local (`EXP-001` & `tests/test_rl.py`):* Verified `T2-self` selects `active_group_frac = 0.25` with zero teacher dependency under `RL-R3` and `I2-Hybrid`.
- **Evidence Against:** None yet (comparative arms not yet run).
- **What We're Missing:** Direct `T0` vs. `T1` vs. `T2-self` vs. `T2-scale` ablation on CPU (`BL-02`) and Stage 2 H100 (`EXP-003`).

### `H2` — Cross-Modal Structural Decorrelation + Inverse-Scaling Oversight Against Low-KL Stealth Goodharting
- **Status:** `open`
- **Claim ([`00-unified-framework.md §4 Tension 2`](../00-unified-framework.md#L169-L173)):** Same-corpus LLM ensembles (WCO/UWO, BTS) share correlated pretraining priors and fail silently against **low-KL stealth Goodharting**. Neutralizing stealth Goodharting requires cross-modal decorrelation: (i) deterministic Tier 0B programmatic committees (`PAJAMA` + metamorphic `IPT`), (ii) pretraining/retrieval-isolated neural surrogates, and (iii) Peer Prediction (inverse scaling against deception, `2601.20299`) paired with 2% out-of-band randomized ground-truth spot-audits.
- **Falsification Criterion:** On a synthesis benchmark with known hidden ground truth and injected decoy traps (`BL-03` synthetic CPU testbed $\to$ `EXP-004` full scale), run GRPO KL-sweeps (`Gao et al.` protocol) comparing (a) a $K$-model same-corpus LLM WCO/UWO ensemble against (b) the modality-decorrelated stack + 2% randomized spot-audits. **Confirm `H2`** iff (b) cuts stealth-Goodhart FP escape rate by $\ge 50\%$ at matched gold reward and triggers drift alerts before gold reward turns negative. **Falsify `H2`** if (b) matches (a)'s overoptimization collapse curve.
- **Evidence For:**
  - *Literature:* Shuo Li Liu (2026) showing correlated-prior BTS drops accuracy vs. majority vote (`0.9724` vs. `0.9793`); Qiu, Carroll, & Allen (`2601.20299`) Peer Prediction inverse scaling (`0.135B` supervising `8B`); `PAJAMA` (`2607.22561`); `IPT` (`2604.15149`).
- **Evidence Against:** No empirical test run in this repo yet.
- **What We're Missing:** Runnable synthetic correlated-prior Goodhart benchmark (`BL-03`, CPU-runnable!) and full-scale Tier 0B–2 prototype (`EXP-004`).

### `H3` — Quasi-Experimental Replay + Co-Adaptive Simulation Quantifies an Irreducible Causal Latency Floor
- **Status:** `open`
- **Claim ([`00-unified-framework.md §4 Tension 3`](../00-unified-framework.md#L174-L178)):** Replacing naive masked observational replay (which collapses under clinician ordering leakage and unobserved counterfactuals) with propensity-logged natural-experiment / staggered-rollout replay scored by doubly robust (AIPW) estimators—paired with co-adaptive multi-agent Tier 2 simulation (`The AI Economist` style)—eliminates confounding collapse inside historical propensity support, while tracing a quantifiable minimum Tier 3 interventional budget outside support.
- **Falsification Criterion:** In a structural causal testbed with known ground-truth dynamics, ordering bias / strategic co-adaptation, and logged propensities (`BL-04` synthetic CPU testbed $\to$ `EXP-005` live loop), compare (a) naive masked observational replay, (b) propensity-weighted AIPW quasi-experimental replay, and (c) AIPW replay + co-adaptive Tier 2 simulator + budgeted Tier 3 randomized micro-pilots. **Confirm `H3`** iff (b) eliminates (a)'s in-support confounding collapse AND (c) achieves strictly superior out-of-support interventional lift with a smooth shift-vs-Tier-3-budget Pareto curve. **Falsify `H3`** if (b) fails to beat (a) or (c) matches (b) out-of-support.
- **Evidence For:**
  - *Literature:* AIPW doubly robust offline evaluation theory; `The AI Economist` (2022) co-adaptive RL; Track 3 §3 Mech. 1 & §4 analysis of EHR ordering leakage and macro-shock correlation ceilings.
- **Evidence Against:** No empirical test run in this repo yet.
- **What We're Missing:** Runnable synthetic structural causal testbed (`BL-04`, CPU-runnable!) and downstream live Tier 3 anchor (`EXP-005`).
