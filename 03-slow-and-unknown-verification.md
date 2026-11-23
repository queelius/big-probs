# Track 3: Slow, Delayed, and "Unknown" Verification in Reinforcement Learning

> **Core Thesis:** Self-supervised learning (SSL) revolutionized unsupervised learning by recognizing that the world already supplies labels for free in the form of future and masked tokens. Extending that paradigm to reinforcement learning (RL) requires dissolving a false dichotomy: almost no practically important domain is truly *unverifiable*. Rather, what ML discourse calls "unverifiable" is overwhelmingly **slow, expensive, noisy, or multi-objective verification**. The true bottleneck to general RL is **verification latency**—and solving it requires engineering latency-compressing stacks (chronological replay, recalibrated surrogates, multi-fidelity funnels, temporal credit assignment, and mechanism design) while explicitly defending against Goodhart collapse, temporal leakage, and causal non-identifiability.

---

## 1. Epistemology of "Unverifiable" Domains: Unpacking *"How Did We Ever Know?"*

When practitioners ask how to build automatic verifiers for domains where *"we don't even know how to verify,"* they inadvertently smuggle in a contradiction. As the guiding intuition of this investigation observes: **if we genuinely had no way to verify whether an answer was right, how did humanity ever come to know it was right in the first place?**

If a medical treatment works, we know because patients survive at higher rates in controlled trials. If a macroeconomic policy succeeds, we know because inflation, employment, and productivity trajectories respond over quarters and years. If a novel is great, we know because readers finish it, recommend it, re-read it, and critics articulate why its structure and themes endure. In every case, **an epistemic feedback channel exists**—otherwise the notion of a "right answer" or "better solution" would be empirically empty.

### 1.1 The Four Pseudo-Unverifiable Regimes (Engineering Bottlenecks, Not Epistemic Impossibilities)

Most domains labeled "non-verifiable" in current Reinforcement Learning with Verifiable Rewards (RLVR) literature actually fall into one or more of four **measurement bottlenecks**:

| Regime | Why It *Looks* Unverifiable to Standard RL | Underlying Ground-Truth Mechanism | Concrete Examples |
| :--- | :--- | :--- | :--- |
| **1. Delayed-Outcome** | Feedback latency $\tau \gg$ policy update step; nothing returns an immediate scalar at $t=0$. | Reality resolves the outcome over $\tau \in [\text{days}, \text{decades}]$. | Geopolitical forecasting, startups, Phase III clinical trials, macroeconomic policy, literary endurance. |
| **2. Expensive-to-Measure** | Verification requires scarce physical apparatus, destructive testing, or specialist labor. | Physical experiment, formal audit, or wet-lab assay yields high-fidelity ground truth at high unit cost $C$. | Drug synthesis, inorganic materials characterization (XRD), fusion plasma control, zero-day security audits. |
| **3. Noisy / Stochastic-Preference** | Individual evaluations disagree; no deterministic `assert(y == y*)` exists. | Outcomes are noisy draws from a population utility distribution $R \sim P(\cdot \mid x, a)$ observable via aggregation. | Book reviews, reader retention, market demand, A/B tests, open-ended advice helpfulness. |
| **4. Multi-Objective / Underspecified** | No single scalar captures quality; optimizing any single metric breaks another. | Each component criterion $r_1, \dots, r_m$ is checkable, but the Pareto trade-off vector $\mathbf{w}$ is context-dependent. | Tax design (growth vs. equity), clinical triage (efficacy vs. toxicity), creative prose (originality vs. clarity). |

**Crucial distinction:** In all four regimes, verification is *possible*—it is simply **too slow, too costly, too high-variance, or too high-dimensional** to plug naively into an on-policy RL loop (like PPO or GRPO) that expects millions of cheap, immediate, deterministic rollouts.

### 1.2 The Genuine Epistemic Remainder: What *Truly* Resists Even Slow Verification

Demystifying the four pseudo-unverifiable regimes lets us isolate the **genuine epistemic remainder**—four classes of problems where waiting indefinitely *still* does not yield an objective scalar verifier:

1. **Irreducible Value Judgments & Social Choice Impossibility:** Even with complete empirical knowledge of every consequence, choosing between Pareto-incomparable outcomes (e.g., 1% higher GDP growth vs. 5% lower Gini coefficient; avant-garde literary difficulty vs. mass accessibility) is normative, not empirical. By Arrow's Impossibility Theorem, no universal aggregation function converts heterogeneous individual rankings into a single consistent social preference ordering without violating mild fairness axioms.
2. **Non-Repeatable Historical Counterfactuals (Fundamental Problem of Causal Inference):** For any single historical intervention $a$ taken in world state $s_t$, we observe the realized outcome $Y_t(a)$—never the counterfactual $Y_t(a')$ for the policy not taken. Even 50 years after a central bank rate hike or a diplomatic treaty, we verify *what happened*, not *what would have happened instead*. Causal identification requires randomization, natural experiments, or structural models—all of which carry untestable identifying assumptions.
3. **Reflexive & Self-Altering Feedback Loops:** In social and financial systems, publishing a prediction or deploying an automated verifier alters the behavior of the agents being measured (Soros's reflexivity; the **Lucas Critique**, 1976; **Goodhart's Law**, 1975). A "true" forecast of a bank run causes the bank run; an optimal tax rule induces new tax-avoidance strategies. The ground-truth mapping $P(Y \mid s, a)$ is non-stationary *as a function of the policy $\pi$ itself*.
4. **Empirical Underdetermination (Duhem–Quine Thesis):** Multiple mutually incompatible explanatory theories or mechanistic hypotheses can yield identical observable predictions over all feasible experiments within a given horizon. Verification constrains equivalence classes of models rather than pinpointing a unique latent truth.

> **Epistemic Takeaway:** An SSL-like revolution for RL does *not* require solving pure normativity or observing alternate universes. It requires attacking the **four engineering bottlenecks** (latency, cost, noise, multi-objectivity) while explicitly quarantining the **genuine remainder** (exposing Pareto frontiers to humans instead of hiding value weights inside a scalar reward, and using causal identification rather than naive correlation).

---

## 2. The Slow-Verification Bottleneck: Why Standard RL Breaks

To see why verification latency $\tau$ is the central wall facing general RL, contrast the operating parameters of fast RLVR (math, competitive programming, formal proofs) against real-world slow-verification domains:

| Dimension | Fast RLVR (Math / Code / Games) | Slow Real-World RL (Science / Econ / Culture) | Scaling Gap |
| :--- | :--- | :--- | :--- |
| **Verification Latency ($\tau$)** | $10^{-3}$ – $10^{1}$ seconds | $10^{5}$ – $10^{8}$ seconds (days to years) | $10^{5}\times$ – $10^{10}\times$ |
| **Reset & Branching** | Exact state reset; $G=64$ parallel GRPO rollouts per state | Irreversible reality; single trajectory per macro state | No counterfactual rollouts |
| **Stationarity** | Fixed compiler / mathematical axioms | Drifting distributions; strategic co-adaptation | Non-stationary MDP / Markov game |
| **Noise Floor** | Near-zero (deterministic execution) | High exogenous variance + asymmetric FP/FN error | Low signal-to-noise ratio (SNR) |
| **Unit Cost per Check** | $\sim \$10^{-6}$ (CPU milliseconds) | $\$10^{1}$ – $\$10^{8}$ (wet-lab assay to clinical trial) | $10^{7}\times$ – $10^{14}\times$ |

When $\tau$ grows from milliseconds to months, five compounding failure modes break standard RL:

### 2.1 Wall-Clock Throughput Collapse
On-policy RL algorithms (PPO, GRPO, ReMax) typically require $10^4$ to $10^7$ policy updates, each consuming batches of fresh rollouts generated by the current policy $\pi_\theta$. If verifying a batch takes $\tau = 30\text{ days}$, even 1,000 sequential policy updates require **82 years**. Parallelism helps only up to the physical carrying capacity of the world (available lab benches, patient cohorts, market liquidity) and cannot compress the *serial* latency of multi-step exploration where iteration $k+1$ depends on discovering what failed in iteration $k$.

### 2.2 Non-Stationarity & Co-Adaptive Drift During the Delay Window
Suppose an agent chooses action $a_t$ at time $t$ and receives reward $r_{t+\tau}$ at time $t + \tau$. In a drifting world, the policy gradient estimated from $(s_t, a_t, r_{t+\tau})$ optimizes for the environment of time $t$, which may no longer exist at $t + \tau$. Worse, in multi-agent systems (markets, economics, cultural platforms), other adaptive agents observe $a_t$ and best-respond—meaning the environment shifts *adversarially* against the agent's policy before feedback even closes (**Lucas Critique**).

### 2.3 Credit Assignment & Variance Explosion
Real-world outcomes resolve at the end of long, noisy causal chains: a startup takes 7 years and thousands of strategic decisions to reach an IPO; a novel's reception reflects plot architecture, prose craft, marketing, and cultural timing. For a policy-gradient estimator $\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (R - b)\right]$, variance scales with both horizon length $T$ and exogenous outcome variance $\sigma_{\text{env}}^2$. When $\sigma_{\text{env}}^2 \gg \text{Var}(\mathbb{E}[R \mid a])$, gradient updates are dominated by luck rather than skill, stalling learning unless massive batch sizes or learned value baselines are introduced.

### 2.4 Irreversibility & Off-Policy Distributional Shift
Algorithms like GRPO eliminate the critic network by sampling a group of $G$ rollouts $\{y_1, \dots, y_G\} \sim \pi_\theta(\cdot \mid x)$ from the *exact same state* $x$ and computing group-relative advantages. Reality does not have a `fork()` syscall: you cannot run 16 parallel tax regimes on the United States in 2026 or test 16 mutually exclusive surgical regimens on the same patient. Learning from historical logs forces **offline / off-policy RL**, where importance weights explode over long horizons and policy optimization drifts into out-of-support actions whose consequences are unmeasured.

### 2.5 Verifier Noise, Asymmetric Error, and the Myth of Noise-Tolerant RLVR
Even when slow verification completes, real-world measurements are imperfect. A critical 2025–2026 correction in the RLVR literature punctures early optimism that RL is naturally immune to noisy rewards:

* **Established Literature — Noise Destroys RLVR:** Prior informal claims suggested RLVR tolerated extreme label corruption (even "100% wrong" answers). **Zhu & Kang (2026, arXiv:2603.16140)** (*"Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards"*) demonstrated that those findings were an artifact of **data contamination**, where purportedly corrupted datasets inadvertently retained clean correct answers. After applying strict re-verification filters, truly noisy labels degraded mathematical reasoning accuracy by **8–10%** and real-world Text2SQL accuracy by **5–12%**, with standard GRPO variants failing to compensate.
* **Established Literature — Asymmetric False Positives vs. False Negatives:** **Cai et al. (2025, arXiv:2510.00915)** (*"Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers"*) model imperfect automated verifiers as an asymmetric stochastic channel. **False negatives (FNs)** reject valid solutions, starving positive reinforcement and slowing convergence; **false positives (FPs)** reward wrong solutions and *directly induce reward hacking and Goodhart collapse*. Cai et al. derive **backward correction** (unbiased de-biased reward estimators) and **forward correction** (score-function reweighting using estimated FN rates) alongside online appeals to secondary verifiers, showing that unmodeled FPs are far deadlier than FNs.

---

## 3. Six Mechanisms to Compress or Bypass Verification Latency

Solving the slow-verification bottleneck requires decoupling **policy optimization speed** (milliseconds per gradient step) from **world resolution speed** (days to years). Below, we synthesize six distinct mechanisms—explicitly separating **established 2022–2026 literature** from **original architectural proposals and open frontiers**.

```
                        LATENCY-COMPRESSED VERIFICATION STACK
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FAST LOOP (ms – min)               │  SLOW GROUND-TRUTH ANCHOR (d–yr)  │
  │                                     │                                   │
  │  [1] Historical Replay              │  • Live Market / Forecast Resolve │
  │      (Elapsed Time as Simulator)    │  • Wet-Lab / Physical Synthesis   │
  │  [2] Recalibrated Surrogates &      │  • Randomized Field Rollouts      │
  │      World Models (RLEMF, RaR, GRM) │  • Real Reader / User Telemetry   │
  │  [3] Multi-Fidelity Funnels (T0-T2) │                                   │
  │  [4] Progress / Milestone Critics   │        ▲                 │        │
  │  [5] Debate / Peer Prediction /     │  VoI   │  Promote        │ Drift  │
  │      Conditional Markets            │ Routing│  Diverse        │ Audit &│
  │  [6] Cross-Domain RLVR Transfer     │        │  Top-K          ▼ Recal. │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

### Mechanism 1: Historical Replay & Chronological Backtesting ("Elapsed Time as a Free Simulator")

**Core Idea:** If verification takes $\tau = 6\text{ months}$, you do not have to wait 6 months for every episode—**history has already waited for you**. Any archived decision point $t_0$ whose outcome resolved at $t_0 + \tau \le t_{\text{now}}$ can be replayed in milliseconds as an instant, verifiable RL episode, *provided* the agent's information set is strictly restricted to $\mathcal{F}_{t_0}$ (facts available at $t_0$).

#### Established Literature
* **Outcome-Based RL on Prediction Markets:** **Turtel et al. (2025, arXiv:2505.17989)** (*"Outcome-based Reinforcement Learning to Predict the Future"*) trained a compact 14B reasoning LLM via adapted GRPO (removing per-question variance scaling) and ReMax on ~100k temporally consistent synthetic forecasting questions plus historical prediction-market questions paired with contemporaneous news headlines. Remarkably, outcome-only RL produced a 14B model that matched or exceeded OpenAI's frontier `o1` on holdout forecasting accuracy while dramatically lowering Expected Calibration Error (ECE) and achieving **>10% simulated ROI** on Polymarket backtests.
* **Open-Ended News-Archive Forecasting RL:** **Chandak et al. (2025, arXiv:2512.25070)** (*"Scaling Open-Ended Reasoning to Predict the Future"* / **OpenForecaster**) synthesized the *OpenForesight* dataset from historical news archives under strict offline retrieval cutoffs and trained open models (e.g., OpenForecaster-8B) via RL with Brier-score-derived proper scoring rewards, enabling compact models to rival much larger proprietary models.
* **Contamination-Free Live Evaluation:** **Karger et al. (2025, ICLR 2025, arXiv:2409.19839)** (*ForecastBench*) established a dynamic benchmark of unresolved real-time questions, showing that while LLMs improve rapidly, elite human forecasters still outperform frontier LLMs when temporal leakage is strictly impossible.

#### Structural Failure Modes
1. **Temporal Leakage & Pretraining Contamination:** Pretrained LLMs have memorized post-$t_0$ world history. Even with cutoff-filtered retrieval, subtle phrasing in news articles or implicit parametric memory leaks the resolution—turning "forecasting RL" into glorified recall. Worse, recent methodological critiques demonstrate that recency effects mimic leakage, making naive pre/post-cutoff splits unreliable.
2. **Passive Prediction vs. Active Control (The Counterfactual Wall):** Historical replay works cleanly when the agent is a *price-taker* (forecasting an election or hurricane does not alter the outcome). It **breaks for active control** (setting a nation's interest rate, launching a viral product, or choosing a novel drug trial protocol), because once the agent deviates from the historical action $a_t^{\text{hist}}$, the archived future $Y_{t+\tau}^{\text{hist}}$ no longer corresponds to the agent's trajectory.
3. **Regime Non-Stationarity:** Policies overfit to historical macro/market regimes (e.g., zero-interest-rate era) fail abruptly after structural breaks.

#### Original Proposal: Time-Capsule Pretraining & Natural-Experiment Replay
* **Time-Capsule Base Models:** To make historical replay epistemically airtight, train dedicated **chronologically gated foundation models** (e.g., pretraining checkpoints strictly frozen at yearly intervals $T_1, T_2, \dots, T_k$ using verified cryptographic timestamps / Wayback-verified corpora). RL rollouts at date $t \in (T_i, T_{i+1}]$ run strictly on checkpoint $T_i$, guaranteeing *zero* parametric leakage by construction.
  * *Hard Bottlenecks to Steelman:* First, retraining frontier-scale base models from scratch for each yearly cutoff incurs prohibitive compute costs and creates a structural **capability lag** (older cutoffs have strictly less pretraining data and weaker world models than current frontier models). Second, even leak-free historical archives face a **correlated macro-shock sample-size ceiling**: tens of thousands of historical markets across 2015–2025 are driven by only a handful of statistically independent global regime shocks (e.g., COVID-19, the 2022 inflation surge), inflating apparent sample size while leaving macro tail-risk generalization fragile.
* **Quasi-Experimental Replay for Active Control:** Extend historical replay from passive prediction to *causal decision-making* by curating archives of **natural experiments, staggered policy rollouts, randomized controlled trials (RCTs), and logged A/B tests with recorded propensity scores**. Because propensity scores $p(a \mid s)$ are known, doubly robust off-policy estimators (AIPW) can verify counterfactual actions from elapsed history without waiting for new real-world trials.

---

### Mechanism 2: Surrogate / Amortized Verifiers & World Models — Plus Defenses Against Goodhart Collapse

**Core Idea:** Distill expensive, slow real-world outcomes into fast neural evaluators—either **predictive world models** (which simulate what nature/markets will do) or **structured rubric/generative judges** (which decompose complex evaluation into checkable criteria). Use the surrogate to run millions of fast RL updates.

#### Established Literature
* **Biological World Models as Soft Verifiers (`rbio1` / RLEMF):** **Istrate et al. (Chan Zuckerberg Initiative, 2025, bioRxiv:10.1101/2025.08.18.670981)** introduced **`rbio1`** (built on Qwen2.5-3B-Instruct via GRPO), replacing slow wet-lab validation with **RLEMF (Reinforcement Learning with Experimental Model Feedback)**—querying Virtual Cell Models (like *TranscriptFormer* and task-specific MLPs) as soft verifiers—and **RLPK (Reinforcement Learning from Prior Knowledge)** grounded in Gene Ontology annotations. `rbio1` achieved SOTA on PerturbQA without direct experimental training labels and exhibited zero-shot transfer to disease-state prediction.
* **Structured Rubric & Checklist Verifiers:**
  * **Rubrics as Rewards (RaR) — Gunjal et al. (Scale AI, 2025, arXiv:2507.17746):** Replaces opaque scalar reward models with instance-specific checklist rubrics (7–20 weighted items categorized as Essential, Important, Optional, Pitfalls) inside on-policy GRPO, delivering **28–31% relative gains** over naive LLM-as-a-judge baselines on HealthBench and strong gains on GPQA-Diamond while stabilizing variance across smaller judge models.
  * **Rubric Curriculum RL (RcRL) — Motwani et al. (ICML 2026, arXiv:2611.06244):** Addresses static-rubric exploitation in open-ended domains by combining **pairwise rollout preferences** with a **non-stationary curriculum over rubric criteria**, sustaining entropy and preventing reward hacking over thousands of steps.
  * **Calibration Rewards (RLCR) — Damani et al. (2025, arXiv:2507.16806):** Adds a Brier-score penalty over verbalized confidence directly into RLVR rewards, curing the systematic bluffing/overconfidence induced by binary pass/fail rewards (connecting directly to the proper-scoring-rule analysis in Track 1).
* **Generative & Programmatic Judges:**
  * **DeepSeek-GRM / SPCT (DeepSeek-AI, 2025, arXiv:2504.02495):** Scales generalist reward modeling at inference time via *Self-Principled Critique Tuning* (dynamically generating task-specific principles and critiques before scoring) plus parallel meta-RM voting.
  * **Tool-Integrated & Code-Based Judges:** **TIR-Judge (ICLR 2026, arXiv:2510.23038)** trains judges with code-execution sandboxes via RL; **JudgeRLVR (2026, arXiv:2601.08468)** trains the judge first and initializes the generator from it; **PAJAMA / *Codifying the Judge* (Huang, Qiu, & Sala, 2026, arXiv:2607.22561)** distills fuzzy LLM-judge criteria into committees of transparent, executable Python programs with weak-supervision aggregation and selective LLM fallback.
* **Formal Theory & Mitigations of Goodhart Collapse:**
  * **Skalse et al. (2022, arXiv:2209.13085)** proved that over the set of all stochastic policies, an **unhackable proxy reward function is mathematically almost impossible** unless the proxy is trivial or exact.
  * **Gao, Schulman, & Hilton (2023, arXiv:2210.10760)** empirically derived **scaling laws for reward model overoptimization**: as policy KL divergence $d = \sqrt{D_{\text{KL}}(\pi \| \pi_0)}$ grows against a fixed proxy RM, true gold reward initially rises and then *collapses* along predictable curves governed by RM parameter count and data size.
  * **Coste et al. (2024, ICLR 2024, arXiv:2310.02743)** showed that **Reward Model Ensembles** using **Worst-Case Optimization (WCO)** ($\min_i \hat{R}_i$) and **Uncertainty-Weighted Optimization (UWO)** ($\mu_{\hat{R}} - \lambda \sigma_{\hat{R}}$) substantially curb overoptimization even under heavy label noise.

#### Synthesized Goodhart Defense Playbook (Established + Original Architecture)
Any surrogate verifier used in RL must be treated not as a permanent oracle, but as a **decaying local Taylor approximation of reality** valid only within a trust region around the training distribution:

1. **Epistemic Uncertainty Penalties (WCO/UWO):** Score candidates via diverse surrogate ensembles (varying architecture, data slice, and evaluation modality—e.g., combining a world model, a checklist rubric judge, and a programmatic check) and penalize ensemble disagreement (Coste et al., 2024).
2. **Original Proposal — KL-Gated "Anchor-and-Drift" Recalibration Loop (with Stealth-Goodhart Guards):** Treat the surrogate verifier as a cache with an explicit **distributional Time-To-Live (TTL)**. Track $d_k = \sqrt{D_{\text{KL}}(\pi_k \| \pi_{\text{anchor}})}$ and ensemble variance $\sigma_k$. Once $d_k$ approaches the Gao et al. overoptimization inflection threshold, **freeze policy optimization**, sample high-disagreement/high-proxy-reward candidates (specifically hunting for proxy false positives), submit them to the slow ground-truth channel, and retrain the surrogate on the newly discovered failure modes before unlocking the next RL epoch.
   * *Critical Blind Spot — Low-KL Stealth Goodharting:* Keeping average token-level KL divergence small does **not** guarantee immunity to reward hacking. A policy can flip just a few high-leverage tokens (e.g., stylistic markers, confident phrasing, or shared-prior triggers) that exploit correlated blind spots across an entire ensemble of pretrained judges (echoing the shared-pretraining correlation failure in Shuo Li Liu, 2026) while keeping both $D_{\text{KL}}$ and ensemble variance $\sigma_k$ deceptively low. Therefore, KL gating must be paired with **out-of-band randomized spot-audits** (sampling ground-truth checks uniformly across training time regardless of $d_k$) and **adversarial probe suites** (injecting known decoy/spurious-correlation traps into the policy's prompt stream to detect low-KL exploitation early).
3. **Adversarial & Non-Stationary Criteria:** Periodically rotate or evolve rubric criteria (RcRL) and co-train adversarial "sneaky provers" (Kirchner et al., 2024) specifically rewarded for achieving high surrogate score with low ground-truth quality.
4. **Asymmetric Noise Correction:** Apply Cai et al. (2025) forward/backward gradient corrections using empirical FP/FN rates measured during recalibration audits.

---

### Mechanism 3: Multi-Fidelity Verification Funnels & Autonomous Closed-Loop Labs

**Core Idea:** Instead of paying slow, high-cost verification on every candidate, arrange verifiers in a **progressive cascade** from ultra-fast/cheap filters to slow/expensive reality checks, allocating scarce real-world verification capacity where information gain is highest.

#### Established Literature
* **Multi-Fidelity Active Learning with GFlowNets:** **Hernández-García et al. (2023, arXiv:2306.11715)** coupled multi-fidelity active learning with **Generative Flow Networks (GFlowNets)**, which sample candidates proportional to reward ($p(x) \propto R(x)$) rather than collapsing onto a single argmax mode. Preserving *mode diversity* is essential under surrogate uncertainty: if a standard RL policy collapses onto a single proxy peak that happens to be a simulation artifact, the entire expensive physical validation budget is wasted.
* **Autonomous Closed-Loop Scientific Labs (2025–2026):**
  * **Periodic Labs:** Treats physical nature as the RL environment for materials science (superconductors, batteries, magnets), pairing automated synthesis and characterization (e.g., XRD) with large models (e.g., *Periodic Neon*) explicitly post-trained on messy physical outputs and **negative experimental results**—turning failed syntheses into dense RL training signal rather than discarded lab waste.
  * **Lila Sciences:** Builds *AI Science Factories* integrating reasoning models, programmatic verifiers, and parallel robotic wet labs across drug discovery, catalysts, and carbon capture to create a 24/7 compounding experimental flywheel.
  * **FutureHouse (`Robin` — arXiv:2505.13400, May 2025; `Aviary`):** Built *Aviary* (an RL environment gymnasium for scientific agents) and *Robin* (a multi-agent discovery system combining literature synthesis, hypothesis generation, experimental planning, and data analysis) that autonomously identified and wet-lab-validated a novel therapeutic hypothesis for dry age-related macular degeneration (dAMD).
* **The Cautionary Tale — Even Physical Labs Have Hackable Characterization Layers:** **Szymanski et al. (2023, *Nature*)** reported the autonomous "A-Lab" synthesizing dozens of novel inorganic compounds. Independent re-analysis by **Leeman et al. (2024, *PRX Energy*)** challenged the automated powder XRD phase-identification and novelty claims, noting cases where known disordered or legacy phases were misattributed by automated scripts. In a subsequent *Nature* Author Correction, the A-Lab team clarified that "novel" referred to absence from the computational reference databases (Materials Project/ICSD) rather than unprecedented existence in solid-state chemistry, and manual re-evaluation still confirmed target synthesis in a majority of experimental runs. **Nuanced Lesson:** Even when physical execution succeeds most of the time, automated *characterization and phase-matching scripts* at the bottom of the loop remain vulnerable to systematic false positives—if left unaudited in a tight closed-loop RL cycle, optimization pressure will disproportionately select for those exact instrument/parsing artifacts.

#### Original Proposal: Value-of-Information ($\text{VoI} / (\$\cdot\tau)$) Multi-Fidelity Funnel Routing
Rather than naive top-$K$ threshold promotion (which funnels the worst proxy-hacking outliers into the most expensive tier), design a 4-tier cascade governed by **expected Value of Information per dollar-day**:

| Tier | Verifier Class | Latency | Cost / Check | Role in RL Loop |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0** | Deterministic invariants (conservation laws, syntax, constraints, sanity bounds) | $10^{-3}$ – $10^{-1}$ s | $\sim \$0$ | Hard rejection gate; zero-FP structural filter. |
| **Tier 1** | Amortized neural surrogates / rubric ensembles (`rbio1`, RaR, GRM) | $10^{-1}$ – $10^{1}$ s | $\$10^{-4}$ – $\$10^{-2}$ | High-throughput on-policy RL / GFlowNet sampling. |
| **Tier 2** | Mechanistic simulators (DFT, MD, multi-agent economic sims, persona panels) | $10^{2}$ – $10^{4}$ s | $\$10^{-1}$ – $\$10^{2}$ | Mid-fidelity pruning & ensemble calibration. |
| **Tier 3** | Physical / live reality (autonomous wet-lab + human-audited XRD, live markets, RCTs) | $10^{5}$ – $10^{7}$ s | $\$10^{2}$ – $\$10^{6}$ | Ground-truth anchor; triggers Tier 1–2 recalibration. |

**Routing Rule:** Promote a diverse portfolio of candidates (sampled via GFlowNet / Thompson sampling) from Tier $i$ to Tier $i+1$ maximizing $\frac{\text{Expected Policy Improvement} + \beta \cdot \text{Surrogate Epistemic Uncertainty}}{\text{Cost}_{i+1} \cdot \tau_{i+1}}$. Including the uncertainty term $\beta$ deliberately spends Tier 3 budget to *falsify* regions where the surrogate is overconfident—starving Goodhart collapse at the root.

---

### Mechanism 4: Credit Assignment, Temporal Abstraction & Leading Indicators

**Core Idea:** Even when terminal verification takes years, intermediate states along the trajectory carry strong predictive signal about the terminal outcome. Temporal-difference (TD) learning and process verifiers compress *effective* feedback latency from the episode horizon $T$ down to the step/milestone timescale $\Delta t \ll T$.

#### Established Literature
* **Temporal-Difference (TD) Bootstrapping (Sutton, 1988):** By training a value function $V_\phi(s_t) \approx \mathbb{E}[R_{t+\tau} \mid s_t]$, an agent updates its policy immediately after transitioning from $s_t \to s_{t+1}$ using the TD error $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ without waiting for terminal resolution. However, TD still requires *eventually* grounding $V_\phi$ in terminal outcomes—and suffers from deadly-triad instability under off-policy function approximation.
* **Process Advantage Verifiers (PAVs):** **Setlur et al. (2024, arXiv:2410.08146)** (*"Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning"*) proved a fundamental result for step-level verification: a good process reward should *not* simply predict ultimate success probability $Q(s,a)$, which conflates easy states with good actions. Instead, it must measure **progress (step-level advantage)** $A^{\mu}(s_t, a_t) = Q^{\mu}(s_t, a_t) - V^{\mu}(s_t)$ relative to a prover policy $\mu$. Optimizing progress rewards yields substantially better sample efficiency and online RL performance than outcome-only RL or naive process reward models.
* **The Surrogate Endpoint / Leading-Indicator Trap:** In medicine and business, practitioners routinely replace slow terminal outcomes (5-year survival, 10-year enterprise value) with fast leading indicators (tumor shrinkage, blood biomarkers, 7-day user retention). Biostatistics formalizes when a leading indicator is valid via the **Prentice Criterion**: the treatment effect on the terminal outcome must be *fully mediated* by the surrogate. History is littered with catastrophic violations (e.g., drugs that successfully lowered arrhythmia biomarkers in the CAST trial while *increasing* patient mortality; engagement metrics that boosted 7-day clicks while destroying long-term trust).

#### Original Proposal: Causal-Chain Milestone Verification (CCMV) with Auto-Decaying Weights
To prevent leading indicators from turning into Goodhart traps during RL:
1. Decompose the slow terminal goal $Y_{t+\tau}$ into a causal DAG of intermediate, fast-verifying milestones $\{M_1, M_2, \dots, M_k\}$ (e.g., for drug discovery: binding affinity $\to$ cellular permeability $\to$ metabolic stability $\to$ low toxicity $\to$ in-vivo efficacy).
2. Assign step-level progress rewards $\sum_j w_j \Delta M_j$, **but treat weights $w_j$ as dynamically re-estimated causal parameters**, not fixed constants.
3. Whenever a slow terminal outcome $Y$ finally resolves, run a **surrogate-validity audit**: compute the residual partial correlation $\text{Corr}(Y, M_j \mid \pi_{\text{recent}})$. If the policy has Goodharted milestone $M_j$ (pushing $M_j$ up while $Y$ stops responding), automatically decay $w_j \to 0$ and trigger causal discovery for the next unmodeled bottleneck in the DAG.

---

### Mechanism 5: Mechanism Design, Prediction Markets & Adversarial Social Verification

**Core Idea:** When ground truth is unavailable today—or permanently costly to check—use **game-theoretic incentives and adversarial interaction** so that truth-telling and error-exposing become equilibrium strategies, turning *relative* epistemic asymmetry (checking an exposed flaw is easier than hiding one) into an immediate reward signal.

#### Established Literature
* **Prover-Verifier Games (Legibility as Checkability):** **Kirchner et al. (OpenAI, 2024, arXiv:2407.13692)** (*"Prover-Verifier Games improve legibility of LLM outputs"*) co-trained a small verifier against a **helpful prover** (rewarded for correct solutions accepted by the verifier) and a **sneaky prover** (rewarded for *incorrect* solutions that fool the verifier). Crucially, optimizing against sneaky provers forced helpful provers to produce step-by-step, easy-to-audit reasoning whose legibility transferred directly to time-constrained human judges.
* **AI Debate & Doubly-Efficient Oversight:**
  * **Khan et al. (2024, ICML Best Paper, arXiv:2402.06782)** (*"Debating with More Persuasive LLMs Leads to More Truthful Answers"*) showed that pitting two expert LLMs with access to hidden evidence against each other in front of a weaker judge boosted non-expert accuracy dramatically (**76% LLM judge, 88% human judge**). Counterintuitively, optimizing debaters purely for *persuasiveness* increased truthfulness, because coherent truth is structurally easier to defend under adversarial cross-examination than fabricated lies.
  * **Brown-Cohen, Irving, & Piliouras (Google DeepMind, 2024, arXiv:2311.14125)** (*"Scalable AI Safety via Doubly-Efficient Debate"*) proved theoretical conditions under which an honest debater running in polynomial time can reliably defeat a dishonest debater with exponential simulation power, even for stochastic environments.
* **Peer Prediction & Bayesian Truth Serum (BTS) — Breakthroughs AND Structural Limits (2026):**
  * Classical mechanism design (Prelec, 2004 *Bayesian Truth Serum*; Miller, Resnick, & Zeckhauser, 2005 *Peer Prediction*) scores agents via mutual predictability when no ground truth exists.
  * **Breakthrough (Inverse Scaling Against Deception):** **Qiu, Carroll, & Allen (2026, arXiv:2601.20299)** (*"Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction"*) applied peer prediction to LLM evaluation and post-training. While standard LLM-as-a-judge collapses below random when evaluating deceptive models larger than itself, peer prediction exhibits an **inverse scaling property**: its resistance to deception *strengthens* as the capability gap widens. Training an 8B model with peer-prediction rewards from a tiny 0.135B model recovered most truthfulness lost from prior malicious fine-tuning.
  * **Critical Nuance (BTS Fails Raw Accuracy on Correlated LLM Panels):** **Shuo Li Liu (2026, OpenReview / ICML AgenticUQ Workshop)** (*"Bayesian Truth Serum for LLMs?"*) tested BTS-family rules (BTS-E, BTS-clamp, BTS-info, Surprisingly Popular) on GPT-5 model panels across code review and standard benchmarks. **BTS consistently underperformed plain majority vote in raw accuracy** (e.g., 0.9724 vs. 0.9793), because LLMs share overlapping pretraining corpora (violating conditional independence) and emit decoder samples rather than coherent Bayesian posteriors. *However*, BTS cut binned Expected Calibration Error (ECE) by **21–32%**, proving valuable as a **calibration & abstention statistic** rather than an accuracy maximizer.
* **Self-Verifiable Task Transformation (`SpyRL` / RLSVR):** **RLSVR / SpyRL (2026, arXiv:2607.23802)** converts open-ended writing and summarization into an asymmetric multi-agent social-deduction game ("Who Is the Spy?"). Because the spy's identity is fixed by the game rules, voting accuracy supplies a **100% deterministic, judge-free verifiable reward** tied to semantic precision and discourse coherence.
* **Prediction Markets & Futarchy (Hanson):** Liquid prediction markets with proper scoring rules (Brier / log score) aggregate dispersed private information into a real-time price $p_t = \mathbb{E}[R_{t+\tau} \mid \mathcal{F}_t]$. In principle, conditional markets ("What will 5-year GDP be *if* Policy A is enacted vs. Policy B?") compress a 5-year verification delay into a market-equilibration window of hours. However, conditional decision markets face three severe structural pathologies: (1) **selection bias / winner's curse**—automatically enacting whichever policy posts the highest conditional price systematically selects for positive pricing errors rather than true causal superiority; (2) **voided-branch illiquidity and manipulation**—trades on un-enacted counterfactual branches are refunded (voided), stripping arbitrageurs of incentives to correct mispricings on unlikely policies and allowing cheap manipulation of the decision trigger; and (3) thin liquidity and correlated market-maker priors.

#### Original Proposal: Hybrid Market–Debate Verifiers
Combine the complementary strengths of **Debate** and **Prediction Markets**:
* **Stage 1 (Adversarial Stress-Testing):** Two specialist AI debaters cross-examine a candidate policy, scientific hypothesis, or economic plan over $k$ rounds, forced to cite checkable evidence and surface hidden tail risks (preventing groupthink and correlated blind spots).
* **Stage 2 (Calibrated Pricing):** A diverse panel of forecasting agents (trained via outcome-based RL as in Turtel et al., 2025, with **intentionally heterogeneous pretraining/retrieval feeds** to restore the conditional independence assumption whose violation broke BTS in Shuo Li Liu, 2026) and/or human forecasters price the conditional outcome distribution *after* reading the debate transcript.
* **Stage 3 (Settlement):** The immediate RL reward is the **post-debate expected utility** minus an uncertainty penalty (using BTS-style peer-calibration scores for abstention), while the forecasting agents themselves are continuously scored via proper scoring rules whenever slow real-world outcomes eventually resolve.

---

### Mechanism 6: Cross-Domain Transfer from Fast-Verifiable to Slow/Open-Ended Domains

**Core Idea:** If slow domains cannot support $10^7$ native RL rollouts, train general reasoning, planning, and verification habits on *fast-verifiable* domains (math, code, logic, forecasting replay) and rely on **cross-domain transfer** to bootstrap performance in slow domains—spending precious slow-domain samples only on domain calibration rather than learning how to reason from scratch.

#### Established Literature
* **The Transferability Gap — and Why On-Policy Learning Beats Off-Policy Distillation:** **Huan et al. (2025, arXiv:2507.00432)** (*"Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning"*) audited 20+ open-weight math-reasoning models and found that **most off-the-shelf math models fail to transfer their gains to non-math domains** (coding, scientific QA, agent planning, general instruction following). Crucially, controlled experiments on Qwen3-14B trained on *math-only* data showed that the decisive axis for transferability is **on-policy vs. off-policy training** (learning from the model's own generated rollouts vs. static distilled trajectories), rather than RL vs. SFT in the abstract:
  * **On-policy RL preserves general latent geometry** (verified via PCA of hidden representations) because updates stay close to the model's own rollout distribution, enabling **positive transfer** to both reasoning and non-reasoning domains.
  * **Off-policy SFT on static distilled math traces** forces the model onto an external teacher's token distribution, inducing severe representation drift and **catastrophic forgetting** of general capabilities.
* **Directional Asymmetry in Creative & Open-Ended Transfer:**
  * Pure math/code RL alone often falls short on stylistic nuance, audience empathy, and discourse flexibility in creative writing without intermediate bridges (rubrics, pairwise critique, or task transformations).
  * **Writing-Zero (Jia et al., 2025, arXiv:2506.00103)** (writing-principle pairwise GenRM + Bootstrapped Relative Policy Optimization, BRPO) and **Writing-RL (2025, arXiv:2506.05760)** (margin-aware curriculum + dynamic reference scheduling) demonstrate a striking **reverse transfer**: RL on long-form creative writing—which enforces long-range structural coherence, thematic consistency, and multi-constraint planning—transfers *positively* to complex, long-input logical reasoning tasks.
  * Similarly, **SpyRL (2026, arXiv:2607.23802)** demonstrates that discourse precision learned in self-verifiable deduction games transfers to creative writing and summarization.

#### What Transfers vs. What Does Not (Empirical Boundary)

| Capability Axis | Transfers from Fast RLVR? | Why / Evidence |
| :--- | :--- | :--- |
| **Search & Meta-Cognition** (decomposition, backtracking, self-verification, long-range coherence) | **Yes** (when trained on-policy, not off-policy distillation) | Preserved latent structure under on-policy updates (Huan et al., 2025); reverse writing-to-reasoning transfer (Writing-RL, 2025). |
| **Epistemic Calibration** (knowing what you don't know; proper probabilities) | **Partially** (requires calibration-explicit rewards) | Standard binary RLVR induces bluffing; requires Brier/calibration rewards (Turtel et al., 2025; RLCR, Damani et al., 2025). |
| **Domain World Priors & Causal Dynamics** (cell biology, macro reflexivity, physical synthesis) | **No** | Pure symbolic reasoning cannot deduce empirical laws of nature or human markets without grounded feedback (e.g., `rbio1`, Periodic Labs). |
| **Taste, Norms & Pareto Value Weights** | **No** | Normative and cultural preferences cannot be derived from formal logic; require human grounding or live behavioral telemetry. |

---

## 4. Worked Case Studies: Mapping the User's Three Domains

To make the latency-compression framework concrete, we map the user's three canonical slow/ambiguous domains—**economic systems**, **scientific experiments**, and **novels/creative writing**—showing the exact verification stack, failure modes, and irreducible epistemic remainders for each.

---

### Case Study A: Economic Systems (Macroeconomic Policy, Taxation & Market Design)

* **Epistemic Diagnosis:**
  * *Latency:* Quarters to decades (GDP, productivity, inequality, inflation inertia).
  * *Irreversibility & Counterfactual Wall:* You cannot fork a national economy to run parallel GRPO rollouts; historical macro data suffers from confounding and tiny effective sample size ($N \approx 50$ postwar business cycles).
  * *Reflexivity:* High (**Lucas Critique**)—citizens, firms, and financial markets immediately adapt their expectations and tax/regulatory arbitrage strategies whenever policy rules change.
  * *Multi-Objective Remainder:* Social welfare requires normative weights trading off growth, equality, inflation stability, and individual liberty.
* **Latency-Compressed Verification Stack:**
  1. **Tier 0 — Accounting & Equilibrium Invariants ($\sim$ms):** Budget identities, flow-of-funds conservation, no-arbitrage bounds, and legal/constitutional constraints.
  2. **Tier 1 — Historical & Quasi-Experimental Replay ($\sim$s):** Backtest passive macro forecasting using chronologically gated archives (Mechanism 1) and evaluate localized interventions via archives of natural experiments / staggered municipal/state policy rollouts using doubly robust causal estimators.
  3. **Tier 2 — Co-Adaptive Multi-Agent RL Simulation ($\sim$min–hrs):** Directly address the Lucas Critique using two-level RL as pioneered by **The AI Economist (Zheng et al., *Science Advances*, 2022)**, where lower-level RL worker/firm agents dynamically co-adapt and invent strategic gaming behaviors (e.g., intertemporal income shifting across tax brackets) in response to an upper-level planner RL policy. Stress-test proposed policies across ensembles of heterogeneous agent bounded-rationality models (behavioral + rational expectations) so the planner does not overfit to one idealized utility model.
  4. **Tier 3 — Conditional Prediction Markets & Staggered Field RCTs (hours $\to$ years):** Use conditional futarchy-style forecasting markets + AI Debate transcripts (Mechanism 5) to obtain immediate skin-in-the-game expected-outcome estimates before enactment, followed by staggered regional pilots / randomized field experiments (where ethical and feasible) to recalibrate the Tier 2 micro-foundational agent simulators.
* **Primary Failure Mode:** **Sim-to-Real Structural Misspecification**—simulated agents in Tier 2 may miss real-world institutional frictions, political coalitions, or panic dynamics (bank runs), causing a policy that looks Pareto-optimal in simulation to trigger crises in reality.
* **Irreducible Epistemic Remainder:** Even a perfect economic simulator can only output a **Pareto frontier** (e.g., $\Delta\text{GDP}$ vs. $\Delta\text{Gini}$ vs. $\Delta\text{Tail-Risk}$). Selecting a point on that frontier is permanently a democratic/normative choice, not an automatic verification problem.

---

### Case Study B: Scientific Experiments (Drug Discovery, Materials & Perturbation Biology)

* **Epistemic Diagnosis:**
  * *Latency:* Days (high-throughput screening) to months (animal models, synthesis optimization) to 5–10 years (Phase III clinical endpoints).
  * *Physical Bottleneck & Cost:* Wet-lab reagents, robotic throughput, and clinical enrollment cap total Tier 3 evaluations at $10^2$–$10^5$ per program.
  * *Measurement Noise:* Automated physical characterization can generate systematic false positives (**A-Lab XRD critique**, Leeman et al., 2024).
  * *Surrogate Endpoint Risk:* Biomarker improvements frequently fail to translate to human clinical survival (Prentice Criterion violations).
* **Latency-Compressed Verification Stack:**
  1. **Tier 0 — Physical, Chemical & Synthesizability Invariants ($\sim$ms):** Valence rules, thermodynamic stability bounds, retrosynthetic route existence, toxicity/PAINS structural alerts.
  2. **Tier 1 — Amortized World Models as Soft Verifiers ($\sim$s):** Use domain foundation models—e.g., Virtual Cell Models via **RLEMF (`rbio1`, Istrate et al., 2025)** plus **RLPK** (Gene Ontology / pathway graphs)—for rapid on-policy RL over biological reasoning and candidate generation, regularized by ensemble uncertainty penalties (Coste et al., 2024).
  3. **Tier 2 — Mechanistic Physics & Multi-Fidelity Diversity Sampling ($\sim$min–hrs):** Filter candidates through physics-based simulators (DFT, molecular dynamics, free-energy perturbation) orchestrated by **Multi-Fidelity GFlowNets (Hernández-García et al., 2023)** to preserve structural diversity across distinct chemical/mechanistic modes rather than collapsing onto a single surrogate hack.
  4. **Tier 3 — Autonomous Closed-Loop Labs with Adversarial Characterization Audits (days–months):** Execute physical synthesis and assays in automated labs (**Periodic Labs**, **Lila Sciences**, **FutureHouse `Robin`**), explicitly logging **both positive and negative results** back into Tier 1 world-model retraining. Crucially—learning directly from the A-Lab controversy—insert **independent orthogonal characterization checks** (e.g., blind secondary assay / human expert spot-auditing of raw spectra rather than trusting a single automated XRD script) to cap the Tier 3 false-positive rate (Cai et al., 2025).
* **Primary Failure Mode:** **Translational Cliff**—optimizing Tier 1–3 in-silico and in-vitro proxies until candidates are ultra-potent in cell culture yet fail due to unmodeled human systemic pharmacokinetics or off-target organ toxicity in Phase II clinical trials.
* **Irreducible Epistemic Remainder:** Biological complexity ensures that *novel mechanisms of action* always step outside the training support of existing world models; physical human/in-vivo validation remains the irreducible floor of latency for clinical safety, compressible via organoids/virtual-cell advances but never to zero.

---

### Case Study C: Novels & Creative Writing (*"Novels That Get Good Reviews"*)

* **Epistemic Diagnosis:**
  * *Latency:* Months to years (editorial review, publication, reader word-of-mouth, literary criticism, cultural staying power).
  * *Noisy, Heterogeneous Preferences:* There is no single scalar "goodness"—thriller readers, literary critics, and genre fans weight pacing, subtext, originality, and emotional resonance differently.
  * *Goodhart Pathology ("AI Slop"):* Naive scalar reward models trained on static human ratings rapidly degenerate into length bias, purple prose, melodramatic clichés, and sycophantic trope-checking.
* **Latency-Compressed Verification Stack:**
  1. **Tier 0 — Verifiable Narrative & Continuity Invariants ($\sim$ms–s):** Programmatic checks—combining hard constraint verification with dynamic quality-conditioned weighting as in **RLMR (Liao et al., 2025, arXiv:2508.18642)** and tool-integrated judges—verifying hard structural facts across 100k tokens: character entity tracking, timeline consistency, POV discipline, Chekhov's-gun setup/payoff resolution, and absence of lexical repetition loops.
  2. **Tier 1 — Dynamic Rubric, Pairwise Critique & Task-Transformation RL ($\sim$s):** Replace static scalar RMs with **instance-specific, non-stationary rubrics** (**RaR**, Gunjal et al., 2025; **RcRL**, Motwani et al., 2026, `arXiv:2611.06244`) and **pairwise self-principled critique** (**Writing-Zero / BRPO**, Jia et al., 2025; **DeepSeek-GRM**, 2025; **Writing-RL**, 2025). Supplement with judge-free **SpyRL task transformations** (2026) to sharpen semantic precision and voice distinctiveness without reward-model exploitation.
  3. **Tier 2 — Persona-Diverse Reader Simulation & Adversarial Literary Debate ($\sim$min):** Evaluate chapter arcs via a **multi-perspective critic panel** (distinct reader personas explicitly conditioned on divergent tastes to prevent collapse to bland "median-reader" prose) coupled with **adversarial literary cross-examination** (Mechanism 5: a "hostile critic" agent rewarded for pinpointing unearned plot twists, flat character motivation, or derivative tropes, while a "defender" agent must justify choices from textual evidence).
  4. **Tier 3 — Serialized Behavioral Telemetry & Delayed Durability Anchoring (weeks–years):** Publish serialized chapters or pilot excerpts to real reader cohorts and measure **revealed behavioral preferences**—chapter-to-chapter completion rates, discussion depth, voluntary re-read rates, and independent critic reviews.
     * *Engagement Goodharting Warning:* Fast behavioral telemetry (chapter completion and comment volume) is itself a notorious **Prentice Criterion trap** (Mechanism 4): optimizing short-term retention and discussion volume is the exact mechanism that produces cliffhanger-padded web serials, repetitive dopamine loops, and rage-bait rather than literature. Mitigation requires treating short-term telemetry strictly as an *auto-decaying leading indicator* under **CCMV** (Mechanism 4), anchored to **delayed durability outcomes** (post-completion satisfaction surveys weeks later, voluntary re-reads, unprompted recommendations, and independent literary criticism) via the Anchor-and-Drift recalibration loop so engagement proxies lose reward weight whenever they diverge from lasting reader value.
* **Primary Failure Mode:** **The Novelty / Pastiche Paradox**—every historical-review surrogate or rubric trained on *past* acclaimed novels inherently scores adherence to established literary conventions. Truly paradigm-shifting art (e.g., *Ulysses*, *Beloved*, *Infinite Jest*) often looks *worse* under existing rubrics before reshaping reader taste itself. Pure optimization against a historical surrogate selects for hyper-polished pastiche rather than cultural originality.
* **Irreducible Epistemic Remainder:** Aesthetic taste is genuinely plural and culturally evolving. Automatic verification can reliably separate *craft competence* (coherence, pacing, voice control, avoidance of clichés) from slop, and can predict commercial appeal within established genres—but determining whether a novel achieves *enduring artistic significance* requires living human communities experiencing it across time.

---

## 5. Comparative Synthesis & Audit of Established Literature vs. Original Proposals

### 5.1 Mechanism Trade-Off Matrix

| Mechanism | Latency Compression | Handles Active Counterfactuals? | Goodhart / Exploitation Risk | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **1. Historical Replay / Backtesting** | $10^{6}\times$ (years $\to$ ms) | **No** (passive prediction only, unless RCT/natural exp.) | Low (if leak-free) | Temporal leakage & regime shift |
| **2. Surrogate Verifiers & World Models** | $10^{5}\times$ – $10^{7}\times$ | **Yes** (within model support) | **Very High** (without recalibration) | Distribution shift & overoptimization |
| **3. Multi-Fidelity Funnels & Closed-Loop Labs** | $10^{2}\times$ – $10^{4}\times$ effective | **Yes** (grounded in physics) | Moderate (characterization FPs) | Tier 3 hardware cost & assay artifacts |
| **4. TD Value / Process Progress / Milestones** | $10^{1}\times$ – $10^{3}\times$ | **Yes** (if causally valid) | High (surrogate endpoint paradox) | Prentice criterion failure / off-policy drift |
| **5. Mechanism Design / Debate / Markets** | $10^{3}\times$ – $10^{5}\times$ | **Partially** (conditional markets) | Moderate–Low (inverse scaling for peer pred.) | Correlated LLM priors; market liquidity |
| **6. Cross-Domain RLVR Transfer** | Amortizes reasoning cost | N/A (bootstraps policy) | Low (on-policy RL preserves geometry) | Does not supply empirical world priors |

### 5.2 Epistemic Audit: Established Literature vs. Original Proposals

To maintain strict scientific hygiene, the table below separates empirically validated results (2022–2026) from our original architectural syntheses and untested hypotheses:

| Category | Claims & Citations | Status |
| :--- | :--- | :--- |
| **Established Literature** | Outcome-based RL on forecasting achieves frontier accuracy & superior calibration (**Turtel et al., 2025, arXiv:2505.17989**; **Chandak et al., 2025, arXiv:2512.25070**); live human forecasters still lead on contamination-free **ForecastBench (Karger et al., 2025, arXiv:2409.19839)**. | Verified empirical results |
| **Established Literature** | Soft biological world-model verifiers (**`rbio1` / RLEMF**, Istrate et al., 2025, bioRxiv:10.1101/2025.08.18.670981); closed-loop labs (**FutureHouse `Robin`**, arXiv:2505.13400; Periodic Labs; Lila Sciences); automated XRD characterization nuance in **A-Lab** (Szymanski et al., 2023; critique by Leeman et al., *PRX Energy*, 2024; *Nature* Author Correction). | Verified empirical results |
| **Established Literature** | Rubric/critique/calibration RL (**RaR**, Gunjal et al., 2025, arXiv:2507.17746; **RcRL**, Motwani et al., ICML 2026, arXiv:2611.06244; **RLCR**, Damani et al., 2025, arXiv:2507.16806; **RLMR**, Liao et al., 2025, arXiv:2508.18642; **DeepSeek-GRM**, arXiv:2504.02495; **Writing-Zero**, arXiv:2506.00103; **Writing-RL**, arXiv:2506.05760; **TIR-Judge**, arXiv:2510.23038; **JudgeRLVR**, arXiv:2601.08468; **PAJAMA**, Huang et al., 2026, arXiv:2607.22561). | Verified empirical results |
| **Established Literature** | Noisy data breaks RLVR once contamination is removed (**Zhu & Kang, 2026, arXiv:2603.16140**); asymmetric FP/FN corrections (**Cai et al., 2025, arXiv:2510.00915**); unhackability theorem (**Skalse et al., 2022, arXiv:2209.13085**); overoptimization scaling (**Gao et al., 2023, arXiv:2210.10760**); ensemble WCO/UWO (**Coste et al., 2024, arXiv:2310.02743**). | Verified theory & empirical results |
| **Established Literature** | Prover-Verifier Games (**Kirchner et al., 2024, arXiv:2407.13692**); Persuasive Debate (**Khan et al., 2024, arXiv:2402.06782**); Doubly-Efficient Debate (**Brown-Cohen et al., 2024, arXiv:2311.14125**); Peer Prediction inverse scaling (**Qiu et al., 2026, arXiv:2601.20299**); BTS accuracy drop vs. ECE gains (**Shuo Li Liu, 2026**); **SpyRL** (arXiv:2607.23802). | Verified theory & empirical results |
| **Established Literature** | Process Advantage Verifiers (**Setlur et al., 2024, arXiv:2410.08146**); Multi-Fidelity GFlowNets (**Hernández-García et al., 2023, arXiv:2306.11715**); Two-level co-adaptive economic RL (**Zheng et al., 2022**); On-policy vs. off-policy transferability (**Huan et al., 2025, arXiv:2507.00432**). | Verified empirical results |
| **Original Proposals / Open Hypotheses** | (1) **Time-Capsule Pretraining + Quasi-Experimental Replay** for zero-leakage causal RL; (2) **KL-Gated "Anchor-and-Drift" Surrogate Recalibration** (augmented with randomized spot-audits & adversarial probes against low-KL stealth Goodharting); (3) **$\text{VoI}/(\$\cdot\tau)$ Funnel Routing** with epistemic uncertainty promotion; (4) **Causal-Chain Milestone Verification (CCMV)** with auto-decaying weights; (5) **Hybrid Market–Debate Verifiers** using pretraining-heterogeneous forecaster panels. | Original architectural proposals requiring empirical validation |

---

## 6. Cross-Track Connections & Concluding Synthesis

The slow-verification lens completes the unifying picture of *"doing for RL what SSL did for unsupervised learning"*:

1. **Link to Track 1 (SSL as RL with Latent CoT):** Just as chronological backtesting turns *elapsed world history* into free verifiable RL episodes (Mechanism 1), next-token prediction with latent CoT turns *every text corpus* into a zero-latency verification stream where the next token is the future outcome. Forecasting RL (Turtel et al., 2025; OpenForecaster) is literally the **macro-horizon generalization of SSL-as-RL**—predicting next-quarter world states instead of next-millisecond tokens, scored by proper scoring rules.
2. **Link to Track 2 (Structural Verification Asymmetry & Inversion):** Structural asymmetries (invariants, conservation laws, cycle-consistency, NP witnesses) form **Tier 0** of our multi-fidelity funnel—instant, zero-FP guardrails that shrink the search space before noisy neural surrogates (Tier 1) or slow physical/social verification (Tier 3) are ever invoked. On **adversarial verification**, the two tracks divide labor cleanly along the nature of the *leaf check*: **Track 2** covers adversarial symmetry breaking whose leaf claims terminate in **mechanical, deterministic checks** (fraud-proof opcodes, fuzzing PoV crashes, unit-test counterexamples), whereas **Track 3** covers **epistemic, ungrounded oversight** (AI Debate, Peer Prediction, Conditional Markets) where even the leaf claims remain uncertain, noisy, or delayed and must be anchored via game-theoretic incentives and delayed settlement.
3. **The Core Strategic Lesson:** Generalizing RL beyond math and code will not come from discovering a mythical universal scalar verifier. It will come from building **hierarchical, self-recalibrating verification stacks** that (a) mine elapsed history without leakage, (b) amortize slow outcomes into uncertainty-aware surrogates and progress critics, (c) route diverse candidates through multi-fidelity physical/social funnels, and (d) continuously re-anchor every proxy to slow ground truth before Goodhart's Law turns optimization into self-deception.
