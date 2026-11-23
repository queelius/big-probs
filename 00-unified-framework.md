# Self-Supervised RL & Universal Verification: A Unified Framework

> **Master Synthesis** — Integrating [Track 1: SSL-as-RL & Latent CoT](file:///home/spinoza/github/big-probs/01-ssl-as-rl-latent-cot.md), [Track 2: Verification Asymmetry & Inversion](file:///home/spinoza/github/big-probs/02-verification-asymmetry-and-inversion.md), and [Track 3: Slow & "Unknown" Verification](file:///home/spinoza/github/big-probs/03-slow-and-unknown-verification.md).
>
> **Epistemic Convention:** Throughout this synthesis, claims, theorems, and metrics grounded in peer-reviewed or preprint literature (2022–2026 + classical anchors) verified across Tracks 1–3 are marked **[Established Literature]**; cross-track conceptual frameworks, architectural designs, and falsifiable hypotheses are marked **[Original Synthesis]** or **[Original Proposal]**.

---

## Executive Summary

1. **Direct Answer to the Overarching Thesis:** Doing for reinforcement learning (RL) and verification what self-supervised learning (SSL) did for unsupervised learning does *not* require inventing a mythical universal scalar oracle. SSL scaled because next-token prediction is a **degenerate boundary case of automatic verification**—trivial masking inversion ($\text{zero pushforward shift}$) intersected with lossless predictive compression at **zero verification latency ($\tau = 0$)**. Generalizing SSL to arbitrary domains requires leaving that boundary systematically along three axes: **latent compute depth** (Track 1), **structural verification asymmetry** (Track 2), and **verification latency compression** (Track 3).
2. **One Continuous Four-Tier Architecture:** Rather than treating SSL, RLVR, simulators, and real-world experiments as separate paradigms, they compose into a single **Unified Verification Stack**: **Tier 0A** (corpus-native latent-CoT RL trading temporal depth for parameter width), **Tier 0B** (zero-false-positive structural gates—grounded inversions, NP witnesses, conservation residuals, duality gaps, and mechanical-leaf adversaries), **Tier 1** (chronological replay, recalibrated world models, dynamic rubrics, and progress critics compressing latency by $10^5\times$–$10^7\times$), **Tier 2** (mechanistic simulators and diversity-preserving GFlowNet search), and **Tier 3** (slow physical/market reality anchors and epistemic oversight that continuously recalibrate Tiers 1–2).
3. **Four Recurring Mathematical Primitives:** Across every timescale—from millisecond token prediction to multi-year clinical trials—four exact mechanisms govern whether RL compounds or collapses: **(i) strictly proper / continuous progress rewards** over mode-collapsing binary pass/fail; **(ii) asymmetric defense against false positives** ($\text{FP} \gg \text{FN}$); **(iii) immutable anchors & stop-gradients** against co-adaptation and Goodharting; and **(iv) information-theoretic compression & compute elasticity** targeting reducible epistemic uncertainty.
4. **Three Genuine Research Frontiers:** Cross-track auditing surfaces three deep, unresolved collisions—**(1) Cold-Start Support vs. On-Policy Transfer**, **(2) Correlated Priors Breaking Ensemble Goodhart Defenses**, and **(3) The Causal Counterfactual Ceiling**—for which we formulate explicit, falsifiable hypotheses (H1–H3).
5. **Actionable Execution Path:** We distill a four-phase experimental roadmap anchored immediately by a runnable **~30–45 H100 GPU-hour small-model latent-CoT RL pilot** (Track 1 Stage 1), scaling into H1 transfer audits (~250–400 GPU-hrs), a cross-domain Tier 0B–2 prototype testing H2, and a latency-compressed Tier 3 loop testing H3.

---

## 1. Answering the Thesis: From Degenerate SSL to Universal Verification

Why did SSL unlock web-scale representation learning while RL with Verifiable Rewards (RLVR) appears trapped inside formal mathematics and software engineering? Viewing the landscape through all three tracks dissolves this apparent chasm **[Original Synthesis]**.

### 1.1 SSL as the Zero-Shift, Zero-Latency Origin of Automatic Verification
Standard autoregressive next-token prediction over $x_{1:T} \in \mathcal{V}^T$ is mathematically a **1-step contextual bandit** whose environment is time-shifted observation (Track 1, §1.1):
* **Free Ground-Truth Oracle:** Masking/predicting real data $z \sim P_{\text{real}}$ via $x = M(z)$, $y = \bar{M}(z)$ sits at the exact intersection of **Generative Asymmetry with zero pushforward distribution shift** ($M_\# P_{\text{real}}$ stays on the natural data manifold; Track 2, §1.2) and **Lossless Predictive Compression** ($L_{\text{bits}}(D \mid P) = -\log_2 P(D)$; Delétang et al., ICLR 2024, `arXiv:2309.10668` **[Established Literature]**).
* **Why Standard SSL Doesn't *Look* Like RL:** Because the 1-step distributional action $q_t = \pi_\theta(\cdot \mid x_{<t})$ is directly differentiable, teacher-forced cross-entropy computes an exact, Rao-Blackwellized policy gradient with zero sampling variance.
* **Where SSL Strictly Becomes RL:** The instant we insert an unobserved **latent reasoning trajectory** $z_t$ before predicting $y_t = x_{t:t+m-1}$, exact marginalization $\log \sum_{z_t} p_\theta(z_t \mid x_{<t})\,p_\theta(y_t \mid x_{<t}, z_t)$ becomes intractable, and its variational/EM gradient splits into a coupled REINFORCE/GRPO update over $z_t$ and a teacher-forced update over $y_t$ (Track 1, §1.2).

### 1.2 Why Real-World Domains Look "Unverifiable"—and What Actually Breaks
Leaving the SSL boundary exposes two engineering walls, neither of which implies epistemic unverifiability:
1. **Structural Asymmetry Wall (Track 2):** Real synthesis tasks (proving theorems, designing proteins, writing PDE solvers, auditing contracts) admit vast equivalence classes of valid solutions. Checking exact string equality against one corpus token fails; verification must exploit **Evaluative Asymmetry** ($C_{\text{verify}}(x,y,w) \ll C_{\text{solve}}(x)$ without gold labels $y^*$) or **Generative Asymmetry** (cheap forward/corruption maps $x = g(y)$ anchored to real artifacts so $g_\# Q \approx P_{\text{real}}$).
2. **Verification Latency Wall (Track 3):** Almost every domain called "unverifiable" (medicine, macroeconomics, materials science, literature) obeys an elementary epistemic test: *if we truly had no way to verify whether an answer was right, how did humanity ever come to know it was right?* What breaks standard on-policy RL (PPO/GRPO) is not unverifiability, but four **measurement bottlenecks** (Track 3, §1.1 & §2):
   * **Latency ($\tau$):** $10^5\times$–$10^{10}\times$ slower than math/code ($10^5$–$10^8\,\text{s}$ vs. milliseconds; 1,000 sequential 30-day updates take 82 years).
   * **Unit Cost & Irreversibility:** $\$10^1$–$\$10^8$ per physical/clinical check, with no `fork()` syscall to run $G=64$ counterfactual worlds in parallel.
   * **Stochastic Noise & Multi-Objectivity:** High exogenous variance $\sigma_{\text{env}}^2$, asymmetric FP/FN measurement error, and Pareto-incomparable criteria.

### 1.3 Quarantining the Genuine Epistemic Remainder
A rigorous framework must state clearly what **even infinite verification time cannot turn into an objective scalar reward** (Track 3, §1.2):
1. **Irreducible Normative Trade-Offs (Arrow's Impossibility Theorem):** Choosing between growth vs. equity or avant-garde difficulty vs. mass accessibility is a social value choice, not an empirical fact.
2. **Non-Repeatable Historical Counterfactuals:** Real history reveals $Y_t(a)$ for the action taken, never $Y_t(a')$ for the path foregone, without identifying causal assumptions.
3. **Reflexivity (Lucas Critique & Goodhart's Law):** Publishing a verifier or deploying an optimized policy shifts the strategic environment $P(Y \mid s, a)$ endogenously.
4. **Empirical Underdetermination (Duhem–Quine):** Distinct latent theories can yield identical observables over any finite experimental horizon.

> **Architectural Principle [Original Synthesis]:** Automate and compress the four engineering bottlenecks (latent depth, structural asymmetry, latency, cost) while explicitly quarantining the genuine remainder—exposing multi-objective Pareto frontiers and calibrated epistemic uncertainty to humans rather than hiding normative weights inside brittle scalar reward models.

---

## 2. The Unified Verification Architecture **[Original Synthesis]**

Tracks 1, 2, and 3 are not competing alternatives; they form one **continuous, latency-and-fidelity hierarchy** in which each lower tier dramatically shrinks the search space and sample complexity required by the tier above it, while higher tiers supply ground-truth anchors that recalibrate the proxies below:

```
                      UNIFIED UNIVERSAL VERIFICATION STACK
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: SLOW REALITY ANCHORS & EPISTEMIC OVERSIGHT   (days–yrs | $10²–$10⁶)  │
│  [Track 3] Closed-loop labs (orthogonal assay audits), live markets, RCTs,   │
│            delayed durability telemetry; Epistemic-leaf Debate / Peer Pred.  │
│        ▲                                                            │        │
│        │ Promote diverse portfolio via                              │ Drift  │
│        │ VoI / ($ · τ) + β · epistemic uncertainty                  │ audit &│
│        │                                                            ▼ recal. │
│ TIER 2: MULTI-FIDELITY SIMULATORS & DIVERSE SEARCH   (min–hrs | $10⁻¹–$10²)  │
│  [Tracks 2–3] DFT/MD, FEA/SPICE, co-adaptive econ RL (AI Economist),         │
│               Multi-Fidelity GFlowNets (p(x) ∝ R(x)), worst-case PVT corners │
│        ▲                                                            │        │
│ TIER 1: FAST SURROGATES, REPLAY & PROGRESS CRITICS   (0.1–10s | $10⁻⁴–$10⁻²) │
│  [Track 3] Chronological / quasi-experimental replay, world models (rbio1),  │
│            non-stationary rubrics (RaR/RcRL), programmatic judges (PAJAMA),  │
│            Process Advantage Verifiers (PAVs) & CCMV auto-decaying weights   │
│        ▲                                                            │        │
│ TIER 0B: MECHANICAL & STRUCTURAL ASYMMETRY GATES     (ms–s | ~$0, zero-FP)   │
│  [Track 2] Grounded inversions (SWE-smith/AZR), NP witnesses (CyberGym/Lean),│
│            conservation/PDE residuals (RLVP), duality bounds, frozen         │
│            round-trips + priors, metamorphic IPT, mechanical-leaf adversaries│
│        ▲                                                            │        │
│ TIER 0A: PREDICTIVE SSL-AS-RL & LATENT COMPUTE       (ms | $0 corpus-native) │
│  [Track 1] Latent-CoT predictive RL (continuous pretraining + sparse hybrid  │
│            RL on compute-elastic tokens) — trades temporal depth for params  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Tier 0A $\to$ Tier 0B: From Compute-for-Parameter SSL to Structural Gates
* **Why Tier 0A Matters for Small Models (Track 1, §3.1–3.2):** By circuit-complexity bounds (Feng et al., 2023, `arXiv:2305.15408`; Merrill & Sabharwal, ICLR 2024, `arXiv:2310.07923` **[Established Literature]**), bounded-depth transformers without scratchpads hit hard walls on serial arithmetic and state tracking, whereas constant-size transformers with polynomial CoT recognize $\textsf{P}$. Decomposing parameter capacity into **parametric memory ($N_{\text{mem}}$)** and **algorithmic depth ($N_{\text{comp}}$)** yields the **Parametric De-Congestion Hypothesis [Original Proposal]**: offloading serial computation into temporal latent steps during pretraining frees weights for world knowledge—explaining why **PonderLM-2-1.4B** beats vanilla **Pythia-2.8B** at matched inference cost (`arXiv:2509.23184`) and **Huginn-3.5B** scales toward ~50B-equivalent compute (`arXiv:2502.05171`), while **Snell et al.** (`arXiv:2408.03314`) show FLOPs-matched test-time scaling can beat a ~14× larger model when base support is non-trivial **[Established Literature]**.
* **Scaling Beyond Next-Token Likelihood (Track 2, §2–§4):** Tier 0B replaces token equality with **eight structural mechanisms**—(1) grounded real-artifact inversion (`SWE-smith`, `arXiv:2504.21798`, >50k tasks, 40.2% Pass@1 on SWE-bench Verified; `AZR`, `arXiv:2505.03335`), (2) NP witnesses & autoformalization (`CyberGym` PoVs; `Rtl2lean`, `2607.16855`; `CircuitProver`, `2607.27259`; `Law And Order`, `2610.02792`; `VPRMs`, `2601.17223`), (3) conservation laws, continuous PDE residuals (`RLVP`, `2607.10474`), and metamorphic invariance (`IPT`, `2604.15149`; `LLMORPH`, `2603.23611`), (4) primal–dual optimality gaps (`AlphaEvolve`, `2506.13131`), (5) grounded round-trips (`scRMSD`/`scTM`; `RTRL`, `2510.01527`; `AutoDecompiler`, `2606.16162`), (6) sandboxed execution (`FunSearch`, `EHRMind`, `LLMol`), (7) mechanical-leaf adversarial games, and (8) MDL compression (`DreamCoder`, `LILO`) **[Established Literature]**.

### 2.2 Tier 1 $\to$ Tier 3: Compressing Latency & Drawing the Clean Adversarial Boundary
* **Macro-Horizon Generalization of SSL-as-RL (Track 3, §3 Mech. 1):** Chronological forecasting RL (`Turtel et al.`, `2505.17989`: 14B model matches/exceeds `o1` with >10% simulated Polymarket ROI; `OpenForecaster`, `2512.25070` **[Established Literature]**) is literally the macro-horizon limit of Track 1: replacing next-millisecond tokens with next-quarter resolved world states scored by proper scoring rules.
* **Amortizing Slow Truth into Recalibrated Stacks (Track 3, §3 Mech. 2–4):** Biological world models (`rbio1`/RLEMF, `bioRxiv:10.1101/2025.08.18.670981`), structured non-stationary rubrics (`RaR`, `2507.17746`, +28–31% relative gain on HealthBench; `RcRL`, `2611.06244`), executable code judges (`PAJAMA`, `2607.22561`; `TIR-Judge`, `2510.23038`), and step-level Process Advantage Verifiers (`PAVs`, `2410.08146`) supply Tier 1 speed, while **Multi-Fidelity GFlowNets** (`2306.11715`) and autonomous closed-loop labs (`Periodic Labs`, `Lila Sciences`, `FutureHouse Robin`, `2505.13400`) close the physical loop **[Established Literature]**.
* **The Clean Adversarial Boundary (Track 2 vs. Track 3) [Original Synthesis]:** Prover–Verifier Games (`Kirchner et al.`, `2407.13692`) and adversarial challengers operate at two distinct strata separated by **how the leaf claim is adjudicated**:
  * **Track 2 (Tier 0B — Mechanical Leaf Checks):** The challenger bisects a complex artifact to a localized defect checked deterministically in $O(1)$—a Layer-2 fraud-proof CPU opcode, a PoV sanitizer crash, a failing unit test, or an executable contract-loophole trace. Persuasion bias is impossible at the leaf.
  * **Track 3 (Tier 3 — Epistemic Leaf Oversight):** Even after bisection, the leaf claim remains empirically uncertain, delayed, or value-laden—requiring **AI Debate** (`Khan et al.`, ICML 2024, `2402.06782`: 76% LLM / 88% human judge accuracy; `Brown-Cohen et al.`, `2311.14125`), **Peer Prediction** (`Qiu, Carroll, & Allen`, `2601.20299`: inverse scaling against deception where a 0.135B model supervises an 8B model), or **Conditional Prediction Markets** anchored to delayed reality settlement **[Established Literature]**.

### 2.3 Cross-Domain Instantiation of the Unified Stack **[Original Synthesis]**

Six canonical "unverifiable" domains from Tracks 2 & 3 instantiate the exact same four-tier blueprint:

| Domain | Tier 0B (Zero-FP Structural / Mechanical) | Tier 1 (Fast Surrogate / Replay / Rubric) | Tier 2 (Mechanistic Sim / Diverse Search) | Tier 3 (Slow Reality Anchor) | Irreducible Epistemic Remainder |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Medicine & Clinical Care** | Drug-interaction/dose bounds, `MedCalc-R1` formulas, VPRM guideline witnesses (`2601.17223`), red-flag challenger | Masked EHR inversion (stripped of clinician order-leakage), `RaR` clinical rubrics | Compartmental PK/PD ODE titration simulators | Prospective clinical trials & longitudinal patient survival | Novel unmodeled biology; quality-of-life preference weights |
| **Statutory Law & Contracts** | Autoformalized tax/covenant logic (`Law & Order`, `2610.02792`), `IPT` equivariance, executable Legal PoV loophole finder | Grounded contract corruption suites, `PAJAMA` programmatic compliance committees | Multi-scenario counterparty stress testing | Judicial precedent, regulatory enforcement & arbitration | Open-textured standards ("reasonable"); societal norm trade-offs |
| **Physical Eng. & PDEs** | Mesh/DRC/LVS checks, conservation laws & `RLVP` PDE residuals (`2607.10474`), Lean RTL proofs (`Rtl2lean`), CLF/CBF bounds | Fast neural PDE/surrogate scorers + grounded CAD/circuit repair inversions | High-fidelity FEA/CFD, SPICE transient, worst-case PVT/sim-to-real adversary | Physical prototyping, silicon tape-out & wear testing | Unmodeled fatigue/environmental contact dynamics |
| **Economic Systems** | Accounting/flow-of-funds identities, no-arbitrage & constitutional constraints | Chronological forecast replay (`2505.17989`) + quasi-experimental AIPW natural-experiment replay | Co-adaptive multi-agent RL (`The AI Economist`, 2022) across behavioral ensembles | Conditional prediction markets + Debate; staggered regional RCTs | Normative Pareto weights (growth vs. equity); Lucas regime breaks |
| **Scientific Discovery** | Valence/stoichiometry, thermodynamics, synthesizability & PAINS filters | Virtual Cell world models (`rbio1`/RLEMF, RLPK) + UWO uncertainty penalties | DFT/MD/FEP physics + Multi-Fidelity GFlowNets preserving mode diversity | Autonomous closed-loop labs (`Periodic`/`Lila`/`Robin`) + blind orthogonal assay audits | Novel mechanisms outside support; human Phase II/III translational cliff |
| **Novels & Creative Writing** | Continuity/entity/timeline/POV checks (`RLMR`, `2508.18642`) | Non-stationary `RcRL` rubrics (`2611.06244`), pairwise `Writing-Zero`/`Writing-RL`, judge-free `SpyRL` (`2607.23802`) | Persona-diverse reader panels + adversarial literary cross-examination | Serialized reader retention (CCMV auto-decaying) anchored to delayed re-reads & reviews | Plural evolving aesthetic taste; paradigm-breaking novelty (pastiche paradox) |

---

## 3. Four Cross-Cutting Mathematical Primitives **[Established Literature + Original Synthesis]**

Despite spanning twelve orders of magnitude in latency, Tracks 1–3 repeatedly rediscover the **exact same four mathematical invariants**. Ignoring any one of them causes RL to collapse.

### 3.1 Strictly Proper Scoring Rules & Dense Progress vs. Binary Pass/Fail Collapse
Binary exact-match or $\{0,1\}$ pass/fail rewards systematically fail outside deterministic puzzles:
* In **Track 1**, exact-match $\mathbb{I}[\hat{x}_t = x_t]$ (`RPT`) is *not* strictly proper—it collapses predictive distributions onto the single mode $\arg\max_v p_{\text{data}}(v \mid x_{<t})$ and creates zero-variance ($\sigma=0$) batches on hard tokens. Conversely, unbounded log-likelihood suffers catastrophic $-\infty$ tail spikes on rare tokens. **Bounded strictly proper scoring rules**—specifically token-level **Brier score** (`R3`, Track 1 §1.3 & §5.4) or clipped log-likelihood gain (`R2`)—eliminate tail blowups while preserving calibration:
  $$\text{BS}(q, v^*) = 2\,q(v^*) - \sum_{v \in \mathcal{V}} q(v)^2 - 1 \in [-2, 0]$$
  trading weaker cold-start slope near $p \approx 0$ against bounded variance and late-stage stability.
* In **Track 2**, continuous PDE residuals (`RLVP`, `2607.10474`) and primal–dual gaps $\varepsilon(y,\lambda) = p(y) - d(\lambda)$ replace sparse binary feedback with smooth physical/optimization landscapes.
* In **Track 3**, binary RLVR trains overconfident bluffers; **RLCR** (`Damani et al.`, `2507.16806`) embeds a Brier-score penalty over verbalized confidence directly into RLVR rewards, while **Process Advantage Verifiers** (`Setlur et al.`, `2410.08146`) prove step rewards must measure *marginal progress* rather than raw success probability:
  $$A^\mu(s_t, a_t) = Q^\mu(s_t, a_t) - V^\mu(s_t)$$

### 3.2 Asymmetric Noise Geometry ($\text{FP} \gg \text{FN}$) & Mode-Diversity Preservation
Early optimism that RLVR tolerates heavy reward noise was overturned in 2025–2026 **[Established Literature]**:
* **Zhu & Kang (`2603.16140`)** proved that prior "noise-tolerant RLVR" claims stemmed from dataset contamination; genuinely noisy labels degrade math accuracy by **8–10%** and Text2SQL by **5–12%**.
* **Cai et al. (`2510.00915`)** formalized imperfect verifiers as an asymmetric channel: **false negatives (FNs)** slow learning by dropping valid rollouts, whereas **false positives (FPs)** actively reward invalid shortcuts and trigger Goodhart collapse—requiring backward/forward channel corrections and secondary verifier appeals.
* **Universal Manifestation:** The identical FP asymmetry appears across all three tracks:
  * *Track 1:* Weak small-model implicit verifiers reward lexical n-gram priming shortcuts unless checked across multi-token horizons ($m \ge 4$).
  * *Track 2:* **Hadamard ill-posedness** in continuous inverse problems lets unphysical, high-frequency solutions satisfy $\|g(y)-x\|\le\varepsilon$, requiring explicit physical priors $\Omega_{\text{prior}}(y)$. Meanwhile, naively multiplying $m$ hard zero-FP gates causes cold-start $\sigma=0$ gradient collapse—fixed by **lexicographic staged gating** (Transform 3):
    $$R(x, y) = \frac{\lambda_{\text{gate}}}{m}\sum_{j=1}^m s_j(x, y) \;+\; \prod_{j=1}^m \mathbb{I}[I_j(x, y)] \cdot \Big(\alpha\, R_{\text{residual}}(x, y) + (1-\alpha)\,\mathbb{E}_{T}[\text{Inv}(f(x), f(T(x)))]\Big)$$
  * *Track 3:* Even autonomous physical labs (**A-Lab**, *PRX Energy* 2024 critique & *Nature* Author Correction) face characterization-layer FPs when automated XRD phase-matching scripts go unaudited inside closed loops.
* **Diversity Defense:** Under imperfect proxies, argmax RL collapses onto the single worst FP peak. Sampling proportionally to reward via **Multi-Fidelity GFlowNets** (`2306.11715`) or Thompson sampling preserves diverse modes across expensive verification tiers.

### 3.3 Anchor & Stop-Gradient Defenses Against Co-Adaptation & Goodharting
Whenever the policy and verifier share parameters, co-train, or optimize indefinitely against a static proxy, optimization manufactures fake reward:
* **Track 1 (Shared-$\theta$ Baseline Sabotage):** Because the same weights $\theta$ parameterize both the thought policy $p_\theta(z_t \mid x_{<t})$ and implicit verifier $p_\theta(y_t \mid x_{<t}, z_t)$, subtracting a live no-thought baseline lets gradient descent inflate advantage by *degrading* $p_\theta(y_t \mid x_{<t}, \varnothing)$. Safe training requires evaluating counterfactual gain against a **stop-gradiented, frozen reference anchor** plus an explicit auxiliary no-thought NTP loss $\lambda_{\text{ntp}} \mathcal{L}_{\text{NTP}}(\theta)$ (Track 1, §1.2 & §4.2):
  $$R_{\text{gain}}(x_{<t}, z_t, y_t) = \frac{1}{m}\sum_{k=0}^{m-1} \operatorname{sg}\!\Big[\text{score}\big(p_\theta(\cdot \mid x_{<t}, z_t, x_{t:t+k-1}),\, x_{t+k}\big) - \text{score}\big(p_{\theta_{\text{ref}}}(\cdot \mid x_{<t}, x_{t:t+k-1}),\, x_{t+k}\big)\Big]$$
* **Track 2 (Steganographic Cycle Collusion):** Co-training forward and reverse neural maps $(f_\theta, g_\phi)$ collapses into uninterpretable private codes; defenses freeze/alternate $g$ (`RTRL`, `2510.01527`) or anchor $g$ to a deterministic compiler/physics engine.
* **Track 3 (Goodhart Scaling & Surrogate TTL):** **Skalse et al.** (`2209.13085`) prove unhackable proxies almost never exist; **Gao et al.** (`2210.10760`) establish predictable KL-overoptimization collapse curves. Defenses treat every surrogate as a decaying local Taylor approximation with a **distributional TTL**: ensemble **WCO/UWO** penalties (`Coste et al.`, ICLR 2024, `2310.02743`), **KL-Gated Anchor-and-Drift recalibration** augmented with **out-of-band randomized spot-audits** (catching low-KL stealth Goodharting), and **Causal-Chain Milestone Verification (CCMV)** that auto-decays leading-indicator weights $w_j \to 0$ whenever Prentice-criterion audits detect divergence from terminal reality (e.g., cliffhanger engagement-Goodharting in serialized fiction or biomarker failures like the CAST trial).

### 3.4 Information-Theoretic Compression, MDL & Compute Elasticity
Verification and reasoning compute must be allocated strictly to **reducible epistemic/computational uncertainty**, never wasted on predictable syntax or irreducible aleatoric noise:
* **Track 1:** ~80% of tokens are predictable low-entropy syntax, while ~20% high-entropy forking tokens drive RLVR gains (`Wang et al.`, NeurIPS 2025, `2506.01939`). Because total predictive uncertainty decomposes as:
  $$H(X_t \mid X_{<t}) = H_{\text{aleatoric}}(X_t \mid X_{<t}) + H_{\text{epistemic/comp}}(X_t \mid X_{<t})$$
  raw entropy filtering conflates arbitrary proper nouns with multi-step deductions. Track 1 therefore targets **compute-elastic tokens** via teacher-free within-group variance (`T2-self`, $\sigma > 0$) or cross-scale likelihood gap $\Delta^{\text{scale}}_t$ (Track 1, §4.1).
* **Track 2:** Mechanism 8 formalizes structural discovery as two-part MDL minimization $L_{\text{total}}(h, D) = L_{\text{model}}(h) + L_{\text{data}}(D \mid h)$ (`DreamCoder`, `2006.08381`; `LILO`, `2310.19791`), linking symbolic library compression directly to SSL's arithmetic coding (`2309.10668`).
* **Track 3:** Tier promotion routes scarce physical/human checks by **Value of Information per dollar-day**:
  $$\text{Route}(x) \propto \frac{\text{Expected Policy Improvement}(x) + \beta \cdot \text{Epistemic Uncertainty}(x)}{\text{Cost}_{i+1} \cdot \tau_{i+1}}$$
  deliberately spending Tier 3 budget to falsify overconfident surrogate regions before Goodhart collapse sets in.

### 3.5 Cross-Primitive Summary Matrix

| Primitive | Core Failure Mode If Ignored | Mandatory Cross-Track Defense |
| :--- | :--- | :--- |
| **Proper Scoring / Progress** | Mode collapse (`RPT`), bluffing (binary RLVR), $-\infty$ tail variance | Bounded strictly proper scores (Brier / `RLCR`), continuous residuals (`RLVP`), progress advantages (`PAVs`) |
| **Asymmetric Noise ($\text{FP}\gg\text{FN}$)** | Reward hacking, Hadamard unphysical inverses, lab script FPs, $\sigma=0$ stalls | Forward/backward noise correction (`2510.00915`), $\Omega_{\text{prior}}(y)$, lexicographic gating, GFlowNet diversity |
| **Immutable Anchors & TTL** | Shared-$\theta$ baseline sabotage, steganographic cycle collapse, KL Goodharting | `sg` frozen reference $p_{\theta_{\text{ref}}}$ + $\lambda_{\text{ntp}}$, frozen/grounded $g$, WCO/UWO + Anchor-and-Drift + spot-audits + CCMV |
| **Compression & Compute Elasticity** | Wasting FLOPs on 80% syntax or fitting irreducible aleatoric noise | `T2-self` / top-20% forking-token updates, prequential MDL, $\text{VoI}/(\$\cdot\tau)$ multi-fidelity routing |

---

## 4. Three Deep Open Tensions & Falsifiable Hypotheses **[Original Synthesis / Proposal]**

Where Tracks 1, 2, and 3 intersect, **three fundamental contradictions** remain unsolved by 2026 literature. Resolving them defines the core scientific agenda of universal verification.

### Tension 1: Cold-Start Support vs. On-Policy Cross-Domain Transfer
* **The Collision:** Track 1 documents that pure discrete-token RL on raw small base models stalls at the **cold-start support boundary**: `DeepSeek-R1-Zero-Qwen-32B` (47.0% AIME 2024 `pass@1`, 60.0% `cons@64`, 91.6% MATH-500 `pass@1`) consistently trails simple teacher SFT distillation (`DeepSeek-R1-Distill-Qwen-32B`: 72.6% AIME `pass@1`, 83.3% `cons@64`, 94.3% MATH-500 `pass@1`; `2501.12948` Sec. 4.1), `RPT` (`2506.08007`) relied on an already-distilled 14B initialization, and `Yue et al.` (`2504.13837`) show RLVR shrinks `pass@k` at large $k$. This suggests using off-policy teacher distillation to warm-start small models. Yet Track 3 cites **Huan et al. (`2507.00432`)**, proving that **off-policy SFT distillation forces severe latent representation drift and destroys cross-domain transfer**, whereas strictly *on-policy RL* preserves general latent geometry and transfers positively!
* **Falsifiable Hypothesis H1 (The Continuous-Pretraining $\to$ On-Policy Hybrid Bridge):** *Pretraining a small model with differentiable continuous latent computation (`PonderLM-2`/`Huginn` style—eliminating discrete REINFORCE cold-start variance) followed by strictly on-policy hybrid RL (`HRPO`, `2505.18454`) on teacher-free compute-elastic tokens (`T2-self`) simultaneously breaks through the cold-start support wall AND preserves cross-domain transfer geometry.*
* **Decisive Test & Kill Criterion:** On matched small backbones (`160M` from-scratch; `Qwen2.5-0.5B/1.5B-Base`), evaluate (a) cold-start discrete RL, (b) off-policy 7B-distilled SFT + RL, and (c) continuous latent pretraining + on-policy hybrid RL (`I2`). **Confirm H1** iff (c) strictly outperforms (a) on OOD algorithmic `pass@64` ($d=6, 8$) *and* strictly outperforms (b) on non-target transfer domains (coding, general QA, PCA latent preservation) at matched inference FLOPs. **Falsify H1** if (c) fails either inequality.

### Tension 2: Correlated Priors Breaking Ensemble Goodhart Defenses
* **The Collision:** Track 2 (Mechanism 5 learned round-trips) and Track 3 (Mechanism 2 WCO/UWO ensembles, `2310.02743`; Mechanism 5 Bayesian Truth Serum) assume diverse evaluators fail independently. Yet **Shuo Li Liu (2026)** showed that frontier LLM panels share overlapping web pretraining corpora—violating conditional independence so badly that BTS underperforms plain majority vote in raw accuracy (0.9724 vs. 0.9793, though cutting binned ECE by 21–32%). This exposes a lethal vulnerability: **low-KL stealth Goodharting** (Track 3, §3 Mech. 2), where a policy flips a few high-leverage tokens that exploit shared-prior blind spots across an *entire* LLM ensemble while keeping both token-level $D_{\text{KL}}$ and ensemble variance $\sigma_k$ deceptively low.
* **Falsifiable Hypothesis H2 (Cross-Modal Structural Decorrelation + Inverse-Scaling Oversight):** *Neutralizing low-KL stealth Goodharting requires orthogonalizing verifier modalities—combining (i) deterministic Track 2 checks (compiled programmatic committees via `PAJAMA`, `2607.22561` + metamorphic `IPT`, `2604.15149`), (ii) pretraining/retrieval-isolated neural surrogates, and (iii) Peer Prediction scores (which exhibit inverse scaling against deception, `2601.20299`) paired with out-of-band randomized ground-truth spot-audits.*
* **Decisive Test & Kill Criterion:** On an open-ended synthesis benchmark with known hidden ground truth and injected decoy traps, run GRPO KL-sweeps (`Gao et al.` protocol) comparing (a) a $K$-model same-corpus LLM WCO/UWO ensemble against (b) the modality-decorrelated composite stack + 2% randomized spot-audits. **Confirm H2** iff (b) reduces stealth-Goodhart FP escape rate by $\ge 50\%$ at matched gold reward and flags drift before gold reward turns negative. **Falsify H2** if (b) matches (a)'s overoptimization collapse curve.

### Tension 3: The Causal Counterfactual Ceiling (Passive Replay vs. Active Intervention)
* **The Collision:** Historical replay (Track 3 Mech. 1: `Turtel et al.`, `OpenForecaster`) and grounded record inversion (Track 2 Blueprint A: masked EHR trajectories) compress latency by $10^6\times$ for **passive prediction**. Both break structurally under **active intervention**: archived futures reflect historical actions $a^{\text{hist}}$, suffer from **clinician ordering leakage** (lab orders telegraphing diagnoses in EHRs) and **correlated macro-shock sample-size ceilings** (tens of thousands of markets driven by a few global shocks), and cannot observe counterfactual outcomes $Y(a')$ or strategic Lucas best-responses.
* **Falsifiable Hypothesis H3 (Quasi-Experimental Replay + Co-Adaptive Simulation Quantifies an Irreducible Causal Latency Floor):** *Replacing naive observational replay with propensity-logged natural-experiment / staggered-rollout replay scored by doubly robust (AIPW) estimators—paired with co-adaptive multi-agent world models (`The AI Economist` style)—eliminates ordering-leakage and confounding collapse inside historical propensity support, while exposing a quantifiable minimum Tier 3 interventional budget required whenever policies push outside support.*
* **Decisive Test & Kill Criterion:** Build a structural causal testbed with known ground-truth dynamics, clinician ordering bias / strategic co-adaptation, and logged propensities. Compare (a) naive masked observational replay, (b) propensity-weighted AIPW quasi-experimental replay, and (c) AIPW replay + co-adaptive Tier 2 simulator + budgeted Tier 3 randomized micro-pilots. **Confirm H3** iff (b) eliminates (a)'s confounding collapse in-support and (c) achieves strictly superior out-of-support interventional lift while tracing a smooth Pareto curve between distributional shift and required Tier 3 sample budget. **Falsify H3** if (b) fails to beat (a) or (c) matches (b) out-of-support.

---

## 5. Concrete Research & Experiment Roadmap **[Original Proposal]**

We prioritize execution from lowest-cost foundational falsification to full cross-domain stack validation, gating each phase on explicit quantitative criteria:

| Phase | Concrete Deliverable | Compute / Cost | Targets & Hypotheses | Explicit Go / No-Go Gate | Source Protocol |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Small-Model Latent-CoT Pilot** *(Immediate)* | From-scratch `160M` vs. `410M` (`2.5×`) & `1.0B` (`6×`) NTP baselines on 2B tokens stratified across *Algorithmic* (ProsQA DAGs $d\in\{2,4,6,8\}$, Python trace, modular arithmetic), *Natural Web* (OpenWebMath/Stack/FineWeb-Edu), and *Aleatoric Negative Control* (TriviaQA). Runs `B1`–`B4` + core teacher-free RL arm (`T2-self` + frozen-anchor `R2`/`R3`). | **~30–45 H100 GPU-hrs** (1–4× H100 weekend) | Parametric De-Congestion; Knowledge–Algorithm split; Brier (`R3`) vs. Log (`R2`) vs. EM (`R1`); swap & truncation faithfulness | **Go:** `160M` + latent CoT matches/beats `410M` at $\le 1.0\times$ inference FLOPs, beats pause baseline `B3` by $\ge 3$ pp on algorithmic tasks, passes $\Delta\text{NLL}_{\text{swap}} > 0.15$ nats + format-matched $\Delta\text{NLL}_{\text{trunc}} > 0$ + monotonic OOD depth scaling, and raises `pass@64` over base. | [Track 1 §5](file:///home/spinoza/github/big-probs/01-ssl-as-rl-latent-cot.md#L232-L313) |
| **Phase 2: Factorized Matrix & H1 Transfer Audit** *(Near-Term)* | Continued-pretraining RL on `Qwen2.5-0.5B/1.5B-Base` vs. `3B/7B-Base` (50M–100M filtered tokens, $G=8$, Dr. GRPO + DAPO Clip-Higher $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$, $\lambda_{\text{ntp}}=0.2$). Full factorized ablation (`T0`/`T1`/`T2-self`/`T2-scale` × `R1`–`R3` × `I1` discrete vs. `I2` continuous$\to$hybrid HRPO) + off-policy distillation `B5`. | **~250–400 H100 GPU-hrs** (~2 days on 8× H100) | **Tension 1 / H1** (Cold-start support vs. on-policy transfer geometry); teacher-free vs. cross-scale filter lift | **Go:** Resolves cold-start threshold at 0.5B–1.5B scale and tests whether `I2` strictly dominates both raw discrete RL (on OOD `pass@64`) and off-policy SFT `B5` (on cross-domain transfer & PCA latent preservation). | [Track 1 §4.4 & §5.4](file:///home/spinoza/github/big-probs/01-ssl-as-rl-latent-cot.md#L223-L295) |
| **Phase 3: Cross-Domain Tier 0B–2 Stack Prototype** *(Medium-Term)* | Build an end-to-end composite verification stack outside math/code—targeting **(a) Scientific PDE Solver Synthesis** (`RLVP` residuals + `IPT` + grounded inversion) or **(b) Executable Contract/Tax PoV Auditing** (autoformalized rules + Legal PoV loophole finder + `RcRL` rubrics)—paired with `PAJAMA` programmatic committees + Peer Prediction + 2% randomized spot-audits. | **~500–1,000 H100 GPU-hrs** *(illustrative estimate; in-silico / sandbox only)* | **Tension 2 / H2** (Cross-modal stealth-Goodhart defense); lexicographic staged gating vs. $\sigma=0$ collapse | **Go:** Sustains rising held-out ground-truth accuracy over $10^3+$ RL steps while cutting stealth-Goodhart FP escape rate $\ge 50\%$ vs. same-corpus LLM WCO ensembles at matched KL. | [Track 2 §4](file:///home/spinoza/github/big-probs/02-verification-asymmetry-and-inversion.md#L247-L323); [Track 3 §3 Mech. 2](file:///home/spinoza/github/big-probs/03-slow-and-unknown-verification.md#L117-L143) |
| **Phase 4: Latency-Compressed Tier 3 Reality Loop** *(Long-Term)* | Validate H3 on a causal interventional simulator first (AIPW quasi-experimental replay + co-adaptive Tier 2), then couple the Tier 0–2 stack to a live slow anchor: either **contamination-free live forecasting** (`ForecastBench`-style + Debate/Peer Prediction) or an **autonomous lab/assay loop** ($\text{VoI}/(\$\cdot\tau)$ GFlowNet routing + CCMV auto-decaying milestones + blind orthogonal characterization audits). | Domain-specific (live market or lab partner) | **Tension 3 / H3** (Causal latency floor); Anchor-and-Drift TTL; CCMV Prentice-criterion auto-decay | **Go:** Demonstrates compounding real-world lift across $\ge 3$ slow recalibration epochs, automatic CCMV decay of a Goodharted proxy milestone, and higher real hit-rate per dollar-day than naive top-$K$ promotion. | [Track 3 §3–§4](file:///home/spinoza/github/big-probs/03-slow-and-unknown-verification.md#L96-L291) |

---

## 6. Deep-Dive Navigation Index

| Research Question / Component | Deep-Dive Location |
| :--- | :--- |
| Formal SSL $\leftrightarrow$ RL EM identity, shared-$\theta$ baseline sabotage, & 7-reward taxonomy | [01-ssl-as-rl-latent-cot.md §1](file:///home/spinoza/github/big-probs/01-ssl-as-rl-latent-cot.md#L19-L79) |
| 2022–2026 SOTA lineages (Quiet-STaR, RPT, VeriFree, RLPR, Coconut, Huginn, PonderLM-2, HRPO) & counter-evidence | [01-ssl-as-rl-latent-cot.md §2](file:///home/spinoza/github/big-probs/01-ssl-as-rl-latent-cot.md#L82-L142) |
| Knowledge–Algorithm Decomposition, 5 small-model failure modes, & epistemic vs. aleatoric filtering | [01-ssl-as-rl-latent-cot.md §3–§4](file:///home/spinoza/github/big-probs/01-ssl-as-rl-latent-cot.md#L145-L229) |
| Full Stage 1 & Stage 2 small-model experimental protocols, faithfulness checks, & falsification criteria | [01-ssl-as-rl-latent-cot.md §5](file:///home/spinoza/github/big-probs/01-ssl-as-rl-latent-cot.md#L232-L313) |
| Evaluative vs. Generative Asymmetry, Pushforward Shift Theorem, & 8-Mechanism Taxonomy (37 citations) | [02-verification-asymmetry-and-inversion.md §1–§3](file:///home/spinoza/github/big-probs/02-verification-asymmetry-and-inversion.md#L19-L244) |
| 6-Transform Verification Playbook (lexicographic gating, Hadamard priors) & Medicine/Law/Engineering blueprints | [02-verification-asymmetry-and-inversion.md §4](file:///home/spinoza/github/big-probs/02-verification-asymmetry-and-inversion.md#L247-L323) |
| Four pseudo-unverifiable regimes, 4 genuine epistemic remainders, latency wall, & noisy RLVR (`2603.16140`, `2510.00915`) | [03-slow-and-unknown-verification.md §1–§2](file:///home/spinoza/github/big-probs/03-slow-and-unknown-verification.md#L7-L70) |
| Six latency-compression mechanisms (Replay, Surrogates/Goodhart, GFlowNet Funnels, PAVs/CCMV, Debate/BTS, Transfer) | [03-slow-and-unknown-verification.md §3](file:///home/spinoza/github/big-probs/03-slow-and-unknown-verification.md#L73-L234) |
| Worked slow-domain stacks (Economics, Science, Novels) & full Established vs. Proposed epistemic audit | [03-slow-and-unknown-verification.md §4–§5](file:///home/spinoza/github/big-probs/03-slow-and-unknown-verification.md#L237-L320) |
