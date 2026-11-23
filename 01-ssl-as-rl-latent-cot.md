# Track 1: Framing Self-Supervised Learning as Reinforcement Learning with Latent Chain-of-Thought

> **Core Question:** If self-supervised next-token prediction (NTP) is reframed as a reinforcement learning (RL) problem with an automatic verifier—allowing a model to execute unsupervised *latent* reasoning before emitting its prediction—can much smaller models punch far above their parameter weight class by trading test-time / latent compute for parameter count?

> **Epistemic Convention:** Throughout this report, claims grounded in published or preprint literature (2022–2026) are marked as **[Established Literature]** with verified citations, while theoretical syntheses, taxonomies, diagnostic criteria, and experimental proposals developed here are explicitly marked as **[Original Synthesis / Proposal]**.

---

## Executive Summary

1. **Formal Equivalence and Its Breaking Point:** Standard self-supervised learning (SSL) is mathematically a degenerate, single-step contextual bandit with an automatic oracle verifier (the observed next token $x_t$) and a strictly proper reward (log-likelihood). Standard pretraining does not *look* like RL only because the action space is directly differentiable: teacher forcing computes the exact Rao-Blackwellized gradient over the entire vocabulary with zero sampling variance. The moment we insert an *unobserved* latent reasoning variable $z_t$ before predicting $x_t$, exact marginalization $\log \sum_{z_t} p_\theta(z_t \mid x_{<t})\,p_\theta(x_t \mid x_{<t}, z_t)$ becomes intractable, and SSL *genuinely becomes* an RL / stochastic variational inference problem.
2. **State of the Art (2024–2026):** Four lines of work have converged on this exact interface: (a) *discrete-token latent RL pretraining* (Quiet-STaR, RPT), (b) *verifier-free likelihood/intrinsic RL* (VeriFree, RLPR, Intuitor/RLIF, RLCR), (c) *continuous latent thought & recurrent depth* (Coconut, Huginn, PonderLM-2, HRPO), and (d) *critical diagnostic studies* showing that current RLVR primarily sharpens `pass@1` within a base model's existing support rather than expanding `pass@k` (Yue et al., 2025; DeepSeek-R1 Sec. 4.1).
3. **Verdict on the Small-Model Hypothesis (Conditional Yes, with Sharp Boundaries):** Latent-CoT RL *can* let small models punch 2×–14× above their parameter weight class—**specifically on depth-limited algorithmic, compositional, and planning tasks** (where bounded-depth transformers hit hard circuit-complexity walls, whereas constant-size models with scratchpads reach $\textsf{P}$). Conversely, latent compute **cannot substitute for missing parametric world knowledge**, and direct discrete-token RL on raw small base models frequently fails due to the **cold-start support boundary** (near-zero probability of sampling a winning trace), **implicit-verifier weakness**, and **token-level credit-assignment noise**.
4. **The High-Leverage Bottleneck:** Most tokens (~80%) are syntactic/predictable, and a large fraction of the remaining high-entropy tokens reflect *irreducible aleatoric uncertainty* (e.g., an arbitrary proper noun or stylistic choice) rather than *reducible computational uncertainty*. Naive entropy filtering conflates the two. Progress hinges on targeting **compute-elastic tokens** (via cross-scale likelihood gaps or multi-token predictive gain), using **bounded strictly proper rewards** (e.g., debiased Brier or clipped log-likelihood gain) to curb variance and mode collapse, and bridging continuous latent pretraining (for differentiable compute-for-parameter de-congestion) with stochastic RL exploration.
5. **Actionable Testbed:** Section 5 specifies a self-contained, FLOPs-matched experimental protocol with itemized H100 compute budgets (~30–45 H100 GPU-hours for the minimal core pilot; ~250–400 H100 GPU-hours for the full factorized matrix and Track B) designed to falsify or confirm the small-model hypothesis across algorithmic, natural-text, and aleatoric negative-control regimes.

---

## 1. Formal Framing: SSL as a Degenerate Case of RL, and What Latent CoT Changes

### 1.1 Standard Next-Token Prediction as a Degenerate 1-Step MDP **[Established Literature + Original Synthesis]**

Let a corpus define a data distribution over token sequences $x_{1:T} \in \mathcal{V}^T$. Standard autoregressive SSL can be cast as a 1-step contextual bandit (a finite-horizon MDP with horizon $H=1$) at every position $t$:

* **State:** Prefix context $s_t = x_{<t}$.
* **Policy / Action:** Either (i) a discrete token action $\hat{x}_t \sim \pi_\theta(\cdot \mid x_{<t})$, or (ii) a distributional action $q_t = \pi_\theta(\cdot \mid x_{<t}) \in \Delta(\mathcal{V})$.
* **Automatic Verifier:** The corpus itself reveals ground truth $x_t$ at zero external labeling cost—verification is *tautologically free* because the environment is time-shifted observation.
* **Reward:**
  * Under a discrete action, exact-match verification gives sparse binary reward $R_{\text{EM}}(\hat{x}_t, x_t) = \mathbb{I}[\hat{x}_t = x_t]$.
  * Under a distributional action, a *scoring rule* $S(q_t, x_t)$ grades the entire predictive distribution—most canonically the logarithmic score $S_{\log}(q_t, x_t) = \log \pi_\theta(x_t \mid x_{<t})$.

Why does standard SSL never look or behave like RL? For two structural reasons:

1. **Analytic Expectation (Zero Sampling Variance):** Because $S_{\log}(\pi_\theta(\cdot \mid x_{<t}), x_t)$ is directly differentiable with respect to $\theta$ without sampling $\hat{x}_t$, maximum-likelihood pretraining performs exact gradient ascent on $\log \pi_\theta(x_t \mid x_{<t})$. In RL terms, teacher-forced cross-entropy is a *fully Rao-Blackwellized, exact-model policy gradient* over a 1-step horizon.
2. **Strict Propriety:** Because $S_{\log}$ is a *strictly proper scoring rule*, its unique maximizer in expectation over $x_t \sim p_{\text{data}}(\cdot \mid x_{<t})$ is the true conditional distribution $\pi^*(\cdot \mid x_{<t}) = p_{\text{data}}(\cdot \mid x_{<t})$. By contrast, maximizing expected exact-match reward $\mathbb{E}_{\hat{x}_t \sim \pi_\theta}[\mathbb{I}[\hat{x}_t = x_t]]$ is *not* strictly proper: its optimum collapses all probability mass onto the single mode $\arg\max_v p_{\text{data}}(v \mid x_{<t})$, destroying distributional calibration.

### 1.2 Inserting Unsupervised Latent Actions $z_t$: Where SSL Strictly Becomes RL **[Established Literature]**

Now suppose that between observing $x_{<t}$ and predicting a future window $y_t = x_{t:t+m-1}$ (where $m \ge 1$), the model generates an *unobserved* latent reasoning trajectory $z_t$ (either discrete tokens $z_t \in \mathcal{V}^L$ or continuous vectors $z_t \in \mathbb{R}^{L \times d}$). The true marginal log-likelihood of the observed continuation is:

$$\mathcal{J}_{\text{marg}}(\theta) = \log p_\theta(y_t \mid x_{<t}) = \log \sum_{z_t} p_\theta(z_t \mid x_{<t})\, p_\theta(y_t \mid x_{<t}, z_t)$$

Because $z_t$ is never supervised in the corpus and the summation over $z_t$ is exponential ($|\mathcal{V}|^L$), exact marginalization is intractable. Taking the gradient of $\mathcal{J}_{\text{marg}}(\theta)$ yields the classic **Latent-Variable / Expectation-Maximization (EM) identity** (cf. Phan et al., 2023, `arXiv:2312.02179`):

$$\nabla_\theta \log p_\theta(y_t \mid x_{<t}) = \mathbb{E}_{z_t \sim p_\theta(z_t \mid x_{<t}, y_t)} \Big[ \nabla_\theta \log p_\theta(z_t \mid x_{<t}) + \nabla_\theta \log p_\theta(y_t \mid x_{<t}, z_t) \Big]$$

Notice the catch: the expectation is under the **intractable posterior** $p_\theta(z_t \mid x_{<t}, y_t) \propto p_\theta(z_t \mid x_{<t})\,p_\theta(y_t \mid x_{<t}, z_t)$, which depends on the future answer $y_t$. Different families of algorithms correspond to different approximations of this single identity:

1. **Prior-Sampled RL / Variational Lower Bound (ELBO with $q_\phi = p_\theta$):**
   By Jensen's inequality, for any variational distribution $q(z_t \mid x_{<t})$:
   $$\log p_\theta(y_t \mid x_{<t}) \ge \mathbb{E}_{z_t \sim q} \big[ \log p_\theta(y_t \mid x_{<t}, z_t) \big] - D_{\text{KL}}\big(q(z_t \mid x_{<t}) \,\|\, p_\theta(z_t \mid x_{<t})\big)$$
   When the amortized proposal *is* the on-policy thought generator $q = p_\theta(\cdot \mid x_{<t})$—which matches test-time inference where $y_t$ is unseen—the KL term vanishes and we obtain the **on-policy RL objective with log-likelihood reward**:
   $$\mathcal{J}_{\text{RL-LL}}(\theta) = \mathbb{E}_{z_t \sim p_\theta(\cdot \mid x_{<t})} \big[ \underbrace{\log p_\theta(y_t \mid x_{<t}, z_t)}_{R(x_{<t}, z_t, y_t)} \big]$$
   Its score-function gradient splits cleanly into two coupled terms:
   $$\nabla_\theta \mathcal{J}_{\text{RL-LL}}(\theta) = \mathbb{E}_{z_t \sim p_\theta} \Big[ \underbrace{R(x_{<t}, z_t, y_t)\,\nabla_\theta \log p_\theta(z_t \mid x_{<t})}_{\text{REINFORCE / GRPO on latent CoT } z_t} + \underbrace{\nabla_\theta \log p_\theta(y_t \mid x_{<t}, z_t)}_{\text{Teacher-forced SFT on } y_t \text{ given } z_t} \Big]$$
   This single equation (central to Quiet-STaR and VeriFree's unified policy/implicit-verifier framing) exposes why SSL-with-latent-CoT is structurally richer—and harder—than both standard SSL and standard RLVR: **the same parameters $\theta$ simultaneously act as the reasoning policy $p_\theta(z_t \mid x_{<t})$ and the implicit reward model $p_\theta(y_t \mid x_{<t}, z_t)$**. Crucially, this shared parameterization introduces a **co-adaptation and baseline-sabotage hazard [Original Synthesis]**: if a debiased reward subtracts an unanchored live no-thought baseline $p_\theta(y_t \mid x_{<t})$ without `stop_gradient` or frozen-reference anchoring, gradient updates can trivially inflate relative advantage by *degrading* $p_\theta(y_t \mid x_{<t})$ rather than improving $z_t$; meanwhile, the coupled teacher-forced term $\nabla_\theta \log p_\theta(y_t \mid x_{<t}, z_t)$ can drift the thought-conditioned decoder $p_\theta(y_t \mid x_{<t}, z_t)$ away from the calibrated base distribution $p_\theta(y_t \mid x_{<t})$ via brittle conditioning codes. Safe optimization therefore requires stop-gradienting / freezing the reference baseline $p_{\theta_{\text{ref}}}(y_t \mid x_{<t})$ inside the reward and retaining an explicit auxiliary NTP loss on the no-thought branch.

2. **Importance-Weighted / MCMC Posterior Inference (STaR, TRICE):**
   Sampling $z_t$ from the prior $p_\theta(z_t \mid x_{<t})$ has near-zero hit rate when the correct reasoning path is rare. STaR (Zelikman et al., 2022, `arXiv:2203.14465`) approximates the posterior via rejection sampling + answer-conditioned *rationalization*; TRICE (Phan et al., 2023, `arXiv:2312.02179`) uses MCMC-EM with control variates to directly target $\nabla_\theta \log p_\theta(y_t \mid x_{<t})$ rather than the looser Jensen bound $\mathbb{E}[\log p_\theta(y_t \mid x_{<t}, z_t)]$.

3. **Pathwise Continuous Latent Transitions (Coconut, PonderLM-2, Huginn):**
   If $z_t = f_\theta(x_{<t}) \in \mathbb{R}^{L \times d}$ is a deterministic continuous recurrence (feeding hidden states directly back into the transformer), the expectation collapses to a Dirac delta and $\nabla_\theta \log p_\theta(y_t \mid x_{<t}, f_\theta(x_{<t}))$ is computed via **backpropagation through time (BPTT)**—eliminating REINFORCE variance at the cost of losing discrete stochastic exploration unless stochasticity is re-introduced (e.g., HRPO, `arXiv:2505.18454`).

### 1.3 Reward Design Taxonomy for SSL-as-RL **[Established Literature + Original Synthesis]**

Replacing an external verifier with corpus-native rewards forces a critical choice of reward functional $R(x_{<t}, z_t, y_t)$. Each option induces distinct optimization geometry, variance profiles, and failure modes:

| Reward Functional $R(x, z, y^*)$ | Representative Works | Bounded? | Strictly Proper? | Length-Invariant? | Primary Strength | Primary Pathology / Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Exact-Match Binary** $\mathbb{I}[\hat{y} = y^*]$ | RPT (`2506.08007`), RLVR | Yes $\{0,1\}$ | **No** (mode-seeking) | Yes | Zero implicit-verifier coupling; immune to likelihood hacking | Extreme sparsity on high-entropy tokens ($p \ll 0.1$); collapses distribution to mode; zero-gradient batches ($\sigma=0$) |
| **Raw Sequence Likelihood** $p_\theta(y^* \mid x, z)$ | VeriFree theoretical objective (`2505.21493`) | Yes $[0,1]$ | Marginal-exact via $\log\mathbb{E}[p]$ | **No** ($\prod_i p_i \to 0$) | Directly optimizes marginal probability $\mathbb{E}_z[p(y^*|x,z)]$ | Exponential underflow on multi-token $y^*$; flatlines gradients on long targets |
| **Log-Likelihood / Log-Score** $\frac{1}{m}\sum_{k=1}^m \log p_\theta(y^*_k \mid x, z, y^*_{<k})$ | Quiet-STaR (`2403.09629`), VeriFree practical (`2505.21493`), ELBO | **No** $(-\infty, 0]$ | **Yes** | Yes (if averaged) | Dense signal; aligned with pretraining NLL; additive across tokens | Catastrophic negative-tail variance from rare tokens ($p \approx 10^{-8}$); requires baseline subtraction & clipping |
| **Mean Token Probability (PR)** $\frac{1}{|y^*|}\sum_{k=1}^{|y^*|} p_\theta(y^*_k \mid x, z, y^*_{<k})$ | RLPR (`2506.18254`) | Yes $[0,1]$ | **No** (for joint seq) | Yes | Bounded variance; no underflow; single forward pass | Over-weights easy/predictable tokens in $y^*$; requires no-CoT debiasing & std-filtering |
| **Brier Score (Proper Scoring)** $-(1 - p_\theta(y^* \mid x, z))^2 - \sum_{v \ne y^*} p_\theta(v \mid x, z)^2$ | RLCR (`2507.16806`, verbalized variant); **[Proposal: token-level Brier]** | Yes $[-2, 0]$ | **Yes** | Yes (if averaged) | **Bounded AND strictly proper**; eliminates $-\infty$ tail spikes while preserving calibration | Weaker gradient than log-score when $p_\theta(y^*) \approx 0$ (small slope near 0 can slow cold-start learning) |
| **KL-Regularized Likelihood** $R(x,z,y^*) - \beta D_{\text{KL}}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$ | Control-as-inference; RLHF/RLVR variants | Inherits $R$ | Depends on $R, \beta$ | Depends on $R$ | Prevents policy entropy collapse and catastrophic forgetting of base language | Over-regularizes away novel latent formats if $\beta$ is too large; anchors policy to non-reasoning prior |
| **Intrinsic Self-Certainty (Zero-Label)** $\text{SelfCertainty}(\pi_\theta(\cdot \mid x, z))$ | Intuitor / RLIF (`2505.19590`) | Model-dependent | **No** | Yes | Requires *zero* ground-truth targets $y^*$ or verifiers | **Epistemic echo chamber:** model can maximize self-certainty via confident hallucinations or degenerate loops |

> **Key Mathematical Insight [Original Synthesis]:** The tension between *log-score* (unbounded variance, strong gradient at $p \approx 0$) and *Brier score* (bounded variance, strictly proper, vanishing gradient at $p \approx 0$) explains why no single reward dominates across all training phases. Cold-start discovery benefits from log-like sensitivity on rare correct paths (or posterior inference), whereas late-stage optimization and calibration benefit from bounded proper scoring rules (Brier / clipped log-gain) that prevent variance blowups and overconfident collapse.

---

## 2. State of the Art (2022–2026) **[Established Literature]**

Research on turning self-supervised prediction and verifier-free signals into latent reasoning has crystallized into four distinct lineages.

### 2.1 Lineage A: Discrete-CoT Latent-Variable & Token-Level RL Pretraining

* **STaR (Zelikman et al., 2022, `arXiv:2203.14465`, NeurIPS 2022):** Demonstrated that models can bootstrap explicit CoT rationales without human CoT labels via an iterative generate $\to$ rationalize (condition on ground-truth answer when wrong) $\to$ fine-tune loop. Established the core bootstrapping paradigm, though limited to QA tasks with known final answers.
* **TRICE / Latent-Variable MCMC-EM (Phan et al., 2023, `arXiv:2312.02179`):** Formalized unsupervised CoT strictly as latent-variable marginal likelihood maximization and introduced MCMC-EM with control variates (outperforming STaR and naive variational EM on GSM8K and BBH by reducing gradient estimator variance as competence improves).
* **Quiet-STaR (Zelikman et al., 2024, `arXiv:2403.09629`):** Generalized STaR from QA to *every token of arbitrary internet text*. Key mechanisms:
  1. *Tokenwise parallel sampling* of internal thoughts bounded by `<|startofthought|>` and `<|endofthought|>`.
  2. *Mixing head:* a shallow MLP interpolating base and thought-conditioned logits ($\log p_{\text{talk}} = w_j \log p_{\text{base}} + (1 - w_j)\log p_{\text{thought}}$) to prevent cold-start degradation.
  3. *Non-myopic teacher-forced reward:* evaluates log-likelihood over $n_{\text{true}}$ ($n_{\text{ahead}}$) future tokens, subtracted from the mean across sibling thoughts: $r_j = \log p_{\text{talk}}(X_{j+1:j+m+1}) - \log \bar{p}_{\text{talk}}(X_{j+1:j+m+1})$.
  4. *Positive-only REINFORCE truncation:* excludes negative advantages ($r_j \le 0$) for training stability at the cost of introducing estimator bias.
  * *Empirical result:* Continued pretraining of **Mistral 7B** on general internet text improved zero-shot direct reasoning on **GSM8K from 5.9% to 10.9%** and **CommonsenseQA from 36.3% to 47.2%** without task-specific fine-tuning, with largest NLL gains concentrated on difficult, low-probability tokens.
* **Reinforcement Pre-Training / RPT (Dong et al., 2025, `arXiv:2506.08007`):** Reframed next-token prediction directly as a GRPO-trained reasoning task ($G=8$, zero KL penalty, exact prefix/token match reward). Crucially, rather than thinking at every token, RPT uses a proxy model's top-16 next-token entropy to **filter out predictable syntax tokens** (punctuation, spacing, trivial transitions) and focus RL exclusively on high-entropy positions.
  * *Empirical result:* Starting from **DeepSeek-R1-Distill-Qwen-14B**, RPT-14B raised challenging "Hard" token prediction accuracy from **20.43% to 23.75%**, matching larger 32B baseline models, and served as a stronger initialization for downstream RL fine-tuning.
  * *Crucial caveat:* RPT initialized from an *already reasoning-distilled 14B checkpoint* (distilled from 671B DeepSeek-R1), not a raw small base model—leaving the cold-start question open at small scales.

### 2.2 Lineage B: Verifier-Free & Intrinsic-Reward RL

To scale RLVR beyond math and code (where deterministic checkers exist) to general domains, 2025 saw a wave of verifier-free methods that collapse the boundary between SSL likelihood and RL:

* **VeriFree (Zhou et al., 2025, `arXiv:2505.21493`):** Replaces rule-based or secondary LLM verifiers by optimizing RL directly to maximize the model's own likelihood of generating the reference answer $y^*$ given its reasoning trace $z$. Provides a variational optimization analysis showing this co-trains policy and implicit verifier in one model with lower gradient variance (Rao-Blackwellization) and zero auxiliary verifier memory overhead. Matches or surpasses verifier-based baselines on **MMLU-Pro, GPQA, SuperGPQA**, and math benchmarks.
* **RLPR (Yu et al., 2025, `arXiv:2506.18254`):** Uses the **arithmetic mean of token decoding probabilities** $\text{PR}(y^* \mid x, z) = \frac{1}{|y^*|}\sum_{t=1}^{|y^*|} P_\theta(y^*_t \mid x, z, y^*_{<t})$ to avoid sequence-length decay and log-probability instability, paired with **reward debiasing** (contrasting against a no-reasoning baseline) and **standard deviation filtering** (dropping low-variance prompts). Evaluated on Llama, Gemma, and Qwen families, outperforming VeriFree by **+7.6 on TheoremQA** and **+7.5 on Minerva**, and beating verifier-model baselines like General-Reasoner.
* **Intuitor / RLIF (Zhao et al., 2025, `arXiv:2505.19590`, ICLR 2026):** Removes even ground-truth reference answers $y^*$, training GRPO using solely the model's internal **self-certainty** as the reward signal. Matches supervised RL in-domain and improves out-of-domain transfer (e.g., code generation), though its long-horizon stability without grounding remains vulnerable to overconfident collapse.
* **RLCR (Damani et al., 2025, `arXiv:2507.16806`):** Shows standard binary RLVR creates overconfident guessers, and replaces binary rewards with a composite correctness + **Brier score** reward over verbalized confidence estimates—improving calibration and reducing hallucinations without sacrificing reasoning accuracy.

### 2.3 Lineage C: Continuous Latent Thought & Recurrent Depth

A parallel paradigm drops discrete text tokens for intermediate thoughts altogether, executing reasoning in continuous hidden states:

* **Pause Tokens (Goyal et al., 2023/ICLR 2024, `arXiv:2310.02226`):** Appends learnable `<pause>` tokens before generation, demonstrating that simply delaying output to run extra transformer forward passes yields measurable gains on QA and reasoning *when trained with pauses from pretraining/fine-tuning*—providing the essential "raw extra FLOPs" baseline for any latent-CoT claim.
* **Coconut (Hao et al., 2024, `arXiv:2412.06769`):** Feeds the model's last hidden state directly back as the next input embedding ("continuous thought") via a curriculum replacing text CoT steps. On **GPT-2**, Coconut enables implicit **breadth-first search (BFS) / superposition** over multiple candidate reasoning branches simultaneously: on the planning-heavy **ProsQA** benchmark, Coconut reaches **~97% accuracy vs. ~77.5% for text CoT** using far fewer steps—though standard text CoT still outperformed Coconut on GPT-2 GSM8K, highlighting that discrete verbalization remains advantageous when symbolic arithmetic precision is needed.
* **Huginn / Recurrent Depth (Geiping et al., 2025, `arXiv:2502.05171`):** Trains a **3.5B-parameter** model from scratch on **800B tokens** with a shared recurrent transformer block unrolled to arbitrary test-time depth per token. Requires *zero* CoT supervision; dynamically increasing recurrent iterations at test time enables the 3.5B model to match effective compute capabilities of up to **50B-scale models** on math and coding tasks while preserving KV-cache efficiency.
* **PonderLM-2 (Zeng et al., 2025, `arXiv:2509.23184`) & Token-Level Adaptive Latent CoT (Zeng et al., 2026, `arXiv:2602.08220`):** Integrates continuous latent thought directly into standard LLM pretraining. **PonderLM-2-Pythia-1.4B** (pretrained on 300B Pile tokens with one latent thought per token) **outperforms vanilla Pythia-2.8B** (2× parameters) at matched inference cost on language modeling and downstream tasks. Follow-up `arXiv:2602.08220` demonstrates that token-level adaptive halting (allocating deeper latent trajectories to hard tokens and skipping easy ones) emerges naturally in one-stage pretraining.
* **RL for Continuous/Hybrid Latent Space:** **HRPO** (Yue et al., 2025, `arXiv:2505.18454`) introduces a learnable gate mixing prior hidden states into sampled discrete tokens, re-injecting stochasticity so full RL can optimize latent reasoning without predefined CoT traces. **Özeren & Assenmacher (2025, `arXiv:2512.11816`)** similarly explore direct RL shaping of continuous latent-space thinking.

### 2.4 Lineage D: Theoretical Foundations & Critical Counter-Evidence

Any serious evaluation of SSL-as-RL must grapple with both the theoretical upper bounds and the harsh empirical failure modes discovered in 2025:

* **Circuit-Complexity Expressivity (Feng et al., 2023, `arXiv:2305.15408`, NeurIPS Oral; Merrill & Sabharwal, 2023/ICLR 2024, `arXiv:2310.07923`):** Bounded-depth Transformers without intermediate steps cannot solve basic arithmetic, linear equations, or sequential state-tracking unless model size scales super-polynomially. Conversely, **constant-size** transformers equipped with polynomial-length CoT simulate Dynamic Programming and recognize **exactly $\textsf{P}$** (all polynomial-time problems, including $\textsf{P}$-complete tasks like Horn-SAT). *This is the formal mathematical justification for why small models can punch above their parameter weight class.*
* **Compute-Optimal Test-Time Scaling (Snell et al., 2024, `arXiv:2408.03314`):** In FLOPs-matched comparisons, compute-optimal test-time scaling (adaptive search/revision per prompt difficulty) lets **PaLM 2-S\*** outperform a **~14× larger model** using standard inference (>4× more efficient than naive Best-of-N)—*critically conditioned on the smaller base model already having a non-trivial success rate on the prompt*.
* **Counter-Evidence 1 — Direct RL Fails to Match Distillation on Small Models (DeepSeek-R1, `arXiv:2501.12948`, Sec. 4.1):** Large-scale direct RL on `Qwen-32B-Base` (`DeepSeek-R1-Zero-Qwen-32B`) scored **47.0%** on AIME 2024 (`pass@1`), **60.0%** on AIME 2024 (`cons@64`), **91.6%** on MATH-500 (`pass@1`), **55.0%** on GPQA Diamond (`pass@1`), and **40.2%** on LiveCodeBench—consistently trailing simple SFT distillation from the 671B teacher (`DeepSeek-R1-Distill-Qwen-32B`: **72.6%** AIME 2024 `pass@1`, **83.3%** AIME 2024 `cons@64`, **94.3%** MATH-500 `pass@1`, **62.1%** GPQA Diamond `pass@1`, and **57.2%** LiveCodeBench). Smaller base models suffer chaotic exploration, repetition loops, and formatting collapse under cold-start RL.
* **Counter-Evidence 2 — The `pass@k` Support-Boundary Paradox (Yue et al., 2025, `arXiv:2504.13837`, NeurIPS 2025):** While RLVR substantially boosts `pass@1`, base models frequently achieve *higher* `pass@k` at large $k$ than their RLVR-trained counterparts. Current RLVR acts primarily as an **elicitation/sharpening tool** (concentrating mass on paths already inside the base distribution while narrowing exploration), whereas distillation genuinely expands the support boundary.
* **Counter-Evidence 3 — RL Optimization Pathologies:** **DAPO** (Yu et al., 2025, `arXiv:2503.14476`) documents severe **entropy collapse** and zero-gradient batches ($\sigma=0$), mitigated via *Clip-Higher* (asymmetric upper PPO clip) and dynamic sampling. **Dr. GRPO** (Liu et al., 2025, `arXiv:2503.20783`) exposes how standard GRPO's length and $\sigma$-denominator normalizations induce artificial length inflation ("overthinking" on wrong answers). **Wang et al. (2025, `arXiv:2506.01939`, NeurIPS 2025)** show that **~80% of CoT tokens are low-entropy** while **~20% high-entropy "forking" tokens** drive virtually all RLVR capability gains (restricting gradient updates exclusively to the top 20% high-entropy tokens matches or exceeds full-gradient updates on `Qwen3-8B` and boosts `Qwen3-32B` on AIME'24/25).

### 2.5 Comparative Summary of Key Methods

| Method (Year, arXiv) | Latent Representation | Reward / Objective | Verifier Needed? | Scale Tested | Key Empirical Finding | Core Limitation |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **Quiet-STaR** (2024, `2403.09629`) | Discrete text thoughts at every token | Multi-token mixed log-prob gain + pos-only REINFORCE | No (corpus text) | Mistral 7B | GSM8K 5.9%→10.9%, CSQA 36.3%→47.2% zero-shot | Heavy compute overhead thinking at every token; biased pos-only gradient |
| **RPT** (2025, `2506.08007`) | Discrete CoT on filtered tokens | Exact prefix/token match via GRPO ($G=8$) | No (corpus text) | R1-Distill-Qwen-14B | Hard-token acc 20.43%→23.75% (matches 32B) | Requires already-distilled reasoning init; exact-match mode-seeking |
| **VeriFree** (2025, `2505.21493`) | Discrete CoT | Reference answer likelihood (variational / implicit verifier) | No (ref answers only) | General QA / STEM | Matches/beats verifier baselines on MMLU-Pro, GPQA, SuperGPQA | Needs Q&A pairs with reference answers; implicit verifier bounded by model quality |
| **RLPR** (2025, `2506.18254`) | Discrete CoT | Debiased mean token prob (PR) + std-filtering | No (ref answers only) | Llama, Gemma, Qwen | Beats VeriFree (+7.6 TheoremQA, +7.5 Minerva) | Mean token prob is not strictly proper for joint sequence distributions |
| **Intuitor / RLIF** (2025, `2505.19590`) | Discrete CoT | Intrinsic self-certainty via GRPO | None (zero labels) | Reasoning & code | Matches supervised RL in-domain; strong OOD code transfer | Vulnerable to epistemic echo chamber / confident hallucination |
| **Coconut** (2024, `2412.06769`) | Continuous hidden states (BPTT) | Supervised next-token NLL (curriculum from CoT) | CoT curriculum | GPT-2 | ProsQA ~97% vs ~77.5% text CoT; emergent BFS | Requires CoT-to-latent curriculum; trails text CoT on GPT-2 GSM8K |
| **Huginn** (2025, `2502.05171`) | Continuous recurrent depth (BPTT) | Standard pretraining NLL (no CoT data) | No (corpus text) | 3.5B (800B tokens) | Dynamically scales to ~50B-equivalent compute on math/code | Architectural redesign required; lacks discrete symbolic exploration |
| **PonderLM-2** (2025, `2509.23184`) | Continuous latent thought/token (BPTT) | Standard pretraining NLL | No (corpus text) | Pythia 1.4B (300B tokens) | 1.4B beats vanilla Pythia-2.8B at matched inference cost | Fixed latent step count (addressed by adaptive halting in `2602.08220`) |
| **HRPO** (2025, `2505.18454`) | Hybrid gated hidden + discrete token | RL reward over stochastic hybrid trajectory | Task reward | General reasoning | Enables RL over latent reasoning without CoT SFT | Early-stage architecture; stability at pretraining scale unverified |

---

## 3. The Small-Model Hypothesis: When Can Latent-CoT RL Beat Parameter Scaling—and When Does It Collapse?

To answer the user's central question without hype or hand-waving, we must separate two fundamentally distinct functions that parameters perform inside a language model.

### 3.1 The Knowledge–Algorithm Decomposition **[Original Synthesis grounded in Feng et al. 2023 & Merrill/Sabharwal 2023]**

A Transformer's parameter budget $N$ serves two orthogonal masters:

1. **Parametric Knowledge Capacity ($N_{\text{mem}}$):** Storing irreducible facts, lexical mappings, rare idioms, entity relations, and statistical priors. By information-theoretic bounds, storing $K$ independent bits of world knowledge requires $\Omega(K)$ weight capacity (a few bits per parameter). **Extra test-time compute cannot synthesize missing facts.** Letting a 300M model think for 10,000 latent steps will not let it derive the atomic mass of technetium or an obscure 19th-century historical date that was never compressed into its weights.
2. **Serial Compositional / Algorithmic Capacity ($N_{\text{comp}}$):** Executing multi-step deductions, maintaining state over long dependencies, exploring search trees, or resolving syntactic/logical constraints. In a standard single-forward-pass model of depth $D$, the network can perform at most $O(D)$ serial operations per token. To approximate inherently serial functions (e.g., $k$-hop graph traversal or multi-digit arithmetic) in a single pass, standard NTP forces the model to memorize exponential lookup tables ("shortcut circuits") across huge parameter widths—wasting vast amounts of $N$ on computation that should be *time*, not *space*.

This decomposition yields a sharp structural prediction:

> **The Parametric De-Congestion Hypothesis [Original Proposal]:** When latent computation is integrated *during pretraining* (rather than bolted on purely during post-training), offloading multi-step computation from spatial width ($N_{\text{comp}}$) into temporal depth ($T_{\text{latent}}$) **frees up parameter capacity for $N_{\text{mem}}$**. This explains *why* PonderLM-2-1.4B beats Pythia-2.8B even on general language modeling and knowledge-adjacent benchmarks—not because thinking invents facts out of thin air, but because the 1.4B model stops wasting half its weights memorizing brittle 1-step computational shortcuts!

### 3.2 Four Regimes Where Small Models Punch Above Their Weight Class **[Established Literature + Original Synthesis]**

1. **Depth-Limited Compositional & Algorithmic Domains:** Math, code execution, formal logic, graph planning (ProsQA), and in-context multi-hop reasoning (where all necessary facts are *in the prompt*, so $N_{\text{mem}}$ is not the bottleneck). Here, Merrill & Sabharwal (`2310.07923`) and Feng et al. (`2305.15408`) prove constant-size models with scratchpads escape bounded-depth circuit ceilings.
2. **Pretraining-Integrated Continuous/Recurrent Latent Compute:** PonderLM-2 (`2509.23184`: 1.4B > 2.8B) and Huginn (`2502.05171`: 3.5B scaling toward ~50B compute equivalence) show that when small models learn *from step zero* to use recurrent latent steps via differentiable gradients, they achieve massive parameter-efficiency gains without suffering discrete RL exploration collapse.
3. **Branching / Search-Heavy Planning via Continuous Superposition:** Small models suffer disproportionately from early greedy mistakes in discrete text CoT (once a small model emits a wrong intermediate symbol, it lacks the self-correction sophistication of a 671B model). Continuous latent thoughts (Coconut, `2412.06769`) keep multiple branches in superposition (implicit BFS), turning a small GPT-2 into a ~97% solver on ProsQA vs. ~77.5% for text CoT.
4. **Adaptive Compute Allocation on "Fork" Tokens:** If a small model spends 1 forward pass on the ~80% routine tokens and allocates 8–32 latent steps *only* on the ~10–20% critical decision tokens (Quiet-STaR, RPT, Wang et al. `2506.01939`, Zeng et al. `2602.08220`), the *amortized* inference FLOPs increase by only ~2×–4× while effective depth on hard decisions jumps by an order of magnitude—maintaining FLOPs-matched Pareto dominance over a larger dense model.

### 3.3 Five Failure Modes That Kill Small-Model Gains (Steelmanning the Skeptic) **[Established Literature + Original Synthesis]**

A rigorous research program must face five severe bottlenecks where small-model latent-CoT RL **fails outright**:

1. **The Cold-Start Support-Boundary Trap (Most Fatal Failure Mode):**
   DeepSeek-R1 Sec. 4.1 (`2501.12948`) and Yue et al. (`2504.13837`) establish that on-policy RL cannot reinforce trajectories the base model never samples. At 100M–3B scale, a raw pretrained base model has near-zero probability of spontaneously generating coherent, self-verifying multi-step discrete CoT on hard tokens. Notice that **RPT (`2506.08007`) sidestepped this entirely by starting from `DeepSeek-R1-Distill-Qwen-14B`** (a checkpoint already distilled from a 671B teacher!). Without either (i) differentiable continuous latent pretraining, (ii) lightweight SFT/distillation warm-start, or (iii) posterior rationalization (STaR/TRICE), pure discrete RL on a raw small base model stalls in noise or collapses into repetition loops.
2. **The Implicit-Verifier Competence Floor:**
   In Quiet-STaR, VeriFree, and RLPR, the *small model itself* computes $p_\theta(y^* \mid x, z)$ to score its own thoughts $z$. A weak small model is a noisy, poorly calibrated likelihood evaluator: it can fail to recognize that a genuinely valid reasoning chain $z$ implies $y^*$, *or* reward degenerate "shortcut" thoughts that prime lexical n-grams of $y^*$ without performing valid deduction.
3. **Token-Overhead FLOPs Inversion:**
   If a 1B model generates 400 discrete CoT tokens to predict a single hard token (or 4,000 tokens to answer a question), its inference compute is $400\times$ higher per token—vastly exceeding the FLOPs of a 70B dense model doing 1 forward pass! Small models punch above their weight *in parameter count* easily, but punching above their weight **in FLOPs-matched latency/cost** requires either *sparse adaptive triggering* (thinking only on hard tokens) or *compact continuous thoughts* (where 4 continuous vectors replace 100 text tokens).
4. **Scratchpad Context Drift & Error Accumulation:**
   Small models have weaker attention fidelity over self-generated text. Longer discrete CoT chains often *degrade* accuracy ("overthinking" in Dr. GRPO, `2503.20783`), as a 0.5B–1.5B model gets derailed by its own noisy intermediate tokens.
5. **Parametric Knowledge Hallucination:**
   On knowledge-bound tasks where $N_{\text{mem}}$ is the bottleneck, rewarding high likelihood or self-certainty (Intuitor/RLIF) encourages *plausible-sounding confabulation* rather than epistemic humility—unless paired with strictly proper calibration rewards (RLCR, `2507.16806`) and abstention mechanisms.

---

## 4. Key Technical Bottlenecks & Design Principles

### 4.1 Which Tokens Benefit from Thinking? The Epistemic vs. Aleatoric Trap **[Original Synthesis]**

Both RPT (`2506.08007`) and Wang et al. (`2506.01939`) show that uniform token-level RL is hugely wasteful because ~80% of tokens are predictable low-entropy continuations. However, **filtering purely by raw token entropy $H(X_t \mid X_{<t})$ has a major blind spot:**

Total predictive uncertainty decomposes into two radically different components:

$$H(X_t \mid X_{<t}) = \underbrace{H_{\text{aleatoric}}(X_t \mid X_{<t})}_{\text{Irreducible randomness}} + \underbrace{H_{\text{epistemic/comp}}(X_t \mid X_{<t})}_{\text{Reducible via computation or knowledge}}$$

* **Irreducible Aleatoric High-Entropy Tokens:** The first name of a newly introduced character in fiction (`"The detective met ___"`), an arbitrary date, or an author's stylistic word choice. *No amount of latent reasoning can deduce the exact ground-truth token.* Running RL on these tokens injects pure label noise into the policy gradient—causing the model to either overfit spurious correlations or collapse its CoT into meaningless filler.
* **Computationally Reducible High-Entropy Tokens:** The result of an arithmetic expression, a closing variable identifier in code, a deductive conclusion from earlier premises, or the resolution of a coreference chain. Here, latent compute dramatically drops NLL.

**Proposed Selection Criterion — Compute-Elasticity / Value of Thought (VoT) [Original Proposal]:**
Instead of raw entropy $H(X_t \mid X_{<t})$, target tokens with high **cross-scale likelihood gap** (using a cheap frozen larger teacher or proxy model $p_{\text{ref-large}}$ vs. the current small model $p_{\text{small}}$):

$$\Delta^{\text{scale}}_t = \log p_{\text{ref-large}}(x_t \mid x_{<t}) - \log p_{\text{small}}(x_t \mid x_{<t})$$

Why this works: on an *aleatoric* token (e.g., an arbitrary character name), **both** the large model and the small model have high entropy, so $\Delta^{\text{scale}}_t \approx 0$! On a *computationally hard* token (e.g., multi-step deduction or code state), the larger model assigns much higher probability than the small model, yielding large positive $\Delta^{\text{scale}}_t$. *(Two caveats: first, $\Delta^{\text{scale}}_t$ also fires on pure parametric-fact gaps where the large model memorized a fact the small model lacks, so combining it with multi-token predictability checks or technical/code/math corpora helps filter pure trivia; second, using $p_{\text{ref-large}}$ introduces subtle **teacher distillation into token selection**, meaning rigorous SSL-as-RL benchmarks must pair it with a strictly teacher-free control.)*

In a purely self-contained (**teacher-free**) setting, an analogous online proxy is the **within-group empirical variance of predictive gain** across sampled thoughts $z^{(1)}, \dots, z^{(G)}$ (as in RLPR's std-filtering and DAPO's $\sigma > 0$ dynamic sampling): if *all* $G$ thoughts yield identically low likelihood for $x_t$, the token is either aleatoric or beyond current support and should receive zero policy gradient.

### 4.2 Sparse vs. Dense Rewards, Variance Reduction, and Credit Assignment **[Established Literature + Original Synthesis]**

* **Sparsity vs. Hacking Trade-off:** Exact-match rewards (RPT) are unhackable by the implicit verifier, but at small scales almost all $G$ rollouts fail to match a hard token exactly ($\sigma = 0$), starving the gradient. Dense likelihood rewards (Quiet-STaR, VeriFree, RLPR) give smooth signal on every rollout, *but* require three safeguards:
  1. **Multi-token horizon ($m > 1$, Quiet-STaR's $n_{\text{true}}$):** Rewarding only $x_t$ ($m=1$) incentivizes myopic lexical priming; scoring $x_{t:t+m-1}$ ($m \in \{4, 8\}$) forces $z_t$ to encode semantic/structural plans that explain an entire clause or expression.
  2. **Stop-Gradient / Frozen-Reference Debiasing + Leave-One-Out Group Baselines (RLOO / Dr. GRPO):** Always measure the *counterfactual marginal gain* of thought $z_t$ against a **stop-gradiented, frozen-reference zero-thought baseline** $\operatorname{sg}[\log p_{\theta_{\text{ref}}}(y_t \mid x_{<t}, \varnothing)]$ (never a live differentiable $p_\theta(y_t \mid x_{<t}, \varnothing)$ that the optimizer can sabotage to manufacture fake advantage!), paired with an explicit auxiliary standard NTP loss on the no-thought branch so $p_\theta(y_t \mid x_{<t}, z_t)$ cannot drift away from $p_\theta(y_t \mid x_{<t})$ via degenerate co-adaptation. Center advantages within the sampled group without dividing by $\sigma$ (Dr. GRPO, `2503.20783`) to avoid length inflation and $\sigma \approx 0$ gradient explosions.
  3. **Stop-Gradient on the Implicit Reward Score:** Treat the reward scalar $R(x_{<t}, z_t, y_t)$ strictly as $\operatorname{sg}[R]$ inside the REINFORCE/GRPO term (allowing direct likelihood gradients into $p_\theta(y_t \mid x_{<t}, z_t)$ only through a separate, weight-controlled SFT term if desired), preventing reward-model self-hacking.
* **Intra-Thought Credit Assignment (Open Bottleneck):** Even with a dense sequence reward $R(x_{<t}, z_t, y_t)$, REINFORCE/GRPO broadcasts the *same* scalar advantage to every token inside $z_t = (z_t^{(1)}, \dots, z_t^{(L)})$. For long thoughts ($L > 64$), variance scales linearly with $L$. Restricting policy updates to the **top-20% high-entropy forking tokens inside $z_t$** (Wang et al., 2025, `arXiv:2506.01939`) directly slashes credit-assignment noise without requiring a learned critic/value network.

### 4.3 Preventing the Two Mirror-Image Collapses **[Established Literature + Original Synthesis]**

Latent-CoT RL sits on a razor's edge between two opposing failure modes:

1. **Posterior Collapse (Latent Bypass):** Early in training, untrained thoughts $z_t$ are noise that *hurt* next-token NLL. If the decoder $p_\theta(y_t \mid x_{<t}, z_t)$ can attend directly to $x_{<t}$ (or via a mixing head with $w_j \to 1$), gradient descent rapidly learns to ignore $z_t$ altogether—reducing the model back to standard NTP with wasted compute.
   * *Mitigations:* Quiet-STaR's learned mixing head initialized smoothly; **thought-dropout / causal bottlenecking** (forcing a subset of attention heads or final-layer features to route through $z_t$ during warmup); and **explicit swap-sensitivity monitoring** (Section 5.5).
2. **Policy Entropy Collapse (Template Fossilization):** Once a few phrasing templates in $z_t$ yield positive advantage, aggressive policy gradients collapse $p_\theta(z_t \mid x_{<t})$ onto a narrow set of repetitive patterns, shrinking `pass@k` (Yue et al., `2504.13837`) and halting further discovery.
   * *Mitigations:* **DAPO Clip-Higher** (`2503.14476`, e.g., $\epsilon_{\text{low}} = 0.2, \epsilon_{\text{high}} = 0.28$) to protect low-probability exploratory tokens from being clipped out; adaptive KL/entropy floor; and continuous-to-discrete hybrid exploration (HRPO, `2505.18454`).

### 4.4 The Continuous-Pretraining + Discrete-RL Bridge **[Original Proposal]**

Synthesizing Lineage A (Quiet-STaR/RPT) and Lineage C (Coconut/Huginn/PonderLM-2) suggests a concrete architectural hypothesis: **Discrete RL and continuous latent pretraining solve each other's primary failure modes.**
* Pure discrete RL fails on raw small models because of the cold-start exploration cliff and high score-function variance.
* Pure continuous BPTT (PonderLM-2, Huginn) scales cleanly from scratch on small models and achieves 2×–10× parameter efficiency, *but* deterministic forward passes cannot perform stochastic trial-and-error search or escape local minima on hard combinatorial problems.
* **Two-Stage / Hybrid Synthesis:** Pretrain small models with differentiable *adaptive continuous latent thoughts* (building internal reasoning circuits and de-congesting parameters with zero REINFORCE variance), *then* apply verifier-free / SSL-as-RL (hybrid HRPO or selective discrete/continuous policy exploration) on compute-elastic tokens to unlock stochastic search and self-correction.

---

## 5. Concrete, Runnable Small-Scale Experimental Design **[Original Proposal]**

To rigorously test whether SSL-as-RL with latent CoT enables small models to punch above their weight class—without conflating parameter count, FLOPs, distillation leakage, or aleatoric noise—we propose a staged, self-contained experimental suite with explicit H100 compute accounting (assuming BF16 at ~40% MFU, i.e., $\approx 1.44 \times 10^{18}$ FLOPs/H100-hour):

* **Stage 1 — Minimal Core Pilot (~30–45 H100 GPU-hours; 1–4× H100 over a weekend):** Runs Track A from-scratch pretraining on **2B tokens** for the 160M target ($\sim 1.9\times 10^{18}$ FLOPs, ~1.5 GPU-hrs), 410M ($\sim 4.9\times 10^{18}$ FLOPs, ~3.5 GPU-hrs), and 1.0B ($\sim 1.2\times 10^{19}$ FLOPs, ~9 GPU-hrs) NTP baselines, plus B3 (pause), B4 (continuous latent), and one core RL arm (`T2-self` + `R2`/`R3`).
* **Stage 2 — Full Factorized Matrix + Track B (~250–400 H100 GPU-hours; ~2 days on an 8× H100 node):** Executes the full Track A token-filter × reward × interface ablation matrix and Track B continued-pretraining RL on `Qwen2.5-0.5B/1.5B-Base` ($G=8$ GRPO rollouts over 50M–100M filtered high-signal tokens, plus B5 teacher distillation).

### 5.1 Experimental Axes & Model Ladder

We run two complementary tracks to isolate **pretraining de-congestion** from **post-pretraining elicitation / cold-start limits**:

* **Track A — Controlled From-Scratch Lab Scale (Tests Parametric De-Congestion):**
  * Small target model: **160M** Llama-architecture decoder-only Transformer (depth 12, width 768).
  * Weight-class ceilings (standard NTP baselines): **410M** (~2.5× params) and **1.0B** (~6× params), trained on the *exact same token sequence* (**2B tokens** for Stage 1—approximately Chinchilla-proportional for the 160M target while keeping 1.0B ceiling pretraining under ~10 H100 GPU-hours; optionally scaled to 10B tokens only if Stage 1 passes falsification checks).
* **Track B — Realistic Small-LM Continued Pretraining (Tests Cold-Start vs. Warm-Start):**
  * Small target models: **`Qwen2.5-0.5B-Base`** and **`Qwen2.5-1.5B-Base`** (raw base, testing true cold-start) AND an ablation arm with **5,000-example lightweight CoT SFT warmup** (testing whether crossing the Yue et al. / DeepSeek-R1 support boundary unlocks RPT-style gains at 0.5B–1.5B scale!).
  * Weight-class ceilings: **`Qwen2.5-3B-Base`** and **`Qwen2.5-7B-Base`** (2×–14× larger, standard 1-pass inference).

### 5.2 Stratified Dataset Suite (Separating Algorithmic, Natural, and Aleatoric Regimes)

A single aggregate perplexity number hides where latent CoT works vs. fails. Training and evaluation are explicitly stratified into three buckets:

1. **Regime 1 — Controllable Algorithmic & Compositional (Predicted Strong Win):**
   * *DAG Multi-Hop Planning (ProsQA-style synthetic graphs)* with controllable hop depth $d \in \{2, 4, 6, 8\}$ (OOD depth generalization tests true algorithmic execution vs. memorization!).
   * *Python Trace Execution* (synthetically generated 5–20 line stateful scripts predicting final printed state, plus held-out CRUXEval-O / MBPP).
   * *Multi-step Modular Arithmetic & Variable Binding* (known circuit-complexity failure mode for shallow Transformers).
2. **Regime 2 — Structured Natural Web Text (Predicted Selective Win on Hard Tokens):**
   * Mixture of **OpenWebMath** (40%), **The Stack v2 (Python/C++)** (40%), and **FineWeb-Edu** (20%)—trained purely via corpus next-token prediction with *no external labels*.
   * Zero-shot downstream eval: **GSM8K**, **CommonsenseQA**, **MBPP**, and **Hard-Token Bits-Per-Byte (Hard-BPB)** on held-out web text.
3. **Regime 3 — Pure Parametric Fact Recall / Aleatoric Negative Control (Predicted No Win / Potential Degradation):**
   * **TriviaQA / SimpleQA-style entity cloze** and fiction proper-noun prediction.
   * *Purpose:* Explicitly verifies our Knowledge–Algorithm Decomposition (Section 3.1) and tests whether RL rewards cause hallucinated overconfidence on unknowable tokens.

### 5.3 Controlled Baselines (Strictly FLOPs-Matched & Parameter-Matched)

Every latent-CoT RL arm is compared against **five mandatory baselines**:

1. **B1 — Same-Size Standard NTP ($N$ params, $1\times$ inference FLOPs):** Lower bound.
2. **B2 — Larger Standard NTP ($2.5N$ and $6N$ params):** Defines the target "weight class" to punch into.
3. **B3 — Same-Size Pause-Token Model ($N$ params, $L$ learnable `<pause>` tokens, matched inference FLOPs):** *Critical control* (Goyal et al., `2310.02226`)—separates raw extra forward-pass compute from *RL-learned reasoning*.
4. **B4 — Continuous Latent Pretraining Only (PonderLM-2 / Coconut-style BPTT with $L \in \{2, 4\}$ continuous steps, matched inference FLOPs):** Tests differentiable latent compute without RL.
5. **B5 — Teacher-Distilled SFT ($N$ params distilled from 7B teacher CoT traces):** Benchmarks autonomous RL discovery against supervised teacher transfer (DeepSeek-R1 Sec. 4.1 control).

### 5.4 Methodological Ablation Matrix (Token Selection × Reward × Latent Interface)

On the small target models (160M from-scratch; 0.5B/1.5B continued pretraining), run a factorized ablation (using Dr. GRPO + DAPO Clip-Higher $\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28$, $G=8$, dynamic $\sigma=0$ filtering, $n_{\text{ahead}} = m = 4$ future tokens, plus an auxiliary standard no-thought NTP loss $\lambda_{\text{ntp}} \mathcal{L}_{\text{NTP}}(\theta)$ with $\lambda_{\text{ntp}} = 0.2$ to prevent decoder drift):

* **Factor 1 — Token Triggering / Filtering:**
  * `T0`: Every token (Quiet-STaR baseline, fixed small thought length $L=8$).
  * `T1`: Top-20% highest **proxy entropy** tokens (RPT baseline).
  * `T2-self`: Strictly **teacher-free** online filter selecting the top-20% tokens by **within-group predictive-gain standard deviation** across $G$ sampled thoughts (isolates pure SSL-as-RL with zero external model leakage).
  * `T2-scale`: Top-20% highest **Cross-Scale Probability Gap** tokens $\Delta^{\text{scale}}_t = \log p_{\text{ref-large}}(x_t|x_{<t}) - \log p_{\text{small}}(x_t|x_{<t})$ **[Proposed epistemic/computational upper-bound filter; explicitly audited against `T2-self` to measure teacher-selection lift]**.
* **Factor 2 — Corpus-Native Reward Function (with Frozen-Reference Anchoring):**
  * `R1`: Exact-match binary $\mathbb{I}[\hat{x}_t = x_t]$ (RPT baseline).
  * `R2`: Debiased, lower-clipped multi-token log-likelihood gain against a **frozen reference no-thought anchor** $p_{\theta_{\text{ref}}}$ (with $\operatorname{sg}[\cdot]$ preventing baseline sabotage and reward-model hacking):
    $$R_{\text{log}}(x_{<t}, z_t, y_t) = \frac{1}{m}\sum_{k=0}^{m-1} \operatorname{sg}\!\left[\text{clip}\!\left(\log p_\theta(x_{t+k} \mid x_{<t}, z_t, x_{t:t+k-1}) - \log p_{\theta_{\text{ref}}}(x_{t+k} \mid x_{<t}, x_{t:t+k-1}),\, -\tau,\, +\tau\right)\right]$$
    with $\tau = 5.0$ to prevent catastrophic negative-tail variance (and $\theta_{\text{ref}}$ updated only via slow periodic EMA or held fixed to the pre-RL checkpoint).
  * `R3`: Debiased multi-token **Brier-score gain** **[Proposed bounded strictly proper reward]**, identically anchored to frozen $p_{\theta_{\text{ref}}}$ under $\operatorname{sg}[\cdot]$:
    $$R_{\text{Brier}}(x_{<t}, z_t, y_t) = \frac{1}{m}\sum_{k=0}^{m-1} \operatorname{sg}\!\Big[ \text{BS}\big(p_\theta(\cdot \mid x_{<t}, z_t, x_{t:t+k-1}),\, x_{t+k}\big) - \text{BS}\big(p_{\theta_{\text{ref}}}(\cdot \mid x_{<t}, x_{t:t+k-1}),\, x_{t+k}\big) \Big]$$
    where $\text{BS}(q, v^*) = 2 q(v^*) - \sum_{v \in \mathcal{V}} q(v)^2 - 1 \in [-2, 0]$ (computed exactly in $O(|\mathcal{V}|)$ from the softmax logits at negligible overhead!).
* **Factor 3 — Latent Interface:**
  * `I1`: Discrete text thoughts ($L \in \{8, 16, 32\}$ tokens, gradient masked to top-20% high-entropy tokens inside $z_t$ per Wang et al. `2506.01939`).
  * `I2`: Continuous latent pretraining (Track A: `B4`) followed by **Hybrid HRPO-style gated RL** (`2505.18454`) on `T2-self` / `T2-scale` tokens.

### 5.5 Diagnostic Metrics & Explicit Falsification Criteria

Standard accuracy alone can be fooled by length inflation, posterior collapse, or memorization. Every run reports five mandatory diagnostics:

1. **FLOPs-Matched Pareto Frontier:** Plot task accuracy and Hard-BPB against *total inference FLOPs per predicted token* (comparing small model + latent steps directly against $2.5N$ and $6N$ standard models).
2. **Support-Boundary Probe (`pass@1` vs. `pass@64`):** Following Yue et al. (`2504.13837`), measure whether SSL-as-RL expands `pass@64` over the base model (genuine capability expansion) or only sharpens `pass@1` while shrinking `pass@64` (pure elicitation).
3. **Causal Thought Faithfulness (Swap-Sensitivity + Format-Matched Truncation & Depth Scaling):** At evaluation, replace a model's generated thought $z_t$ with a mismatched thought $z'_{t'}$ sampled from another prompt of equal length:
   $$\Delta \text{NLL}_{\text{swap}} = \text{NLL}(y_t \mid x_{<t}, z'_{t'}) - \text{NLL}(y_t \mid x_{<t}, z_t)$$
   If $\Delta \text{NLL}_{\text{swap}} \approx 0$, the model suffered **posterior collapse** (or is merely using $z_t$ as a content-free pause token). *Crucial caveat:* $\Delta \text{NLL}_{\text{swap}} > 0$ alone only proves decoder *dependence* on $z_t$—which brittle formatting delimiters or learned checksum codes can also trigger without genuine reasoning. We therefore pair $\Delta\text{NLL}_{\text{swap}}$ with two stronger faithfulness checks: (a) a **format-matched intermediate-step truncation/perturbation ablation** ($\Delta\text{NLL}_{\text{trunc}}$, corrupting internal reasoning steps while preserving formatting delimiters and token count), and (b) **monotonic depth scaling**, testing whether $\Delta\text{NLL}_{\text{swap}}$ and accuracy lift grow monotonically with OOD algorithmic hop depth $d \in \{2, 4, 6, 8\}$ rather than remaining flat.
4. **Distributional Calibration (ECE & Token Brier Score):** Measured on both Regime 2 (natural text) and Regime 3 (TriviaQA negative control) to verify whether `R3` (Brier) avoids the overconfidence collapse of `R1` (exact match).
5. **Training Stability Telemetry:** Track policy entropy $H(\pi_\theta(z_t \mid x_{<t}))$, fraction of zero-variance groups ($\sigma = 0$), no-thought baseline NLL drift $\text{NLL}(y_t \mid x_{<t}, \varnothing)$ (checking for baseline sabotage), and mean thought length over training steps.

#### Explicit Confirmation vs. Falsification Criteria

* **Hypothesis Confirmed (Strong Form):** On Regimes 1 & 2, a small model trained with strictly teacher-free adaptive latent-CoT RL (`T2-self` + `R2`/`R3` or `I2`) matches or exceeds the **$2.5\times$ larger NTP baseline (B2)** at **$\le 1.0\times$ matched inference FLOPs**, beats the FLOPs-matched pause-token control (**B3**) by $\ge 3$ percentage points on algorithmic/reasoning tasks, passes both causal thought checks ($\Delta\text{NLL}_{\text{swap}} > 0.15$ nats, positive format-matched $\Delta\text{NLL}_{\text{trunc}}$, and monotonic scaling with OOD hop depth $d$), maintains stable no-thought calibration, and exhibits **strictly higher `pass@64`** than its base model on OOD hop depths ($d=6, 8$).
* **Hypothesis Partially Confirmed (Pretraining-Only / Warm-Start / Teacher-Filtered Conditional):** Gains materialize only under continuous latent pretraining (`B4`/`I2`), post-warmstart RL, or teacher-assisted token filtering (`T2-scale` beating `T2-self`), while cold-start teacher-free discrete RL on raw 0.5B/1.5B base models fails to beat B3. *(This would sharply confirm that continuous differentiable pretraining or teacher bootstrapping is a prerequisite for small-model discrete RL.)*
* **Hypothesis Falsified:** Even on Regime 1 (algorithmic depth-limited tasks) with `T2-self` or `T2-scale` epistemic filtering, no small-model latent-CoT RL variant beats either (a) the FLOPs-matched pause-token baseline **B3**, or (b) the FLOPs-matched larger NTP baseline **B2**—indicating that extra compute at small parameter scales is better spent on larger dense/MoE weights or deterministic recurrent depth than on RL over unsupervised latent thoughts.

---

## 6. Bridge to Tracks 2 & 3: Where SSL-as-RL Ends

SSL-as-RL is the **universal base layer** of self-verified reinforcement learning: because every byte of text, code, or sensory data reveals its own future continuation, verification is automatic and ubiquitous. Yet predictive verification has two structural ceilings that motivate Tracks 2 and 3 of this investigation:

1. **Beyond Next-Token Prediction (Track 2 — Verification Asymmetry & Inversion):** Many of the highest-value reasoning tasks (proving a theorem, integrating a function, synthesizing a bug-free system) cannot be reduced to predicting a single canonical continuation in a corpus—multiple valid solutions exist, and the verifier is a *structural check* (differentiation, proof checking, residual equations, NP witness checking) rather than string likelihood.
2. **Beyond Immediate Observation (Track 3 — Slow & Delayed Verification):** SSL-as-RL works because the "next token" arrives immediately in a recorded corpus. In science, economics, and long-horizon decision-making, the ground-truth continuation of reality may not arrive for months or years—turning the problem from fast token-level RL into slow, non-stationary, multi-fidelity verification.

---

## Verified References (2022–2026)

1. **Damani, M., Puri, I., Slocum, S., Shenfeld, I., Choshen, L., Kim, Y., & Andreas, J. (2025).** *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR).* `arXiv:2507.16806`.
2. **DeepSeek-AI (2025).** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* `arXiv:2501.12948`.
3. **Dong, Q., et al. (2025).** *Reinforcement Pre-Training (RPT).* `arXiv:2506.08007`.
4. **Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., & Wang, L. (2023).** *Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective.* `arXiv:2305.15408` (NeurIPS 2023 Oral).
5. **Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B., Kailkhura, B., Bhatele, A., & Goldstein, T. (2025).** *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn).* `arXiv:2502.05171`.
6. **Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., & Nagarajan, V. (2023/2024).** *Think before you speak: Training Language Models With Pause Tokens.* `arXiv:2310.02226` (ICLR 2024).
7. **Hao, S., et al. (2024).** *Training Large Language Models to Reason in a Continuous Latent Space (Coconut).* `arXiv:2412.06769`.
8. **Liu, Z., et al. (2025).** *Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO).* `arXiv:2503.20783`.
9. **Merrill, W., & Sabharwal, A. (2023/2024).** *The Expressive Power of Transformers with Chain of Thought.* `arXiv:2310.07923` (ICLR 2024).
10. **Özeren, E., & Assenmacher, M. (2025).** *Reinforcement Learning for Latent-Space Thinking in LLMs.* `arXiv:2512.11816`.
11. **Phan, D., Hoffman, M. D., et al. (2023).** *Training Chain-of-Thought via Latent-Variable Inference (TRICE).* `arXiv:2312.02179`.
12. **Snell, C., Lee, J., Xu, K., & Kumar, A. (2024).** *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* `arXiv:2408.03314`.
13. **Wang, S., Lin, J., Huang, G., et al. (2025).** *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.* `arXiv:2506.01939` (NeurIPS 2025).
14. **Yu, Q., et al. (2025).** *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* `arXiv:2503.14476`.
15. **Yu, T., Liu, Z., Sun, M., Chua, T.-S., et al. (2025).** *RLPR: Extrapolating RLVR to General Domains without Verifiers.* `arXiv:2506.18254`.
16. **Yue, Y., Chen, Z., Lu, R., et al. (2025).** *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* `arXiv:2504.13837` (NeurIPS 2025).
17. **Yue, Z., Jin, B., Zeng, H., Zhuang, H., Qin, Z., Yoon, J., Shang, L., Han, J., & Wang, D. (2025).** *Hybrid Latent Reasoning via Reinforcement Learning (HRPO).* `arXiv:2505.18454`.
18. **Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022).** *STaR: Bootstrapping Reasoning With Reasoning.* `arXiv:2203.14465` (NeurIPS 2022).
19. **Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., & Goodman, N. D. (2024).** *Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking.* `arXiv:2403.09629`.
20. **Zeng, B., Li, H., Song, S., Wang, Y., Wang, Z., He, Z., Wang, X., & Lin, Z. (2025).** *PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space.* `arXiv:2509.23184`.
21. **Zeng, B., Hao, Y., Li, H., Song, S., Song, F., Wang, Z., Huang, S., Xu, Y., He, Z., Wang, X., & Lin, Z. (2026).** *Pretraining with Token-Level Adaptive Latent Chain-of-Thought.* `arXiv:2602.08220`.
22. **Zhao, X., Kang, Z., Feng, A., Levine, S., & Song, D. (2025).** *Learning to Reason without External Rewards (Intuitor / RLIF).* `arXiv:2505.19590` (ICLR 2026).
23. **Zhou, X., Liu, Z., Sims, A., Wang, H., Pang, T., Li, C., Wang, L., Lin, M., & Du, C. (2025).** *Reinforcing General Reasoning without Verifiers (VeriFree).* `arXiv:2505.21493`.
