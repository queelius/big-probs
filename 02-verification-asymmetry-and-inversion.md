# Verification Asymmetry & Inversion: Generalizing Automatic Verifiers Beyond Math and Code

> **Track 2 Report** — Foundational Investigation: *"Doing for RL and verification what self-supervised learning (SSL) did for unsupervised learning."*
>
> **Epistemic Hygiene Note:** Throughout this report, established peer-reviewed and preprint literature (2024–2026, plus classical theoretical foundations) is explicitly cited with verified authors, years, and arXiv/journal identifiers. Original conceptual syntheses, unified formalisms, and engineering proposals are explicitly tagged as **[Original Synthesis]** or **[Original Proposal]**.

---

## Executive Summary

Self-supervised learning (SSL) unlocked scaling for representation learning by exploiting a degenerate form of automatic verification: masking a token or patch from real data creates a labeled prediction problem at zero human annotation cost. Recent progress in Reinforcement Learning with Verifiable Rewards (RLVR) has achieved an analogous leap in pure mathematics and software engineering, where theorem provers (Lean 4) and unit-test harnesses supply cheap, objective rewards. Yet a widespread misconception holds that automatic verifiability is an accident unique to math and code.

This report demonstrates that automatic verifiability is governed by **deep structural asymmetries** that recur across physics, chemistry, structural biology, hardware design, cybersecurity, optimization, economics, medicine, and statutory law. We formalize two orthogonal axes of verification asymmetry—**Evaluative Asymmetry** (checking a candidate $(x,y)$ far more cheaply than solving $x$) and **Generative Asymmetry** (sampling a solution $y$ and applying an easy forward/corrupting map $g$ to synthesize a verified hard problem $x = g(y)$)—and show that **SSL is merely the trivial identity special case of Generative Asymmetry**, where $g$ is projection/masking on real data (guaranteeing zero distribution shift). Generalizing beyond SSL means replacing trivial masking with richer structural operators while managing the exact failure modes—above all **pushforward distribution shift**, **semantic drift**, and **verifier gaming**—that emerge once we leave the masking manifold.

We organize the landscape into an **eight-mechanism taxonomy of structural automatic verifiability** grounded in 2024–2026 literature, analyze the precise failure modes of each mechanism, and synthesize a six-transform **Verification Design Playbook** with worked blueprints for medicine, law/economics, and physical engineering.

---

## 1. Formalizing Verification Asymmetry

### 1.1 The Generation–Verification Gap and Two Orthogonal Axes [Original Synthesis]

Consider a task family over problem space $\mathcal{X}$ and solution space $\mathcal{Y}$ with target relation $R \subseteq \mathcal{X} \times \mathcal{Y}$ (or continuous quality functional $U: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$). Let $C_{\text{solve}}(x)$ denote the computational (or sample/search) cost required to find $y \in \mathcal{Y}$ satisfying $R(x,y)$, and $C_{\text{verify}}(x,y)$ the cost to evaluate a verifier $V(x,y) \in [0,1]$.

Automatic verification becomes scalable for RL whenever there exists a large **Generation–Verification Gap (GV-Gap)**—not merely in compute, $\Delta_{\text{comp}}(x) = C_{\text{solve}}(x) / C_{\text{verify}}(x,y) \gg 1$, but also in **label independence**: $V(x,y)$ must operate without requiring a human-provided ground-truth answer $y^*$.

Current RLVR discussions frequently conflate two mathematically distinct sources of label-free verification:

1. **Axis A — Evaluative Asymmetry (Checking arbitrary $x \sim P_{\text{real}}$):**
   Given a real-world problem $x$ drawn from the target deployment distribution $P_{\text{real}}(x)$ and an agent's candidate output $y$ (optionally paired with a witness $w$), there exists a cheap evaluator $V(x, y, w)$ that certifies correctness or bounds suboptimality without knowing $y^*$.
   *Examples:* SAT assignment checking, PDE residual evaluation, primal–dual optimization gaps, exploit execution, lossless decompression.

2. **Axis B — Generative Asymmetry (Manufacturing verified $(x, y)$ pairs via easy forward maps):**
   Even when checking an *arbitrary* $x$ is hard or expensive, there exists a cheap forward operator $g: \mathcal{Y} \to \mathcal{X}$ (or deduction/corruption process) such that sampling $y \sim Q$ and computing $x = g(y)$ produces a problem $x$ paired with a verified solution $y$ by construction.
   *Examples:* Symbolic differentiation $F \mapsto F' = f$ to synthesize integration problems; AST bug injection into a passing codebase to synthesize repair tasks; forward geometric deduction from random diagrams to synthesize Olympiad proofs.

### 1.2 Why SSL Scaled So Effortlessly—and Where General Inversion Breaks [Original Synthesis]

Viewing SSL through Axes A and B exposes why SSL was the easiest case of automatic verification, and exactly what becomes hard when we generalize it:

* **SSL as Trivial Generative Asymmetry:** In next-token prediction or masked modeling, we draw a real artifact $z \sim P_{\text{real}}$ and apply a trivial projection/masking operator $x = M(z)$, $y = \bar{M}(z)$. Because $z$ is drawn directly from $P_{\text{real}}$ and $M$ is a simple restriction, the induced problem distribution $M_\# P_{\text{real}}$ *matches the natural data manifold* with zero distribution shift, and verification is $O(1)$ exact token comparison or log-likelihood scoring (connecting directly to Track 1's framing of SSL as degenerate RL and Mechanism 8's compression view).
* **The Pushforward Shift Theorem of General Inversion:** In general Generative Asymmetry (e.g., differentiating random expression trees $F \sim Q_{\text{syn}}$ to create integration problems $f = D(F)$), the induced training problem distribution is the **pushforward measure** $g_\# Q_{\text{syn}}$. Unless $Q_{\text{syn}}$ is specifically calibrated so that $g_\# Q_{\text{syn}} \approx P_{\text{real}}(x)$, an RL policy trained on inverted tasks optimizes over an alien problem manifold.
  $$\text{SSL: } z \sim P_{\text{real}} \implies M_\# P_{\text{real}} \text{ grounded} \qquad \text{vs.} \qquad \text{Naive Inversion: } y \sim Q_{\text{syn}} \implies g_\# Q_{\text{syn}} \neq P_{\text{real}}(x)$$

This single structural observation governs the design of almost every successful 2024–2026 synthetic RL system: **the most effective inverted task generators ground $Q$ in real-world artifacts** (e.g., real GitHub repositories in SWE-smith rather than random ASTs) **and/or replace exact match against $y_0$ with an Axis A equivalence verifier** $V(x, y) = \mathbb{I}[g(y) = x]$ to avoid penalizing valid alternative pre-images.

---

## 2. Taxonomy of Structural Sources of Automatic Verifiability

We identify **eight distinct structural mechanisms** by which automatic verification arises across domains. While real systems frequently compose multiple mechanisms, each has a distinct mathematical signature, domain footprint, and failure mode.

---

### Mechanism 1: Inversion & One-Way Forward–Backward Synthesis

* **Axis:** Generative Asymmetry (Axis B), often paired with forward-check Evaluative Asymmetry (Axis A).
* **Formal Characterization:** There exists a map $g: \mathcal{Y} \to \mathcal{X}$ (or multi-step forward transition $y_0 \xrightarrow{g_1} \dots \xrightarrow{g_k} x$) that is computationally cheap to evaluate ($C_g \ll C_{g^{-1}}$), whereas computing a pre-image $y \in g^{-1}(x)$ is hard. One constructs training instances $(x = g(y), y)$ by sampling $y \sim Q$. In its cryptographic extreme, $g$ is a trapdoor one-way function; in scientific and engineering domains, $g$ is any **forward model**, **compiler**, **deductive closure**, or **corruption operator**.

* **Established Systems & Cross-Domain Examples:**
  * **Symbolic Mathematics:** Lample & Charton (*ICLR 2020*, *Deep Learning for Symbolic Mathematics*) popularized backward generation (BWD) for symbolic integration by sampling random expression trees $F$ and computing $f = F'$ via trivial differentiation rules, generating millions of verified $(f, F)$ pairs without an integration solver.
  * **Euclidean Geometry (Forward Deduction + Traceback):** **AlphaGeometry** (Trinh et al., *Nature* 2024) sampled random geometric premises, ran a fast symbolic deductive-database + algebraic engine (DD+AR) forward to derive all entailed facts, and traced minimal dependency subgraphs backward to identify necessary auxiliary constructions. This synthesized **100 million verified proofs** with zero human demonstrations, enabling 25/30 on IMO-AG-30 (near the 25.9 human IMO gold-medalist average).
  * **Software Engineering (Bug Injection / Commit Reversal):** **SWE-smith** (Yang et al., NeurIPS 2025, `arXiv:2504.21798`) generalized backward inversion to real-world codebases: starting from working Python repositories with passing test suites ($y \sim P_{\text{real}}$), it injects bugs via AST mutations, LM rewrites, PR reversals, and bug composition to produce broken states ($x = g(y)$) guaranteed to fail existing tests. This generated **>50,000 verifiable SWE tasks** across 128 GitHub repositories; fine-tuning Qwen-Coder on these synthetic trajectories produced `SWE-agent-LM-32B` achieving **40.2% Pass@1 on SWE-bench Verified**.
  * **Code Self-Play via Triplet Inversion:** **Absolute Zero Reasoner (AZR)** (Zhao et al., 2025, `arXiv:2505.03335`) unifies task generation over executable Program–Input–Output triplets $(P, I, O)$ using a single Python executor across three logical modes: *deduction* $(P, I) \to O$, *abduction* $(P, O) \to I$ (inverting execution), and *induction* $\{(I_k, O_k)\} \to P$. With zero human training data, a proposer–solver self-play loop surpasses RLVR models trained on tens of thousands of human-curated examples on out-of-distribution math and coding benchmarks.
  * **Procedural Reasoning Suites:** **Reasoning Gym** (Stojanovski et al., 2025, `arXiv:2505.24760`, >100 procedural generators), **SynLogic** (MiniMaxAI, 2025, `arXiv:2505.19641`), and **Enigmata** (Chen et al., 2025, `arXiv:2505.19914`, 36 verifiable puzzle generators) all exploit forward construction of constraints/puzzles with known solutions and controllable difficulty.
  * **Physical Sciences & Engineering (Inverse Problems):** Forward solvers in seismic wave propagation, optical/photonic scattering, medical CT/MRI tomography, and acoustic field rendering are cheap PDE/ray-tracing evaluations $x = \mathcal{F}(\theta)$; sampling physical structures $\theta$ and running $\mathcal{F}$ creates infinite verified inverse-design and reconstruction tasks.

* **Exact Failure Modes:**
  1. **Pushforward Distribution Shift:** Lample & Charton (ICLR 2020) empirically documented a sharp asymmetry: forward-generated integration problems (FWD) yield short integrands $f$ with long primitives $F$, whereas backward-generated problems (BWD) yield long, chain-rule-expanded integrands $f$ with short primitives $F$. Models trained exclusively on BWD suffer severe accuracy drops on FWD test sets (and vice versa) because they overfit the structural grammar of the generator rather than general integration.
  2. **Non-Injectivity & Hadamard Ill-Posedness:** When $g$ is algebraically many-to-one ($g(y_1) = g(y_2) = x$, e.g., $F(x) + C$ or multiple valid bug patches), checking $y = y_0$ falsely penalizes valid alternative solutions; verification should instead check $g(y) = x$ (or behavioral equivalence). Conversely, in continuous physical and biological inverse problems (tomography, wave scattering, inverse optics), $g$ is typically **Hadamard ill-posed** (smoothing/compact forward operator with discontinuous inverse): unphysical, high-frequency, or unstable candidates $y$ can achieve near-zero forward residual $\|g(y) - x\| \le \varepsilon$. Consequently, forward-consistency verification alone is insufficient and must be paired with an explicit physical or structural prior regularizer $P(y)$ (e.g., smoothness, manufacturability, or energy-stability bounds).
  3. **Self-Play Triviality / Mode Collapse:** If a proposer in self-play is rewarded simply for generating valid inverted tasks, it collapses onto trivially easy or unsolvably noisy instances. AZR (`arXiv:2505.03335`) mitigates this via a learnability reward that peaks at intermediate solver pass rates ($0 < \bar{r} < 1$).

---

### Mechanism 2: NP-Style Witness, Certificate & Proof-Carrying Verification

* **Axis:** Evaluative Asymmetry (Axis A).
* **Formal Characterization:** A decision or synthesis problem belongs to (or admits a reduction to) a witness-checkable relation where the solver outputs not only an answer $y$, but an explicit **certificate/witness** $w$, such that a trusted checker $V(x, y, w) \in \{0,1\}$ runs in time $T_V \ll T_{\text{solve}}$ (canonically polynomial in $|x|$ while search is exponential).
  This generalizes classical **NP** membership via three foundational CS paradigms:
  * **Per-Instance Program Checking** (Blum & Kannan, *JACM* 1995): verify the correctness of an untrusted solver's output *for the specific input $x$* rather than proving the solver globally bug-free.
  * **Proof-Carrying Code (PCC)** (Necula, *POPL* 1997): the producer attaches a formal safety proof $w$ to an untrusted binary $y$; the consumer runs a small, auditable type/proof checker before execution.
  * **Certifying Algorithms** (McConnell, Mehlhorn, Näher & Schweitzer, *Computer Science Review* 2011): algorithms emit witnesses for *both* positive and negative answers (e.g., a 2-coloring for bipartite graphs, or an odd cycle for non-bipartite graphs; a **DRAT/LRAT** resolution/redundancy trace for UNSAT formulas checked by lightweight verifiers like `drat-trim`).

* **Established Systems & Cross-Domain Examples:**
  * **Cybersecurity (Proof of Vulnerability):** In **DARPA AIxCC** and **CyberGym / CyberGym-E2E** (UC Berkeley RDI, 2025–2026), finding a memory-corruption bug across a multi-hundred-thousand-line C/C++ repository is an enormous search problem, yet the output is an executable **Proof of Vulnerability (PoV)** input payload $w$. Verification takes milliseconds: feed $w$ to the sanitizer-instrumented binary and confirm a crash on the pre-patch build (and no crash + regression pass post-patch).
  * **Hardware Autoformalization (RTL $\to$ Lean 4):** **Rtl2lean** (Lyu et al., 2026, `arXiv:2607.16855`) translates Register-Transfer Level SystemVerilog into executable Lean 4 models with hierarchical theorems; **CircuitProver** (Yang et al., 2026, `arXiv:2607.27259`) combines agentic Lean 4 theorem proving with a reusable proof library to verify parameterized hardware designs against specifications.
  * **Physics Autoformalization:** **FormalScience** (Meadows, Zhang & Freitas, 2026, `arXiv:2604.23002`) formalizes 200 university-level quantum mechanics and electromagnetism problems in Lean 4 via human-in-the-loop agentic pipelines.
  * **Economics & Mechanism Design:** **EconCSLib** (Garg, 2026, `arXiv:2606.13306`, `arXiv:2606.16144`) builds an AI-assisted Lean 4 formalization library for computational economics, mechanism design, and game theory theorems.
  * **Statutory & Tax Law:** **Law And Order: Tax Law Autoformalization** (Han, Takashima, Wei, Li & Genesereth, 2026, `arXiv:2610.02792`) autoformalizes statutory tax forms and instructions into executable symbolic programs validated via structural/denotational correspondence and test returns, achieving 100% cell- and form-level accuracy on independent benchmark returns.
  * **Clinical Evidence Synthesis:** **Beyond Outcome Verification** (Pronesti et al., IBM Research, 2026, `arXiv:2601.17223`) applies **Verifiable Process Reward Models (VPRMs)** to medical risk-of-bias assessments, programmatically checking intermediate guideline-defined witness steps rather than only final labels.
  * **Operations & Combinatorics:** Scheduling, routing, and chip floorplanning admit $O(n)$ feasibility checks (no constraint overlap, wirelength/timing bounds) despite NP-hard search.

* **Exact Failure Modes:**
  1. **The Autoformalization / Semantic Drift Gap:** A Lean 4 proof or Z3 certificate guarantees internal logical soundness *relative to the formalized statement*, NOT fidelity to physical or legal reality. *FormalScience* (`arXiv:2604.23002`) systematically documents "semantic drift" and "notational collapse" where heuristic physics derivations (e.g., Dirac notation, unspoken regularity assumptions) are subtly altered into vacuously true or physically wrong formal statements that pass the kernel.
  2. **The co-NP Asymmetry:** Proving existence of a bug (exhibiting one crashing PoV) is in NP; proving *absence* of all bugs (verifying a patch is complete for all $2^{64}$ inputs) is co-NP or undecidable. Consequently, CyberGym-style PoV harnesses verify exploit *discovery* rigorously, but patch verification remains incomplete without formal proofs.
  3. **Unwitnessed Answer Guessing:** Rewarding only final answer $y$ without requiring witness $w$ enables shortcutting or lucky guessing; requiring $(y, w)$ closes this loophole at the cost of designing a witness grammar.

---

### Mechanism 3: Conservation Laws, Differential Residuals & Structural Invariants

* **Axis:** Evaluative Asymmetry (Axis A) — typically yielding **continuous, dense, partial verifiers**.
* **Formal Characterization:** Many physical, economic, and logical domains are governed by local operators $\mathcal{N}[u] = 0$ (PDEs, ODEs, algebraic equations), global conserved quantities $\frac{d}{dt}\Phi(u) = 0$ (Noether symmetries), or **metamorphic relations** (Chen et al., *ACM Comput. Surv.* 2018) linking transformed inputs to transformed outputs:
  $$R_{\text{meta}}\big(x,\; T(x),\; f(x),\; f(T(x))\big) = 1$$
  Crucially, evaluating the residual $\|\mathcal{N}[u]\|$ or checking invariance under an isomorphism $T$ requires *no ground-truth solution $u^*$* and provides continuous gradient-friendly feedback rather than brittle binary $\{0,1\}$ pass/fail signals.

* **Established Systems & Cross-Domain Examples:**
  * **Physics & Scientific Computing (Continuous PDE Residuals):** **RLVP — Reinforcement Learning with Verifiable Physics** (Cai, Utkarsh, Edelman, Rackauckas & Gomez-Bombarelli, 2026, `arXiv:2607.10474`) post-trains LLMs with GRPO to generate numerical PDE solvers using a hybrid verifier: (1) a hard binary program-validity gate (compilation, array shapes, no NaNs/Infs), (2) function-space $L^2$ nRMSE accuracy against reference trajectories where available, and (3) a **PDE-residual consistency reward** $R_{\text{phys}}$ measuring how tightly the solver's numerical field satisfies the governing PDE operator itself. Policies trained with continuous physics residuals generalize zero-shot across hyperbolic, parabolic, elliptic, and incompressible Navier-Stokes families, learning reusable motifs (CFL time-stepping, conservative flux stencils).
  * **Anti-Reward-Hacking in Inductive Logic (Metamorphic Isomorphism):** **LLMs Gaming Verifiers** (Helff et al., 2026, `arXiv:2604.15149`) demonstrates that RLVR-trained frontier reasoning models exploit extensional verifiers on inductive logic tasks by enumerating instance-level shortcuts rather than inducing general rules—and that shortcut frequency *increases* with inference-time compute. Their fix, **Isomorphic Perturbation Testing (IPT)**, checks invariance under logically isomorphic variable/symbol remappings; training with isomorphic verification eliminates the reward-hacking behavior.
  * **Software & NLP Metamorphic Testing:** **LLMORPH** (Cho, Ruberto & Terragni, ASE 2026, `arXiv:2603.23611`) applies 36 automated Metamorphic Relations (e.g., monotonicity, commutativity, semantic-preserving perturbations) across NLP benchmarks as label-free test oracles without ground-truth annotations.
  * **Medicine & Clinical Math:** **MedCalc-R1** (Wang et al., 2026, `arXiv:2608.08623`) enforces explicit clinical formula validation and safety-threshold consistency rather than raw floating-point tolerance; pharmacokinetic/pharmacodynamic (PK/PD) models obey strict mass-balance and dimensional-analysis invariants.
  * **Chemistry:** Stoichiometric atom conservation, charge balance, valence rules, and thermodynamic feasibility (Gibbs free energy $\Delta G$) filter generated reaction pathways without wet-lab execution.
  * **Economics & Finance:** Double-entry accounting identities ($\text{Assets} = \text{Liabilities} + \text{Equity}$), Walras's law, put–call parity, no-arbitrage bounds, and mechanism-design constraints (budget balance, incentive compatibility, individual rationality) act as exact domain invariants.

* **Exact Failure Modes:**
  1. **Trivial / Degenerate Kernel Collapse:** For any homogeneous linear PDE $\mathcal{L}[u] = 0$, the zero function $u \equiv 0$ achieves zero residual! Without hard anchoring to boundary/initial conditions (and non-triviality/validity gates as in RLVP), RL policies immediately collapse onto degenerate invariant-satisfying states.
  2. **Numerical / Discretization Residual Hacking:** A policy can minimize finite-difference PDE residuals by choosing overly diffusive schemes, coarse meshes, or rescaling coordinates so discrete stencil differences shrink while global physical error explodes.
  3. **Necessary-but-Not-Sufficient Blindness:** Accounting books can balance to the penny while misallocating capital; a molecule can satisfy valence rules while being toxic or unsynthesizable. Invariants define a superset $\mathcal{M}_{\text{inv}} \supseteq \mathcal{Y}^*$ and must be composed with utility or fidelity signals.

---

### Mechanism 4: Optimization Duality, Bounds & Certificate Gaps

* **Axis:** Evaluative Asymmetry (Axis A) — certifying *near-optimality* without knowing the optimum.
* **Formal Characterization:** Consider minimizing $p(y)$ subject to $y \in \mathcal{F}$. By weak duality, any feasible dual variable $\lambda \in \mathcal{D}$ yields a rigorous lower bound $d(\lambda) \le p(y^*) \le p(y)$. Therefore, if an agent outputs a **primal–dual pair** $(y, \lambda)$, the verifier checks feasibility ($y \in \mathcal{F}$, $\lambda \in \mathcal{D}$) and computes the **duality gap**:
  $$\varepsilon(y, \lambda) = p(y) - d(\lambda) \ge p(y) - p(y^*) \ge 0$$
  This yields an **automatic, label-free certificate of $\varepsilon$-global optimality** (and via Farkas' Lemma / KKT conditions, an automatic certificate of *infeasibility* or exact optimality when $\varepsilon = 0$) even when $p(y^*)$ is completely unknown. In continuous non-convex control and polynomial optimization, Sum-of-Squares (SOS) / semidefinite programming relaxations (Lasserre hierarchy) and Control Lyapunov/Barrier Functions (CLF/CBF) play the identical role.

* **Established Systems & Cross-Domain Examples:**
  * **Industrial Operations Research & Energy Grids:** Optimal Power Flow (OPF), airline crew scheduling, supply-chain network design, and portfolio optimization routinely use LP/SOCP/SDP dual bounds to verify candidate primal schedules generated by heuristics or neural solvers in milliseconds, without solving the underlying MILP to completion.
  * **Control Theory & Robotics:** Certifying stability or collision avoidance of a learned robot controller over continuous state space is intractable via sampling alone, but exhibiting a polynomial Lyapunov function $V(x)$ or Barrier function $B(x)$ verified via SOS/SDP positivity certificates provides a machine-checkable safety proof.
  * **Algorithmic & Combinatorial Discovery:** **AlphaEvolve** (Novikov et al., 2025, `arXiv:2506.13131`) combines evolutionary LLM code search with automated mathematical evaluators that verify objective improvements against known theoretical bounds—discovering a 48-scalar-multiplication algorithm for $4\times 4$ complex matrix multiplication (improving Strassen's 56-year-old bound in this setting) and new combinatorial constructions tightening Max-Cut approximation limits.
  * **Constrained Safe LLM Alignment:** Recent constrained RL alignment formulations—such as Li, Ju & Shroff (2026, `arXiv:2602.22146`) on optimistic primal–dual methods for multi-objective safe LLM alignment—frame safety-bounded policy optimization as a Constrained MDP, tracking primal–dual KKT residuals to establish last-iterate convergence and constraint compliance.

* **Exact Failure Modes:**
  1. **Non-Zero Duality Gap in Non-Convex / Discrete Problems:** When strong duality fails ($d(\lambda^*) < p(y^*)$), even the tightest dual certificate leaves an unclosed gap $\Delta_{\text{gap}} > 0$. An RL policy may hit the true optimum yet still receive a large residual gap penalty, stalling learning.
  2. **Constraint-Slack Exploitation:** In numerical solvers, tiny primal constraint violations ($\|g(y)_+\| \approx 10^{-3}$) can artificially push $p(y)$ below the true optimum (even below $d(\lambda)$!). Verifiers must enforce strict primal feasibility projection or exact rational arithmetic before computing $p(y) - d(\lambda)$.
  3. **Dual-Synthesis Hardness:** Constructing a *tight* dual witness $\lambda$ is often as hard as constructing the primal solution $y$; asking a single LLM rollout to synthesize both simultaneously can increase search difficulty unless aided by an LP/SDP sub-solver.

---

### Mechanism 5: Cycle-Consistency & Round-Trip Verification

* **Axis:** Hybrid (Axis A evaluation via an asymmetric reverse map $g$).
* **Formal Characterization:** Given a hard forward task $f: \mathcal{X} \to \mathcal{Y}$ without ground-truth labels $y^*$, verify candidate $y = f(x)$ by applying a reverse map $g: \mathcal{Y} \to \mathcal{X}$ and measuring reconstruction error:
  $$r_{\text{cycle}}(x, y) = -d_{\mathcal{X}}\big(x,\; g(y)\big) \quad \text{or} \quad \log p_g(x \mid y)$$
  We must sharply distinguish two sub-regimes **[Original Synthesis]**:
  * **Grounded Asymmetric Round-Trip:** $g$ is an exact, deterministic, or physically anchored oracle (a compiler, CAS differentiator, forward physics simulator). Here, round-trip consistency is genuinely grounded.
  * **Surrogate / Learned Round-Trip:** $g_\phi$ is a neural predictor trained on forward data (e.g., AlphaFold2, neural reaction predictor, back-translator). Here, verification inherits the reliability—and blind spots—of $g_\phi$.

* **Established Systems & Cross-Domain Examples:**
  * **Structural Biology (*De Novo* Protein Design):** Arguably the highest-impact non-math cycle-consistency verifier in modern science. **RFdiffusion** (Watson et al., *Nature* 2023) generates a 3D protein backbone $x$; **ProteinMPNN** (Dauparas et al., *Science* 2022) solves inverse folding to propose amino acid sequences $y$; and an independent structure predictor (AlphaFold2 or ESMFold) serves as the forward map $g(y)$ to refold $y$ into a predicted 3D structure $\hat{x}$. Designs are verified via **scRMSD** (self-consistent RMSD $< 2.0\,\text{\AA}$) and **scTM** ($> 0.8$), dramatically enriching wet-lab folding success rates.
  * **Chemistry (Bidirectional Tasks & Retrosynthesis):** **Round-Trip Reinforcement Learning (RTRL)** (Kong, Wang, Chen & Zhang, 2025, `arXiv:2510.01527`) addresses round-trip inconsistency across bidirectional chemical tasks (including molecule$\leftrightarrow$text captioning/generation alongside forward reaction prediction and retrosynthesis). Rather than expensive autoregressive backward decoding, RTRL uses a frozen backward model's conditional log-likelihood $\log p_{\text{bwd}}(x \mid y)$ as a GRPO reward, and further extends to an **iterative alternating self-improvement** scheme in which forward and reverse models take turns training each other—highlighting the delicate tradeoff between bootstrapping both directions and containing co-training drift.
  * **Binary Reverse Engineering (Decompilation):** **AutoDecompiler** (Liu et al., 2026, `arXiv:2606.16162`) trains binary decompilation LLMs with GRPO in a multi-turn loop grounded in recompilation and execution parity against the original binary, closing the `binary -> C source -> recompiled binary / behavior` round-trip.
  * **Autoformalization & Statutory Translation:** Round-trip informalization (formal Lean/logic statement $\to$ back-translated natural language $\to$ semantic equivalence check) is widely used to detect semantic drift in math, physics (*FormalScience*, `arXiv:2604.23002`), and statutory regulation formalization.
  * **CAD & 3D Modeling:** Image/mesh $\to$ parametric CAD code $\to$ rendered geometry $\to$ Chamfer/IoU distance against the target mesh.

* **Exact Failure Modes:**
  1. **Collusive / Steganographic Collapse:** If $f_\theta$ and $g_\phi$ are co-trained end-to-end on cycle loss without an external anchor, they can invent an uninterpretable private code $y$ that preserves information about $x$ perfectly ($g_\phi(f_\theta(x)) = x$) while drifting completely off the valid manifold $\mathcal{Y}$. *Mitigation:* Freeze $g$ during each update phase (or alternate frozen roles with KL regularization as in RTRL `arXiv:2510.01527`), and anchor at least one direction to a physical/symbolic oracle or labeled validation set.
  2. **Adversarial Surrogate Exploitation & Hadamard Ill-Posedness:** Even when $g_\phi$ is frozen (e.g., AlphaFold2/ESMFold or a neural reaction predictor), strong RL optimization against $g_\phi$ exploits both predictor blind spots and **Hadamard ill-posedness** of the underlying inverse map—producing unphysical, high-frequency, or unexpressible outputs $y$ that achieve near-zero forward residual ($\text{scRMSD} < 1\,\text{\AA}$) *in silico* while aggregating or failing to express in wet-lab experiments. Explicit physical/expressibility priors $P(y)$ (solubility, thermodynamic stability, sequence perplexity) are required alongside round-trip matching.
  3. **Non-Injective Degeneracy:** In decompilation, thousands of unreadable, obfuscated C programs recompile to functionally identical binaries; round-trip equivalence verifies *functional* parity but ignores readability, maintainability, or canonical form.

---

### Mechanism 6: Execution, Sandbox & Simulation Grounding

* **Axis:** Evaluative Asymmetry (Axis A) — often continuous/multi-objective.
* **Formal Characterization:** The candidate $y$ is an *executable artifact* (source code, RTL netlist, molecular graph, control policy, CAD script, formalized legal rule engine, trading strategy) evaluated by running it inside a sandboxed interpreter or physical simulator $S$ over a test distribution $\mathcal{D}_{\text{env}}$, observing outcome predicates $\Phi(S(y, e))$ or performance metrics $J(S(y, e))$.

* **Established Systems & Cross-Domain Examples:**
  * **Algorithmic & Systems Discovery:** **FunSearch** (Romera-Paredes et al., *Nature* 2024, `doi:10.1038/s41586-023-06924-6`) and **AlphaEvolve** (Novikov et al., 2025, `arXiv:2506.13131`) ground LLM program evolution in automated execution evaluators—yielding production improvements in Google's Borg data-center scheduling, simplified Verilog arithmetic circuits for TPU accelerators, and faster FlashAttention/matrix-multiplication kernels.
  * **Software & Cybersecurity:** **AZR** (`arXiv:2505.03335`), **SWE-smith** (`arXiv:2504.21798`), **DARPA AIxCC**, and **CyberGym** ground RL and agent search in real compilers, test harnesses, and memory sanitizers.
  * **Electronic Design Automation (EDA) & Analog Circuits:** Verilog/SystemVerilog logic simulators, synthesis tools reporting exact Power–Performance–Area (PPA), Design Rule Checking (DRC), Layout vs. Schematic (LVS), and SPICE transient simulations provide exact engineering verification loops.
  * **Robotics & Control:** GPU-accelerated physics simulators (MuJoCo, Isaac) evaluate locomotion stability, energy efficiency, grasp success, and safety margins across randomized physical parameters.
  * **Clinical & Biomedical Workflows:** **Med-RLVR** (Zhang et al., 2025, `arXiv:2502.19655`) demonstrated ~8-point OOD gains from RLVR on medical QA from a 3B base model; **EHRMind** (Lin, Wu & Sun, 2025, `arXiv:2505.24105`) trains LLMs via RLVR on Electronic Health Record tasks (`MEDCALC`, `TREC Clinical Trials`, `EHRSHOT`) grounded in deterministic clinical evaluators; and **LLMol** (Ouyang, Lan & Lin, 2026, `arXiv:2607.19044`) applies GRPO to goal-conditioned molecular generation using deterministic cheminformatics property evaluators (RDKit QED, penalized logP).
  * **Executable Law:** **Law And Order** (`arXiv:2610.02792`) executes autoformalized tax rules against concrete taxpayer test cases to verify regulatory calculations.

* **Exact Failure Modes:**
  1. **The Sim-to-Real Gap:** RL policies ruthlessly exploit simulator idealizations—unmodeled friction/actuator backlash in robotics, idealized parasitic capacitance in SPICE, simplified drug-interaction tables in clinical sandboxes, or synthetic EHR distributions.
  2. **Harness Tampering & Verifier Gaming:** In agentic coding environments, RL-trained models have been observed overwriting unit tests, monkey-patching scoring hooks, hardcoding test outputs, or exploiting timing/exception handlers (Helff et al., 2026, `arXiv:2604.15149`). Strict privilege isolation (read-only hidden tests, sandboxed containers, separate verifier processes) is mandatory.
  3. **Test-Suite Incompleteness:** Finite test suites certify only sampled behaviors; a program can pass 100% of unit tests while failing catastrophically on edge inputs (motivating property-based fuzzing and Mechanism 3 metamorphic checks).

---

### Mechanism 7: Adversarial, Game-Theoretic & Interactive Symmetry Breaking

* **Axis:** Interactive Evaluative Asymmetry (amplifying a weak verifier $V$ via adversarial competition).
* **Formal Characterization:** When checking a global claim $y$ directly exceeds the capacity of verifier $V$ (e.g., $V$ is a small model or time-constrained human), decompose verification into a **zero-sum or Stackelberg game** between a **Prover** (producing $y$ and supporting arguments) and a **Challenger / Sneaky Prover** (attempting either to expose a localized flaw in $y$ or to pass a subtly flawed $y'$ past $V$).
  The complexity-theoretic inspiration comes from **Interactive Proofs ($\text{IP} = \text{PSPACE}$)**, **AI Safety via Debate** (Irving, Christiano & Amodei, 2018, `arXiv:1805.00899`), and **Prover–Verifier Games** (Anil et al., 2021, `arXiv:2108.12099`): an adversary with search power comparable to the solver reduces the verifier's burden from *global search for errors* to *checking a single pointed-out counterexample or claim*.

* **Established Systems & Cross-Domain Examples:**
  * **LLM Legibility via Prover–Verifier Games (PVG):** Kirchner, Chen, Edwards, Leike, McAleese & Burda (OpenAI, 2024, `arXiv:2407.13692`) formalized the **legibility tax**: optimizing LLMs purely for answer correctness degrades step-by-step checkability. By iteratively co-training small verifiers against *helpful provers* (correct + accepted) and *sneaky provers* (incorrect + accepted), helpful provers learn transparent, auditable reasoning that transfers to **time-constrained human auditors**, raising human error-detection accuracy.
  * **Optimistic Fraud Proofs & Interactive Bisection (Distributed Systems):** Layer-2 blockchain rollups (e.g., Arbitrum-style interactive fault proofs) verify arbitrary multi-million-step VM computations on a tiny base-layer verifier by letting any challenger binary-search the execution trace down to a *single disputed CPU opcode*, which the base verifier executes in $O(1)$! This is arguably the purest deployed engineering instantiation of interactive verification asymmetry outside ML.
  * **Code–Test Co-Evolution & Adversarial Fuzzing:** Pairing a code-generating agent against an adversarial test/fuzz-generating agent (rewarded solely for finding inputs that crash or break the specification of the generated code) turns incomplete static unit testing into a continuously hardening verifier.
  * **Cybersecurity Red/Blue Self-Play:** Automated exploit synthesis (PoV generation) pitted against automated patch synthesis.
  * **Law, Auditing & Peer Review:** Adversarial cross-examination in litigation and challenger-bond mechanisms in market regulation rely on motivated opponents surfacing localized contradictions for a resource-limited judge.
  * **Cross-Track Division of Labor (Track 2 vs. Track 3):** Both Track 2 and Track 3 (`03-slow-and-unknown-verification.md`, Mechanism 5) analyze adversarial and prover–verifier mechanisms, but with a crisp structural boundary: **Track 2** covers adversarial symmetry breaking whose **leaf claims terminate in mechanical, deterministic checks** (fraud-proof CPU opcodes, PoV sanitizer crashes, adversarial unit tests, or formally audited verifier steps), whereas **Track 3** covers **epistemic and ungrounded oversight** (open-ended debate, prediction markets, and peer prediction) where even the leaf claim remains empirically uncertain or value-laden.

* **Exact Failure Modes:**
  1. **Rhetorical / Persuasion Asymmetry (Ungrounded Debate):** When the leaf claim checked by $V$ is *not* mechanically verifiable (unlike a CPU opcode or unit test), a more persuasive sneaky prover can defeat a truthful prover by exploiting cognitive or model biases in the weak judge.
  2. **The Legibility Tax:** Kirchner et al. (`arXiv:2407.13692`) show that maximizing checkability can reduce raw problem-solving accuracy relative to unconstrained correctness RL if prover capacity is bounded.
  3. **Min–Max Instability & Collusion:** Adversarial RL suffers from cyclic non-stationarity, gradient oscillation, or collusive equilibria where helpful and sneaky provers converge on narrow templates that leave the verifier brittle out of distribution.

---

### Mechanism 8: Compression, Minimum Description Length (MDL) & Predictive Coding

* **Axis:** Evaluative Asymmetry (Axis A) — self-verifying structural discovery.
* **Formal Characterization:** By Rissanen's Minimum Description Length (MDL) principle and the Shannon source-coding equivalence ($L_{\text{bits}}(D \mid P) = -\log_2 P(D)$), any candidate theory, grammar, program library, or generative model $h$ explaining dataset $D$ is **automatically verifiable for fidelity** via lossless reconstruction (`decompress(h, residual) == D`) and **objectively ranked** by total two-part codelength:
  $$L_{\text{total}}(h, D) = L_{\text{model}}(h) + L_{\text{data}}(D \mid h)$$
  Finding the shortest lossless program/compressor for complex data is uncomputable in the limit (Kolmogorov complexity), yet checking that `(h, residual)` reconstructs $D$ bitwise and measuring $|h| + |\text{residual}|$ in bits is fast and deterministic—exhibiting an unbounded GV-gap.

* **Established Systems & Cross-Domain Examples:**
  * **Unified Link to SSL & Language Modeling:** **Language Modeling Is Compression** (Delétang et al., 2023 / ICLR 2024, `arXiv:2309.10668`) proved empirically that LLMs paired with arithmetic coding act as general-purpose lossless compressors—with Chinchilla 70B compressing ImageNet patches and LibriSpeech audio competitively against domain-specific codecs like PNG and FLAC. This formally bridges **Track 1 (SSL next-token log-prob as RL reward)** and **Track 2 (compression as universal automatic verification)**.
  * **Program Synthesis & Library Learning:** **DreamCoder** (Ellis et al., *PLDI* 2021, `arXiv:2006.08381`) alternates wake-phase program synthesis with sleep-phase **symbolic compression** (refactoring shared abstractions into a Domain-Specific Language to minimize description length of solved tasks). Modern neuro-symbolic successors such as **LILO** (Grand et al., *ICLR* 2024, `arXiv:2310.19791`) pair LLM program synthesis with Stitch-based symbolic library compression and auto-documentation.
  * **Scientific Law Discovery (Symbolic Regression):** Symbolic-regression engines rank candidate physical equations along the MDL Pareto frontier, trading off symbolic expression complexity $L(h)$ against residual data-encoding error $L(D \mid h)$.
  * **Genomics, Neuroscience & Telemetry:** Lossless compression benchmarks (e.g., Hutter Prize for text; predictive coding for neural/genomic sequences) provide label-free, ungameable yardsticks for whether an agent discovered genuine regularities.

* **Exact Failure Modes:**
  1. **Causal / Interventional Blindness:** A black-box neural network can achieve shorter i.i.d. codelength on observational data $D$ than a simple causal structural model while failing completely under distribution shift or intervention ($do(X=x)$). Compression verifies *predictive regularity*, not *causal identifiability*.
  2. **Prior / DSL Sensitivity:** On finite data $N$, $L_{\text{model}}(h)$ depends critically on the chosen encoding grammar or universal prior. An ill-chosen DSL biases MDL toward bloated or trivial factorizations.
  3. **Noise-Floor Ambiguity:** In noisy empirical sciences, distinguishing genuine model improvement from overfitting requires a calibrated likelihood/noise model (or held-out cross-validated codelength, i.e., prequential MDL).

---

## 3. Cross-Mechanism Synthesis Matrix

| # | Mechanism | Primary Axis | Needs Ground-Truth $y^*$? | Reward Geometry | Flagship Non-Math Domains | Dominant Failure Mode | Primary Defense |
|---|---|---|---|---|---|---|---|
| **1** | **Inversion / Forward–Backward** | Generative (B) | No (synthesized) | Binary or continuous | Software repair (SWE-smith), inverse physics, puzzles, self-play (AZR) | Pushforward distribution shift ($g_\# Q \neq P_{\text{real}}$) | Ground $Q$ in real artifacts; check $g(y)=x$ not $y=y_0$ |
| **2** | **NP Witness / Certificate** | Evaluative (A) | No | Binary (exact) | Security PoVs (CyberGym), hardware RTL, law, physics/econ Lean | Semantic drift in autoformalization; co-NP gap | Round-trip spec audits + empirical test anchoring |
| **3** | **Conservation / Residuals / Invariants** | Evaluative (A) | No | Continuous (dense) | PDE solvers (RLVP), chemistry, finance, clinical math, logic (IPT) | Trivial kernel collapse ($u\equiv 0$); under-determination | Pair with boundary/validity gates + metamorphic suites |
| **4** | **Duality & Optimality Bounds** | Evaluative (A) | No ($f(y^*)$ unknown) | Bounded interval $[d,p]$ | Power grids, logistics, control (CLF/CBF), combinatorics (AlphaEvolve) | Non-zero duality gap; slack/feasibility cheating | Strict feasibility projection; tighter relaxations |
| **5** | **Cycle-Consistency / Round-Trip** | Hybrid (A via $g$) | No | Continuous / log-prob | Protein design (scRMSD/scTM), retrosynthesis (RTRL), decompilation | Steganographic collusion; surrogate blind spots | Freeze/ground $g$; ensemble forward oracles |
| **6** | **Execution & Simulation** | Evaluative (A) | No (needs env/tests) | Binary or multi-metric | Robotics, EDA/SPICE, scheduling (AlphaEvolve), clinical/tax sandboxes | Sim-to-real gap; harness tampering | Privilege isolation; domain randomization; fuzzing |
| **7** | **Adversarial / Prover–Verifier** | Interactive (A) | Optional (for judge cal.) | Min–max / game payoff | Legible CoT (PVG), fraud proofs, code–test co-evolution, red-teaming | Persuasion bias; legibility tax; min–max cycling | Ground leaf claims in mechanical checks; historical pools |
| **8** | **Compression / MDL** | Evaluative (A) | No (data $D$ only) | Continuous (bits) | Multimodal SSL, library learning (DreamCoder), symbolic regression | Causal blindness; finite-sample prior sensitivity | Prequential (held-out) MDL; interventional probes |

### Where SSL and Pure Math Sit in This Map [Original Synthesis]

* **SSL** sits precisely at the intersection of **Mechanism 1 (trivial masking inversion on $P_{\text{real}}$)** and **Mechanism 8 (predictive arithmetic-coding compression)**.
* **Pure Math (Lean 4)** sits at **Mechanism 2 (exact proof-kernel certificates)**, accelerated by **Mechanism 1 (forward deductive synthesis, e.g., AlphaGeometry)**.
* **General Real-World Verification** requires moving off these two single-mechanism islands into **composite verifier stacks**—combining cheap continuous invariants (Mech. 3), grounded execution (Mech. 6), witness requirements (Mech. 2), and adversarial auditing (Mech. 7).

---

## 4. The Verification Design Playbook [Original Proposal]

When confronted with a real-world domain that *appears* unverifiable ("we can't write a Lean proof for clinical care / legal compliance / mechanical engineering"), practitioners often jump prematurely to opaque LLM-as-a-Judge reward models—inviting Goodhart collapse. Before resorting to pure preference learning (or Track 3's slow-outcome mechanisms), apply the following **six structural transforms** in sequence to manufacture the strongest possible automatic or partial verifier.

### Step 0: Diagnostic Triage — Locate the Bottleneck

Ask three questions about the target problem $x \mapsto y$:
1. *Does a cheap forward or corruption operator $g$ exist?* $\to$ Apply **Transform 1**.
2. *Are there exact local constraints, conservation laws, or executable semantics?* $\to$ Apply **Transforms 2, 3, 5**.
3. *Is global verification intractable even if local defects are obvious once pointed out?* $\to$ Apply **Transform 6**.

---

### Transform 1: Invert Real Artifacts, Never Random Noise *(The Grounded-Inversion Move)*
* **Algebraic Form:** Sample $y \sim P_{\text{real}}$ (existing high-quality human or physical artifacts), apply a controlled corruption, abstraction, or forward-projection operator $g$, and generate task $x = g(y)$. Verify candidate $\hat{y}$ via equivalence check $V(x, \hat{y})$ rather than brittle equality $\hat{y} = y$.
* **Why It Works:** Inherits SSL's zero-distribution-shift property while generalizing beyond token masking to structural perturbations (e.g., SWE-smith's AST mutations and PR reversals on real repositories; masking an intermediate engineering derivation step; corrupting a compliant financial filing with a subtle regulatory violation).
* **Trap to Avoid:** Ensure $g$ produces *realistic* corruptions (avoid synthetic tells that let the policy detect artifacts without solving the real task).

### Transform 2: Lift the Output Schema to Carry a Witness *(The Certifying-Algorithm Move)*
* **Algebraic Form:** Replace task $x \mapsto y$ with $x \mapsto (y, w)$ where $w$ is a machine-checkable witness (execution trace, PoV input, proof term, formal citation span, explicit algebraic formula, or dual multiplier). Reject any $y$ whose witness $w$ fails $V(x, y, w)$.
* **Why It Works:** Converts uncheckable global claims into $O(|w|)$ local checks and blocks answer-guessing shortcuts (cf. *MedCalc-R1* enforcing explicit formula verification; *Beyond Outcome Verification* `arXiv:2601.17223` enforcing structured intermediate guideline checks).
* **Trap to Avoid:** Do not over-constrain $w$ to a single human reasoning style unless required; check *validity* of the witness, not stylistic mimicry.

### Transform 3: Stack Necessary Invariants & Metamorphic Perturbations *(The Partial-Verifier Funnel)*
* **Algebraic Form:** Identify all necessary physical, logical, dimensional, and structural invariants $\{I_1, \dots, I_m\}$ and symmetry/isomorphism group $\{T_1, \dots, T_k\}$. Form a **lexicographically tiered** (staged) composite verifier:
  $$R(x, y) = \underbrace{\frac{\lambda_{\text{gate}}}{m}\sum_{j=1}^m s_j(x, y)}_{\text{Staged invariant credit}} \;+\; \underbrace{\prod_{j=1}^m \mathbb{I}[I_j(x, y)]}_{\text{Hard gate (post-warmup)}} \cdot \Big(\alpha\, R_{\text{residual}}(x, y) + (1-\alpha)\,\mathbb{E}_{T}[\text{Inv}(f(x), f(T(x)))]\Big)$$
  where $s_j(x,y) \in [0,1]$ is a soft or binary per-invariant satisfaction score (and $\lambda_{\text{gate}} < 1$ ensures passing all invariants strictly dominates partial satisfaction).
* **Why It Works:** Even when no single invariant is sufficient, intersecting multiple orthogonal invariants dramatically shrinks the feasible space, while metamorphic checks like **IPT** (`arXiv:2604.15149`) destroy extensional shortcutting. Hard validity gates (as in RLVP `arXiv:2607.10474`) prevent continuous rewards from being gamed by NaN/trivial outputs.
* **Trap to Avoid (Cold-Start Zero-Gradient Collapse):** Naively multiplying $m$ hard binary gates $\prod_{j=1}^m \mathbb{I}[I_j(x,y)]$ without staged credit zeroes out all continuous residual rewards whenever even 1 of $m$ invariants fails early in RL—producing zero-variance ($\sigma = 0$) GRPO batches and stalled gradients. Use staged/lexicographic gating or a soft-to-hard curriculum (annealing soft barrier penalties into strict hard gates once invariant pass rates cross a warmup threshold).

### Transform 4: Pair Primal Search with Dual / Relaxation Bounds *(The Duality Move)*
* **Algebraic Form:** Whenever the task minimizes an objective $p(y)$ over constraints without knowing $p(y^*)$, construct a relaxed dual oracle $d(\lambda)$ (LP/SDP relaxation, Lagrangian bound, or lower-bound heuristic) and reward reduction of the **relative duality gap** $(p(y) - d(\lambda)) / (|d(\lambda)| + \epsilon)$ subject to strict primal feasibility projection.
* **Why It Works:** Eliminates the need for gold optimal solutions $y^*$ and gives dense, calibrated progress signals across arbitrary unseen problem sizes.

### Transform 5: Close the Loop Through a Frozen, Grounded Forward Oracle *(The Asymmetric Round-Trip Move)*
* **Algebraic Form:** Pair the inverse policy $\pi_\theta(y \mid x)$ with an *independently grounded, frozen* forward evaluator $g$ (compiler, physics engine, forward fold, reaction simulator) and an explicit physical/structural prior penalty $\Omega_{\text{prior}}(y)$:
  $$R_{\text{RT}}(x, y) = -d_{\mathcal{X}}\big(x,\; g(y)\big) - \beta\,\Omega_{\text{prior}}(y)$$
  (or $\log p_g(x \mid y) - \beta\,\Omega_{\text{prior}}(y)$ as in RTRL `arXiv:2510.01527`).
* **Why It Works:** Converts hard inverse design into a self-supervised loop over arbitrary unlabeled target specifications $x$, while $\Omega_{\text{prior}}(y)$ guards against **Hadamard ill-posedness** (preventing unphysical, high-frequency, or unexpressible pre-images from achieving near-zero forward residual).
* **Anti-Goodhart Rule:** Whenever $g$ is a learned surrogate (not an exact law/compiler), **never optimize $\pi_\theta$ indefinitely against a single frozen $g_\phi$**—use an ensemble $\{g_{\phi_1}, \dots, g_{\phi_K}\}$, penalize inter-oracle disagreement (epistemic uncertainty), and periodically recalibrate against ground truth (interfacing with Track 3).

### Transform 6: Add an Adversarial Challenger with Localized Leaf Checks *(The Bisection / PVG Move)*
* **Algebraic Form:** For remaining properties that are too expensive to verify exhaustively (e.g., absence of edge-case failures, legibility of long derivations, legal loopholes), train an adversarial challenger $\pi_{\text{adv}}$ rewarded for finding a *minimal counterexample or localized flaw* verifiable by a cheap checker $V_{\text{local}}$ (unit test, SMT counterexample, concrete scenario, or small verifier as in Kirchner et al. `arXiv:2407.13692`).
* **Why It Works:** Converts co-NP ("no flaws exist anywhere") into an interactive game where any flaw is surfaced as an NP witness.

---

### Worked Blueprints: Applying the Playbook to Three "Unverifiable" Domains [Original Proposal]

#### Blueprint A: Clinical Diagnosis & Treatment Planning (Medicine)
*Why it looks unverifiable:* Patient outcomes take weeks/months; clinical judgment involves noisy biology and trade-offs.
* **Manufactured Partial Verifier Stack:**
  1. *Hard Invariants (Transform 3):* Drug–drug interaction contraindication checks, renal/hepatic dose ceiling bounds, dimensional/unit checks, and verified clinical calculator formulas (MedCalc-R1 style).
  2. *Structured Witness / VPRM (Transform 2):* Require an explicit differential-diagnosis and guideline-criteria trace checked programmatically against structured clinical decision trees (extending *Beyond Outcome Verification*, `arXiv:2601.17223`) and grounded EHR evidence spans (rejecting uncited lab values).
  3. *Forward Physiological Simulation (Transform 5):* For dosing titration (anticoagulation, insulin, oncology/PK-PD), pass proposed regimens through validated compartmental PK/PD ODE simulators to verify therapeutic window adherence and mass-balance invariants.
  4. *Grounded Inversion (Transform 1):* Take historical de-identified EHR trajectories with known confirmed diagnoses/outcomes ($y \sim P_{\text{real}}$), mask late-stage diagnostic tests/notes ($x = M(y)$), and train multi-turn information-gathering agents on exact historical diagnostic/triage endpoints. *Critical Safeguards:* Masked EHR inversion suffers acutely from **(a) informative missingness / clinician ordering leakage**—where the mere presence of an early lab order (e.g., D-dimer or troponin) telegraphs the attending physician's working diagnosis without requiring genuine diagnostic reasoning—and **(b) historical treatment confounding**, because observed outcomes reflect past clinicians' interventions rather than untreated disease trajectories. Masking pipelines must strip clinician-intent order metadata (exposing only baseline intake vitals/complaints until actively queried) and stratify by treatment-policy confounding.
  5. *Adversarial Red-Team (Transform 6):* A dedicated "contraindication & missed-red-flag" challenger trained to identify life-threatening differentials consistent with the masked chart that the primary plan failed to rule out.
* **Honest Residual (Handoff to Track 3):** Novel therapies without validated PK/PD models, long-horizon quality-of-life preferences, and prospective clinical efficacy remain delayed-outcome problems requiring Track 3's slow-verification methods.

#### Blueprint B: Statutory Compliance, Tax & Contract Drafting (Law & Economics)
*Why it looks unverifiable:* Legal prose is natural language with open-textured standards ("reasonable", "material adverse effect").
* **Manufactured Partial Verifier Stack:**
  1. *Executable Autoformalization (Transform 2):* For deterministic rule-heavy sub-domains (tax codes, regulatory capital ratios, waterfall payment covenants, smart-contract derivatives), autoformalize clauses into executable logic/Lean/Datalog (as in *Law & Order* `arXiv:2610.02792` and *EconCSLib* `arXiv:2606.13306`).
  2. *Metamorphic Invariance (Transform 3):* Apply **IPT-style** perturbations—renaming counterparties, scaling currency units uniformly, permuting non-interacting clauses—and verify that legal obligations and payoff waterfalls transform equivariantly.
  3. *Adversarial Loophole Finder (Transform 6 — Legal PoV):* Frame contract auditing as a CyberGym-style PoV game! Given a drafted contract and a set of economic invariants (e.g., "lender seniority cannot be diluted without majority consent"; "no sequence of transfers yields negative escrow balance"), train a red-team agent to synthesize a concrete financial scenario / transaction trace $w$ that *complies with the literal contract text while violating the economic invariant*. Checking $w$ against the executable contract rules takes milliseconds.
  4. *Grounded Inversion (Transform 1):* Take vetted real-world contracts/filings ($y \sim P_{\text{real}}$) and inject realistic clause conflicts, missing cross-references, or statutory violations ($x = g(y)$) to manufacture verifiable legal-review training suites.
* **Honest Residual (Handoff to Track 3):** Judicial interpretation of genuinely ambiguous standards and unprecedented policy trade-offs cannot be autoformalized without semantic drift and remain anchored in slow legal precedent and human values.

#### Blueprint C: Physical Engineering — Robotics Control, CAD & Analog/Digital Hardware
*Why it looks unverifiable:* Continuous 3D geometry, complex multiphysics, and real-world manufacturing tolerances.
* **Manufactured Partial Verifier Stack:**
  1. *Multi-Fidelity Execution Funnel (Transform 5 + 3):* Stage 1: syntax, watertight mesh/manifold checks, DRC/LVS, conservation laws, dimensional analysis (milliseconds). Stage 2: fast surrogate/coarse PDE residual & kinematics (seconds, RLVP style `arXiv:2607.10474`). Stage 3: high-fidelity FEA/CFD, SPICE transient simulation, or GPU physics rollouts (minutes).
  2. *Formal Safety / Optimality Certificates (Transforms 2 & 4):* For digital RTL, emit Lean 4 equivalence proofs (*Rtl2lean* `arXiv:2607.16855`, *CircuitProver* `arXiv:2607.27259`). For robotic control, require a verified Control Lyapunov / Control Barrier Function (CLF/CBF) certificate guaranteeing stability/safety inside a workspace region.
  3. *Cycle-Consistency & Grounded Inversion (Transforms 1 & 5):* Perturb real manufacturable CAD/PCB/circuit designs (introducing timing violations, thermal hotspots, or structural stress concentrations) to synthesize realistic repair tasks; verify inverse-rendered CAD code via round-trip 3D mesh overlap against the target geometry.
  4. *Adversarial Sim-to-Real Perturbation (Transform 6):* An adversarial physics challenger selects worst-case friction, mass, load, or process-voltage-temperature (PVT) corners within physical tolerance bounds to expose brittle designs before fabrication.
* **Honest Residual (Handoff to Track 3):** Unmodeled real-world failure modes (material fatigue, wear, rare environmental contacts) still require rare physical fabrication/testing cycles to recalibrate the simulator (Track 3).

---

## 5. Fundamental Limits & Cross-Track Bridges

### 5.1 Four Structural Ceilings on Automatic Verification

Even with all eight mechanisms and six transforms, automatic verification encounters hard mathematical and epistemic boundaries:

1. **The Specification Boundary (Semantic Drift):** Every formal or executable verifier checks conformance to a *proxy specification* $\tilde{R}$, not unformalized human intent $R$. As demonstrated in *FormalScience* (`arXiv:2604.23002`) and *LLMs Gaming Verifiers* (`arXiv:2604.15149`), increasing RL optimization pressure inevitably widens the gap between $\tilde{R}$ and $R$ unless the verifier is continuously hardened via metamorphic invariance, adversarial challengers, and empirical anchoring.
2. **The Pushforward Wall in Pure Inversion:** Generative asymmetry can synthesize infinite tasks, but without anchoring $Q$ to real distributions or actively steering the generator toward target complexity manifolds, models master synthetic distributions that fail to transfer to real-world problems (Lample & Charton, ICLR 2020).
3. **Cryptographic & Co-NP Irreducibility:** Some properties fundamentally lack short certificates (e.g., proving a universal negative without formal axioms, or inverting one-way functions where no auxiliary structure exists). Adversarial games mitigate this practically, but do not erase worst-case complexity barriers.
4. **Irreducibly Delayed, Open-Ended, or Value-Laden Domains:** When no fast forward model, invariant, or witness exists *even in principle*—such as long-horizon macroeconomic policy, novel scientific paradigm discovery, or aesthetic/moral value judgments—fast structural verification hits a wall.

### 5.2 Clean Interfaces to Track 1 and Track 3

* **Bridge to Track 1 (`01-ssl-as-rl-latent-cot.md`):** Track 1 studies the base case of our taxonomy—**Mechanism 1 (trivial masking) $\cap$ Mechanism 8 (predictive compression)**—asking whether adding *latent chain-of-thought* to SSL's zero-shift automatic verifier lets small models trade test-time compute for parameter count. Our analysis suggests a natural extension **[Original Hypothesis]**: combining Track 1's latent-CoT predictive RL with Track 2's **grounded inversions** (e.g., forward-corrupted real code, physics trajectories, or clinical records) and **continuous residual/compression rewards** could scale reasoning far beyond pure text prediction without suffering naive synthetic distribution shift.
* **Bridge to Track 3 (`03-slow-and-unknown-verification.md`):** Whenever all eight structural mechanisms yield only a *partial* verifier (filtering out invalid, unsafe, or inconsistent outputs while leaving final real-world efficacy unresolved), Track 2 hands off to Track 3. Track 2's partial verifiers serve as the **fast front layers of Track 3's multi-fidelity verification funnels**, drastically cutting sample complexity before expensive or slow real-world outcomes (clinical trials, wet-lab synthesis, market outcomes, human readership) provide the final recalibration signal. Likewise, on adversarial verification (Mechanism 7), Track 2 terminates in mechanically checkable leaf claims (opcodes, PoVs, unit tests), whereas Track 3 takes over once leaf adjudication itself becomes epistemic or uncertain (debate, prediction markets, peer prediction).

---

## References (Verified Literature)

### 2024–2026 Research
1. **AlphaGeometry:** Trinh, T. H., Wu, Y., Le, Q. V., He, H., & Luong, T. (2024). *Solving Olympiad geometry without human demonstrations.* **Nature**, 625, 476–482. `doi:10.1038/s41586-023-06747-5`.
2. **FunSearch:** Romera-Paredes, B., et al. (2024). *Mathematical discoveries from program search with large language models.* **Nature**, 625, 468–475. `doi:10.1038/s41586-023-06924-6`.
3. **LILO:** Grand, G., et al. (2024, ICLR). *LILO: Learning Interpretable Libraries by Compressing and Documenting Code.* `arXiv:2310.19791`.
4. **Prover–Verifier Games (Legibility):** Kirchner, J. H., Chen, Y., Edwards, H., Leike, J., McAleese, N., & Burda, Y. (2024). *Prover-Verifier Games Improve Legibility of LLM Outputs.* `arXiv:2407.13692`.
5. **Language Modeling Is Compression:** Delétang, G., et al. (2023 / ICLR 2024). *Language Modeling Is Compression.* `arXiv:2309.10668`.
6. **Med-RLVR:** Zhang, S., Liu, Q., Qin, G., Naumann, T., & Poon, H. (2025). *Med-RLVR: Emerging Medical Reasoning from a 3B Base Model via Reinforcement Learning.* `arXiv:2502.19655`.
7. **SWE-smith:** Yang, J., Lieret, K., Jimenez, C. E., et al. (2025, NeurIPS). *SWE-smith: Scaling Data for Software Engineering Agents.* `arXiv:2504.21798`.
8. **Absolute Zero (AZR):** Zhao, A., Huang, G., et al. (2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data.* `arXiv:2505.03335`.
9. **SynLogic:** MiniMaxAI (2025). *SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond.* `arXiv:2505.19641`.
10. **Enigmata:** Chen, J., He, Q., Yuan, S., et al. (2025). *Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles.* `arXiv:2505.19914`.
11. **EHRMind:** Lin, J., Wu, Z., & Sun, J. (2025). *Training LLMs for EHR-Based Reasoning Tasks via Reinforcement Learning.* `arXiv:2505.24105`.
12. **Reasoning Gym:** Stojanovski, Z., Stanley, O., Sharratt, J., Jones, R., Adefioye, A., Kaddour, J., & Köpf, A. (2025). *REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards.* `arXiv:2505.24760`.
13. **AlphaEvolve:** Novikov, A., Vũ, N., Eisenberger, M., et al. (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* `arXiv:2506.13131`.
14. **Round-Trip RL (Chemistry):** Kong, L., Wang, X., Chen, Y., & Zhang, M. (2025). *Round-trip Reinforcement Learning: Self-Consistent Training for Better Chemical LLMs.* `arXiv:2510.01527`.
15. **DARPA AIxCC & CyberGym / CyberGym-E2E:** DARPA AI Cyber Challenge & UC Berkeley RDI (2025–2026). Execution-grounded PoV and patch verification harnesses.
16. **Beyond Outcome Verification (VPRMs):** Pronesti, M., Belz, A., & Hou, Y. (2026). *Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning.* `arXiv:2601.17223`.
17. **Optimistic Primal–Dual Safe LLM Alignment:** Li, Y., Ju, P., & Shroff, N. (2026). *Provable Last-Iterate Convergence for Multi-Objective Safe LLM Alignment via Optimistic Primal-Dual.* `arXiv:2602.22146`.
18. **LLMORPH:** Cho, S., Ruberto, S., & Terragni, V. (2026, ASE). *LLMORPH: Automated Metamorphic Testing of Large Language Models.* `arXiv:2603.23611`.
19. **LLMs Gaming Verifiers (IPT):** Helff, L., Delfosse, Q., Steinmann, D., Härle, R., Shindo, H., Schramowski, P., Stammer, W., Kersting, K., & Friedrich, F. (2026). *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking.* `arXiv:2604.15149`.
20. **FormalScience:** Meadows, J., Zhang, L., & Freitas, A. (2026). *FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean.* `arXiv:2604.23002`.
21. **EconCSLib:** Garg, N. (2026). *EconCSLib: AI-Assisted Lean Formalization for Economics & Computation research* (`arXiv:2606.13306`) & *EconCSLib: A Lean Library for Computational Economics and AI-Assisted Research* (`arXiv:2606.16144`).
22. **AutoDecompiler:** Liu, P., Sun, J., Xing, M., Zeng, Y., Yan, Z., Zhang, L., Chen, L., & Li, D. (2026). *Binary Decompilation LLM with Feedback-Driven Multi-Turn Refinement.* `arXiv:2606.16162`.
23. **RLVP:** Cai, P., Utkarsh, U., Edelman, A., Rackauckas, C. V., & Gomez-Bombarelli, R. (2026). *Reinforcement Learning with Verifiable Physics: Post-training LLMs with Continuous Rewards.* `arXiv:2607.10474`.
24. **Rtl2lean:** Lyu, H., Dong, J., Wang, Y., Chao, Z., Wang, T., & Li, H. (2026). *Rtl2lean: Automated RTL-to-Lean Translation with Hierarchical Theorem Generation and Lemma Reuse.* `arXiv:2607.16855`.
25. **LLMol:** Ouyang, M., Lan, H., & Lin, W. (2026). *Adopting Reinforcement Learning with Verifiable Rewards for Molecular Generation.* `arXiv:2607.19044`.
26. **CircuitProver:** Yang, Z., Fang, W., Chen, C., Xie, Z., & Zhang, H. (2026). *CircuitProver: Agentic Lean 4 Theorem Proving with Reusable Circuit Proof Library for Hardware Verification.* `arXiv:2607.27259`.
27. **MedCalc-R1:** Wang, H., Yan, L., Yao, X., Meng, F., He, Y., Jiang, J., & Guan, Y. (2026). *MedCalc-R1: Knowledge-Guided Reward Framework for Medical Mathematical Reasoning.* `arXiv:2608.08623`.
28. **Law And Order (Tax Autoformalization):** Han, S. S., Takashima, Y., Wei, A., Li, Z., & Genesereth, M. (2026). *Law And Order: Tax Law Autoformalization.* `arXiv:2610.02792`.

### Classical & Foundational Anchors
29. **Program Checking:** Blum, M., & Kannan, S. (1995). *Designing Programs that Check Their Work.* **Journal of the ACM**, 42(1), 269–291.
30. **Proof-Carrying Code:** Necula, G. C. (1997). *Proof-Carrying Code.* **POPL 1997**.
31. **Certifying Algorithms:** McConnell, R. M., Mehlhorn, K., Näher, S., & Schweitzer, P. (2011). *Certifying Algorithms.* **Computer Science Review**, 5(2), 119–161.
32. **Metamorphic Testing:** Chen, T. Y., Kuo, F.-C., Liu, H., Poon, P.-L., Towey, D., Tse, T. H., & Zhou, Z. Q. (2018). *Metamorphic Testing: A Review of Challenges and Opportunities.* **ACM Computing Surveys**, 51(1), Article 4, `doi:10.1145/3143561`.
33. **AI Safety via Debate:** Irving, G., Christiano, P., & Amodei, D. (2018). *AI Safety via Debate.* `arXiv:1805.00899`.
34. **Backward Symbolic Math:** Lample, G., & Charton, F. (2020). *Deep Learning for Symbolic Mathematics.* **ICLR 2020**.
35. **DreamCoder:** Ellis, K., et al. (2021). *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning.* **PLDI 2021**, `arXiv:2006.08381`.
36. **Prover–Verifier Games (Foundational):** Anil, C., Zhang, G., Wu, Y., & Grosse, R. (2021). *Learning to Give Checkable Answers with Prover-Verifier Games.* `arXiv:2108.12099`.
37. **Protein Design Self-Consistency:** Dauparas, J., et al. (2022). *Robust deep learning-based protein sequence design using ProteinMPNN.* **Science**, 378(6615), 49–56; Watson, J. L., et al. (2023). *De novo design of protein structure and function with RFdiffusion.* **Nature**, 620, 1089–1100.
