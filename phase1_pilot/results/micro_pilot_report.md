# Phase 1 Small-Model Latent-CoT RL Pilot — CPU Micro-Scale Report

> **Scope Notice (Proof-of-Life & Pipeline Validation):** These results are from a live, reproducible **CPU micro-scale run** (`micro-small` ~0.9M params vs `micro-large` ~2.5× params, trained for 80 supervised steps + 20 RL steps on synthetic 3-regime data in 93.1s on CPU). This validates end-to-end training stability, anti-sabotage invariants, and all 5 Track 1 §5.5 diagnostics. It is **distinct from** the full 160M / 410M / 1.0B H100 Stage 1 run (`scripts/launch_h100_stage1.sh`, ~30–45 H100 GPU-hrs).

- **Generated:** 2026-11-23 22:39 UTC
- **Seed:** `42` | **Train samples:** `600` | **Eval samples (Regime 1 / Regime 3):** `64` / `48`
- **Supervised steps:** `80` (`batch_size=16`) | **RL steps:** `20` (`batch_size=8`, `G=4`, `thought_len=6`, `n_ahead=4`)
- **Support-Boundary Probe (`pass@k`):** `n=16` rollouts/prompt across `d in {2, 4, 6, 8}`

---

## 1. Track 1 §5.5 Falsification Verdict (CPU Micro-Scale)

**Verdict Status:** `FALSIFIED`

Falsified: no small-model latent-CoT or RL arm simultaneously beat pause control B3 (0.0%) and FLOPs-matched B2 (6.2%). Best candidate `I2-Hybrid` reached 3.1% at 2.51x B2 FLOPs.

### Summary Comparison Table (All 7 Arms)

| Arm | Params | FLOPs/tok (vs B2) | Exact Acc | Hard-BPB | OOD pass@max_k | ΔNLL_swap | ΔNLL_trunc | Mono Depth | Reg3 ECE | Reg3 Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| `B1` | 984,192 | 0.34× | 3.1% | 2.318 | 31.2% | — | — | — | 0.005 | -0.994 |
| `B2` | 2,853,312 | 1.00× | 6.2% | 2.266 | 31.2% | — | — | — | 0.040 | -1.006 |
| `B3` | 984,704 | 1.71× | 0.0% | 2.317 | 31.2% | — | — | — | 0.032 | -1.003 |
| `B4` | 1,000,704 | 1.74× | 3.1% | 2.272 | 18.8% | -0.001 | +2.909 | No | 0.045 | -0.989 |
| `RL-R1` | 984,192 | 3.10× | 3.1% | 2.326 | 31.2% | -0.002 | +0.001 | No | 0.026 | -0.997 |
| `RL-R3` | 984,192 | 3.10× | 1.6% | 2.309 | 25.0% | +0.000 | -0.001 | No | 0.040 | -1.005 |
| `I2-Hybrid` | 1,033,600 | 2.51× | 3.1% | 2.271 | 37.5% | -0.003 | +2.732 | No | 0.032 | -0.998 |

---

## 2. Diagnostic 1 & 2 — Accuracy & `pass@k` by Hop Depth (ID `d={2,4}` vs OOD `d={6,8}`)

| Arm | Greedy Acc `d=2` | Greedy Acc `d=4` | Greedy Acc `d=6` (OOD) | Greedy Acc `d=8` (OOD) | ID `pass@1` | OOD `pass@1` | OOD `pass@16` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1` | 12.5% | 0.0% | 0.0% | 0.0% | 3.1% | 2.0% | 31.2% |
| `B2` | 6.2% | 0.0% | 6.2% | 12.5% | 2.7% | 2.3% | 31.2% |
| `B3` | 0.0% | 0.0% | 0.0% | 0.0% | 2.3% | 2.7% | 31.2% |
| `B4` | 12.5% | 0.0% | 0.0% | 0.0% | 5.1% | 1.2% | 18.8% |
| `RL-R1` | 6.2% | 0.0% | 0.0% | 6.2% | 3.9% | 2.0% | 31.2% |
| `RL-R3` | 0.0% | 0.0% | 0.0% | 6.2% | 2.0% | 2.3% | 25.0% |
| `I2-Hybrid` | 0.0% | 0.0% | 6.2% | 6.2% | 3.5% | 2.7% | 37.5% |

---

## 3. Diagnostic 3 — Causal Thought Faithfulness (Swap, Format-Matched Truncation & Depth Scaling)

| Arm | Mode | NLL(thought) | ΔNLL_swap (nats) | ΔNLL_trunc (nats) | Swap `d=2` | Swap `d=4` | Swap `d=6` | Swap `d=8` | Monotonic? |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `B1` | `ntp` | 3.440 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | No |
| `B2` | `ntp` | 3.362 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | No |
| `B3` | `pause` | 3.437 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | No |
| `B4` | `continuous` | 3.371 | -0.0008 | +2.9091 | +0.0141 | -0.0125 | +0.0051 | -0.0101 | No |
| `RL-R1` | `discrete` | 3.452 | -0.0016 | +0.0008 | -0.0063 | +0.0001 | -0.0005 | +0.0002 | No |
| `RL-R3` | `discrete` | 3.426 | +0.0003 | -0.0009 | -0.0016 | +0.0005 | +0.0022 | +0.0002 | No |
| `I2-Hybrid` | `hybrid` | 3.369 | -0.0026 | +2.7323 | -0.0005 | -0.0064 | -0.0029 | -0.0007 | No |

---

## 4. Diagnostic 4 — Distributional Calibration (Regime 1 vs Regime 3 Aleatoric Negative Control)

| Arm | Reg1 ECE | Reg1 Brier (`[-2,0]`) | Reg3 ECE (Aleatoric) | Reg3 Brier (`[-2,0]`) | Reg3 Mean Conf | Reg3 Acc | Reg3 Overconf Gap |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1` | 0.0235 | -0.9629 | 0.0046 | -0.9936 | 0.0162 | 0.0208 | -0.0046 |
| `B2` | 0.0131 | -0.9595 | 0.0397 | -1.0056 | 0.0397 | 0.0000 | +0.0397 |
| `B3` | 0.0517 | -0.9640 | 0.0323 | -1.0027 | 0.0323 | 0.0000 | +0.0323 |
| `B4` | 0.0197 | -0.9586 | 0.0447 | -0.9889 | 0.0178 | 0.0625 | -0.0447 |
| `RL-R1` | 0.0262 | -0.9644 | 0.0255 | -0.9971 | 0.0255 | 0.0000 | +0.0255 |
| `RL-R3` | 0.0380 | -0.9632 | 0.0400 | -1.0046 | 0.0400 | 0.0000 | +0.0400 |
| `I2-Hybrid` | 0.0354 | -0.9581 | 0.0325 | -0.9982 | 0.0325 | 0.0000 | +0.0325 |

---

## 5. Diagnostic 5 — Training Stability Telemetry

| Arm | Steps | Final Train Loss / `loss_pg` | Final Val NLL / `loss_ntp` | Mean Reward | Policy Entropy | `sigma=0` Frac | No-Thought NLL Drift |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1` | 80 | 1.4134 | 1.2866 | — | — | — | — |
| `B2` | 80 | 1.3435 | 1.2898 | — | — | — | — |
| `B3` | 80 | 1.4950 | 1.2776 | — | — | — | — |
| `B4` | 80 | 1.5092 | 1.3047 | — | — | — | — |
| `RL-R1` | 20 | -0.0000 | 1.3647 | +0.5000 | 2.727 | 1.00 | -0.0686 |
| `RL-R3` | 20 | +0.0000 | 1.3456 | -0.0124 | 2.269 | 0.00 | +0.0559 |
| `I2-Hybrid` | 20 | -0.0000 | 1.4726 | +0.1331 | 4.122 | 0.00 | -0.5746 |

### Headline Criterion Checklist (`I2-Hybrid`)

- `teacher_free`: **PASS**
- `b2_accuracy_parity`: **FAIL**
- `b2_flops_matched`: **FAIL**
- `beats_b3_by_3pp`: **PASS**
- `beats_b3_strict`: **PASS**
- `swap_sensitivity_gt_0_15`: **FAIL**
- `trunc_faithfulness_gt_0`: **PASS**
- `monotonic_ood_depth_scaling`: **FAIL**
- `ood_pass_at_k_expansion`: **PASS**
- `calibration_stable`: **PASS**
