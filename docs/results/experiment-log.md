# Entropy Attention: Experiment Log

## Origin and Question

Entropy attention came from a personally theorized negentropic mechanism behind reasoning and informational processing. The question was whether attention could explore broadly first, measure the information landscape it found, then contract based on that measurement.

It also came directly out of my Rössler attention experiments. Those experiments produced a clear negative result, but they surfaced a useful observation: standard attention commits to an attention pattern in a single step without knowing what it will find. What if attention could explore first, measure the information landscape, then contract—with the contraction informed by what the exploration discovered?

The idea was to split attention into two phases: an exploration phase that surveys the context with a learned temperature, and an exploitation phase that uses the entropy of what was found to modulate how sharply each position contracts. Positions that found clear signal should contract harder; positions that found noise should stay broader.

---

## Experiment 1: Smoke Test

**Goal:** Quick validation before committing to a full battery. Does the mechanism train? Do the parameters differentiate across heads? Is there any accuracy signal on the hardest task?

**Setup:** Entropy attention vs standard on two tasks—associative recall (hardest, 23.4% baseline) and dual-stream (easy sanity check, 100% baseline). 50 epochs, dim 64, 4 heads, 1 layer, sequence length 256.

**Results:**

| Task | Standard Critical | Entropy Critical | Delta |
|------|------------------:|-----------------:|------:|
| Associative recall | 26.6% | 26.2% | -0.4% |
| Dual-stream | 100% | 100% | 0.0% |

**Diagnostics:**

- **Exploration-exploitation divergence:** KL = 0.40 (associative recall), 0.53 (dual-stream). The two phases produced genuinely different attention distributions.
- **Beta effective range:** [1.15, 1.73] on associative recall. The entropy modulation was creating meaningful position-dependent sharpness variation.
- **Tau differentiation:** Heads found different exploration temperatures (1.24 to 1.60).

- **Alpha (entropy coupling) went negative** on 2 of 4 heads for associative recall (-0.40, -0.34). These heads learned to invert the entropy relationship—positions that found clear signal got broader attention, positions that found noise got sharper. This is the same pattern as the Rössler reinjection parameter `b` going negative in my Rössler attention experiments: the network may be disagreeing with the assumed direction of the mechanism.

- However, 2 other heads kept alpha positive (+0.06, +0.18). The network was split—some heads used entropy as designed, others inverted it.

### Decision Log

**Decision:** Run a quick battery across all 8 tasks at 50 epochs (~30 minutes) to determine if the alpha inversion is task-specific, universal, or mixed.

**Rationale:** Two heads inverted alpha on associative recall, but the other two kept it positive. This split could mean (a) the inversion is task-specific, (b) some heads naturally adopt the opposite relationship, or (c) the signal is too weak to learn consistently. Testing across all 8 tasks maps which tasks produce which pattern, which is necessary before iterating on formulation.

This avoids the Rössler attention mistake of iterating on a mechanism based on a single task before understanding whether the behavior generalizes.

---

## Experiment 2: Quick Battery

**Goal:** Determine if entropy attention helps on ANY task, and map the alpha sign pattern across the full task battery.

**Setup:** Standard vs entropy attention on all 8 tasks, 50 epochs each, dim 64, 4 heads, 1 layer. Total runtime: ~25 minutes.

**Results—critical accuracy:**

| Task | Standard | Entropy | Delta |
|------|---------:|--------:|------:|
| Selective copy | 44.8% | 37.8% | **-7.1%** |
| Associative recall | 26.6% | 25.6% | -1.0% |
| Mode interference | 99.5% | 98.8% | -0.8% |
| Pattern confounders | 99.9% | 99.7% | -0.2% |
| Dual-stream | 100% | 100% | 0.0% |
| Sparse needle | 100% | 100% | 0.0% |
| Compositional lookup | 100% | 100% | 0.0% |
| Nested periodicity | 88.5% | 88.5% | 0.0% |

**Zero wins across all 8 tasks** in this battery. Entropy attention matched standard on easy tasks and lost on every task where there was room to differentiate. The largest loss was selective copy (-7.1%).

**Alpha sign pattern:**

| Task | Head 0 | Head 1 | Head 2 | Head 3 | Pattern |
|------|-------:|-------:|-------:|-------:|---------|
| Associative recall | +0.06 | +0.18 | -0.40 | -0.34 | Mixed |
| Selective copy | +0.28 | +0.45 | -0.35 | -0.35 | Mixed |
| Dual-stream | +0.19 | +0.23 | +0.12 | +0.15 | All positive |
| Nested periodicity | -0.09 | +0.22 | -0.29 | -0.02 | Mixed |
| Sparse needle | +0.16 | +0.18 | +0.21 | +0.10 | All positive |
| Pattern confounders | -0.19 | -0.15 | +0.13 | -0.03 | Mixed |
| Mode interference | +0.40 | +0.44 | +0.50 | +0.24 | All positive |
| Compositional lookup | +0.11 | +0.08 | +0.15 | +0.18 | All positive |

4 tasks all-positive, 4 tasks mixed, 0 tasks all-negative. Alpha was consistently positive on easy tasks (where both models score 99–100%) and mixed on hard/medium tasks (where performance actually mattered). On the easy tasks, the entropy signal may be harmless noise—the model solves them regardless. On the hard tasks, the entropy signal is actively conflicted, with some heads inverting it.

**Mechanism health:** KL divergence between exploration and exploitation phases ranged from 0.33 to 0.75 across tasks—the two phases were genuinely distinct everywhere. Beta effective varied meaningfully. The mechanism appears to work mechanically in this setup; it simply did not correlate with improved accuracy on these tasks.

### Decision Log

**Decision:** Do not run a full 200-epoch battery or scale test at this time. Stop and document.

**Rationale:** Zero wins across 8 tasks with clear signal (alpha inversion on hard tasks, consistent pattern) suggests that the current formulation—using value entropy to drive contraction—does not provide useful task signal in this setup. Running longer epochs or larger scale is unlikely to change this pattern if the learned parameter itself inverts on the tasks where performance matters.

This is a clear stopping point: the mechanism trains and differentiates, but the specific signal (value entropy) may not encode what we assumed it encodes. Future experiments should test alternative exploitation signals before committing more compute.

---

## Observations Across Both Experiments

### Confirmed

- The two-phase structure does not collapse during training. KL between `A_explore` and `A_final` remains nonzero across tasks and epochs.
- Heads learn different exploration temperatures (`tau`) and contraction behaviors (`beta`).
- The mechanism differentiates across heads in this setup.
- On this 8-task battery, entropy attention matched or underperformed standard attention.

### Observed

- Alpha was consistently positive on tasks where both models score near 100%, and mixed or negative on tasks where there was room to improve. This pattern appeared in both experiments.
- On hard tasks, some heads inverted the entropy-sharpness relationship. This may indicate that the assumed direction (more signal → sharper attention) is not what the optimizer finds useful on these tasks.
- The largest accuracy losses appeared on selective copying tasks. This may suggest that broadening attention during exploration actively harms tasks that require precise token retrieval, though this hypothesis was not isolated in these experiments.

### Unexplained

- **Why does alpha invert on hard tasks but stay positive on easy ones?** One possibility: on easy tasks, the entropy signal is too weak to matter, so alpha drifts positive by initialization bias. On hard tasks, the signal matters enough that the optimizer actively learns to use or invert it. Another possibility: the inversion is a local minimum that the optimizer falls into when loss gradients are stronger. This was not tested.
- **Does value entropy actually measure what we think it measures?** Value entropy after broadened attention reflects the distribution of energy across embedding dimensions of the attended value vector. Whether this correlates with "task relevance" was assumed, not verified. A direct test (e.g., compare value entropy against ground-truth importance labels) was not run.
- **Would the pattern hold with constrained alpha?** All experiments used unconstrained alpha. A comparison of fixed-positive alpha, softplus-constrained alpha, and unconstrained alpha was not tested and may change the observed behavior.

---

## Where This Stands

**What has been tested:**

- The two-phase explore/exploit structure trains stably on an 8-task synthetic battery (dim 64, 4 heads, 1 layer, 50 epochs).
- The learned parameters differentiate across heads.
- The exploration and exploitation phases produce different attention distributions.
- On this battery, entropy attention did not outperform standard attention on any task.

**What has not been tested:**

- Alternative exploitation signals (query-value alignment, prediction confidence, gradient-derived proxies).
- Larger scale (more layers, heads, dimensions, epochs).
- Constrained vs. unconstrained alpha.
- Whether value entropy correlates with ground-truth token importance.
- Recurrent or across-layer application of the two-phase structure.

**What the next experiments should address:**

1. **Test alternative exploitation signals.** Value entropy was one hypothesis for what the exploitation phase should measure. Other signals may encode task relevance more directly.
2. **Constrain alpha.** Unconstrained alpha can invert. Testing whether constraining it to positive values changes the accuracy pattern would isolate whether the inversion is causing the losses or merely reflecting them.
3. **Larger scale if an alternative signal shows promise.** Scaling up is only justified if a smaller-scale test shows signal. The current battery provides no basis for scaling this specific formulation.

---

## Open Questions

1. **Would entropy-driven attention help at larger scale?** The 1-layer, dim-64 model has limited capacity. At larger scale with more heads and layers, the entropy signal might provide useful inductive bias—but this battery provides no evidence that it would.

2. **Is the two-phase explore/exploit structure useful with a different exploitation signal?** The structure itself (broaden first, then sharpen based on what you found) is not invalidated by these results. Value entropy was one possible signal. Whether other signals would work better remains open.

3. **Does alpha inversion cause the accuracy losses, or correlate with them?** If alpha is constrained to positive values, does accuracy improve? This would separate mechanism failure from signal failure.

4. **What would a direct test of value entropy vs. task relevance show?** Comparing explored value entropy against ground-truth important-token labels would test the core assumption that entropy encodes relevance.

---

## Experimental Timeline

| Experiment | Duration | Key Finding |
|------------|----------|-------------|
| Smoke test | ~5 min | Mechanism alive but no accuracy signal; alpha inversion on 2/4 heads |
| Quick battery | ~25 min | Zero wins; alpha positive on easy tasks, mixed on hard tasks |
| **Total** | **~30 min** | |

All experiments run on Apple M1, MLX 0.31.1. Entropy experiments: April 12-14, 2026.
