# Entropy Attention: Explore-Then-Exploit Attention

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![MLX](https://img.shields.io/badge/MLX-0.31+-orange.svg)](https://ml-explore.github.io/mlx/)

**A two-phase attention mechanism that explores broadly, measures the information landscape, then contracts based on what it found.**

Standard attention commits to an attention pattern in one step. Entropy attention splits that commitment into two phases: an exploration phase that surveys the context with a learned per-head temperature, and an exploitation phase that uses the entropy of explored values to modulate contraction sharpness.

This line of investigation came from my own theory of a negentropic mechanism behind reasoning and informational processing. It also came directly out of my Rössler attention experiments, which produced a clear negative result but surfaced a useful observation: standard attention commits to an attention pattern in one step without knowing what it will find. What if it could explore first?

**Result:** The mechanism is mechanically alive—exploration and exploitation produce genuinely different attention distributions, and heads learn different temperatures—but it did not improve over standard attention across an 8-task synthetic battery. See `docs/results/experiment-log.md` for the full experimental story, alpha sign patterns, and conclusions.

## Quick Start

```bash
git clone https://github.com/davidkny22/entropy-attention.git
cd entropy-attention
pip install -r requirements.txt

# Smoke test: 2 tasks, ~5 minutes
python smoke_test.py

# Quick battery: all 8 tasks, ~25 minutes
python quick_battery.py
```

## Module Reference

Every module can be used standalone. Below are practical examples for each domain.

### Entropy Attention (`entropy_attention.py`)

```python
from entropy_attention import EntropyAttention
from attention import StandardAttention
from engine import Model, run_experiment

# Train with entropy attention
model = Model(vocab_size=16, dim=64, num_heads=4,
              max_seq_len=255, attention_cls=EntropyAttention)

# Or use standard attention as a baseline
model_std = Model(vocab_size=16, dim=64, num_heads=4,
                  max_seq_len=255, attention_cls=StandardAttention)
```

### Standard Attention Baseline (`attention.py`)

```python
from attention import StandardAttention, _BaseAttention

# Standard causal dot-product attention
attn = StandardAttention(dim=64, num_heads=4)
output = attn(x)  # x: (B, T, D)

# Diagnostics: attention matrix
diag = attn.get_diagnostics(x)
print(diag["A"].shape)  # (B, H, T, T)
```

### Training & Evaluation (`engine.py`)

```python
from engine import run_experiment, DEFAULT_CONFIG

# Run a single experiment with full diagnostics
result = run_experiment(
    name="associative_recall",
    attention_cls=EntropyAttention,
    variant_name="entropy",
    train_x=train_x, train_y=train_y,
    test_x=test_x, test_y=test_y,
    target_masks=masks,
    config={"epochs": 50},
)

# Result keys: losses, acc, diag, train_time, param_history, model
print(f"Overall accuracy: {result['acc']['overall']:.4f}")
print(f"Entropy per head: {result['diag']['entropy_per_head']}")
```

### Task Generators (`tasks.py`)

```python
from tasks import ALL_TASKS

# Generate data for any of the 8 tasks
task_fn = ALL_TASKS["associative_recall"]
train_seqs, train_masks, info = task_fn(num_seqs=2048, seed=42)

# info contains vocab_size, critical_mask, description
critical = info["critical_mask"]  # which positions test the core capability
```

## How It Works

Standard attention commits to a distribution in one step:

```
A = softmax(QK^T / sqrt(d))
O = A @ V
```

Entropy attention splits that commitment into two phases:

### Phase 1: Exploration

```
S = Q @ K^T / sqrt(d) + causal_mask
tau = softplus(tau_raw) + 0.5           # per-head learned temperature
A_explore = softmax(S / tau)            # broadened distribution
V_explored = A_explore @ V              # what broad attention sees
```

### Phase 2: Exploitation

```
V_abs = |V_explored| + eps
V_norm = V_abs / sum(V_abs, dim=-1)     # normalize to distribution over dims
H = -sum(V_norm * log(V_norm), dim=-1)  # per-position entropy
negentropy = log(d) - H                 # high = clear signal found

beta_base = softplus(beta_raw) + 1.0    # per-head base sharpness
beta_effective = beta_base * (1 + alpha * negentropy)  # per-position sharpness
A_final = softmax(S * beta_effective)   # entropy-informed contraction
O = A_final @ V
```

Three learnable scalars per head: `tau` (exploration temperature), `alpha` (entropy coupling), `beta` (contraction sharpness). 12 extra parameters total for 4 heads.

## Project Structure

```
entropy-attention/
  entropy_attention.py      # Two-phase attention module
  attention.py              # Standard attention baseline + shared QKV logic
  engine.py                 # Model, training loop, evaluation, diagnostics
  tasks.py                  # 8 synthetic task generators
  smoke_test.py             # 2-task validation run (~5 min)
  quick_battery.py          # 50-epoch eight-task battery (~25 min)
  docs/
    results/
      experiment-log.md     # Full experiment log: smoke test, quick battery, conclusions
```

## Installation

```bash
git clone https://github.com/davidkny22/entropy-attention.git
cd entropy-attention
pip install -r requirements.txt
```

For development:
```bash
pip install -e .
```

## Related Work

Attention entropy has been studied as a training diagnostic—[Zhai et al. (2023)](https://arxiv.org/abs/2303.09417) showed that entropy collapse correlates with training instability, and [Varre, Rofin & Flammarion (2026)](https://arxiv.org/abs/2603.06248) proved that gradient flow drives softmax outputs toward low-entropy solutions. Learned temperature in attention has been explored by [Vasylenko et al. (2026)](https://arxiv.org/abs/2506.00590) (ASEntmax) and [Zhang et al. (2024)](https://arxiv.org/abs/2411.12892) (Selective Self-Attention), while [Lee, Lee & Song (2022)](https://arxiv.org/abs/2112.13492) showed that Vision Transformers consistently learn lower-than-standard temperatures. Information-theoretic regularizers such as negative entropy have been used to induce sparse or diverse attention ([Sun et al., 2021](https://arxiv.org/abs/2112.07688); [Martins, Niculae & McNamee, 2023](https://arxiv.org/abs/2304.12810)). [Agarwal, Dalal & Misra (2025)](https://arxiv.org/abs/2512.22471) showed that transformer attention implements Bayesian inference via a value manifold parameterized by posterior entropy. The explore/exploit framing has appeared in attention budgeting ([Faye et al., 2026](https://arxiv.org/abs/2604.22583)) and in-context bandit learning ([Dai, Tomasi & Ghiassian, 2024](https://arxiv.org/abs/2403.06826)), but to my knowledge, not as a two-phase mechanism driven by value entropy.

## License

[AGPL-3.0-only](LICENSE)

## Citation

```bibtex
@misc{kogan2026entropy,
  author = {Kogan, David},
  title = {Entropy Attention: Explore-Then-Exploit Attention},
  year = {2026},
  url = {https://github.com/davidkny22/entropy-attention}
}
```
