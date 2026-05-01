"""Entropy Attention — Smoke Test

Standard vs Entropy on associative_recall (hard) and dual_stream (easy).
50 epochs. Quick signal check before committing to the full battery.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from attention import StandardAttention
from entropy_attention import EntropyAttention
from engine import run_experiment, DEFAULT_CONFIG
from tasks import ALL_TASKS


def main():
    print("=" * 60)
    print("  Entropy Attention — Smoke Test")
    print("=" * 60)

    test_tasks = ["associative_recall", "dual_stream"]
    cfg = {**DEFAULT_CONFIG, "epochs": 50}

    for task_name in test_tasks:
        task_fn = ALL_TASKS[task_name]
        train_seqs, train_masks, info = task_fn(2048, seed=42)
        test_seqs, test_masks, _ = task_fn(512, seed=123)

        train_x = mx.array(train_seqs[:, :-1])
        train_y = mx.array(train_seqs[:, 1:])
        test_x = mx.array(test_seqs[:, :-1])
        test_y = mx.array(test_seqs[:, 1:])
        target_masks = {k: v[:, 1:] for k, v in test_masks.items()}
        task_cfg = {**cfg, "vocab_size": info["vocab_size"]}
        crit = info["critical_mask"]

        print(f"\n{'=' * 60}")
        print(f"  Task: {task_name} (critical: {crit})")
        print(f"{'=' * 60}")

        r_std = run_experiment(
            "Standard", StandardAttention, "standard",
            train_x, train_y, test_x, test_y, target_masks, config=task_cfg,
        )

        r_ent = run_experiment(
            "Entropy", EntropyAttention, "entropy",
            train_x, train_y, test_x, test_y, target_masks, config=task_cfg,
        )

        std_crit = r_std["acc"].get(crit, 0)
        ent_crit = r_ent["acc"].get(crit, 0)
        delta = ent_crit - std_crit

        print(f"\n  {'Metric':<20} {'Standard':>10} {'Entropy':>10} {'Delta':>10}")
        print(f"  {'-' * 50}")
        print(f"  {'Overall':<20} {r_std['acc']['overall']:>10.4f} {r_ent['acc']['overall']:>10.4f} {r_ent['acc']['overall'] - r_std['acc']['overall']:>+10.4f}")
        print(f"  {crit:<20} {std_crit:>10.4f} {ent_crit:>10.4f} {delta:>+10.4f}")
        print(f"  {'Final loss':<20} {r_std['losses'][-1]:>10.4f} {r_ent['losses'][-1]:>10.4f} {r_ent['losses'][-1] - r_std['losses'][-1]:>+10.4f}")

        # Entropy-specific diagnostics
        attn = r_ent["model"].block.attn
        print(f"\n  Entropy attention learned parameters:")
        print(f"  {'Head':<6} {'tau':>8} {'alpha':>8} {'beta':>8}")
        print(f"  {'-' * 30}")
        for h in range(attn.num_heads):
            tau_val = (nn.softplus(attn.tau_raw[h]) + 0.5).item()
            alpha_val = attn.alpha[h].item()
            beta_val = (nn.softplus(attn.beta_raw[h]) + 1.0).item()
            print(f"  {h:<6} {tau_val:>8.4f} {alpha_val:>8.4f} {beta_val:>8.4f}")

        # Exploration-exploitation divergence
        diag = r_ent["model"].get_attn_diagnostics(test_x[:4])
        A_exp = diag["A_explore"]
        A_fin = diag["A"]
        mx.eval(A_exp, A_fin)
        eps = 1e-8
        kl = mx.mean(mx.sum((A_fin + eps) * mx.log((A_fin + eps) / (A_exp + eps)), axis=-1))
        mx.eval(kl)
        print(f"\n  Explore-exploit KL divergence: {kl.item():.4f}")

        beta_eff = diag["beta_effective"]
        mx.eval(beta_eff)
        print(f"  Beta effective range: [{beta_eff.min().item():.3f}, {beta_eff.max().item():.3f}]")
        print(f"  Beta effective std: {mx.std(beta_eff).item():.4f}")

    print(f"\n{'=' * 60}")
    print(f"  SMOKE TEST COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
