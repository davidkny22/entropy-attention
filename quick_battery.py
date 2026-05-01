"""Entropy Attention — Quick Battery (50 epochs, all 8 tasks)

Focus: does entropy attention help on ANY task? What's the alpha sign pattern?
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from attention import StandardAttention
from entropy_attention import EntropyAttention
from engine import run_experiment, DEFAULT_CONFIG
from tasks import ALL_TASKS


def main():
    print("=" * 70)
    print("  Entropy Attention — Quick Battery (50 epochs)")
    print("=" * 70)

    cfg = {**DEFAULT_CONFIG, "epochs": 50}
    results = []

    for task_name, task_fn in ALL_TASKS.items():
        train_seqs, train_masks, info = task_fn(2048, seed=42)
        test_seqs, test_masks, _ = task_fn(512, seed=123)

        train_x = mx.array(train_seqs[:, :-1])
        train_y = mx.array(train_seqs[:, 1:])
        test_x = mx.array(test_seqs[:, :-1])
        test_y = mx.array(test_seqs[:, 1:])
        target_masks = {k: v[:, 1:] for k, v in test_masks.items()}
        task_cfg = {**cfg, "vocab_size": info["vocab_size"]}
        crit = info["critical_mask"]

        print(f"\n{'_' * 70}")
        print(f"  {task_name} (critical: {crit})")
        print(f"{'_' * 70}")

        r_std = run_experiment(
            f"std/{task_name}", StandardAttention, "standard",
            train_x, train_y, test_x, test_y, target_masks, config=task_cfg,
        )
        r_ent = run_experiment(
            f"ent/{task_name}", EntropyAttention, "entropy",
            train_x, train_y, test_x, test_y, target_masks, config=task_cfg,
        )

        attn = r_ent["model"].block.attn
        alphas = [attn.alpha[h].item() for h in range(attn.num_heads)]
        taus = [(nn.softplus(attn.tau_raw[h]) + 0.5).item() for h in range(attn.num_heads)]
        betas = [(nn.softplus(attn.beta_raw[h]) + 1.0).item() for h in range(attn.num_heads)]

        diag = r_ent["model"].get_attn_diagnostics(test_x[:4])
        A_exp = diag["A_explore"]
        A_fin = diag["A"]
        mx.eval(A_exp, A_fin)
        eps = 1e-8
        kl = mx.mean(mx.sum((A_fin + eps) * mx.log((A_fin + eps) / (A_exp + eps)), axis=-1))
        mx.eval(kl)

        beta_eff = diag["beta_effective"]
        mx.eval(beta_eff)

        std_crit = r_std["acc"].get(crit, 0)
        ent_crit = r_ent["acc"].get(crit, 0)

        results.append({
            "task": task_name,
            "crit": crit,
            "std_crit": std_crit,
            "ent_crit": ent_crit,
            "delta": ent_crit - std_crit,
            "alphas": alphas,
            "taus": taus,
            "betas": betas,
            "kl_div": kl.item(),
            "beta_eff_std": mx.std(beta_eff).item(),
        })

    # Results
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Critical Accuracy")
    print(f"{'=' * 70}")
    print(f"\n  {'Task':<25} {'Standard':>10} {'Entropy':>10} {'Delta':>10}")
    print(f"  {'-' * 55}")

    wins = 0
    for r in sorted(results, key=lambda x: x["delta"]):
        marker = " ***" if r["delta"] > 0.01 else ""
        print(f"  {r['task']:<25} {r['std_crit']:>10.4f} {r['ent_crit']:>10.4f} {r['delta']:>+10.4f}{marker}")
        if r["delta"] > 0.01:
            wins += 1

    print(f"\n  Entropy wins (>1%): {wins}/{len(results)} tasks")

    # Alpha pattern
    print(f"\n{'=' * 70}")
    print(f"  ALPHA SIGN PATTERN")
    print(f"{'=' * 70}")
    print(f"\n  {'Task':<25} {'H0':>7} {'H1':>7} {'H2':>7} {'H3':>7} {'Mean':>7} {'Sign':>6}")
    print(f"  {'-' * 65}")

    all_positive = 0
    all_negative = 0
    mixed = 0

    for r in results:
        a = r["alphas"]
        mean_a = np.mean(a)
        signs = sum(1 for x in a if x > 0)
        if signs == 4:
            sign_str = "all +"
            all_positive += 1
        elif signs == 0:
            sign_str = "all -"
            all_negative += 1
        else:
            sign_str = "mixed"
            mixed += 1
        print(f"  {r['task']:<25} {a[0]:>+7.3f} {a[1]:>+7.3f} {a[2]:>+7.3f} {a[3]:>+7.3f} {mean_a:>+7.3f} {sign_str:>6}")

    print(f"\n  All positive: {all_positive}  All negative: {all_negative}  Mixed: {mixed}")

    # Mechanism health
    print(f"\n{'=' * 70}")
    print(f"  MECHANISM HEALTH")
    print(f"{'=' * 70}")
    print(f"\n  {'Task':<25} {'KL div':>8} {'Beta std':>10} {'Tau range':>12}")
    print(f"  {'-' * 55}")

    for r in results:
        tau_range = f"{min(r['taus']):.2f}-{max(r['taus']):.2f}"
        print(f"  {r['task']:<25} {r['kl_div']:>8.4f} {r['beta_eff_std']:>10.4f} {tau_range:>12}")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    if wins >= 3:
        print(f"  STRONG SIGNAL — Entropy attention wins on {wins} tasks. Run full 200-epoch battery.")
    elif wins >= 1:
        print(f"  WEAK SIGNAL — Entropy attention wins on {wins} task(s). Investigate which and why.")
    else:
        if all_negative >= 5:
            print(f"  NO SIGNAL, CONSISTENT INVERSION — Alpha negative on {all_negative} tasks.")
            print(f"  Recommendation: flip the entropy-sharpness relationship and re-test.")
        elif mixed >= 5:
            print(f"  NO SIGNAL, MIXED ALPHA — Mechanism is task-adaptive but not helping.")
            print(f"  Recommendation: investigate positive-alpha vs negative-alpha tasks.")
        else:
            print(f"  NO SIGNAL — Entropy modulation not providing useful information.")

    print()
    return results


if __name__ == "__main__":
    main()
