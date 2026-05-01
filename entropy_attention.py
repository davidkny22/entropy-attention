"""Entropy Attention — Two-Phase Explore/Exploit Mechanism.

Phase 1 (Exploration): Broadened attention surveys the context.
Phase 2 (Exploitation): Entropy of explored values modulates contraction sharpness.
"""

import mlx.core as mx
import mlx.nn as nn
from attention import make_causal_mask, _BaseAttention


class EntropyAttention(_BaseAttention):
    """Two-phase attention: explore broadly, then exploit with entropy-informed sharpness.

    Per-head learnable parameters:
      tau_raw  -> exploration temperature (how broadly to look)
      alpha    -> entropy coupling (how much value entropy modulates contraction)
      beta_raw -> base contraction sharpness (how sharply to exploit)
    """

    def __init__(self, dim, num_heads):
        super().__init__(dim, num_heads)
        self.tau_raw = mx.array([0.5] * num_heads)    # -> softplus + 0.5 ~ 1.13
        self.alpha = mx.array([0.1] * num_heads)
        self.beta_raw = mx.array([0.0] * num_heads)   # -> softplus + 1.0 ~ 1.69

    def _forward(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q, K, V = self._project(x)

        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)

        # ── PHASE 1: EXPLORATION ──
        tau = (nn.softplus(self.tau_raw) + 0.5).reshape(1, H, 1, 1)
        A_explore = mx.softmax(S / tau, axis=-1)                    # (B, H, T, T)

        # What does broadened attention see?
        V_explored = A_explore @ V                                   # (B, H, T, d)

        # Entropy of explored values per position
        eps = 1e-8
        V_abs = mx.abs(V_explored) + eps
        V_norm = V_abs / mx.sum(V_abs, axis=-1, keepdims=True)      # (B, H, T, d)
        H_val = -mx.sum(V_norm * mx.log(V_norm), axis=-1)           # (B, H, T)
        H_max = mx.log(mx.array(float(d)))
        negentropy = H_max - H_val                                   # (B, H, T) high = clear signal

        # ── PHASE 2: EXPLOITATION ──
        alpha = self.alpha.reshape(1, H, 1)
        beta_base = (nn.softplus(self.beta_raw) + 1.0).reshape(1, H, 1)

        # Per-position sharpness: positions that found clear signal contract harder
        beta_effective = beta_base * (1.0 + alpha * negentropy)      # (B, H, T)
        beta_effective = beta_effective.reshape(B, H, T, 1)          # broadcast over key dim

        A_final = mx.softmax(S * beta_effective, axis=-1)            # (B, H, T, T)
        O = A_final @ V                                              # (B, H, T, d)

        return O, A_explore, A_final, H_val, negentropy, beta_effective.squeeze(-1)

    def __call__(self, x):
        B, T, D = x.shape
        O, *_ = self._forward(x)
        return self._output(O, B, T, D)

    def get_diagnostics(self, x):
        O, A_explore, A_final, H_val, negentropy, beta_eff = self._forward(x)
        return {
            "A": A_final,
            "A_explore": A_explore,
            "H_val": H_val,
            "negentropy": negentropy,
            "beta_effective": beta_eff,
        }
