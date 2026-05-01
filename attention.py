"""Baseline attention utilities for entropy-attention experiments."""

import mlx.core as mx
import mlx.nn as nn


def make_causal_mask(T):
    rows = mx.arange(T).reshape(T, 1)
    cols = mx.arange(T).reshape(1, T)
    return mx.where(rows >= cols, mx.array(0.0), mx.array(-1e9))


class _BaseAttention(nn.Module):
    """Shared QKV projection logic."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def _project(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        return Q, K, V

    def _output(self, O, B, T, D):
        return self.o_proj(O.transpose(0, 2, 1, 3).reshape(B, T, D))


class StandardAttention(_BaseAttention):
    """Vanilla causal dot-product attention baseline."""

    def __call__(self, x):
        B, T, D = x.shape
        Q, K, V = self._project(x)
        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(T)
        A = mx.softmax(S, axis=-1)
        return self._output(A @ V, B, T, D)

    def get_diagnostics(self, x):
        Q, K, V = self._project(x)
        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale + make_causal_mask(x.shape[1])
        A = mx.softmax(S, axis=-1)
        return {"A": A}
