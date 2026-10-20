r"""A permutation-equivariant pair-scoring policy, implemented in NumPy.

The policy applies one shared MLP ``f_\theta : R^{2kn} -> R`` to every row of
the observation (one row per S-pair) and normalises the resulting preference
scores with a softmax over the *current* pair set:

.. math::

    \pi_\theta(a \mid s) = \frac{\exp f_\theta(x_a)}{\sum_{b \in P_t} \exp f_\theta(x_b)} .

Weight sharing makes the policy independent of the number and the order of
the pairs, which is what an action set of varying size requires. Scores can
also be used directly as sort keys inside a production implementation of
Buchberger's algorithm.

Why NumPy? The network is tiny (a single hidden layer suffices, see the
ablations), and writing the forward and backward passes explicitly keeps the
whole method -- including the PPO gradient -- transparent and dependency-free.
Every gradient is verified against finite differences in
``tests/test_gradients.py``.

Ragged batches
--------------
A training batch contains many states, each with a different number of pairs.
All rows are concatenated into a single matrix and the state boundaries are
given by ``starts`` (the index of the first row of each state). Per-state
softmax reductions then become ``np.maximum.reduceat`` / ``np.add.reduceat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class _Cache:
    inputs: list[np.ndarray]
    pre: list[np.ndarray]


class PairScorer:
    """MLP ``R^d -> R`` with ReLU hidden layers and a linear output."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (128,), seed: int | None = None,
                 output_init_scale: float = 0.01):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden = tuple(hidden)
        sizes = (input_dim, *hidden, 1)
        self.params: list[np.ndarray] = []
        for li, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = li == len(sizes) - 2
            std = np.sqrt(2.0 / fan_in) * (output_init_scale if last else 1.0)  # He init
            self.params.append((rng.standard_normal((fan_in, fan_out)) * std).astype(np.float64))
            self.params.append(np.zeros(fan_out, dtype=np.float64))

    # -- forward / backward -------------------------------------------------

    def forward(self, X: np.ndarray, keep_cache: bool = False):
        h = np.asarray(X, dtype=np.float64)
        cache = _Cache([], [])
        n_layers = len(self.params) // 2
        for li in range(n_layers):
            W, b = self.params[2 * li], self.params[2 * li + 1]
            if keep_cache:
                cache.inputs.append(h)
            z = h @ W + b
            if li < n_layers - 1:
                if keep_cache:
                    cache.pre.append(z)
                h = np.maximum(z, 0.0)
            else:
                h = z
        scores = h[:, 0]
        return (scores, cache) if keep_cache else scores

    def backward(self, cache: _Cache, dscores: np.ndarray) -> list[np.ndarray]:
        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        g = dscores[:, None]
        n_layers = len(self.params) // 2
        for li in reversed(range(n_layers)):
            W = self.params[2 * li]
            grads[2 * li] = cache.inputs[li].T @ g
            grads[2 * li + 1] = g.sum(axis=0)
            if li > 0:
                g = (g @ W.T) * (cache.pre[li - 1] > 0)
        return grads

    # -- policy helpers -----------------------------------------------------

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        s = self.forward(X)
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    def act(self, X: np.ndarray, rng: np.random.Generator, greedy: bool = False) -> tuple[int, float]:
        """Return ``(action, log pi(action))`` for a single state."""
        s = self.forward(X)
        if greedy:
            a = int(np.argmax(s))
        z = s - s.max()
        logp = z - np.log(np.exp(z).sum())
        if not greedy:
            a = int(rng.choice(len(s), p=np.exp(logp)))
        return a, float(logp[a])

    # -- persistence --------------------------------------------------------

    def get_params(self) -> list[np.ndarray]:
        return [p.copy() for p in self.params]

    def set_params(self, params: list[np.ndarray]) -> None:
        assert len(params) == len(self.params)
        self.params = [np.array(p, dtype=np.float64) for p in params]

    def save(self, path: str | Path) -> None:
        np.savez(path, input_dim=self.input_dim, hidden=np.array(self.hidden),
                 **{f"p{i}": p for i, p in enumerate(self.params)})

    @classmethod
    def load(cls, path: str | Path) -> "PairScorer":
        data = np.load(path)
        net = cls(int(data["input_dim"]), tuple(int(h) for h in data["hidden"]))
        net.set_params([data[f"p{i}"] for i in range(len(net.params))])
        return net


def segment_log_softmax(scores: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Log-softmax of ``scores`` computed independently within each segment."""
    seg = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(scores))))
    mx = np.maximum.reduceat(scores, starts)
    z = scores - mx[seg]
    lse = np.log(np.add.reduceat(np.exp(z), starts))
    return z - lse[seg]


class Adam:
    """Adam optimiser (Kingma & Ba, 2015) for a list of NumPy arrays, updated in place."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-4, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        c1 = 1 - self.b1**self.t
        c2 = 1 - self.b2**self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
