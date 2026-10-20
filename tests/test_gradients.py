"""Finite-difference checks of every analytic gradient used in training."""

import unittest

import numpy as np

from buchberger_rl.agent.network import PairScorer, segment_log_softmax
from buchberger_rl.agent.ppo import ppo_loss_and_grads


def numerical_grad(f, params, eps=1e-6):
    grads = []
    for p in params:
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            fp = f()
            p[i] = old - eps
            fm = f()
            p[i] = old
            g[i] = (fp - fm) / (2 * eps)
        grads.append(g)
    return grads


def ragged_batch(rng, n_states=6, dim=12):
    lengths = rng.integers(1, 7, size=n_states)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    X = rng.normal(size=(lengths.sum(), dim))
    actions = starts + np.array([rng.integers(l) for l in lengths])
    return X, starts, actions


class TestGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.net = PairScorer(12, hidden=(8, 5), seed=1, output_init_scale=1.0)

    def assertGradsClose(self, analytic, numeric, tol=1e-5):
        for a, n in zip(analytic, numeric):
            np.testing.assert_allclose(a, n, rtol=tol, atol=tol)

    def test_mlp_backward(self):
        X = self.rng.normal(size=(9, 12))
        w = self.rng.normal(size=9)
        f = lambda: float(self.net.forward(X) @ w)
        _, cache = self.net.forward(X, keep_cache=True)
        self.assertGradsClose(self.net.backward(cache, w), numerical_grad(f, self.net.params))

    def test_segment_log_softmax_normalises(self):
        X, starts, _ = ragged_batch(self.rng)
        lp = segment_log_softmax(self.net.forward(X), starts)
        np.testing.assert_allclose(np.add.reduceat(np.exp(lp), starts), 1.0)

    def _check_ppo(self, entropy_coef, perturb):
        X, starts, act = ragged_batch(self.rng)
        logp_now = segment_log_softmax(self.net.forward(X), starts)[act]
        # old log-probs differ from the current ones so that some ratios fall
        # outside the clipping range and the zero-gradient branch is exercised
        logp_old = logp_now + self.rng.normal(scale=perturb, size=len(act))
        adv = self.rng.normal(size=len(act))
        args = (X, starts, act, logp_old, adv, 0.2, entropy_coef)
        _, grads, _ = ppo_loss_and_grads(self.net, *args)
        f = lambda: ppo_loss_and_grads(self.net, *args)[0]
        self.assertGradsClose(grads, numerical_grad(f, self.net.params))

    def test_ppo_gradient_inside_clip_range(self):
        self._check_ppo(entropy_coef=0.0, perturb=0.01)

    def test_ppo_gradient_with_clipping(self):
        self._check_ppo(entropy_coef=0.0, perturb=0.5)

    def test_ppo_gradient_with_entropy_bonus(self):
        self._check_ppo(entropy_coef=0.05, perturb=0.3)


if __name__ == "__main__":
    unittest.main()
