"""The MDP, the featuriser, GAE, ragged batching and a tiny end-to-end training run."""

import tempfile
import unittest

import numpy as np

from buchberger_rl.agent import Batch, PairScorer, PPOTrainer, compute_gae, run_episode
from buchberger_rl.agent.ppo import make_env
from buchberger_rl.algebra import BuchbergerState, PolynomialRing
from buchberger_rl.config import EnvConfig, ExperimentConfig, PPOConfig, RunConfig
from buchberger_rl.env import BuchbergerEnv, PairFeaturizer, rollout_cost
from buchberger_rl.evaluation import sample_ideals, strategy_costs
from buchberger_rl.ideals import RandomBinomialIdealGenerator
from buchberger_rl.strategies import DegreeStrategy


class TestEnvironment(unittest.TestCase):
    def test_features_encode_two_leading_terms(self):
        R = PolynomialRing(3, "grevlex")
        F = [R.parse("x*y^6 + 9*z^4"), R.parse("x*y^3 + 13*z^4"), R.parse("x^2*y^2 + 91*z")]
        state = BuchbergerState.from_generators(F, elimination="none")
        X = PairFeaturizer(3, k=2)(state)
        self.assertEqual(X.shape, (3, 12))
        np.testing.assert_array_equal(X[0], [1, 6, 0, 0, 0, 4, 1, 3, 0, 0, 0, 4])

    def test_rewards_match_strategy_cost(self):
        """Following the degree strategy inside the environment reproduces its cost exactly."""
        ideals = sample_ideals("3-12-6-weighted", 5, seed=2)
        expected = strategy_costs(DegreeStrategy(), ideals)
        gen = RandomBinomialIdealGenerator.from_name("3-12-6-weighted")
        env = BuchbergerEnv(gen, 3)
        strat = DegreeStrategy()
        for F, cost in zip(ideals, expected):
            env.reset(F)
            total, done = 0.0, env.state.done
            while not done:
                _, r, done, _ = env.step(env.state.pairs.index(strat(env.state)))
                total -= r
            self.assertEqual(total, cost)
            self.assertEqual(rollout_cost(BuchbergerState.from_generators(F)), cost)

    def test_reset_skips_trivial_ideals(self):
        env = make_env(EnvConfig(distribution="3-5-2-weighted"), seed=0)
        for _ in range(20):
            env.reset()
            self.assertGreater(len(env.state.pairs), 0)


class TestPPOComponents(unittest.TestCase):
    def test_gae_reduces_to_discounted_return_without_baseline(self):
        r = np.array([-1.0, -2.0, -3.0])
        adv = compute_gae(r, np.zeros(3), gamma=0.9, lam=1.0)
        np.testing.assert_allclose(adv, [-1 - 0.9 * 2 - 0.81 * 3, -2 - 0.9 * 3, -3])

    def test_gae_is_zero_for_perfect_values(self):
        r = np.array([-1.0, -2.0, -3.0])
        v = np.array([-6.0, -5.0, -3.0])  # exact undiscounted values-to-go
        np.testing.assert_allclose(compute_gae(r, v, gamma=1.0, lam=0.95), 0.0, atol=1e-12)

    def test_batch_select_recovers_rows(self):
        env = make_env(EnvConfig(distribution="3-10-5-weighted"), seed=0)
        net = PairScorer(env.observation_dim, (16,), seed=0)
        rng = np.random.default_rng(0)
        episodes = [run_episode(env, net, rng) for _ in range(3)]
        batch = Batch(episodes, PPOConfig(normalize_advantages=False))
        obs = [o for ep in episodes for o in ep.obs]
        idx = np.array([len(obs) - 1, 0, 2])
        X, starts, act_rows, _, _ = batch.select(idx)
        np.testing.assert_array_equal(X[starts[1]: starts[2]], obs[0])
        self.assertEqual(act_rows[0] - starts[0], batch.actions[idx[0]])

    def test_training_smoke(self):
        cfg = ExperimentConfig(
            name="smoke",
            env=EnvConfig(distribution="3-8-4-weighted"),
            ppo=PPOConfig(hidden=(16,), policy_updates=3, batch_size=64, value="none"),
            run=RunConfig(epochs=2, episodes_per_epoch=6, workers=1, smoothing=1, checkpoint_every=1),
        )
        with tempfile.TemporaryDirectory() as d:
            best = PPOTrainer(cfg, d, log=lambda *_: None).train()
            net = PairScorer.load(best)
            self.assertEqual(net.input_dim, 12)


if __name__ == "__main__":
    unittest.main()
