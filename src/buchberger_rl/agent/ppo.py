r"""Proximal Policy Optimization for S-pair selection.

The update follows Schulman et al. (2017) with the clipped surrogate

.. math::

    L^{CLIP}(\theta) = -\,\mathbb{E}_t\Big[\min\big(\rho_t(\theta) A_t,\;
        \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\Big],
    \qquad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)},

advantages from Generalized Advantage Estimation (Schulman et al., 2016), and
early stopping of the policy updates once the approximate KL divergence to the
data-collecting policy exceeds ``1.5 * target_kl`` (as in OpenAI Spinning Up).

Gradient of the surrogate
-------------------------
Writing ``g_t = dL/d log pi(a_t|s_t)``, we have ``g_t = -rho_t A_t / B`` when the
unclipped term attains the minimum and ``g_t = 0`` otherwise. For a softmax
over a state's pairs, ``d log pi(a_t|s_t) / d s_j = 1[j = a_t] - pi_j``, so

.. math::

    \frac{\partial L}{\partial s_j} = g_t\,\big(1[j = a_t] - \pi_j\big),

which is then back-propagated through the shared MLP. An optional entropy
bonus ``-c H`` contributes ``(c/B)\,\pi_j(\log\pi_j + H)``.
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..algebra.polynomial import Polynomial
from ..config import EnvConfig, ExperimentConfig, PPOConfig
from ..env import BuchbergerEnv, rollout_cost
from ..ideals import RandomBinomialIdealGenerator
from .network import Adam, PairScorer, segment_log_softmax

# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    obs: list[np.ndarray]
    actions: np.ndarray
    logps: np.ndarray
    rewards: np.ndarray
    values: np.ndarray

    @property
    def cost(self) -> float:
        """Total polynomial additions (minus the undiscounted return)."""
        return float(-self.rewards.sum())

    def __len__(self) -> int:
        return len(self.actions)


def make_env(cfg: EnvConfig, seed: int | None = None) -> BuchbergerEnv:
    gen = RandomBinomialIdealGenerator.from_name(cfg.distribution, order=cfg.order, seed=seed)
    return BuchbergerEnv(gen, gen.nvars, cfg.elimination, cfg.k, cfg.feature_scale)


def run_episode(
    env: BuchbergerEnv,
    policy: PairScorer,
    rng: np.random.Generator,
    greedy: bool = False,
    value: str = "none",
    ideal: Sequence[Polynomial] | None = None,
) -> Episode:
    obs = env.reset(ideal)
    observations, actions, logps, rewards, values = [], [], [], [], []
    done = env.state.done
    while not done:
        values.append(-rollout_cost(env.state) if value == "degree" else 0.0)
        a, logp = policy.act(obs, rng, greedy=greedy)
        observations.append(obs)
        actions.append(a)
        logps.append(logp)
        obs, r, done, _ = env.step(a)
        rewards.append(r)
    return Episode(
        observations,
        np.asarray(actions, dtype=np.int64),
        np.asarray(logps, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """Generalized Advantage Estimation; the episode ends in a terminal state (``V = 0``)."""
    adv = np.zeros_like(rewards)
    next_value, running = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        next_value = values[t]
    return adv


def _collect_worker(args) -> list[Episode]:
    params, hidden, env_cfg, n_episodes, seed, value = args
    env = make_env(env_cfg, seed)
    policy = PairScorer(env.observation_dim, hidden)
    policy.set_params(params)
    rng = np.random.default_rng(seed + 1)
    return [run_episode(env, policy, rng, value=value) for _ in range(n_episodes)]


# ---------------------------------------------------------------------------
# Ragged batches
# ---------------------------------------------------------------------------


class Batch:
    """All states of an epoch, stored as one ragged matrix of pair features."""

    def __init__(self, episodes: Sequence[Episode], cfg: PPOConfig):
        obs = [o for ep in episodes for o in ep.obs]
        self.lengths = np.array([len(o) for o in obs], dtype=np.int64)
        self.starts = np.concatenate(([0], np.cumsum(self.lengths)[:-1]))
        self.X = np.concatenate(obs).astype(np.float64)
        self.actions = np.concatenate([ep.actions for ep in episodes])
        self.logp_old = np.concatenate([ep.logps for ep in episodes])
        adv = np.concatenate([compute_gae(ep.rewards, ep.values, cfg.gamma, cfg.lam) for ep in episodes])
        if cfg.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.adv = adv

    def __len__(self) -> int:
        return len(self.actions)

    def select(self, idx: np.ndarray):
        """Gather the rows of the chosen states into a contiguous ragged minibatch."""
        lengths = self.lengths[idx]
        new_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        rows = np.repeat(self.starts[idx] - new_starts, lengths) + np.arange(lengths.sum())
        return self.X[rows], new_starts, new_starts + self.actions[idx], self.logp_old[idx], self.adv[idx]


def ppo_loss_and_grads(
    policy: PairScorer,
    X: np.ndarray,
    starts: np.ndarray,
    act_rows: np.ndarray,
    logp_old: np.ndarray,
    adv: np.ndarray,
    clip_eps: float,
    entropy_coef: float = 0.0,
):
    """Clipped PPO loss, its exact gradient, and diagnostics for one minibatch."""
    B = len(starts)
    scores, cache = policy.forward(X, keep_cache=True)
    logp_all = segment_log_softmax(scores, starts)
    pi = np.exp(logp_all)
    seg = np.repeat(np.arange(B), np.diff(np.append(starts, len(scores))))

    logp_new = logp_all[act_rows]
    ratio = np.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    loss = -np.mean(np.minimum(unclipped, clipped))
    g = np.where(unclipped <= clipped, -unclipped / B, 0.0)  # dL / d log pi(a_t)

    dscores = -pi * g[seg]
    dscores[act_rows] += g

    entropy = -np.add.reduceat(pi * logp_all, starts)
    if entropy_coef:
        loss -= entropy_coef * entropy.mean()
        dscores += (entropy_coef / B) * pi * (logp_all + entropy[seg])

    grads = policy.backward(cache, dscores)
    stats = {
        "kl": float(np.mean(logp_old - logp_new)),
        "clip_frac": float(np.mean(np.abs(ratio - 1) > clip_eps)),
        "entropy": float(entropy.mean()),
    }
    return float(loss), grads, stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class PPOTrainer:
    def __init__(self, cfg: ExperimentConfig, out_dir: str | Path, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.out = Path(out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.log = log
        env = make_env(cfg.env, seed=0)
        self.policy = PairScorer(env.observation_dim, cfg.ppo.hidden, seed=cfg.run.seed)
        self.optimizer = Adam(self.policy.params, lr=cfg.ppo.lr)
        self.rng = np.random.default_rng(cfg.run.seed)
        cfg.to_yaml(self.out / "config.yaml")

    def collect(self, epoch: int, pool: ProcessPoolExecutor | None) -> list[Episode]:
        run, ppo = self.cfg.run, self.cfg.ppo
        w = max(1, run.workers)
        counts = [run.episodes_per_epoch // w + (i < run.episodes_per_epoch % w) for i in range(w)]
        jobs = [
            (self.policy.get_params(), ppo.hidden, self.cfg.env, n,
             (run.seed * 1_000_003 + epoch * 1_009 + i) % 2**32, ppo.value)
            for i, n in enumerate(counts) if n
        ]
        if pool is None:
            chunks = map(_collect_worker, jobs)
        else:
            chunks = pool.map(_collect_worker, jobs)
        return [ep for chunk in chunks for ep in chunk]

    def update(self, batch: Batch) -> dict:
        ppo = self.cfg.ppo
        steps, stats = 0, {}
        for _ in range(ppo.policy_updates):
            size = min(ppo.batch_size, len(batch))
            idx = self.rng.choice(len(batch), size=size, replace=False)
            loss, grads, s = ppo_loss_and_grads(self.policy, *batch.select(idx), ppo.clip_eps, ppo.entropy_coef)
            stats = {**s, "loss": loss}
            if s["kl"] > 1.5 * ppo.target_kl:
                break  # early stopping; this minibatch's gradient is discarded
            self.optimizer.step(self.policy.params, grads)
            steps += 1
        stats["policy_steps"] = steps
        return stats

    def train(self) -> Path:
        run = self.cfg.run
        history: list[dict] = []
        best, best_path = np.inf, self.out / "checkpoints" / "best.npz"
        log_path = self.out / "train_log.csv"
        pool = ProcessPoolExecutor(run.workers) if run.workers > 1 else None
        try:
            with log_path.open("w", newline="") as fh:
                writer = None
                for epoch in range(1, run.epochs + 1):
                    t0 = time.time()
                    episodes = self.collect(epoch, pool)
                    costs = np.array([ep.cost for ep in episodes])
                    t_collect = time.time() - t0
                    stats = self.update(Batch(episodes, self.cfg.ppo))
                    row = {
                        "epoch": epoch,
                        "mean_additions": costs.mean(),
                        "std_additions": costs.std(),
                        "mean_length": np.mean([len(ep) for ep in episodes]),
                        **stats,
                        "seconds_collect": t_collect,
                        "seconds_total": time.time() - t0,
                    }
                    history.append(row)
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    fh.flush()

                    smoothed = np.mean([h["mean_additions"] for h in history[-run.smoothing:]])
                    if epoch >= run.smoothing and smoothed < best:
                        best = smoothed
                        self.policy.save(best_path)
                    if epoch % run.checkpoint_every == 0 or epoch == run.epochs:
                        self.policy.save(self.out / "checkpoints" / f"epoch_{epoch:04d}.npz")
                    self.log(
                        f"epoch {epoch:4d} | additions {row['mean_additions']:8.2f} "
                        f"(smoothed {smoothed:8.2f}) | kl {stats['kl']:.4f} | "
                        f"steps {stats['policy_steps']:2d} | {row['seconds_total']:.1f}s"
                    )
        finally:
            if pool is not None:
                pool.shutdown()
        if not best_path.exists():
            self.policy.save(best_path)
        (self.out / "summary.json").write_text(json.dumps(
            {"best_smoothed_training_additions": float(best), "epochs": run.epochs}, indent=2))
        return best_path
