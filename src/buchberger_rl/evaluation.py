"""Paired evaluation of selection strategies and trained agents.

All methods are run on the *same* held-out ideals, so comparisons are paired:
per-ideal cost differences remove most of the (very large) variance in ideal
difficulty. We report the mean and standard deviation of the cost, the relative
improvement over a reference strategy, a bootstrap confidence interval for the
mean paired difference, and a two-sided Wilcoxon signed-rank test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats as sps

from .agent.network import PairScorer
from .agent.ppo import make_env, run_episode
from .algebra.buchberger import BuchbergerState
from .algebra.polynomial import Polynomial
from .config import EnvConfig
from .ideals import RandomBinomialIdealGenerator
from .strategies import Strategy

Ideal = Sequence[Polynomial]


def sample_ideals(distribution: str, n: int, seed: int, order: str = "grevlex",
                  elimination: str = "gebauermoeller") -> list[list[Polynomial]]:
    """Sample ``n`` ideals with a non-empty initial pair set (same rule as training)."""
    gen = RandomBinomialIdealGenerator.from_name(distribution, order=order, seed=seed)
    ideals = []
    while len(ideals) < n:
        F = gen.sample()
        if BuchbergerState.from_generators(F, elimination).pairs:
            ideals.append(F)
    return ideals


def strategy_costs(strategy: Strategy, ideals: Sequence[Ideal], elimination: str = "gebauermoeller") -> np.ndarray:
    costs = np.empty(len(ideals))
    for i, F in enumerate(ideals):
        state = BuchbergerState.from_generators(F, elimination)
        total = 0
        while not state.done:
            total += state.step(strategy(state)).additions
        costs[i] = total
    return costs


def agent_costs(policy: PairScorer, ideals: Sequence[Ideal], env_cfg: EnvConfig,
                greedy: bool = True, seed: int = 0) -> np.ndarray:
    env = make_env(env_cfg, seed=seed)
    rng = np.random.default_rng(seed)
    return np.array([run_episode(env, policy, rng, greedy=greedy, ideal=F).cost for F in ideals])


@dataclass
class Comparison:
    name: str
    mean: float
    std: float
    improvement: float          # relative reduction of the mean vs. the reference (positive = better)
    diff_mean: float            # mean of (reference - method), in additions
    diff_ci: tuple[float, float]
    wilcoxon_p: float | None

    def as_row(self) -> dict:
        return {
            "method": self.name,
            "mean_additions": round(self.mean, 2),
            "std_additions": round(self.std, 2),
            "improvement_vs_ref_%": round(100 * self.improvement, 1),
            "mean_paired_diff": round(self.diff_mean, 2),
            "ci95_low": round(self.diff_ci[0], 2),
            "ci95_high": round(self.diff_ci[1], 2),
            "wilcoxon_p": None if self.wilcoxon_p is None else float(f"{self.wilcoxon_p:.3g}"),
        }


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare(results: Mapping[str, np.ndarray], reference: str) -> list[Comparison]:
    ref = results[reference]
    rows = []
    for name, c in results.items():
        d = ref - c
        p = None
        if name != reference and np.any(d != 0):
            p = float(sps.wilcoxon(ref, c).pvalue)
        rows.append(Comparison(
            name=name,
            mean=float(c.mean()),
            std=float(c.std()),
            improvement=float(1 - c.mean() / ref.mean()),
            diff_mean=float(d.mean()),
            diff_ci=bootstrap_ci(d) if name != reference else (0.0, 0.0),
            wilcoxon_p=p,
        ))
    return rows


def best_strategy(results: Mapping[str, np.ndarray], candidates: Sequence[str]) -> str:
    return min(candidates, key=lambda k: results[k].mean())


def format_table(rows: Sequence[dict]) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    if not rows:
        return ""
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")
    return "\n".join(out)

