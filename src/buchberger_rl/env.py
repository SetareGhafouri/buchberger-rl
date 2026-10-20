"""Buchberger's algorithm as a Markov decision process.

* **State** ``s_t = (G_t, P_t)``: the current generating set and pair set.
* **Action** ``a_t in P_t``: the S-pair to process next. The action set changes
  size at every step, which rules out standard fixed-output policy networks.
* **Transition**: deterministic -- remove the pair, reduce its S-polynomial,
  and update ``(G, P)`` with the Gebauer--Möller criteria. Randomness enters
  only through the initial distribution over ideals.
* **Reward** ``r_t = -(polynomial additions spent on a_t)``. With ``gamma = 1``
  the return is minus the total cost, so maximising expected return is exactly
  minimising the expected cost of computing a Gröbner basis.

Observation
-----------
Each pair ``(f, g)`` is encoded by the exponent vectors of the ``k`` leading
terms of ``f`` and of ``g`` (zero-padded), giving a fixed-length vector of size
``2 k n``. The observation is the matrix whose rows are these vectors, one row
per pair. Coefficients are not observed, so the MDP is only partially
observed; for binomial ideals and ``k = 2``, however, every monomial occurring
in the computation is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .algebra.buchberger import BuchbergerState, Pair
from .algebra.polynomial import Polynomial
from .strategies import DegreeStrategy, Strategy


@dataclass
class StepInfo:
    additions: int
    pair: Pair


class PairFeaturizer:
    """Encode pairs as fixed-length vectors of leading-term exponents."""

    def __init__(self, nvars: int, k: int = 2, scale: float = 1.0):
        self.nvars = nvars
        self.k = k
        self.scale = scale
        self.dim = 2 * k * nvars
        self._cache: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._cache.clear()

    def poly_vector(self, idx: int, f: Polynomial) -> np.ndarray:
        v = self._cache.get(idx)
        if v is None:
            v = np.zeros(self.k * self.nvars, dtype=np.float32)
            for t, (m, _) in enumerate(f.terms[: self.k]):
                v[t * self.nvars : (t + 1) * self.nvars] = m
            v *= self.scale
            self._cache[idx] = v
        return v

    def __call__(self, state: BuchbergerState) -> np.ndarray:
        if not state.pairs:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = [
            np.concatenate((self.poly_vector(i, state.polys[i]), self.poly_vector(j, state.polys[j])))
            for i, j in state.pairs
        ]
        return np.stack(rows)


class BuchbergerEnv:
    """Gym-style environment: ``reset() -> obs``, ``step(a) -> (obs, reward, done, info)``.

    Actions are row indices into the current observation (equivalently, into
    ``env.state.pairs``).
    """

    def __init__(
        self,
        ideal_generator: Callable[[], Sequence[Polynomial]],
        nvars: int,
        elimination: str = "gebauermoeller",
        k: int = 2,
        feature_scale: float = 1.0,
        max_resamples: int = 1000,
    ):
        self.ideal_generator = ideal_generator
        self.elimination = elimination
        self.featurizer = PairFeaturizer(nvars, k, feature_scale)
        self.max_resamples = max_resamples
        self.state: BuchbergerState | None = None

    @property
    def observation_dim(self) -> int:
        return self.featurizer.dim

    def reset(self, ideal: Sequence[Polynomial] | None = None) -> np.ndarray:
        """Start a new episode.

        Without an explicit ``ideal``, ideals whose initial pair set is empty
        (the generators already form a Gröbner basis after pair elimination)
        are resampled, since they contain no decision to learn from.
        """
        self.featurizer.reset()
        if ideal is not None:
            self.state = BuchbergerState.from_generators(ideal, self.elimination)
            return self.featurizer(self.state)
        for _ in range(self.max_resamples):
            self.state = BuchbergerState.from_generators(self.ideal_generator(), self.elimination)
            if self.state.pairs:
                return self.featurizer(self.state)
        raise RuntimeError("ideal generator keeps producing trivial ideals")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        assert self.state is not None, "call reset() first"
        pair = self.state.pairs[action]
        result = self.state.step(pair)
        obs = self.featurizer(self.state)
        return obs, -float(result.additions), self.state.done, StepInfo(result.additions, pair)


def rollout_cost(state: BuchbergerState, strategy: Strategy | None = None) -> int:
    """Total additions needed to finish ``state`` with ``strategy`` (default: degree).

    Used as a state-value estimate ``V(s) = -rollout_cost(s)``, as in Peifer et
    al. Expensive -- quadratic in episode length overall -- but it gives the
    agent an informative baseline from the very first epoch.
    """
    strategy = strategy or DegreeStrategy()
    s = state.copy()
    cost = 0
    while not s.done:
        cost += s.step(strategy(s)).additions
    return cost
