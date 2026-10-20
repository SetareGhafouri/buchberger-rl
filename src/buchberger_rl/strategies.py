"""Hand-designed S-pair selection strategies.

Each strategy is a callable ``state -> pair``. Ties are always broken by
insertion order (the "first" strategy), which keeps every strategy fully
deterministic and makes comparisons between them reproducible.

The classical strategies are

* **first**  -- FIFO queue order;
* **degree** -- minimal total degree of ``lcm(LM f, LM g)``;
* **normal** -- minimal ``lcm(LM f, LM g)`` in the monomial order
  (Buchberger, 1985);
* **sugar**  -- minimal sugar degree of the S-polynomial (Giovini et al., 1991).

Two further strategies make explicit the behaviour that the trained agents
appear to learn (see ``scripts/analyze_policy.py``). Both need to look one step
ahead by forming the S-polynomial of every candidate pair:

* **true_degree**   -- minimal degree of ``LM(S(f, g))``;
* **monomial_first** -- pairs whose S-polynomial is a single term first, then
  the degree strategy.
"""

from __future__ import annotations

import random
from typing import Callable

from .algebra import monomials as M
from .algebra.buchberger import BuchbergerState, Pair
from .algebra.reduction import spoly


class Strategy:
    name = "base"

    def __call__(self, state: BuchbergerState) -> Pair:  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _KeyStrategy(Strategy):
    """Pick the pair minimising ``key(state, pair)``; ties go to the earliest pair."""

    def key(self, state: BuchbergerState, pair: Pair):  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, state: BuchbergerState) -> Pair:
        return min(state.pairs, key=lambda p: self.key(state, p))


class FirstStrategy(Strategy):
    name = "first"

    def __call__(self, state: BuchbergerState) -> Pair:
        return state.pairs[0]


class RandomStrategy(Strategy):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def __call__(self, state: BuchbergerState) -> Pair:
        return self.rng.choice(state.pairs)


class DegreeStrategy(_KeyStrategy):
    name = "degree"

    def key(self, state, pair):
        return M.degree(state.pair_lcm(pair))


class NormalStrategy(_KeyStrategy):
    name = "normal"

    def key(self, state, pair):
        # the smallest lcm with respect to the monomial order
        return state.ring.order.key(state.pair_lcm(pair))


class SugarStrategy(_KeyStrategy):
    name = "sugar"

    def key(self, state, pair):
        return (state.pair_sugar(pair), state.ring.order.key(state.pair_lcm(pair)))


class TrueDegreeStrategy(_KeyStrategy):
    name = "true_degree"

    def key(self, state, pair):
        s = spoly(state.polys[pair[0]], state.polys[pair[1]])
        return (-1 if s.is_zero() else M.degree(s.LM), M.degree(state.pair_lcm(pair)))


class MonomialFirstStrategy(_KeyStrategy):
    name = "monomial_first"

    def key(self, state, pair):
        s = spoly(state.polys[pair[0]], state.polys[pair[1]])
        return (0 if len(s) <= 1 else 1, M.degree(state.pair_lcm(pair)))


STRATEGIES: dict[str, Callable[..., Strategy]] = {
    cls.name: cls
    for cls in (
        FirstStrategy,
        RandomStrategy,
        DegreeStrategy,
        NormalStrategy,
        SugarStrategy,
        TrueDegreeStrategy,
        MonomialFirstStrategy,
    )
}

CLASSICAL = ("first", "degree", "normal", "sugar")


def make_strategy(name: str, **kwargs) -> Strategy:
    try:
        return STRATEGIES[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from exc
