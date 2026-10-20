"""Buchberger's algorithm as an explicit, steppable state machine.

The algorithm is written as a state object rather than a single function so
that exactly the same code drives (i) the classical algorithm with hand-designed
selection strategies and (ii) the reinforcement-learning environment, in which
an agent chooses the next S-pair. Any difference in measured cost between a
heuristic and a learned policy is therefore due to the selection decisions
alone.

Pair elimination
----------------
Three variants are supported:

``"none"``
    Every pair is processed.
``"lcm"``
    Buchberger's first criterion: pairs with coprime leading monomials are
    discarded, since their S-polynomials always reduce to zero.
``"gebauermoeller"`` (default)
    The Gebauer--Möller installation of Buchberger's criteria (criteria M, F
    and B), in the formulation of Becker & Weispfenning, *Gröbner Bases*
    (1993), Sect. 5.5. Redundant basis elements are dropped from the active
    reducer set as soon as a new leading monomial divides theirs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from . import monomials as M
from .polynomial import Polynomial, PolynomialRing
from .reduction import interreduce, reduce, spoly, spoly_sugar

Pair = tuple[int, int]
ELIMINATIONS = ("none", "lcm", "gebauermoeller")


@dataclass
class StepResult:
    additions: int            # polynomial additions spent on this pair (>= 1)
    new_index: int | None     # index of the polynomial added to the basis, if any


@dataclass
class RunStatistics:
    additions: int = 0
    pairs_processed: int = 0
    zero_reductions: int = 0
    max_pairs: int = 0
    basis_size: int = 0


class BuchbergerState:
    """Mutable state ``(G, P)`` of Buchberger's algorithm.

    Attributes
    ----------
    polys:
        Every polynomial ever added (monic). Indices into this list are stable,
        so pairs can refer to polynomials that are no longer active reducers.
    basis:
        Indices of the active basis ``G`` used for reduction.
    pairs:
        Current pair set ``P``, in insertion order (which defines the "first"
        strategy).
    sugars:
        Sugar degree of each polynomial in ``polys``.
    """

    def __init__(self, ring: PolynomialRing, elimination: str = "gebauermoeller"):
        if elimination not in ELIMINATIONS:
            raise ValueError(f"elimination must be one of {ELIMINATIONS}, got {elimination!r}")
        self.ring = ring
        self.elimination = elimination
        self.polys: list[Polynomial] = []
        self.lms: list[M.Monomial] = []
        self.sugars: list[int] = []
        self.basis: list[int] = []
        self.pairs: list[Pair] = []

    # -- construction -------------------------------------------------------

    @classmethod
    def from_generators(
        cls, generators: Iterable[Polynomial], elimination: str = "gebauermoeller"
    ) -> "BuchbergerState":
        generators = list(generators)
        if not generators:
            raise ValueError("need at least one generator")
        state = cls(generators[0].ring, elimination)
        for f in generators:
            if not f.is_zero():
                state._add(f.monic(), sugar=f.degree())
        return state

    def copy(self) -> "BuchbergerState":
        """Cheap copy: polynomials are immutable, so only the index lists are copied."""
        other = BuchbergerState.__new__(BuchbergerState)
        other.ring = self.ring
        other.elimination = self.elimination
        other.polys = list(self.polys)
        other.lms = list(self.lms)
        other.sugars = list(self.sugars)
        other.basis = list(self.basis)
        other.pairs = list(self.pairs)
        return other

    # -- queries ------------------------------------------------------------

    @property
    def done(self) -> bool:
        return not self.pairs

    def pair_lcm(self, pair: Pair) -> M.Monomial:
        i, j = pair
        return M.lcm(self.lms[i], self.lms[j])

    def pair_sugar(self, pair: Pair) -> int:
        i, j = pair
        return spoly_sugar(self.sugars[i], self.polys[i], self.sugars[j], self.polys[j])

    def basis_polynomials(self) -> list[Polynomial]:
        return [self.polys[k] for k in self.basis]

    def reduced_basis(self) -> list[Polynomial]:
        return interreduce(self.basis_polynomials())

    # -- dynamics -----------------------------------------------------------

    def step(self, pair: Pair) -> StepResult:
        """Process one S-pair: remove it, reduce its S-polynomial, update ``(G, P)``."""
        self.pairs.remove(pair)
        i, j = pair
        s = spoly(self.polys[i], self.polys[j])
        sugar = self.pair_sugar(pair)
        additions = 1  # forming S(f, g) is itself one subtraction
        if s.is_zero():
            return StepResult(additions, None)
        reducers = self.basis
        red = reduce(
            s,
            [self.polys[k] for k in reducers],
            sugar=sugar,
            divisor_sugars=[self.sugars[k] for k in reducers],
        )
        additions += red.additions
        if red.remainder.is_zero():
            return StepResult(additions, None)
        idx = self._add(red.remainder.monic(), sugar=red.sugar)
        return StepResult(additions, idx)

    def _add(self, h: Polynomial, sugar: int) -> int:
        """Append ``h`` to the basis and update the pair set."""
        t = len(self.polys)
        self.polys.append(h)
        self.lms.append(h.LM)
        self.sugars.append(sugar)
        if self.elimination == "gebauermoeller":
            self._update_gebauer_moeller(t)
        else:
            lm_h = h.LM
            for g in self.basis:
                if self.elimination == "lcm" and M.is_coprime(self.lms[g], lm_h):
                    continue
                self.pairs.append((g, t))
            self.basis.append(t)
        return t

    def _update_gebauer_moeller(self, t: int) -> None:
        lms = self.lms
        lm_h = lms[t]

        # Candidate new pairs (g, t), with their lcms.
        C = [(g, M.lcm(lms[g], lm_h)) for g in self.basis]
        D: list[tuple[int, M.Monomial]] = []
        while C:
            g, L = C.pop(0)
            coprime = M.is_coprime(lms[g], lm_h)
            # Criteria M and F: drop (g, t) if another candidate's lcm divides L.
            # Pairs with equal lcm eliminate all but the last one examined.
            if coprime or (
                not any(M.divides(L2, L) for _, L2 in C) and not any(M.divides(L2, L) for _, L2 in D)
            ):
                D.append((g, L))
        # Criterion B (product criterion) applied to the survivors.
        new_pairs = [(g, t) for g, L in D if not M.is_coprime(lms[g], lm_h)]

        # Remove old pairs (a, b) made redundant by the chain criterion through t.
        kept = []
        for a, b in self.pairs:
            L = M.lcm(lms[a], lms[b])
            if M.divides(lm_h, L) and M.lcm(lms[a], lm_h) != L and M.lcm(lms[b], lm_h) != L:
                continue
            kept.append((a, b))
        self.pairs = kept + new_pairs

        # Drop reducers whose leading monomial is divisible by LM(h).
        self.basis = [g for g in self.basis if not M.divides(lm_h, lms[g])] + [t]


# ---------------------------------------------------------------------------
# Convenience driver
# ---------------------------------------------------------------------------

SelectionFn = Callable[[BuchbergerState], Pair]


@dataclass
class BuchbergerResult:
    basis: list[Polynomial]
    stats: RunStatistics = field(default_factory=RunStatistics)


def buchberger(
    generators: Sequence[Polynomial],
    select: SelectionFn | None = None,
    elimination: str = "gebauermoeller",
    reduced: bool = True,
) -> BuchbergerResult:
    """Compute a Gröbner basis of ``<generators>``.

    ``select`` maps the current state to the next pair; it defaults to the
    normal strategy. With ``reduced=True`` the reduced Gröbner basis is returned.
    """
    if select is None:
        from ..strategies import NormalStrategy

        select = NormalStrategy()
    state = BuchbergerState.from_generators(generators, elimination)
    stats = RunStatistics(max_pairs=len(state.pairs))
    while not state.done:
        result = state.step(select(state))
        stats.additions += result.additions
        stats.pairs_processed += 1
        stats.zero_reductions += result.new_index is None
        stats.max_pairs = max(stats.max_pairs, len(state.pairs))
    basis = state.reduced_basis() if reduced else state.basis_polynomials()
    stats.basis_size = len(basis)
    return BuchbergerResult(basis, stats)
