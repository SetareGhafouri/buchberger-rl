"""S-polynomials, multivariate division, and the cost model.

Cost model
----------
Following Peifer, Stillman & Halpern-Leistner (2020), the cost of Buchberger's
algorithm is measured by the number of *polynomial additions*: every step of
the form ``p <- p - c * x^m * g`` counts as one addition, including the single
subtraction that forms the S-polynomial itself. Unlike wall-clock time, this
measure is independent of hardware and of low-level implementation details,
yet it tracks the dominant cost of a real implementation closely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import monomials as M
from .polynomial import Polynomial


@dataclass(frozen=True)
class Reduction:
    """Result of reducing a polynomial by a list of divisors."""

    remainder: Polynomial
    additions: int
    sugar: int | None = None


def spoly(f: Polynomial, g: Polynomial) -> Polynomial:
    """S-polynomial ``S(f, g) = (L / LT(f)) f - (L / LT(g)) g`` with ``L = lcm(LM f, LM g)``."""
    L = M.lcm(f.LM, g.LM)
    ring = f.ring
    a = f.scale(ring.inv(f.LC), M.div(L, f.LM))
    return a.sub_scaled(ring.inv(g.LC), M.div(L, g.LM), g)


def spoly_sugar(f_sugar: int, f: Polynomial, g_sugar: int, g: Polynomial) -> int:
    """Sugar degree of ``S(f, g)`` (Giovini, Mora, Niesi, Robbiano & Traverso, 1991)."""
    L = M.lcm(f.LM, g.LM)
    return max(f_sugar + M.degree(L) - M.degree(f.LM), g_sugar + M.degree(L) - M.degree(g.LM))


def reduce(
    f: Polynomial,
    divisors: Sequence[Polynomial],
    *,
    full: bool = True,
    sugar: int | None = None,
    divisor_sugars: Sequence[int] | None = None,
) -> Reduction:
    """Reduce ``f`` by ``divisors`` and count the polynomial additions used.

    Parameters
    ----------
    f:
        Polynomial to reduce.
    divisors:
        Reducers; the first divisor whose leading monomial divides the current
        leading monomial is used (the classical "first-divisor" rule).
    full:
        If ``True`` (default) the tail is reduced as well, so that no term of the
        remainder is divisible by any leading monomial. If ``False`` only the
        leading term is reduced (top reduction).
    sugar, divisor_sugars:
        Optional sugar bookkeeping. If given, the sugar of the remainder is
        returned, updated as ``max(sugar, sugar(g) + deg(x^m))`` at every
        reduction step ``p <- p - c x^m g``.
    """
    ring = f.ring
    lms = [g.LM for g in divisors]
    remainder: list[tuple[M.Monomial, int]] = []
    additions = 0
    p = f
    while not p.is_zero():
        m, c = p.LT
        for k, lm in enumerate(lms):
            if M.divides(lm, m):
                g = divisors[k]
                q = M.div(m, lm)
                coef = c if g.LC == 1 else (c * ring.inv(g.LC)) % ring.characteristic
                p = p.sub_scaled(coef, q, g)
                additions += 1
                if sugar is not None and divisor_sugars is not None:
                    sugar = max(sugar, divisor_sugars[k] + M.degree(q))
                break
        else:
            if not full:
                remainder.extend(p.terms)
                break
            remainder.append(p.terms[0])
            p = type(p)(ring, p.terms[1:])
    return Reduction(type(f)(ring, tuple(remainder)), additions, sugar)


def interreduce(polys: Sequence[Polynomial]) -> list[Polynomial]:
    """Compute the reduced Gröbner basis from *any* Gröbner basis.

    First discard every element whose leading monomial is divisible by the
    leading monomial of another element (minimalisation), then fully reduce
    each remaining element by the others and normalise to be monic. By
    uniqueness of the reduced Gröbner basis, the output is canonical, which
    makes it the right object to compare in tests.
    """
    polys = [g.monic() for g in polys if not g.is_zero()]
    if not polys:
        return []
    order_key = polys[0].ring.order.key
    polys.sort(key=lambda g: order_key(g.LM))  # small leading monomials first
    minimal: list[Polynomial] = []
    for g in polys:
        if not any(M.divides(h.LM, g.LM) for h in minimal):
            minimal.append(g)
    reduced = []
    for i, g in enumerate(minimal):
        others = minimal[:i] + minimal[i + 1 :]
        reduced.append(reduce(g, others).remainder.monic())
    reduced.sort(key=lambda g: order_key(g.LM), reverse=True)
    return reduced


def is_groebner_basis(G: Sequence[Polynomial]) -> bool:
    """Check Buchberger's criterion: every S-polynomial reduces to zero modulo ``G``."""
    G = [g for g in G if not g.is_zero()]
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            if not reduce(spoly(G[i], G[j]), G).remainder.is_zero():
                return False
    return True
