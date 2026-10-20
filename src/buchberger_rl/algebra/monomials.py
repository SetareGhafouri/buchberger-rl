"""Monomials and monomial orders.

A monomial ``x_1^{a_1} ... x_n^{a_n}`` is represented by its exponent vector,
a tuple of non-negative integers. Monomial orders are represented by *sort
keys*: a function ``key`` such that ``m1 > m2`` in the order if and only if
``key(m1) > key(m2)`` under Python's lexicographic tuple comparison. This keeps
every comparison inside the C implementation of tuple ordering, which matters
because reduction performs a very large number of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, Iterator

Monomial = tuple[int, ...]


def degree(m: Monomial) -> int:
    """Total degree ``|m| = a_1 + ... + a_n``."""
    return sum(m)


def mul(a: Monomial, b: Monomial) -> Monomial:
    return tuple(x + y for x, y in zip(a, b))


def div(a: Monomial, b: Monomial) -> Monomial:
    """Quotient ``a / b``; the caller guarantees that ``b`` divides ``a``."""
    return tuple(x - y for x, y in zip(a, b))


def divides(a: Monomial, b: Monomial) -> bool:
    """Return ``True`` if ``a`` divides ``b``."""
    return all(x <= y for x, y in zip(a, b))


def lcm(a: Monomial, b: Monomial) -> Monomial:
    return tuple(x if x >= y else y for x, y in zip(a, b))


def is_coprime(a: Monomial, b: Monomial) -> bool:
    """``True`` if ``a`` and ``b`` share no variable, i.e. ``lcm(a, b) = a * b``."""
    return all(x == 0 or y == 0 for x, y in zip(a, b))


def monomials_of_degree(n: int, d: int) -> Iterator[Monomial]:
    """Enumerate all monomials of total degree ``d`` in ``n`` variables.

    There are ``C(n + d - 1, d)`` of them; they are produced in lex-descending
    order.
    """
    if n == 1:
        yield (d,)
        return
    for first in range(d, -1, -1):
        for rest in monomials_of_degree(n - 1, d - first):
            yield (first, *rest)


def count_monomials_of_degree(n: int, d: int) -> int:
    return comb(n + d - 1, d)


# ---------------------------------------------------------------------------
# Monomial orders
# ---------------------------------------------------------------------------


def _lex_key(m: Monomial) -> tuple:
    return m


def _grlex_key(m: Monomial) -> tuple:
    return (sum(m), m)


def _grevlex_key(m: Monomial) -> tuple:
    # Ties in total degree are broken in favour of the monomial with the
    # *smaller* exponent in the last variable that differs.
    return (sum(m), tuple(-e for e in reversed(m)))


@dataclass(frozen=True)
class MonomialOrder:
    """A named monomial order given by a sort key (larger key = larger monomial)."""

    name: str
    key: Callable[[Monomial], tuple]

    def max(self, *monomials: Monomial) -> Monomial:
        return max(monomials, key=self.key)

    def gt(self, a: Monomial, b: Monomial) -> bool:
        return self.key(a) > self.key(b)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MonomialOrder({self.name!r})"


LEX = MonomialOrder("lex", _lex_key)
GRLEX = MonomialOrder("grlex", _grlex_key)
GREVLEX = MonomialOrder("grevlex", _grevlex_key)

_ORDERS = {o.name: o for o in (LEX, GRLEX, GREVLEX)}


def get_order(order: str | MonomialOrder) -> MonomialOrder:
    if isinstance(order, MonomialOrder):
        return order
    try:
        return _ORDERS[order.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown monomial order {order!r}; choose from {sorted(_ORDERS)}") from exc
