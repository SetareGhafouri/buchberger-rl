"""Sparse multivariate polynomials over a prime field GF(p).

Design notes
------------
* Polynomials are immutable. Terms are stored as a tuple of ``(monomial,
  coefficient)`` pairs sorted in *descending* order with respect to the ring's
  monomial order, so the leading term is always ``terms[0]``.
* Coefficients live in ``{1, ..., p - 1}``; zero coefficients are never stored.
* Working over a large prime field (the default is p = 32003, the customary
  choice in Singular and Macaulay2) avoids coefficient growth, which would
  otherwise dominate the cost of Buchberger's algorithm and obscure the effect
  of the selection strategy we actually want to study.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import monomials as M
from .monomials import Monomial, MonomialOrder, get_order

DEFAULT_CHARACTERISTIC = 32003


@dataclass(frozen=True)
class PolynomialRing:
    """The ring ``GF(p)[x_1, ..., x_n]`` equipped with a monomial order."""

    nvars: int
    order: MonomialOrder = field(default_factory=lambda: get_order("grevlex"))
    characteristic: int = DEFAULT_CHARACTERISTIC
    variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.nvars < 1:
            raise ValueError("a polynomial ring needs at least one variable")
        object.__setattr__(self, "order", get_order(self.order))
        if not self.variables:
            names = ("x", "y", "z", "w") if self.nvars <= 4 else tuple(f"x{i}" for i in range(self.nvars))
            object.__setattr__(self, "variables", tuple(names[: self.nvars]))
        if len(self.variables) != self.nvars:
            raise ValueError("number of variable names must equal nvars")

    # -- construction -------------------------------------------------------

    def zero(self) -> "Polynomial":
        return Polynomial(self, ())

    def one(self) -> "Polynomial":
        return self.monomial((0,) * self.nvars)

    def monomial(self, m: Monomial, c: int = 1) -> "Polynomial":
        return self.from_dict({tuple(m): c})

    def from_dict(self, terms: dict[Monomial, int]) -> "Polynomial":
        p = self.characteristic
        cleaned = {tuple(m): c % p for m, c in terms.items() if c % p}
        return Polynomial(self, self._sorted(cleaned))

    def from_terms(self, terms: Iterable[tuple[Monomial, int]]) -> "Polynomial":
        acc: dict[Monomial, int] = {}
        for m, c in terms:
            acc[tuple(m)] = acc.get(tuple(m), 0) + c
        return self.from_dict(acc)

    def parse(self, text: str) -> "Polynomial":
        """Parse expressions such as ``"x^3 - 2*x*y + 1/2*y^2"``.

        Rational coefficients ``a/b`` are mapped to ``a * b^{-1}`` in GF(p).
        Intended for tests and examples rather than heavy input.
        """
        return _parse(self, text)

    # -- helpers ------------------------------------------------------------

    def inv(self, c: int) -> int:
        return pow(c, -1, self.characteristic)

    def _sorted(self, terms: dict[Monomial, int]) -> tuple[tuple[Monomial, int], ...]:
        key = self.order.key
        return tuple(sorted(terms.items(), key=lambda t: key(t[0]), reverse=True))


class Polynomial:
    """An immutable sparse polynomial. Construct through :class:`PolynomialRing`."""

    __slots__ = ("ring", "terms", "_hash")

    def __init__(self, ring: PolynomialRing, terms: tuple[tuple[Monomial, int], ...]):
        self.ring = ring
        self.terms = terms
        self._hash: int | None = None

    # -- leading data -------------------------------------------------------

    def is_zero(self) -> bool:
        return not self.terms

    @property
    def LM(self) -> Monomial:
        """Leading monomial."""
        return self.terms[0][0]

    @property
    def LC(self) -> int:
        """Leading coefficient."""
        return self.terms[0][1]

    @property
    def LT(self) -> tuple[Monomial, int]:
        return self.terms[0]

    def __len__(self) -> int:
        return len(self.terms)

    def degree(self) -> int:
        return max((M.degree(m) for m, _ in self.terms), default=-1)

    def monomials(self) -> list[Monomial]:
        return [m for m, _ in self.terms]

    # -- arithmetic ---------------------------------------------------------

    def monic(self) -> "Polynomial":
        if self.is_zero() or self.LC == 1:
            return self
        return self.scale(self.ring.inv(self.LC))

    def scale(self, c: int, m: Monomial | None = None) -> "Polynomial":
        """Return ``c * x^m * self``. Multiplication by a monomial preserves the term order."""
        p = self.ring.characteristic
        c %= p
        if c == 0:
            return self.ring.zero()
        if m is None:
            return Polynomial(self.ring, tuple((t, (a * c) % p) for t, a in self.terms))
        return Polynomial(self.ring, tuple((M.mul(t, m), (a * c) % p) for t, a in self.terms))

    def sub_scaled(self, c: int, m: Monomial, other: "Polynomial") -> "Polynomial":
        """Return ``self - c * x^m * other`` -- the single primitive used by reduction."""
        p = self.ring.characteristic
        acc = dict(self.terms)
        for t, a in other.terms:
            tm = M.mul(t, m)
            v = (acc.get(tm, 0) - c * a) % p
            if v:
                acc[tm] = v
            else:
                acc.pop(tm, None)
        return Polynomial(self.ring, self.ring._sorted(acc))

    def __add__(self, other: "Polynomial") -> "Polynomial":
        return self.sub_scaled(-1, (0,) * self.ring.nvars, other)

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        return self.sub_scaled(1, (0,) * self.ring.nvars, other)

    def __neg__(self) -> "Polynomial":
        return self.scale(-1)

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        acc: dict[Monomial, int] = {}
        p = self.ring.characteristic
        for m1, c1 in self.terms:
            for m2, c2 in other.terms:
                m = M.mul(m1, m2)
                acc[m] = (acc.get(m, 0) + c1 * c2) % p
        return self.ring.from_dict(acc)

    # -- comparison & display -----------------------------------------------

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Polynomial) and self.ring == other.ring and self.terms == other.terms

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.terms)
        return self._hash

    def __repr__(self) -> str:
        return f"Polynomial({self})"

    def __str__(self) -> str:
        if self.is_zero():
            return "0"
        p = self.ring.characteristic
        parts = []
        for m, c in self.terms:
            # print coefficients in the symmetric range (-p/2, p/2] for readability
            cs = c if c <= p // 2 else c - p
            mono = "*".join(
                v if e == 1 else f"{v}^{e}" for v, e in zip(self.ring.variables, m) if e
            )
            if not mono:
                body = str(abs(cs))
            elif abs(cs) == 1:
                body = mono
            else:
                body = f"{abs(cs)}*{mono}"
            parts.append(("- " if cs < 0 else "+ ") + body)
        s = " ".join(parts)
        return s[2:] if s.startswith("+ ") else "-" + s[1:]


# ---------------------------------------------------------------------------
# A small, forgiving parser (tests / examples only)
# ---------------------------------------------------------------------------

_TERM = re.compile(r"([+-]?)\s*([^+-]+)")


def _parse(ring: PolynomialRing, text: str) -> Polynomial:
    index = {v: i for i, v in enumerate(ring.variables)}
    p = ring.characteristic
    acc: dict[Monomial, int] = {}
    src = text.replace(" ", "").replace("**", "^")
    if not src:
        raise ValueError("empty polynomial")
    for sign, body in _TERM.findall(src):
        coeff = -1 if sign == "-" else 1
        exps = [0] * ring.nvars
        for factor in body.split("*"):
            if not factor:
                continue
            if factor[0].isdigit():
                num, _, den = factor.partition("/")
                coeff = coeff * int(num) * (pow(int(den), -1, p) if den else 1)
                continue
            var, _, exp = factor.partition("^")
            if var not in index:
                raise ValueError(f"unknown variable {var!r} in {text!r}")
            exps[index[var]] += int(exp) if exp else 1
        m = tuple(exps)
        acc[m] = (acc.get(m, 0) + coeff) % p
    return ring.from_dict(acc)


def parse_many(ring: PolynomialRing, texts: Sequence[str]) -> list[Polynomial]:
    return [ring.parse(t) for t in texts]
