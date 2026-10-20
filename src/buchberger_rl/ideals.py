"""Random distributions of polynomial ideals.

The main family is random **binomial** ideals, written ``n-d-s-<degrees>``:
``s`` generators in ``n`` variables, each a binomial ``c1 x^a + c2 x^b`` with
distinct monomials of degree at most ``d`` and coefficients drawn uniformly
from ``GF(p)^*``. Binomial ideals are a natural testbed: they are ubiquitous in
applications (toric ideals, chemical reaction networks, integer programming),
every polynomial produced by Buchberger's algorithm on them is again a
binomial, and yet the cost of the computation varies enormously with the
order in which S-pairs are processed.

How the two monomials of each generator are drawn is controlled by
``degrees``:

``"uniform"``
    a monomial uniformly from *all* monomials of degree ``1..d``; equivalently,
    degree ``k`` is chosen with probability proportional to ``C(n+k-1, k)``,
    which heavily favours high degrees.
``"weighted"``
    first a degree ``k`` uniformly from ``{1, ..., d}``, then a monomial
    uniformly among those of degree ``k``. Monomials are thus *weighted* so
    that every degree carries equal mass, and low degrees are well represented.
``"maximum"``
    both monomials uniformly among those of degree exactly ``d``.

Setting ``constants=True`` also allows the monomial ``1`` (degree 0).

These names follow Peifer, Stillman & Halpern-Leistner (2020); the precise
sampling rules above are this project's own and are stated explicitly so that
results can be reproduced and compared. With these rules the mean cost of the
classical strategies on ``3-20-10-weighted`` is close to the value reported by
Peifer et al., which suggests the definitions agree, but this has not been
verified against their code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .algebra import monomials as M
from .algebra.polynomial import DEFAULT_CHARACTERISTIC, Polynomial, PolynomialRing

DEGREE_DISTRIBUTIONS = ("uniform", "weighted", "maximum")


@dataclass
class RandomBinomialIdealGenerator:
    nvars: int
    max_degree: int
    num_generators: int
    degrees: str = "uniform"
    constants: bool = False
    order: str = "grevlex"
    characteristic: int = DEFAULT_CHARACTERISTIC
    seed: int | None = None
    ring: PolynomialRing = field(init=False)

    def __post_init__(self) -> None:
        if self.degrees not in DEGREE_DISTRIBUTIONS:
            raise ValueError(f"degrees must be one of {DEGREE_DISTRIBUTIONS}")
        self.ring = PolynomialRing(self.nvars, self.order, self.characteristic)
        self.rng = np.random.default_rng(self.seed)
        lo = 0 if self.constants else 1
        self._by_degree = {
            k: list(M.monomials_of_degree(self.nvars, k)) for k in range(lo, self.max_degree + 1)
        }
        sizes = np.array([len(v) for v in self._by_degree.values()], dtype=float)
        self._degrees = np.array(list(self._by_degree), dtype=int)
        if self.degrees == "uniform":
            self._degree_probs = sizes / sizes.sum()
        elif self.degrees == "weighted":
            self._degree_probs = np.full(len(sizes), 1.0 / len(sizes))
        else:  # maximum
            self._degree_probs = (self._degrees == self.max_degree).astype(float)

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "RandomBinomialIdealGenerator":
        """Build a generator from a label such as ``"3-20-10-weighted"``."""
        try:
            n, d, s, degrees = name.split("-")
            return cls(int(n), int(d), int(s), degrees=degrees, **kwargs)
        except ValueError as exc:
            raise ValueError(f"expected 'n-d-s-<degrees>', got {name!r}") from exc

    @property
    def name(self) -> str:
        return f"{self.nvars}-{self.max_degree}-{self.num_generators}-{self.degrees}"

    def seed_(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def _monomial(self) -> M.Monomial:
        k = self.rng.choice(self._degrees, p=self._degree_probs)
        pool = self._by_degree[int(k)]
        return pool[self.rng.integers(len(pool))]

    def _binomial(self) -> Polynomial:
        while True:
            a, b = self._monomial(), self._monomial()
            if a != b:
                break
        c1, c2 = self.rng.integers(1, self.characteristic, size=2)
        return self.ring.from_dict({a: int(c1), b: int(c2)})

    def sample(self) -> list[Polynomial]:
        return [self._binomial() for _ in range(self.num_generators)]

    def __call__(self) -> list[Polynomial]:
        return self.sample()
