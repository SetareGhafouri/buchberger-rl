"""Correctness of monomial orders, polynomial arithmetic and Buchberger's algorithm."""

import itertools
import unittest

from buchberger_rl.algebra import (
    BuchbergerState,
    PolynomialRing,
    buchberger,
    interreduce,
    is_groebner_basis,
    reduce,
    spoly,
)
from buchberger_rl.algebra import monomials as M
from buchberger_rl.ideals import RandomBinomialIdealGenerator
from buchberger_rl.strategies import STRATEGIES, make_strategy


class TestMonomialOrders(unittest.TestCase):
    def test_textbook_comparisons(self):
        # Cox, Little & O'Shea, Ideals, Varieties, and Algorithms, Sect. 2.2
        lex, grlex, grevlex = (M.get_order(o) for o in ("lex", "grlex", "grevlex"))
        self.assertTrue(lex.gt((1, 2, 0), (0, 3, 4)))
        self.assertTrue(lex.gt((3, 2, 4), (3, 2, 1)))
        self.assertTrue(grlex.gt((1, 2, 3), (3, 2, 0)))
        self.assertTrue(grlex.gt((1, 2, 4), (1, 1, 5)))
        self.assertTrue(grevlex.gt((4, 7, 1), (4, 2, 3)))
        self.assertTrue(grevlex.gt((1, 5, 2), (4, 1, 3)))

    def test_orders_are_total_and_respect_multiplication(self):
        mons = [m for d in range(4) for m in M.monomials_of_degree(3, d)]
        for name in ("lex", "grlex", "grevlex"):
            o = M.get_order(name)
            for a, b in itertools.combinations(mons, 2):
                self.assertNotEqual(o.key(a), o.key(b))
                for c in mons[:10]:
                    self.assertEqual(o.gt(a, b), o.gt(M.mul(a, c), M.mul(b, c)))

    def test_monomial_counts(self):
        for n, d in [(1, 5), (2, 4), (3, 6), (4, 3)]:
            self.assertEqual(len(list(M.monomials_of_degree(n, d))), M.count_monomials_of_degree(n, d))


class TestPolynomials(unittest.TestCase):
    def setUp(self):
        self.R = PolynomialRing(3, "grevlex", 32003)

    def test_parse_and_arithmetic(self):
        f, g = self.R.parse("x^2*y - 3*z + 1"), self.R.parse("x*y - z")
        self.assertEqual(f - f, self.R.zero())
        self.assertEqual((f + g) - g, f)
        self.assertEqual(f * self.R.one(), f)
        self.assertEqual(self.R.parse("1/2*x") + self.R.parse("1/2*x"), self.R.parse("x"))
        self.assertEqual(self.R.parse("x"), self.R.monomial((1, 0, 0)))

    def test_leading_terms_follow_order(self):
        f = self.R.parse("z^3 + x*y + x^2*z")
        self.assertEqual(f.LM, (2, 0, 1))  # grevlex: x^2 z > z^3 (smaller power of the last variable)

    def test_division_identity(self):
        # f = sum q_i g_i + r must hold; check via ideal arithmetic on a small example.
        R = PolynomialRing(2, "lex")
        f = R.parse("x^2*y + x*y^2 + y^2")
        G = [R.parse("x*y - 1"), R.parse("y^2 - 1")]
        r = reduce(f, G).remainder
        self.assertEqual(r, R.parse("x + y + 1"))  # Cox-Little-O'Shea, Example 2.3.2

    def test_spoly_cancels_leading_terms(self):
        f, g = self.R.parse("x^3*y^2 - x^2*y^3 + x"), self.R.parse("3*x^4*y + y^2")
        s = spoly(f, g)
        L = M.lcm(f.LM, g.LM)
        self.assertNotIn(L, s.monomials())


class TestBuchberger(unittest.TestCase):
    def test_textbook_example_grlex(self):
        # The worked example of Chapter 3 of the thesis (Cox-Little-O'Shea, Example 2.7.1).
        R = PolynomialRing(2, "grlex")
        F = [R.parse("x^3 - 2*x*y"), R.parse("x^2*y - 2*y^2 + x")]
        expected = interreduce([R.parse("x^2"), R.parse("x*y"), R.parse("y^2 - 1/2*x")])
        self.assertEqual(buchberger(F).basis, expected)

    def test_twisted_cubic_lex(self):
        # Ideal of the twisted cubic (t, t^2, t^3): reduced lex basis is classical.
        R = PolynomialRing(3, "lex")
        F = [R.parse("x^2 - y"), R.parse("x^3 - z")]
        expected = interreduce([R.parse(s) for s in ("x^2 - y", "x*y - z", "x*z - y^2", "y^3 - z^2")])
        self.assertEqual(buchberger(F).basis, expected)

    def test_unit_ideal(self):
        R = PolynomialRing(2, "grevlex")
        F = [R.parse("x*y - 1"), R.parse("x"), R.parse("y^2 + 1")]
        self.assertEqual(buchberger(F).basis, [R.one()])

    def test_random_ideals_all_strategies_and_eliminations_agree(self):
        """The reduced Gröbner basis is unique, so every strategy must produce the same one."""
        gen = RandomBinomialIdealGenerator(3, 8, 5, degrees="weighted", seed=7)
        for _ in range(15):
            F = gen.sample()
            reference = None
            for elim in ("none", "lcm", "gebauermoeller"):
                for name in STRATEGIES:
                    kwargs = {"seed": 0} if name == "random" else {}
                    res = buchberger(F, make_strategy(name, **kwargs), elimination=elim)
                    self.assertTrue(is_groebner_basis(res.basis))
                    if reference is None:
                        reference = res.basis
                    self.assertEqual(res.basis, reference, f"{name}/{elim}")
            # every generator lies in the ideal: it reduces to zero modulo the basis
            for f in F:
                self.assertTrue(reduce(f, reference).remainder.is_zero())

    def test_non_binomial_inputs(self):
        R = PolynomialRing(3, "grevlex", 101)
        F = [R.parse("x^2 + y*z + 2*z - 1"), R.parse("x*y + y^2 + 3"), R.parse("x*z - y + z^2")]
        bases = [buchberger(F, make_strategy(s)).basis for s in ("first", "degree", "sugar")]
        self.assertTrue(is_groebner_basis(bases[0]))
        self.assertEqual(bases[0], bases[1])
        self.assertEqual(bases[0], bases[2])

    def test_gebauer_moeller_saves_work(self):
        gen = RandomBinomialIdealGenerator(3, 10, 6, degrees="weighted", seed=3)
        F = gen.sample()
        cost = {e: buchberger(F, make_strategy("degree"), elimination=e).stats.additions
                for e in ("none", "gebauermoeller")}
        self.assertLessEqual(cost["gebauermoeller"], cost["none"])

    def test_state_copy_is_independent(self):
        gen = RandomBinomialIdealGenerator(3, 10, 6, seed=1)
        state = BuchbergerState.from_generators(gen.sample())
        pairs, polys, basis = list(state.pairs), list(state.polys), list(state.basis)
        clone = state.copy()
        while not clone.done:
            clone.step(clone.pairs[0])
        self.assertEqual((state.pairs, state.polys, state.basis), (pairs, polys, basis))


try:  # optional cross-check against an independent implementation
    import sympy
except ImportError:  # pragma: no cover
    sympy = None


@unittest.skipIf(sympy is None, "sympy not installed")
class TestAgainstSympy(unittest.TestCase):
    def test_random_binomial_ideals(self):
        x, y, z = sympy.symbols("x y z")
        gen = RandomBinomialIdealGenerator(3, 6, 4, degrees="weighted", characteristic=101, seed=11)
        R = gen.ring
        for _ in range(10):
            F = gen.sample()
            exprs = [sum(c * x**a * y**b * z**e for (a, b, e), c in f.terms) for f in F]
            G = sympy.groebner(exprs, x, y, z, order="grevlex", modulus=101)
            theirs = interreduce([R.from_dict({m: int(c) for m, c in sympy.Poly(g, x, y, z).terms()}) for g in G.exprs])
            self.assertEqual(buchberger(F).basis, theirs)


if __name__ == "__main__":
    unittest.main()
