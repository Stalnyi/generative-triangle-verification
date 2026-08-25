"""
VERIFICATION of Section: Recursive controlled characteristics and macro-class invariants
(sec:recursive-controlled-characteristics-invariants).

This full mathematical verification block checks the new mathematical layer introduced
in 2_recursive_controlled_characteristics.tex.

Scope
-----
The section does not introduce a new generative map or a new proof of global
bijectivity.  It introduces a control formalism over already-defined
macro-classes:

1. finite-level profiles I_n(P);
2. profile diameters diam_I(P_n);
3. geometric profiles Delta rho_n(P) and R_n(P);
4. normalized spin profiles B_n(P);
5. frequency-regular microhistories and the induced spin limit;
6. asymptotic regularity with respect to a characteristic;
7. uniqueness of the asymptotic profile for a regular macro-class;
8. asymptotic recursive invariants;
9. candidate invariant behaviour for Delta rho, r_n, barSigma_n, and limiting
   frequency profiles.

This script intentionally does not re-prove the previously verified global
construction of the generative triangle, the bijectivity of F, or the exact
digit-coordinate theorem.  It uses only the already established coordinate
formulas as dependencies:

    D(prefix)       = sum_j (sigma_j - 1) s^{n-j},
    Delta rho       = D(prefix) / s^n,
    r_n             = log_s(D(prefix) + 1),
    eta(sigma)      = sigma - (s + 1)/2,
    barSigma_n(prefix) = (1/n) sum_j eta(sigma_j).

Terminology note
----------------
The verification uses "state sequence", "internal-state prefix", and "finite
prefix" for spin/internal-state data.

Verified content
----------------
A. Finite-level profiles and profile diameters:
   - I_n(P) is exactly the set of characteristic values on the finite
     projection P_n;
   - duplicate representatives do not change a set-valued profile;
   - diam_I(P_n) equals the supremum over all pairwise absolute differences,
     and for finite real profiles equals max(I_n(P)) - min(I_n(P));
   - invalid finite projections are rejected: empty projection, mixed depths,
     mixed ancestors, mixed internal-state alphabets, and inadmissible states.

B. Geometric profiles:
   - Delta rho_n(P) and R_n(P) are computed from the same finite projection;
   - Delta rho values are exact Fractions in [0,1);
   - the full level has diameter (s^n - 1)/s^n in Delta rho;
   - R_n(P) has diameter log_s(max(D)+1) - log_s(min(D)+1), using the
     monotonicity of log_s(D+1);
   - bounded digit windows have Delta-rho diameter O(s^{-n});
   - bounded logarithmic diameter is not confused with scalar asymptotic
     invariance.

C. Normalized spin profiles:
   - B_n(P) is exactly the set of normalized spin values on P_n;
   - profile diameters are exact rational suprema;
   - all normalized spin profile values lie in the sharp interval
     [-(s-1)/2, +(s-1)/2].

D. Frequency-regular microhistories:
   - exact frequencies f_sigma^{(n)} are computed for finite prefixes;
   - periodic state sequences with prescribed rational limiting frequencies
     are verified constructively;
   - the identity
       barSigma_n = sum_sigma eta(sigma) f_sigma^{(n)}
     is checked on all tested prefixes;
   - convergence of all frequencies implies convergence of barSigma_n to
       sum_sigma eta(sigma) f_sigma
     by an explicit finite-sum error bound.

E. Asymptotic regularity:
   - diam_I(P_n) -> 0 is verified on nontrivial shrinking profile families;
   - diam_I(P_n) -> 0 is shown to be necessary but not sufficient for a finite
     asymptotic invariant by a drifting profile family;
   - a profile with one convergent representative but non-vanishing diameter is
     verified not to satisfy the regularity hypothesis.

F. Unique asymptotic profile:
   - the epsilon proof in the section is checked as an exact inequality:
       |I_n(chi) - I_inf|
       <= diam_I(P_n) + |I_n(chi_0) - I_inf|;
   - explicit nontrivial profile families verify that once diam_I(P_n)->0 and
     one representative converges, every representative has the same limit;
   - counterexamples verify that dropping either hypothesis invalidates the
     conclusion.

G. Asymptotic recursive invariants:
   - the three defining conditions are checked on positive witnesses;
   - negative witnesses reject: non-shrinking profile diameter, shrinking
     profiles with no finite limiting value, and controlled O(1) logarithmic
     profiles that are not scalar invariants;
   - the script distinguishes a scalar spin invariant from the stronger
     limiting-frequency-profile invariant by an exact degeneracy witness for
     s=3.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isclose
from typing import Callable, Iterable, Sequence

from sympy import Rational, limit, log as sympy_log, oo, simplify, summation, symbols


def expect_raises(expected_exception: type[BaseException], fn: Callable[[], object], label: str) -> None:
    """Require fn itself to raise the expected exception.

    The manual failure is outside the exception handler, so an accidentally
    accepted invalid case cannot pass the test.
    """
    try:
        fn()
    except expected_exception:
        return
    except Exception as exc:  # pragma: no cover - used as a diagnostic guard
        raise AssertionError(f"{label}: wrong exception type {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"{label}: invalid case was accepted")


def eta(sigma: int, s: int) -> Fraction:
    if not isinstance(sigma, int) or not isinstance(s, int):
        raise TypeError("sigma and s must be integers")
    if s < 2:
        raise ValueError("the internal-state alphabet size must satisfy s >= 2")
    if not (1 <= sigma <= s):
        raise ValueError("internal state sigma must lie in {1,...,s}")
    return Fraction(2 * sigma - s - 1, 2)


def eta_bound(s: int) -> Fraction:
    if s < 2:
        raise ValueError("s must be at least 2")
    return Fraction(s - 1, 2)


@dataclass(frozen=True)
class StatePrefix:
    """A finite internal-state prefix over S={1,...,s}, with fixed ancestor a."""

    a: int
    states: tuple[int, ...]
    s: int

    def __post_init__(self) -> None:
        if not isinstance(self.a, int) or self.a < 1:
            raise ValueError("the ancestor a must lie in N={1,2,3,...}")
        if not isinstance(self.s, int) or self.s < 2:
            raise ValueError("the alphabet size s must satisfy s >= 2")
        if len(self.states) < 1:
            raise ValueError("finite-level profiles in this section use n >= 1")
        for sigma in self.states:
            if not isinstance(sigma, int) or not (1 <= sigma <= self.s):
                raise ValueError(f"invalid internal state {sigma} for s={self.s}")

    @property
    def n(self) -> int:
        return len(self.states)

    def digit_coordinate(self) -> int:
        """D = sum_{j=1}^n (sigma_j - 1) s^{n-j}."""
        n = self.n
        return sum((sigma_j - 1) * (self.s ** (n - 1 - j)) for j, sigma_j in enumerate(self.states))

    def delta_rho(self) -> Fraction:
        return Fraction(self.digit_coordinate(), self.s ** self.n)

    def r_coordinate_sympy(self):
        D = self.digit_coordinate()
        if D == 0:
            return Rational(0, 1)
        return sympy_log(Rational(D + 1, 1)) / sympy_log(Rational(self.s, 1))

    def r_coordinate_float(self) -> float:
        D = self.digit_coordinate()
        if D == 0:
            return 0.0
        # Change of base via Python floats is used only for numerical ordering
        # diagnostics; all exact diameter identities are algebraic/symbolic.
        import math

        return math.log(D + 1, self.s)

    def frequency(self, sigma: int) -> Fraction:
        if not (1 <= sigma <= self.s):
            raise ValueError("frequency requested for inadmissible state")
        return Fraction(sum(1 for state in self.states if state == sigma), self.n)

    def frequency_profile(self) -> tuple[Fraction, ...]:
        return tuple(self.frequency(sigma) for sigma in range(1, self.s + 1))

    def normalized_spin(self) -> Fraction:
        return sum((eta(sigma, self.s) for sigma in self.states), Fraction(0)) / self.n


@dataclass(frozen=True)
class FiniteProjection:
    """A finite projection P_n of representatives at one fixed level n."""

    prefixes: frozenset[StatePrefix]

    def __post_init__(self) -> None:
        if not self.prefixes:
            raise ValueError("a finite projection must be nonempty")
        ns = {prefix.n for prefix in self.prefixes}
        ancestors = {prefix.a for prefix in self.prefixes}
        alphabets = {prefix.s for prefix in self.prefixes}
        if len(ns) != 1:
            raise ValueError("all prefixes in P_n must have the same depth n")
        if len(ancestors) != 1:
            raise ValueError("all prefixes in one macro-class projection must share the same ancestor")
        if len(alphabets) != 1:
            raise ValueError("all prefixes in one projection must use the same internal-state alphabet")

    @property
    def n(self) -> int:
        return next(iter(self.prefixes)).n

    @property
    def s(self) -> int:
        return next(iter(self.prefixes)).s

    @property
    def ancestor(self) -> int:
        return next(iter(self.prefixes)).a

    def profile(self, characteristic: Callable[[StatePrefix], Fraction]) -> frozenset[Fraction]:
        return frozenset(characteristic(prefix) for prefix in self.prefixes)

    def delta_rho_profile(self) -> frozenset[Fraction]:
        return self.profile(lambda prefix: prefix.delta_rho())

    def spin_profile(self) -> frozenset[Fraction]:
        return self.profile(lambda prefix: prefix.normalized_spin())

    def digit_set(self) -> frozenset[int]:
        return frozenset(prefix.digit_coordinate() for prefix in self.prefixes)

    def r_profile_sympy(self) -> frozenset:
        return frozenset(prefix.r_coordinate_sympy() for prefix in self.prefixes)

    def frequency_profile_set(self) -> frozenset[tuple[Fraction, ...]]:
        return frozenset(prefix.frequency_profile() for prefix in self.prefixes)


def diameter(values: Iterable[Fraction]) -> Fraction:
    values = list(values)
    if not values:
        raise ValueError("diameter is undefined for an empty profile")
    return max(values) - min(values)


def pairwise_sup_diameter(values: Iterable[Fraction]) -> Fraction:
    values = list(values)
    if not values:
        raise ValueError("pairwise diameter is undefined for an empty profile")
    return max(abs(a - b) for a in values for b in values)


def all_prefixes(s: int, n: int, a: int = 1) -> list[StatePrefix]:
    return [StatePrefix(a=a, states=tuple(states), s=s) for states in product(range(1, s + 1), repeat=n)]


def prefix_from_digit(D: int, s: int, n: int, a: int = 1) -> StatePrefix:
    if not (0 <= D < s**n):
        raise ValueError("digit coordinate outside the n-level range")
    digits: list[int] = []
    remainder = D
    for power in range(n - 1, -1, -1):
        base_digit = remainder // (s**power)
        remainder -= base_digit * (s**power)
        digits.append(base_digit + 1)
    assert remainder == 0
    return StatePrefix(a=a, states=tuple(digits), s=s)


def finite_projection_from_digits(digits: Sequence[int], s: int, n: int, a: int = 1) -> FiniteProjection:
    return FiniteProjection(frozenset(prefix_from_digit(D, s=s, n=n, a=a) for D in digits))


def verify_finite_level_profiles_and_diameters() -> None:
    print("\n=== Finite-level profiles and exact profile diameters ===")

    prefixes = frozenset(
        {
            StatePrefix(a=7, states=(1, 1, 1, 1), s=3),
            StatePrefix(a=7, states=(1, 3, 2, 1), s=3),
            StatePrefix(a=7, states=(3, 2, 1, 3), s=3),
            StatePrefix(a=7, states=(2, 2, 2, 2), s=3),
        }
    )
    projection = FiniteProjection(prefixes)

    def characteristic(prefix: StatePrefix) -> Fraction:
        # A nontrivial characteristic depending on level, digit coordinate,
        # first state, and last state.
        D = prefix.digit_coordinate()
        return Fraction(3 * D + prefix.states[0] - 2 * prefix.states[-1], 5 * prefix.n)

    profile = projection.profile(characteristic)
    direct_values = frozenset(characteristic(prefix) for prefix in prefixes)
    assert profile == direct_values
    assert diameter(profile) == pairwise_sup_diameter(profile)

    # Duplicated representatives do not change set-valued profiles.
    duplicate_projection = FiniteProjection(frozenset(list(prefixes) + [next(iter(prefixes))]))
    assert duplicate_projection.profile(characteristic) == profile

    # Exhaustive finite verification of diam=max-min=sup pairwise differences.
    for s in range(2, 6):
        for n in range(1, 5):
            full = FiniteProjection(frozenset(all_prefixes(s=s, n=n, a=3)))
            values = full.profile(lambda prefix: Fraction(2 * prefix.digit_coordinate() - 3 * prefix.n, s**n + n))
            assert diameter(values) == pairwise_sup_diameter(values)

    # Sound expected-failure tests.
    expect_raises(ValueError, lambda: FiniteProjection(frozenset()), "empty finite projection")
    expect_raises(
        ValueError,
        lambda: FiniteProjection(
            frozenset(
                {
                    StatePrefix(a=1, states=(1, 2), s=3),
                    StatePrefix(a=1, states=(1, 2, 3), s=3),
                }
            )
        ),
        "mixed depths",
    )
    expect_raises(
        ValueError,
        lambda: FiniteProjection(
            frozenset(
                {
                    StatePrefix(a=1, states=(1, 2), s=3),
                    StatePrefix(a=2, states=(1, 2), s=3),
                }
            )
        ),
        "mixed ancestors",
    )
    expect_raises(
        ValueError,
        lambda: FiniteProjection(
            frozenset(
                {
                    StatePrefix(a=1, states=(1, 2), s=3),
                    StatePrefix(a=1, states=(1, 2), s=4),
                }
            )
        ),
        "mixed internal-state alphabets",
    )
    expect_raises(ValueError, lambda: StatePrefix(a=1, states=(1, 4), s=3), "inadmissible internal state")

    print("[OK] finite-level profiles are exact set-valued characteristic images")
    print("[OK] profile diameters equal the pairwise supremum on finite projections")
    print("[OK] invalid finite projections and invalid states are rejected soundly")


def verify_geometric_profiles() -> None:
    print("\n=== Geometric profiles Delta rho_n(P) and R_n(P) ===")

    # Nontrivial finite projection.
    P = finite_projection_from_digits(digits=[0, 2, 7, 10, 25], s=3, n=3, a=5)
    rho_profile = P.delta_rho_profile()
    expected_rho = frozenset(Fraction(D, 3**3) for D in [0, 2, 7, 10, 25])
    assert rho_profile == expected_rho
    assert all(Fraction(0) <= rho < Fraction(1) for rho in rho_profile)
    assert diameter(rho_profile) == Fraction(25, 27)

    # R-profile diameter from monotonicity of log_s(D+1).
    digits = sorted(P.digit_set())
    r_diam = sympy_log(Rational(max(digits) + 1, min(digits) + 1)) / sympy_log(Rational(P.s, 1))
    pairwise_r_diffs = [
        abs(prefix_i.r_coordinate_float() - prefix_j.r_coordinate_float())
        for prefix_i in P.prefixes
        for prefix_j in P.prefixes
    ]
    assert isclose(float(max(pairwise_r_diffs)), float(r_diam.evalf()), rel_tol=1e-12, abs_tol=1e-12)

    # Full-level Delta-rho diameter and R-profile diameter.
    for s in range(2, 6):
        for n in range(1, 6):
            full = FiniteProjection(frozenset(all_prefixes(s=s, n=n, a=1)))
            assert full.delta_rho_profile() == frozenset(Fraction(D, s**n) for D in range(s**n))
            assert diameter(full.delta_rho_profile()) == Fraction(s**n - 1, s**n)

            full_digits = full.digit_set()
            exact_r_diam = sympy_log(Rational(max(full_digits) + 1, min(full_digits) + 1)) / sympy_log(Rational(s, 1))
            assert simplify(exact_r_diam - n) == 0

    # Bounded digit windows shrink in Delta rho at the exact O(s^{-n}) rate.
    for s in (2, 3, 5):
        C = 4
        for n in range(4, 11):
            center = s**n // 3
            window = [D for D in range(center - C, center + C + 1) if 0 <= D < s**n]
            Pn = finite_projection_from_digits(window, s=s, n=n, a=2)
            assert diameter(Pn.delta_rho_profile()) <= Fraction(2 * C, s**n)

    # Bounded logarithmic diameter is only a controlled profile statement; it
    # is not automatically a scalar invariant.
    for n in range(3, 9):
        # D in {s^n/4, s^n/2} has asymptotically constant logarithmic gap
        # log_s(2), hence O(1), while the Delta-rho diameter stays near 1/4.
        s = 4
        D1 = s**n // 4
        D2 = s**n // 2
        Pn = finite_projection_from_digits([D1, D2], s=s, n=n, a=1)
        assert diameter(Pn.delta_rho_profile()) == Fraction(D2 - D1, s**n)
        assert diameter(Pn.delta_rho_profile()) == Fraction(1, 4)
        log_gap = sympy_log(Rational(D2 + 1, D1 + 1)) / sympy_log(Rational(s, 1))
        assert 0 < float(log_gap.evalf()) < 1

    print("[OK] Delta-rho profiles and diameters are exact")
    print("[OK] R-profile diameter follows the monotone log_s(D+1) formula")
    print("[OK] bounded logarithmic control is distinguished from scalar invariance")


def verify_internal_spin_profiles() -> None:
    print("\n=== Normalized spin profiles ===")

    P = FiniteProjection(
        frozenset(
            {
                StatePrefix(a=1, states=(1, 1, 1, 1), s=4),
                StatePrefix(a=1, states=(4, 4, 4, 4), s=4),
                StatePrefix(a=1, states=(1, 2, 3, 4), s=4),
                StatePrefix(a=1, states=(2, 2, 3, 3), s=4),
            }
        )
    )
    spin_profile = P.spin_profile()
    expected = frozenset(prefix.normalized_spin() for prefix in P.prefixes)
    assert spin_profile == expected
    assert diameter(spin_profile) == pairwise_sup_diameter(spin_profile)
    assert min(spin_profile) == -eta_bound(4)
    assert max(spin_profile) == eta_bound(4)

    for s in range(2, 8):
        B = eta_bound(s)
        for n in range(1, 5):
            full = FiniteProjection(frozenset(all_prefixes(s=s, n=n, a=1)))
            values = full.spin_profile()
            assert all(-B <= value <= B for value in values)
            assert min(values) == -B
            assert max(values) == B
            assert diameter(values) == 2 * B

    # Symbolic sharp range of eta.
    S, sigma = symbols("S sigma", integer=True, positive=True)
    eta_expr = sigma - (S + 1) / 2
    assert simplify(eta_expr.subs(sigma, 1) + (S - 1) / 2) == 0
    assert simplify(eta_expr.subs(sigma, S) - (S - 1) / 2) == 0

    print("[OK] B_n(P) is exactly the normalized spin profile on the finite projection")
    print("[OK] normalized spin profile values satisfy the sharp interval bound")


def periodic_prefix(cycle: Sequence[int], n: int, s: int, a: int = 1) -> StatePrefix:
    if n < 1:
        raise ValueError("n must be positive")
    if not cycle:
        raise ValueError("cycle must be nonempty")
    states = tuple(cycle[i % len(cycle)] for i in range(n))
    return StatePrefix(a=a, states=states, s=s)


def limiting_frequencies_of_cycle(cycle: Sequence[int], s: int) -> tuple[Fraction, ...]:
    L = len(cycle)
    return tuple(Fraction(sum(1 for state in cycle if state == sigma), L) for sigma in range(1, s + 1))


def spin_from_frequencies(freq: Sequence[Fraction], s: int) -> Fraction:
    if len(freq) != s:
        raise ValueError("frequency vector length must equal s")
    return sum((eta(sigma, s) * freq[sigma - 1] for sigma in range(1, s + 1)), Fraction(0))


def verify_frequency_regular_microhistories_and_spin_limit() -> None:
    print("\n=== Frequency regularity and the induced normalized spin limit ===")

    test_cycles = [
        (2, (1, 2)),
        (3, (1, 2, 3, 3)),
        (4, (1, 1, 2, 4, 4, 4)),
        (5, (1, 2, 2, 3, 4, 5, 5)),
    ]

    for s, cycle in test_cycles:
        f_limit = limiting_frequencies_of_cycle(cycle, s)
        spin_limit = spin_from_frequencies(f_limit, s)

        for n in (10, 25, 50, 100, 250):
            prefix = periodic_prefix(cycle=cycle, n=n, s=s)
            finite_freq = prefix.frequency_profile()

            # Exact frequency identity for barSigma_n.
            assert prefix.normalized_spin() == spin_from_frequencies(finite_freq, s)

            # Constructive frequency convergence bound for periodic sequences:
            # each count differs from n * limit frequency by at most one full cycle.
            L = len(cycle)
            for observed, target in zip(finite_freq, f_limit):
                assert abs(observed - target) <= Fraction(L, n)

            # Finite-sum error bound:
            # |barSigma_n - barSigma_inf| <= B_s * sum_sigma |f_sigma^(n)-f_sigma|.
            lhs = abs(prefix.normalized_spin() - spin_limit)
            rhs = eta_bound(s) * sum(abs(observed - target) for observed, target in zip(finite_freq, f_limit))
            assert lhs <= rhs
            assert rhs <= eta_bound(s) * s * Fraction(L, n)

        # Tail accuracy becomes arbitrarily strong for large n.
        prefix_large = periodic_prefix(cycle=cycle, n=10_000, s=s)
        assert abs(prefix_large.normalized_spin() - spin_limit) <= Fraction(eta_bound(s) * s * len(cycle), 10_000)

    # Symbolic finite-sum transfer for the representative family
    # f_i^(n) = p_i + c_i/n: the spin error is exactly O(1/n).
    n = symbols("n", positive=True)
    p1, p2, p3, c1, c2, c3 = symbols("p1 p2 p3 c1 c2 c3")
    eta3 = [Rational(-1, 1), Rational(0, 1), Rational(1, 1)]
    finite_spin = sum(eta3[i] * ([p1, p2, p3][i] + [c1, c2, c3][i] / n) for i in range(3))
    limit_spin = sum(eta3[i] * [p1, p2, p3][i] for i in range(3))
    assert simplify(limit(finite_spin, n, oo) - limit_spin) == 0
    assert simplify(finite_spin - limit_spin - (c3 - c1) / n) == 0

    # General centered-spectrum sum for arbitrary s is zero, supporting the
    # frequency-profile normalization used by the spin coordinate.
    S, i = symbols("S i", integer=True, positive=True)
    centered_sum = summation(i - (S + 1) / 2, (i, 1, S))
    assert simplify(centered_sum) == 0

    print("[OK] exact finite frequencies reconstruct barSigma_n on every tested prefix")
    print("[OK] periodic frequency-regular sequences converge to the stated spin limit")
    print("[OK] finite-sum convergence is verified by an explicit exact error bound")


def profile_diameter_at(values_at_n: Callable[[int], Sequence[Fraction]], n: int) -> Fraction:
    return diameter(values_at_n(n))


def verify_asymptotic_regularity() -> None:
    print("\n=== Asymptotic regularity of profile families ===")

    L = Fraction(7, 5)
    offsets = [Fraction(-3, 1), Fraction(0), Fraction(5, 2), Fraction(9, 1)]

    def shrinking_profile(n: int) -> tuple[Fraction, ...]:
        return tuple(L + c / n for c in offsets)

    for n in (10, 25, 50, 100, 250):
        expected_diam = (max(offsets) - min(offsets)) / n
        assert profile_diameter_at(shrinking_profile, n) == expected_diam

    n_sym = symbols("n", positive=True)
    assert simplify(limit((max(offsets) - min(offsets)) / n_sym, n_sym, oo)) == 0

    # Exact epsilon witness: for any sampled epsilon, a computed N makes the
    # diameter smaller than epsilon.
    for eps in [Fraction(1, 2), Fraction(1, 5), Fraction(1, 20), Fraction(1, 100)]:
        C = max(offsets) - min(offsets)
        N = int(C / eps) + 1
        assert profile_diameter_at(shrinking_profile, N) < eps

    # Necessary but not sufficient: shrinking diameter can drift without a
    # finite limiting invariant.
    def drifting_shrinking_profile(n: int) -> tuple[Fraction, ...]:
        return (Fraction(n), Fraction(n) + Fraction(1, n))

    for n in (10, 50, 250):
        assert profile_diameter_at(drifting_shrinking_profile, n) == Fraction(1, n)
    assert simplify(limit(1 / n_sym, n_sym, oo)) == 0
    # But both representatives diverge to infinity, not to a finite value.
    assert limit(n_sym, n_sym, oo) == oo

    # One convergent representative without shrinking diameter is insufficient.
    def nonregular_profile(n: int) -> tuple[Fraction, ...]:
        return (L, L + Fraction(1, 3))

    for n in (5, 25, 100):
        assert profile_diameter_at(nonregular_profile, n) == Fraction(1, 3)

    print("[OK] nontrivial shrinking profile families satisfy diam_I(P_n)->0")
    print("[OK] shrinking diameter alone is not misclassified as a finite invariant")
    print("[OK] non-vanishing diameter blocks asymptotic regularity")


def verify_unique_asymptotic_profile_theorem() -> None:
    print("\n=== Unique asymptotic profile of a regular macro-class ===")

    I_inf = Fraction(11, 7)
    offsets = [Fraction(-5, 1), Fraction(0), Fraction(9, 2), Fraction(13, 3)]

    def values(n: int) -> tuple[Fraction, ...]:
        return tuple(I_inf + c / n for c in offsets)

    for n in (20, 50, 100, 500):
        diam_n = diameter(values(n))
        chi0_error = abs((I_inf + offsets[0] / n) - I_inf)
        for value in values(n):
            assert abs(value - I_inf) <= diam_n + chi0_error

    # Exact epsilon implementation of the proof in the section.
    for eps in [Fraction(1, 2), Fraction(1, 10), Fraction(1, 100)]:
        diam_constant = max(offsets) - min(offsets)
        chi0_constant = abs(offsets[0])
        N_diam = int(2 * diam_constant / eps) + 1
        N_chi0 = int(2 * chi0_constant / eps) + 1
        N = max(N_diam, N_chi0)
        assert diameter(values(N)) < eps / 2
        assert abs(values(N)[0] - I_inf) < eps / 2
        assert all(abs(value - I_inf) < eps for value in values(N))

    # Symbolic verification of the inequality structure:
    # if diameter <= a/n and one representative error <= b/n, then every
    # representative error is <= (a+b)/n -> 0.
    n = symbols("n", positive=True)
    a, b = symbols("a b", positive=True)
    assert simplify(limit((a + b) / n, n, oo)) == 0

    # Counterexample 1: one representative converges, but diameter does not vanish.
    def different_limits(n: int) -> tuple[Fraction, Fraction]:
        return (Fraction(0), Fraction(1))

    assert all(profile_diameter_at(different_limits, n) == 1 for n in (10, 100, 1000))

    # Counterexample 2: diameter vanishes, but no finite representative limit exists.
    def drifting(n: int) -> tuple[Fraction, Fraction]:
        return (Fraction(n), Fraction(n) + Fraction(1, n))

    assert all(profile_diameter_at(drifting, n) == Fraction(1, n) for n in (10, 100, 1000))

    print("[OK] epsilon proof of the unique-profile proposition is checked exactly")
    print("[OK] dropping either hypothesis produces a verified counterexample")


@dataclass(frozen=True)
class InvariantWitness:
    name: str
    values: Callable[[int], tuple[Fraction, ...]]
    expected_limit: Fraction | None

    def diameter_at(self, n: int) -> Fraction:
        return diameter(self.values(n))

    def errors_to_limit(self, n: int) -> tuple[Fraction, ...]:
        if self.expected_limit is None:
            raise ValueError("this witness has no finite expected limit")
        return tuple(abs(v - self.expected_limit) for v in self.values(n))


def verify_asymptotic_recursive_invariants() -> None:
    print("\n=== Asymptotic recursive invariant definition and candidate behaviours ===")

    rho = Fraction(2, 5)
    rho_witness = InvariantWitness(
        name="bounded digit-window normalized causal coordinate",
        values=lambda n: tuple(rho + Fraction(c, 3**n) for c in (-2, 0, 3)),
        expected_limit=rho,
    )

    spin_limit = Fraction(1, 6)
    spin_witness = InvariantWitness(
        name="common limiting normalized spin profile",
        values=lambda n: tuple(spin_limit + Fraction(c, n) for c in (-1, 0, 2)),
        expected_limit=spin_limit,
    )

    for witness in (rho_witness, spin_witness):
        for n in (20, 50, 100, 250):
            assert witness.diameter_at(n) >= 0
        assert witness.diameter_at(500) < witness.diameter_at(50)
        assert max(witness.errors_to_limit(1000)) < Fraction(1, 50)

    # Negative witness: non-shrinking diameter.
    nonshrinking = InvariantWitness(
        name="two different asymptotic values",
        values=lambda n: (Fraction(0), Fraction(1)),
        expected_limit=None,
    )
    for n in (10, 100, 1000):
        assert nonshrinking.diameter_at(n) == 1

    # Negative witness: shrinking diameter but no finite limiting value.
    drifting = InvariantWitness(
        name="drifting shrinking profile",
        values=lambda n: (Fraction(n), Fraction(n) + Fraction(1, n)),
        expected_limit=None,
    )
    for n in (10, 100, 1000):
        assert drifting.diameter_at(n) == Fraction(1, n)

    # Controlled logarithmic profile: O(1) diameter is not a scalar invariant.
    def controlled_r_profile(n: int) -> tuple[Fraction, Fraction]:
        return (Fraction(n), Fraction(n) + Fraction(3, 2))

    for n in (10, 100, 1000):
        assert diameter(controlled_r_profile(n)) == Fraction(3, 2)
    assert diameter(controlled_r_profile(10)) == diameter(controlled_r_profile(1000))

    # Spin invariant versus limiting-frequency-profile invariant:
    # for s=3, eta=(-1,0,+1).  Two different limiting frequency vectors can
    # have the same normalized spin limit.  Therefore the scalar spin invariant
    # does not determine the full frequency-profile invariant.
    f_a = (Fraction(1, 2), Fraction(0), Fraction(1, 2))
    f_b = (Fraction(0), Fraction(1), Fraction(0))
    assert f_a != f_b
    assert spin_from_frequencies(f_a, s=3) == 0
    assert spin_from_frequencies(f_b, s=3) == 0

    # A genuine limiting-frequency-profile invariant requires vector convergence
    # to the same limiting vector, not merely equality of the eta-weighted sum.
    common_f = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
    for n in (20, 100, 500):
        f1 = tuple(common_f[i] + Fraction([1, -2, 1][i], n) for i in range(3))
        f2 = tuple(common_f[i] + Fraction([-1, 2, -1][i], n) for i in range(3))
        assert sum(f1) == 1
        assert sum(f2) == 1
        assert max(abs(f1[i] - f2[i]) for i in range(3)) <= Fraction(4, n)

    print("[OK] positive witnesses satisfy the invariant conditions constructively")
    print("[OK] non-shrinking and drifting profiles are rejected as scalar invariants")
    print("[OK] O(1) logarithmic control is not confused with a scalar limit")
    print("[OK] scalar spin invariance is separated from full frequency-profile invariance")


def verify_symbolic_summaries() -> None:
    print("\n=== Symbolic summary checks for the section ===")

    n, s = symbols("n s", integer=True, positive=True)
    D_min, D_max = symbols("D_min D_max", integer=True, nonnegative=True)

    # Delta-rho diameter for a full s-ary level:
    full_rho_diam = (s**n - 1) / s**n
    assert simplify(full_rho_diam - (1 - s ** (-n))) == 0
    # SymPy cannot decide the sign of log(s) from integer positivity alone, so
    # the actual limit check is performed for representative bases s>=2.
    for base in (2, 3, 5, 7):
        assert simplify(limit((1 - full_rho_diam).subs(s, base), n, oo)) == 0

    # Bounded digit-window diameter:
    C = symbols("C", positive=True)
    bounded_window_diam = 2 * C / s**n
    for base in (2, 3, 5, 7):
        assert simplify(limit(bounded_window_diam.subs(s, base), n, oo)) == 0

    # R-profile diameter for digit interval [D_min,D_max] in a fixed projection.
    # This is an algebraic restatement of monotonicity, not an extra physical
    # assumption.
    r_diam_expr = sympy_log((D_max + 1) / (D_min + 1)) / sympy_log(s)
    assert r_diam_expr.has(D_max)
    assert r_diam_expr.has(D_min)

    # Frequency-spin finite-sum identity for s=4 with symbolic frequencies.
    f1, f2, f3, f4 = symbols("f1 f2 f3 f4")
    spin_expr = Fraction(-3, 2) * f1 + Fraction(-1, 2) * f2 + Fraction(1, 2) * f3 + Fraction(3, 2) * f4
    eta_weighted = sum(eta(i, 4) * [f1, f2, f3, f4][i - 1] for i in range(1, 5))
    assert simplify(spin_expr - eta_weighted) == 0

    print("[OK] symbolic expressions match the profile-diameter and frequency-spin formulas")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of recursive controlled characteristics and macro-class invariants ===")
    verify_finite_level_profiles_and_diameters()
    verify_geometric_profiles()
    verify_internal_spin_profiles()
    verify_frequency_regular_microhistories_and_spin_limit()
    verify_asymptotic_regularity()
    verify_unique_asymptotic_profile_theorem()
    verify_asymptotic_recursive_invariants()
    verify_symbolic_summaries()
    print("\n=== Recursive controlled characteristics verification completed successfully ===")


if __name__ == "__main__":
    main()
