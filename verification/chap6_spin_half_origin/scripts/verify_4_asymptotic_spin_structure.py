"""
VERIFICATION of Section: Asymptotic structure of the cumulative spin coordinate
(sec:asymptotic-spin-structure).

This script provides a full mathematical verification block for the claims in
4_asymptotic_spin_structure.tex.  It verifies only the new asymptotic layer of
the binary cumulative-spin construction and uses the exact finite spectrum of
the cumulative coordinate as an already established dependency.

No physical spin representation is added here: the checks concern the internal
binary-state sequence, its cumulative centered coordinate, the normalized
coordinate, the asymptotic densification of its finite spectra, and the
central combinatorial multiplicity.

Verified content
----------------
1. Normalized cumulative coordinate:
      barSigma_n = Sigma_n / n
   is meaningful only for n >= 1.  The script explicitly rejects n = 0.

2. Exact normalized spectrum:
      Spec_n(barSigma) = {-1/2 + q/n : q = 0,...,n}.
   It verifies endpoints, cardinality, strict spacing 1/n, and exact agreement
   with exhaustive finite enumeration for small n.

3. Densification:
   for every rational open interval (a,b) inside [-1/2,1/2], the script
   constructs an explicit n and an explicit spectral point inside (a,b).
   It also checks nearest-grid approximation bounds.

4. Balanced trajectories:
   Sigma_n = 0 exists exactly for even n, and for a fixed initial element the
   number of balanced finite internal-state prefixes is C(n,n/2).

5. Central-layer maximality:
   for even n, the central value Sigma_n = 0 has the unique maximal
   multiplicity C(n,n/2).  For odd n, the script verifies the expected two
   adjacent maximal layers and rejects treating Sigma_n = 0 as available.

6. Central-binomial asymptotic:
      C(n,n/2) ~ 2^n / sqrt(pi n/2),  n even.
   Written as n = 2q, this is
     C(2q,q) ~ 4^q / sqrt(pi q).
   The script verifies finite values, checks the Wallis-type bounding regime,
   and symbolically verifies that the bounding interval shrinks to 1.

7. Corollary verification:
   the final corollary is checked as the conjunction of densification and
   central maximality, without adding a new independent assumption.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb, exp, lgamma, log, pi, sqrt
from typing import Callable, Iterable

from sympy import Rational, limit, simplify, sqrt as sympy_sqrt, symbols, oo


BINARY_STATES: tuple[int, int] = (1, 2)


def assert_raises(expected_exception: type[BaseException], fn: Callable, *args, **kwargs) -> None:
    """Require fn(*args, **kwargs) to raise the expected exception type."""
    try:
        fn(*args, **kwargs)
    except expected_exception:
        return
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            f"Expected {expected_exception.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {expected_exception.__name__}, but no exception was raised")


def require_binary_state(sigma: int) -> None:
    if sigma not in BINARY_STATES:
        raise ValueError(f"invalid internal state {sigma}; expected one of {BINARY_STATES}")


def require_prefix(prefix: tuple[int, ...], expected_length: int | None = None) -> None:
    if expected_length is not None and len(prefix) != expected_length:
        raise ValueError(f"prefix length {len(prefix)} does not match expected {expected_length}")
    for sigma in prefix:
        require_binary_state(sigma)


def require_positive_depth(n: int) -> None:
    if not isinstance(n, int) or n < 1:
        raise ValueError("normalized spin coordinate requires integer depth n >= 1")


def require_nonnegative_depth(n: int) -> None:
    if not isinstance(n, int) or n < 0:
        raise ValueError("depth must be a nonnegative integer")


def centered_binary_value(sigma: int) -> Fraction:
    require_binary_state(sigma)
    return Fraction(-1, 2) if sigma == 1 else Fraction(1, 2)


def cumulative_spin(prefix: tuple[int, ...]) -> Fraction:
    require_prefix(prefix)
    return sum((centered_binary_value(sigma) for sigma in prefix), Fraction(0))


def normalized_spin(prefix: tuple[int, ...]) -> Fraction:
    require_prefix(prefix)
    n = len(prefix)
    require_positive_depth(n)
    return cumulative_spin(prefix) / n


def all_prefixes(n: int) -> Iterable[tuple[int, ...]]:
    require_nonnegative_depth(n)
    return product(BINARY_STATES, repeat=n)


def sigma_from_plus_count(n: int, k_plus: int) -> Fraction:
    require_nonnegative_depth(n)
    if not isinstance(k_plus, int) or not 0 <= k_plus <= n:
        raise ValueError("plus-state count must be an integer in [0,n]")
    return Fraction(2 * k_plus - n, 2)


def normalized_from_plus_count(n: int, k_plus: int) -> Fraction:
    require_positive_depth(n)
    return sigma_from_plus_count(n, k_plus) / n


def cumulative_spectrum(n: int) -> list[Fraction]:
    require_nonnegative_depth(n)
    return [sigma_from_plus_count(n, k) for k in range(n + 1)]


def normalized_spectrum(n: int) -> list[Fraction]:
    require_positive_depth(n)
    return [normalized_from_plus_count(n, k) for k in range(n + 1)]


def multiplicity_for_plus_count(n: int, k_plus: int) -> int:
    require_nonnegative_depth(n)
    if not isinstance(k_plus, int) or not 0 <= k_plus <= n:
        raise ValueError("plus-state count must be an integer in [0,n]")
    return comb(n, k_plus)


def multiplicity_by_cumulative_value(n: int) -> dict[Fraction, int]:
    require_nonnegative_depth(n)
    return {sigma_from_plus_count(n, k): multiplicity_for_plus_count(n, k) for k in range(n + 1)}


def balanced_count_for_even_depth(n: int) -> int:
    require_nonnegative_depth(n)
    if n % 2 != 0:
        raise ValueError("balanced cumulative spin Sigma_n = 0 requires even depth")
    return comb(n, n // 2)


def floor_fraction(q: Fraction) -> int:
    return q.numerator // q.denominator


def nearest_integer_fraction(q: Fraction) -> int:
    return floor_fraction(q + Fraction(1, 2))


def require_interval_inside_spin_range(a: Fraction, b: Fraction) -> None:
    if not (Fraction(-1, 2) <= a < b <= Fraction(1, 2)):
        raise ValueError("expected a nonempty open interval inside [-1/2,1/2]")


def explicit_spectral_point_inside(a: Fraction, b: Fraction) -> tuple[int, Fraction]:
    """Construct n and a normalized spectral point strictly inside (a,b)."""
    require_interval_inside_spin_range(a, b)
    width = b - a
    n = floor_fraction(Fraction(1, 1) / width) + 2
    assert Fraction(1, n) < width
    shifted_left = a + Fraction(1, 2)
    k = floor_fraction(n * shifted_left) + 1
    point = Fraction(-1, 2) + Fraction(k, n)
    assert a < point < b
    assert point in set(normalized_spectrum(n))
    return n, point


@dataclass(frozen=True)
class CentralBinomialRatio:
    m: int

    @property
    def exact_count(self) -> int:
        return comb(2 * self.m, self.m)

    @property
    def asymptotic_main_term(self) -> float:
        return (4.0**self.m) / sqrt(pi * self.m)

    @property
    def ratio(self) -> float:
        # log form avoids overflow for large m.
        log_ratio = (
            lgamma(2 * self.m + 1)
            - 2 * lgamma(self.m + 1)
            + 0.5 * (log(pi) + log(self.m))
            - self.m * log(4)
        )
        return exp(log_ratio)

    @property
    def wallis_lower(self) -> float:
        return sqrt(self.m / (self.m + 0.5))


def verify_symbolic_normalized_coordinate() -> None:
    print("\n=== Symbolic verification of normalized cumulative coordinate ===")

    n, q = symbols("n q", positive=True, integer=True)

    Sigma_n = q - Rational(1, 2) * n
    normalized = Sigma_n / n
    expected = -Rational(1, 2) + q / n

    assert simplify(normalized - expected) == 0
    assert simplify(expected.subs(q, 0) + Rational(1, 2)) == 0
    assert simplify(expected.subs(q, n) - Rational(1, 2)) == 0
    assert simplify((expected.subs(q, q + 1) - expected) - Rational(1, 1) / n) == 0

    # Cumulative spectrum spacing is one before normalization.
    cumulative_expected = q - Rational(1, 2) * n
    assert simplify((cumulative_expected.subs(q, q + 1) - cumulative_expected) - 1) == 0

    print("[OK] barSigma_n = Sigma_n / n equals -1/2 + q/n for q=N_+")
    print("[OK] endpoints and exact normalized spacing 1/n hold symbolically")


def verify_exact_normalized_spectrum_by_enumeration() -> None:
    print("\n=== Exact finite enumeration of normalized spectra ===")

    total_checked = 0
    for n in range(1, 15):
        enumerated_cumulative = Counter(cumulative_spin(tuple(prefix)) for prefix in all_prefixes(n))
        enumerated_normalized = Counter(normalized_spin(tuple(prefix)) for prefix in all_prefixes(n))

        expected_cumulative = multiplicity_by_cumulative_value(n)
        expected_normalized = {
            normalized_from_plus_count(n, k): multiplicity_for_plus_count(n, k)
            for k in range(n + 1)
        }

        assert enumerated_cumulative == expected_cumulative
        assert enumerated_normalized == expected_normalized

        spectrum = normalized_spectrum(n)
        assert spectrum[0] == Fraction(-1, 2)
        assert spectrum[-1] == Fraction(1, 2)
        assert len(spectrum) == n + 1
        assert len(set(spectrum)) == n + 1
        assert all(spectrum[i + 1] - spectrum[i] == Fraction(1, n) for i in range(n))
        total_checked += 2**n

    assert_raises(ValueError, normalized_spin, tuple())
    assert cumulative_spectrum(0) == [Fraction(0)]

    print(f"[OK] Exhaustively checked {total_checked} finite internal-state prefixes for n=1,...,14")
    print("[OK] n=0 is correctly rejected for the normalized coordinate and accepted only for Sigma_0=0")


def verify_asymptotic_densification_constructively() -> None:
    print("\n=== Constructive verification of asymptotic spectral densification ===")

    endpoints = sorted(
        {
            Fraction(-1, 2),
            Fraction(1, 2),
            Fraction(-2, 5),
            Fraction(-1, 3),
            Fraction(-1, 7),
            Fraction(0),
            Fraction(1, 11),
            Fraction(1, 4),
            Fraction(2, 5),
        }
    )

    checked_intervals = 0
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1 :]:
            if a < b:
                n, point = explicit_spectral_point_inside(a, b)
                assert a < point < b
                assert point in set(normalized_spectrum(n))
                checked_intervals += 1

    # Nearest-grid approximation: every target in the interval is within
    # 1/(2n) of a spectral point at level n.
    targets = {
        Fraction(-1, 2),
        Fraction(-3, 8),
        Fraction(-1, 10),
        Fraction(0),
        Fraction(7, 31),
        Fraction(1, 3),
        Fraction(1, 2),
    }
    approximation_checks = 0
    for n in (5, 8, 13, 21, 34, 55, 89):
        spectrum = set(normalized_spectrum(n))
        for target in targets:
            shifted = target + Fraction(1, 2)
            k = min(max(nearest_integer_fraction(n * shifted), 0), n)
            point = Fraction(-1, 2) + Fraction(k, n)
            assert point in spectrum
            assert abs(point - target) <= Fraction(1, 2 * n)
            approximation_checks += 1

    assert checked_intervals > 0 and approximation_checks > 0
    print(f"[OK] Constructed spectral points inside {checked_intervals} rational open intervals")
    print(f"[OK] Checked {approximation_checks} nearest-grid approximations with error <= 1/(2n)")


def verify_balanced_trajectories() -> None:
    print("\n=== Exact verification of balanced trajectories ===")

    exact_formula_checks = 0
    enumeration_checks = 0

    for n in range(0, 31):
        has_zero = Fraction(0) in set(cumulative_spectrum(n))
        if n % 2 == 0:
            assert has_zero
            assert multiplicity_by_cumulative_value(n)[Fraction(0)] == comb(n, n // 2)
            assert balanced_count_for_even_depth(n) == comb(n, n // 2)
            exact_formula_checks += 1
        else:
            assert not has_zero
            assert_raises(ValueError, balanced_count_for_even_depth, n)

    for n in range(0, 15, 2):
        count = sum(1 for prefix in all_prefixes(n) if cumulative_spin(tuple(prefix)) == 0)
        assert count == comb(n, n // 2)
        enumeration_checks += 1

    initial_count = 7
    for n in range(0, 20, 2):
        assert initial_count * balanced_count_for_even_depth(n) == initial_count * comb(n, n // 2)

    print(f"[OK] Checked parity and balanced-count formulas for {exact_formula_checks} even depths")
    print(f"[OK] Exhaustively enumerated balanced prefixes for {enumeration_checks} even depths")
    print("[OK] Multiplicity scales linearly with the number of fixed initial elements when that factor is included")


def verify_central_layer_maximality() -> None:
    print("\n=== Verification of central-layer maximality ===")

    n, q = symbols("n q", positive=True, integer=True)

    # For adjacent binomial multiplicities, the standard quotient is
    # C(n,q+1)/C(n,q) = (n-q)/(q+1).  Its excess over one is exactly
    # (n-2q-1)/(q+1), which determines monotonicity around the center.
    neighbor_ratio = (n - q) / (q + 1)
    assert simplify((neighbor_ratio - 1) - (n - 2 * q - 1) / (q + 1)) == 0

    m = symbols("m", positive=True, integer=True)
    left_of_center_ratio_minus_one = simplify((2 * m - (m - 1)) / m - 1)
    right_of_center_ratio_minus_one = simplify((2 * m - m) / (m + 1) - 1)
    assert simplify(left_of_center_ratio_minus_one - 1 / m) == 0
    assert simplify(right_of_center_ratio_minus_one + 1 / (m + 1)) == 0

    even_checks = 0
    for n_value in range(2, 101, 2):
        counts = [comb(n_value, q_value) for q_value in range(n_value + 1)]
        maximum = max(counts)
        maximal_indices = [q_value for q_value, count in enumerate(counts) if count == maximum]
        assert maximal_indices == [n_value // 2]
        assert maximum == comb(n_value, n_value // 2)
        assert sigma_from_plus_count(n_value, n_value // 2) == 0

        for q_value in range(0, n_value // 2):
            assert counts[q_value] < counts[q_value + 1]
        for q_value in range(n_value // 2, n_value):
            assert counts[q_value] > counts[q_value + 1]
        even_checks += 1

    odd_checks = 0
    for n_value in range(1, 100, 2):
        counts = [comb(n_value, q_value) for q_value in range(n_value + 1)]
        maximum = max(counts)
        maximal_indices = [q_value for q_value, count in enumerate(counts) if count == maximum]
        assert maximal_indices == [n_value // 2, n_value // 2 + 1]
        assert Fraction(0) not in set(cumulative_spectrum(n_value))
        odd_checks += 1

    print("[OK] Symbolic binomial-neighbor ratio gives the center as the unique even-depth maximum")
    print(f"[OK] Checked central uniqueness for {even_checks} even depths")
    print(f"[OK] Checked the two adjacent maxima and absence of Sigma_n=0 for {odd_checks} odd depths")


def verify_central_binomial_asymptotic() -> None:
    print("\n=== Verification of central-binomial asymptotic regime ===")

    # For n=2m, the section's estimate becomes:
    #   C(2m,m) ~ 4^m / sqrt(pi*m).
    # We verify it using finite high-precision log-ratio checks and the
    # Wallis-type squeeze:
    #   sqrt(m/(m+1/2)) < C(2m,m) * sqrt(pi*m) / 4^m < 1.
    # The lower bound tends to 1, hence the ratio is squeezed to 1.

    m = symbols("m", positive=True)
    lower_bound = sympy_sqrt(m / (m + Rational(1, 2)))
    assert limit(lower_bound, m, oo) == 1

    checked = 0
    previous_error = None
    for m_value in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610):
        ratio_data = CentralBinomialRatio(m_value)
        ratio = ratio_data.ratio
        lower = ratio_data.wallis_lower

        assert lower < ratio < 1.0
        error = 1.0 - ratio
        assert error > 0
        if previous_error is not None:
            # Along this increasing sequence, the asymptotic error must shrink.
            assert error < previous_error
        previous_error = error
        checked += 1

    # Exact finite identities for small m connect the asymptotic expression to
    # the central multiplicity claimed in the section.
    for m_value in range(1, 25):
        n_value = 2 * m_value
        assert balanced_count_for_even_depth(n_value) == comb(2 * m_value, m_value)
        assert multiplicity_by_cumulative_value(n_value)[Fraction(0)] == comb(2 * m_value, m_value)

    print("[OK] Wallis-type lower bound tends symbolically to 1")
    print(f"[OK] Checked {checked} central-binomial ratios inside the squeeze interval")
    print("[OK] Central multiplicity and balanced multiplicity agree exactly for n=2m")


def verify_corollary_as_conjunction() -> None:
    print("\n=== Verification of the final corollary as a conjunction ===")

    # The corollary contains exactly two ingredients already established in the
    # section: spectral densification and even-depth central maximality.  This
    # block checks their simultaneous availability on representative levels and
    # does not add a new premise.

    intervals = [
        (Fraction(-1, 2), Fraction(-1, 3)),
        (Fraction(-1, 9), Fraction(1, 10)),
        (Fraction(3, 20), Fraction(1, 2)),
    ]

    for a, b in intervals:
        n_dense, point = explicit_spectral_point_inside(a, b)
        assert a < point < b
        assert point in set(normalized_spectrum(n_dense))

    for n_value in (2, 4, 8, 12, 20, 40):
        counts = multiplicity_by_cumulative_value(n_value)
        assert counts[Fraction(0)] == max(counts.values())
        assert list(value for value, count in counts.items() if count == max(counts.values())) == [Fraction(0)]

    print("[OK] Densification witnesses and even-depth central maximality coexist exactly as stated")


def verify_negative_domain_guards() -> None:
    print("\n=== Negative domain and consistency checks ===")

    assert_raises(ValueError, require_binary_state, 0)
    assert_raises(ValueError, require_binary_state, 3)
    assert_raises(ValueError, require_prefix, (1, 2, 3))
    assert_raises(ValueError, require_prefix, (1, 2), 3)
    assert_raises(ValueError, require_positive_depth, 0)
    assert_raises(ValueError, require_positive_depth, -2)
    assert_raises(ValueError, normalized_spectrum, 0)
    assert_raises(ValueError, normalized_spectrum, -1)
    assert_raises(ValueError, sigma_from_plus_count, 5, -1)
    assert_raises(ValueError, sigma_from_plus_count, 5, 6)
    assert_raises(ValueError, balanced_count_for_even_depth, 7)
    assert_raises(ValueError, require_interval_inside_spin_range, Fraction(-2, 3), Fraction(0))
    assert_raises(ValueError, require_interval_inside_spin_range, Fraction(0), Fraction(2, 3))
    assert_raises(ValueError, require_interval_inside_spin_range, Fraction(1, 3), Fraction(1, 3))

    print("[OK] Invalid states, invalid depths, invalid intervals and odd balanced requests are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of asymptotic spin structure (sec:asymptotic-spin-structure) ===")
    verify_symbolic_normalized_coordinate()
    verify_exact_normalized_spectrum_by_enumeration()
    verify_asymptotic_densification_constructively()
    verify_balanced_trajectories()
    verify_central_layer_maximality()
    verify_central_binomial_asymptotic()
    verify_corollary_as_conjunction()
    verify_negative_domain_guards()
    print("\n=== Asymptotic spin structure verification completed successfully ===")


if __name__ == "__main__":
    main()
