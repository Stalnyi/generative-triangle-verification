"""
VERIFICATION of Section:
Sector causal-geometric projection
(sec:sector-causal-geometric-projection).

Source file:
    3_sector_causal_geometric_projection.tex

This script verifies the mathematical content of the section as a standalone
sector-projection layer over the previously verified normalized causal
coordinate, finite causal spectra, and full combinatorial perturbation measure.

The section proves that the primary causal sectors

    I_sigma = [(sigma-1)/s, sigma/s),    sigma=1,...,s,

partition the normalized causal interval [0,1), that the first digit of a
causal microhistory determines the primary sector, that each primary sector
contains exactly a 1/s fraction of the finite spectrum at every depth
Delta n>=1, and that the unique equal primary-sector projection of the full
measure

    G_comb = ln s

is

    G_s = (ln s)/s.

Verified content
----------------
1. Primary causal sectors:
       I_sigma=[(sigma-1)/s, sigma/s), sigma=1,...,s.
   They form a disjoint half-open partition of [0,1).

2. Finite causal spectrum at depth d=Delta n:
       Sigma_d = {q/s^d : q=0,...,s^d-1}.
   It has exactly s^d points and lies in [0,1).

3. Digit coordinate:
       D(sigma_1,...,sigma_d)
       =
       sum_{j=1}^d (sigma_j-1)s^{d-j}.
   The normalized coordinate is Delta rho = D/s^d.

4. First-digit sector lemma:
       Delta rho in I_sigma  iff  sigma_1=sigma.
   The residual after the first digit satisfies
       0 <= R < 1/s
   exactly.

5. Finite primary-sector fraction:
       |Sigma_d cap I_sigma| = s^{d-1},
       |Sigma_d cap I_sigma| / |Sigma_d| = 1/s
   for every d>=1 and sigma=1,...,s in the checked windows.

6. Finite-continuum agreement:
       length(I_sigma)=1/s
   agrees exactly with the finite spectrum fraction 1/s.

7. Sector projection of the full measure:
       G_comb = ln s.
   A primary sector projection is a tuple (G_1,...,G_s).

8. Symmetry of primary sector contributions:
   in the equal primary-sector class, all contributions are the same:
       G_sigma=G_tau.

9. Unique primary-sector normalization:
   if
       sum_{sigma=1}^s G_sigma = G_comb
   and all G_sigma are equal, then
       G_sigma = G_comb/s = (ln s)/s.
   This coefficient is unique.

10. Reconstruction:
       sum_{sigma=1}^s (ln s)/s = ln s.

11. Binary case:
       s=2  ->  G_2=(ln 2)/2.

12. Normalization guards:
   - division by s^q reconstructs the full measure only when q=1 in the
     primary equal-sector class;
   - division by s-1 does not reconstruct the full measure for s>=2;
   - division by sqrt(s) does not reconstruct the full measure for s>=2 except
     not in the integer primary-sector count sense; numerically it would require
     s=1, outside the model;
   - division by s^alpha reconstructs the full measure only for alpha=1;
   - deeper q-level sub-sector projections have s^q equal pieces, not s primary
     pieces.

13. Negative guards:
   - s<2 is rejected;
   - depth d<1 is rejected for primary-sector causal paths;
   - invalid sigma is rejected;
   - invalid states are rejected;
   - invalid digit D is rejected;
   - rho outside [0,1) is rejected;
   - unequal sector contributions are rejected for the equal-primary-sector
     normalization theorem;
   - wrong denominators fail full reconstruction;
   - a same-number sub-sector normalization is not accepted as the primary
     sector constant.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isclose, log, sqrt
from typing import Callable, Sequence

from sympy import sqrt
from sympy import log as slog
from sympy import simplify, symbols


class DomainError(ValueError):
    """Raised when a mathematical domain condition is violated."""


def expect_raises(expected_exception: type[BaseException] | tuple[type[BaseException], ...],
                  fn: Callable[[], object]) -> None:
    try:
        fn()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception} was not raised")


def require_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def require_branching(value: object) -> int:
    value = require_int(value, "s")
    if value < 2:
        raise DomainError("s must satisfy s>=2")
    return value


def require_positive_depth(value: object) -> int:
    value = require_int(value, "depth")
    if value < 1:
        raise DomainError("primary-sector causal depth must satisfy depth>=1")
    return value


def require_nonnegative_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 0:
        raise DomainError(f"{name} must be nonnegative")
    return value


def validate_sigma(s: int, sigma: object) -> int:
    s = require_branching(s)
    sigma = require_int(sigma, "sigma")
    if not (1 <= sigma <= s):
        raise DomainError("sigma must belong to {1,...,s}")
    return sigma


def validate_state_sequence(s: int, states: Sequence[int]) -> tuple[int, ...]:
    s = require_branching(s)
    if not isinstance(states, tuple):
        raise TypeError("state sequence must be a tuple")
    if len(states) < 1:
        raise DomainError("primary-sector path requires positive depth")
    for sigma in states:
        validate_sigma(s, sigma)
    return states


@dataclass(frozen=True, slots=True)
class PrimarySector:
    s: int
    sigma: int

    def __post_init__(self) -> None:
        require_branching(self.s)
        validate_sigma(self.s, self.sigma)

    @property
    def left(self) -> Fraction:
        return Fraction(self.sigma - 1, self.s)

    @property
    def right(self) -> Fraction:
        return Fraction(self.sigma, self.s)

    @property
    def length(self) -> Fraction:
        return self.right - self.left

    def contains(self, rho: Fraction) -> bool:
        if not isinstance(rho, Fraction):
            raise TypeError("rho must be an exact Fraction")
        if rho < 0 or rho >= 1:
            raise DomainError("rho must belong to [0,1)")
        return self.left <= rho < self.right


def primary_sectors(s: int) -> tuple[PrimarySector, ...]:
    s = require_branching(s)
    return tuple(PrimarySector(s=s, sigma=sigma) for sigma in range(1, s + 1))


def digit_from_states(s: int, states: Sequence[int]) -> int:
    states = validate_state_sequence(s, states)
    depth = len(states)
    return sum((sigma - 1) * s ** (depth - index - 1) for index, sigma in enumerate(states))


def states_from_digit(s: int, depth: int, D: object) -> tuple[int, ...]:
    s = require_branching(s)
    depth = require_positive_depth(depth)
    D = require_nonnegative_int(D, "D")
    if D >= s**depth:
        raise DomainError("D is outside the depth range")
    states: list[int] = []
    remaining = D
    for power in range(depth - 1, -1, -1):
        base_power = s**power
        digit = remaining // base_power
        remaining %= base_power
        states.append(digit + 1)
    return tuple(states)


def normalized_rho_from_states(s: int, states: Sequence[int]) -> Fraction:
    states = validate_state_sequence(s, states)
    depth = len(states)
    D = digit_from_states(s, states)
    return Fraction(D, s**depth)


def normalized_rho_from_digit(s: int, depth: int, D: object) -> Fraction:
    s = require_branching(s)
    depth = require_positive_depth(depth)
    D = require_nonnegative_int(D, "D")
    if D >= s**depth:
        raise DomainError("D is outside the depth range")
    return Fraction(D, s**depth)


def spectrum(s: int, depth: int) -> tuple[Fraction, ...]:
    s = require_branching(s)
    depth = require_positive_depth(depth)
    return tuple(Fraction(q, s**depth) for q in range(s**depth))


def sector_index_of_rho(s: int, rho: Fraction) -> int:
    s = require_branching(s)
    if not isinstance(rho, Fraction):
        raise TypeError("rho must be an exact Fraction")
    if rho < 0 or rho >= 1:
        raise DomainError("rho must belong to [0,1)")
    # Half-open sectors make floor(s*rho)+1 exact.
    return int(s * rho) + 1


def sector_points(s: int, depth: int, sigma: int) -> tuple[Fraction, ...]:
    s = require_branching(s)
    depth = require_positive_depth(depth)
    sigma = validate_sigma(s, sigma)
    sector = PrimarySector(s, sigma)
    return tuple(rho for rho in spectrum(s, depth) if sector.contains(rho))


def residual_after_first_digit(s: int, states: Sequence[int]) -> Fraction:
    states = validate_state_sequence(s, states)
    if len(states) == 1:
        return Fraction(0, 1)
    depth = len(states)
    tail_digit = sum((sigma - 1) * s ** (depth - index - 1) for index, sigma in enumerate(states[1:], start=1))
    return Fraction(tail_digit, s**depth)


def full_comb_measure(s: int) -> float:
    s = require_branching(s)
    return log(s)


def primary_sector_constant(s: int) -> float:
    s = require_branching(s)
    return log(s) / s


def equal_sector_projection(s: int) -> tuple[float, ...]:
    return tuple(primary_sector_constant(s) for _ in range(require_branching(s)))


def reconstruct_full_measure(contributions: Sequence[float]) -> float:
    if not isinstance(contributions, tuple):
        raise TypeError("contributions must be a tuple")
    if len(contributions) == 0:
        raise DomainError("contribution tuple cannot be empty")
    return sum(contributions)


def verify_equal_primary_projection(s: int, contributions: Sequence[float]) -> None:
    s = require_branching(s)
    if not isinstance(contributions, tuple):
        raise TypeError("contributions must be a tuple")
    if len(contributions) != s:
        raise DomainError("there must be exactly s primary-sector contributions")
    first = contributions[0]
    for value in contributions:
        if not isclose(value, first, rel_tol=1e-14, abs_tol=1e-14):
            raise DomainError("primary-sector uniqueness theorem assumes equal primary-sector contributions")
    if not isclose(sum(contributions), full_comb_measure(s), rel_tol=1e-14, abs_tol=1e-14):
        raise DomainError("contributions do not reconstruct the full combinatorial measure")


def denominator_projection_constant(s: int, denominator: float) -> float:
    s = require_branching(s)
    if denominator <= 0:
        raise DomainError("denominator must be positive")
    return full_comb_measure(s) / denominator


def primary_reconstructs_with_denominator(s: int, denominator: float) -> bool:
    candidate = denominator_projection_constant(s, denominator)
    return isclose(s * candidate, full_comb_measure(s), rel_tol=1e-14, abs_tol=1e-14)


def q_level_subsector_constant(s: int, q: int) -> float:
    s = require_branching(s)
    q = require_positive_depth(q)
    return full_comb_measure(s) / (s**q)


def q_level_reconstructs_with_q_subsectors(s: int, q: int) -> bool:
    s = require_branching(s)
    q = require_positive_depth(q)
    return isclose((s**q) * q_level_subsector_constant(s, q), full_comb_measure(s), rel_tol=1e-14, abs_tol=1e-14)


def q_level_reconstructs_with_primary_sectors(s: int, q: int) -> bool:
    s = require_branching(s)
    q = require_positive_depth(q)
    return isclose(s * q_level_subsector_constant(s, q), full_comb_measure(s), rel_tol=1e-14, abs_tol=1e-14)


def verify_symbolic_sector_formulas() -> None:
    print("\n=== Symbolic verification of sector formulas ===")

    s, sigma, d = symbols("s sigma d", integer=True, positive=True)
    G = slog(s)
    Gs = G / s

    assert simplify(s * Gs - G) == 0

    left = (sigma - 1) / s
    right = sigma / s
    assert simplify((right - left) - 1 / s) == 0

    # Sector count at depth d: fixing first digit leaves d-1 free digits.
    sector_count = s ** (d - 1)
    total_count = s**d
    assert simplify(sector_count / total_count - 1 / s) == 0

    q = symbols("q", integer=True, positive=True)
    primary_sum_with_q_depth = simplify(s * (G / (s**q)) - G)
    assert simplify(primary_sum_with_q_depth.subs(q, 1)) == 0

    alpha = symbols("alpha", real=True)
    # s*(G/s^alpha)=G iff s^(1-alpha)=1; for s>1 this is alpha=1.
    reconstruction_factor = s / (s**alpha)
    assert simplify(reconstruction_factor.subs(alpha, 1) - 1) == 0

    print("[OK] s*(ln s/s)=ln s is symbolic")
    print("[OK] primary sector length is 1/s symbolically")
    print("[OK] finite sector fraction s^(d-1)/s^d=1/s is symbolic")
    print("[OK] q-depth denominator reconstructs primary sum only at q=1 in the primary class")


def verify_primary_sector_partition() -> None:
    print("\n=== Verification of primary-sector partition of [0,1) ===")

    checked_sectors = 0
    checked_grid_points = 0
    checked_boundaries = 0

    for s in range(2, 20):
        sectors = primary_sectors(s)
        checked_sectors += len(sectors)

        assert sectors[0].left == Fraction(0, 1)
        assert sectors[-1].right == Fraction(1, 1)
        assert all(sector.length == Fraction(1, s) for sector in sectors)

        for left, right in zip(sectors, sectors[1:]):
            checked_boundaries += 1
            assert left.right == right.left

        # A refined exact grid verifies disjoint assignment and coverage.
        test_grid = tuple(Fraction(q, s * 17) for q in range(s * 17))
        for rho in test_grid:
            checked_grid_points += 1
            containing = [sector.sigma for sector in sectors if sector.contains(rho)]
            assert len(containing) == 1
            assert containing[0] == sector_index_of_rho(s, rho)

        expect_raises(DomainError, lambda s=s: sector_index_of_rho(s, Fraction(1, 1)))
        expect_raises(DomainError, lambda s=s: sector_index_of_rho(s, Fraction(-1, s)))

    print(f"[OK] Checked {checked_sectors} primary sectors")
    print(f"[OK] Checked {checked_boundaries} adjacent half-open boundaries")
    print(f"[OK] Checked {checked_grid_points} exact partition grid points")


def verify_first_digit_sector_lemma() -> None:
    print("\n=== Verification of first-digit sector lemma ===")

    checked_paths = 0
    checked_residuals = 0
    checked_digits = 0

    for s in range(2, 9):
        for depth in range(1, 7):
            if s <= 5 and depth <= 5:
                state_samples = [tuple(states) for states in product(range(1, s + 1), repeat=depth)]
            else:
                state_samples = []
                for first in range(1, s + 1):
                    state_samples.append(tuple([first] + [1] * (depth - 1)))
                    state_samples.append(tuple([first] + [s] * (depth - 1)))
                    if depth >= 2:
                        alternating = tuple([first] + [1 + (j % s) for j in range(1, depth)])
                        state_samples.append(alternating)
                state_samples = sorted(set(state_samples))

            for states in state_samples:
                checked_paths += 1

                rho = normalized_rho_from_states(s, states)
                first = states[0]
                sector = PrimarySector(s, first)

                assert sector.contains(rho)
                assert sector_index_of_rho(s, rho) == first

                for sigma in range(1, s + 1):
                    in_sector = PrimarySector(s, sigma).contains(rho)
                    assert in_sector == (sigma == first)

                R = residual_after_first_digit(s, states)
                checked_residuals += 1
                assert Fraction(0, 1) <= R < Fraction(1, s)
                assert rho == Fraction(first - 1, s) + R

                D = digit_from_states(s, states)
                checked_digits += 1
                assert 0 <= D <= s**depth - 1
                assert states_from_digit(s, depth, D) == states
                assert normalized_rho_from_digit(s, depth, D) == rho

    print(f"[OK] Checked {checked_paths} causal digit paths")
    print(f"[OK] Checked {checked_residuals} residual bounds 0<=R<1/s")
    print(f"[OK] Checked {checked_digits} digit/rho inverse decodings")


def verify_finite_sector_fraction_and_spectrum() -> None:
    print("\n=== Verification of finite primary-sector fractions ===")

    checked_spectra = 0
    checked_sector_counts = 0
    checked_points = 0
    checked_formula_ranges = 0
    checked_sampled_points = 0

    for s in range(2, 20):
        for depth in range(1, 9):
            total = s**depth
            checked_spectra += 1

            first_point = Fraction(0, total)
            last_point = Fraction(total - 1, total)
            assert first_point == Fraction(0, 1)
            assert last_point < Fraction(1, 1)

            # Exhaustive construction is kept for moderate windows.  For large
            # windows, the exact q-range formula verifies the same counting
            # statement without materializing millions of fractions.
            exhaustive = s <= 5 and depth <= 5
            if exhaustive:
                spec = spectrum(s, depth)
                assert len(spec) == total
                assert len(set(spec)) == total
                union_points = set()
            else:
                spec = None
                union_points = None

            for sigma in range(1, s + 1):
                checked_sector_counts += 1
                q_low = (sigma - 1) * s ** (depth - 1)
                q_high = sigma * s ** (depth - 1) - 1
                sector_count = q_high - q_low + 1

                assert sector_count == s ** (depth - 1)
                assert Fraction(sector_count, total) == Fraction(1, s)
                assert PrimarySector(s, sigma).length == Fraction(1, s)
                checked_formula_ranges += 1

                sample_qs = sorted({q_low, q_high, (q_low + q_high) // 2})
                for q in sample_qs:
                    rho = Fraction(q, total)
                    checked_sampled_points += 1
                    assert PrimarySector(s, sigma).contains(rho)
                    assert sector_index_of_rho(s, rho) == sigma

                if exhaustive:
                    pts = sector_points(s, depth, sigma)
                    checked_points += len(pts)
                    expected = tuple(Fraction(q, total) for q in range(q_low, q_high + 1))
                    assert pts == expected
                    union_points.update(pts)

            if exhaustive:
                assert union_points == set(spec)

    print(f"[OK] Checked {checked_spectra} finite spectra")
    print(f"[OK] Checked {checked_sector_counts} sector count/fraction cases")
    print(f"[OK] Checked {checked_points} exhaustively materialized spectrum points")
    print(f"[OK] Checked {checked_formula_ranges} exact q-range sector formulas")
    print(f"[OK] Checked {checked_sampled_points} boundary/midpoint sector samples")


def verify_sector_projection_uniqueness() -> None:
    print("\n=== Verification of unique equal primary-sector projection ===")

    checked_s = 0
    checked_wrong_equal_values = 0
    checked_unequal_rejections = 0
    checked_wrong_lengths = 0

    for s in range(2, 80):
        G = full_comb_measure(s)
        Gs = primary_sector_constant(s)
        contributions = equal_sector_projection(s)
        checked_s += 1

        assert len(contributions) == s
        assert all(isclose(value, Gs, rel_tol=1e-14, abs_tol=1e-14) for value in contributions)
        assert isclose(reconstruct_full_measure(contributions), G, rel_tol=1e-14, abs_tol=1e-14)
        verify_equal_primary_projection(s, contributions)

        for multiplier in (Fraction(1, 2), Fraction(2, 3), Fraction(3, 2), Fraction(2, 1)):
            wrong_value = float(multiplier) * Gs
            wrong_contributions = tuple(wrong_value for _ in range(s))
            checked_wrong_equal_values += 1
            expect_raises(DomainError, lambda s=s, wrong_contributions=wrong_contributions: verify_equal_primary_projection(s, wrong_contributions))

        if s >= 3:
            unequal = tuple([Gs + 0.01, Gs - 0.01] + [Gs] * (s - 2))
            checked_unequal_rejections += 1
            # It may or may not reconstruct after perturbation depending on the
            # chosen perturbation, but it is not in the equal-sector class.
            expect_raises(DomainError, lambda s=s, unequal=unequal: verify_equal_primary_projection(s, unequal))

        checked_wrong_lengths += 2
        expect_raises(DomainError, lambda s=s, Gs=Gs: verify_equal_primary_projection(s, tuple([Gs] * (s - 1))))
        expect_raises(DomainError, lambda s=s, Gs=Gs: verify_equal_primary_projection(s, tuple([Gs] * (s + 1))))

    print(f"[OK] Checked {checked_s} unique primary-sector projections")
    print(f"[OK] Checked {checked_wrong_equal_values} wrong equal-sector normalizations")
    print(f"[OK] Checked {checked_unequal_rejections} unequal-contribution rejections")
    print(f"[OK] Checked {checked_wrong_lengths} wrong primary-sector count rejections")


def verify_normalization_guards() -> None:
    print("\n=== Verification of normalization guards ===")

    checked_q = 0
    checked_denominators = 0
    checked_alpha_cases = 0
    checked_subsector_cases = 0

    for s in range(2, 50):
        G = full_comb_measure(s)

        assert primary_reconstructs_with_denominator(s, s)
        assert not primary_reconstructs_with_denominator(s, s - 1)
        assert not primary_reconstructs_with_denominator(s, sqrt(s))

        checked_denominators += 3

        for q in range(1, 8):
            checked_q += 1
            primary_reconstruction = q_level_reconstructs_with_primary_sectors(s, q)
            assert primary_reconstruction == (q == 1)

            q_subsector_reconstruction = q_level_reconstructs_with_q_subsectors(s, q)
            checked_subsector_cases += 1
            assert q_subsector_reconstruction

            if q > 1:
                assert q_level_subsector_constant(s, q) != primary_sector_constant(s)

        for alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
            checked_alpha_cases += 1
            candidate = G / (s**alpha)
            reconstructs = isclose(s * candidate, G, rel_tol=1e-14, abs_tol=1e-14)
            assert reconstructs == isclose(alpha, 1.0, rel_tol=0.0, abs_tol=0.0)

    print(f"[OK] Checked {checked_q} q-depth denominator cases")
    print(f"[OK] Checked {checked_denominators} s, s-1, sqrt(s) denominator guards")
    print(f"[OK] Checked {checked_alpha_cases} s^alpha denominator guards")
    print(f"[OK] Checked {checked_subsector_cases} deeper sub-sector reconstruction cases")


def verify_binary_sector_constant() -> None:
    print("\n=== Verification of binary sector constant ===")

    G2 = primary_sector_constant(2)
    assert isclose(G2, log(2) / 2, rel_tol=1e-15, abs_tol=1e-15)
    assert isclose(2 * G2, log(2), rel_tol=1e-15, abs_tol=1e-15)

    print("[OK] Checked binary sector constant G_2=ln(2)/2")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(DomainError, lambda: primary_sectors(1))
    expect_raises(TypeError, lambda: primary_sectors(2.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: PrimarySector(2, 0))
    expect_raises(DomainError, lambda: PrimarySector(2, 3))
    expect_raises(TypeError, lambda: PrimarySector(2, 1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: spectrum(2, 0))
    expect_raises(DomainError, lambda: spectrum(2, -1))
    expect_raises(TypeError, lambda: spectrum(2, 1.5))  # type: ignore[arg-type]

    expect_raises(TypeError, lambda: validate_state_sequence(2, [1, 2]))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: validate_state_sequence(2, tuple()))
    expect_raises(DomainError, lambda: validate_state_sequence(2, (1, 3)))

    expect_raises(DomainError, lambda: normalized_rho_from_digit(2, 3, -1))
    expect_raises(DomainError, lambda: normalized_rho_from_digit(2, 3, 8))
    expect_raises(TypeError, lambda: normalized_rho_from_digit(2, 3, 1.5))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: states_from_digit(2, 0, 0))
    expect_raises(DomainError, lambda: states_from_digit(2, 3, 8))

    expect_raises(DomainError, lambda: sector_index_of_rho(2, Fraction(-1, 2)))
    expect_raises(DomainError, lambda: sector_index_of_rho(2, Fraction(1, 1)))
    expect_raises(TypeError, lambda: sector_index_of_rho(2, 0.5))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: denominator_projection_constant(2, 0))
    expect_raises(DomainError, lambda: q_level_subsector_constant(2, 0))
    expect_raises(DomainError, lambda: q_level_subsector_constant(2, -1))

    expect_raises(TypeError, lambda: reconstruct_full_measure([1.0, 1.0]))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: reconstruct_full_measure(tuple()))

    print("[OK] Invalid branching, depth, sigma, states, digits, rho, denominators, and contribution tuples are rejected")


def main() -> None:
    print("=== Verification of sector causal-geometric projection ===")
    verify_symbolic_sector_formulas()
    verify_primary_sector_partition()
    verify_first_digit_sector_lemma()
    verify_finite_sector_fraction_and_spectrum()
    verify_sector_projection_uniqueness()
    verify_normalization_guards()
    verify_binary_sector_constant()
    verify_negative_domain_tests()
    print("\n=== Sector causal-geometric projection verification completed successfully ===")


if __name__ == "__main__":
    main()
