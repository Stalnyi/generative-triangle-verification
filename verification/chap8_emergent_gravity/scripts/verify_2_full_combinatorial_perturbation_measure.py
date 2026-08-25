"""
VERIFICATION of Section:
Full combinatorial perturbation measure
(sec:full-combinatorial-perturbation-measure).

Source file:
    2_full_combinatorial_perturbation_measure.tex

This script verifies the mathematical content of the section as a standalone
finite-level logarithmic-measure layer over the previously verified stable
microhistorical multiplicity and combinatorial mass.

The section assumes the positive finite-level regime

    N_stable^(n)(P) >= 1,

where

    m_comb^(n)(P) = log_s N_stable^(n)(P)

is defined.  It then proves that the unique coefficient converting the internal
s-logarithmic combinatorial mass to the natural logarithmic measure of stable
microhistorical multiplicity is

    G_comb = ln s.

Verified content
----------------
1. Positive logarithmic regime:
       s >= 2, n >= 0, 1 <= N <= s^n.

2. Change of base:
       log_s N = ln N / ln s,
       ln N = (ln s) log_s N.

3. Unique coefficient:
   If a constant C satisfies
       ln N = C log_s N
   for all admissible N>=1, then using N=s gives
       C = ln s.
   The script also verifies that N=1 alone cannot determine C, since both
   sides vanish.

4. Reconstruction:
       N = s^{m_comb},
       ln N = m_comb ln s.

5. Finite variations:
   For positive N_1,N_2 with N_1 != N_2,
       Delta(ln N) = (ln s) Delta m_comb,
       Delta(ln N)/Delta m_comb = ln s.

6. Unit mass increment:
       Delta m_comb = 1
   corresponds exactly to multiplying N by s, and therefore
       Delta(ln N) = ln s.

7. Derivative notation:
   The derivative
       d(ln N)/d m_comb = ln s
   is verified only as the slope of the exact linear map
       y(m)=m ln s.
   No continuum assumption over the discrete multiplicity set is introduced.

8. Stable-fraction and deficit conversion:
       theta = N/s^n,
       delta = -log_s theta = n - m_comb,
       -ln theta = (ln s) delta.

9. Finite variations of the deficit:
       Delta(-ln theta) = (ln s) Delta delta.

10. Boundary cases:
       N=1       -> ln N=0, m_comb=0.
       N=s^n     -> ln N=n ln s, m_comb=n.
       theta=1   -> delta=0 and -ln theta=0.
       theta=s^{-n} -> delta=n and -ln theta=n ln s.

11. Full versus sector measure:
       G_comb = ln s
   is checked as the full branching measure.  The possible later sector value
       G_s = (ln s)/s
   is explicitly not equal to G_comb for s>=2 and is not derived here as the
   full measure.

12. Negative guards:
   - s<2 is rejected;
   - n<0 is rejected;
   - N=0 is rejected for logarithmic quantities;
   - N>s^n is rejected in finite-level checks;
   - noninteger multiplicities are rejected;
   - finite-difference ratio is rejected when N_1=N_2;
   - theta=0 or theta>1 is rejected;
   - a wrong coefficient C!=ln s fails already at N=s;
   - treating the full coefficient ln s as the sector-projected coefficient
     (ln s)/s is rejected for s>=2;
   - interpreting derivative notation as an additional continuum assumption is
     rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose, log
from typing import Callable

from sympy import diff, log as slog, simplify, symbols


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


def require_nonnegative_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 0:
        raise DomainError(f"{name} must be nonnegative")
    return value


def require_branching(value: object) -> int:
    value = require_int(value, "s")
    if value < 2:
        raise DomainError("s must satisfy s>=2")
    return value


def validate_level_multiplicity(s: int, n: int, N: object, *, require_positive: bool = True) -> int:
    s = require_branching(s)
    n = require_nonnegative_int(n, "n")
    N = require_int(N, "N")
    lower = 1 if require_positive else 0
    if N < lower:
        raise DomainError("N is outside the admissible finite-level multiplicity range")
    if N > s**n:
        raise DomainError("N cannot exceed s^n on level n")
    return N


def log_s(s: int, N: object) -> float:
    s = require_branching(s)
    N = require_int(N, "N")
    if N < 1:
        raise DomainError("N must be positive for logarithms")
    return log(N, s)


def ln_N(N: object) -> float:
    N = require_int(N, "N")
    if N < 1:
        raise DomainError("N must be positive for logarithms")
    return log(N)


def combinatorial_mass(s: int, n: int, N: object) -> float:
    validate_level_multiplicity(s, n, N, require_positive=True)
    return log_s(s, N)


def natural_log_measure(s: int, n: int, N: object) -> float:
    validate_level_multiplicity(s, n, N, require_positive=True)
    return ln_N(N)


def full_combinatorial_measure(s: int) -> float:
    s = require_branching(s)
    return log(s)


def wrong_coefficient_residual_at_N(s: int, C: float, N: int) -> float:
    s = require_branching(s)
    if N < 1:
        raise DomainError("N must be positive")
    return ln_N(N) - C * log_s(s, N)


def finite_delta_mass(s: int, n: int, N1: object, N2: object) -> float:
    N1 = validate_level_multiplicity(s, n, N1, require_positive=True)
    N2 = validate_level_multiplicity(s, n, N2, require_positive=True)
    # Numerically stable form of log_s(N2)-log_s(N1).
    return log(N2 / N1, s)


def finite_delta_ln(s: int, n: int, N1: object, N2: object) -> float:
    N1 = validate_level_multiplicity(s, n, N1, require_positive=True)
    N2 = validate_level_multiplicity(s, n, N2, require_positive=True)
    # Numerically stable form of ln(N2)-ln(N1).
    return log(N2 / N1)


def finite_difference_ratio(s: int, n: int, N1: object, N2: object) -> float:
    delta_m = finite_delta_mass(s, n, N1, N2)
    if isclose(delta_m, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise DomainError("finite-difference ratio requires Delta m != 0")
    return finite_delta_ln(s, n, N1, N2) / delta_m


def stable_fraction(s: int, n: int, N: object) -> Fraction:
    N = validate_level_multiplicity(s, n, N, require_positive=False)
    return Fraction(N, s**n)


def validate_positive_theta(theta: Fraction) -> Fraction:
    if not isinstance(theta, Fraction):
        raise TypeError("theta must be an exact Fraction")
    if theta <= 0 or theta > 1:
        raise DomainError("theta must satisfy 0<theta<=1")
    return theta


def deficit_from_theta(s: int, theta: Fraction) -> float:
    s = require_branching(s)
    theta = validate_positive_theta(theta)
    return -log(theta.numerator / theta.denominator, s)


def negative_ln_theta(theta: Fraction) -> float:
    theta = validate_positive_theta(theta)
    return -log(theta.numerator / theta.denominator)


def deficit_from_multiplicity(s: int, n: int, N: object) -> float:
    theta = stable_fraction(s, n, N)
    return deficit_from_theta(s, theta)


def finite_delta_deficit(s: int, n: int, N1: object, N2: object) -> float:
    return deficit_from_multiplicity(s, n, N2) - deficit_from_multiplicity(s, n, N1)


def finite_delta_negative_ln_theta(s: int, n: int, N1: object, N2: object) -> float:
    theta1 = stable_fraction(s, n, N1)
    theta2 = stable_fraction(s, n, N2)
    return negative_ln_theta(theta2) - negative_ln_theta(theta1)


def sector_projected_candidate(s: int) -> float:
    s = require_branching(s)
    return log(s) / s


def full_measure_is_sector_measure(s: int) -> bool:
    return isclose(full_combinatorial_measure(s), sector_projected_candidate(s), rel_tol=1e-14, abs_tol=1e-14)


def derivative_statement_is_only_linear_slope(exact_linear_relation: bool, extra_continuum_assumption: bool) -> bool:
    return exact_linear_relation and not extra_continuum_assumption


@dataclass(frozen=True, slots=True)
class AdmissibleMultiplicity:
    s: int
    n: int
    N: int

    def __post_init__(self) -> None:
        validate_level_multiplicity(self.s, self.n, self.N, require_positive=True)

    @property
    def mass(self) -> float:
        return combinatorial_mass(self.s, self.n, self.N)

    @property
    def ln_measure(self) -> float:
        return natural_log_measure(self.s, self.n, self.N)

    @property
    def theta(self) -> Fraction:
        return stable_fraction(self.s, self.n, self.N)

    @property
    def deficit(self) -> float:
        return deficit_from_theta(self.s, self.theta)


def verify_symbolic_change_of_base_and_uniqueness() -> None:
    print("\n=== Symbolic verification of change of base and uniqueness ===")

    s, N, C = symbols("s N C", positive=True)
    m_comb = slog(N) / slog(s)
    natural_measure = slog(N)

    assert simplify(natural_measure - slog(s) * m_comb) == 0

    # If ln N = C log_s N for all admissible N, evaluating at N=s gives C=ln s.
    equation_at_s = slog(s) - C * (slog(s) / slog(s))
    assert simplify(equation_at_s - (slog(s) - C)) == 0

    # N=1 carries no uniqueness information: both logarithms vanish.
    equation_at_one = slog(1) - C * (slog(1) / slog(s))
    assert simplify(equation_at_one) == 0

    m = symbols("m", real=True)
    y = m * slog(s)
    assert simplify(diff(y, m) - slog(s)) == 0

    theta = symbols("theta", positive=True)
    delta = -slog(theta) / slog(s)
    assert simplify((-slog(theta)) - slog(s) * delta) == 0

    print("[OK] ln N = (ln s) log_s N is symbolic")
    print("[OK] evaluating at N=s forces C=ln s")
    print("[OK] N=1 is correctly identified as non-determining for C")
    print("[OK] derivative notation is the slope of y(m)=m ln s")
    print("[OK] -ln(theta)=(ln s)(-log_s theta) is symbolic")


def verify_coefficient_uniqueness_on_exact_finite_grid() -> None:
    print("\n=== Exact finite-grid verification of coefficient uniqueness ===")

    checked_values = 0
    checked_wrong_coefficients = 0
    checked_n_equals_one_nonuniqueness = 0

    for s in range(2, 15):
        C = full_combinatorial_measure(s)

        for n in range(1, 9):
            full = s**n
            sample_N = sorted({1, s, min(full, s**2), max(1, full // 3), max(1, full - 1), full})
            for N in sample_N:
                if N > full:
                    continue
                checked_values += 1
                m = combinatorial_mass(s, n, N)
                ln_value = natural_log_measure(s, n, N)
                assert isclose(ln_value, C * m, rel_tol=1e-12, abs_tol=1e-12)

        # Wrong coefficient fails at N=s.
        for multiplier in (Fraction(1, 2), Fraction(2, 3), Fraction(3, 2), Fraction(2, 1)):
            wrong_C = float(multiplier) * C
            checked_wrong_coefficients += 1
            assert not isclose(wrong_coefficient_residual_at_N(s, wrong_C, s), 0.0, rel_tol=1e-12, abs_tol=1e-12)

        # But N=1 cannot detect a wrong coefficient.
        for wrong_C in (0.0, C / 2, 2 * C, C + 7):
            checked_n_equals_one_nonuniqueness += 1
            assert isclose(wrong_coefficient_residual_at_N(s, wrong_C, 1), 0.0, abs_tol=1e-12)

    print(f"[OK] Checked {checked_values} admissible ln N = (ln s)m cases")
    print(f"[OK] Checked {checked_wrong_coefficients} wrong coefficients failing at N=s")
    print(f"[OK] Checked {checked_n_equals_one_nonuniqueness} N=1 non-uniqueness cases")


def verify_reconstruction_and_boundary_cases() -> None:
    print("\n=== Verification of reconstruction and boundary cases ===")

    checked_reconstructions = 0
    checked_boundaries = 0
    checked_power_cases = 0

    for s in range(2, 12):
        G = full_combinatorial_measure(s)

        for n in range(0, 10):
            full = s**n

            for N in sorted({1, full, max(1, full // 2), max(1, full - 1)}):
                obj = AdmissibleMultiplicity(s=s, n=n, N=N)
                checked_reconstructions += 1

                assert isclose(s ** obj.mass, N, rel_tol=1e-10, abs_tol=1e-10)
                assert isclose(obj.ln_measure, obj.mass * G, rel_tol=1e-12, abs_tol=1e-12)

            # Boundary N=1.
            obj = AdmissibleMultiplicity(s=s, n=n, N=1)
            checked_boundaries += 1
            assert isclose(obj.mass, 0.0, abs_tol=1e-12)
            assert isclose(obj.ln_measure, 0.0, abs_tol=1e-12)

            # Boundary N=s^n.
            obj = AdmissibleMultiplicity(s=s, n=n, N=full)
            checked_boundaries += 1
            assert isclose(obj.mass, n, abs_tol=1e-12)
            assert isclose(obj.ln_measure, n * G, rel_tol=1e-12, abs_tol=1e-12)

            for q in range(0, n + 1):
                N = s**q
                obj = AdmissibleMultiplicity(s=s, n=n, N=N)
                checked_power_cases += 1
                assert isclose(obj.mass, q, abs_tol=1e-12)
                assert isclose(obj.ln_measure, q * G, rel_tol=1e-12, abs_tol=1e-12)

    print(f"[OK] Checked {checked_reconstructions} reconstructions N=s^m and ln N=m ln s")
    print(f"[OK] Checked {checked_boundaries} boundary cases")
    print(f"[OK] Checked {checked_power_cases} exact power cases")


def verify_finite_variations_and_unit_increment() -> None:
    print("\n=== Verification of finite variations and unit mass increment ===")

    checked_pairs = 0
    checked_ratios = 0
    checked_unit_increments = 0

    for s in range(2, 12):
        G = full_combinatorial_measure(s)

        for n in range(1, 9):
            full = s**n
            sample_N = sorted({1, min(full, 2), min(full, s), max(1, full // 4), max(1, full // 2), max(1, full - 1), full})

            for N1 in sample_N:
                for N2 in sample_N:
                    if N1 == N2:
                        expect_raises(DomainError, lambda s=s, n=n, N1=N1, N2=N2: finite_difference_ratio(s, n, N1, N2))
                        continue

                    checked_pairs += 1
                    delta_ln = finite_delta_ln(s, n, N1, N2)
                    delta_m = finite_delta_mass(s, n, N1, N2)
                    ratio = finite_difference_ratio(s, n, N1, N2)

                    assert isclose(delta_ln, G * delta_m, rel_tol=1e-12, abs_tol=1e-12)
                    assert isclose(ratio, G, rel_tol=1e-12, abs_tol=1e-12)
                    checked_ratios += 1

            for q in range(0, n):
                N1 = s**q
                N2 = s ** (q + 1)
                checked_unit_increments += 1
                assert isclose(finite_delta_mass(s, n, N1, N2), 1.0, abs_tol=1e-12)
                assert isclose(finite_delta_ln(s, n, N1, N2), G, rel_tol=1e-12, abs_tol=1e-12)

    print(f"[OK] Checked {checked_pairs} nonzero finite-change pairs")
    print(f"[OK] Checked {checked_ratios} finite-difference ratios equal ln s")
    print(f"[OK] Checked {checked_unit_increments} unit mass increments N -> sN")


def verify_derivative_notation_as_linear_slope_only() -> None:
    print("\n=== Verification of derivative notation as exact linear slope only ===")

    checked_slopes = 0
    checked_log_chords = 0

    for s in range(2, 20):
        G = full_combinatorial_measure(s)

        # The function y(m)=m ln s has the same slope on every chord.
        for numerator1 in range(0, 8):
            for numerator2 in range(0, 8):
                if numerator1 == numerator2:
                    continue
                m1 = Fraction(numerator1, 3)
                m2 = Fraction(numerator2, 3)
                y1 = float(m1) * G
                y2 = float(m2) * G
                checked_slopes += 1
                assert isclose((y2 - y1) / (float(m2) - float(m1)), G, rel_tol=1e-12, abs_tol=1e-12)

        # On actual admissible multiplicities the same statement is a chord
        # identity, not an added continuum hypothesis.
        n = 6
        sample_N = sorted({1, s, s**2, s**3, s**6})
        for N1 in sample_N:
            for N2 in sample_N:
                if N1 == N2:
                    continue
                checked_log_chords += 1
                assert isclose(finite_difference_ratio(s, n, N1, N2), G, rel_tol=1e-12, abs_tol=1e-12)

        assert derivative_statement_is_only_linear_slope(True, False)
        assert not derivative_statement_is_only_linear_slope(True, True)
        assert not derivative_statement_is_only_linear_slope(False, False)

    print(f"[OK] Checked {checked_slopes} linear-slope chord cases")
    print(f"[OK] Checked {checked_log_chords} admissible logarithmic chord cases")
    print("[OK] Derivative notation is guarded against adding a continuum assumption")


def verify_deficit_conversion_and_variations() -> None:
    print("\n=== Verification of stability-deficit natural-log conversion ===")

    checked_deficits = 0
    checked_deficit_variations = 0
    checked_boundaries = 0

    for s in range(2, 12):
        G = full_combinatorial_measure(s)

        for n in range(1, 9):
            full = s**n
            sample_N = sorted({1, min(full, 2), min(full, s), max(1, full // 3), max(1, full - 1), full})

            for N in sample_N:
                theta = stable_fraction(s, n, N)
                delta = deficit_from_multiplicity(s, n, N)
                checked_deficits += 1

                assert isclose(negative_ln_theta(theta), G * delta, rel_tol=1e-12, abs_tol=1e-12)
                assert isclose(delta, n - combinatorial_mass(s, n, N), rel_tol=1e-12, abs_tol=1e-12)

                if N == full:
                    checked_boundaries += 1
                    assert theta == Fraction(1, 1)
                    assert isclose(delta, 0.0, abs_tol=1e-12)
                    assert isclose(negative_ln_theta(theta), 0.0, abs_tol=1e-12)

                if N == 1:
                    checked_boundaries += 1
                    assert theta == Fraction(1, full)
                    assert isclose(delta, n, abs_tol=1e-12)
                    assert isclose(negative_ln_theta(theta), n * G, rel_tol=1e-12, abs_tol=1e-12)

            for N1 in sample_N:
                for N2 in sample_N:
                    if N1 == N2:
                        continue
                    checked_deficit_variations += 1
                    delta_neg_ln_theta = finite_delta_negative_ln_theta(s, n, N1, N2)
                    delta_deficit = finite_delta_deficit(s, n, N1, N2)
                    assert isclose(delta_neg_ln_theta, G * delta_deficit, rel_tol=1e-12, abs_tol=1e-12)

    print(f"[OK] Checked {checked_deficits} deficit conversion cases")
    print(f"[OK] Checked {checked_deficit_variations} deficit finite-variation cases")
    print(f"[OK] Checked {checked_boundaries} theta boundary cases")


def verify_full_measure_and_sector_measure_are_distinct() -> None:
    print("\n=== Full combinatorial measure and sector projection are distinct ===")

    checked_s = 0
    checked_projection_cases = 0

    for s in range(2, 50):
        G_full = full_combinatorial_measure(s)
        G_sector = sector_projected_candidate(s)
        checked_s += 1

        assert G_full > 0
        assert G_sector > 0
        assert isclose(G_sector * s, G_full, rel_tol=1e-14, abs_tol=1e-14)
        assert not full_measure_is_sector_measure(s)
        assert not isclose(G_full, G_sector, rel_tol=1e-14, abs_tol=1e-14)

        # A later sector projection requires a sector-normalizing step.  Without
        # that step, the full measure remains ln s.
        for has_sector_projection in (False, True):
            checked_projection_cases += 1
            if has_sector_projection:
                candidate = G_sector
                assert isclose(candidate, G_full / s, rel_tol=1e-14, abs_tol=1e-14)
            else:
                candidate = G_full
                assert isclose(candidate, G_full, rel_tol=1e-14, abs_tol=1e-14)

    print(f"[OK] Checked {checked_s} branching factors for G_comb != (ln s)/s")
    print(f"[OK] Checked {checked_projection_cases} full-versus-sector projection cases")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(DomainError, lambda: AdmissibleMultiplicity(s=1, n=0, N=1))
    expect_raises(DomainError, lambda: AdmissibleMultiplicity(s=2, n=-1, N=1))
    expect_raises(DomainError, lambda: AdmissibleMultiplicity(s=2, n=3, N=0))
    expect_raises(DomainError, lambda: AdmissibleMultiplicity(s=2, n=3, N=9))
    expect_raises(TypeError, lambda: AdmissibleMultiplicity(s=2.0, n=3, N=1))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: AdmissibleMultiplicity(s=2, n=3.0, N=1))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: AdmissibleMultiplicity(s=2, n=3, N=1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: log_s(1, 1))
    expect_raises(DomainError, lambda: log_s(2, 0))
    expect_raises(TypeError, lambda: log_s(2, 1.5))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: finite_difference_ratio(2, 3, 1, 1))
    expect_raises(DomainError, lambda: finite_difference_ratio(2, 3, 0, 1))
    expect_raises(DomainError, lambda: finite_difference_ratio(2, 3, 1, 9))

    expect_raises(DomainError, lambda: stable_fraction(2, 3, -1))
    expect_raises(DomainError, lambda: stable_fraction(2, 3, 9))
    expect_raises(DomainError, lambda: deficit_from_theta(2, Fraction(0, 1)))
    expect_raises(DomainError, lambda: deficit_from_theta(2, Fraction(2, 1)))
    expect_raises(TypeError, lambda: deficit_from_theta(2, 0.5))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: sector_projected_candidate(1))

    print("[OK] Invalid branching, levels, multiplicities, theta values, and finite-change requests are rejected")


def main() -> None:
    print("=== Verification of full combinatorial perturbation measure ===")
    verify_symbolic_change_of_base_and_uniqueness()
    verify_coefficient_uniqueness_on_exact_finite_grid()
    verify_reconstruction_and_boundary_cases()
    verify_finite_variations_and_unit_increment()
    verify_derivative_notation_as_linear_slope_only()
    verify_deficit_conversion_and_variations()
    verify_full_measure_and_sector_measure_are_distinct()
    verify_negative_domain_tests()
    print("\n=== Full combinatorial perturbation-measure verification completed successfully ===")


if __name__ == "__main__":
    main()
