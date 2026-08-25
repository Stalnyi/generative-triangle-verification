"""
VERIFICATION of Section: General centered spectrum of internal generative states
(sec:centered-spectrum-general).

This script provides a full mathematical verification block for the claims in
2_centered_spectrum_general.tex.  It verifies the new content of the section:
the unique unit-scale affine centering of S={1,...,s}, the exact centered
spectrum, the integer/half-integer parity classification, and the uniqueness of
the minimal two-state half-integer spectrum.

The script does not re-prove previously verified material about
recursive generation, causal coordinates, macroclasses, stability, localization,
or particle-like structures.  Here s is treated exactly as the section states:
as an abstract classification parameter; inside any one generative space it is a
single fixed global value.

Verified content
----------------
1. Unit-scale affine centering theorem:
      eta(sigma)=sigma+beta
   is centered on S={1,...,s} if and only if
      beta=-(s+1)/2.

2. Non-degeneracy of the theorem's hypotheses:
   if the unit-scale condition is dropped, centering is no longer unique;
   the general affine family eta(sigma)=alpha*sigma+beta is centered exactly when
      beta=-alpha*(s+1)/2.
   This verifies that the theorem's uniqueness statement is tied to alpha=1.

3. Exact centered spectrum:
      eta(S)={-(s-1)/2, -(s-3)/2, ..., (s-3)/2, (s-1)/2}.
   The verification checks endpoints, cardinality, strict unit spacing, symmetry
   under sigma -> s+1-sigma, zero sum, zero arithmetic mean, and radius
      B_s=(s-1)/2.

4. Parity classification:
      s odd  -> all eta-values are integers;
      s even -> all eta-values are strict half-integers.
   The verification is symbolic at the parity-parametrized level and exact over
   a large finite range.

5. Unique minimal two-state half-integer spectrum:
   among all s>=2, the only two-state centered spectrum is obtained at s=2 and is
      {-1/2,+1/2};
   larger even s are half-integer but not two-state and not minimal.

6. Classification discipline:
   spectra for different s are different possible fixed choices of a generative
   space, not several spectra simultaneously living in one fixed space.  The
   script enforces this by validating each finite spectrum against exactly one
   fixed s and by rejecting mixed-state spectra.

7. Negative-domain tests:
   invalid s<2, non-contiguous state sets, wrong beta, parity misclassification,
   duplicate/missing spectral values, and mixed spectra are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Sequence

from sympy import Symbol, simplify, summation, symbols


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(exc_type: type[BaseException], fn, *args, **kwargs) -> None:
    """Run fn and require that it raises exactly the expected exception type."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover - defensive diagnostic
        raise AssertionError(
            f"Expected {exc_type.__name__}, but got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {exc_type.__name__}, but no exception was raised")


def validate_s(s: int) -> None:
    if not isinstance(s, int):
        raise TypeError("s must be an integer")
    if s < 2:
        raise ValueError("s must satisfy s >= 2")


def states(s: int) -> tuple[int, ...]:
    validate_s(s)
    return tuple(range(1, s + 1))


def validate_state_set(candidate: Sequence[int], s: int) -> tuple[int, ...]:
    validate_s(s)
    actual = tuple(candidate)
    expected = states(s)
    if actual != expected:
        raise ValueError(f"state set must be exactly {expected}, got {actual}")
    return actual


def unit_scale_centering_beta(s: int) -> Fraction:
    validate_s(s)
    return -Fraction(s + 1, 2)


def affine_beta_for_centering(s: int, alpha: Fraction) -> Fraction:
    validate_s(s)
    if alpha == 0:
        # The section fixes alpha=1, but this helper is used to show the broader
        # affine family.  Alpha=0 produces the degenerate constant zero spectrum.
        return Fraction(0)
    return -alpha * Fraction(s + 1, 2)


def eta_value(sigma: int, s: int, beta: Fraction | None = None) -> Fraction:
    validate_s(s)
    if sigma not in states(s):
        raise ValueError(f"sigma={sigma} is not in S={{1,...,{s}}}")
    b = unit_scale_centering_beta(s) if beta is None else beta
    return Fraction(sigma) + b


def centered_spectrum(s: int) -> tuple[Fraction, ...]:
    validate_s(s)
    return tuple(eta_value(sigma, s) for sigma in states(s))


def expected_centered_spectrum(s: int) -> tuple[Fraction, ...]:
    validate_s(s)
    return tuple(Fraction(2 * j - (s + 1), 2) for j in range(1, s + 1))


def arithmetic_mean(values: Sequence[Fraction]) -> Fraction:
    if len(values) == 0:
        raise ValueError("mean is undefined for an empty sequence")
    return sum(values, Fraction(0)) / len(values)


def is_integer_value(x: Fraction) -> bool:
    return x.denominator == 1


def is_strict_half_integer_value(x: Fraction) -> bool:
    # A strict half-integer has denominator 2 in lowest terms, e.g. -3/2, -1/2.
    return x.denominator == 2


def radius(values: Sequence[Fraction]) -> Fraction:
    if len(values) == 0:
        raise ValueError("radius is undefined for an empty spectrum")
    return max(abs(v) for v in values)


@dataclass(frozen=True)
class FixedGenerativeSpectrum:
    """A centered spectrum attached to one fixed value of s."""

    s: int
    values: tuple[Fraction, ...]

    @classmethod
    def unit_scale(cls, s: int) -> "FixedGenerativeSpectrum":
        return cls(s=s, values=centered_spectrum(s))

    def validate(self) -> None:
        validate_s(self.s)
        if self.values != centered_spectrum(self.s):
            raise ValueError(
                "values do not equal the unit-scale centered spectrum for the fixed s"
            )


def validate_fixed_s_spectrum(s: int, values: Sequence[Fraction]) -> FixedGenerativeSpectrum:
    spectrum = FixedGenerativeSpectrum(s=s, values=tuple(values))
    spectrum.validate()
    return spectrum


def verify_symbolic_unit_scale_centering() -> None:
    print("\n=== Symbolic verification of unit-scale affine centering ===")

    s, beta, sigma = symbols("s beta sigma", integer=True, positive=True)
    centered_sum = summation(sigma + beta, (sigma, 1, s))
    centered_mean = simplify(centered_sum / s)

    # The centering equation has exactly one beta solution under unit scale.
    beta_solution = simplify(-centered_mean.subs(beta, 0))
    require(simplify(beta_solution + Fraction(1, 2) * (s + 1)) == 0, "wrong symbolic beta")
    require(simplify(centered_mean.subs(beta, beta_solution)) == 0, "centered mean not zero")

    # A wrong offset by any nonzero h leaves a nonzero mean h.
    h = symbols("h", nonzero=True)
    shifted_mean = simplify(centered_mean.subs(beta, beta_solution + h))
    require(simplify(shifted_mean - h) == 0, "wrong beta-offset response")

    # Direct equivalence: sum eta = 0 iff beta=-(s+1)/2 for unit scale.
    residual = simplify(centered_sum.subs(beta, beta_solution + h))
    require(simplify(residual - s * h) == 0, "sum residual should be s*h")

    print("[OK] Unit-scale affine centering has the unique symbolic offset beta=-(s+1)/2")
    print("[OK] Any nonzero offset from the unit-scale beta produces a nonzero mean")


def verify_symbolic_general_affine_non_uniqueness() -> None:
    print("\n=== Symbolic guard: centering is not unique without alpha=1 ===")

    s, sigma, alpha, beta = symbols("s sigma alpha beta", integer=True, positive=True)
    centered_sum = summation(alpha * sigma + beta, (sigma, 1, s))
    centered_mean = simplify(centered_sum / s)
    general_beta = simplify(-alpha * (s + 1) / 2)

    require(simplify(centered_mean.subs(beta, general_beta)) == 0, "general affine centering failed")

    # Different nonzero scales give different centered spectra, so the theorem's
    # uniqueness is correctly restricted to the unit-scale normalization.
    s0 = 7
    spectrum_alpha_1 = tuple(Fraction(1) * sigma + affine_beta_for_centering(s0, Fraction(1)) for sigma in states(s0))
    spectrum_alpha_2 = tuple(Fraction(2) * sigma + affine_beta_for_centering(s0, Fraction(2)) for sigma in states(s0))
    require(spectrum_alpha_1 != spectrum_alpha_2, "different affine scales should produce different centered spectra")
    require(arithmetic_mean(spectrum_alpha_1) == 0, "alpha=1 spectrum should be centered")
    require(arithmetic_mean(spectrum_alpha_2) == 0, "alpha=2 spectrum should be centered")

    print("[OK] General affine centering has beta=-alpha*(s+1)/2")
    print("[OK] The unit-scale hypothesis is necessary for uniqueness of the theorem's normalization")


def verify_exact_spectrum_structure() -> None:
    print("\n=== Exact verification of the centered spectrum structure ===")

    for s in range(2, 81):
        spectrum = centered_spectrum(s)
        expected = expected_centered_spectrum(s)

        require(spectrum == expected, f"spectrum formula mismatch for s={s}")
        require(len(spectrum) == s, f"wrong cardinality for s={s}")
        require(len(set(spectrum)) == s, f"duplicate spectral values for s={s}")
        require(arithmetic_mean(spectrum) == 0, f"mean not zero for s={s}")
        require(sum(spectrum, Fraction(0)) == 0, f"sum not zero for s={s}")

        # Endpoints and radius.
        require(spectrum[0] == -Fraction(s - 1, 2), f"left endpoint mismatch for s={s}")
        require(spectrum[-1] == Fraction(s - 1, 2), f"right endpoint mismatch for s={s}")
        require(radius(spectrum) == Fraction(s - 1, 2), f"radius mismatch for s={s}")

        # Strict unit spacing and ordering.
        for left, right in zip(spectrum, spectrum[1:]):
            require(right - left == 1, f"non-unit spacing for s={s}: {left}, {right}")
            require(left < right, f"non-strict ordering for s={s}")

        # Symmetry under sigma -> s+1-sigma.
        for idx, value in enumerate(spectrum, start=1):
            mirror = eta_value(s + 1 - idx, s)
            require(value + mirror == 0, f"mirror symmetry failed for s={s}, sigma={idx}")

        # Consecutive centered spectrum has no missing values in its unit lattice.
        lattice = {spectrum[0] + j for j in range(s)}
        require(lattice == set(spectrum), f"missing lattice values for s={s}")

    print("[OK] Exact endpoints, spacing, radius, zero mean and mirror symmetry verified for s=2..80")


def verify_symbolic_spectrum_structure() -> None:
    print("\n=== Symbolic verification of endpoint, spacing and zero-sum formulas ===")

    s, sigma = symbols("s sigma", integer=True, positive=True)
    eta = sigma - (s + 1) / 2

    left = simplify(eta.subs(sigma, 1))
    right = simplify(eta.subs(sigma, s))
    require(simplify(left + (s - 1) / 2) == 0, "symbolic left endpoint failed")
    require(simplify(right - (s - 1) / 2) == 0, "symbolic right endpoint failed")

    next_eta = (sigma + 1) - (s + 1) / 2
    require(simplify(next_eta - eta - 1) == 0, "symbolic unit spacing failed")

    mirror_eta = (s + 1 - sigma) - (s + 1) / 2
    require(simplify(eta + mirror_eta) == 0, "symbolic mirror symmetry failed")

    sum_eta = summation(eta, (sigma, 1, s))
    require(simplify(sum_eta) == 0, "symbolic sum should be zero")

    sum_eta_squared = simplify(summation(eta**2, (sigma, 1, s)))
    expected_squared = simplify(s * (s**2 - 1) / 12)
    require(simplify(sum_eta_squared - expected_squared) == 0, "symbolic second moment failed")

    print("[OK] Symbolic endpoint, spacing, mirror-symmetry, zero-sum and second-moment formulas verified")


def verify_parity_classification() -> None:
    print("\n=== Exact and symbolic verification of integer/half-integer parity classification ===")

    q = Symbol("q", integer=True, nonnegative=True)
    sigma = Symbol("sigma", integer=True, positive=True)

    # Odd s = 2q+1 gives eta = sigma-(q+1), an integer expression.
    eta_odd = simplify(sigma - ((2 * q + 1) + 1) / 2)
    require(simplify(eta_odd - (sigma - q - 1)) == 0, "odd-s integer expression failed")

    # Even s = 2q gives eta = sigma-q-1/2, a strict half-integer expression.
    eta_even = simplify(sigma - ((2 * q) + 1) / 2)
    doubled_even = simplify(2 * eta_even)
    require(simplify(doubled_even - (2 * sigma - 2 * q - 1)) == 0, "even-s doubled expression failed")

    for s in range(2, 101):
        spectrum = centered_spectrum(s)
        if s % 2 == 1:
            require(all(is_integer_value(v) for v in spectrum), f"odd s={s} should have integer spectrum")
            require(not any(is_strict_half_integer_value(v) for v in spectrum), f"odd s={s} should not be strict half-integer")
        else:
            require(all(is_strict_half_integer_value(v) for v in spectrum), f"even s={s} should have strict half-integer spectrum")
            require(not any(is_integer_value(v) for v in spectrum), f"even s={s} should not have integer spectral values")

    print("[OK] Odd s gives integer spectra; even s gives strict half-integer spectra")
    print("[OK] Symbolic parity-parametrized expressions match the classification")


def verify_unique_minimal_binary_half_integer_spectrum() -> None:
    print("\n=== Verification of the unique minimal two-state half-integer spectrum ===")

    binary = centered_spectrum(2)
    require(binary == (Fraction(-1, 2), Fraction(1, 2)), "binary centered spectrum mismatch")
    require(len(binary) == 2, "s=2 should be two-state")
    require(all(is_strict_half_integer_value(v) for v in binary), "binary spectrum should be strict half-integer")

    # No other s>=2 is two-state, because |S|=s exactly.
    for s in range(3, 101):
        spectrum = centered_spectrum(s)
        require(len(spectrum) != 2, f"s={s} should not be two-state")
        if s % 2 == 0:
            require(all(is_strict_half_integer_value(v) for v in spectrum), f"even s={s} should be half-integer")
            require(radius(spectrum) > Fraction(1, 2), f"even s={s} should not be minimal radius")
        else:
            require(all(is_integer_value(v) for v in spectrum), f"odd s={s} should be integer")

    # The only centered spectrum with radius 1/2 and unit spacing has two values.
    for s in range(2, 101):
        if radius(centered_spectrum(s)) == Fraction(1, 2):
            require(s == 2, "only s=2 may have radius 1/2")

    print("[OK] s=2 gives exactly {-1/2,+1/2}")
    print("[OK] Larger even s are half-integer but not minimal two-state spectra")


def verify_classification_discipline_fixed_s() -> None:
    print("\n=== Verification of fixed-s classification discipline ===")

    for s in range(2, 30):
        spectrum = FixedGenerativeSpectrum.unit_scale(s)
        spectrum.validate()
        recovered_s = len(spectrum.values)
        require(recovered_s == s, "cardinality should recover the fixed s")
        validate_fixed_s_spectrum(s, spectrum.values)

    # Mixing values from two different fixed spectra is rejected.
    mixed_4 = (centered_spectrum(4)[0], centered_spectrum(4)[1], centered_spectrum(6)[-2], centered_spectrum(6)[-1])
    expect_raises(ValueError, validate_fixed_s_spectrum, 4, mixed_4)

    # Duplicates, missing values, or reordered values are not the unit-scale centered spectrum.
    s = 5
    correct = centered_spectrum(s)
    duplicate = correct[:-1] + (correct[-2],)
    missing = correct[:-1]
    reordered = tuple(reversed(correct))
    expect_raises(ValueError, validate_fixed_s_spectrum, s, duplicate)
    expect_raises(ValueError, validate_fixed_s_spectrum, s, missing)
    expect_raises(ValueError, validate_fixed_s_spectrum, s, reordered)

    print("[OK] Each spectrum is validated against one fixed value of s")
    print("[OK] Mixed, duplicated, missing and reordered spectra are rejected")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative-domain verification ===")

    # Invalid s values.
    for invalid_s in (-5, -1, 0, 1):
        expect_raises(ValueError, centered_spectrum, invalid_s)
        expect_raises(ValueError, states, invalid_s)
        expect_raises(ValueError, unit_scale_centering_beta, invalid_s)

    expect_raises(TypeError, centered_spectrum, Fraction(3, 2))

    # Non-contiguous or wrong state sets are not S={1,...,s}.
    expect_raises(ValueError, validate_state_set, [0, 1, 2], 3)
    expect_raises(ValueError, validate_state_set, [1, 3], 3)
    expect_raises(ValueError, validate_state_set, [1, 2, 2, 3], 3)
    expect_raises(ValueError, validate_state_set, [1, 2, 3, 4], 3)

    # Wrong beta cannot center the unit-scale spectrum.
    for s in range(2, 20):
        wrong_beta = unit_scale_centering_beta(s) + Fraction(1, s + 1)
        wrong_values = tuple(eta_value(sigma, s, beta=wrong_beta) for sigma in states(s))
        require(arithmetic_mean(wrong_values) != 0, f"wrong beta accidentally centered s={s}")

    # Parity misclassification is explicitly rejected.
    for s in range(2, 40):
        spectrum = centered_spectrum(s)
        if s % 2 == 0:
            require(not all(is_integer_value(v) for v in spectrum), f"even s={s} misclassified as integer")
        else:
            require(not all(is_strict_half_integer_value(v) for v in spectrum), f"odd s={s} misclassified as half-integer")

    print("[OK] Invalid s, invalid state sets, wrong offsets and parity misclassifications are rejected")


def verify_dense_exact_grid() -> None:
    print("\n=== Dense exact finite grid verification ===")

    checked = 0
    for s in range(2, 121):
        beta = unit_scale_centering_beta(s)
        spectrum = centered_spectrum(s)
        expected_radius = Fraction(s - 1, 2)

        # Check several exact perturbations of beta.  None except zero may center.
        perturbations = [Fraction(a, b) for b in range(2, 13) for a in range(-12, 13) if a != 0 and gcd(abs(a), b) == 1]
        for delta in perturbations[:60]:
            perturbed = tuple(eta_value(sigma, s, beta=beta + delta) for sigma in states(s))
            require(arithmetic_mean(perturbed) == delta, "mean should shift exactly by delta")
            require(arithmetic_mean(perturbed) != 0, "nonzero perturbation should not center")
            checked += 1

        # Exact pair sums must vanish under mirror pairing.
        for i, j in zip(range(1, s + 1), range(s, 0, -1)):
            require(eta_value(i, s) + eta_value(j, s) == 0, "mirror pair sum failed")
            checked += 1

        require(radius(spectrum) == expected_radius, "radius mismatch on dense grid")
        checked += 1

    print(f"[OK] Checked {checked} exact rational perturbation, mirror and radius cases")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of centered spectrum of internal generative states ===")
    verify_symbolic_unit_scale_centering()
    verify_symbolic_general_affine_non_uniqueness()
    verify_symbolic_spectrum_structure()
    verify_exact_spectrum_structure()
    verify_parity_classification()
    verify_unique_minimal_binary_half_integer_spectrum()
    verify_classification_discipline_fixed_s()
    verify_negative_domain_tests()
    verify_dense_exact_grid()
    print("\n=== Centered spectrum general verification completed successfully ===")


if __name__ == "__main__":
    main()
