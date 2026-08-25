"""
VERIFICATION of Section:
Microhistorical density and combinatorial mass
(sec:microhistorical-density-and-combinatorial-mass).

Source file:
    1_microhistorical_density_and_combinatorial_mass.tex

This script verifies the mathematical content of the section as a standalone
finite-level combinatorial layer over the already established s-ary
microhistorical tree and particle-like recursive structures.

The section uses, for a particle-like recursive structure P and a fixed
level n, the finite stable-realization notation

    H_P^(n) = P_n subseteq S^n

in the particle-like finite-projection convention fixed in Chapter 7.  The
verification represents such finite stable supports as explicit subsets of the
full s-ary layer.

its cardinality

    N_stable^(n)(P) = |H_P^(n)|,

the finite-level combinatorial mass

    m_comb^(n)(P) = log_s N_stable^(n)(P)

on levels where N_stable^(n)(P) >= 1, the stable fraction

    theta_stable^(n)(P) = N_stable^(n)(P) / s^n,

and the logarithmic stability deficit

    delta_stable^(n)(P) = n - m_comb^(n)(P) = -log_s theta_stable^(n)(P).

Verified content
----------------
1. Full s-ary finite layer:
       |S^n| = s^n.

2. Stable-realization support:
       H_P^(n)=P_n in the particle-like finite-projection convention, and the
       tested finite support lies inside S^n;
       N_stable^(n)(P)=|H_P^(n)|,
       0 <= N_stable^(n)(P) <= s^n.

3. Positive regime:
       N_stable^(n)(P) >= 1
   is required before m_comb^(n)(P), log_s theta, and delta_stable^(n)(P)
   are evaluated.

4. Combinatorial mass:
       m_comb^(n)(P)=log_s N,
       0 <= m_comb^(n)(P) <= n
   for 1 <= N <= s^n.

5. Additivity:
       log_s(N1*N2)=log_s(N1)+log_s(N2)
   for independent product multiplicities N1,N2 >= 1.

6. Stable fraction:
       theta=N/s^n,
       0 <= theta <= 1,
       N=s^n*theta.

7. Positive-regime identity:
       m_comb = n + log_s(theta).

8. Stability deficit:
       delta=n-m_comb=-log_s(theta),
       delta >= 0.

9. Boundary cases:
       N=1      -> m_comb=0, theta=s^{-n}, delta=n.
       N=s^n    -> m_comb=n, theta=1, delta=0.
       N>1      <=> m_comb>0.
       N=s^q    -> m_comb=q for 0<=q<=n.

10. Invertibility:
       N = s^{m_comb},
       theta = s^{-delta},
       N = s^{n-delta},
       m_comb + delta = n.

11. Monotonicity:
       N increasing implies m_comb increasing and delta decreasing.

12. Exact finite set checks:
       stable subsets are built explicitly inside {0,...,s^n-1}; the script
       checks their cardinalities, fractions, masses, and deficits.

13. Dependency guard:
       positive combinatorial mass is insufficient for effective gravitational
       deformation without a compatible sector measure and a strict sector deficit.

14. Negative guards:
       - s<2 is rejected;
       - n<0 is rejected;
       - N<0 and N>s^n are rejected;
       - mass is rejected for N=0;
       - theta-log and deficit are rejected for theta=0;
       - nonintegral multiplicities are rejected;
       - stable sets containing indices outside the full s-ary layer are
         rejected;
       - product multiplicities with N1=0 or N2=0 are rejected for logarithmic
         additivity;
       - treating positive mass alone as a gravitational-deficit conclusion is
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

from sympy import expand_log, log as slog, simplify, symbols


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


def require_positive_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 1:
        raise DomainError(f"{name} must be positive")
    return value


def require_branching(value: object) -> int:
    value = require_int(value, "s")
    if value < 2:
        raise DomainError("s must satisfy s>=2")
    return value


@dataclass(frozen=True, slots=True)
class StableLayer:
    s: int
    n: int
    stable_indices: frozenset[int]

    def __post_init__(self) -> None:
        require_branching(self.s)
        require_nonnegative_int(self.n, "n")

        full_size = self.s ** self.n
        for index in self.stable_indices:
            if not isinstance(index, int):
                raise TypeError("stable indices must be integers")
            if not (0 <= index < full_size):
                raise DomainError("stable index lies outside the full s-ary layer")

    @property
    def full_size(self) -> int:
        return self.s ** self.n

    @property
    def N(self) -> int:
        return len(self.stable_indices)

    @property
    def theta(self) -> Fraction:
        return Fraction(self.N, self.full_size)

    @property
    def positive(self) -> bool:
        return self.N >= 1

    def mass(self) -> float:
        return combinatorial_mass(self.s, self.n, self.N)

    def deficit(self) -> float:
        return stability_deficit(self.s, self.n, self.N)


def validate_multiplicity(s: int, n: int, N: object, *, require_positive: bool) -> int:
    s = require_branching(s)
    n = require_nonnegative_int(n, "n")
    N = require_int(N, "N")
    lower = 1 if require_positive else 0
    if N < lower:
        if require_positive:
            raise DomainError("N must be positive for logarithmic quantities")
        raise DomainError("N must be nonnegative")
    if N > s**n:
        raise DomainError("N cannot exceed the full s-ary layer size s^n")
    return N


def combinatorial_mass(s: int, n: int, N: object) -> float:
    N = validate_multiplicity(s, n, N, require_positive=True)
    return log(N, s)


def stable_fraction(s: int, n: int, N: object) -> Fraction:
    N = validate_multiplicity(s, n, N, require_positive=False)
    return Fraction(N, s**n)


def stability_deficit(s: int, n: int, N: object) -> float:
    N = validate_multiplicity(s, n, N, require_positive=True)
    return n - log(N, s)


def mass_from_theta(s: int, n: int, theta: Fraction) -> float:
    s = require_branching(s)
    n = require_nonnegative_int(n, "n")
    if not isinstance(theta, Fraction):
        raise TypeError("theta must be an exact Fraction")
    if theta <= 0 or theta > 1:
        raise DomainError("theta must satisfy 0<theta<=1 for logarithmic quantities")
    return n + log(theta.numerator / theta.denominator, s)


def deficit_from_theta(s: int, theta: Fraction) -> float:
    s = require_branching(s)
    if not isinstance(theta, Fraction):
        raise TypeError("theta must be an exact Fraction")
    if theta <= 0 or theta > 1:
        raise DomainError("theta must satisfy 0<theta<=1")
    return -log(theta.numerator / theta.denominator, s)


def product_mass(s: int, N1: object, N2: object) -> float:
    s = require_branching(s)
    N1 = require_positive_int(N1, "N1")
    N2 = require_positive_int(N2, "N2")
    return log(N1 * N2, s)


def stable_indices_by_count(s: int, n: int, N: int) -> frozenset[int]:
    validate_multiplicity(s, n, N, require_positive=False)
    return frozenset(range(N))


def residue_stable_indices(s: int, n: int, modulus: int, residue: int) -> frozenset[int]:
    require_branching(s)
    require_nonnegative_int(n, "n")
    modulus = require_positive_int(modulus, "modulus")
    residue = require_nonnegative_int(residue, "residue")
    if residue >= modulus:
        raise DomainError("residue must be smaller than modulus")
    return frozenset(index for index in range(s**n) if index % modulus == residue)


def interval_stable_indices(s: int, n: int, start: int, stop: int) -> frozenset[int]:
    require_branching(s)
    require_nonnegative_int(n, "n")
    start = require_nonnegative_int(start, "start")
    stop = require_nonnegative_int(stop, "stop")
    if start > stop:
        raise DomainError("start cannot exceed stop")
    if stop > s**n:
        raise DomainError("stop exceeds the full layer size")
    return frozenset(range(start, stop))


def effective_deformation_ready(positive_mass: bool,
                                compatible_sector_measure: bool,
                                strict_sector_deficit: bool) -> bool:
    return positive_mass and compatible_sector_measure and strict_sector_deficit


def verify_symbolic_logarithmic_identities() -> None:
    print("\n=== Symbolic verification of logarithmic identities ===")

    s, n, N, N1, N2 = symbols("s n N N1 N2", positive=True)

    m_comb = slog(N) / slog(s)
    theta = N / (s**n)
    mass_from_theta_expr = n + slog(theta) / slog(s)
    deficit = n - m_comb
    deficit_from_theta_expr = -slog(theta) / slog(s)

    assert simplify(m_comb - mass_from_theta_expr) == 0
    assert simplify(deficit - deficit_from_theta_expr) == 0
    assert simplify(m_comb + deficit - n) == 0

    additivity_residual = expand_log(slog(N1 * N2), force=True) / slog(s) - slog(N1) / slog(s) - slog(N2) / slog(s)
    assert simplify(additivity_residual) == 0

    q = symbols("q", nonnegative=True)
    assert simplify((slog(s**q) / slog(s)) - q) == 0

    print("[OK] m_comb = n + log_s(theta) is symbolic")
    print("[OK] delta = n - m_comb = -log_s(theta) is symbolic")
    print("[OK] logarithmic additivity under product multiplicity is symbolic")
    print("[OK] N=s^q implies m_comb=q symbolically")


def verify_cardinality_bounds_and_positive_regime() -> None:
    print("\n=== Verification of cardinality bounds and positive logarithmic regime ===")

    checked_layers = 0
    checked_counts = 0
    checked_zero_layers = 0

    for s in range(2, 9):
        for n in range(0, 9):
            full = s**n
            test_counts = sorted(set([0, 1, full, full // 2, max(0, full - 1)]))
            for N in test_counts:
                layer = StableLayer(s=s, n=n, stable_indices=stable_indices_by_count(s, n, N))
                checked_layers += 1
                checked_counts += 1

                assert 0 <= layer.N <= layer.full_size
                assert layer.N == N
                assert layer.full_size == full
                assert layer.theta == Fraction(N, full)

                if N == 0:
                    checked_zero_layers += 1
                    expect_raises(DomainError, lambda s=s, n=n, N=N: combinatorial_mass(s, n, N))
                    expect_raises(DomainError, lambda s=s, n=n, N=N: stability_deficit(s, n, N))
                    expect_raises(DomainError, lambda s=s, n=n: mass_from_theta(s, n, Fraction(0, 1)))
                    expect_raises(DomainError, lambda s=s: deficit_from_theta(s, Fraction(0, 1)))
                else:
                    m = layer.mass()
                    delta = layer.deficit()
                    assert -1e-12 <= m <= n + 1e-12
                    assert -1e-12 <= delta <= n + 1e-12
                    assert isclose(m + delta, n, rel_tol=1e-12, abs_tol=1e-12)
                    assert isclose(mass_from_theta(s, n, layer.theta), m, rel_tol=1e-12, abs_tol=1e-12)
                    assert isclose(deficit_from_theta(s, layer.theta), delta, rel_tol=1e-12, abs_tol=1e-12)
                    assert layer.theta.numerator * full == N * layer.theta.denominator

    print(f"[OK] Checked {checked_layers} finite stable layers")
    print(f"[OK] Checked {checked_counts} cardinality/fraction cases")
    print(f"[OK] Checked {checked_zero_layers} zero-multiplicity logarithmic rejections")


def verify_exact_stable_subsets_inside_full_layer() -> None:
    print("\n=== Exact stable-subset verification inside full s-ary layers ===")

    checked_sets = 0
    checked_indices = 0

    for s in range(2, 7):
        for n in range(1, 8):
            full = s**n
            candidate_sets = [
                frozenset(),
                frozenset({0}),
                frozenset({full - 1}),
                interval_stable_indices(s, n, 0, min(full, max(1, full // 3))),
                interval_stable_indices(s, n, max(0, full // 2), full),
                residue_stable_indices(s, n, 2, 0),
                residue_stable_indices(s, n, min(5, full), 0),
                frozenset(range(full)),
            ]

            for stable_set in candidate_sets:
                layer = StableLayer(s=s, n=n, stable_indices=stable_set)
                checked_sets += 1
                checked_indices += len(stable_set)

                assert layer.N == len(stable_set)
                assert all(0 <= index < full for index in stable_set)
                assert layer.theta == Fraction(layer.N, full)

                if layer.N > 0:
                    assert isclose(2 ** (combinatorial_mass(2, 1, 1)), 1.0, rel_tol=1e-12)  # fixed calibration
                    reconstructed_N = s ** layer.mass()
                    assert isclose(reconstructed_N, layer.N, rel_tol=1e-10, abs_tol=1e-10)
                    assert isclose(s ** (-layer.deficit()), layer.theta.numerator / layer.theta.denominator, rel_tol=1e-10, abs_tol=1e-10)
                else:
                    expect_raises(DomainError, lambda layer=layer: layer.mass())

    expect_raises(DomainError, lambda: StableLayer(s=2, n=3, stable_indices=frozenset({8})))
    expect_raises(DomainError, lambda: StableLayer(s=2, n=3, stable_indices=frozenset({-1})))
    expect_raises(TypeError, lambda: StableLayer(s=2, n=3, stable_indices=frozenset({1.5})))  # type: ignore[arg-type]

    print(f"[OK] Checked {checked_sets} explicit stable subsets")
    print(f"[OK] Checked {checked_indices} stable indices inside full layers")


def verify_boundary_cases_and_equivalences() -> None:
    print("\n=== Verification of boundary cases and equivalences ===")

    checked_boundaries = 0
    checked_power_cases = 0
    checked_nontrivial_cases = 0

    for s in range(2, 12):
        for n in range(0, 12):
            full = s**n

            # N=1 boundary.
            N = 1
            m = combinatorial_mass(s, n, N)
            theta = stable_fraction(s, n, N)
            delta = stability_deficit(s, n, N)
            checked_boundaries += 1
            assert isclose(m, 0.0, abs_tol=1e-12)
            assert theta == Fraction(1, full)
            assert isclose(delta, n, abs_tol=1e-12)
            assert not (m > 0)

            # Full layer boundary.
            N = full
            m = combinatorial_mass(s, n, N)
            theta = stable_fraction(s, n, N)
            delta = stability_deficit(s, n, N)
            checked_boundaries += 1
            assert isclose(m, n, abs_tol=1e-12)
            assert theta == Fraction(1, 1)
            assert isclose(delta, 0.0, abs_tol=1e-12)

            # Powers N=s^q.
            for q in range(0, n + 1):
                N = s**q
                checked_power_cases += 1
                assert isclose(combinatorial_mass(s, n, N), q, abs_tol=1e-12)
                assert stable_fraction(s, n, N) == Fraction(s**q, s**n)
                assert isclose(stability_deficit(s, n, N), n - q, abs_tol=1e-12)

            # N>1 iff mass>0.
            for N in sorted(set([1, min(full, 2), full, max(1, full // 3), max(1, full - 1)])):
                m = combinatorial_mass(s, n, N)
                checked_nontrivial_cases += 1
                assert (N > 1) == (m > 0)

    print(f"[OK] Checked {checked_boundaries} boundary cases N=1 and N=s^n")
    print(f"[OK] Checked {checked_power_cases} exact power cases N=s^q")
    print(f"[OK] Checked {checked_nontrivial_cases} equivalences N>1 iff m_comb>0")


def verify_additivity_and_product_multiplicity() -> None:
    print("\n=== Verification of logarithmic additivity for product multiplicities ===")

    checked_products = 0
    checked_exact_power_products = 0

    for s in range(2, 12):
        for N1 in range(1, 40):
            for N2 in range(1, 40):
                checked_products += 1
                lhs = product_mass(s, N1, N2)
                rhs = log(N1, s) + log(N2, s)
                assert isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12)

        for q1 in range(0, 8):
            for q2 in range(0, 8):
                N1 = s**q1
                N2 = s**q2
                checked_exact_power_products += 1
                assert isclose(product_mass(s, N1, N2), q1 + q2, abs_tol=1e-12)

    expect_raises(DomainError, lambda: product_mass(2, 0, 1))
    expect_raises(DomainError, lambda: product_mass(2, 1, 0))
    expect_raises(TypeError, lambda: product_mass(2, 1.5, 2))  # type: ignore[arg-type]

    print(f"[OK] Checked {checked_products} product-multiplicity additivity cases")
    print(f"[OK] Checked {checked_exact_power_products} exact power-product cases")


def verify_monotonicity_and_invertibility() -> None:
    print("\n=== Verification of monotonicity and invertibility ===")

    checked_monotone_pairs = 0
    checked_reconstructions = 0

    for s in range(2, 9):
        for n in range(1, 10):
            full = s**n
            sample_counts = sorted({
                1,
                min(full, 2),
                min(full, 3),
                max(1, full // 5),
                max(1, full // 3),
                max(1, full // 2),
                max(1, (2 * full) // 3),
                max(1, full - 2),
                max(1, full - 1),
                full,
            })

            previous_mass = None
            previous_delta = None
            previous_N = None

            for N in sample_counts:
                m = combinatorial_mass(s, n, N)
                theta = stable_fraction(s, n, N)
                delta = stability_deficit(s, n, N)

                reconstructed_N_from_mass = s**m
                reconstructed_theta_from_delta = s ** (-delta)
                reconstructed_N_from_delta = s ** (n - delta)

                checked_reconstructions += 1
                assert isclose(reconstructed_N_from_mass, N, rel_tol=1e-10, abs_tol=1e-10)
                assert isclose(reconstructed_theta_from_delta, theta.numerator / theta.denominator, rel_tol=1e-10, abs_tol=1e-10)
                assert isclose(reconstructed_N_from_delta, N, rel_tol=1e-10, abs_tol=1e-10)
                assert isclose(m + delta, n, rel_tol=1e-12, abs_tol=1e-12)

                if previous_mass is not None and N > previous_N:
                    checked_monotone_pairs += 1
                    assert m > previous_mass
                    assert delta < previous_delta

                previous_mass = m
                previous_delta = delta
                previous_N = N

    print(f"[OK] Checked {checked_monotone_pairs} strict monotonicity pairs")
    print(f"[OK] Checked {checked_reconstructions} inverse reconstructions from mass/deficit")


def verify_density_deficit_relationships_on_exact_grids() -> None:
    print("\n=== Exact rational-grid verification of theta and deficit relationships ===")

    checked_theta_values = 0

    for s in range(2, 9):
        for n in range(1, 9):
            full = s**n
            sample_counts = sorted(set([1, 2 if full >= 2 else 1, full // 2 or 1, full - 1, full]))
            for N in sample_counts:
                theta = stable_fraction(s, n, N)
                checked_theta_values += 1

                assert Fraction(1, full) <= theta <= Fraction(1, 1)
                assert theta * full == N
                assert isclose(mass_from_theta(s, n, theta), combinatorial_mass(s, n, N), rel_tol=1e-12, abs_tol=1e-12)
                assert isclose(deficit_from_theta(s, theta), stability_deficit(s, n, N), rel_tol=1e-12, abs_tol=1e-12)

                deficit = stability_deficit(s, n, N)
                assert deficit >= -1e-12
                if theta == 1:
                    assert isclose(deficit, 0.0, abs_tol=1e-12)
                if theta < 1:
                    assert deficit > 0

    expect_raises(DomainError, lambda: mass_from_theta(2, 3, Fraction(0, 1)))
    expect_raises(DomainError, lambda: mass_from_theta(2, 3, Fraction(2, 1)))
    expect_raises(DomainError, lambda: deficit_from_theta(2, Fraction(0, 1)))
    expect_raises(DomainError, lambda: deficit_from_theta(2, Fraction(3, 2)))
    expect_raises(TypeError, lambda: mass_from_theta(2, 3, 0.5))  # type: ignore[arg-type]

    print(f"[OK] Checked {checked_theta_values} exact rational theta values")


def verify_effective_deformation_dependency_guard() -> None:
    print("\n=== Dependency guard for effective deformation ===")

    checked_cases = 0

    for N in range(1, 20):
        positive_mass = N > 1
        for compatible_sector_measure in (False, True):
            for strict_sector_deficit in (False, True):
                checked_cases += 1
                ready = effective_deformation_ready(
                    positive_mass,
                    compatible_sector_measure,
                    strict_sector_deficit,
                )

                assert ready == (positive_mass and compatible_sector_measure and strict_sector_deficit)

                if positive_mass and not compatible_sector_measure:
                    assert not ready
                if positive_mass and compatible_sector_measure and not strict_sector_deficit:
                    assert not ready
                if not positive_mass:
                    assert not ready

    assert effective_deformation_ready(True, True, True)
    assert not effective_deformation_ready(True, False, True)
    assert not effective_deformation_ready(True, True, False)
    assert not effective_deformation_ready(False, True, True)

    print(f"[OK] Checked {checked_cases} effective-deformation dependency cases")
    print("[OK] Positive combinatorial mass alone is not treated as a sector-deficit conclusion")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(DomainError, lambda: StableLayer(s=1, n=0, stable_indices=frozenset()))
    expect_raises(DomainError, lambda: StableLayer(s=2, n=-1, stable_indices=frozenset()))
    expect_raises(TypeError, lambda: StableLayer(s=2.0, n=1, stable_indices=frozenset()))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: StableLayer(s=2, n=1.0, stable_indices=frozenset()))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: stable_fraction(2, 3, -1))
    expect_raises(DomainError, lambda: stable_fraction(2, 3, 9))
    expect_raises(TypeError, lambda: stable_fraction(2, 3, 1.5))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: combinatorial_mass(2, 3, 0))
    expect_raises(DomainError, lambda: combinatorial_mass(2, 3, -1))
    expect_raises(DomainError, lambda: combinatorial_mass(2, 3, 9))
    expect_raises(TypeError, lambda: combinatorial_mass(2, 3, 1.5))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: stability_deficit(2, 3, 0))
    expect_raises(DomainError, lambda: residue_stable_indices(2, 3, 0, 0))
    expect_raises(DomainError, lambda: residue_stable_indices(2, 3, 3, 3))
    expect_raises(DomainError, lambda: interval_stable_indices(2, 3, 5, 4))
    expect_raises(DomainError, lambda: interval_stable_indices(2, 3, 0, 9))

    print("[OK] Invalid branching, levels, multiplicities, theta values, and stable subsets are rejected")


def main() -> None:
    print("=== Verification of microhistorical density and combinatorial mass ===")
    verify_symbolic_logarithmic_identities()
    verify_cardinality_bounds_and_positive_regime()
    verify_exact_stable_subsets_inside_full_layer()
    verify_boundary_cases_and_equivalences()
    verify_additivity_and_product_multiplicity()
    verify_monotonicity_and_invertibility()
    verify_density_deficit_relationships_on_exact_grids()
    verify_effective_deformation_dependency_guard()
    verify_negative_domain_tests()
    print("\n=== Microhistorical density and combinatorial-mass verification completed successfully ===")


if __name__ == "__main__":
    main()
