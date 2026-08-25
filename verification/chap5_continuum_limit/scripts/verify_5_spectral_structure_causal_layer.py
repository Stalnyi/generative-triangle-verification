"""
VERIFICATION of Section 5: Coordinate spectrum of the causal layer
(sec:spectral-structure-causal-layer).

This script verifies the new mathematical content of
5_spectral_structure_causal_layer.tex.  It treats the digit coordinate,
normalized causal coordinate, and discrete causal interval as previously
established dependencies, and checks the additional spectral claims made in
this section.

Verified content
----------------
1. Definition of the coordinate spectrum
      Sigma_Delta(x) = { Delta rho_x(y) | y in C^+(x) cap L_{n+Delta} }.

2. Exact finite spectrum
      Sigma_Delta(x) = { q / s^Delta | q = 0, ..., s^Delta - 1 }.
   The verification checks cardinality, endpoints, strict ordering, absence of
   duplicates, exclusion of 1, and exact uniform spacing.

3. Exact s-adic fractional representation
      q / s^Delta = sum_{j=1}^Delta d_j s^{-j},
      d_j = sigma_j - 1 in {0, ..., s-1},
   including leading zero digits and exact decoding of every digit value.

4. Monotone ordering
      D_1 < D_2  ==>  D_1/s^Delta < D_2/s^Delta.

5. Spectral step and scaling
      rho_{q+1} - rho_q = s^{-Delta},
      s^{-Delta} -> 0 as Delta -> infinity for s >= 2.

6. Recursive self-similarity
      Sigma_{Delta+1}(x)
        = union_{d=0}^{s-1} (d/s + Sigma_Delta(x)/s),
   with disjoint branch blocks and exact equality with the next spectrum.

7. Exact spectral refinement
   Sigma_Delta is nested in Sigma_{Delta+1}; each coarse cell contains exactly
   s refined points, one inherited left endpoint and s-1 new points.

8. Constructive density of the infinite union in [0,1)
   For exact rational intervals (a,b) subset [0,1), the script constructs a
   concrete depth Delta and a concrete spectral point q/s^Delta in (a,b).

9. Finite-depth non-continuum guard
   No finite layer is identified with the full interval [0,1); explicit
   rational points in [0,1) are shown to be absent from each finite spectrum.

10. Integration with level coordinates
    The same spectrum is obtained for different ancestors x and different
    absolute levels; the spectrum depends only on s and Delta, not on x.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Sequence

from sympy import Rational, oo, symbols, limit, simplify


@dataclass(frozen=True)
class ModelParams:
    m: int
    k: int
    s: int

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError("m must be a positive integer")
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if self.s < 2:
            raise ValueError("s must be at least 2")

    def level_start(self, n: int) -> int:
        if n < 0:
            raise ValueError("level index must be nonnegative")
        return self.m + self.k * (self.s**n - 1) // (self.s - 1)

    def level_size(self, n: int) -> int:
        if n < 0:
            raise ValueError("level index must be nonnegative")
        return self.k * self.s**n

@dataclass(frozen=True)
class CausalLayerPoint:
    ancestor_level: int
    ancestor_u: int
    depth: int
    digit: int
    rho: Fraction


def require_s_delta(s: int, depth: int) -> None:
    if not isinstance(s, int):
        raise ValueError("s must be an integer")
    if not isinstance(depth, int):
        raise ValueError("depth must be an integer")
    if s < 2:
        raise ValueError("s must be at least 2")
    if depth < 1:
        raise ValueError("depth must be at least 1")


def require_digit(s: int, depth: int, digit: int) -> None:
    require_s_delta(s, depth)
    if not isinstance(digit, int):
        raise ValueError("digit coordinate must be an integer")
    if not 0 <= digit <= s**depth - 1:
        raise ValueError("digit coordinate is outside the causal layer")


def coordinate_spectrum(s: int, depth: int) -> tuple[Fraction, ...]:
    require_s_delta(s, depth)
    denom = s**depth
    return tuple(Fraction(k, denom) for k in range(denom))


def spectral_point(s: int, depth: int, digit: int) -> Fraction:
    require_digit(s, depth, digit)
    return Fraction(digit, s**depth)


def digit_from_state_prefix(s: int, states: Sequence[int]) -> int:
    if s < 2:
        raise ValueError("s must be at least 2")
    if len(states) == 0:
        raise ValueError("state prefix must be nonempty")
    digit = 0
    for state in states:
        if not isinstance(state, int) or not 1 <= state <= s:
            raise ValueError("state prefix contains an invalid internal state")
        digit = s * digit + (state - 1)
    return digit


def rho_from_state_prefix(s: int, states: Sequence[int]) -> Fraction:
    digit = digit_from_state_prefix(s, states)
    return Fraction(digit, s**len(states))


def decode_digit(s: int, depth: int, digit: int) -> tuple[int, ...]:
    require_digit(s, depth, digit)
    digits = [0] * depth
    remaining = digit
    for idx in range(depth - 1, -1, -1):
        digits[idx] = remaining % s
        remaining //= s
    assert remaining == 0
    return tuple(d + 1 for d in digits)


def descendant_u(parent_u: int, s: int, depth: int, states: Sequence[int]) -> int:
    require_s_delta(s, depth)
    if parent_u < 0:
        raise ValueError("ancestor coordinate must be nonnegative")
    if len(states) != depth:
        raise ValueError("state prefix length must equal depth")
    digit = digit_from_state_prefix(s, states)
    return s**depth * parent_u + digit


def quotient_remainder(parent_u: int, child_u: int, s: int, depth: int) -> tuple[int, int]:
    require_s_delta(s, depth)
    if parent_u < 0 or child_u < 0:
        raise ValueError("coordinates must be nonnegative")
    q, r = divmod(child_u, s**depth)
    if q != parent_u:
        raise ValueError("child coordinate is not in the causal layer of the ancestor")
    return q, r


def exact_s_adic_fraction(s: int, states: Sequence[int]) -> Fraction:
    if len(states) == 0:
        raise ValueError("state prefix must be nonempty")
    total = Fraction(0)
    for j, state in enumerate(states, start=1):
        if not 1 <= state <= s:
            raise ValueError("state prefix contains an invalid internal state")
        total += Fraction(state - 1, s**j)
    return total


def floor_fraction(x: Fraction) -> int:
    return x.numerator // x.denominator


def point_inside_rational_interval(s: int, a: Fraction, b: Fraction) -> tuple[int, int, Fraction]:
    if s < 2:
        raise ValueError("s must be at least 2")
    if not (Fraction(0) <= a < b <= Fraction(1)):
        raise ValueError("interval must satisfy 0 <= a < b <= 1")

    depth = 1
    while Fraction(1, s**depth) >= b - a:
        depth += 1
    denom = s**depth
    k = floor_fraction(a * denom) + 1
    candidate = Fraction(k, denom)
    assert a < candidate < b
    return depth, k, candidate


def expect_value_error(action, label: str) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError was not raised: {label}")


def verify_exact_uniform_spectrum() -> None:
    print("\n=== Exact uniform coordinate spectrum ===")
    checked = 0
    for s in range(2, 9):
        for depth in range(1, 6):
            spec = coordinate_spectrum(s, depth)
            denom = s**depth
            assert len(spec) == denom
            assert len(set(spec)) == denom
            assert spec[0] == 0
            assert spec[-1] == Fraction(denom - 1, denom)
            assert spec[-1] < 1
            assert Fraction(1) not in spec
            assert all(Fraction(0) <= rho < Fraction(1) for rho in spec)
            assert all(spec[i] < spec[i + 1] for i in range(len(spec) - 1))
            gaps = [spec[i + 1] - spec[i] for i in range(len(spec) - 1)]
            assert all(gap == Fraction(1, denom) for gap in gaps)
            assert spec == tuple(spectral_point(s, depth, digit) for digit in range(denom))
            checked += denom
    print(f"[OK] Verified exact spectra, endpoints, order and spacing for {checked} spectral points")


def verify_s_adic_fractional_representation() -> None:
    print("\n=== Exact s-adic fractional representation and decoding ===")
    checked = 0
    leading_zero_cases = 0
    for s in range(2, 8):
        for depth in range(1, 6):
            denom = s**depth
            for digit in range(denom):
                states = decode_digit(s, depth, digit)
                assert len(states) == depth
                assert all(1 <= state <= s for state in states)
                reconstructed_digit = digit_from_state_prefix(s, states)
                assert reconstructed_digit == digit
                rho1 = Fraction(digit, denom)
                rho2 = rho_from_state_prefix(s, states)
                rho3 = exact_s_adic_fraction(s, states)
                assert rho1 == rho2 == rho3
                if depth > 1 and states[0] == 1:
                    leading_zero_cases += 1
                checked += 1
    assert leading_zero_cases > 0
    print(f"[OK] Checked {checked} exact digit/state-prefix/fraction conversions")
    print(f"[OK] Leading zero digit cases preserved: {leading_zero_cases}")


def verify_monotone_ordering() -> None:
    print("\n=== Monotone ordering induced by digit coordinates ===")
    pair_checks = 0
    for s in range(2, 9):
        for depth in range(1, 5):
            denom = s**depth
            for d1 in range(0, denom):
                rho1 = spectral_point(s, depth, d1)
                for d2 in range(d1 + 1, min(denom, d1 + 7)):
                    rho2 = spectral_point(s, depth, d2)
                    assert rho1 < rho2
                    assert rho2 - rho1 == Fraction(d2 - d1, denom)
                    pair_checks += 1
    assert pair_checks > 0
    print(f"[OK] Verified strict monotonicity for {pair_checks} exact ordered digit pairs")


def verify_spectral_step_and_limit() -> None:
    print("\n=== Spectral step and vanishing scale ===")
    for s in range(2, 12):
        previous_step = None
        for depth in range(1, 12):
            step = Fraction(1, s**depth)
            if depth <= 5 and s <= 8:
                spec = coordinate_spectrum(s, depth)
                assert spec[1] - spec[0] == step
                assert spec[-1] - spec[-2] == step
            assert step <= Fraction(1, 2**depth)
            if previous_step is not None:
                assert step == previous_step / s
                assert step < previous_step
            previous_step = step

    Delta = symbols("Delta", positive=True, integer=True)
    assert simplify(limit(Rational(1, 2) ** Delta, Delta, oo)) == 0
    print("[OK] Exact step rho_{k+1}-rho_k = s^{-Delta} verified")
    print("[OK] Vanishing follows from s^{-Delta} <= 2^{-Delta} and lim 2^{-Delta}=0")


def scaled_branch(s: int, depth: int, branch_digit: int) -> set[Fraction]:
    if not 0 <= branch_digit <= s - 1:
        raise ValueError("branch digit must be in {0, ..., s-1}")
    return {Fraction(branch_digit, s) + Fraction(1, s) * rho for rho in coordinate_spectrum(s, depth)}


def verify_recursive_self_similarity() -> None:
    print("\n=== Recursive self-similarity of spectra ===")
    checked = 0
    for s in range(2, 7):
        for depth in range(1, 5):
            branches = [scaled_branch(s, depth, branch_digit) for branch_digit in range(s)]
            union = set().union(*branches)
            next_spec = set(coordinate_spectrum(s, depth + 1))
            assert union == next_spec
            assert sum(len(branch) for branch in branches) == len(union)
            assert all(len(branch) == s**depth for branch in branches)
            for branch_digit, branch in enumerate(branches):
                expected_low = Fraction(branch_digit, s)
                expected_high = Fraction(branch_digit + 1, s)
                assert min(branch) == expected_low
                assert max(branch) == expected_high - Fraction(1, s ** (depth + 1))
                assert all(expected_low <= point < expected_high for point in branch)
            checked += len(next_spec)
    print(f"[OK] Verified exact self-similar decomposition for {checked} refined spectral points")


def verify_exact_spectral_refinement() -> None:
    print("\n=== Exact spectral refinement and nesting ===")
    inherited = 0
    new_points = 0
    cell_checks = 0
    for s in range(2, 7):
        for depth in range(1, 5):
            coarse = set(coordinate_spectrum(s, depth))
            refined = set(coordinate_spectrum(s, depth + 1))
            assert coarse.issubset(refined)
            inherited += len(coarse)
            new_points += len(refined - coarse)
            assert len(refined - coarse) == (s - 1) * len(coarse)

            coarse_den = s**depth
            refined_den = s ** (depth + 1)
            for cell in range(coarse_den):
                left = Fraction(cell, coarse_den)
                right = Fraction(cell + 1, coarse_den)
                expected = [Fraction(cell * s + a, refined_den) for a in range(s)]
                assert all(point in refined for point in expected)
                assert all(left <= point < right for point in expected)
                assert expected[0] in coarse
                assert all(point not in coarse for point in expected[1:])
                cell_checks += 1
    print(f"[OK] Nested inherited points checked: {inherited}; new refined points checked: {new_points}")
    print(f"[OK] Each coarse cell has exactly s refined points in {cell_checks} exact cells")


def verify_constructive_density_of_union() -> None:
    print("\n=== Constructive density of the infinite spectral union in [0,1) ===")
    intervals_checked = 0
    max_depth_used = 0

    endpoints: list[Fraction] = []
    for den in range(2, 18):
        for num in range(0, den + 1):
            endpoints.append(Fraction(num, den))
    endpoints = sorted(set(endpoints))

    for s in range(2, 8):
        for i, a in enumerate(endpoints):
            for b in endpoints[i + 1 : min(i + 9, len(endpoints))]:
                if Fraction(0) <= a < b <= Fraction(1):
                    depth, digit, point = point_inside_rational_interval(s, a, b)
                    assert point == Fraction(digit, s**depth)
                    assert point in coordinate_spectrum(s, depth)
                    assert a < point < b
                    max_depth_used = max(max_depth_used, depth)
                    intervals_checked += 1

    assert intervals_checked > 0
    print(f"[OK] Constructed spectral points inside {intervals_checked} rational intervals")
    print(f"[OK] Largest constructed depth used: {max_depth_used}")


def verify_finite_depth_is_not_continuum() -> None:
    print("\n=== Finite-depth non-continuum guard ===")
    missing_checks = 0
    for s in range(2, 10):
        for depth in range(1, 12):
            denom = s**depth
            missing_midpoint = Fraction(1, 2 * denom)
            assert Fraction(0) < missing_midpoint < Fraction(1)
            # If missing_midpoint = k/denom, then k = 1/2, impossible in integers.
            assert (missing_midpoint * denom).denominator != 1
            # If 1 = k/denom with k in {0,...,denom-1}, then k=denom, outside range.
            assert denom not in range(0, denom)
            missing_checks += 2
    print(f"[OK] Verified {missing_checks} finite-depth exclusions from the full interval [0,1]")


def verify_integration_with_level_coordinates() -> None:
    print("\n=== Integration with level coordinates and ancestor independence ===")
    models = [
        ModelParams(m=1, k=2, s=2),
        ModelParams(m=3, k=4, s=3),
        ModelParams(m=5, k=3, s=4),
    ]
    checked_layers = 0
    checked_descendants = 0
    for model in models:
        for level in range(0, 4):
            level_size = model.level_size(level)
            candidate_parent_us = sorted({0, min(1, level_size - 1), level_size // 2, level_size - 1})
            for parent_u in candidate_parent_us:
                parent_x = model.level_start(level) + parent_u
                assert model.level_start(level) <= parent_x < model.level_start(level) + level_size
                for depth in range(1, 5):
                    observed: set[Fraction] = set()
                    for states in product(range(1, model.s + 1), repeat=depth):
                        child_u = descendant_u(parent_u, model.s, depth, states)
                        _, digit = quotient_remainder(parent_u, child_u, model.s, depth)
                        child_x = model.level_start(level + depth) + child_u
                        assert model.level_start(level + depth) <= child_x < model.level_start(level + depth) + model.level_size(level + depth)
                        observed.add(spectral_point(model.s, depth, digit))
                        checked_descendants += 1
                    assert observed == set(coordinate_spectrum(model.s, depth))
                    checked_layers += 1
    print(f"[OK] Checked {checked_layers} causal layers across absolute levels and ancestors")
    print(f"[OK] Checked {checked_descendants} generated descendants with identical spectra")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain and corruption tests ===")
    expect_value_error(lambda: coordinate_spectrum(1, 3), "s < 2 rejected")
    expect_value_error(lambda: coordinate_spectrum(2, 0), "zero depth rejected")
    expect_value_error(lambda: coordinate_spectrum(2.5, 2), "non-integer s rejected")
    expect_value_error(lambda: coordinate_spectrum(2, 1.5), "non-integer depth rejected")
    expect_value_error(lambda: spectral_point(3, 2, -1), "negative digit rejected")
    expect_value_error(lambda: spectral_point(3, 2, 9), "digit beyond layer rejected")
    expect_value_error(lambda: decode_digit(3, 2, 9), "decode beyond layer rejected")
    expect_value_error(lambda: digit_from_state_prefix(3, []), "empty state prefix rejected")
    expect_value_error(lambda: digit_from_state_prefix(3, [1, 4]), "invalid internal state rejected")
    expect_value_error(lambda: descendant_u(-1, 3, 2, [1, 2]), "negative ancestor coordinate rejected")
    expect_value_error(lambda: descendant_u(0, 3, 2, [1]), "wrong prefix length rejected")
    expect_value_error(lambda: quotient_remainder(2, 10, 3, 2), "wrong ancestor quotient rejected")
    expect_value_error(lambda: scaled_branch(3, 2, 3), "invalid self-similar branch rejected")
    expect_value_error(lambda: point_inside_rational_interval(2, Fraction(1, 3), Fraction(1, 3)), "empty interval rejected")
    expect_value_error(lambda: point_inside_rational_interval(2, Fraction(-1, 3), Fraction(1, 3)), "negative interval endpoint rejected")
    expect_value_error(lambda: point_inside_rational_interval(2, Fraction(1, 3), Fraction(4, 3)), "interval exceeding one rejected")
    expect_value_error(lambda: ModelParams(m=0, k=2, s=2), "invalid m rejected")
    expect_value_error(lambda: ModelParams(m=1, k=1, s=2), "invalid k rejected")
    expect_value_error(lambda: ModelParams(m=1, k=2, s=1), "invalid model s rejected")
    print("[OK] Invalid domains and nonmatching causal-layer data are rejected soundly")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of coordinate spectrum of the causal layer ===")
    verify_exact_uniform_spectrum()
    verify_s_adic_fractional_representation()
    verify_monotone_ordering()
    verify_spectral_step_and_limit()
    verify_recursive_self_similarity()
    verify_exact_spectral_refinement()
    verify_constructive_density_of_union()
    verify_finite_depth_is_not_continuum()
    verify_integration_with_level_coordinates()
    verify_negative_domain_tests()
    print("\n=== Coordinate spectrum verification completed successfully ===")


if __name__ == "__main__":
    main()
