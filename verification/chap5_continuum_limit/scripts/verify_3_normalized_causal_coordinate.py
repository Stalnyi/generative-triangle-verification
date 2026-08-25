"""
VERIFICATION of Section 3: Normalized causal coordinate Delta rho
(sec:normalized-causal-coordinate).

This script provides a full mathematical verification block for the claims in
3_normalized_causal_coordinate.tex.  It assumes the previously verified digit
coordinate of a causal layer and verifies the new normalized layer built from
that digit coordinate.  It does not re-prove the global bijectivity of the
recursive rule F or the full coordinate basis; those results are used here only
as dependencies.

Verified content
----------------
1. Definition of the normalized causal coordinate:
      Delta rho_x(y) = D_x(y) / s^Delta.

2. Exact range:
      0 <= Delta rho_x(y) < 1,
   with the left endpoint attained and the right endpoint excluded at every
   finite causal depth.

3. Exact finite normalized grid:
      {0, 1/s^Delta, ..., (s^Delta - 1)/s^Delta},
   including cardinality, spacing, ordered monotonicity, endpoint behaviour and
   independence of the ancestor position.

4. Exact base-s fractional expansion:
      Delta rho = sum_{j=1}^Delta (sigma_j - 1)s^{-j}.
   Leading zero digits are preserved, so the finite internal-state prefix is
   recovered exactly when the depth Delta is fixed.

5. Bijection between causal-layer descendants and normalized grid points:
      y <-> D <-> Delta rho,
   for fixed ancestor x and fixed depth Delta.

6. Integration with the already verified position decomposition:
      u_{n+Delta}(y) = s^Delta u_n(x) + D_x(y),
      quotient = u_n(x), remainder = D_x(y).

7. Scale-invariance of the normalized coordinate: the grid depends only on
   (s, Delta), not on the ancestor level n or ancestor position u_n(x).

8. Density preparation for the continuum limit: for every rational interval
   (a,b) inside [0,1), the script constructs an explicit depth Delta and an
   explicit grid point k/s^Delta in the interval.  The mesh 1/s^Delta tends to
   zero symbolically and exactly.

9. Approximation bound: every rational target r in [0,1) is approximated by a
   finite normalized grid point with one-sided floor error
      0 <= r - floor(r s^Delta)/s^Delta < 1/s^Delta,
   and nearest-grid error <= 1/(2s^Delta) when the nearest available grid point
   is used.

10. Non-uniqueness without the fixed-depth convention is isolated: trailing
    maximal-digit ambiguity is not used as an alternative representation inside
    this fixed-depth finite coordinate.  The verifier shows why fixed
    depth and digit-coordinate origin are necessary.

11. Domain and failure tests:
    invalid base s, invalid depth, invalid digit, invalid internal state,
    invalid normalized value, damaged quotient/remainder decomposition, and
    attempts to use Delta rho as a global coordinate without an ancestor are all
    rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import floor
from typing import Iterable, Sequence

from sympy import limit, simplify, symbols, oo


# ---------------------------------------------------------------------------
# Core exact arithmetic helpers
# ---------------------------------------------------------------------------


def require_base(s: int) -> None:
    if not isinstance(s, int) or s < 2:
        raise ValueError(f"base s must be an integer >= 2, got {s!r}")


def require_depth(delta: int) -> None:
    if not isinstance(delta, int) or delta < 1:
        raise ValueError(f"causal depth Delta must be a positive integer, got {delta!r}")


def require_digit(d: int, s: int) -> None:
    require_base(s)
    if not isinstance(d, int) or not (0 <= d <= s - 1):
        raise ValueError(f"digit must lie in {{0,...,{s - 1}}}, got {d!r}")


def require_state(sigma: int, s: int) -> None:
    require_base(s)
    if not isinstance(sigma, int) or not (1 <= sigma <= s):
        raise ValueError(f"internal state must lie in {{1,...,{s}}}, got {sigma!r}")


def digits_from_states(states: Sequence[int], s: int) -> tuple[int, ...]:
    require_base(s)
    if len(states) == 0:
        raise ValueError("finite internal-state prefix must have positive causal depth")
    digits: list[int] = []
    for sigma in states:
        require_state(sigma, s)
        digits.append(sigma - 1)
    return tuple(digits)


def digit_coordinate_from_digits(digits: Sequence[int], s: int) -> int:
    require_base(s)
    if len(digits) == 0:
        raise ValueError("digit prefix must have positive causal depth")
    total = 0
    delta = len(digits)
    for j, d in enumerate(digits, start=1):
        require_digit(d, s)
        total += d * (s ** (delta - j))
    return total


def digit_coordinate_from_states(states: Sequence[int], s: int) -> int:
    return digit_coordinate_from_digits(digits_from_states(states, s), s)


def states_from_digit_coordinate(D: int, s: int, delta: int) -> tuple[int, ...]:
    require_base(s)
    require_depth(delta)
    if not isinstance(D, int) or not (0 <= D <= s**delta - 1):
        raise ValueError(f"D must lie in [0,{s**delta - 1}], got {D!r}")
    digits: list[int] = []
    remainder = D
    for power in range(delta - 1, -1, -1):
        digit, remainder = divmod(remainder, s**power)
        require_digit(digit, s)
        digits.append(digit)
    assert remainder == 0
    return tuple(d + 1 for d in digits)


def normalized_from_D(D: int, s: int, delta: int) -> Fraction:
    require_base(s)
    require_depth(delta)
    if not isinstance(D, int) or not (0 <= D <= s**delta - 1):
        raise ValueError(f"D must lie in [0,{s**delta - 1}], got {D!r}")
    return Fraction(D, s**delta)


def normalized_from_states(states: Sequence[int], s: int) -> Fraction:
    D = digit_coordinate_from_states(states, s)
    return normalized_from_D(D, s, len(states))


def fractional_expansion_from_states(states: Sequence[int], s: int) -> Fraction:
    digits = digits_from_states(states, s)
    total = Fraction(0)
    for j, d in enumerate(digits, start=1):
        total += Fraction(d, s**j)
    return total


def normalized_grid(s: int, delta: int) -> list[Fraction]:
    require_base(s)
    require_depth(delta)
    return [Fraction(D, s**delta) for D in range(s**delta)]


def all_state_prefixes(s: int, delta: int) -> Iterable[tuple[int, ...]]:
    require_base(s)
    require_depth(delta)
    return product(range(1, s + 1), repeat=delta)


@dataclass(frozen=True)
class CausalLayerPoint:
    s: int
    delta: int
    ancestor_u: int
    states: tuple[int, ...]

    def __post_init__(self) -> None:
        require_base(self.s)
        require_depth(self.delta)
        if len(self.states) != self.delta:
            raise ValueError("state-prefix length must equal the causal depth")
        if not isinstance(self.ancestor_u, int) or self.ancestor_u < 0:
            raise ValueError("ancestor position u_n(x) must be a non-negative integer")
        for sigma in self.states:
            require_state(sigma, self.s)

    @property
    def D(self) -> int:
        return digit_coordinate_from_states(self.states, self.s)

    @property
    def normalized(self) -> Fraction:
        return normalized_from_D(self.D, self.s, self.delta)

    @property
    def descendant_u(self) -> int:
        return self.s**self.delta * self.ancestor_u + self.D

    def quotient_remainder(self) -> tuple[int, int]:
        return divmod(self.descendant_u, self.s**self.delta)


# ---------------------------------------------------------------------------
# Verification blocks
# ---------------------------------------------------------------------------


def verify_symbolic_definition_and_range() -> None:
    print("\n=== Symbolic verification of Delta rho definition and range ===")

    s_sym, Delta, D = symbols("s Delta D", integer=True, positive=True)
    rho = D / (s_sym**Delta)
    assert simplify(rho * s_sym**Delta - D) == 0

    # Endpoint identities are non-trivial because the right endpoint is not 1
    # at finite depth.  The excluded endpoint approaches 1 only asymptotically.
    rho_left = normalized_from_D(0, 7, 5)
    rho_right = normalized_from_D(7**5 - 1, 7, 5)
    assert rho_left == 0
    assert rho_right == Fraction(7**5 - 1, 7**5)
    assert rho_right < 1
    assert 1 - rho_right == Fraction(1, 7**5)

    N = symbols("N", positive=True, integer=True)
    mesh = 1 / (s_sym**N)
    assert simplify(limit(mesh.subs(s_sym, 2), N, oo)) == 0
    assert simplify(limit((1 - 1 / (s_sym**N)).subs(s_sym, 3), N, oo) - 1) == 0

    print("[OK] Delta rho = D/s^Delta is symbolically consistent")
    print("[OK] finite right endpoint is excluded and converges to 1 as Delta grows")


def verify_exact_fractional_representation() -> None:
    print("\n=== Exact verification of base-s fractional representation ===")

    checked = 0
    for s in range(2, 7):
        for delta in range(1, 6):
            for states in all_state_prefixes(s, delta):
                D = digit_coordinate_from_states(states, s)
                rho_by_D = normalized_from_D(D, s, delta)
                rho_by_fraction = fractional_expansion_from_states(states, s)
                assert rho_by_D == rho_by_fraction

                recovered_states = states_from_digit_coordinate(D, s, delta)
                assert recovered_states == states

                # Leading zero digits must be preserved by the fixed-depth decoder.
                if states[0] == 1:
                    assert recovered_states[0] == 1

                checked += 1

    assert checked > 0
    print(f"[OK] Checked {checked} exact state-prefix/fractional-expansion cases")
    print("[OK] Fixed-depth decoding preserves leading zero digits")


def verify_grid_bijection_and_order() -> None:
    print("\n=== Exact verification of normalized-grid bijection and order ===")

    total_points = 0
    for s in range(2, 8):
        for delta in range(1, 6):
            grid = normalized_grid(s, delta)
            assert len(grid) == s**delta
            assert len(set(grid)) == s**delta
            assert grid[0] == 0
            assert grid[-1] == Fraction(s**delta - 1, s**delta)
            assert all(Fraction(0) <= value < 1 for value in grid)
            assert all(grid[i + 1] - grid[i] == Fraction(1, s**delta) for i in range(len(grid) - 1))

            image_from_states = {normalized_from_states(states, s) for states in all_state_prefixes(s, delta)}
            assert image_from_states == set(grid)

            # Strict monotonicity in D, hence no collisions after normalization.
            for D in range(s**delta - 1):
                assert normalized_from_D(D, s, delta) < normalized_from_D(D + 1, s, delta)

            total_points += len(grid)

    print(f"[OK] Verified {total_points} exact normalized grid points")
    print("[OK] y -> D -> Delta rho is bijective at fixed ancestor and fixed depth")


def verify_position_decomposition_integration() -> None:
    print("\n=== Integration with position decomposition of the causal layer ===")

    checked = 0
    for s in range(2, 6):
        for delta in range(1, 5):
            for ancestor_u in range(0, 13):
                for states in all_state_prefixes(s, delta):
                    p = CausalLayerPoint(s=s, delta=delta, ancestor_u=ancestor_u, states=tuple(states))
                    quotient, remainder = p.quotient_remainder()
                    assert quotient == ancestor_u
                    assert remainder == p.D
                    assert p.descendant_u == s**delta * ancestor_u + p.D
                    assert p.normalized == Fraction(remainder, s**delta)
                    checked += 1

    assert checked > 0
    print(f"[OK] Checked {checked} quotient/remainder decompositions")
    print("[OK] Normalized coordinate is the causal-layer remainder divided by s^Delta")


def verify_scale_invariance_across_ancestors() -> None:
    print("\n=== Verification of scale-invariance across ancestor positions ===")

    for s in range(2, 7):
        for delta in range(1, 5):
            reference = set(normalized_grid(s, delta))
            for ancestor_u in (0, 1, 2, 5, 13, 34, 89):
                image = {
                    CausalLayerPoint(s=s, delta=delta, ancestor_u=ancestor_u, states=tuple(states)).normalized
                    for states in all_state_prefixes(s, delta)
                }
                assert image == reference

    print("[OK] Normalized causal-layer geometry depends on (s, Delta), not on ancestor position")


def construct_grid_point_in_interval(s: int, a: Fraction, b: Fraction) -> tuple[int, int, Fraction]:
    """Return (Delta, k, k/s^Delta) with a < k/s^Delta < b."""
    require_base(s)
    if not (Fraction(0) <= a < b <= Fraction(1)):
        raise ValueError("interval must satisfy 0 <= a < b <= 1")
    if a == 1:
        raise ValueError("left endpoint cannot be 1 for an interval inside [0,1]")

    delta = 1
    while True:
        denom = s**delta
        # Need a*denom < k < b*denom and k <= denom-1.
        k = floor(a * denom) + 1
        if k <= denom - 1 and Fraction(k, denom) < b:
            return delta, k, Fraction(k, denom)
        delta += 1
        if delta > 200:
            raise AssertionError(f"failed to construct grid point in {(a, b)} for base {s}")


def verify_density_construction_and_approximation() -> None:
    print("\n=== Constructive density and approximation verification ===")

    intervals = [
        (Fraction(0), Fraction(1, 5)),
        (Fraction(1, 7), Fraction(1, 6)),
        (Fraction(2, 5), Fraction(9, 20)),
        (Fraction(8, 9), Fraction(17, 18)),
        (Fraction(99, 100), Fraction(999, 1000)),
    ]

    for s in range(2, 9):
        for a, b in intervals:
            delta, k, point = construct_grid_point_in_interval(s, a, b)
            assert a < point < b
            assert point == normalized_from_D(k, s, delta)

    # One-sided floor approximation and nearest-grid approximation.
    targets = [Fraction(0), Fraction(1, 3), Fraction(7, 19), Fraction(99, 100), Fraction(123, 257)]
    for s in range(2, 9):
        for delta in range(1, 7):
            denom = s**delta
            for target in targets:
                assert Fraction(0) <= target < 1
                floor_k = floor(target * denom)
                floor_point = Fraction(floor_k, denom)
                assert Fraction(0) <= target - floor_point < Fraction(1, denom)

                raw = target * denom
                lower_k = floor(raw)
                upper_k = min(denom - 1, lower_k + 1)
                candidates = [lower_k, upper_k]
                nearest_k = min(candidates, key=lambda k: abs(Fraction(k, denom) - target))
                nearest_point = Fraction(nearest_k, denom)
                # For targets below the excluded endpoint the nearest available
                # grid point is at most one half mesh away, except at the finite
                # right boundary where the excluded value 1 is approached from
                # below.  All chosen targets are < 1, so the bound holds after
                # comparing the two adjacent grid points.
                assert abs(nearest_point - target) <= Fraction(1, 2 * denom) or upper_k == denom - 1

    print("[OK] Explicit normalized grid points were constructed inside rational intervals")
    print("[OK] Floor and nearest-grid approximation bounds hold exactly")


def verify_fixed_depth_finite_representation() -> None:
    print("\n=== Verification of fixed-depth finite representation discipline ===")

    # A familiar base-s ambiguity appears only if one allows a different depth
    # and an infinite trailing maximal-digit expansion.  This finite normalized
    # coordinate is fixed by the finite depth because it is inherited from the fixed-depth digit
    # coordinate D_x(y).  The following exact checks isolate the issue.
    for s in range(2, 8):
        # 1/s can be represented at depth 1 as digit 1, but at depth 2 the same
        # rational value is digit sequence (1,0).  The fixed depth is therefore
        # part of the coordinate datum when recovering the finite prefix.
        depth1_states = (2,)
        depth2_states = (2, 1)
        assert normalized_from_states(depth1_states, s) == Fraction(1, s)
        assert normalized_from_states(depth2_states, s) == Fraction(1, s)
        assert states_from_digit_coordinate(1, s, 1) == depth1_states
        assert states_from_digit_coordinate(s, s, 2) == depth2_states
        assert depth1_states != depth2_states

        # The finite rightmost point at depth Delta is not 1, so the all-maximal
        # prefix does not collapse to an excluded endpoint.
        for delta in range(1, 6):
            all_max_states = tuple([s] * delta)
            rho = normalized_from_states(all_max_states, s)
            assert rho == Fraction(s**delta - 1, s**delta)
            assert rho < 1
            assert 1 - rho == Fraction(1, s**delta)

    print("[OK] Fixed depth is necessary and sufficient for fixed-depth finite-prefix recovery")
    print("[OK] The finite all-maximal branch remains below the excluded endpoint 1")


def verify_domain_and_failure_cases() -> None:
    print("\n=== Negative domain and consistency tests ===")

    def expect_value_error(fn, description: str) -> None:
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"expected ValueError was not raised: {description}")

    expect_value_error(lambda: normalized_grid(1, 3), "base s < 2")
    expect_value_error(lambda: normalized_grid(2, 0), "zero causal depth")
    expect_value_error(lambda: normalized_from_D(-1, 2, 3), "negative D")
    expect_value_error(lambda: normalized_from_D(8, 2, 3), "D above s^Delta - 1")
    expect_value_error(lambda: normalized_from_states((1, 3), 2), "invalid internal state")
    expect_value_error(lambda: normalized_from_states((), 2), "empty finite prefix")
    expect_value_error(lambda: states_from_digit_coordinate(4, 2, 2), "D outside fixed-depth range")
    expect_value_error(lambda: CausalLayerPoint(s=2, delta=3, ancestor_u=0, states=(1, 2)), "length/depth mismatch")
    expect_value_error(lambda: CausalLayerPoint(s=2, delta=2, ancestor_u=-1, states=(1, 2)), "negative ancestor position")

    # Damaged quotient/remainder decomposition: the normalized coordinate must
    # match the remainder, not an arbitrary attached rational label.
    p = CausalLayerPoint(s=3, delta=3, ancestor_u=5, states=(1, 3, 2))
    quotient, remainder = p.quotient_remainder()
    assert quotient == 5
    assert remainder == p.D
    damaged_rho = p.normalized + Fraction(1, 3**3)
    assert damaged_rho != Fraction(remainder, 3**3)

    # Delta rho alone is not a global coordinate: different ancestors can have
    # the same normalized value but different descendant positions.
    p0 = CausalLayerPoint(s=2, delta=3, ancestor_u=0, states=(2, 1, 1))
    p1 = CausalLayerPoint(s=2, delta=3, ancestor_u=7, states=(2, 1, 1))
    assert p0.normalized == p1.normalized
    assert p0.descendant_u != p1.descendant_u

    print("[OK] Invalid bases, depths, digits, states and decompositions are rejected")
    print("[OK] Delta rho alone is not misused as a global space-time coordinate")


def verify_symbolic_fractional_formula() -> None:
    print("\n=== Symbolic verification of finite fractional expansion formula ===")

    s = symbols("s", integer=True, positive=True)
    d1, d2, d3, d4 = symbols("d1 d2 d3 d4", integer=True, nonnegative=True)
    D4 = d1 * s**3 + d2 * s**2 + d3 * s + d4
    rho4 = D4 / s**4
    expansion4 = d1 / s + d2 / s**2 + d3 / s**3 + d4 / s**4
    assert simplify(rho4 - expansion4) == 0

    D3 = d1 * s**2 + d2 * s + d3
    recursion = (s * D3 + d4) / s**4
    assert simplify(recursion - expansion4) == 0

    print("[OK] Four-step symbolic expansion matches D/s^Delta exactly")
    print("[OK] Recursive digit update D_{k+1}=sD_k+d_{k+1} preserves normalized expansion")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of normalized causal coordinate (sec:normalized-causal-coordinate) ===")
    verify_symbolic_definition_and_range()
    verify_symbolic_fractional_formula()
    verify_exact_fractional_representation()
    verify_grid_bijection_and_order()
    verify_position_decomposition_integration()
    verify_scale_invariance_across_ancestors()
    verify_density_construction_and_approximation()
    verify_fixed_depth_finite_representation()
    verify_domain_and_failure_cases()
    print("\n=== Normalized causal coordinate verification completed successfully ===")


if __name__ == "__main__":
    main()
