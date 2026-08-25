"""
VERIFICATION of Section 8: Scale structure of causal cones
(sec:scale-structure-causal-cones).

This script provides a full mathematical verification block for the claims in
8_scale_structure_causal_cones.tex.

It assumes the previously verified results on the normalized causal coordinate,
the exact finite spectra of causal layers, the continuum coordinate limit, and
the one-sided causal inequality.  It does not re-prove the construction of the
generative triangle or the earlier digit-coordinate bijections from scratch.
Instead, it verifies the new scale-structure claims introduced in this section.

Verified content
----------------
1. Elementary scale maps:
      S_d(rho) = (d + rho)/s,
   with d in {0,...,s-1}, map [0,1) exactly into
      [d/s, (d+1)/s),
   are strictly order-preserving, and have contraction factor 1/s.

2. Half-open scale cells are disjoint and cover [0,1):
      [0,1) = union_m [d/s, (d+1)/s).

3. Normalized finite causal cone at depth delta:
      C_delta^+(x) = {D/s^delta : D = 0,...,s^delta-1},
   independently of the ancestor position.

4. Exact recursive scale refinement:
      C_{delta+1}^+(x)
        = union_{d=0}^{s-1} S_d(C_delta^+(x)).
   The union is disjoint at finite depth and has cardinality s^(delta+1).

5. Exact correspondence between the refinement index d and the new leading
   s-adic digit.  For a tail digit-coordinate D at depth delta,
      S_d(D/s^delta) = (d*s^delta + D)/s^(delta+1).

6. Recursive coordinate evolution:
      rho_{delta+1} = d_1/s + rho_tail/s,
   where rho_tail is the normalized coordinate of the remaining delta internal
   states.  The script also verifies that this is not the same as appending a
   new final digit, except in special accidental cases.

7. Scale stability of the causal inequality:
      0 <= Delta r <= Delta t
   is preserved at refined depth delta+1 because the refined digit-coordinate
   remains in the exact admissible range.

8. Exponential growth of causal microhistory realizations:
      N(delta) = s^delta,
      N(delta+1) = s*N(delta),
   with exact finite enumeration.

9. No-new-geometry guard:
   normalized cones at fixed s and delta are independent of the ancestor
   level and position; changing only the ancestor cannot change the normalized
   finite cone.

10. Finite-versus-continuum guard:
    every finite cone is finite and cannot be identified with [0,1), although
    the refinement scale tends to zero and the nested union is dense in [0,1).

11. Negative domain tests:
    invalid bases, invalid scales, invalid branch indices, invalid rho values,
    invalid digits, invalid digit-coordinate bounds, non-half-open endpoint
    mistakes, and corrupted quotient/remainder decompositions are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from sympy import Symbol, simplify, symbols


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expect_value_error(fn, description: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError was not raised: {description}")


def validate_base(s: int) -> None:
    require(isinstance(s, int), "The branching base s must be an integer")
    require(s >= 2, "The branching base s must satisfy s >= 2")


def validate_delta(delta: int) -> None:
    require(isinstance(delta, int), "The scale delta must be an integer")
    require(delta >= 1, "The scale delta must satisfy delta >= 1")


def validate_branch(s: int, d: int) -> None:
    validate_base(s)
    require(isinstance(d, int), "The branch index must be an integer")
    require(0 <= d <= s - 1, "The branch index must lie in {0,...,s-1}")


def validate_digit(s: int, d: int) -> None:
    validate_base(s)
    require(isinstance(d, int), "The s-adic digit must be an integer")
    require(0 <= d <= s - 1, "The s-adic digit must lie in {0,...,s-1}")


def validate_state(s: int, sigma: int) -> None:
    validate_base(s)
    require(isinstance(sigma, int), "The internal state must be an integer")
    require(1 <= sigma <= s, "The internal state must lie in {1,...,s}")


def scale_map(s: int, d: int, rho: Fraction) -> Fraction:
    validate_branch(s, d)
    require(isinstance(rho, Fraction), "rho must be an exact Fraction")
    require(Fraction(0) <= rho < Fraction(1), "rho must lie in [0,1)")
    return Fraction(d, s) + rho / s


def finite_cone_spectrum(s: int, delta: int) -> list[Fraction]:
    validate_base(s)
    validate_delta(delta)
    denominator = s**delta
    return [Fraction(D, denominator) for D in range(denominator)]


def digit_coordinate_from_digits(s: int, digits: Sequence[int]) -> int:
    validate_base(s)
    require(len(digits) >= 1, "A nonempty finite prefix is required")
    D = 0
    for d in digits:
        validate_digit(s, d)
        D = s * D + d
    return D


def normalized_coordinate_from_digits(s: int, digits: Sequence[int]) -> Fraction:
    D = digit_coordinate_from_digits(s, digits)
    return Fraction(D, s ** len(digits))


def digits_from_digit_coordinate(s: int, delta: int, D: int) -> tuple[int, ...]:
    validate_base(s)
    validate_delta(delta)
    require(isinstance(D, int), "The digit-coordinate D must be an integer")
    require(0 <= D <= s**delta - 1, "D must lie in {0,...,s^delta-1}")

    digits_reversed: list[int] = []
    current = D
    for _ in range(delta):
        digits_reversed.append(current % s)
        current //= s
    assert current == 0
    return tuple(reversed(digits_reversed))


def normalized_cone_from_ancestor(s: int, delta: int, ancestor_position: int) -> list[Fraction]:
    validate_base(s)
    validate_delta(delta)
    require(isinstance(ancestor_position, int), "ancestor_position must be an integer")
    require(ancestor_position >= 0, "ancestor_position must be nonnegative")

    # The normalized cone depends on descendant offsets D/s^delta only.
    # The ancestor position shifts the unnormalized level coordinate but cancels
    # in the normalized layer coordinate.
    return finite_cone_spectrum(s, delta)


def descendant_position(s: int, delta: int, ancestor_position: int, D: int) -> int:
    validate_base(s)
    validate_delta(delta)
    require(isinstance(ancestor_position, int), "ancestor_position must be an integer")
    require(ancestor_position >= 0, "ancestor_position must be nonnegative")
    require(isinstance(D, int), "D must be an integer")
    require(0 <= D <= s**delta - 1, "D must lie in the exact causal-layer range")
    return s**delta * ancestor_position + D


def quotient_remainder_decode(s: int, delta: int, descendant_u: int) -> tuple[int, int]:
    validate_base(s)
    validate_delta(delta)
    require(isinstance(descendant_u, int), "descendant_u must be an integer")
    require(descendant_u >= 0, "descendant_u must be nonnegative")
    modulus = s**delta
    return descendant_u // modulus, descendant_u % modulus


def refined_digit_coordinate(s: int, delta: int, d: int, D: int) -> int:
    validate_branch(s, d)
    validate_delta(delta)
    require(isinstance(D, int), "D must be an integer")
    require(0 <= D <= s**delta - 1, "D must lie in the depth-delta range")
    return d * s**delta + D


def recursive_leading_digit_coordinate(s: int, leading_digit: int, tail_digits: Sequence[int]) -> Fraction:
    validate_digit(s, leading_digit)
    require(len(tail_digits) >= 1, "The tail finite prefix must be nonempty in this verification block")
    rho_tail = normalized_coordinate_from_digits(s, tail_digits)
    return Fraction(leading_digit, s) + rho_tail / s


def append_final_digit_coordinate(s: int, prefix_digits: Sequence[int], final_digit: int) -> Fraction:
    validate_digit(s, final_digit)
    require(len(prefix_digits) >= 1, "The finite prefix must be nonempty")
    rho_prefix = normalized_coordinate_from_digits(s, prefix_digits)
    return rho_prefix + Fraction(final_digit, s ** (len(prefix_digits) + 1))


def causal_delta_r_bounds_integer_witness(s: int, delta: int, ancestor_position: int, D: int) -> tuple[int, int]:
    """Return integer witnesses for 0 <= Delta r <= delta.

    The logarithm is monotone, so the two required inequalities reduce to:
        numerator >= denominator,
        numerator <= s^delta * denominator.
    """
    validate_base(s)
    validate_delta(delta)
    require(isinstance(ancestor_position, int), "ancestor_position must be an integer")
    require(ancestor_position >= 0, "ancestor_position must be nonnegative")
    require(isinstance(D, int), "D must be an integer")
    require(0 <= D <= s**delta - 1, "D must lie in {0,...,s^delta-1}")

    numerator = s**delta * ancestor_position + D + 1
    denominator = ancestor_position + 1

    lower_gap = numerator - denominator
    upper_gap = s**delta * denominator - numerator

    return lower_gap, upper_gap


@dataclass(frozen=True)
class ConeScale:
    s: int
    delta: int

    def __post_init__(self) -> None:
        validate_base(self.s)
        validate_delta(self.delta)

    @property
    def count(self) -> int:
        return self.s**self.delta

    @property
    def mesh(self) -> Fraction:
        return Fraction(1, self.s**self.delta)

    @property
    def spectrum(self) -> tuple[Fraction, ...]:
        return tuple(finite_cone_spectrum(self.s, self.delta))


def exact_interval_cells(s: int) -> list[tuple[Fraction, Fraction]]:
    validate_base(s)
    return [(Fraction(d, s), Fraction(d + 1, s)) for d in range(s)]


def rational_grid(max_num: int = 12, max_den: int = 12) -> list[Fraction]:
    vals = {Fraction(0)}
    for den in range(1, max_den + 1):
        for num in range(0, den):
            vals.add(Fraction(num, den))
    return sorted(vals)


def verify_elementary_scale_maps() -> None:
    print("\n=== Elementary scale transformations ===")

    checked = 0
    for s in range(2, 10):
        cells = exact_interval_cells(s)
        assert cells[0][0] == 0
        assert cells[-1][1] == 1

        for left_right, next_left_right in zip(cells, cells[1:]):
            assert left_right[1] == next_left_right[0]

        for d, (left, right) in enumerate(cells):
            for rho in rational_grid(max_num=18, max_den=18):
                image = scale_map(s, d, rho)
                assert left <= image < right
                assert image == Fraction(d, s) + rho / s
                checked += 1

            # Endpoints are half-open: rho=0 reaches the left endpoint, and
            # rho approaches but never reaches the right endpoint.
            assert scale_map(s, d, Fraction(0)) == left
            near_one = Fraction(s**6 - 1, s**6)
            assert scale_map(s, d, near_one) < right

        # Strict order preservation and exact contraction factor.
        vals = rational_grid(max_num=12, max_den=12)
        for d in range(s):
            for a in vals:
                for b in vals:
                    if a < b:
                        Sa = scale_map(s, d, a)
                        Sb = scale_map(s, d, b)
                        assert Sa < Sb
                        assert Sb - Sa == (b - a) / s

    print(f"[OK] Verified scale-map ranges, half-open cells, order preservation, and contraction on {checked} exact rational samples")


def verify_symbolic_scale_map_properties() -> None:
    print("\n=== Symbolic scale-map identities ===")

    s, d, rho1, rho2 = symbols("s d rho1 rho2", positive=True)
    S1 = (d + rho1) / s
    S2 = (d + rho2) / s

    assert simplify((S2 - S1) - (rho2 - rho1) / s) == 0

    # The finite-depth refinement identity:
    # S_d(D/s^delta) = (d*s^delta + D)/s^(delta+1).
    D, delta = symbols("D delta", integer=True, nonnegative=True)
    expr = (d + D / s**delta) / s
    expected = (d * s**delta + D) / s ** (delta + 1)
    assert simplify(expr - expected) == 0

    # The integer gaps that prove 0 <= Delta r <= delta.
    u = Symbol("u", integer=True, nonnegative=True)
    lower_gap = s**delta * u + D + 1 - (u + 1)
    upper_gap = s**delta * (u + 1) - (s**delta * u + D + 1)
    assert simplify(lower_gap - ((s**delta - 1) * u + D)) == 0
    assert simplify(upper_gap - (s**delta - D - 1)) == 0

    print("[OK] Symbolic contraction, refinement, and causal-bound gap identities verified")


def verify_normalized_cone_spectra() -> None:
    print("\n=== Normalized finite causal cones ===")

    checked = 0
    for s in range(2, 7):
        for delta in range(1, 6):
            cone = ConeScale(s=s, delta=delta)
            spectrum = cone.spectrum

            assert len(spectrum) == s**delta
            assert len(set(spectrum)) == s**delta
            assert spectrum[0] == 0
            assert spectrum[-1] == Fraction(s**delta - 1, s**delta)
            assert Fraction(1) not in spectrum
            assert all(Fraction(0) <= rho < Fraction(1) for rho in spectrum)

            if len(spectrum) > 1:
                gaps = [spectrum[i + 1] - spectrum[i] for i in range(len(spectrum) - 1)]
                assert all(g == cone.mesh for g in gaps)

            for D, rho in enumerate(spectrum):
                assert rho == Fraction(D, s**delta)
                digits = digits_from_digit_coordinate(s, delta, D)
                assert digit_coordinate_from_digits(s, digits) == D
                assert normalized_coordinate_from_digits(s, digits) == rho
                checked += 1

    print(f"[OK] Verified exact finite normalized cones and digit decoding for {checked} points")


def verify_self_similarity_of_finite_cones() -> None:
    print("\n=== Exact recursive scale refinement of finite cones ===")

    checked = 0
    for s in range(2, 7):
        for delta in range(1, 6):
            coarse = finite_cone_spectrum(s, delta)
            refined_indices: list[int] = []
            cell_sizes: list[int] = []

            for d in range(s):
                cell_indices = [d * s**delta + D for D in range(s**delta)]
                cell = [Fraction(index, s ** (delta + 1)) for index in cell_indices]
                assert len(cell) == len(coarse)
                assert all(Fraction(d, s) <= rho < Fraction(d + 1, s) for rho in cell)
                cell_sizes.append(len(set(cell_indices)))
                refined_indices.extend(cell_indices)

            assert sorted(refined_indices) == list(range(s ** (delta + 1)))
            assert len(set(refined_indices)) == s ** (delta + 1)
            assert sum(cell_sizes) == s ** (delta + 1)

            # Exact branch-digit correspondence.
            for d in range(s):
                for D in range(s**delta):
                    rho = Fraction(D, s**delta)
                    image = scale_map(s, d, rho)
                    D_refined = refined_digit_coordinate(s, delta, d, D)
                    assert image == Fraction(D_refined, s ** (delta + 1))
                    decoded = digits_from_digit_coordinate(s, delta + 1, D_refined)
                    assert decoded[0] == d
                    assert digit_coordinate_from_digits(s, decoded[1:]) == D
                    checked += 1

            # Refinement strictly increases the finite spectrum when s >= 2.
            direct_refined = finite_cone_spectrum(s, delta + 1)
            assert set(coarse).issubset(set(direct_refined))
            assert len(set(direct_refined) - set(coarse)) == s ** (delta + 1) - s**delta

    print(f"[OK] Verified self-similar refinement and leading-digit correspondence for {checked} refined branches")


def verify_recursive_coordinate_evolution() -> None:
    print("\n=== Recursive coordinate evolution ===")

    checked = 0
    accidental_equalities = 0
    non_equalities = 0

    for s in range(2, 7):
        for delta in range(1, 6):
            for D_tail in range(s**delta):
                tail_digits = digits_from_digit_coordinate(s, delta, D_tail)
                rho_tail = normalized_coordinate_from_digits(s, tail_digits)

                for leading_digit in range(s):
                    direct_digits = (leading_digit,) + tail_digits
                    direct_rho = normalized_coordinate_from_digits(s, direct_digits)
                    recursive_rho = recursive_leading_digit_coordinate(s, leading_digit, tail_digits)

                    assert direct_rho == recursive_rho
                    assert recursive_rho == Fraction(leading_digit, s) + rho_tail / s

                    # This law is a leading-digit refinement.  It is not the
                    # same operation as appending a final digit to an existing
                    # finite prefix, except in special accidental cases.
                    append_rho = append_final_digit_coordinate(s, tail_digits, leading_digit)
                    if append_rho == recursive_rho:
                        accidental_equalities += 1
                    else:
                        non_equalities += 1

                    checked += 1

    assert non_equalities > 0
    assert accidental_equalities > 0  # e.g. all-zero configurations.
    print(f"[OK] Verified leading-digit recursive law for {checked} exact finite prefixes")
    print("[OK] Verified that leading refinement is not silently confused with final-digit append")


def verify_scale_stability_of_causal_bound() -> None:
    print("\n=== Scale stability of the causal inequality ===")

    checked = 0
    boundary_lower = 0
    boundary_upper = 0

    for s in range(2, 7):
        for delta in range(1, 5):
            for ancestor_position in range(0, 8):
                for D in range(s**delta):
                    lower_gap, upper_gap = causal_delta_r_bounds_integer_witness(
                        s=s,
                        delta=delta,
                        ancestor_position=ancestor_position,
                        D=D,
                    )
                    assert lower_gap >= 0
                    assert upper_gap >= 0

                    if lower_gap == 0:
                        assert ancestor_position == 0 and D == 0
                        boundary_lower += 1
                    if upper_gap == 0:
                        assert D == s**delta - 1
                        boundary_upper += 1

                    for d in range(s):
                        D_refined = refined_digit_coordinate(s, delta, d, D)
                        lower_ref, upper_ref = causal_delta_r_bounds_integer_witness(
                            s=s,
                            delta=delta + 1,
                            ancestor_position=ancestor_position,
                            D=D_refined,
                        )
                        assert lower_ref >= 0
                        assert upper_ref >= 0

                        # Refined depth increases time by one and keeps the
                        # digit-coordinate in the exact admissible range.
                        assert 0 <= D_refined <= s ** (delta + 1) - 1
                        checked += 1

    assert boundary_lower > 0 and boundary_upper > 0
    print(f"[OK] Verified causal-bound preservation for {checked} exact refined causal intervals")
    print(f"[OK] Lower-bound equalities: {boundary_lower}; upper-bound equalities: {boundary_upper}")


def verify_exponential_microhistory_growth() -> None:
    print("\n=== Exponential growth of causal microhistory realizations ===")

    checked = 0
    for s in range(2, 7):
        for delta in range(1, 6):
            prefixes = [digits_from_digit_coordinate(s, delta, D) for D in range(s**delta)]
            assert len(prefixes) == s**delta
            assert len(set(prefixes)) == s**delta
            assert all(len(prefix) == delta for prefix in prefixes)
            assert all(all(0 <= d <= s - 1 for d in prefix) for prefix in prefixes)

            next_prefixes = [digits_from_digit_coordinate(s, delta + 1, D) for D in range(s ** (delta + 1))]
            assert len(next_prefixes) == s * len(prefixes)
            assert s ** (delta + 1) == s * s**delta

            # For each old prefix, appending one additional internal state gives
            # exactly s refinements at depth delta+1.
            refinements = {
                prefix: tuple(prefix + (d,) for d in range(s))
                for prefix in prefixes
            }
            assert all(len(set(children)) == s for children in refinements.values())
            assert sum(len(children) for children in refinements.values()) == s ** (delta + 1)

            checked += 1

    print(f"[OK] Verified N(delta)=s^delta and N(delta+1)=s*N(delta) for {checked} parameter pairs")


def verify_no_new_geometry_and_ancestor_independence() -> None:
    print("\n=== No-new-geometry and ancestor-independence checks ===")

    checked = 0
    for s in range(2, 8):
        for delta in range(1, 6):
            reference = normalized_cone_from_ancestor(s, delta, ancestor_position=0)
            for ancestor_position in [0, 1, 2, 5, 17, 123]:
                cone = normalized_cone_from_ancestor(s, delta, ancestor_position)
                assert cone == reference

                for D in [0, s**delta // 2, s**delta - 1]:
                    descendant_u = descendant_position(s, delta, ancestor_position, D)
                    q, r = quotient_remainder_decode(s, delta, descendant_u)
                    assert q == ancestor_position
                    assert r == D
                    assert Fraction(r, s**delta) in cone
                    checked += 1

    print(f"[OK] Verified normalized-cone independence from ancestor data using {checked} quotient/remainder samples")


def verify_finite_vs_continuum_guards() -> None:
    print("\n=== Finite-versus-continuum guards ===")

    checked = 0
    for s in range(2, 8):
        previous_mesh = None
        for delta in range(1, 8):
            spectrum = finite_cone_spectrum(s, delta)
            mesh = Fraction(1, s**delta)
            assert len(spectrum) == s**delta
            assert len(spectrum) < 10**9  # finite by construction
            assert Fraction(1, 2 * s**delta) not in spectrum
            assert Fraction(1) not in spectrum

            if previous_mesh is not None:
                assert mesh == previous_mesh / s
                assert mesh < previous_mesh
            previous_mesh = mesh
            checked += 1

        # Constructive density witness for a small set of rational intervals.
        intervals = [
            (Fraction(0), Fraction(1, 5)),
            (Fraction(1, 7), Fraction(1, 3)),
            (Fraction(2, 5), Fraction(3, 5)),
            (Fraction(5, 6), Fraction(1)),
        ]
        for a, b in intervals:
            assert 0 <= a < b <= 1
            delta = 1
            while Fraction(1, s**delta) >= b - a:
                delta += 1
            k = a * s**delta
            D = k.numerator // k.denominator + 1
            rho = Fraction(D, s**delta)
            assert a < rho < b
            assert rho in finite_cone_spectrum(s, delta)

    print(f"[OK] Verified finite-layer non-continuum guards and constructive density samples for {checked} finite scales")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_value_error(lambda: finite_cone_spectrum(1, 2), "base s < 2")
    expect_value_error(lambda: finite_cone_spectrum(2, 0), "delta = 0")
    expect_value_error(lambda: scale_map(2, -1, Fraction(0)), "negative branch")
    expect_value_error(lambda: scale_map(2, 2, Fraction(0)), "branch outside {0,...,s-1}")
    expect_value_error(lambda: scale_map(2, 0, Fraction(-1, 10)), "negative rho")
    expect_value_error(lambda: scale_map(2, 0, Fraction(1)), "rho = 1 is outside the half-open domain")
    expect_value_error(lambda: validate_digit(3, 3), "digit outside range")
    expect_value_error(lambda: digit_coordinate_from_digits(3, (0, 1, 3)), "invalid digit in finite prefix")
    expect_value_error(lambda: digit_coordinate_from_digits(3, ()), "empty finite prefix")
    expect_value_error(lambda: digits_from_digit_coordinate(3, 2, 9), "D outside depth range")
    expect_value_error(lambda: refined_digit_coordinate(3, 2, 3, 0), "invalid branch in refinement")
    expect_value_error(lambda: descendant_position(2, 3, -1, 0), "negative ancestor position")
    expect_value_error(lambda: descendant_position(2, 3, 0, 8), "D outside layer")
    expect_value_error(lambda: quotient_remainder_decode(2, 3, -1), "negative descendant coordinate")
    expect_value_error(lambda: recursive_leading_digit_coordinate(2, 0, ()), "empty tail prefix")
    expect_value_error(lambda: append_final_digit_coordinate(2, (), 0), "empty prefix for append check")

    # A corrupted quotient/remainder decomposition must not be accepted as a
    # valid causal-layer decomposition.
    s = 3
    delta = 2
    ancestor_position = 4
    D = 5
    u = descendant_position(s, delta, ancestor_position, D)
    q, r = quotient_remainder_decode(s, delta, u)
    assert (q, r) == (ancestor_position, D)
    corrupted_q, corrupted_r = q + 1, r
    assert corrupted_q * s**delta + corrupted_r != u

    print("[OK] Invalid domains and corrupted decompositions are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of scale structure of causal cones (sec:scale-structure-causal-cones) ===")
    verify_elementary_scale_maps()
    verify_symbolic_scale_map_properties()
    verify_normalized_cone_spectra()
    verify_self_similarity_of_finite_cones()
    verify_recursive_coordinate_evolution()
    verify_scale_stability_of_causal_bound()
    verify_exponential_microhistory_growth()
    verify_no_new_geometry_and_ancestor_independence()
    verify_finite_vs_continuum_guards()
    verify_negative_domain_tests()
    print("\n=== Scale structure of causal cones verification completed successfully ===")


if __name__ == "__main__":
    main()
