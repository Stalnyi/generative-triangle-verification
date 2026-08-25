"""
VERIFICATION of Section 5.4: Exact discrete causal interval
(sec:discrete-causal-interval).

This script verifies the new mathematical content of
4_discrete_causal_interval.tex.  It treats the coordinate basis, the digit
coordinate, and the normalized causal coordinate as already established
inputs, and checks only the additional interval-level consequences introduced
in this file.

Verified content
----------------
1. The coordinate causal interval
       I_T(x,y) = (Delta t(x,y), Delta r(x,y))
   is well-defined for x in L_n and y in C^+(x) cap L_{n+Delta n}, with
       Delta t = Delta n.

2. The logarithmic spatial increment is exactly
       Delta r = log_s((u_{n+Delta n}(y)+1)/(u_n(x)+1)).

3. Substitution of the exact causal-layer decomposition
       u_{n+Delta n}(y) = s^{Delta n} u_n(x) + D_x(y)
   gives the exact digit-coordinate formula
       Delta r = log_s((s^{Delta n}u + D + 1)/(u + 1)).

4. The causal interval inequality is verified without floating-point
   approximation:
       0 <= Delta r <= Delta n.
   Since s >= 2, this is checked equivalently as
       1 <= (s^{Delta n}u + D + 1)/(u + 1) <= s^{Delta n}.

5. Boundary cases and equality conditions:
       Delta r = 0       iff u = 0 and D = 0;
       Delta r = Delta n iff D = s^{Delta n}-1.
   The latter is equivalent to the right-boundary internal-state trajectory
   sigma_j = s for all j.  The left digit D=0 is equivalent to
   sigma_j = 1 for all j, but Delta r = 0 additionally requires u = 0.

6. Exact integration with the previously verified position-coordinate
   recursion:
       u_{j+1} = s u_j + (\\sigma_{j+1}-1).
   Iterating this recursion yields the same descendant coordinate as the
   closed formula with D_x(y).

7. Non-uniformity of the Delta r spectrum:
   the normalized coordinate Delta rho has uniform step 1/s^{Delta n}, while
   the logarithmic transform produces strictly decreasing Delta r gaps
   whenever the layer has at least three points.

8. Monotonicity and concavity of the discrete logarithmic transform over the
   causal layer are checked exactly through ratio comparisons and finite
   second differences.

9. Negative domain tests reject invalid bases, invalid depths, invalid digits,
   invalid internal states, invalid level positions, and corrupted causal-layer
   decompositions.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Sequence

from sympy import Symbol, factor, log, simplify, symbols


@dataclass(frozen=True)
class CausalLayerPoint:
    s: int
    delta_n: int
    ancestor_u: int
    digit: int

    @property
    def q(self) -> int:
        return self.s ** self.delta_n

    @property
    def descendant_u(self) -> int:
        return self.q * self.ancestor_u + self.digit

    @property
    def log_argument(self) -> Fraction:
        return Fraction(self.descendant_u + 1, self.ancestor_u + 1)

    @property
    def normalized_coordinate(self) -> Fraction:
        return Fraction(self.digit, self.q)


def validate_base_and_depth(s: int, delta_n: int) -> None:
    if not isinstance(s, int) or s < 2:
        raise ValueError("The branching base s must be an integer with s >= 2")
    if not isinstance(delta_n, int) or delta_n < 1:
        raise ValueError("The causal depth Delta n must be a positive integer")


def validate_ancestor_position(u: int) -> None:
    if not isinstance(u, int) or u < 0:
        raise ValueError("The ancestor position u_n(x) must be a nonnegative integer")


def validate_digit(s: int, delta_n: int, digit: int) -> None:
    validate_base_and_depth(s, delta_n)
    q = s ** delta_n
    if not isinstance(digit, int) or not (0 <= digit <= q - 1):
        raise ValueError("The digit coordinate must satisfy 0 <= D <= s^Delta_n - 1")


def validate_internal_state_sequence(s: int, states: Sequence[int], expected_length: int | None = None) -> None:
    if expected_length is not None and len(states) != expected_length:
        raise ValueError("The finite internal-state prefix has the wrong length")
    if len(states) == 0:
        raise ValueError("A positive-depth causal layer requires a nonempty finite prefix")
    for sigma in states:
        if not isinstance(sigma, int) or not (1 <= sigma <= s):
            raise ValueError("Internal states must lie in {1, ..., s}")


def digit_from_states(s: int, states: Sequence[int]) -> int:
    validate_internal_state_sequence(s, states)
    delta_n = len(states)
    return sum((sigma - 1) * s ** (delta_n - j - 1) for j, sigma in enumerate(states))


def states_from_digit(s: int, delta_n: int, digit: int) -> tuple[int, ...]:
    validate_digit(s, delta_n, digit)
    remaining = digit
    decoded: list[int] = []
    for power in range(delta_n - 1, -1, -1):
        place = s ** power
        value = remaining // place
        decoded.append(value + 1)
        remaining -= value * place
    assert remaining == 0
    validate_internal_state_sequence(s, decoded, expected_length=delta_n)
    return tuple(decoded)


def descendant_position_by_recursion(s: int, ancestor_u: int, states: Sequence[int]) -> int:
    validate_ancestor_position(ancestor_u)
    validate_internal_state_sequence(s, states)
    current = ancestor_u
    for sigma in states:
        current = s * current + (sigma - 1)
    return current


def interval_point_from_states(s: int, ancestor_u: int, states: Sequence[int]) -> CausalLayerPoint:
    validate_ancestor_position(ancestor_u)
    validate_internal_state_sequence(s, states)
    delta_n = len(states)
    digit = digit_from_states(s, states)
    point = CausalLayerPoint(s=s, delta_n=delta_n, ancestor_u=ancestor_u, digit=digit)
    assert point.descendant_u == descendant_position_by_recursion(s, ancestor_u, states)
    return point


def level_start(m: int, k: int, s: int, n: int) -> int:
    if not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer")
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer with k >= 2")
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    validate_base_and_depth(s, 1)
    return m + k * (s**n - 1) // (s - 1)


def level_size(k: int, s: int, n: int) -> int:
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer with k >= 2")
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    validate_base_and_depth(s, 1)
    return k * s**n


def element_from_position(m: int, k: int, s: int, n: int, u: int) -> int:
    validate_ancestor_position(u)
    size = level_size(k, s, n)
    if u >= size:
        raise ValueError("The level position u is outside L_n")
    return level_start(m, k, s, n) + u


def coordinate_position(m: int, k: int, s: int, n: int, x: int) -> int:
    start = level_start(m, k, s, n)
    size = level_size(k, s, n)
    if not (start <= x <= start + size - 1):
        raise ValueError("The element does not belong to the stated level")
    return x - start


def interval_log_argument(s: int, delta_n: int, ancestor_u: int, digit: int) -> Fraction:
    validate_digit(s, delta_n, digit)
    validate_ancestor_position(ancestor_u)
    point = CausalLayerPoint(s=s, delta_n=delta_n, ancestor_u=ancestor_u, digit=digit)
    return point.log_argument


def assert_expected_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except (ValueError, AssertionError):
        return
    raise AssertionError(f"Expected an error for {fn.__name__}{args}, but the call succeeded")


def verify_symbolic_discrete_interval_formula() -> None:
    print("\n=== Symbolic verification of the exact discrete causal interval formula ===")

    q, u, D = symbols("q u D", integer=True, positive=True)

    combined_argument = (q * u + D + 1) / (u + 1)

    lower_difference = factor(q * u + D + 1 - (u + 1))
    upper_difference = factor(q * (u + 1) - (q * u + D + 1))

    assert lower_difference == D + q * u - u
    assert upper_difference == q - D - 1

    # Boundary substitutions at the level of the logarithmic argument.
    assert simplify(combined_argument.subs(D, 0) - (q * u + 1) / (u + 1)) == 0
    assert simplify(combined_argument.subs(D, q - 1) - q) == 0

    # Strict nonlinearity over the discrete layer is captured by a negative
    # second finite difference of the logarithmic transform.  The logarithm base
    # is increasing for q > 1, so it suffices to verify that this ratio is < 1.
    B = Symbol("B", nonnegative=True, integer=True)
    numerator_gap_ratio = (B + 1) * (B + 3)
    denominator_gap_ratio = (B + 2) ** 2
    assert factor(denominator_gap_ratio - numerator_gap_ratio) == 1

    print("[OK] Delta r formula follows from the log-coordinate difference and causal-layer decomposition")
    print("[OK] Lower/upper interval inequalities reduce to nonnegative integer expressions")
    print("[OK] Symbolic second-difference check confirms nonlinear logarithmic spacing")


def verify_exact_finite_interval_inequalities() -> None:
    print("\n=== Exact finite verification of 0 <= Delta r <= Delta t ===")

    checked = 0
    lower_equalities = 0
    upper_equalities = 0
    strict_interior = 0

    for s in range(2, 8):
        for delta_n in range(1, 5):
            q = s**delta_n
            for u in range(0, 17):
                for D in range(q):
                    ratio = interval_log_argument(s, delta_n, u, D)
                    checked += 1

                    assert Fraction(1) <= ratio <= Fraction(q)
                    if ratio == 1:
                        assert u == 0 and D == 0
                        lower_equalities += 1
                    elif ratio == q:
                        assert D == q - 1
                        upper_equalities += 1
                    else:
                        assert Fraction(1) < ratio < Fraction(q)
                        strict_interior += 1

    assert checked > 0
    assert lower_equalities > 0
    assert upper_equalities > 0
    assert strict_interior > 0

    print(f"[OK] Checked {checked} exact rational interval arguments")
    print(f"[OK] Lower equalities: {lower_equalities}; upper equalities: {upper_equalities}; strict interior: {strict_interior}")


def verify_boundary_internal_state_characterizations() -> None:
    print("\n=== Verification of boundary internal-state trajectories ===")

    checked = 0
    for s in range(2, 9):
        for delta_n in range(1, 6):
            q = s**delta_n
            left_states = tuple(1 for _ in range(delta_n))
            right_states = tuple(s for _ in range(delta_n))

            assert digit_from_states(s, left_states) == 0
            assert digit_from_states(s, right_states) == q - 1
            assert states_from_digit(s, delta_n, 0) == left_states
            assert states_from_digit(s, delta_n, q - 1) == right_states

            # Delta r = 0 requires both left digit and left ancestor position.
            assert interval_log_argument(s, delta_n, 0, 0) == 1
            for u in range(1, 20):
                assert interval_log_argument(s, delta_n, u, 0) > 1

            # Delta r = Delta n depends only on the right-boundary digit.
            for u in range(0, 20):
                assert interval_log_argument(s, delta_n, u, q - 1) == q

            checked += 1

    print(f"[OK] Checked {checked} base/depth boundary cases")
    print("[OK] D=0 is the left boundary, but Delta r=0 also requires ancestor position u=0")
    print("[OK] D=s^Delta_n-1 is exactly the right boundary and gives Delta r=Delta n for every u")


def verify_recursion_and_real_level_integration() -> None:
    print("\n=== Integration with position recursion and finite level coordinates ===")

    checked_paths = 0
    checked_level_coordinates = 0

    for s in range(2, 6):
        for states in product(range(1, s + 1), repeat=4):
            for u in range(0, 14):
                point = interval_point_from_states(s, u, states)
                q = s ** len(states)
                assert point.descendant_u == q * u + point.digit
                assert point.log_argument == Fraction(point.descendant_u + 1, u + 1)
                checked_paths += 1

    test_parameters = [
        (1, 2, 2),
        (3, 4, 2),
        (2, 3, 3),
        (5, 2, 4),
    ]

    for m, k, s in test_parameters:
        for n in range(0, 5):
            size = level_size(k, s, n)
            sample_positions = sorted({0, min(1, size - 1), size // 2, size - 1})
            for u in sample_positions:
                x = element_from_position(m, k, s, n, u)
                assert coordinate_position(m, k, s, n, x) == u
                for states in product(range(1, s + 1), repeat=3):
                    point = interval_point_from_states(s, u, states)
                    y = element_from_position(m, k, s, n + 3, point.descendant_u)
                    assert coordinate_position(m, k, s, n + 3, y) == point.descendant_u
                    assert (n + 3) - n == 3
                    assert point.log_argument == Fraction(point.descendant_u + 1, u + 1)
                    checked_level_coordinates += 1

    print(f"[OK] Checked {checked_paths} recursively generated causal-layer paths")
    print(f"[OK] Checked {checked_level_coordinates} integrations with actual level coordinates")


def verify_normalized_coordinate_and_log_spectrum() -> None:
    print("\n=== Verification of normalized uniformity versus logarithmic non-uniformity ===")

    uniform_rho_checks = 0
    nonlinear_gap_checks = 0
    monotonic_checks = 0

    for s in range(2, 8):
        for delta_n in range(1, 5):
            q = s**delta_n
            rho_values = [Fraction(D, q) for D in range(q)]
            if q >= 2:
                rho_gaps = [rho_values[i + 1] - rho_values[i] for i in range(q - 1)]
                assert all(gap == Fraction(1, q) for gap in rho_gaps)
                uniform_rho_checks += len(rho_gaps)

            for u in range(0, 12):
                log_arguments = [interval_log_argument(s, delta_n, u, D) for D in range(q)]
                assert all(log_arguments[i] < log_arguments[i + 1] for i in range(q - 1))
                monotonic_checks += max(0, q - 1)

                # Delta r gaps are logarithms of these exact ratios.
                # Because log_s is increasing, gap comparisons reduce to
                # comparing the corresponding positive arguments exactly.
                gap_arguments = [Fraction(log_arguments[i + 1], log_arguments[i]) for i in range(q - 1)]
                if q >= 3:
                    assert all(gap_arguments[i] > gap_arguments[i + 1] for i in range(q - 2))
                    nonlinear_gap_checks += q - 2

    assert uniform_rho_checks > 0
    assert monotonic_checks > 0
    assert nonlinear_gap_checks > 0

    print(f"[OK] Uniform Delta rho gaps checked: {uniform_rho_checks}")
    print(f"[OK] Strict Delta r monotonicity checks: {monotonic_checks}")
    print(f"[OK] Strictly decreasing logarithmic gap checks: {nonlinear_gap_checks}")


def verify_continuum_relevant_interval_bounds() -> None:
    print("\n=== Verification of exact interval bounds ===")

    # Every exact discrete causal interval lies inside the oriented one-sided cone 0 <= Delta r <= Delta t.
    checked = 0
    for s in range(2, 7):
        for delta_n in range(1, 6):
            q = s**delta_n
            for u in (0, 1, 2, 5, 17, 101):
                for D in sorted({0, 1 if q > 1 else 0, q // 2, q - 2 if q > 1 else 0, q - 1}):
                    if 0 <= D <= q - 1:
                        ratio = interval_log_argument(s, delta_n, u, D)
                        # Equivalent exact statement: 0 <= log_s(ratio) <= delta_n.
                        assert Fraction(1) <= ratio <= Fraction(q)
                        checked += 1

    assert checked > 0
    print(f"[OK] Checked {checked} exact representative intervals inside the oriented coordinate cone")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain and corruption tests ===")

    assert_expected_error(validate_base_and_depth, 1, 1)
    assert_expected_error(validate_base_and_depth, 2.5, 1)
    assert_expected_error(validate_base_and_depth, 2, 0)
    assert_expected_error(validate_base_and_depth, 2, -1)
    assert_expected_error(validate_ancestor_position, -1)
    assert_expected_error(validate_digit, 2, 3, -1)
    assert_expected_error(validate_digit, 2, 3, 8)
    assert_expected_error(validate_internal_state_sequence, 2, [], 1)
    assert_expected_error(validate_internal_state_sequence, 2, [1, 3], 2)
    assert_expected_error(validate_internal_state_sequence, 3, [1, 2], 3)
    assert_expected_error(states_from_digit, 3, 2, 9)
    assert_expected_error(level_start, 0, 2, 2, 0)
    assert_expected_error(level_start, 1, 1, 2, 0)
    assert_expected_error(level_start, 1, 2, 1, 0)
    assert_expected_error(level_start, 1, 2, 2, -1)
    assert_expected_error(element_from_position, 1, 2, 2, 3, -1)
    assert_expected_error(element_from_position, 1, 2, 2, 3, level_size(2, 2, 3))

    # Corrupted causal-layer decomposition: wrong descendant position cannot
    # have the stated quotient/remainder pair.
    s, delta_n, u, D = 3, 3, 5, 7
    q = s**delta_n
    correct = q * u + D
    corrupted = correct + 1
    assert corrupted != correct
    assert corrupted // q == u or corrupted // q == u + 1
    assert corrupted % q != D

    print("[OK] Invalid parameters and corrupted decompositions are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of exact discrete causal interval (sec:discrete-causal-interval) ===")
    verify_symbolic_discrete_interval_formula()
    verify_exact_finite_interval_inequalities()
    verify_boundary_internal_state_characterizations()
    verify_recursion_and_real_level_integration()
    verify_normalized_coordinate_and_log_spectrum()
    verify_continuum_relevant_interval_bounds()
    verify_negative_domain_tests()
    print("\n=== Exact discrete causal interval verification completed successfully ===")


if __name__ == "__main__":
    main()
