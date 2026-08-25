"""
VERIFICATION of Section 4: Causal cones in (t,r) coordinates
(sec:causal-cones-tr).

This verification block checks the new mathematical content of
4_causal_cones_tr.tex: the causal-cone inequality in logarithmic coordinates,
the exact right and left boundary trajectories, the two asymptotic regimes of
the left boundary, the integer interval structure of a depth-ell causal slice,
shift invariance, and the guard that the speed inequality is necessary for
descendants but not a converse causality test.

Verified content
----------------
1. For a depth-ell descendant of an ancestor with positional coordinate u,

       u_{n+\ell}(y) = s^ell u + D,       0 <= D <= s^ell - 1.

2. The logarithmic coordinate difference is

       Delta r = log_s((s^ell u + D + 1)/(u + 1)).

3. The causal-cone bound Delta r <= ell is exactly controlled by

       s^ell(u+1) - (s^ell u + D + 1) = s^ell - D - 1 >= 0.

4. The lower bound Delta r >= 0 is exactly controlled by

       (s^ell u + D + 1) - (u + 1) = (s^ell - 1)u + D >= 0.

5. Equality Delta r = ell occurs iff D = s^ell - 1, equivalently all internal
   states in the depth-ell extension are s.

6. The left boundary D = 0, equivalently all internal states are 1, satisfies

       Delta r = ell + log_s(u+s^{-ell}) - log_s(u+1).

7. Left-boundary asymptotics:
    - fixed ell and u -> infinity: Delta r -> ell;
    - fixed u > 0 and ell -> infinity: Delta r - ell generally has a nonzero
     limit, while Delta r/ell -> 1;
    - u = 0 on the left boundary gives Delta r = 0 for all ell.

8. The depth-ell causal slice is the exact integer interval

       [s^ell u, s^ell u + s^ell - 1].

9. The coordinate description is invariant under shifts of the initial interval.

10. The speed inequality is not used as a converse criterion: a future-level
    non-descendant can satisfy Delta r <= Delta t.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isclose, log
from typing import Callable, Sequence

from sympy import limit, log as slog, oo, simplify, symbols


@dataclass(frozen=True)
class Model:
    m: int
    k: int
    s: int

    def __post_init__(self) -> None:
        if not isinstance(self.m, int) or not isinstance(self.k, int) or not isinstance(self.s, int):
            raise TypeError("m, k, and s must be integers")
        if self.m < 1:
            raise ValueError("m must be a positive integer")
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if self.s < 2:
            raise ValueError("s must be at least 2")

    @property
    def c(self) -> int:
        return (self.s - 1) * self.m + 1 - self.k

    def level_start(self, n: int) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError("level index must be a nonnegative integer")
        numerator = self.k * (self.s**n - 1)
        denominator = self.s - 1
        if numerator % denominator != 0:
            raise ArithmeticError("level-start formula did not give an integer")
        return self.m + numerator // denominator

    def level_size(self, n: int) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError("level index must be a nonnegative integer")
        return self.k * self.s**n

    def level_end(self, n: int) -> int:
        return self.level_start(n) + self.level_size(n) - 1

    def contains_level_point(self, x: int, n: int) -> bool:
        return self.level_start(n) <= x <= self.level_end(n)

    def positional_coordinate(self, x: int, n: int) -> int:
        if not self.contains_level_point(x, n):
            raise ValueError("x does not belong to the requested level")
        return x - self.level_start(n)

    def point_from_position(self, u: int, n: int) -> int:
        if not isinstance(u, int):
            raise TypeError("u must be an integer")
        if not (0 <= u < self.level_size(n)):
            raise ValueError("u is outside the positional range of level n")
        return self.level_start(n) + u

    def F(self, x: int, sigma: int) -> int:
        if sigma < 1 or sigma > self.s:
            raise ValueError("internal state is outside S")
        return self.s * x + sigma - self.c

    def shifted(self, delta: int) -> "Model":
        if not isinstance(delta, int) or self.m + delta < 1:
            raise ValueError("shift must keep the initial interval inside positive integers")
        return Model(m=self.m + delta, k=self.k, s=self.s)


def validate_depth(depth: int) -> None:
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("depth ell must be a positive integer")


def validate_state_sequence(states: Sequence[int], s: int, expected_depth: int | None = None) -> tuple[int, ...]:
    if not isinstance(states, tuple):
        raise TypeError("internal-state sequence must be a tuple")
    if expected_depth is not None and len(states) != expected_depth:
        raise ValueError("internal-state sequence has the wrong depth")
    if len(states) == 0:
        raise ValueError("positive-depth extension cannot be empty")
    for sigma in states:
        if not isinstance(sigma, int) or sigma < 1 or sigma > s:
            raise ValueError("internal state is outside S")
    return states


def digit_from_states(states: Sequence[int], s: int) -> int:
    checked = validate_state_sequence(tuple(states), s)
    depth = len(checked)
    return sum((sigma - 1) * s ** (depth - j - 1) for j, sigma in enumerate(checked))


def states_from_digit(D: int, s: int, depth: int) -> tuple[int, ...]:
    validate_depth(depth)
    if not isinstance(D, int) or not (0 <= D <= s**depth - 1):
        raise ValueError("digit coordinate is outside the depth-ell range")
    remaining = D
    out: list[int] = []
    for power in range(depth - 1, -1, -1):
        base_power = s**power
        digit = remaining // base_power
        remaining %= base_power
        out.append(digit + 1)
    return validate_state_sequence(tuple(out), s, expected_depth=depth)


def descendant_position_from_digit(u: int, D: int, s: int, depth: int) -> int:
    validate_depth(depth)
    if not isinstance(u, int) or u < 0:
        raise ValueError("ancestor coordinate must be a nonnegative integer")
    if not isinstance(D, int) or not (0 <= D <= s**depth - 1):
        raise ValueError("digit coordinate is outside the depth-ell range")
    return s**depth * u + D


def descendant_position_from_states(u: int, states: Sequence[int], s: int) -> int:
    checked = validate_state_sequence(tuple(states), s)
    return descendant_position_from_digit(u, digit_from_states(checked, s), s, len(checked))


def iterate_F(model: Model, root_x: int, states: Sequence[int]) -> int:
    checked = validate_state_sequence(tuple(states), model.s)
    x = root_x
    for sigma in checked:
        x = model.F(x, sigma)
    return x


def log_argument_for_delta_r(u: int, D: int, s: int, depth: int) -> Fraction:
    descendant_u = descendant_position_from_digit(u, D, s, depth)
    return Fraction(descendant_u + 1, u + 1)


def delta_r_float(u: int, D: int, s: int, depth: int) -> float:
    arg = log_argument_for_delta_r(u, D, s, depth)
    return log(arg.numerator / arg.denominator, s)


def upper_speed_gap(D: int, s: int, depth: int) -> int:
    return s**depth - 1 - D


def lower_speed_gap(u: int, D: int, s: int, depth: int) -> int:
    return (s**depth - 1) * u + D


def causal_slice_positions(u: int, s: int, depth: int) -> list[int]:
    validate_depth(depth)
    if u < 0:
        raise ValueError("ancestor coordinate must be nonnegative")
    return [s**depth * u + D for D in range(s**depth)]


def expect_raises(expected_exception: type[BaseException], fn: Callable[[], object]) -> None:
    try:
        fn()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} was not raised")


def verify_symbolic_cone_identities() -> None:
    print("\n=== Symbolic verification of the causal-cone identities ===")

    u, D, s, K = symbols("u D s K", positive=True, integer=True)

    descendant_u = s**K * u + D
    upper_gap = s**K * (u + 1) - (descendant_u + 1)
    lower_gap = (descendant_u + 1) - (u + 1)

    assert simplify(upper_gap - (s**K - D - 1)) == 0
    assert simplify(lower_gap - ((s**K - 1) * u + D)) == 0

    right_boundary_arg = (s**K * u + (s**K - 1) + 1) / (u + 1)
    assert simplify(right_boundary_arg - s**K) == 0

    left_boundary_arg = (s**K * u + 1) / (u + 1)
    expected_left_arg = s**K * (u + s**(-K)) / (u + 1)
    assert simplify(left_boundary_arg - expected_left_arg) == 0

    width_ratio = (s**K * (u + 1)) / (s**K * u + 1)
    expected_width_ratio = (u + 1) / (u + s**(-K))
    assert simplify(width_ratio - expected_width_ratio) == 0

    print("[OK] Upper cone inequality reduces exactly to D <= s^ell - 1")
    print("[OK] Lower cone inequality reduces exactly to (s^ell - 1)u + D >= 0")
    print("[OK] Boundary and width formulas are symbolically consistent")


def verify_symbolic_asymptotic_regimes() -> None:
    print("\n=== Symbolic verification of asymptotic regimes ===")

    U, q, S = symbols("U q S", positive=True)
    correction = slog((U + q) / (U + 1)) / slog(S)

    assert simplify(limit(correction, U, oo)) == 0
    assert simplify(limit(correction, q, 0, dir="+") - slog(U / (U + 1)) / slog(S)) == 0

    K, C = symbols("K C", positive=True)
    assert simplify(limit((K + C) / K, K, oo) - 1) == 0

    print("[OK] Fixed-depth left boundary tends to Delta r = ell as u -> infinity")
    print("[OK] Fixed-u left boundary has a bounded correction, hence Delta r/ell -> 1")
    print("[OK] The nonzero correction for fixed u > 0 is explicitly preserved")


def verify_exact_finite_cone_slices() -> None:
    print("\n=== Exact finite verification of causal slices ===")

    checked_points = 0
    right_cases = 0
    left_cases = 0

    for s in range(2, 6):
        for depth in range(1, 5):
            for u in range(0, 22):
                slice_positions = causal_slice_positions(u, s, depth)
                assert slice_positions[0] == s**depth * u
                assert slice_positions[-1] == s**depth * u + s**depth - 1
                assert len(slice_positions) == s**depth
                assert slice_positions == list(range(slice_positions[0], slice_positions[-1] + 1))

                for D in range(s**depth):
                    checked_points += 1
                    states = states_from_digit(D, s, depth)
                    descendant_u = descendant_position_from_digit(u, D, s, depth)

                    assert digit_from_states(states, s) == D
                    assert descendant_position_from_states(u, states, s) == descendant_u
                    assert descendant_u == slice_positions[D]

                    assert upper_speed_gap(D, s, depth) >= 0
                    assert lower_speed_gap(u, D, s, depth) >= 0

                    if D == s**depth - 1:
                        right_cases += 1
                        assert states == tuple([s] * depth)
                        assert upper_speed_gap(D, s, depth) == 0
                        assert isclose(delta_r_float(u, D, s, depth), float(depth), rel_tol=1e-13, abs_tol=1e-13)
                    else:
                        assert upper_speed_gap(D, s, depth) > 0
                        assert delta_r_float(u, D, s, depth) < depth

                    if D == 0:
                        left_cases += 1
                        assert states == tuple([1] * depth)
                        if u == 0:
                            assert lower_speed_gap(u, D, s, depth) == 0
                            assert isclose(delta_r_float(u, D, s, depth), 0.0, rel_tol=0.0, abs_tol=1e-13)
                        else:
                            assert lower_speed_gap(u, D, s, depth) > 0
                            assert 0.0 < delta_r_float(u, D, s, depth) < depth
                    else:
                        assert lower_speed_gap(u, D, s, depth) > 0

    print(f"[OK] Checked {checked_points} exact descendant coordinates")
    print(f"[OK] Right-boundary cases: {right_cases}; left-boundary cases: {left_cases}")


def verify_integration_with_level_model_and_shift_invariance() -> None:
    print("\n=== Integration with levels, F, and shift invariance ===")

    checked_paths = 0

    models = [
        Model(m=1, k=2, s=2),
        Model(m=2, k=3, s=2),
        Model(m=3, k=2, s=3),
        Model(m=5, k=4, s=4),
    ]

    for model in models:
        for n in range(0, 4):
            samples = sorted({0, 1, model.level_size(n) // 2, model.level_size(n) - 1})
            for u in samples:
                x = model.point_from_position(u, n)
                assert model.positional_coordinate(x, n) == u

                for depth in range(1, 4):
                    for states in product(range(1, model.s + 1), repeat=depth):
                        D = digit_from_states(states, model.s)
                        expected_u = descendant_position_from_digit(u, D, model.s, depth)

                        y_by_F = iterate_F(model, x, states)
                        y_by_position = model.point_from_position(expected_u, n + depth)
                        assert y_by_F == y_by_position
                        assert model.positional_coordinate(y_by_F, n + depth) == expected_u

                        shifted = model.shifted(5)
                        shifted_x = x + 5
                        shifted_y = iterate_F(shifted, shifted_x, states)

                        assert shifted_y == y_by_F + 5
                        assert shifted.positional_coordinate(shifted_x, n) == u
                        assert shifted.positional_coordinate(shifted_y, n + depth) == expected_u

                        checked_paths += 1

    print(f"[OK] Checked {checked_paths} generated paths and shifted copies")


def verify_boundary_width_and_ratio_regimes() -> None:
    print("\n=== Boundary width and ratio-regime verification ===")

    for s in range(2, 7):
        for depth in range(1, 6):
            previous_excess: Fraction | None = None
            for u in (1, 2, 5, 10, 50, 200, 1000):
                # right_arg / left_arg - 1 = (s^depth - 1)/(s^depth u + 1)
                excess = Fraction(s**depth - 1, s**depth * u + 1)
                if previous_excess is not None:
                    assert excess < previous_excess
                previous_excess = excess
            assert previous_excess is not None
            assert previous_excess < Fraction(1, 100)

    for s in range(2, 6):
        for u in range(1, 8):
            target = Fraction(u + 1, u)
            previous_error: Fraction | None = None
            for depth in (2, 4, 8, 12):
                width_ratio = Fraction(s**depth * (u + 1), s**depth * u + 1)
                error = abs(width_ratio - target)
                if previous_error is not None:
                    assert error < previous_error
                previous_error = error
            assert previous_error is not None
            assert previous_error < Fraction(1, 50)

            for depth in (80, 120, 160):
                ratio = delta_r_float(u, 0, s, depth) / depth
                assert 0.95 < ratio < 1.0

    print("[OK] Fixed-depth logarithmic cone width shrinks as u increases")
    print("[OK] Fixed-u left-boundary correction remains bounded, so Delta r/ell tends to 1")
    print("[OK] The u=0 left boundary stays exactly at Delta r = 0")


def verify_speed_bound_is_not_a_converse_criterion() -> None:
    print("\n=== Negative guard: speed bound is not a converse causality test ===")

    model = Model(m=1, k=2, s=2)
    n = 2
    depth = 2
    u_x = 2
    x = model.point_from_position(u_x, n)

    descendant_start = model.s**depth * u_x
    descendant_end = descendant_start + model.s**depth - 1

    u_y = descendant_start - 1
    y = model.point_from_position(u_y, n + depth)

    assert model.contains_level_point(x, n)
    assert model.contains_level_point(y, n + depth)
    assert not (descendant_start <= u_y <= descendant_end)

    delta_r = log((u_y + 1) / (u_x + 1), model.s)
    assert delta_r <= depth

    print("[OK] A future-level non-descendant can still satisfy Delta r <= Delta t")
    print("[OK] The theorem is verified as a necessary bound, not as an equivalence")


def verify_invalid_inputs_are_rejected() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(ValueError, lambda: Model(m=0, k=2, s=2))
    expect_raises(ValueError, lambda: Model(m=1, k=1, s=2))
    expect_raises(ValueError, lambda: Model(m=1, k=2, s=1))
    expect_raises(TypeError, lambda: Model(m=1, k=2, s=2.5))  # type: ignore[arg-type]

    expect_raises(ValueError, lambda: validate_depth(0))
    expect_raises(ValueError, lambda: states_from_digit(-1, 2, 3))
    expect_raises(ValueError, lambda: states_from_digit(8, 2, 3))
    expect_raises(ValueError, lambda: descendant_position_from_digit(0, 4, 2, 2))
    expect_raises(ValueError, lambda: descendant_position_from_digit(-1, 0, 2, 2))
    expect_raises(TypeError, lambda: validate_state_sequence([1, 2], 2))  # type: ignore[arg-type]
    expect_raises(ValueError, lambda: validate_state_sequence(tuple(), 2))
    expect_raises(ValueError, lambda: validate_state_sequence((1, 3), 2))
    expect_raises(ValueError, lambda: validate_state_sequence((1, 2), 2, expected_depth=3))

    model = Model(m=1, k=2, s=2)
    expect_raises(ValueError, lambda: model.positional_coordinate(1000, 0))
    expect_raises(ValueError, lambda: model.point_from_position(-1, 0))
    expect_raises(ValueError, lambda: model.point_from_position(model.level_size(2), 2))
    expect_raises(ValueError, lambda: model.F(1, 0))
    expect_raises(ValueError, lambda: model.F(1, 3))
    expect_raises(ValueError, lambda: model.shifted(-10))

    print("[OK] Invalid parameters, depths, coordinates, and internal states are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of causal cones in (t,r) coordinates (sec:causal-cones-tr) ===")
    verify_symbolic_cone_identities()
    verify_symbolic_asymptotic_regimes()
    verify_exact_finite_cone_slices()
    verify_integration_with_level_model_and_shift_invariance()
    verify_boundary_width_and_ratio_regimes()
    verify_speed_bound_is_not_a_converse_criterion()
    verify_invalid_inputs_are_rejected()
    print("\n=== Causal-cone (t,r) verification completed successfully ===")


if __name__ == "__main__":
    main()
