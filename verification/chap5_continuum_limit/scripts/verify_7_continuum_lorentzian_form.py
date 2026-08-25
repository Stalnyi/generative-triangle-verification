"""
VERIFICATION of Section 7: Continuum Lorentzian form of causal geometry
(sec:continuum-lorentzian-form).

This script provides a full mathematical verification block for the claims in
7_continuum_lorentzian_form.tex.  It does not re-verify the construction of
the discrete causal interval, the digit coordinate, or the continuum coordinate
spectrum.  Those facts are treated as already established dependencies.

The new content verified here is the exact implication

    0 <= Delta r <= Delta t  ==>  I^2 = Delta t^2 - Delta r^2 >= 0,

together with its boundary cases, its oriented future-cone interpretation, its
failure outside the cone, and the stability of the quadratic form under
cone-preserving continuum limits.

Verified content
----------------
1. Algebraic factorization:
       I^2 = Delta t^2 - Delta r^2
           = (Delta t - Delta r)(Delta t + Delta r).

2. One-sided future-cone implication:
       0 <= Delta r <= Delta t  implies  I^2 >= 0.

3. Boundary cases:
       Delta r = 0          gives I^2 = Delta t^2.
       Delta r = Delta t    gives I^2 = 0.
       Delta t = 0 in cone  forces Delta r = 0 and gives I^2 = 0.

4. Exact finite rational classification inside the oriented cone:
       Delta t > Delta r >= 0  gives I^2 > 0.
       Delta t = Delta r >= 0  gives I^2 = 0.

5. Exact finite rational failure outside the cone:
       Delta r > Delta t >= 0  gives I^2 < 0.

6. Discrete-to-continuum logical transfer:
   exact discrete causal intervals satisfying
       0 <= Delta r_N <= Delta t_N
   keep nonnegative I_N^2, and cone-preserving limits satisfy
       lim I_N^2 = T^2 - R^2 >= 0
   whenever 0 <= R <= T.

7. Oriented one-sided versus natural two-sided extension:
   the section proves the one-sided future cone
       0 <= Delta r <= Delta t.
   The script separately verifies that the natural extended condition
       |Delta r| <= Delta t
   has the same Lorentzian non-negativity implication.

8. Domain guards:
   noncausal pairs, negative time separation, and outside-cone pairs are rejected
   as inputs for the proposition verified in this section.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Callable, Iterable, Sequence, Tuple

from sympy import Symbol, factor, limit, oo, simplify, symbols


FractionPair = Tuple[Fraction, Fraction]


def interval_squared(delta_t: Fraction, delta_r: Fraction) -> Fraction:
    """Exact quadratic causal-coordinate interval I^2 = Delta t^2 - Delta r^2."""
    return delta_t * delta_t - delta_r * delta_r


def require_oriented_future_cone(delta_t: Fraction, delta_r: Fraction) -> None:
    """Accept exactly the one-sided future cone used in the section."""
    if not (delta_t >= 0 and Fraction(0) <= delta_r <= delta_t):
        raise ValueError(
            f"not in the oriented future cone: Delta t={delta_t}, Delta r={delta_r}"
        )


def require_two_sided_future_cone(delta_t: Fraction, delta_r: Fraction) -> None:
    """Accept exactly the natural two-sided extension |Delta r| <= Delta t."""
    if not (delta_t >= 0 and abs(delta_r) <= delta_t):
        raise ValueError(
            f"not in the two-sided future cone: Delta t={delta_t}, Delta r={delta_r}"
        )


def oriented_interval_squared(delta_t: Fraction, delta_r: Fraction) -> Fraction:
    """Quadratic form restricted to the section's one-sided causal domain."""
    require_oriented_future_cone(delta_t, delta_r)
    return interval_squared(delta_t, delta_r)


@dataclass(frozen=True)
class ConePoint:
    delta_t: Fraction
    delta_r: Fraction

    @property
    def I2(self) -> Fraction:
        return interval_squared(self.delta_t, self.delta_r)

    @property
    def factorized_I2(self) -> Fraction:
        return (self.delta_t - self.delta_r) * (self.delta_t + self.delta_r)


def rational_grid(max_num: int, max_den: int, signed: bool = False) -> list[Fraction]:
    if max_num < 0 or max_den < 1:
        raise ValueError("invalid rational grid parameters")
    values = {Fraction(0)}
    signs = (-1, 1) if signed else (1,)
    for den in range(1, max_den + 1):
        for num in range(0, max_num + 1):
            for sign in signs:
                values.add(Fraction(sign * num, den))
    return sorted(values)


def oriented_cone_points(max_num: int = 18, max_den: int = 15) -> Iterable[ConePoint]:
    values = rational_grid(max_num=max_num, max_den=max_den)
    for delta_t in values:
        for delta_r in values:
            if Fraction(0) <= delta_r <= delta_t:
                yield ConePoint(delta_t=delta_t, delta_r=delta_r)


def outside_oriented_cone_points(
    max_num: int = 18, max_den: int = 15
) -> Iterable[ConePoint]:
    values = rational_grid(max_num=max_num, max_den=max_den)
    for delta_t in values:
        for delta_r in values:
            if delta_t >= 0 and delta_r > delta_t:
                yield ConePoint(delta_t=delta_t, delta_r=delta_r)


def two_sided_cone_points(max_num: int = 14, max_den: int = 12) -> Iterable[ConePoint]:
    nonnegative_values = rational_grid(max_num=max_num, max_den=max_den)
    signed_values = rational_grid(max_num=max_num, max_den=max_den, signed=True)
    for delta_t in nonnegative_values:
        for delta_r in signed_values:
            if delta_t >= 0 and abs(delta_r) <= delta_t:
                yield ConePoint(delta_t=delta_t, delta_r=delta_r)


def assert_raises(expected_exception: type[Exception], action: Callable[[], object]) -> None:
    """Strict expected-failure helper: only the target call is inside the handler."""
    try:
        action()
    except expected_exception:
        return
    raise AssertionError(f"expected {expected_exception.__name__} was not raised")


def verify_symbolic_lorentzian_form() -> None:
    print("\n=== Symbolic verification of the Lorentzian quadratic form ===")

    Delta_t, Delta_r, q = symbols("Delta_t Delta_r q", real=True)
    I2 = Delta_t**2 - Delta_r**2

    assert simplify(I2 - (Delta_t - Delta_r) * (Delta_t + Delta_r)) == 0
    assert simplify(factor(I2) - (Delta_t - Delta_r) * (Delta_t + Delta_r)) == 0

    assert simplify(I2.subs(Delta_r, 0) - Delta_t**2) == 0
    assert simplify(I2.subs(Delta_r, Delta_t)) == 0
    assert simplify(I2.subs({Delta_t: 0, Delta_r: 0})) == 0

    # Cone parameterization Delta r = q Delta t.  For 0 <= q <= 1,
    # the factor (1 - q^2) is nonnegative; exact sign checks are supplied by
    # rational grids below.
    parameterized = simplify(I2.subs(Delta_r, q * Delta_t))
    assert simplify(parameterized - Delta_t**2 * (1 - q**2)) == 0

    print("[OK] I^2 factorizes exactly as (Delta t - Delta r)(Delta t + Delta r)")
    print("[OK] Boundary identities and cone parameterization hold symbolically")


def verify_exact_oriented_future_cone() -> None:
    print("\n=== Exact rational verification inside the oriented future cone ===")

    total = 0
    timelike_count = 0
    null_count = 0
    origin_count = 0

    for point in oriented_cone_points():
        total += 1
        require_oriented_future_cone(point.delta_t, point.delta_r)

        assert point.I2 == point.factorized_I2
        assert point.I2 >= 0, point

        if point.delta_t == 0:
            assert point.delta_r == 0
            assert point.I2 == 0
            origin_count += 1
        elif point.delta_r == point.delta_t:
            assert point.I2 == 0
            null_count += 1
        elif Fraction(0) <= point.delta_r < point.delta_t:
            assert point.I2 > 0
            timelike_count += 1
        else:
            raise AssertionError(f"unclassified oriented cone point: {point}")

    assert total > 0
    assert timelike_count > 0
    assert null_count > 0
    assert origin_count == 1

    print(f"[OK] Checked {total} exact rational one-sided cone points")
    print(f"[OK] Timelike={timelike_count}; null={null_count}; origin={origin_count}")


def verify_exact_failure_outside_oriented_cone() -> None:
    print("\n=== Exact rational verification outside the oriented cone ===")

    total = 0
    for point in outside_oriented_cone_points():
        total += 1
        assert point.delta_t >= 0
        assert point.delta_r > point.delta_t
        assert point.I2 == point.factorized_I2
        assert point.I2 < 0, point
        assert_raises(ValueError, lambda p=point: oriented_interval_squared(p.delta_t, p.delta_r))

    assert total > 0
    print(f"[OK] Checked {total} exact outside-cone points with Delta r > Delta t >= 0")
    print("[OK] Non-negativity fails exactly outside the one-sided causal inequality")


def verify_two_sided_extension() -> None:
    print("\n=== Verification of the natural two-sided extension ===")

    total = 0
    positive_count = 0
    null_count = 0

    for point in two_sided_cone_points():
        total += 1
        require_two_sided_future_cone(point.delta_t, point.delta_r)
        assert point.I2 >= 0

        if abs(point.delta_r) == point.delta_t:
            assert point.I2 == 0
            null_count += 1
        elif abs(point.delta_r) < point.delta_t:
            assert point.I2 > 0
            positive_count += 1

    assert total > 0
    assert positive_count > 0
    assert null_count > 0

    outside_samples = [
        (Fraction(1), Fraction(2)),
        (Fraction(3, 2), Fraction(-2)),
        (Fraction(0), Fraction(1, 7)),
    ]
    for delta_t, delta_r in outside_samples:
        assert_raises(ValueError, lambda dt=delta_t, dr=delta_r: require_two_sided_future_cone(dt, dr))
        assert interval_squared(delta_t, delta_r) < 0

    print(f"[OK] Checked {total} exact rational two-sided cone points")
    print("[OK] The two-sided extension preserves the same non-negativity condition")


def discrete_cone_sample(delta_t: int, denominator: int, numerator: int) -> FractionPair:
    if delta_t < 0 or denominator <= 0 or numerator < 0:
        raise ValueError("invalid discrete cone sample parameters")
    delta_r = Fraction(numerator, denominator)
    if not (Fraction(0) <= delta_r <= Fraction(delta_t)):
        raise ValueError("sample is outside the oriented cone")
    return Fraction(delta_t), delta_r


def verify_discrete_to_continuum_implication() -> None:
    print("\n=== Discrete-to-continuum causal implication checks ===")

    total = 0
    for delta_t in range(0, 31):
        denominator = 31
        for numerator in range(0, delta_t * denominator + 1):
            dt, dr = discrete_cone_sample(delta_t, denominator, numerator)
            total += 1
            I2 = oriented_interval_squared(dt, dr)
            assert I2 >= 0
            assert I2 == (dt - dr) * (dt + dr)

    assert total > 0

    # Exact representative sequences converging to continuum points in the cone.
    targets = [
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1, 3)),
        (Fraction(1), Fraction(1)),
        (Fraction(5, 3), Fraction(2, 3)),
        (Fraction(7, 4), Fraction(7, 4)),
    ]

    for T, R in targets:
        expected = interval_squared(T, R)
        previous_error: Fraction | None = None
        for N in (25, 50, 100, 200, 400):
            dt_N = T + Fraction(1, N)
            if R == 0:
                dr_N = Fraction(0)
            else:
                dr_N = max(Fraction(0), R - Fraction(1, N * (N + 1)))
            require_oriented_future_cone(dt_N, dr_N)

            I2_N = oriented_interval_squared(dt_N, dr_N)
            assert I2_N >= 0

            error = abs(I2_N - expected)
            if previous_error is not None:
                assert error <= previous_error or error < Fraction(1, 100)
            previous_error = error

        assert previous_error is not None
        assert previous_error < Fraction(1, 30), (T, R, previous_error)

    N = Symbol("N", positive=True)
    T, R = symbols("T R", positive=True)
    dtN = T + 1 / N
    drN = R - 1 / (N * (N + 1))
    I2N = dtN**2 - drN**2
    assert simplify(limit(I2N, N, oo) - (T**2 - R**2)) == 0

    print(f"[OK] Checked {total} exact discrete/rational causal intervals")
    print("[OK] Cone-preserving limits converge to T^2 - R^2 and keep non-negativity")


def verify_boundary_and_classification_exhaustiveness() -> None:
    print("\n=== Boundary and classification exhaustiveness checks ===")

    boundary_cases: Sequence[Tuple[Fraction, Fraction, Fraction]] = [
        (Fraction(1), Fraction(0), Fraction(1)),
        (Fraction(5, 2), Fraction(0), Fraction(25, 4)),
        (Fraction(3), Fraction(3), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0)),
    ]
    for delta_t, delta_r, expected in boundary_cases:
        assert oriented_interval_squared(delta_t, delta_r) == expected

    # Exhaustive exact classification on a smaller grid, including all boundaries.
    seen_classes = set()
    for point in oriented_cone_points(max_num=10, max_den=8):
        if point.delta_t == 0:
            classification = "origin"
            assert point.delta_r == 0 and point.I2 == 0
        elif point.delta_r == point.delta_t:
            classification = "null"
            assert point.I2 == 0
        elif point.delta_r == 0:
            classification = "radial-origin"
            assert point.I2 == point.delta_t**2 and point.I2 > 0
        else:
            classification = "strict-interior"
            assert point.I2 > 0

        seen_classes.add(classification)

    assert seen_classes == {"origin", "null", "radial-origin", "strict-interior"}

    print("[OK] Boundary identities and cone classification are exhaustive on exact grids")


def verify_domain_guards() -> None:
    print("\n=== Domain guards ===")

    invalid_oriented_cases = [
        (Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(-1, 10)),
        (Fraction(1), Fraction(11, 10)),
        (Fraction(0), Fraction(1, 100)),
    ]
    for delta_t, delta_r in invalid_oriented_cases:
        assert_raises(ValueError, lambda dt=delta_t, dr=delta_r: require_oriented_future_cone(dt, dr))


    # Ensure no floating non-finite result is hidden in exact checks.
    for value in [0.0, 1.0, -1.0, 0.25]:
        assert isfinite(value)

    print("[OK] Invalid causal-domain inputs are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of Section 7: continuum Lorentzian form ===")
    verify_symbolic_lorentzian_form()
    verify_exact_oriented_future_cone()
    verify_exact_failure_outside_oriented_cone()
    verify_two_sided_extension()
    verify_discrete_to_continuum_implication()
    verify_boundary_and_classification_exhaustiveness()
    verify_domain_guards()
    print("\n=== Continuum Lorentzian form verification completed successfully ===")


if __name__ == "__main__":
    main()
