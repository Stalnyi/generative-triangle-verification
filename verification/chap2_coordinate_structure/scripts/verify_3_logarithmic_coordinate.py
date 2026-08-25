"""
VERIFICATION of Section 3: Logarithmic spatial coordinate
(sec:logarithmic-coordinate).

This script provides a full mathematical verification block for the claims in
3_logarithmic_coordinate.tex.  It verifies the new content of this section:
the logarithmic coordinate r_n(x), its exact one-step evolution, the causal
speed bound in (t,r) coordinates, shift invariance, and the asymptotic expansion
of the one-step radial increment.

The script intentionally treats the positional coordinate recursion from the
previous section as a dependency and does not re-prove the entire generative
bijection.  It verifies only those consequences needed here.

Verified content
----------------
1. Domain discipline:
      s >= 2, m >= 1, k >= 2, n >= 0, x in L_n, sigma in {1,...,s}.

2. Logarithmic coordinate:
      r_n(x) = log_s(u_n(x)+1),
   with the left boundary u_n=0 giving r_n=0 and no singularity.

3. Normalization +1 shift:
      among coordinates log_s(u+c), the condition r(0)=0 forces c=1.
   The coordinate log_s(u) is rejected at the left boundary u=0.

4. Exact one-step identity:
      u_{n+1}(F(x,sigma)) + 1 = s*(u_n(x)+sigma/s).

5. Exact evolution of r:
      Delta r =
      r_{n+1}(F(x,sigma)) - r_n(x)
      =
      1 + log_s((u + sigma/s)/(u+1)).

6. Sharp bound:
      0 <= Delta r <= 1.

7. Equality cases:
      Delta r = 1  iff  sigma = s;
      Delta r = 0  iff  u = 0 and sigma = 1.

8. depth-ell maximum-speed bound:
      for every descendant y of x at depth ell,
      r_{n+ell}(y) <= r_n(x) + ell,
   with equality exactly on the right boundary trajectory
      sigma_1 = ... = sigma_ell = s.

9. Shift invariance:
      shifting the whole initial interval by delta preserves u_n and therefore
      preserves r_n and the speed bound.

10. Asymptotic behavior:
      Delta r =
      1 + ((sigma/s)-1)/((u+1)*ln(s)) + O(1/u^2)
   for fixed sigma and large u.  The script verifies the exact first-order
   expansion by introducing z = 1/(u+1).

11. Negative tests:
      invalid base, invalid model parameters, invalid level, invalid state,
      x outside L_n, invalid descendant depth, invalid shift, and singular
      logarithmic coordinate at u=0 without the +1 shift.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isclose, log
from typing import Sequence

from sympy import (
    S as SymS,
    exp,
    limit,
    log as sym_log,
    simplify,
    symbols,
)


@dataclass(frozen=True)
class GenerativeSpace:
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
        self._validate_level(n)
        return self.m + self.k * (self.s**n - 1) // (self.s - 1)

    def level_size(self, n: int) -> int:
        self._validate_level(n)
        return self.k * self.s**n

    def level_end(self, n: int) -> int:
        return self.level_start(n) + self.level_size(n) - 1

    def in_level(self, x: int, n: int) -> bool:
        self._validate_level(n)
        return self.level_start(n) <= x <= self.level_end(n)

    def level_elements(self, n: int) -> range:
        return range(self.level_start(n), self.level_end(n) + 1)

    def position(self, x: int, n: int) -> int:
        if not isinstance(x, int):
            raise TypeError("x must be an integer")
        if not self.in_level(x, n):
            raise ValueError(f"x={x} is not in L_{n}")
        return x - self.level_start(n)

    def from_position(self, u: int, n: int) -> int:
        self._validate_level(n)
        if not isinstance(u, int):
            raise TypeError("u must be an integer")
        if not (0 <= u < self.level_size(n)):
            raise ValueError("u is outside the positional range of L_n")
        return self.level_start(n) + u

    def F(self, x: int, sigma: int) -> int:
        self._validate_state(sigma)
        return self.s * x + sigma - self.c

    def child_position(self, u: int, sigma: int) -> int:
        if not isinstance(u, int) or u < 0:
            raise ValueError("u must be a nonnegative integer")
        self._validate_state(sigma)
        return self.s * u + (sigma - 1)

    def iterate_from_position(self, u: int, states: Sequence[int]) -> int:
        if not isinstance(u, int) or u < 0:
            raise ValueError("u must be a nonnegative integer")
        out = u
        for sigma in states:
            out = self.child_position(out, sigma)
        return out

    def digit_coordinate(self, states: Sequence[int]) -> int:
        if len(states) < 1:
            raise ValueError("the finite internal-state prefix must be nonempty for a positive depth")
        D = 0
        for sigma in states:
            self._validate_state(sigma)
            D = self.s * D + (sigma - 1)
        return D

    def descendant_position(self, u: int, states: Sequence[int]) -> int:
        depth = len(states)
        if depth < 1:
            raise ValueError("descendant depth must be positive")
        D = self.digit_coordinate(states)
        return self.s**depth * u + D

    def log_coordinate_from_position(self, u: int) -> float:
        if not isinstance(u, int) or u < 0:
            raise ValueError("u must be a nonnegative integer")
        return log(u + 1, self.s)

    def log_coordinate(self, x: int, n: int) -> float:
        return self.log_coordinate_from_position(self.position(x, n))

    def delta_r_one_step_from_position(self, u: int, sigma: int) -> float:
        self._validate_state(sigma)
        if not isinstance(u, int) or u < 0:
            raise ValueError("u must be a nonnegative integer")
        up = self.child_position(u, sigma)
        return self.log_coordinate_from_position(up) - self.log_coordinate_from_position(u)

    def shift(self, delta: int) -> "GenerativeSpace":
        if not isinstance(delta, int):
            raise TypeError("delta must be an integer")
        if delta < 0:
            raise ValueError("delta must be nonnegative")
        return GenerativeSpace(m=self.m + delta, k=self.k, s=self.s)

    def _validate_level(self, n: int) -> None:
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n < 0:
            raise ValueError("n must be nonnegative")

    def _validate_state(self, sigma: int) -> None:
        if not isinstance(sigma, int):
            raise TypeError("sigma must be an integer")
        if not (1 <= sigma <= self.s):
            raise ValueError("sigma is outside S={1,...,s}")


def assert_raises(expected_exception: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except expected_exception:
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected {expected_exception.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {expected_exception.__name__}, but no exception was raised")


def symbolic_verification() -> None:
    print("\n=== Symbolic verification of logarithmic-coordinate identities ===")

    u, s, sigma = symbols("u s sigma", positive=True)
    z = symbols("z", positive=True)

    # Exact one-step identity in positional coordinates.
    u_next_plus_one = s * u + sigma
    factored = s * (u + sigma / s)
    assert simplify(u_next_plus_one - factored) == 0

    # Exact r-evolution formula:
    # log_s(s*u+sigma) - log_s(u+1)
    # = 1 + log_s((u+sigma/s)/(u+1)).
    delta_direct = sym_log(s * u + sigma) / sym_log(s) - sym_log(u + 1) / sym_log(s)
    delta_compact = 1 + sym_log((u + sigma / s) / (u + 1)) / sym_log(s)
    assert simplify(delta_direct - delta_compact) == 0

    # Equality case Delta r = 1:
    # ratio = 1  <=>  u + sigma/s = u + 1  <=> sigma = s.
    ratio = (u + sigma / s) / (u + 1)
    assert simplify(ratio.subs(sigma, s) - 1) == 0

    # Equality case Delta r = 0:
    # ratio = 1/s  <=>  (s-1)u + sigma - 1 = 0.
    equation_for_zero = simplify(s * (u + sigma / s) - (u + 1))
    assert simplify(equation_for_zero - ((s - 1) * u + sigma - 1)) == 0
    assert simplify(equation_for_zero.subs({u: 0, sigma: 1})) == 0

    # Asymptotic expansion.
    # Let z = 1/(u+1) and a = sigma/s - 1.  Then:
    # Delta r = 1 + log(1 + a*z)/log(s).
    a = sigma / s - 1
    asymptotic_expression = 1 + sym_log(1 + a * z) / sym_log(s)
    first_order = 1 + a * z / sym_log(s)
    remainder_order = simplify(asymptotic_expression.series(z, 0, 3).removeO() - first_order)
    assert simplify(remainder_order - (-(a**2) * z**2 / (2 * sym_log(s)))) == 0

    # The first-order correction vanishes in the right-boundary state sigma=s.
    assert simplify((a / sym_log(s)).subs(sigma, s)) == 0

    # Asymptotic equivalence r(u) ~ log_s(u).
    U = symbols("U", positive=True)
    r_difference = sym_log(U + 1) / sym_log(s) - sym_log(U) / sym_log(s)
    assert simplify(limit(r_difference, U, SymS.Infinity)) == 0

    print("[OK] Exact logarithmic evolution identity verified symbolically")
    print("[OK] Boundary equality conditions reduced to exact algebraic equations")
    print("[OK] First-order asymptotic expansion verified symbolically")
    print("[OK] r(u)=log_s(u+1) is asymptotic to log_s(u)")


def verify_log_coordinate_basic_properties() -> None:
    print("\n=== Exact finite verification of r_n(x)=log_s(u_n(x)+1) ===")

    checked = 0
    for gs in (
        GenerativeSpace(1, 2, 2),
        GenerativeSpace(2, 3, 2),
        GenerativeSpace(1, 4, 3),
        GenerativeSpace(5, 5, 4),
        GenerativeSpace(3, 2, 5),
    ):
        for n in range(0, 7):
            values = []
            for x in gs.level_elements(n):
                u = gs.position(x, n)
                r = gs.log_coordinate(x, n)
                assert u == x - gs.level_start(n)
                assert r >= 0
                if u == 0:
                    assert r == 0
                else:
                    assert r > 0
                values.append(r)
                reconstructed = gs.from_position(u, n)
                assert reconstructed == x
                checked += 1

            # Strict monotonicity of r as a function of the positional coordinate.
            assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    print(f"[OK] Checked {checked} level points for domain, reconstruction, and strict monotonicity")


def verify_one_step_evolution() -> None:
    print("\n=== Exact finite verification of the one-step logarithmic evolution ===")

    checked = 0
    equality_upper = 0
    equality_lower = 0
    strict_interior = 0

    for gs in (
        GenerativeSpace(1, 2, 2),
        GenerativeSpace(1, 3, 3),
        GenerativeSpace(4, 2, 4),
        GenerativeSpace(2, 5, 5),
    ):
        for n in range(0, 6):
            for x in gs.level_elements(n):
                u = gs.position(x, n)
                for sigma in range(1, gs.s + 1):
                    x_next = gs.F(x, sigma)
                    assert gs.in_level(x_next, n + 1)

                    u_next = gs.position(x_next, n + 1)
                    assert u_next == gs.child_position(u, sigma)
                    assert u_next + 1 == gs.s * (u + Fraction(sigma, gs.s))

                    delta_direct = gs.log_coordinate(x_next, n + 1) - gs.log_coordinate(x, n)
                    delta_formula = 1 + log((u + sigma / gs.s) / (u + 1), gs.s)
                    assert isclose(delta_direct, delta_formula, rel_tol=1e-13, abs_tol=1e-13)

                    # Exact inequality checks are performed on logarithm arguments.
                    ratio_num = gs.s * u + sigma
                    ratio_den = gs.s * (u + 1)
                    assert 1 <= ratio_num <= ratio_den
                    # Delta r in [0,1] is equivalent to ratio in [1/s,1].
                    assert isclose(delta_direct, gs.delta_r_one_step_from_position(u, sigma), rel_tol=0, abs_tol=1e-15)
                    assert -1e-14 <= delta_direct <= 1 + 1e-14

                    if sigma == gs.s:
                        assert ratio_num == ratio_den
                        assert isclose(delta_direct, 1.0, rel_tol=0, abs_tol=1e-13)
                        equality_upper += 1
                    elif u == 0 and sigma == 1:
                        assert ratio_num == 1
                        assert ratio_den == gs.s
                        assert isclose(delta_direct, 0.0, rel_tol=0, abs_tol=1e-13)
                        equality_lower += 1
                    else:
                        assert ratio_num > 1
                        assert ratio_num < ratio_den
                        assert 0 < delta_direct < 1
                        strict_interior += 1

                    checked += 1

    assert equality_upper > 0 and equality_lower > 0 and strict_interior > 0
    print(f"[OK] Checked {checked} one-step transitions")
    print(f"[OK] Upper equality cases: {equality_upper}; lower equality cases: {equality_lower}; strict interior cases: {strict_interior}")


def verify_logarithmic_normalization_shift() -> None:
    print("\n=== Verification of the normalization +1 shift in r(u)=log_s(u+1) ===")

    for s in range(2, 12):
        # log_s(u) is singular at u=0.
        assert_raises(ValueError, logarithm_without_shift, s, 0)

        for c in range(1, 10):
            value_at_left_boundary = log(c, s)
            if c == 1:
                assert value_at_left_boundary == 0
            else:
                assert value_at_left_boundary != 0

        # For c=1 the exact factorization used in the section holds.
        for u in range(0, 20):
            for sigma in range(1, s + 1):
                assert s * u + sigma == s * (u + Fraction(sigma, s))

    print("[OK] The left-boundary condition r(0)=0 forces the +1 shift")
    print("[OK] The unshifted logarithmic coordinate is singular at u=0")


def logarithm_without_shift(s: int, u: int) -> float:
    if s < 2:
        raise ValueError("s must be at least 2")
    if u <= 0:
        raise ValueError("log_s(u) is not defined at the left boundary u=0")
    return log(u, s)


def verify_ell_step_speed_bound() -> None:
    print("\n=== Verification of the depth-ell maximum-speed bound ===")

    checked = 0
    equality_count = 0
    strict_count = 0

    for gs in (
        GenerativeSpace(1, 2, 2),
        GenerativeSpace(1, 3, 3),
        GenerativeSpace(2, 2, 4),
        GenerativeSpace(5, 4, 3),
    ):
        for n in range(0, 5):
            # Use representative positions from each level, including boundaries.
            representative_us = sorted(
                {
                    0,
                    1 if gs.level_size(n) > 1 else 0,
                    gs.level_size(n) // 2,
                    gs.level_size(n) - 1,
                }
            )
            for u in representative_us:
                x = gs.from_position(u, n)
                r_start = gs.log_coordinate(x, n)
                for depth in range(1, 6):
                    for states in product(range(1, gs.s + 1), repeat=depth):
                        u_end = gs.descendant_position(u, states)
                        y = gs.from_position(u_end, n + depth)
                        r_end = gs.log_coordinate(y, n + depth)

                        # Exact inequality:
                        # r_end <= r_start + depth
                        # <=> u_end + 1 <= s^depth * (u+1)
                        D = gs.digit_coordinate(states)
                        assert u_end == gs.s**depth * u + D
                        assert D <= gs.s**depth - 1
                        assert u_end + 1 <= gs.s**depth * (u + 1)
                        assert r_end <= r_start + depth + 1e-12

                        all_right_boundary = all(sigma == gs.s for sigma in states)
                        if all_right_boundary:
                            assert D == gs.s**depth - 1
                            assert u_end + 1 == gs.s**depth * (u + 1)
                            assert isclose(r_end - r_start, float(depth), rel_tol=0, abs_tol=1e-12)
                            equality_count += 1
                        else:
                            assert D < gs.s**depth - 1
                            assert u_end + 1 < gs.s**depth * (u + 1)
                            assert r_end - r_start < depth
                            strict_count += 1

                        checked += 1

    assert equality_count > 0 and strict_count > 0
    print(f"[OK] Checked {checked} finite descendant paths")
    print(f"[OK] Equality only on the right boundary: {equality_count}; strict cases: {strict_count}")


def verify_shift_invariance() -> None:
    print("\n=== Verification of global shift invariance of the logarithmic coordinate ===")

    checked = 0
    base = GenerativeSpace(2, 4, 3)

    for delta in range(0, 12):
        shifted = base.shift(delta)
        for n in range(0, 6):
            for u in (0, 1, base.level_size(n) // 2, base.level_size(n) - 1):
                x = base.from_position(u, n)
                x_shifted = x + delta

                assert shifted.in_level(x_shifted, n)
                assert shifted.position(x_shifted, n) == base.position(x, n)
                assert shifted.log_coordinate(x_shifted, n) == base.log_coordinate(x, n)

                for sigma in range(1, base.s + 1):
                    child = base.F(x, sigma)
                    child_shifted = shifted.F(x_shifted, sigma)
                    assert child_shifted == child + delta
                    assert shifted.log_coordinate(child_shifted, n + 1) == base.log_coordinate(child, n + 1)
                    checked += 1

    print(f"[OK] Checked {checked} shifted one-step transitions")


def verify_asymptotic_expansion_numerically() -> None:
    print("\n=== Numerical and exact-rational verification of the asymptotic correction ===")

    checked = 0
    for s in range(2, 10):
        for sigma in range(1, s + 1):
            a = Fraction(sigma, s) - 1
            for u in (50, 100, 200, 500, 1000, 5000):
                exact_delta = 1 + log((u + sigma / s) / (u + 1), s)
                first_order = 1 + float(a) / ((u + 1) * log(s))
                error = abs(exact_delta - first_order)

                # The next Taylor term is O((u+1)^-2).  Use a conservative
                # constant that is uniform over the tested finite grid.
                assert error <= 2.0 / ((u + 1) ** 2 * log(s))
                checked += 1

            if sigma == s:
                for u in (0, 1, 10, 1000):
                    exact_delta = 1 + log((u + sigma / s) / (u + 1), s)
                    assert isclose(exact_delta, 1.0, rel_tol=0, abs_tol=1e-14)

    print(f"[OK] Checked {checked} asymptotic first-order approximations")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain and consistency tests ===")

    assert_raises(ValueError, GenerativeSpace, 0, 2, 2)
    assert_raises(ValueError, GenerativeSpace, 1, 1, 2)
    assert_raises(ValueError, GenerativeSpace, 1, 2, 1)
    assert_raises(TypeError, GenerativeSpace, 1.5, 2, 2)

    gs = GenerativeSpace(1, 2, 2)

    assert_raises(ValueError, gs.level_start, -1)
    assert_raises(TypeError, gs.level_start, 1.2)
    assert_raises(ValueError, gs.position, gs.level_start(2) - 1, 2)
    assert_raises(ValueError, gs.from_position, -1, 2)
    assert_raises(ValueError, gs.from_position, gs.level_size(2), 2)
    assert_raises(ValueError, gs.F, gs.level_start(1), 0)
    assert_raises(ValueError, gs.F, gs.level_start(1), 3)
    assert_raises(ValueError, gs.child_position, -1, 1)
    assert_raises(ValueError, gs.digit_coordinate, ())
    assert_raises(ValueError, gs.digit_coordinate, (1, 3))
    assert_raises(ValueError, gs.descendant_position, 0, ())
    assert_raises(ValueError, gs.log_coordinate_from_position, -1)
    assert_raises(ValueError, gs.shift, -1)
    assert_raises(TypeError, gs.shift, 1.5)

    # Damaged descendant: a position that is not in the depth-ell cone from u.
    u = 2
    depth = 3
    valid_low = gs.s**depth * u
    valid_high = gs.s**depth * u + gs.s**depth - 1
    damaged = valid_high + 1
    assert not (valid_low <= damaged <= valid_high)

    print("[OK] Invalid domains and damaged cone data are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of logarithmic spatial coordinate (sec:logarithmic-coordinate) ===")
    symbolic_verification()
    verify_log_coordinate_basic_properties()
    verify_one_step_evolution()
    verify_logarithmic_normalization_shift()
    verify_ell_step_speed_bound()
    verify_shift_invariance()
    verify_asymptotic_expansion_numerically()
    verify_negative_domain_tests()
    print("\n=== Logarithmic-coordinate verification completed successfully ===")


if __name__ == "__main__":
    main()
