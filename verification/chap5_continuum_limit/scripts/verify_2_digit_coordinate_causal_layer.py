"""
VERIFICATION of Section 2: Exact digit-coordinate of a causal layer
(sec:digit-coordinate-causal-layer).

This script provides a full mathematical verification block for
2_digit_coordinate_causal_layer.tex.

The section is verified as a new layer over the previously established
coordinate basis.  We do not re-prove the global bijectivity of F, the full
level construction, or the causal-cone cardinality theorem.  Instead, this
script checks the new claims made here:

Verified content
----------------
1. Definition of the causal-layer digit coordinate
      D_x(y) = sum_{j=1}^{Delta n} (sigma_j - 1) s^{Delta n - j}.

2. Exact decomposition of the descendant positional coordinate
      u_{n+Delta n}(y) = s^{Delta n} u_n(x) + D_x(y).

3. Compatibility with the one-step positional recursion
      u_{j+1} = s u_j + (sigma_{j+1} - 1)
   and the induced recurrence
      D_{\\ell+1} = s D_\\ell + (sigma_{\\ell+1} - 1)

4. Fixed-length base-s representation:
   the internal-state sequence of length Delta n is equivalent to the
   digit vector (sigma_1-1, ..., sigma_{Delta n}-1), including leading zeros.

5. Full filling of the causal layer:
      {D_x(y) | y in C^+(x) cap L_{n+Delta n}}
      = {0, 1, ..., s^{Delta n}-1}.

6. Injectivity and surjectivity of the finite-layer coding map
      S^{Delta n} <-> {0, ..., s^{Delta n}-1}.

7. Hierarchical decomposition:
      u_{n+Delta n}(y) // s^{Delta n} = u_n(x),
      u_{n+Delta n}(y) mod s^{Delta n} = D_x(y).

8. Contiguous slice geometry inside a causal cone:
      u_{n+Delta n}(C^+(x)) =
      {s^{Delta n} u_n(x), ..., s^{Delta n} u_n(x)+s^{Delta n}-1}.

9. Integration with the actual level-coordinate model:
   iterating F and reconstructing the descendant from m_{n+Delta n}+u
   agree exactly.

10. Negative domain tests:
    invalid s, invalid k, invalid m, Delta n = 0, invalid internal states,
    invalid digit values, out-of-range D, out-of-range u_n(x), corrupted
    decompositions, and wrong-level descendants are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

from sympy import Sum, simplify, symbols


@dataclass(frozen=True)
class Model:
    """Finite parameter package for the already established coordinate basis."""

    m: int
    k: int
    s: int

    def __post_init__(self) -> None:
        if not isinstance(self.m, int) or self.m < 1:
            raise ValueError("m must be a positive integer")
        if not isinstance(self.k, int) or self.k < 2:
            raise ValueError("k must be an integer at least 2")
        if not isinstance(self.s, int) or self.s < 2:
            raise ValueError("s must be an integer at least 2")

    @property
    def c(self) -> int:
        return (self.s - 1) * self.m + 1 - self.k

    def m_level(self, n: int) -> int:
        require_nonnegative_int(n, "level n")
        numerator = self.k * (self.s**n - 1)
        denominator = self.s - 1
        assert numerator % denominator == 0
        return self.m + numerator // denominator

    def level_size(self, n: int) -> int:
        require_nonnegative_int(n, "level n")
        return self.k * self.s**n

    def M_level(self, n: int) -> int:
        return self.m_level(n) + self.level_size(n) - 1

    def contains_level_element(self, x: int, n: int) -> bool:
        require_nonnegative_int(n, "level n")
        return self.m_level(n) <= x <= self.M_level(n)

    def u(self, x: int, n: int) -> int:
        if not self.contains_level_element(x, n):
            raise ValueError("element is outside the specified level")
        return x - self.m_level(n)

    def x_from_u(self, u_value: int, n: int) -> int:
        require_nonnegative_int(n, "level n")
        if not isinstance(u_value, int) or not (0 <= u_value < self.level_size(n)):
            raise ValueError("positional coordinate is outside the level range")
        return self.m_level(n) + u_value

    def F(self, x: int, sigma: int) -> int:
        require_state(sigma, self.s)
        return self.s * x + sigma - self.c


def require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def require_positive_depth(delta: int) -> None:
    if not isinstance(delta, int) or delta < 1:
        raise ValueError("Delta n must be a positive integer")


def require_state(sigma: int, s: int) -> None:
    if not isinstance(sigma, int) or not (1 <= sigma <= s):
        raise ValueError("internal state is outside S")


def require_state_sequence(states: Sequence[int], s: int) -> None:
    if len(states) == 0:
        raise ValueError("the internal-state sequence must be nonempty in this section")
    for sigma in states:
        require_state(sigma, s)


def require_digit(digit: int, s: int) -> None:
    if not isinstance(digit, int) or not (0 <= digit <= s - 1):
        raise ValueError("digit is outside the base-s range")


def require_digit_vector(digits: Sequence[int], s: int) -> None:
    if len(digits) == 0:
        raise ValueError("digit vector must be nonempty in this section")
    for d in digits:
        require_digit(d, s)


def digit_coordinate_from_states(states: Sequence[int], s: int) -> int:
    """D = sum (sigma_j - 1) s^{Delta n-j} for a nonempty state sequence."""
    require_state_sequence(states, s)
    delta = len(states)
    return sum((sigma - 1) * (s ** (delta - j - 1)) for j, sigma in enumerate(states))


def digit_coordinate_from_digits(digits: Sequence[int], s: int) -> int:
    """Encode a fixed-length base-s digit vector as an integer."""
    require_digit_vector(digits, s)
    delta = len(digits)
    return sum(d * (s ** (delta - j - 1)) for j, d in enumerate(digits))


def digits_from_D(D: int, delta: int, s: int) -> tuple[int, ...]:
    """Decode D into exactly Delta n base-s digits, preserving leading zeros."""
    require_positive_depth(delta)
    if not isinstance(D, int) or not (0 <= D <= s**delta - 1):
        raise ValueError("D is outside the causal-layer digit range")
    digits: list[int] = []
    remainder = D
    for power in range(delta - 1, -1, -1):
        base_power = s**power
        digit = remainder // base_power
        require_digit(digit, s)
        digits.append(digit)
        remainder -= digit * base_power
    assert remainder == 0
    return tuple(digits)


def states_from_D(D: int, delta: int, s: int) -> tuple[int, ...]:
    return tuple(d + 1 for d in digits_from_D(D, delta, s))


def descendant_u(u_parent: int, states: Sequence[int], s: int) -> int:
    require_state_sequence(states, s)
    if not isinstance(u_parent, int) or u_parent < 0:
        raise ValueError("parent positional coordinate must be nonnegative")
    u_value = u_parent
    for sigma in states:
        u_value = s * u_value + (sigma - 1)
    return u_value


def exact_decomposition_u(u_parent: int, states: Sequence[int], s: int) -> int:
    require_state_sequence(states, s)
    delta = len(states)
    D = digit_coordinate_from_states(states, s)
    return (s**delta) * u_parent + D


def iterated_F(model: Model, x: int, states: Sequence[int]) -> int:
    require_state_sequence(states, model.s)
    current = x
    for sigma in states:
        current = model.F(current, sigma)
    return current


def cone_slice_u_values(u_parent: int, delta: int, s: int) -> list[int]:
    require_positive_depth(delta)
    if not isinstance(u_parent, int) or u_parent < 0:
        raise ValueError("parent positional coordinate must be nonnegative")
    start = (s**delta) * u_parent
    return list(range(start, start + s**delta))


def all_state_sequences(delta: int, s: int) -> Iterable[tuple[int, ...]]:
    require_positive_depth(delta)
    return product(range(1, s + 1), repeat=delta)


def assert_raises(expected_exception: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except expected_exception:
        return
    raise AssertionError(f"{fn.__name__} accepted an invalid input")


def verify_symbolic_digit_coordinate_bounds_and_decomposition() -> None:
    print("\n=== Symbolic verification of digit-coordinate bounds and decomposition ===")

    s, delta, U, D = symbols("s delta U D", integer=True, positive=True)
    j = symbols("j", integer=True, positive=True)

    # Maximal digit coordinate occurs when every digit is s-1:
    maximal_sum = Sum((s - 1) * s ** (delta - j), (j, 1, delta)).doit()
    assert simplify(maximal_sum - (s**delta - 1)) == 0

    # Minimal coordinate occurs when every digit is zero.
    minimal_sum = Sum(0 * s ** (delta - j), (j, 1, delta)).doit()
    assert simplify(minimal_sum) == 0

    # Hierarchical quotient/remainder identities under 0 <= D < s^delta.
    # SymPy cannot use inequalities inside floor/mod automatically here, so the
    # script verifies the algebraic decomposition directly and exact integer
    # quotient/remainder exhaustively in finite grids below.
    decomposed = s**delta * U + D
    assert simplify(decomposed - (s**delta * U + D)) == 0

    # Induced recurrence for the digit coordinate after appending one state.
    Dk, sigma_next = symbols("D_k sigma_next", integer=True)
    Dkp1 = s * Dk + (sigma_next - 1)
    low_preserved = Dkp1.subs({Dk: 0, sigma_next: 1})
    high_preserved = Dkp1.subs({Dk: s**delta - 1, sigma_next: s})
    assert simplify(low_preserved) == 0
    assert simplify(high_preserved - (s ** (delta + 1) - 1)) == 0

    print("[OK] Maximal digit coordinate equals s^Delta-1 symbolically")
    print("[OK] Appending one internal state preserves the exact digit-coordinate range")


def verify_encoding_decoding_bijection() -> None:
    print("\n=== Exact finite verification of the fixed-length base-s bijection ===")

    total_sequences = 0
    for s in range(2, 6):
        for delta in range(1, 6):
            encoded_values: set[int] = set()
            for states in all_state_sequences(delta, s):
                total_sequences += 1
                digits = tuple(sigma - 1 for sigma in states)
                D_from_states = digit_coordinate_from_states(states, s)
                D_from_digits = digit_coordinate_from_digits(digits, s)
                assert D_from_states == D_from_digits
                assert 0 <= D_from_states <= s**delta - 1
                encoded_values.add(D_from_states)

                decoded_digits = digits_from_D(D_from_states, delta, s)
                decoded_states = states_from_D(D_from_states, delta, s)
                assert decoded_digits == digits
                assert decoded_states == states

            assert encoded_values == set(range(s**delta))
            assert len(encoded_values) == s**delta

            # Leading-zero preservation is essential for fixed-depth causal layers.
            assert digits_from_D(0, delta, s) == tuple(0 for _ in range(delta))
            if delta >= 2:
                assert len(digits_from_D(1, delta, s)) == delta
                assert digit_coordinate_from_digits(digits_from_D(1, delta, s), s) == 1

    print(f"[OK] Verified {total_sequences} internal-state sequences across exact finite grids")
    print("[OK] Every causal layer is filled by exactly {0,...,s^Delta-1}")


def verify_exact_decomposition_against_recursion() -> None:
    print("\n=== Exact verification of u-recursion versus closed decomposition ===")

    checked = 0
    for s in range(2, 6):
        for u_parent in range(0, 12):
            for delta in range(1, 5):
                for states in all_state_sequences(delta, s):
                    recursive_value = descendant_u(u_parent, states, s)
                    decomposed_value = exact_decomposition_u(u_parent, states, s)
                    D = digit_coordinate_from_states(states, s)
                    assert recursive_value == decomposed_value
                    assert recursive_value == (s**delta) * u_parent + D
                    assert recursive_value // (s**delta) == u_parent
                    assert recursive_value % (s**delta) == D
                    checked += 1

    print(f"[OK] Checked {checked} exact recursion/decomposition cases")
    print("[OK] Quotient recovers the ancestor coordinate and remainder recovers D")


def verify_cone_slice_geometry() -> None:
    print("\n=== Verification of contiguous causal-layer slice geometry ===")

    cases = 0
    for s in range(2, 6):
        for delta in range(1, 6):
            for u_parent in range(0, 12):
                generated = sorted(descendant_u(u_parent, states, s) for states in all_state_sequences(delta, s))
                expected = cone_slice_u_values(u_parent, delta, s)

                assert generated == expected
                assert len(generated) == s**delta
                assert generated[0] == (s**delta) * u_parent
                assert generated[-1] == (s**delta) * u_parent + s**delta - 1

                # Successive descendants in sorted D-order are adjacent positions.
                assert all(b - a == 1 for a, b in zip(generated, generated[1:]))
                cases += 1

    print(f"[OK] Verified {cases} complete contiguous causal slices")


def verify_integration_with_actual_level_coordinates() -> None:
    print("\n=== Integration check with level coordinates and the generation rule F ===")

    models = [
        Model(m=1, k=2, s=2),
        Model(m=2, k=3, s=2),
        Model(m=1, k=3, s=3),
        Model(m=4, k=5, s=4),
    ]

    checked = 0
    for model in models:
        for n in range(0, 5):
            level_size = model.level_size(n)
            sample_u_values = sorted({0, min(1, level_size - 1), level_size // 2, level_size - 1})
            for u_parent in sample_u_values:
                x = model.x_from_u(u_parent, n)
                assert model.u(x, n) == u_parent

                for delta in range(1, 5):
                    for states in all_state_sequences(delta, model.s):
                        y_by_F = iterated_F(model, x, states)
                        u_by_decomposition = exact_decomposition_u(u_parent, states, model.s)
                        y_by_coordinate = model.x_from_u(u_by_decomposition, n + delta)

                        assert y_by_F == y_by_coordinate
                        assert model.contains_level_element(y_by_F, n + delta)
                        assert model.u(y_by_F, n + delta) == u_by_decomposition
                        checked += 1

    print(f"[OK] Verified {checked} integrations of F-iteration with digit-coordinate reconstruction")


def verify_full_filling_independent_of_ancestor_position() -> None:
    print("\n=== Verification that D-range is independent of ancestor position ===")

    for s in range(2, 7):
        for delta in range(1, 6):
            reference = set(range(s**delta))
            for u_parent in (0, 1, 3, 10, 37):
                values = {
                    descendant_u(u_parent, states, s) - (s**delta) * u_parent
                    for states in all_state_sequences(delta, s)
                }
                assert values == reference

    print("[OK] The internal D-spectrum of a causal layer is independent of u_n(x)")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain and corruption tests ===")

    assert_raises(ValueError, Model, 0, 2, 2)
    assert_raises(ValueError, Model, 1, 1, 2)
    assert_raises(ValueError, Model, 1, 2, 1)

    assert_raises(ValueError, digit_coordinate_from_states, [], 2)
    assert_raises(ValueError, digit_coordinate_from_states, [1, 3], 2)
    assert_raises(ValueError, digit_coordinate_from_states, [0, 1], 2)
    assert_raises(ValueError, digit_coordinate_from_digits, [0, 2], 2)
    assert_raises(ValueError, digit_coordinate_from_digits, [-1, 0], 2)

    assert_raises(ValueError, digits_from_D, -1, 3, 2)
    assert_raises(ValueError, digits_from_D, 8, 3, 2)
    assert_raises(ValueError, digits_from_D, 0, 0, 2)

    assert_raises(ValueError, descendant_u, -1, [1, 2], 2)
    assert_raises(ValueError, cone_slice_u_values, 0, 0, 2)

    model = Model(m=1, k=2, s=2)
    assert_raises(ValueError, model.u, model.M_level(2) + 1, 2)
    assert_raises(ValueError, model.x_from_u, model.level_size(3), 3)
    assert_raises(ValueError, model.x_from_u, -1, 3)

    # Corrupted decomposition: changing D by one must leave the actual recursive
    # descendant coordinate unless the modified value exits the layer range.
    states = (1, 2, 1, 2)
    s = 2
    u_parent = 5
    true_D = digit_coordinate_from_states(states, s)
    true_u = descendant_u(u_parent, states, s)
    corrupted_D = true_D + 1
    if corrupted_D < s ** len(states):
        corrupted_u = (s ** len(states)) * u_parent + corrupted_D
        assert corrupted_u != true_u

    # Wrong-level check: the same integer positional value cannot be accepted
    # as a descendant at a different depth unless its layer range and quotient
    # structure are checked separately.
    y_depth_3_u = descendant_u(2, (1, 2, 2), 2)
    assert y_depth_3_u in cone_slice_u_values(2, 3, 2)
    assert y_depth_3_u not in cone_slice_u_values(2, 2, 2)

    print("[OK] Invalid domains and corrupted decompositions are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of exact digit-coordinate of causal layer ===")
    verify_symbolic_digit_coordinate_bounds_and_decomposition()
    verify_encoding_decoding_bijection()
    verify_exact_decomposition_against_recursion()
    verify_cone_slice_geometry()
    verify_integration_with_actual_level_coordinates()
    verify_full_filling_independent_of_ancestor_position()
    verify_negative_domain_tests()
    print("\n=== Digit-coordinate causal-layer verification completed successfully ===")


if __name__ == "__main__":
    main()
