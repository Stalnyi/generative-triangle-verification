"""
VERIFICATION of Section 2: Positional coordinate on a level
(sec:positional-coordinate).

This script provides a full mathematical verification block for
2_positional_coordinate.tex.  It verifies the new claims of the section:
the level-relative positional coordinate, its recursion under one causal
generation step, its exact base-s path representation, and the numerical
examples for the binary base case.

The script deliberately treats the previously established facts about the
existence of the levels, the generation map, and the unique path
decomposition as structural dependencies.  It does not re-prove the global
theory from earlier sections; it verifies that the positional coordinate
introduced here is algebraically and computationally compatible with those
dependencies.

Verified content
----------------
1. Level-relative coordinate:
      u_n(x) = x - m_n
   maps L_n bijectively onto {0, ..., k*s^n - 1}.

2. Exact one-step recursion:
      u_{n+1}(F(x,sigma)) = s*u_n(x) + (sigma - 1).

3. Base-s path representation:
      u_n(x) = u_0(a)*s^n + sum_{i=1}^n (sigma_i - 1)*s^{n-i},
   where u_0(a)=a-m and sigma_i in {1,...,s}.

4. Decoding:
   the integer u_n uniquely recovers the root index u_0(a) and all internal
   states sigma_1,...,sigma_n.

5. Range separation:
   the leading digit u_0(a) belongs to {0,...,k-1}, while each internal
   digit sigma_i-1 belongs to {0,...,s-1}.  The verification covers cases
   k < s, k = s, and k > s, so the leading digit is not silently confused
   with an internal base-s digit.

6. Numerical examples in the section:
      s=2, A={1,2}, n=2, x=11  ->  u_2(11)=4 and path (a=2; 1,1),
      s=2, A={1,2}, n=2, x=14  ->  u_2(14)=7 and path (a=2; 2,2).

7. Negative-domain tests:
   invalid model parameters, invalid internal states, roots outside A,
   elements outside the stated level, malformed coordinate values, and
   corrupted encodings are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Tuple

from sympy import Symbol, summation, simplify, symbols


StateSequence = Tuple[int, ...]


@dataclass(frozen=True)
class GenerativeModel:
    """Finite-parameter instance of the generative triangle used by the section."""

    m: int
    k: int
    s: int

    def __post_init__(self) -> None:
        if not isinstance(self.m, int) or not isinstance(self.k, int) or not isinstance(self.s, int):
            raise ValueError("m, k, and s must be integers")
        if self.m < 1:
            raise ValueError("m must be a positive natural number")
        if self.k < 2:
            raise ValueError("k must be at least 2")
        if self.s < 2:
            raise ValueError("s must be at least 2")

    @property
    def A(self) -> range:
        return range(self.m, self.m + self.k)

    @property
    def states(self) -> range:
        return range(1, self.s + 1)

    @property
    def c_shift(self) -> int:
        return (self.s - 1) * self.m + 1 - self.k

    def F(self, x: int, sigma: int) -> int:
        if sigma not in self.states:
            raise ValueError("internal state outside S")
        return self.s * x + sigma - self.c_shift

    def m_n(self, n: int) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError("level index must be a nonnegative integer")
        # m + k*(1 + s + ... + s^{n-1}); avoids divisibility assumptions in code.
        return self.m + self.k * sum(self.s**j for j in range(n))

    def level_size(self, n: int) -> int:
        if not isinstance(n, int) or n < 0:
            raise ValueError("level index must be a nonnegative integer")
        return self.k * self.s**n

    def level_bounds(self, n: int) -> tuple[int, int]:
        start = self.m_n(n)
        return start, start + self.level_size(n) - 1

    def level(self, n: int) -> range:
        start, end = self.level_bounds(n)
        return range(start, end + 1)

    def positional_coordinate(self, n: int, x: int) -> int:
        start, end = self.level_bounds(n)
        if x < start or x > end:
            raise ValueError("x is outside L_n")
        return x - start

    def reconstruct_from_u(self, n: int, u: int) -> int:
        if not isinstance(u, int):
            raise ValueError("u must be an integer")
        if u < 0 or u >= self.level_size(n):
            raise ValueError("u is outside the positional range of L_n")
        return self.m_n(n) + u

    def generate_from_path(self, root: int, seq: StateSequence) -> int:
        if root not in self.A:
            raise ValueError("root is outside A")
        if not isinstance(seq, tuple):
            raise ValueError("internal-state sequence must be a tuple")
        x = root
        for sigma in seq:
            x = self.F(x, sigma)
        return x

    def positional_formula(self, root: int, seq: StateSequence) -> int:
        if root not in self.A:
            raise ValueError("root is outside A")
        if not isinstance(seq, tuple):
            raise ValueError("internal-state sequence must be a tuple")
        n = len(seq)
        u0 = root - self.m
        if not 0 <= u0 <= self.k - 1:
            raise AssertionError("root digit outside its required range")
        total = u0 * self.s**n
        for i, sigma in enumerate(seq, start=1):
            if sigma not in self.states:
                raise ValueError("internal state outside S")
            total += (sigma - 1) * self.s ** (n - i)
        return total

    def decode_u(self, n: int, u: int) -> tuple[int, StateSequence]:
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if not isinstance(u, int) or u < 0 or u >= self.level_size(n):
            raise ValueError("u is outside the positional range of L_n")

        if n == 0:
            return self.m + u, tuple()

        root_digit = u // (self.s**n)
        if root_digit < 0 or root_digit >= self.k:
            raise ValueError("decoded root digit outside {0,...,k-1}")

        remainder = u - root_digit * self.s**n
        digits: list[int] = []
        for power in range(n - 1, -1, -1):
            base = self.s**power
            digit = remainder // base
            if digit < 0 or digit >= self.s:
                raise AssertionError("decoded internal digit outside {0,...,s-1}")
            digits.append(digit)
            remainder -= digit * base

        assert remainder == 0
        return self.m + root_digit, tuple(d + 1 for d in digits)

    def decode_element(self, n: int, x: int) -> tuple[int, StateSequence]:
        u = self.positional_coordinate(n, x)
        return self.decode_u(n, u)


def expect_raises(exc_type: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} was not raised")


def all_sequences(s: int, n: int) -> Iterable[StateSequence]:
    return product(range(1, s + 1), repeat=n)


def verify_symbolic_recursion_and_base_s_formula() -> None:
    print("\n=== Symbolic verification of positional-coordinate formulas ===")

    x, m_n_sym, c_A, sigma, s = symbols("x m_n c_A sigma s", integer=True)
    F_x_sigma = s * x + sigma - c_A
    m_np1 = s * m_n_sym + 1 - c_A
    u_next = simplify(F_x_sigma - m_np1)
    expected = s * (x - m_n_sym) + (sigma - 1)
    assert simplify(u_next - expected) == 0

    n = Symbol("n", integer=True, nonnegative=True)
    i = Symbol("i", integer=True, positive=True)
    u0 = Symbol("u0", integer=True, nonnegative=True)

    # Verification of the nontrivial algebraic induction step of the section:
    # multiplying the n-level expression by s and appending the next internal
    # state produces exactly the (n+1)-level expression.
    sigma_next = Symbol("sigma_next", integer=True)

    # Because SymPy cannot represent an indexed symbolic finite sum of arbitrary
    # symbols without introducing IndexedBase complexity, we verify the structural
    # exponent shift using a generic summand A_i = sigma_i - 1.
    A_i = Symbol("A_i", integer=True)
    shifted_summand = s * A_i * s ** (n - i)
    expected_shifted_summand = A_i * s ** ((n + 1) - i)
    assert simplify(shifted_summand - expected_shifted_summand) == 0

    # The closed formula for m_n used by u_n is compatible with the level
    # recursion m_{n+1}=s*m_n+1-c(A), once c(A) is substituted.
    m, k = symbols("m k", integer=True, positive=True)
    c_expr = (s - 1) * m + 1 - k
    m_n_expr = m + k * summation(s**i, (i, 0, n - 1))
    m_np1_expr = m + k * summation(s**i, (i, 0, n))
    recurrence_rhs = s * m_n_expr + 1 - c_expr
    assert simplify(m_np1_expr - recurrence_rhs) == 0

    print("[OK] One-step positional recursion verified symbolically")
    print("[OK] Base-s exponent shift in the path formula verified symbolically")
    print("[OK] Level-start formula is compatible with the required recursion")


def verify_level_coordinate_bijection() -> None:
    print("\n=== Exact finite verification of u_n as a level coordinate ===")

    models = [
        GenerativeModel(m=1, k=2, s=2),
        GenerativeModel(m=3, k=2, s=5),   # k < s
        GenerativeModel(m=2, k=4, s=4),   # k = s
        GenerativeModel(m=5, k=7, s=3),   # k > s
        GenerativeModel(m=4, k=5, s=6),
    ]

    checked_levels = 0
    checked_elements = 0

    for model in models:
        for n in range(0, 6):
            level = list(model.level(n))
            u_values = [model.positional_coordinate(n, x) for x in level]
            assert u_values == list(range(model.level_size(n)))
            assert len(set(u_values)) == model.level_size(n)

            for u in u_values:
                x = model.reconstruct_from_u(n, u)
                assert model.positional_coordinate(n, x) == u
                checked_elements += 1

            checked_levels += 1

    print(f"[OK] Verified bijective u_n coordinate on {checked_levels} levels")
    print(f"[OK] Checked {checked_elements} exact level elements")


def verify_one_step_recursion_exhaustively() -> None:
    print("\n=== Exhaustive verification of u_{n+1}=s*u_n+(sigma-1) ===")

    models = [
        GenerativeModel(m=1, k=2, s=2),
        GenerativeModel(m=2, k=3, s=2),
        GenerativeModel(m=3, k=5, s=4),
        GenerativeModel(m=6, k=4, s=7),
    ]

    checks = 0
    for model in models:
        for n in range(0, 5):
            for x in model.level(n):
                u = model.positional_coordinate(n, x)
                for sigma in model.states:
                    x_next = model.F(x, sigma)
                    assert x_next in model.level(n + 1)
                    u_next = model.positional_coordinate(n + 1, x_next)
                    assert u_next == model.s * u + (sigma - 1)
                    checks += 1

    print(f"[OK] Verified {checks} exact one-step recursion instances")


def verify_base_s_path_representation_and_decoding() -> None:
    print("\n=== Exact verification of the base-s path representation and decoding ===")

    models = [
        GenerativeModel(m=1, k=2, s=2),
        GenerativeModel(m=3, k=2, s=5),
        GenerativeModel(m=2, k=4, s=4),
        GenerativeModel(m=5, k=7, s=3),
    ]

    checks = 0
    for model in models:
        for n in range(0, 6):
            seen_u: set[int] = set()
            for root in model.A:
                for seq in all_sequences(model.s, n):
                    x = model.generate_from_path(root, seq)
                    assert x in model.level(n)

                    u_from_level = model.positional_coordinate(n, x)
                    u_from_formula = model.positional_formula(root, seq)
                    assert u_from_level == u_from_formula

                    decoded_root, decoded_seq = model.decode_u(n, u_from_formula)
                    assert decoded_root == root
                    assert decoded_seq == seq

                    decoded_from_x = model.decode_element(n, x)
                    assert decoded_from_x == (root, seq)

                    reconstructed_x = model.reconstruct_from_u(n, u_from_formula)
                    assert reconstructed_x == x

                    seen_u.add(u_from_formula)
                    checks += 1

            assert seen_u == set(range(model.level_size(n)))

    print(f"[OK] Verified {checks} generated paths against positional formula")
    print("[OK] Decoding from u_n recovers root and every internal state exactly")


def verify_leading_digit_is_not_an_internal_digit() -> None:
    print("\n=== Verification of leading-root digit versus internal-state digits ===")

    # The section explicitly states that the leading digit belongs to
    # {0,...,k-1}, while internal digits belong to {0,...,s-1}.
    # This distinction matters when k != s.
    cases = [
        GenerativeModel(m=10, k=2, s=5),  # leading range smaller than internal digit range
        GenerativeModel(m=20, k=6, s=3),  # leading range larger than internal digit range
    ]

    for model in cases:
        n = 3
        for root in model.A:
            for seq in all_sequences(model.s, n):
                u = model.positional_formula(root, seq)
                root_digit = u // (model.s**n)
                assert root_digit == root - model.m
                assert 0 <= root_digit <= model.k - 1

                remainder = u - root_digit * model.s**n
                internal_digits = []
                for power in range(n - 1, -1, -1):
                    digit = remainder // (model.s**power)
                    internal_digits.append(digit)
                    remainder -= digit * model.s**power

                assert all(0 <= d <= model.s - 1 for d in internal_digits)
                assert tuple(d + 1 for d in internal_digits) == seq

    print("[OK] Leading-root digit and internal-state digits are separated exactly")
    print("[OK] Verified cases k<s and k>s without conflating k with s")


def verify_section_examples() -> None:
    print("\n=== Verification of the numerical examples in the section ===")

    model = GenerativeModel(m=1, k=2, s=2)
    assert model.c_shift == 0

    x_11 = model.generate_from_path(2, (1, 1))
    assert x_11 == 11
    assert model.m_n(2) == 7
    assert model.positional_coordinate(2, x_11) == 4
    assert model.positional_formula(2, (1, 1)) == 4
    assert model.decode_element(2, 11) == (2, (1, 1))

    x_14 = model.generate_from_path(2, (2, 2))
    assert x_14 == 14
    assert model.positional_coordinate(2, x_14) == 7
    assert model.positional_formula(2, (2, 2)) == 7
    assert model.decode_element(2, 14) == (2, (2, 2))

    # Guard against a plausible transcription error: the first path cannot
    # produce 11 from root 1.
    wrong_candidate = model.generate_from_path(1, (1, 1))
    assert wrong_candidate == 7
    assert wrong_candidate != 11

    print("[OK] Verified x=11 example: root 2, states (1,1), u_2=4")
    print("[OK] Verified x=14 example: root 2, states (2,2), u_2=7")
    print("[OK] Rejected the incorrect root-1 reconstruction for x=11")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative-domain verification ===")

    expect_raises(ValueError, GenerativeModel, 0, 2, 2)
    expect_raises(ValueError, GenerativeModel, 1, 1, 2)
    expect_raises(ValueError, GenerativeModel, 1, 2, 1)
    expect_raises(ValueError, GenerativeModel, 1.5, 2, 2)
    expect_raises(ValueError, GenerativeModel, 1, 2.5, 2)
    expect_raises(ValueError, GenerativeModel, 1, 2, 2.5)

    model = GenerativeModel(m=2, k=3, s=4)

    expect_raises(ValueError, model.F, 2, 0)
    expect_raises(ValueError, model.F, 2, 5)
    expect_raises(ValueError, model.m_n, -1)
    expect_raises(ValueError, model.level_size, -1)
    expect_raises(ValueError, model.positional_coordinate, 1, model.m_n(1) - 1)
    expect_raises(ValueError, model.positional_coordinate, 1, model.level_bounds(1)[1] + 1)
    expect_raises(ValueError, model.reconstruct_from_u, 2, -1)
    expect_raises(ValueError, model.reconstruct_from_u, 2, model.level_size(2))
    expect_raises(ValueError, model.generate_from_path, 1, (1, 2))
    expect_raises(ValueError, model.generate_from_path, 2, [1, 2])
    expect_raises(ValueError, model.generate_from_path, 2, (1, 5))
    expect_raises(ValueError, model.positional_formula, 1, (1, 2))
    expect_raises(ValueError, model.positional_formula, 2, [1, 2])
    expect_raises(ValueError, model.positional_formula, 2, (1, 5))
    expect_raises(ValueError, model.decode_u, -1, 0)
    expect_raises(ValueError, model.decode_u, 2, -3)
    expect_raises(ValueError, model.decode_u, 2, model.level_size(2))

    print("[OK] Invalid domains are rejected explicitly")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of positional coordinate on a level (sec:positional-coordinate) ===")
    verify_symbolic_recursion_and_base_s_formula()
    verify_level_coordinate_bijection()
    verify_one_step_recursion_exhaustively()
    verify_base_s_path_representation_and_decoding()
    verify_leading_digit_is_not_an_internal_digit()
    verify_section_examples()
    verify_negative_domain_tests()
    print("\n=== Positional-coordinate verification completed successfully ===")


if __name__ == "__main__":
    main()
