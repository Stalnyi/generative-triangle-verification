"""
VERIFICATION of Section: Coordinate basis of causal structure
(sec:coordinate-basis-causal-structure).

This script provides a full mathematical verification block for
1_coordinate_basis_causal_structure.tex.

The section is a synthesis section: it does not introduce an additional
physical axiom.  It assembles three previously constructed internal coordinates
of the generative triangle T = T_(A,S,F):

    t(x)          level index;
    u_{t(x)}(x)   positional coordinate inside the level;
    r_{t(x)}(x)   logarithmic spatial coordinate log_s(u_{t(x)}(x)+1).

Accordingly, this script verifies the coordinate-basis claims themselves:
well-definedness, uniqueness, reconstruction, exact recursion, genealogical
decoding, logarithmic normalization, and causal compatibility.  It uses the
recursive model equations as dependencies and does not re-verify unrelated
physical interpretations.

Verified content
----------------
1. Admissible model domain:
      m in N, k >= 2, s >= 2,
      A = {m, ..., m+k-1}, S = {1, ..., s},
      c(A) = (s-1)m + 1 - k.

2. Level structure needed for the coordinate basis:
      L_n = {m_n, ..., M_n},
      m_n = m + k(s^n-1)/(s-1),
      M_n = m_n + k s^n - 1,
      M_n + 1 = m_{n+1}.
   These identities are checked symbolically and then exhaustively over finite
   parameter grids.

3. Time coordinate:
      t(x)=n iff x in L_n.
   Finite checks verify that every generated element has exactly one level
   coordinate and that level disjointness is respected.

4. Positional coordinate:
      u_n(x)=x-m_n in {0, ..., k s^n - 1}.
   The map x |-> u_n(x) is verified to be a bijection from L_n onto this full
   integer interval, and x=m_n+u reconstructs the element.

5. Exact positional recursion along an edge:
      if x' = F(x,sigma), then
      u_{n+1}(x') = s u_n(x) + (sigma-1).

6. Exact genealogical decoding of u_n(x):
      u_n(x)=u_0(a)s^n + sum_{i=1}^n (sigma_i-1)s^{n-i}.
   Exhaustive finite checks verify both directions:
      coordinate -> root and internal-state sequence,
      root and internal-state sequence -> coordinate.

7. Logarithmic coordinate:
      r_n(x)=log_s(u_n(x)+1).
   The verification checks boundary behavior, monotonicity, injectivity through
   u, and the exact additive form inherited from the positional recursion:
      u_{n+1}(x')+1 = s(u_n(x)+sigma/s),
      r_{n+1}(x') = 1 + log_s(u_n(x)+sigma/s).

8. Coordinate completeness:
      (t(x),u_t(x)(x)) reconstructs x uniquely;
      r_t(x)(x) is a deterministic function of u_t(x)(x);
      if two generated elements share the same (t,u,r) tuple, they are equal.

9. Causal compatibility:
      x precedes y implies t(x)<t(y).
   The converse is explicitly rejected by finite counterexamples: later level
   membership alone does not imply descent from a fixed earlier element.

10. Future-cone coordinate slices:
      for y at depth Delta above x,
      u_{n+Delta}(y)=s^Delta u_n(x)+D,
      0 <= D <= s^Delta-1.
   The slice cardinality is exactly s^Delta, and D is decoded as the base-s
   digit of the internal-state suffix.

11. Domain and failure tests:
      invalid parameters, invalid internal states, invalid levels, out-of-range
      u, elements not belonging to the requested level, invalid logarithm base,
      malformed coordinate tuples, and non-causal later elements are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isclose, log as real_log
from typing import Callable, Iterable, Sequence

from sympy import Rational, Symbol, log, simplify, symbols


class DomainError(ValueError):
    """Raised when a tested object lies outside the model domain."""


@dataclass(frozen=True)
class ParameterCase:
    m: int
    k: int
    s: int
    max_n: int


PARAMETER_GRID: tuple[ParameterCase, ...] = (
    ParameterCase(m=1, k=2, s=2, max_n=7),
    ParameterCase(m=2, k=3, s=2, max_n=6),
    ParameterCase(m=1, k=2, s=3, max_n=5),
    ParameterCase(m=3, k=4, s=3, max_n=4),
    ParameterCase(m=5, k=2, s=4, max_n=4),
    ParameterCase(m=2, k=5, s=5, max_n=3),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DomainError(message)


def require_parameters(m: int, k: int, s: int) -> None:
    require(isinstance(m, int) and m >= 1, "m must belong to N")
    require(isinstance(k, int) and k >= 2, "k must be an integer at least 2")
    require(isinstance(s, int) and s >= 2, "s must be an integer at least 2")


def shift_c(m: int, k: int, s: int) -> int:
    require_parameters(m, k, s)
    return (s - 1) * m + 1 - k


def F(case: ParameterCase, x: int, sigma: int) -> int:
    require_parameters(case.m, case.k, case.s)
    require(isinstance(sigma, int) and 1 <= sigma <= case.s, "invalid internal state")
    return case.s * x + sigma - shift_c(case.m, case.k, case.s)


def m_start(case: ParameterCase, n: int) -> int:
    require_parameters(case.m, case.k, case.s)
    require(isinstance(n, int) and n >= 0, "level must belong to N0")
    numerator = case.k * (case.s**n - 1)
    denominator = case.s - 1
    require(numerator % denominator == 0, "level start must be integral")
    return case.m + numerator // denominator


def level_size(case: ParameterCase, n: int) -> int:
    require(isinstance(n, int) and n >= 0, "level must belong to N0")
    return case.k * case.s**n


def M_end(case: ParameterCase, n: int) -> int:
    return m_start(case, n) + level_size(case, n) - 1


def level_interval(case: ParameterCase, n: int) -> list[int]:
    return list(range(m_start(case, n), M_end(case, n) + 1))


def in_level(case: ParameterCase, x: int, n: int) -> bool:
    return m_start(case, n) <= x <= M_end(case, n)


def time_coordinate(case: ParameterCase, x: int, max_n: int | None = None) -> int:
    search_to = case.max_n if max_n is None else max_n
    hits = [n for n in range(search_to + 1) if in_level(case, x, n)]
    require(len(hits) == 1, f"element {x} must have exactly one tested level")
    return hits[0]


def positional_coordinate(case: ParameterCase, x: int, n: int) -> int:
    require(in_level(case, x, n), f"element {x} does not belong to level {n}")
    u = x - m_start(case, n)
    require(0 <= u < level_size(case, n), "positional coordinate out of range")
    return u


def reconstruct_from_tu(case: ParameterCase, n: int, u: int) -> int:
    require(isinstance(n, int) and n >= 0, "level must belong to N0")
    require(isinstance(u, int), "positional coordinate must be integral")
    require(0 <= u < level_size(case, n), "positional coordinate out of range")
    return m_start(case, n) + u


def log_coordinate(case: ParameterCase, u: int) -> float:
    require_parameters(case.m, case.k, case.s)
    require(isinstance(u, int) and u >= 0, "logarithmic coordinate requires u >= 0")
    return real_log(u + 1, case.s)


def coordinate_tuple(case: ParameterCase, x: int, n: int):
    u = positional_coordinate(case, x, n)
    r = log_coordinate(case, u)
    return (n, u, r)


def internal_state_sequences(s: int, length: int) -> Iterable[tuple[int, ...]]:
    require(isinstance(s, int) and s >= 2, "invalid internal-state alphabet size")
    require(isinstance(length, int) and length >= 0, "length must belong to N0")
    return product(range(1, s + 1), repeat=length)


def apply_sequence(case: ParameterCase, root: int, state_seq: Sequence[int]) -> int:
    require(case.m <= root <= case.m + case.k - 1, "root outside A")
    current = root
    for sigma in state_seq:
        current = F(case, current, sigma)
    return current


def decode_from_u(case: ParameterCase, n: int, u: int) -> tuple[int, tuple[int, ...]]:
    require(0 <= u < level_size(case, n), "positional coordinate out of range")
    root_offset = u // (case.s**n)
    remainder = u % (case.s**n)
    require(0 <= root_offset < case.k, "decoded root offset outside A")
    digits: list[int] = []
    for power in range(n - 1, -1, -1):
        place = case.s**power
        digit = remainder // place
        remainder = remainder % place
        require(0 <= digit <= case.s - 1, "decoded digit outside range")
        digits.append(digit)
    state_seq = tuple(d + 1 for d in digits)
    return case.m + root_offset, state_seq


def decode_from_element(case: ParameterCase, x: int, n: int) -> tuple[int, tuple[int, ...]]:
    u = positional_coordinate(case, x, n)
    return decode_from_u(case, n, u)


def inverse_parent(case: ParameterCase, y: int) -> tuple[int, int]:
    z = y + shift_c(case.m, case.k, case.s)
    sigma = ((z - 1) % case.s) + 1
    parent_numerator = z - sigma
    require(parent_numerator % case.s == 0, "parent numerator not divisible by s")
    parent = parent_numerator // case.s
    return parent, sigma


def is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    return len(prefix) <= len(full) and tuple(full[: len(prefix)]) == tuple(prefix)


def causally_precedes(case: ParameterCase, x: int, n_x: int, y: int, n_y: int) -> bool:
    if n_y <= n_x:
        return False
    root_x, seq_x = decode_from_element(case, x, n_x)
    root_y, seq_y = decode_from_element(case, y, n_y)
    return root_x == root_y and is_prefix(seq_x, seq_y)


def suffix_digit(case: ParameterCase, state_suffix: Sequence[int]) -> int:
    delta = len(state_suffix)
    total = 0
    for i, sigma in enumerate(state_suffix, start=1):
        require(1 <= sigma <= case.s, "invalid internal state in suffix")
        total += (sigma - 1) * case.s ** (delta - i)
    return total


def expect_domain_error(thunk: Callable[[], object], label: str) -> None:
    try:
        thunk()
    except DomainError:
        return
    raise AssertionError(f"Expected DomainError was not raised: {label}")


def verify_symbolic_core_identities() -> None:
    print("\n=== Symbolic verification of coordinate-basis identities ===")

    m, k, s, n, u, sigma = symbols("m k s n u sigma", positive=True, integer=True)
    c = (s - 1) * m + 1 - k

    m_n = m + k * (s**n - 1) / (s - 1)
    m_np1_explicit = m + k * (s ** (n + 1) - 1) / (s - 1)
    m_np1_recursive = s * m_n + 1 - c
    assert simplify(m_np1_recursive - m_np1_explicit) == 0

    M_n = m_n + k * s**n - 1
    assert simplify((M_n + 1) - m_np1_explicit) == 0

    x = m_n + u
    x_next = s * x + sigma - c
    u_next = simplify(x_next - m_np1_explicit)
    assert simplify(u_next - (s * u + sigma - 1)) == 0

    assert simplify((u_next + 1) - s * (u + Rational(1, 1) * sigma / s)) == 0

    # The logarithmic recursion is represented by equality of the arguments:
    # log_s(u_next+1)=log_s(s*(u+sigma/s))=1+log_s(u+sigma/s).
    assert simplify((u_next + 1) / s - (u + sigma / s)) == 0

    u0 = Symbol("u0", integer=True, nonnegative=True)
    d1, d2, d3 = symbols("d1 d2 d3", integer=True, nonnegative=True)
    base_expr = u0 * s**3 + d1 * s**2 + d2 * s + d3
    after_one_more = s * base_expr + (sigma - 1)
    expected = u0 * s**4 + d1 * s**3 + d2 * s**2 + d3 * s + (sigma - 1)
    assert simplify(after_one_more - expected) == 0

    assert simplify(log(1, s)) == 0

    print("[OK] Explicit level starts satisfy the recursive level-start equation")
    print("[OK] Consecutive level intervals are adjacent: M_n + 1 = m_{n+1}")
    print("[OK] Edge recursion gives u_{n+1}=s*u_n+(sigma-1)")
    print("[OK] The logarithmic-coordinate recursion follows from equality of arguments")
    print("[OK] One-step extension of the base-s positional expansion is symbolically consistent")


def verify_levels_and_time(case: ParameterCase) -> None:
    print(f"\n=== Level and time-coordinate checks: m={case.m}, k={case.k}, s={case.s} ===")

    all_seen: dict[int, int] = {}
    previous_end: int | None = None

    for n in range(case.max_n + 1):
        L_n = level_interval(case, n)
        assert len(L_n) == level_size(case, n)
        assert L_n[0] == m_start(case, n)
        assert L_n[-1] == M_end(case, n)
        assert L_n == list(range(L_n[0], L_n[-1] + 1))

        if previous_end is not None:
            assert L_n[0] == previous_end + 1
        previous_end = L_n[-1]

        for x in L_n:
            assert x not in all_seen
            all_seen[x] = n
            assert time_coordinate(case, x) == n

    for x, n in all_seen.items():
        hits = [j for j in range(case.max_n + 1) if in_level(case, x, j)]
        assert hits == [n]

    print(f"[OK] Checked {len(all_seen)} generated elements with unique time coordinates")


def verify_positional_bijection_and_reconstruction(case: ParameterCase) -> None:
    print("\n=== Positional-coordinate bijection and reconstruction ===")

    total = 0
    for n in range(case.max_n + 1):
        L_n = level_interval(case, n)
        u_values = [positional_coordinate(case, x, n) for x in L_n]
        expected_u = list(range(level_size(case, n)))
        assert u_values == expected_u
        assert len(set(u_values)) == len(u_values)

        for u in expected_u:
            x = reconstruct_from_tu(case, n, u)
            assert x in L_n
            assert positional_coordinate(case, x, n) == u
            total += 1

    print(f"[OK] Positional coordinates bijectively index {total} finite-level elements")


def verify_edge_recursion_and_inverse(case: ParameterCase) -> None:
    print("\n=== Exact edge recursion and inverse-parent decoding ===")

    edge_count = 0
    for n in range(case.max_n):
        for x in level_interval(case, n):
            u_x = positional_coordinate(case, x, n)
            produced = []
            for sigma in range(1, case.s + 1):
                y = F(case, x, sigma)
                assert in_level(case, y, n + 1)
                assert positional_coordinate(case, y, n + 1) == case.s * u_x + (sigma - 1)

                parent, recovered_sigma = inverse_parent(case, y)
                assert parent == x
                assert recovered_sigma == sigma
                produced.append(y)
                edge_count += 1
            assert len(produced) == case.s
            assert len(set(produced)) == case.s

    print(f"[OK] Checked {edge_count} generated edges and exact inverse parents")


def verify_genealogical_decoding(case: ParameterCase) -> None:
    print("\n=== Genealogical decoding of positional coordinates ===")

    total = 0
    for n in range(case.max_n + 1):
        for x in level_interval(case, n):
            u = positional_coordinate(case, x, n)
            root, state_seq = decode_from_u(case, n, u)

            assert len(state_seq) == n
            assert case.m <= root <= case.m + case.k - 1
            assert all(1 <= sigma <= case.s for sigma in state_seq)

            reconstructed_x = apply_sequence(case, root, state_seq)
            assert reconstructed_x == x

            direct_root, direct_seq = decode_from_element(case, x, n)
            assert direct_root == root
            assert direct_seq == state_seq

            # Backward inverse traversal must recover the same finite prefix.
            current = x
            recovered: list[int] = []
            for level in range(n, 0, -1):
                parent, sigma = inverse_parent(case, current)
                assert in_level(case, parent, level - 1)
                recovered.append(sigma)
                current = parent
            assert current == root
            assert tuple(reversed(recovered)) == state_seq
            total += 1

        if n <= 5:
            for root_offset in range(case.k):
                root = case.m + root_offset
                for state_seq in internal_state_sequences(case.s, n):
                    x = apply_sequence(case, root, state_seq)
                    assert in_level(case, x, n)
                    u = positional_coordinate(case, x, n)
                    decoded_root, decoded_seq = decode_from_u(case, n, u)
                    assert decoded_root == root
                    assert decoded_seq == state_seq

    print(f"[OK] Decoded and reconstructed {total} generated elements")


def verify_logarithmic_coordinate(case: ParameterCase) -> None:
    print("\n=== Logarithmic-coordinate checks ===")

    checked = 0
    for n in range(case.max_n + 1):
        previous_arg: int | None = None
        previous_r: float | None = None
        for x in level_interval(case, n):
            u = positional_coordinate(case, x, n)
            r = log_coordinate(case, u)

            if u == 0:
                assert r == 0.0

            if previous_arg is not None and previous_r is not None:
                assert u + 1 > previous_arg
                assert r > previous_r
            previous_arg = u + 1
            previous_r = r

            # Exact recovery is checked on the argument level: r is defined from
            # the unique integer argument u+1, and log_s is strictly injective on
            # positive reals.  Powers are checked exactly when u+1 is a pure
            # power of s.
            arg = u + 1
            power = 1
            exponent = 0
            while power < arg:
                power *= case.s
                exponent += 1
            if power == arg:
                assert isclose(r, float(exponent), rel_tol=0.0, abs_tol=1e-12)
            checked += 1

    for n in range(case.max_n):
        for x in level_interval(case, n):
            u_x = positional_coordinate(case, x, n)
            for sigma in range(1, case.s + 1):
                y = F(case, x, sigma)
                u_y = positional_coordinate(case, y, n + 1)
                assert u_y + 1 == case.s * u_x + sigma
                assert u_y + 1 == case.s * (u_x + Rational(sigma, case.s))

                left = log_coordinate(case, u_y)
                right = 1.0 + real_log(float(u_x + Rational(sigma, case.s)), case.s)
                assert isclose(left, right, rel_tol=0.0, abs_tol=1e-12)

    print(f"[OK] Checked {checked} logarithmic coordinates and edge-additive identities")


def verify_coordinate_completeness(case: ParameterCase) -> None:
    print("\n=== Coordinate completeness ===")

    seen_tu: dict[tuple[int, int], int] = {}
    seen_tur: dict[tuple[int, int, object], int] = {}
    checked = 0

    for n in range(case.max_n + 1):
        for x in level_interval(case, n):
            t, u, r = coordinate_tuple(case, x, n)
            assert t == n
            assert reconstruct_from_tu(case, t, u) == x

            key_tu = (t, u)
            key_tur = (t, u, r)
            assert key_tu not in seen_tu or seen_tu[key_tu] == x
            assert key_tur not in seen_tur or seen_tur[key_tur] == x
            seen_tu[key_tu] = x
            seen_tur[key_tur] = x

            root, state_seq = decode_from_element(case, x, n)
            assert apply_sequence(case, root, state_seq) == x
            checked += 1

    # r does not add independent information once t and u are fixed.
    for (t, u), x in seen_tu.items():
        assert log_coordinate(case, u) == coordinate_tuple(case, x, t)[2]

    print(f"[OK] Verified unique reconstruction of {checked} elements from (t,u)")


def verify_causal_compatibility_and_future_slices(case: ParameterCase) -> None:
    print("\n=== Causal compatibility and future-cone coordinate slices ===")

    causal_pairs = 0
    future_slice_points = 0

    for n in range(case.max_n + 1):
        for x in level_interval(case, n):
            root_x, seq_x = decode_from_element(case, x, n)
            u_x = positional_coordinate(case, x, n)

            for delta in range(1, case.max_n - n + 1):
                descendants = []
                D_values = []
                for suffix in internal_state_sequences(case.s, delta):
                    y = apply_sequence(case, root_x, seq_x + tuple(suffix))
                    assert in_level(case, y, n + delta)
                    assert causally_precedes(case, x, n, y, n + delta)
                    assert n < n + delta

                    D = suffix_digit(case, suffix)
                    u_y = positional_coordinate(case, y, n + delta)
                    assert u_y == case.s**delta * u_x + D
                    assert 0 <= D <= case.s**delta - 1
                    descendants.append(y)
                    D_values.append(D)
                    causal_pairs += 1

                assert len(descendants) == case.s**delta
                assert len(set(descendants)) == case.s**delta
                assert sorted(D_values) == list(range(case.s**delta))
                future_slice_points += len(descendants)

    # Later time alone is not enough for causality.
    if case.max_n >= 1:
        early = case.m
        later_from_other_root = F(case, case.m + 1, 1)
        assert time_coordinate(case, early) == 0
        assert time_coordinate(case, later_from_other_root) == 1
        assert not causally_precedes(case, early, 0, later_from_other_root, 1)

    print(f"[OK] Checked {causal_pairs} causal pairs and {future_slice_points} future-slice points")
    print("[OK] Verified by counterexample that t(x)<t(y) alone does not imply x precedes y")


def verify_domain_guards(case: ParameterCase) -> None:
    print("\n=== Domain and failure-mode checks ===")

    expect_domain_error(lambda: shift_c(0, case.k, case.s), "m=0")
    expect_domain_error(lambda: shift_c(case.m, 1, case.s), "k=1")
    expect_domain_error(lambda: shift_c(case.m, case.k, 1), "s=1")
    expect_domain_error(lambda: F(case, case.m, 0), "internal state below range")
    expect_domain_error(lambda: F(case, case.m, case.s + 1), "internal state above range")
    expect_domain_error(lambda: m_start(case, -1), "negative level")
    expect_domain_error(lambda: positional_coordinate(case, m_start(case, 0) - 1, 0), "element before level")
    expect_domain_error(lambda: reconstruct_from_tu(case, 0, -1), "negative u")
    expect_domain_error(lambda: reconstruct_from_tu(case, 0, case.k), "u outside L_0")
    expect_domain_error(lambda: log_coordinate(case, -1), "negative log argument offset")
    expect_domain_error(lambda: decode_from_u(case, 2, level_size(case, 2)), "u equal to level size")
    expect_domain_error(lambda: apply_sequence(case, case.m - 1, (1,)), "root outside A")
    expect_domain_error(lambda: apply_sequence(case, case.m, (case.s + 1,)), "invalid internal state in sequence")

    # Malformed coordinate tuple: right t but u outside the level interval.
    expect_domain_error(lambda: reconstruct_from_tu(case, 1, level_size(case, 1)), "malformed coordinate tuple")

    print("[OK] Invalid parameter, coordinate, state, and reconstruction cases are rejected")


def run_case(case: ParameterCase) -> None:
    require_parameters(case.m, case.k, case.s)
    verify_levels_and_time(case)
    verify_positional_bijection_and_reconstruction(case)
    verify_edge_recursion_and_inverse(case)
    verify_genealogical_decoding(case)
    verify_logarithmic_coordinate(case)
    verify_coordinate_completeness(case)
    verify_causal_compatibility_and_future_slices(case)
    verify_domain_guards(case)


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of coordinate basis of causal structure (sec:coordinate-basis-causal-structure) ===")
    verify_symbolic_core_identities()
    for case in PARAMETER_GRID:
        run_case(case)
    print("\n=== Coordinate-basis causal-structure verification completed successfully ===")


if __name__ == "__main__":
    main()
