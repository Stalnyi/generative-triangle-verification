"""
VERIFICATION of Section 6: Strict continuum coordinate limit
(sec:continuum-coordinate-limit).

This script provides a full mathematical verification block for
6_continuum_coordinate_limit.tex.  It checks only the new content of this
section: the passage from exact finite s-adic spectra of causal layers to the
continuum coordinate limit.  It uses the previously verified finite spectrum
formula as an input dependency and does not re-prove the construction of the
generative triangle, causal cones, digit coordinate, or normalized causal
coordinate.

Verified content
----------------
1. Definition of the continuum coordinate limit:
      Sigma_infty(x) = union_{N>=1} Sigma_N(x),
      Sigma_N(x) = {q / s^N : q = 0, ... ,s^N-1}.

2. Nested exact refinement:
      Sigma_N subset Sigma_{N+1},
   and the finite truncation union_{j=1}^N Sigma_j is exactly Sigma_N.

3. Exact refinement cell structure:
   each interval between neighboring coarse points is subdivided into s equal
   parts, with s-1 genuinely new interior points; the terminal cell before 1
   is refined analogously, although 1 itself is never present at finite depth.

4. Density in [0,1):
   for every rational interval (a,b) with 0 <= a < b < 1, the constructive
   proof point
      q0 = floor(a s^N) + 1
   is checked exactly after choosing N such that s^{-N} < b-a.

5. One-sided mesh approximation:
   for every rational r in [0,1], finite spectra provide an exact lower grid
   approximation with error at most s^{-N}; for r=1 the best finite point is
   1 - s^{-N}.

6. Infinite s-adic expansion:
   the greedy digit algorithm produces digits d_j in {0,...,s-1},
      rho = sum_{j=1}^m d_j s^{-j} + remainder_m s^{-m},
   and every partial sum lies in Sigma_m.

7. Non-uniqueness where expected:
   positive finite s-adic rationals have both terminating and repeating
   (s-1)-tail expansions, and the two expansions converge to the same point.

8. Self-similarity of the continuum coordinate limit:
      Sigma_infty = union_{d=0}^{s-1} (d/s + Sigma_infty/s),
   verified by exact left-to-right and right-to-left finite witnesses.

9. The continuum limit is dense but not equal to [0,1):
   explicit rational points such as 1/(s+1) are proved not to belong to
   Sigma_infty, while every interval contains Sigma_infty points.

10. Boundary discipline:
    0 belongs to Sigma_infty; 1 never belongs to Sigma_infty, but is an
    accumulation point through 1 - s^{-N}.

11. Negative domain tests:
    invalid bases, invalid depths, malformed intervals, invalid digits, and
    invalid finite prefixes are rejected by the checked constructors.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Callable, Sequence

from sympy import simplify, symbols, summation


def expect_raises(exc_type: type[BaseException], fn: Callable, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            f"Expected {exc_type.__name__}, but got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {exc_type.__name__}, but no exception was raised")


def require_integer(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def validate_base(s: int) -> None:
    require_integer(s, "s")
    if s < 2:
        raise ValueError("s must satisfy s >= 2")


def validate_depth(N: int) -> None:
    require_integer(N, "N")
    if N < 1:
        raise ValueError("N must satisfy N >= 1")


def validate_digit(d: int, s: int) -> None:
    validate_base(s)
    require_integer(d, "digit")
    if not (0 <= d <= s - 1):
        raise ValueError("digit is outside {0,...,s-1}")


def validate_prefix(prefix: Sequence[int], s: int) -> None:
    validate_base(s)
    if len(prefix) == 0:
        raise ValueError("finite prefix must be nonempty")
    for d in prefix:
        validate_digit(d, s)


def spectrum(s: int, N: int) -> list[Fraction]:
    validate_base(s)
    validate_depth(N)
    denominator = s**N
    return [Fraction(k, denominator) for k in range(denominator)]


def spectrum_set(s: int, N: int) -> set[Fraction]:
    return set(spectrum(s, N))


def truncation_union(s: int, N: int) -> set[Fraction]:
    validate_base(s)
    validate_depth(N)
    acc: set[Fraction] = set()
    for depth in range(1, N + 1):
        acc.update(spectrum(s, depth))
    return acc


def integer_to_digits(k: int, s: int, N: int) -> tuple[int, ...]:
    validate_base(s)
    validate_depth(N)
    if not isinstance(k, int):
        raise ValueError("k must be an integer")
    if not (0 <= k <= s**N - 1):
        raise ValueError("k is outside the finite spectrum range")
    digits = []
    remainder = k
    for power in range(N - 1, -1, -1):
        place = s**power
        digit = remainder // place
        digits.append(digit)
        remainder -= digit * place
    assert remainder == 0
    for d in digits:
        validate_digit(d, s)
    return tuple(digits)


def digits_to_integer(prefix: Sequence[int], s: int) -> int:
    validate_prefix(prefix, s)
    value = 0
    for d in prefix:
        value = s * value + d
    return value


def prefix_value(prefix: Sequence[int], s: int) -> Fraction:
    validate_prefix(prefix, s)
    N = len(prefix)
    return Fraction(digits_to_integer(prefix, s), s**N)


def is_in_spectrum(rho: Fraction, s: int, N: int) -> bool:
    validate_base(s)
    validate_depth(N)
    scaled = rho * (s**N)
    return scaled.denominator == 1 and 0 <= scaled.numerator <= s**N - 1


def construct_density_point(s: int, a: Fraction, b: Fraction) -> tuple[int, int, Fraction]:
    validate_base(s)
    if not (Fraction(0) <= a < b < Fraction(1)):
        raise ValueError("density interval must satisfy 0 <= a < b < 1")
    width = b - a
    N = 1
    while Fraction(1, s**N) >= width:
        N += 1
    scaled_a = a * (s**N)
    q0 = scaled_a.numerator // scaled_a.denominator + 1
    rho = Fraction(q0, s**N)
    assert a < rho < b
    assert 0 <= q0 <= s**N - 1
    assert is_in_spectrum(rho, s, N)
    return N, q0, rho


def lower_grid_approximation(s: int, N: int, r: Fraction) -> Fraction:
    validate_base(s)
    validate_depth(N)
    if not (Fraction(0) <= r <= Fraction(1)):
        raise ValueError("r must lie in [0,1]")
    denominator = s**N
    scaled = r * denominator
    k = scaled.numerator // scaled.denominator
    if k == denominator:
        k = denominator - 1
    rho = Fraction(k, denominator)
    assert is_in_spectrum(rho, s, N)
    return rho


@dataclass(frozen=True)
class GreedyExpansion:
    rho: Fraction
    base: int
    digits: tuple[int, ...]
    remainders: tuple[Fraction, ...]

    def partial_sum(self, m: int) -> Fraction:
        if m < 1 or m > len(self.digits):
            raise ValueError("m must select a nonempty available prefix")
        return prefix_value(self.digits[:m], self.base)

    def reconstruction_error(self, m: int) -> Fraction:
        return self.rho - self.partial_sum(m)


def greedy_expansion(rho: Fraction, s: int, depth: int) -> GreedyExpansion:
    validate_base(s)
    validate_depth(depth)
    if not (Fraction(0) <= rho < Fraction(1)):
        raise ValueError("rho must lie in [0,1)")
    remainder = rho
    digits: list[int] = []
    remainders: list[Fraction] = [remainder]
    for _ in range(depth):
        scaled = s * remainder
        digit = scaled.numerator // scaled.denominator
        validate_digit(digit, s)
        remainder = scaled - digit
        assert Fraction(0) <= remainder < Fraction(1)
        digits.append(digit)
        remainders.append(remainder)
    return GreedyExpansion(rho=rho, base=s, digits=tuple(digits), remainders=tuple(remainders))


def nonterminating_tail_value(s: int, start_index: int, tail_length: int) -> Fraction:
    """Finite approximation to sum_{j=start_index}^{infty} (s-1)s^{-j}."""
    validate_base(s)
    validate_depth(start_index)
    if tail_length < 1:
        raise ValueError("tail_length must be positive")
    return sum(Fraction(s - 1, s**j) for j in range(start_index, start_index + tail_length))


def exact_self_similarity_left_witness(rho: Fraction, s: int, N: int) -> tuple[int, Fraction]:
    """Represent rho in Sigma_N as m/s + eta/s with eta in Sigma_infty."""
    validate_base(s)
    validate_depth(N)
    if not is_in_spectrum(rho, s, N):
        raise ValueError("rho must belong to Sigma_N")
    k = (rho * s**N).numerator
    if N == 1:
        m = k
        eta = Fraction(0)
    else:
        block = s ** (N - 1)
        m = k // block
        q = k % block
        eta = Fraction(q, block)
        assert is_in_spectrum(eta, s, N - 1)
    assert 0 <= m <= s - 1
    assert rho == Fraction(m, s) + eta / s
    return m, eta


def exact_self_similarity_right_value(m: int, eta: Fraction, s: int, eta_depth: int) -> Fraction:
    validate_base(s)
    validate_depth(eta_depth)
    validate_digit(m, s)
    if not is_in_spectrum(eta, s, eta_depth):
        raise ValueError("eta must belong to the selected finite spectrum")
    rho = Fraction(m, s) + eta / s
    assert is_in_spectrum(rho, s, eta_depth + 1)
    return rho


def prove_fraction_not_in_limit(rho: Fraction, s: int) -> None:
    validate_base(s)
    if not (Fraction(0) <= rho < Fraction(1)):
        raise ValueError("rho must lie in [0,1)")
    # This helper is specialized to rho = 1/(s+1), the fixed witness used below.
    if rho != Fraction(1, s + 1):
        raise ValueError("this proof helper expects rho = 1/(s+1)")
    # If 1/(s+1) = k/s^N, then k(s+1) = s^N.  Since gcd(s+1, s^N)=1,
    # this would force s+1 to divide 1, impossible for s >= 2.
    for N in range(1, 20):
        assert gcd(s + 1, s**N) == 1
        assert not is_in_spectrum(Fraction(1, s + 1), s, N)


def verify_definition_and_nested_refinement() -> None:
    print("\n=== Continuum-limit definition and nested finite refinements ===")

    for s in range(2, 7):
        for N in range(1, 6):
            sigma_N = spectrum_set(s, N)
            sigma_next = spectrum_set(s, N + 1)
            assert sigma_N.issubset(sigma_next)

            truncated = truncation_union(s, N)
            assert truncated == sigma_N
            assert len(truncated) == s**N

            assert Fraction(0) in sigma_N
            assert Fraction(1) not in sigma_N
            assert max(sigma_N) == Fraction(s**N - 1, s**N)
            assert Fraction(1) - max(sigma_N) == Fraction(1, s**N)

    q, N = symbols("q N", integer=True, positive=True)
    s_sym = symbols("s", integer=True, positive=True)
    # Algebraic refinement identity:
    # q/s^N = (q*s)/s^(N+1), hence every old point is retained.
    assert simplify(q / s_sym**N - (q * s_sym) / s_sym**(N + 1)) == 0

    print("[OK] Sigma_N is nested in Sigma_{N+1}")
    print("[OK] finite truncation union_{j<=N} Sigma_j equals Sigma_N")
    print("[OK] 0 is always present; 1 is never finite-level present but is approached by 1-s^{-N}")


def verify_exact_cell_refinement() -> None:
    print("\n=== Exact cell-level spectral refinement ===")

    for s in range(2, 7):
        for N in range(1, 5):
            old = spectrum(s, N)
            refined = spectrum_set(s, N + 1)
            old_den = s**N

            for j in range(0, old_den - 1):
                left = Fraction(j, old_den)
                right = Fraction(j + 1, old_den)
                interior = [
                    Fraction(j * s + offset, s ** (N + 1))
                    for offset in range(1, s)
                ]
                assert all(left < point < right for point in interior)
                assert all(point in refined for point in interior)
                assert all(point not in old for point in interior)
                assert len(interior) == s - 1

            # Terminal cell [last old point, 1) is also refined, but 1 is excluded.
            left = old[-1]
            terminal = [
                Fraction((old_den - 1) * s + offset, s ** (N + 1))
                for offset in range(1, s)
            ]
            assert all(left < point < Fraction(1) for point in terminal)
            assert all(point in refined for point in terminal)
            assert all(point not in old for point in terminal)
            assert Fraction(1) not in refined

            new_points = refined - set(old)
            assert len(new_points) == (s - 1) * s**N

    print("[OK] each coarse cell receives exactly s-1 new refined points")
    print("[OK] the terminal cell before 1 is refined while 1 remains excluded")


def verify_density_constructively() -> None:
    print("\n=== Constructive exact density in [0,1) ===")

    intervals = [
        (Fraction(0), Fraction(1, 3)),
        (Fraction(1, 10), Fraction(11, 100)),
        (Fraction(7, 19), Fraction(8, 19)),
        (Fraction(97, 100), Fraction(999, 1000)),
        (Fraction(123, 1000), Fraction(124, 1000)),
        (Fraction(999_001, 1_000_000), Fraction(999_999, 1_000_000)),
    ]

    checked = 0
    for s in range(2, 12):
        for a, b in intervals:
            N, q0, rho = construct_density_point(s, a, b)
            assert a < rho < b
            assert rho in spectrum_set(s, N)
            assert Fraction(1, s**N) < b - a
            assert q0 == (a * s**N).numerator // (a * s**N).denominator + 1
            checked += 1

    # Dense rational stress grid.
    for s in range(2, 8):
        rationals = sorted({Fraction(i, q) for q in range(2, 18) for i in range(0, q)})
        for a, b in zip(rationals, rationals[1:]):
            if a < b < 1:
                _, _, rho = construct_density_point(s, a, b)
                assert a < rho < b
                checked += 1

    print(f"[OK] constructed exact spectral points in {checked} rational intervals")
    print("[OK] the proof choice q0=floor(a*s^N)+1 always lands inside (a,b)")


def verify_mesh_approximation_and_boundary() -> None:
    print("\n=== Mesh approximation and boundary discipline ===")

    samples = sorted({
        Fraction(0),
        Fraction(1),
        Fraction(1, 2),
        Fraction(2, 3),
        Fraction(17, 31),
        Fraction(999, 1000),
        Fraction(1, 97),
    })

    for s in range(2, 10):
        for N in range(1, 8):
            mesh = Fraction(1, s**N)
            for r in samples:
                rho = lower_grid_approximation(s, N, r)
                assert is_in_spectrum(rho, s, N)
                assert rho <= r
                assert Fraction(0) <= r - rho <= mesh
                if r < 1:
                    assert r - rho < mesh
                else:
                    assert rho == Fraction(1) - mesh
                    assert r - rho == mesh

    for base in (2, 3, 10):
        previous = Fraction(1, base)
        for depth in range(2, 20):
            current = Fraction(1, base**depth)
            assert current < previous
            previous = current
        assert Fraction(1, base**19) < Fraction(1, 100_000)

    print("[OK] every sampled r in [0,1] has lower spectral approximants with error <= s^{-N}")
    print("[OK] 1 is not included but is approached by the sequence 1-s^{-N}")


def verify_greedy_infinite_expansion() -> None:
    print("\n=== Greedy infinite s-adic expansion and finite partial spectra ===")

    rho_samples = [
        Fraction(0),
        Fraction(1, 7),
        Fraction(2, 7),
        Fraction(5, 13),
        Fraction(17, 29),
        Fraction(999, 1000),
    ]

    checked = 0
    for s in range(2, 11):
        for rho in rho_samples:
            if rho >= 1:
                continue
            expansion = greedy_expansion(rho, s, depth=14)
            assert expansion.remainders[0] == rho

            for m in range(1, 15):
                prefix = expansion.digits[:m]
                partial = expansion.partial_sum(m)
                remainder = expansion.remainders[m]

                assert is_in_spectrum(partial, s, m)
                assert all(0 <= d <= s - 1 for d in prefix)
                assert rho == partial + remainder * Fraction(1, s**m)
                assert Fraction(0) <= remainder < Fraction(1)
                assert Fraction(0) <= rho - partial < Fraction(1, s**m)
                checked += 1

    print(f"[OK] verified greedy expansion identities for {checked} exact rational partial sums")
    print("[OK] every checked partial sum belongs to the corresponding finite spectrum Sigma_m")


def verify_expected_nonuniqueness_of_expansions() -> None:
    print("\n=== Expected non-uniqueness of s-adic expansions at finite s-adic rationals ===")

    checked = 0
    for s in range(2, 9):
        for N in range(1, 5):
            upper = min(s**N - 1, 25)
            for k in range(1, upper + 1):
                rho = Fraction(k, s**N)
                terminating_digits = integer_to_digits(k, s, N)
                predecessor_digits = integer_to_digits(k - 1, s, N)

                assert prefix_value(terminating_digits, s) == rho
                assert prefix_value(predecessor_digits, s) == rho - Fraction(1, s**N)

                for tail_length in (1, 2, 5, 10, 20):
                    alternative_partial = (
                        prefix_value(predecessor_digits, s)
                        + nonterminating_tail_value(s, N + 1, tail_length)
                    )
                    assert alternative_partial < rho
                    assert rho - alternative_partial == Fraction(1, s ** (N + tail_length))

                checked += 1

    j = symbols("j", positive=True, integer=True)
    s_sym, N_sym, M_sym = symbols("s N M", positive=True, integer=True)
    # Sum_{j=1}^{M} (s-1)s^{-(N+j)} = s^{-N}(1-s^{-M}).
    # This verifies the general finite repeating-tail truncation, not only a
    # fixed three-term special case.
    finite_tail_M = summation((s_sym - 1) * s_sym ** (-(N_sym + j)), (j, 1, M_sym))
    expected_tail_M = s_sym ** (-N_sym) * (1 - s_sym ** (-M_sym))
    assert simplify(finite_tail_M - expected_tail_M) == 0

    # Consequently, after M repeated maximal digits beyond depth N, the exact
    # remaining error to the terminating value is s^{-(N+M)}.
    tail_error_M = s_sym ** (-(N_sym + M_sym))
    assert simplify(
        s_sym ** (-N_sym) - expected_tail_M - tail_error_M
    ) == 0

    print(f"[OK] checked {checked} finite s-adic rationals with terminating and repeating-tail expansions")
    print("[OK] the repeating (s-1)-tail converges to the same value with exact error s^{-(N+M)}")


def verify_self_similarity_of_limit() -> None:
    print("\n=== Self-similarity of the continuum coordinate limit ===")

    left_checked = 0
    right_checked = 0

    for s in range(2, 7):
        for N in range(1, 5):
            for rho in spectrum(s, N):
                m, eta = exact_self_similarity_left_witness(rho, s, N)
                assert rho == Fraction(m, s) + eta / s
                assert eta == 0 or any(is_in_spectrum(eta, s, depth) for depth in range(1, N))
                left_checked += 1

            for eta_depth in range(1, 5):
                for eta in spectrum(s, eta_depth):
                    for m in range(s):
                        rho = exact_self_similarity_right_value(m, eta, s, eta_depth)
                        assert is_in_spectrum(rho, s, eta_depth + 1)
                        right_checked += 1

    print(f"[OK] left-to-right self-similarity witnesses checked: {left_checked}")
    print(f"[OK] right-to-left self-similarity witnesses checked: {right_checked}")


def verify_dense_but_not_equal_to_continuum() -> None:
    print("\n=== Dense but not equal to the full interval ===")

    for s in range(2, 15):
        witness = Fraction(1, s + 1)
        prove_fraction_not_in_limit(witness, s)

        # Despite not being present, every neighborhood around the witness has
        # exact spectral points by the density construction.
        for radius in (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000)):
            a = max(Fraction(0), witness - radius)
            b = min(Fraction(1) - Fraction(1, 10_000_000), witness + radius)
            if a < witness < b:
                _, _, rho = construct_density_point(s, a, b)
                assert a < rho < b

    print("[OK] 1/(s+1) is never an s-adic finite-ratio point for base s")
    print("[OK] neighborhoods of such missing points still contain exact spectral points")


def verify_finite_level_is_not_continuum() -> None:
    print("\n=== Guard against identifying any finite spectrum with the continuum ===")

    for s in range(2, 7):
        for N in range(1, 5):
            sigma_N = spectrum_set(s, N)
            assert len(sigma_N) == s**N
            assert Fraction(1, s**N * 2) not in sigma_N
            assert Fraction(1, s + 1) not in sigma_N

            # There is always a nonempty rational interval between consecutive
            # finite spectral points containing no finite-level spectral point.
            ordered = spectrum(s, N)
            if len(ordered) >= 2:
                left = ordered[0]
                right = ordered[1]
                midpoint = (left + right) / 2
                assert left < midpoint < right
                assert midpoint not in sigma_N

    print("[OK] finite spectra remain finite grids and are never [0,1)")
    print("[OK] gaps exist at every finite depth even though the infinite union is dense")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(ValueError, spectrum, 1, 3)
    expect_raises(ValueError, spectrum, 2, 0)
    expect_raises(ValueError, spectrum, 2.5, 3)
    expect_raises(ValueError, spectrum, 2, 1.5)

    expect_raises(ValueError, construct_density_point, 2, Fraction(1, 2), Fraction(1, 2))
    expect_raises(ValueError, construct_density_point, 2, Fraction(-1, 10), Fraction(1, 2))
    expect_raises(ValueError, construct_density_point, 2, Fraction(1, 2), Fraction(1))
    expect_raises(ValueError, construct_density_point, 2, Fraction(3, 4), Fraction(1, 2))

    expect_raises(ValueError, validate_digit, -1, 2)
    expect_raises(ValueError, validate_digit, 2, 2)
    expect_raises(ValueError, validate_prefix, [], 2)
    expect_raises(ValueError, prefix_value, [0, 1, 2], 2)

    expect_raises(ValueError, greedy_expansion, Fraction(1), 2, 5)
    expect_raises(ValueError, greedy_expansion, Fraction(-1, 3), 2, 5)
    expect_raises(ValueError, greedy_expansion, Fraction(1, 3), 1, 5)
    expect_raises(ValueError, greedy_expansion, Fraction(1, 3), 2, 0)

    expect_raises(ValueError, lower_grid_approximation, 2, 3, Fraction(-1, 10))
    expect_raises(ValueError, lower_grid_approximation, 2, 3, Fraction(11, 10))

    expect_raises(ValueError, exact_self_similarity_left_witness, Fraction(1, 3), 2, 2)
    expect_raises(ValueError, exact_self_similarity_right_value, 2, Fraction(0), 2, 1)
    expect_raises(ValueError, exact_self_similarity_right_value, 0, Fraction(1, 3), 2, 1)

    print("[OK] invalid bases, depths, intervals, digits, prefixes and spectral witnesses are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of strict continuum coordinate limit (sec:continuum-coordinate-limit) ===")
    verify_definition_and_nested_refinement()
    verify_exact_cell_refinement()
    verify_density_constructively()
    verify_mesh_approximation_and_boundary()
    verify_greedy_infinite_expansion()
    verify_expected_nonuniqueness_of_expansions()
    verify_self_similarity_of_limit()
    verify_dense_but_not_equal_to_continuum()
    verify_finite_level_is_not_continuum()
    verify_negative_domain_tests()
    print("\n=== Continuum coordinate limit verification completed successfully ===")


if __name__ == "__main__":
    main()
