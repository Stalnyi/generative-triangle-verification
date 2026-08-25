"""
VERIFICATION of Section 9: Macroscopic indistinguishability and emergent stochasticity
(sec:emergent-stochasticity).

This script provides a verification block for the claims in
9_emergent_stochasticity.tex.  It does not re-prove the previously verified
coordinate-spectrum, digit-coordinate, discrete-interval, Lorentzian-form, or
scale-self-similarity results.  It uses them as dependencies and verifies the
new coarse-grained layer introduced in this section.

Verified content
----------------
1. Deterministic finite microhistory count:
      |S^N| = s^N,
   together with exact bijection between internal-state prefixes, digit
   coordinates D in {0,...,s^N-1}, and normalized points rho = D/s^N.

2. Macroscopic equivalence induced by a finite resolution:
   a half-open coordinate partition of [0,1) defines a genuine equivalence
   relation on finite deterministic microhistories.

3. Exact degeneracy count for a macroscopic coordinate region [a,b):
      N_macro(s,N;[a,b)) = #{D : a <= D/s^N < b}
                          = ceil(b s^N) - ceil(a s^N).
   The script verifies this formula against exhaustive enumeration.

4. Exact two-rounding finite count and error control:
      N_macro in {floor((b-a)s^N), ceil((b-a)s^N)},
      |N_macro - (b-a)s^N| < 1,
   hence, for fixed epsilon=b-a,
      N_macro = epsilon s^N + O(1).

5. Exact finite lower bound used in the chaos-like mechanism:
   from the ceil formula,
      N_macro(s,N;[a,b)) >= (b-a)s^N - 1.
   Hence, if epsilon s^N >= 4, then
      N_macro >= (1/2) epsilon s^N
   without relying on an unspecified O(1) constant.

6. Asymptotic Theta statement with an explicit finite-scale threshold:
   if epsilon s^N >= 4, then
      (1/2) epsilon s^N <= N_macro <= 2 epsilon s^N.

7. Finite-scale guard:
   a positive lower Theta bound cannot be required uniformly on all small
   finite scales for arbitrary half-open regions.  The script exhibits a
   valid small region containing no grid point.

8. Coarse-grained counting measure:
      P_N([a,b)) = N_macro / s^N
   satisfies
      |P_N([a,b)) - (b-a)| < s^(-N),
   so the effective stochastic description is the push-forward of exact
   finite counting over deterministic microhistories.

9. Partition consistency:
   counts over a finite half-open macroscopic partition add exactly to s^N,
   and the induced probabilities add exactly to 1.

10. Exponential degeneracy and deterministic extensions:
   a macroscopic region with at least M indistinguishable N-prefixes produces
      M*s^h
   pairwise distinct deterministic continuations at depth N+h.  Distinct
   prefixes have disjoint continuation sets because their first N states differ.

11. Sensitivity to early internal-state changes:
   changing the j-th state by one unit changes D by s^(N-j) for fixed tail,
   hence changes rho by s^(-j), while the number of possible tails beyond
   that position is s^(N-j).

12. No external randomness:
   the same internal-state prefix always gives the same D, rho, and interval
   data.  All probabilities verified here are finite counting push-forwards,
   not additional primitive random variables.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Iterable, Tuple, Sequence

from sympy import simplify, symbols


StatePrefix = Tuple[int, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_value_error(fn, description: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError was not raised: {description}")


def ceil_fraction(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def floor_fraction(x: Fraction) -> int:
    return x.numerator // x.denominator


def validate_s_N(s: int, N: int) -> None:
    if not isinstance(s, int) or s < 2:
        raise ValueError("The base s must be an integer at least 2.")
    if not isinstance(N, int) or N < 1:
        raise ValueError("The depth N must be a positive integer.")


def validate_prefix(prefix: StatePrefix, s: int, N: int | None = None) -> None:
    if N is not None and len(prefix) != N:
        raise ValueError("The internal-state prefix has the wrong length.")
    if len(prefix) == 0:
        raise ValueError("The internal-state prefix must be nonempty at positive depth.")
    for state in prefix:
        if not isinstance(state, int) or not (1 <= state <= s):
            raise ValueError("An internal state lies outside {1,...,s}.")


def digit_from_prefix(prefix: StatePrefix, s: int) -> int:
    validate_prefix(prefix, s)
    D = 0
    for state in prefix:
        D = s * D + (state - 1)
    return D


def prefix_from_digit(D: int, s: int, N: int) -> StatePrefix:
    validate_s_N(s, N)
    if not isinstance(D, int) or not (0 <= D <= s**N - 1):
        raise ValueError("The digit coordinate is outside the N-depth range.")
    digits = []
    remaining = D
    for power in range(N - 1, -1, -1):
        place = s**power
        d = remaining // place
        remaining -= d * place
        digits.append(d + 1)
    prefix = tuple(digits)
    validate_prefix(prefix, s, N)
    require(digit_from_prefix(prefix, s) == D, "Digit decoding failed.")
    return prefix


def rho_from_digit(D: int, s: int, N: int) -> Fraction:
    validate_s_N(s, N)
    if not isinstance(D, int) or not (0 <= D <= s**N - 1):
        raise ValueError("The digit coordinate is outside the normalized spectrum.")
    return Fraction(D, s**N)


def all_prefixes(s: int, N: int) -> Iterable[StatePrefix]:
    validate_s_N(s, N)
    return product(range(1, s + 1), repeat=N)


def validate_extension_depth(h: int) -> None:
    if not isinstance(h, int) or h < 0:
        raise ValueError("The extension depth h must be a nonnegative integer.")


def deterministic_extensions(prefix: StatePrefix, s: int, h: int) -> tuple[StatePrefix, ...]:
    validate_prefix(prefix, s)
    validate_extension_depth(h)
    return tuple(prefix + tuple(tail) for tail in product(range(1, s + 1), repeat=h))


@dataclass(frozen=True)
class MacroRegion:
    left: Fraction
    right: Fraction

    def __post_init__(self) -> None:
        if not (Fraction(0) <= self.left < self.right <= Fraction(1)):
            raise ValueError("A macroscopic region must be a nonempty half-open subinterval of [0,1).")

    @property
    def width(self) -> Fraction:
        return self.right - self.left

    def contains_rho(self, rho: Fraction) -> bool:
        return self.left <= rho < self.right

    def count_grid_points(self, s: int, N: int) -> int:
        validate_s_N(s, N)
        scale = s**N
        left_index = ceil_fraction(self.left * scale)
        right_index = ceil_fraction(self.right * scale)
        return max(0, right_index - left_index)

    def enumerate_grid_points(self, s: int, N: int) -> list[int]:
        validate_s_N(s, N)
        return [
            D
            for D in range(s**N)
            if self.contains_rho(rho_from_digit(D, s, N))
        ]

    def probability(self, s: int, N: int) -> Fraction:
        return Fraction(self.count_grid_points(s, N), s**N)


def resolution_partition(q: int) -> list[MacroRegion]:
    if not isinstance(q, int) or q < 1:
        raise ValueError("The number of macroscopic cells must be positive.")
    return [
        MacroRegion(Fraction(j, q), Fraction(j + 1, q))
        for j in range(q)
    ]


def validate_half_open_partition(partition: Sequence[MacroRegion]) -> None:
    if len(partition) == 0:
        raise ValueError("The macroscopic partition must be nonempty.")
    expected_left = Fraction(0)
    for region in partition:
        if region.left != expected_left:
            raise ValueError("The macroscopic partition must be adjacent and gap-free.")
        expected_left = region.right
    if expected_left != Fraction(1):
        raise ValueError("The macroscopic partition must cover [0,1).")


def sample_half_open_partitions() -> tuple[tuple[MacroRegion, ...], ...]:
    return (
        tuple(resolution_partition(5)),
        (
            MacroRegion(Fraction(0), Fraction(1, 7)),
            MacroRegion(Fraction(1, 7), Fraction(3, 5)),
            MacroRegion(Fraction(3, 5), Fraction(9, 10)),
            MacroRegion(Fraction(9, 10), Fraction(1)),
        ),
        (
            MacroRegion(Fraction(0), Fraction(2, 9)),
            MacroRegion(Fraction(2, 9), Fraction(1, 3)),
            MacroRegion(Fraction(1, 3), Fraction(8, 11)),
            MacroRegion(Fraction(8, 11), Fraction(1)),
        ),
    )


def macro_cell_index_for_partition(rho: Fraction, partition: Sequence[MacroRegion]) -> int:
    if not (Fraction(0) <= rho < Fraction(1)):
        raise ValueError("rho must lie in [0,1).")
    validate_half_open_partition(partition)
    for index, region in enumerate(partition):
        if region.contains_rho(rho):
            return index
    raise AssertionError("A validated half-open partition failed to cover rho.")


def macro_cell_index(rho: Fraction, q: int) -> int:
    if not (Fraction(0) <= rho < Fraction(1)):
        raise ValueError("rho must lie in [0,1).")
    if q < 1:
        raise ValueError("The resolution denominator must be positive.")
    return min(q - 1, floor_fraction(rho * q))


def macro_equivalent(prefix_a: StatePrefix, prefix_b: StatePrefix, s: int, q: int) -> bool:
    if len(prefix_a) != len(prefix_b):
        raise ValueError("Macroscopic equivalence at fixed depth requires equal depths.")
    N = len(prefix_a)
    validate_prefix(prefix_a, s, N)
    validate_prefix(prefix_b, s, N)
    rho_a = rho_from_digit(digit_from_prefix(prefix_a, s), s, N)
    rho_b = rho_from_digit(digit_from_prefix(prefix_b, s), s, N)
    return macro_cell_index(rho_a, q) == macro_cell_index(rho_b, q)


def macro_equivalent_partition(
    prefix_a: StatePrefix,
    prefix_b: StatePrefix,
    s: int,
    partition: Sequence[MacroRegion],
) -> bool:
    if len(prefix_a) != len(prefix_b):
        raise ValueError("Macroscopic equivalence at fixed depth requires equal depths.")
    N = len(prefix_a)
    validate_prefix(prefix_a, s, N)
    validate_prefix(prefix_b, s, N)
    rho_a = rho_from_digit(digit_from_prefix(prefix_a, s), s, N)
    rho_b = rho_from_digit(digit_from_prefix(prefix_b, s), s, N)
    return macro_cell_index_for_partition(rho_a, partition) == macro_cell_index_for_partition(rho_b, partition)


def interval_data_from_prefix(prefix: StatePrefix, s: int, u_ancestor: int = 0) -> tuple[int, Fraction, int]:
    """Return (Delta t, normalized rho, D)."""
    N = len(prefix)
    validate_s_N(s, N)
    validate_prefix(prefix, s, N)
    if not isinstance(u_ancestor, int) or u_ancestor < 0:
        raise ValueError("The ancestor coordinate must be a nonnegative integer.")
    D = digit_from_prefix(prefix, s)
    return N, rho_from_digit(D, s, N), D


def verify_deterministic_microhistory_bijection() -> None:
    print("\n=== Deterministic finite microhistory count and digit bijection ===")

    checked = 0
    for s in range(2, 7):
        for N in range(1, 7):
            prefixes = list(all_prefixes(s, N))
            digits = [digit_from_prefix(prefix, s) for prefix in prefixes]
            require(len(prefixes) == s**N, "Incorrect number of deterministic prefixes.")
            require(len(set(digits)) == s**N, "Digit map is not injective.")
            require(set(digits) == set(range(s**N)), "Digit map does not fill the causal layer.")
            for D in range(s**N):
                prefix = prefix_from_digit(D, s, N)
                require(digit_from_prefix(prefix, s) == D, "Digit inverse failed.")
                require(rho_from_digit(D, s, N) == Fraction(D, s**N), "rho normalization failed.")
                checked += 1

    print(f"[OK] Checked exact bijections for {checked} digit coordinates across several bases and depths")


def verify_macro_equivalence_relation() -> None:
    print("\n=== Macroscopic equivalence induced by finite resolution ===")

    # Uniform partitions are the simplest finite half-open partitions of [0,1).
    for s, N, q in [(2, 7, 5), (3, 5, 7), (4, 4, 6)]:
        prefixes = list(all_prefixes(s, N))

        for a in prefixes:
            require(macro_equivalent(a, a, s, q), "Reflexivity failed.")

        for a in prefixes[:: max(1, len(prefixes) // 60)]:
            for b in prefixes[:: max(1, len(prefixes) // 55)]:
                ab = macro_equivalent(a, b, s, q)
                ba = macro_equivalent(b, a, s, q)
                require(ab == ba, "Symmetry failed.")

        sample = prefixes[:: max(1, len(prefixes) // 45)]
        for a in sample:
            for b in sample:
                for c in sample:
                    if macro_equivalent(a, b, s, q) and macro_equivalent(b, c, s, q):
                        require(macro_equivalent(a, c, s, q), "Transitivity failed.")

    # Nonuniform finite half-open partitions check the exact formulation used in
    # the manuscript: an arbitrary finite half-open partition of [0,1), not only
    # an equal-width resolution grid.
    checked_nonuniform = 0
    for partition in sample_half_open_partitions():
        validate_half_open_partition(partition)
        for s, N in [(2, 6), (3, 4), (5, 3)]:
            prefixes = list(all_prefixes(s, N))
            sample = prefixes[:: max(1, len(prefixes) // 12)]
            for a in sample:
                require(macro_equivalent_partition(a, a, s, partition), "Partition reflexivity failed.")
                for b in sample:
                    ab = macro_equivalent_partition(a, b, s, partition)
                    ba = macro_equivalent_partition(b, a, s, partition)
                    require(ab == ba, "Partition symmetry failed.")
                    for c in sample:
                        if ab and macro_equivalent_partition(b, c, s, partition):
                            require(
                                macro_equivalent_partition(a, c, s, partition),
                                "Partition transitivity failed.",
                            )
                    checked_nonuniform += 1

    print("[OK] Coarse resolution classes define an equivalence relation on finite deterministic prefixes")
    print(f"[OK] Checked {checked_nonuniform} nonuniform half-open partition witnesses")


def verify_exact_macro_degeneracy_counts() -> None:
    print("\n=== Exact macroscopic degeneracy counts ===")

    regions = [
        MacroRegion(Fraction(0), Fraction(1, 5)),
        MacroRegion(Fraction(1, 7), Fraction(1, 7) + Fraction(2, 9)),
        MacroRegion(Fraction(3, 10), Fraction(9, 10)),
        MacroRegion(Fraction(11, 17), Fraction(15, 17)),
        MacroRegion(Fraction(99, 100), Fraction(1)),
    ]

    checked = 0
    for s in range(2, 8):
        for N in range(1, 8):
            scale = s**N
            for region in regions:
                if region.right > 1:
                    continue
                formula_count = region.count_grid_points(s, N)
                enumerated = region.enumerate_grid_points(s, N)
                require(formula_count == len(enumerated), "Closed-form grid count disagrees with enumeration.")
                expected_main = region.width * scale
                lower_rounding = floor_fraction(expected_main)
                upper_rounding = ceil_fraction(expected_main)
                require(
                    formula_count in {lower_rounding, upper_rounding},
                    "Exact two-rounding estimate N_macro in {floor(epsilon*s^N), ceil(epsilon*s^N)} failed.",
                )
                error = abs(Fraction(formula_count) - expected_main)
                require(error < 1 or error == 0, "Finite boundary error is not controlled by the sharp <1 bound.")
                checked += 1

    print(f"[OK] Verified exact ceil-count formula and finite error bound for {checked} region/base/depth cases")


def verify_aligned_regions_have_exact_counts() -> None:
    print("\n=== Exact counts for grid-aligned macroscopic regions ===")

    checked = 0
    for s in range(2, 7):
        for N in range(2, 8):
            scale = s**N
            for left_index in (0, 1, scale // 3, scale // 2):
                left_index = min(left_index, scale - 2)
                length = max(1, min(scale - left_index, scale // 4))
                region = MacroRegion(Fraction(left_index, scale), Fraction(left_index + length, scale))
                count = region.count_grid_points(s, N)
                require(count == length, "Grid-aligned macroscopic region count is not exact.")
                require(region.probability(s, N) == region.width, "Grid-aligned probability does not equal interval width.")
                checked += 1

    print(f"[OK] Checked {checked} grid-aligned regions with exact degeneracy and exact probability")


def verify_exact_lower_bound_used_in_chaos_mechanism() -> None:
    print("\n=== Exact lower bound for the chaos-like mechanism ===")

    epsilons = [Fraction(1, 10), Fraction(1, 7), Fraction(2, 5), Fraction(3, 8)]
    starts = [Fraction(0), Fraction(1, 11), Fraction(2, 9), Fraction(5, 13)]

    checked = 0
    for s in range(2, 8):
        for epsilon in epsilons:
            N0 = 1
            while epsilon * (s**N0) < 4:
                N0 += 1

            for N in range(N0, N0 + 8):
                scale = s**N
                main = epsilon * scale
                for start in starts:
                    if start + epsilon <= 1:
                        region = MacroRegion(start, start + epsilon)
                        count = region.count_grid_points(s, N)

                        # Exact lower bound used in the text:
                        # ceil(b*s^N)-ceil(a*s^N) >= b*s^N-(a*s^N+1).
                        require(
                            Fraction(count) >= main - 1,
                            "Exact ceil-based lower bound N_macro >= epsilon*s^N - 1 failed.",
                        )
                        require(
                            main - 1 >= Fraction(1, 2) * main,
                            "Threshold epsilon*s^N>=4 should imply main-1 >= main/2.",
                        )
                        require(
                            Fraction(count) >= Fraction(1, 2) * main,
                            "Textual chaos-mechanism lower bound failed.",
                        )
                        checked += 1

    print(f"[OK] Verified exact ceil-based lower bound in {checked} region/base/depth cases")


def verify_asymptotic_theta_with_explicit_threshold() -> None:
    print("\n=== Asymptotic Theta bound with explicit scale threshold ===")

    epsilons = [Fraction(1, 10), Fraction(1, 7), Fraction(2, 5), Fraction(3, 8)]
    starts = [Fraction(0), Fraction(1, 11), Fraction(2, 9), Fraction(5, 13)]

    checked = 0
    for s in range(2, 8):
        for epsilon in epsilons:
            N0 = 1
            while epsilon * (s**N0) < 4:
                N0 += 1

            for N in range(N0, N0 + 8):
                for start in starts:
                    if start + epsilon <= 1:
                        region = MacroRegion(start, start + epsilon)
                        count = region.count_grid_points(s, N)
                        main = epsilon * (s**N)
                        require(2 * Fraction(count) >= main, "Lower asymptotic constant c1=1/2 failed.")
                        require(Fraction(count) <= 2 * main, "Upper asymptotic constant c2=2 failed.")
                        checked += 1

    print(f"[OK] Verified eventual Theta bounds with c1=1/2 and c2=2 in {checked} cases")


def verify_finite_scale_lower_bound_guard() -> None:
    print("\n=== Finite-scale guard for arbitrary macroscopic regions ===")

    s = 2
    N = 1
    epsilon = Fraction(1, 100)
    region = MacroRegion(Fraction(1, 5), Fraction(1, 5) + epsilon)
    count = region.count_grid_points(s, N)

    require(count == 0, "The finite-scale guard example should contain no grid points.")
    require(epsilon * s**N > 0, "The reference main term must be positive.")

    print("[OK] Arbitrary small-scale regions need an eventual-scale hypothesis for positive lower Theta bounds")


def verify_coarse_grained_counting_measure() -> None:
    print("\n=== Coarse-grained counting measure and convergence to interval length ===")

    regions = [
        MacroRegion(Fraction(1, 13), Fraction(5, 13)),
        MacroRegion(Fraction(0), Fraction(3, 11)),
        MacroRegion(Fraction(7, 19), Fraction(18, 19)),
    ]

    for s in range(2, 8):
        for region in regions:
            previous_bound = None
            for N in range(1, 10):
                probability = region.probability(s, N)
                error = abs(probability - region.width)
                bound = Fraction(1, s**N)
                require(error < bound or error == 0, "Counting-measure error exceeds finite grid bound.")
                if previous_bound is not None:
                    require(bound < previous_bound, "Error bound must shrink with depth.")
                previous_bound = bound

    print("[OK] Push-forward counting probabilities converge to macroscopic interval lengths")


def verify_partition_consistency() -> None:
    print("\n=== Macroscopic partition consistency ===")

    checked = 0
    for s in range(2, 7):
        for N in range(1, 8):
            total = s**N
            for q in range(1, 10):
                partition = resolution_partition(q)
                counts = [region.count_grid_points(s, N) for region in partition]
                require(sum(counts) == total, "Partition counts do not add to the full causal layer.")
                probabilities = [Fraction(c, total) for c in counts]
                require(sum(probabilities, Fraction(0)) == 1, "Partition probabilities do not add to one.")
                checked += 1

            for partition in sample_half_open_partitions():
                validate_half_open_partition(partition)
                counts = [region.count_grid_points(s, N) for region in partition]
                require(sum(counts) == total, "Nonuniform partition counts do not add to the full causal layer.")
                probabilities = [Fraction(c, total) for c in counts]
                require(sum(probabilities, Fraction(0)) == 1, "Nonuniform partition probabilities do not add to one.")
                checked += 1

    print(f"[OK] Checked exact count and probability additivity for {checked} partitions")


def verify_exponential_degeneracy() -> None:
    print("\n=== Exponential growth of indistinguishable deterministic realizations ===")

    for s in range(2, 8):
        epsilon = Fraction(1, 4)
        region = MacroRegion(Fraction(0), epsilon)

        previous_count = None
        for N in range(4, 11):
            scale = s**N
            count = region.count_grid_points(s, N)
            main = epsilon * scale

            require(count > 0, "Expected nonzero macro-degeneracy at this scale.")
            require(Fraction(count) >= main - 2, "Exponential lower envelope failed.")
            require(Fraction(count) <= main + 2, "Exponential upper envelope failed.")
            require(Fraction(count, scale) >= epsilon - Fraction(2, scale), "Normalized lower envelope failed.")
            require(Fraction(count, scale) <= epsilon + Fraction(2, scale), "Normalized upper envelope failed.")

            if previous_count is not None:
                require(count > previous_count, "Degeneracy must strictly increase for the tested fixed region.")
            previous_count = count

        if s % 2 == 0:
            # Exact aligned case epsilon=1/4 for even s after N>=1.
            for N in range(2, 8):
                exact = MacroRegion(Fraction(0), Fraction(1, 4)).count_grid_points(s, N)
                require(exact == s**N // 4, "Aligned exponential degeneracy count failed.")

    print("[OK] Fixed-resolution degeneracy has positive exponential lower and upper envelopes")


def verify_indistinguishable_prefix_extensions_are_distinct() -> None:
    print("\n=== Deterministic extensions of indistinguishable prefixes ===")

    cases = [
        (2, 5, 3, MacroRegion(Fraction(0), Fraction(1, 4))),
        (3, 4, 2, MacroRegion(Fraction(1, 9), Fraction(4, 9))),
        (4, 3, 2, MacroRegion(Fraction(1, 4), Fraction(3, 4))),
        (5, 3, 1, MacroRegion(Fraction(0), Fraction(2, 5))),
    ]

    checked_extensions = 0
    for s, N, h, region in cases:
        grid_digits = region.enumerate_grid_points(s, N)
        prefixes = tuple(prefix_from_digit(D, s, N) for D in grid_digits)
        prefix_set = set(prefixes)
        require(len(prefixes) == region.count_grid_points(s, N), "Region prefix count mismatch.")
        require(len(prefix_set) == len(prefixes), "Distinct grid digits should give distinct prefixes.")

        all_extensions: list[StatePrefix] = []
        for prefix in prefixes:
            extensions = deterministic_extensions(prefix, s, h)
            require(len(extensions) == s**h, "Each prefix must have exactly s^h continuations.")
            for extension in extensions:
                require(extension[:N] == prefix, "A deterministic extension changed its prefix.")
            all_extensions.extend(extensions)

        require(
            len(set(all_extensions)) == len(prefixes) * (s**h),
            "Extensions of distinct prefixes must be pairwise distinct.",
        )
        checked_extensions += len(all_extensions)

    print(f"[OK] Checked {checked_extensions} deterministic continuations with disjoint prefix classes")


def verify_sensitivity_to_early_state_changes() -> None:
    print("\n=== Sensitivity to early internal-state changes ===")

    checked = 0
    for s in range(2, 7):
        for N in range(2, 8):
            base_prefix = tuple([1] * N)
            for j in range(1, N + 1):
                changed = list(base_prefix)
                changed[j - 1] = 2
                changed_prefix = tuple(changed)
                D_base = digit_from_prefix(base_prefix, s)
                D_changed = digit_from_prefix(changed_prefix, s)
                require(D_changed - D_base == s ** (N - j), "Digit sensitivity to an early change failed.")
                require(
                    rho_from_digit(D_changed, s, N) - rho_from_digit(D_base, s, N)
                    == Fraction(1, s**j),
                    "Normalized sensitivity value failed.",
                )
                require(s ** (N - j) == len(list(product(range(1, s + 1), repeat=N - j))), "Tail count failed.")
                checked += 1

    # Symbolic companion identity for a unit state change at position j.
    s_sym, N_sym, j_sym = symbols("s N j", integer=True, positive=True)
    digit_jump = s_sym ** (N_sym - j_sym)
    rho_jump = digit_jump / s_sym**N_sym
    require(simplify(rho_jump - s_sym ** (-j_sym)) == 0, "Symbolic normalized sensitivity failed.")

    print(f"[OK] Checked {checked} finite sensitivity cases and the symbolic rho jump s^(-j)")


def verify_no_external_randomness_guard() -> None:
    print("\n=== Determinism guard: probabilities are counting push-forwards ===")

    prefixes = [
        (1, 1, 1, 1),
        (1, 2, 1, 2),
        (2, 2, 1, 1),
        (2, 1, 2, 1),
    ]
    s = 2
    seen = {}
    for prefix in prefixes:
        first = interval_data_from_prefix(prefix, s)
        second = interval_data_from_prefix(prefix, s)
        require(first == second, "The same deterministic prefix produced different data.")
        seen[prefix] = first

    require(len(set(seen.values())) == len(seen), "Distinct listed prefixes unexpectedly collapsed at exact micro-level.")

    region = MacroRegion(Fraction(1, 4), Fraction(3, 4))
    count = region.count_grid_points(s, 4)
    enumerated = region.enumerate_grid_points(s, 4)
    require(count == len(enumerated), "Macroscopic probability is not a pure count of deterministic cases.")
    require(region.probability(s, 4) == Fraction(count, s**4), "Probability is not the counting push-forward.")

    print("[OK] No primitive randomness is used: effective probabilities are induced by finite counting")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_value_error(lambda: validate_s_N(1, 3), "base s<2")
    expect_value_error(lambda: validate_s_N(2, 0), "depth N=0")
    expect_value_error(lambda: digit_from_prefix((), 2), "empty prefix at positive depth")
    expect_value_error(lambda: digit_from_prefix((1, 3), 2), "invalid internal state")
    expect_value_error(lambda: prefix_from_digit(-1, 2, 4), "negative digit coordinate")
    expect_value_error(lambda: prefix_from_digit(16, 2, 4), "digit coordinate beyond layer")
    expect_value_error(lambda: rho_from_digit(8, 2, 3), "rho from out-of-range digit")
    expect_value_error(lambda: MacroRegion(Fraction(1, 2), Fraction(1, 2)), "degenerate macro-region")
    expect_value_error(lambda: MacroRegion(Fraction(-1, 10), Fraction(1, 10)), "negative macro-region start")
    expect_value_error(lambda: MacroRegion(Fraction(9, 10), Fraction(11, 10)), "macro-region exceeds [0,1)")
    expect_value_error(lambda: resolution_partition(0), "zero-cell partition")
    expect_value_error(lambda: validate_half_open_partition(()), "empty partition")
    expect_value_error(
        lambda: validate_half_open_partition((MacroRegion(Fraction(0), Fraction(1, 3)),)),
        "partition does not cover [0,1)",
    )
    expect_value_error(
        lambda: validate_half_open_partition((
            MacroRegion(Fraction(0), Fraction(1, 3)),
            MacroRegion(Fraction(1, 2), Fraction(1)),
        )),
        "partition has a gap",
    )
    expect_value_error(lambda: macro_cell_index(Fraction(1), 4), "rho=1 outside half-open coordinate range")
    expect_value_error(lambda: macro_equivalent((1, 2), (1, 2, 1), 2, 4), "different finite depths")
    expect_value_error(lambda: interval_data_from_prefix((1, 2), 2, u_ancestor=-1), "negative ancestor coordinate")
    expect_value_error(lambda: deterministic_extensions((1, 2), 2, -1), "negative extension depth")

    print("[OK] All invalid-domain cases are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of macroscopic indistinguishability and emergent stochasticity ===")
    verify_deterministic_microhistory_bijection()
    verify_macro_equivalence_relation()
    verify_exact_macro_degeneracy_counts()
    verify_aligned_regions_have_exact_counts()
    verify_exact_lower_bound_used_in_chaos_mechanism()
    verify_asymptotic_theta_with_explicit_threshold()
    verify_finite_scale_lower_bound_guard()
    verify_coarse_grained_counting_measure()
    verify_partition_consistency()
    verify_exponential_degeneracy()
    verify_indistinguishable_prefix_extensions_are_distinct()
    verify_sensitivity_to_early_state_changes()
    verify_no_external_randomness_guard()
    verify_negative_domain_tests()
    print("\n=== Emergent stochasticity verification completed successfully ===")


if __name__ == "__main__":
    main()
