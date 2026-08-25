"""
VERIFICATION of Section: Cumulative spin coordinate of a causal trajectory
(sec:cumulative-spin-coordinate).

This script provides a full mathematical verification block for the claims in
3_cumulative_spin_coordinate.tex.  It verifies only the new cumulative layer:
the accumulated binary internal-state coordinate, its exact spectrum, balanced
trajectories, and completeness of the accumulated profile.  It uses the already
established binary centered spectrum

    eta(1) = -1/2,     eta(2) = +1/2,

as a dependency and does not re-prove the previous centered-spectrum results.

Verified content
----------------
1. Definition of the cumulative coordinate:
      Sigma_n = sum_{j=1}^n eta(sigma_j).

2. Explicit count formula:
      Sigma_n = (N_+ - N_-)/2
   with N_+ + N_- = n.

3. Equivalent closed form using q = N_+:
      Sigma_n = q - n/2.

4. Sharp bounds:
      -n/2 <= Sigma_n <= n/2,
   with both endpoints attained by pure internal-state sequences.

5. Exact level spectrum:
      Spec_n(Sigma) = {-n/2, -n/2 + 1, ..., n/2}.
   The verification checks both symbolic structure and exhaustive finite
   enumeration.

6. Multiplicity strengthening of the exact spectrum:
      value q - n/2 occurs exactly binomial(n,q) times for a fixed initial
      element, and |A| * binomial(n,q) times across the whole initial set.
   This is not an added assumption; it is the finite combinatorial count behind
   attainability of every spectral value.

7. Balanced trajectories:
      Sigma_n = 0 is possible exactly for even n.
   The script also checks the exact number of balanced internal-state sequences.

8. Cumulative profile:
      S_profile(x) = (Sigma_0, Sigma_1, ..., Sigma_n),
      Sigma_0 = 0.

9. Profile completeness:
      the internal-state sequence determines the cumulative profile, and the
      profile recovers the sequence uniquely through increments +/-1/2.

10. Domain and failure checks:
      invalid states, wrong lengths, invalid profile increments, nonzero
      Sigma_0, wrong endpoint, and degenerate state alphabets are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb
from typing import Callable, Iterable, Sequence

from sympy import Rational, binomial, simplify, symbols, summation


State = int
Value = Fraction


ETA = {
    1: Fraction(-1, 2),
    2: Fraction(1, 2),
}


@dataclass(frozen=True)
class GenealogicalCode:
    initial: int
    states: tuple[State, ...]

    @property
    def depth(self) -> int:
        return len(self.states)

    @property
    def sigma(self) -> Value:
        return cumulative_spin(self.states)

    @property
    def cumulative_profile(self) -> tuple[Value, ...]:
        return cumulative_profile(self.states)


def expect_raises(exc_type: type[BaseException], operation: Callable[[], object]) -> None:
    try:
        operation()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}, but the operation completed")


def validate_state_sequence(states: Sequence[State], expected_length: int | None = None) -> tuple[State, ...]:
    seq = tuple(states)
    if expected_length is not None and len(seq) != expected_length:
        raise ValueError(f"Expected length {expected_length}, got {len(seq)}")
    invalid = [state for state in seq if state not in ETA]
    if invalid:
        raise ValueError(f"Invalid internal states: {invalid}")
    return seq


def eta(state: State) -> Value:
    if state not in ETA:
        raise ValueError(f"Invalid internal state: {state}")
    return ETA[state]


def counts_pm(states: Sequence[State]) -> tuple[int, int]:
    seq = validate_state_sequence(states)
    plus = seq.count(2)
    minus = seq.count(1)
    assert plus + minus == len(seq)
    return plus, minus


def cumulative_spin(states: Sequence[State], expected_length: int | None = None) -> Value:
    seq = validate_state_sequence(states, expected_length=expected_length)
    plus = seq.count(2)
    minus = seq.count(1)
    return Fraction(plus - minus, 2)


def cumulative_profile(states: Sequence[State]) -> tuple[Value, ...]:
    seq = validate_state_sequence(states)
    values = [Fraction(0)]
    total = Fraction(0)
    for state in seq:
        total += eta(state)
        values.append(total)
    return tuple(values)


def recover_states_from_profile(profile: Sequence[Value]) -> tuple[State, ...]:
    prof = tuple(Fraction(v) for v in profile)
    if not prof:
        raise ValueError("A cumulative profile must contain Sigma_0")
    if prof[0] != 0:
        raise ValueError("A cumulative profile must start at Sigma_0 = 0")

    states: list[State] = []
    for previous, current in zip(prof, prof[1:]):
        increment = current - previous
        if increment == Fraction(-1, 2):
            states.append(1)
        elif increment == Fraction(1, 2):
            states.append(2)
        else:
            raise ValueError(f"Invalid cumulative-profile increment: {increment}")
    return tuple(states)


def validate_code(code: GenealogicalCode, initial_set: set[int], expected_depth: int | None = None) -> None:
    if code.initial not in initial_set:
        raise ValueError("Initial element is not in A")
    validate_state_sequence(code.states, expected_length=expected_depth)


def all_state_sequences(n: int) -> Iterable[tuple[State, ...]]:
    if n < 0:
        raise ValueError("Depth must be nonnegative")
    return product((1, 2), repeat=n)


def spectrum_expected(n: int) -> set[Value]:
    if n < 0:
        raise ValueError("Depth must be nonnegative")
    return {Fraction(k, 1) - Fraction(n, 2) for k in range(n + 1)}


def verify_symbolic_count_formula() -> None:
    print("\n=== Symbolic verification of the cumulative coordinate formula ===")

    n, q = symbols("n q", integer=True, nonnegative=True)
    N_plus, N_minus = symbols("N_plus N_minus", integer=True, nonnegative=True)

    sigma_from_counts = Rational(1, 2) * (N_plus - N_minus)
    sigma_from_q = sigma_from_counts.subs({N_plus: q, N_minus: n - q})
    assert simplify(sigma_from_q - (q - Rational(1, 2) * n)) == 0

    left_endpoint = sigma_from_q.subs(q, 0)
    right_endpoint = sigma_from_q.subs(q, n)
    assert simplify(left_endpoint + Rational(1, 2) * n) == 0
    assert simplify(right_endpoint - Rational(1, 2) * n) == 0

    q = symbols("q", integer=True, nonnegative=True)
    h = symbols("h", integer=True, nonnegative=True)
    even_zero_value = (q - Rational(1, 2) * n).subs({n: 2 * h, q: h})
    assert simplify(even_zero_value) == 0

    odd_obstruction = (q - Rational(1, 2) * n).subs(n, 2 * h + 1)
    assert simplify(odd_obstruction - (q - h - Rational(1, 2))) == 0

    i = symbols("i", integer=True, nonnegative=True)
    total_binary_prefixes = summation(binomial(n, i), (i, 0, n))
    assert simplify(total_binary_prefixes - 2**n) == 0

    print("[OK] Sigma_n = (N_+ - N_-)/2 and Sigma_n = q - n/2 verified symbolically")
    print("[OK] Endpoint and parity-zero conditions verified symbolically")


def verify_exact_enumeration_and_spectrum() -> None:
    print("\n=== Exact finite enumeration of spectrum and multiplicities ===")

    for n in range(0, 13):
        sequences = list(all_state_sequences(n))
        assert len(sequences) == 2**n

        observed_values = [cumulative_spin(seq) for seq in sequences]
        observed_spectrum = set(observed_values)
        expected_spectrum = spectrum_expected(n)
        assert observed_spectrum == expected_spectrum

        assert len(observed_spectrum) == n + 1
        assert min(observed_spectrum) == Fraction(-n, 2)
        assert max(observed_spectrum) == Fraction(n, 2)

        sorted_values = sorted(observed_spectrum)
        for left, right in zip(sorted_values, sorted_values[1:]):
            assert right - left == 1

        multiplicities = Counter(observed_values)
        for plus_count in range(n + 1):
            value = Fraction(plus_count, 1) - Fraction(n, 2)
            assert multiplicities[value] == comb(n, plus_count)

        assert sum(multiplicities.values()) == 2**n

    print("[OK] Exact spectra checked for n = 0..12")
    print("[OK] Spectral multiplicities match binomial(n,k) for every value")


def verify_explicit_formula_and_bounds() -> None:
    print("\n=== Exact verification of count formula, sharp bounds, and extremizers ===")

    for n in range(0, 15):
        lower = Fraction(-n, 2)
        upper = Fraction(n, 2)

        for seq in all_state_sequences(n):
            plus, minus = counts_pm(seq)
            sigma = cumulative_spin(seq)
            assert sigma == Fraction(plus - minus, 2)
            assert plus + minus == n
            assert lower <= sigma <= upper

        lower_seq = tuple(1 for _ in range(n))
        upper_seq = tuple(2 for _ in range(n))
        assert cumulative_spin(lower_seq) == lower
        assert cumulative_spin(upper_seq) == upper

    print("[OK] Explicit count formula verified over all binary internal-state sequences up to n = 14")
    print("[OK] Bounds are sharp and both endpoints are attained")


def verify_balanced_trajectories() -> None:
    print("\n=== Exact verification of balanced trajectories ===")

    for n in range(0, 15):
        balanced = [seq for seq in all_state_sequences(n) if cumulative_spin(seq) == 0]
        if n % 2 == 0:
            assert len(balanced) == comb(n, n // 2)
            for seq in balanced:
                plus, minus = counts_pm(seq)
                assert plus == minus == n // 2
        else:
            assert len(balanced) == 0

    print("[OK] Sigma_n = 0 occurs exactly for even n")
    print("[OK] Balanced multiplicity equals binomial(n,n/2) when n is even")


def verify_cumulative_profile_completeness() -> None:
    print("\n=== Exact verification of cumulative-profile completeness ===")

    for n in range(0, 12):
        seen_profiles: set[tuple[Value, ...]] = set()

        for seq in all_state_sequences(n):
            prof = cumulative_profile(seq)
            assert len(prof) == n + 1
            assert prof[0] == 0
            assert prof[-1] == cumulative_spin(seq)

            for ell in range(n + 1):
                assert prof[ell] == cumulative_spin(seq[:ell])
                assert Fraction(-ell, 2) <= prof[ell] <= Fraction(ell, 2)

            increments = [prof[j] - prof[j - 1] for j in range(1, n + 1)]
            assert all(increment in {Fraction(-1, 2), Fraction(1, 2)} for increment in increments)

            recovered = recover_states_from_profile(prof)
            assert recovered == seq
            assert cumulative_profile(recovered) == prof
            seen_profiles.add(prof)

        assert len(seen_profiles) == 2**n

    print("[OK] Every internal-state sequence maps to exactly one cumulative profile")
    print("[OK] Every valid cumulative profile recovers exactly one internal-state sequence")


def verify_initial_element_separation() -> None:
    print("\n=== Verification that the profile does not encode the initial element ===")

    initial_set = {10, 11, 12}
    n = 7
    seq = (1, 2, 2, 1, 2, 1, 1)
    codes = [GenealogicalCode(initial=a, states=seq) for a in initial_set]

    for code in codes:
        validate_code(code, initial_set, expected_depth=n)
        assert code.sigma == cumulative_spin(seq)
        assert code.cumulative_profile == cumulative_profile(seq)

    assert len({code.initial for code in codes}) == len(initial_set)
    assert len({code.cumulative_profile for code in codes}) == 1

    # For the whole initial set, each spectral value receives |A| times
    # the fixed-initial multiplicity.
    initial_count = len(initial_set)
    for n in range(0, 9):
        level_codes = [
            GenealogicalCode(initial=a, states=seq)
            for a in initial_set
            for seq in all_state_sequences(n)
        ]
        assert len(level_codes) == initial_count * 2**n
        multiplicities = Counter(code.sigma for code in level_codes)
        for plus_count in range(n + 1):
            value = Fraction(plus_count, 1) - Fraction(n, 2)
            assert multiplicities[value] == initial_count * comb(n, plus_count)

    print("[OK] The cumulative profile encodes the internal-state sequence, not the initial element")
    print("[OK] Level multiplicities scale by |A| exactly")


def verify_profile_domain_failures() -> None:
    print("\n=== Negative domain checks for state sequences and profiles ===")

    expect_raises(ValueError, lambda: cumulative_spin((1, 3, 2)))
    expect_raises(ValueError, lambda: cumulative_spin((0, 1)))
    expect_raises(ValueError, lambda: cumulative_spin((1, 2), expected_length=3))
    expect_raises(ValueError, lambda: validate_code(GenealogicalCode(99, (1, 2)), {1, 2}, expected_depth=2))
    expect_raises(ValueError, lambda: validate_code(GenealogicalCode(1, (1, 2, 1)), {1, 2}, expected_depth=2))

    expect_raises(ValueError, lambda: recover_states_from_profile(()))
    expect_raises(ValueError, lambda: recover_states_from_profile((Fraction(1, 2), Fraction(1))))
    expect_raises(ValueError, lambda: recover_states_from_profile((Fraction(0), Fraction(0))))
    expect_raises(ValueError, lambda: recover_states_from_profile((Fraction(0), Fraction(1))))
    expect_raises(ValueError, lambda: recover_states_from_profile((Fraction(0), Fraction(1, 2), Fraction(1, 4))))

    # Endpoint inconsistency: the same sequence cannot have a profile ending at
    # any value other than its cumulative coordinate.
    seq = (1, 2, 2, 1)
    prof = list(cumulative_profile(seq))
    prof[-1] += Fraction(1, 2)
    expect_raises(ValueError, lambda: recover_states_from_profile(tuple(prof)))

    print("[OK] Invalid internal states and wrong lengths are rejected")
    print("[OK] Invalid cumulative profiles are rejected without vacuous passing")


def verify_prefix_spectral_consistency() -> None:
    print("\n=== Prefix-level spectral consistency of cumulative profiles ===")

    for n in range(0, 11):
        for seq in all_state_sequences(n):
            prof = cumulative_profile(seq)
            for ell in range(n + 1):
                prefix_value = prof[ell]
                assert prefix_value in spectrum_expected(ell)

            # The final spectrum is generated by the last entry of all profiles.
            assert prof[-1] in spectrum_expected(n)

    for ell in range(0, 11):
        attainable_prefix_values = {
            cumulative_profile(seq)[ell]
            for seq in all_state_sequences(ell)
        }
        assert attainable_prefix_values == spectrum_expected(ell)

    print("[OK] Every partial coordinate Sigma_ell lies in the exact spectrum at depth ell")
    print("[OK] Every depth-ell spectral value is attained by a cumulative profile")



def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of cumulative spin coordinate (sec:cumulative-spin-coordinate) ===")
    verify_symbolic_count_formula()
    verify_exact_enumeration_and_spectrum()
    verify_explicit_formula_and_bounds()
    verify_balanced_trajectories()
    verify_cumulative_profile_completeness()
    verify_initial_element_separation()
    verify_profile_domain_failures()
    verify_prefix_spectral_consistency()
    print("\n=== Cumulative spin coordinate verification completed successfully ===")


if __name__ == "__main__":
    main()
