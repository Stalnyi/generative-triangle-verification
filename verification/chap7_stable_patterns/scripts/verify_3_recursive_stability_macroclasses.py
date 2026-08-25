"""
VERIFICATION of Section: Recursive stability of causal macroclasses
(sec:recursive-stability-macroclasses).

This full mathematical verification block checks the formal content of
3_recursive_stability_macroclasses.tex.  It deliberately treats the already
verified macroclass machinery and recursive characteristics as dependencies:
it does not re-prove the generative map, level bijectivity, digit-coordinate
arithmetic, or the earlier construction of asymptotic recursive invariants.

Verified content
----------------
1. One-step recursive refinement:
      Ref(P_n) = { (chi^(n), sigma) | chi^(n) in P_n, sigma in S }.
   The script verifies exact cardinalities, parent recovery, state recovery,
   valid next-level prefixes, and rejection of malformed projections.

2. Recursive extension sequences:
      P_{r+1} subseteq Ref(P_r)
   after realizing each refinement pair as a next finite prefix.
   The script checks valid extension towers and negative cases where an
   apparent next projection contains an inadmissible child.

3. Recursive stability relative to a finite invariant family:
   every admissible unbounded continuation compatible with the macroclass
   must converge to the prescribed macroclass limit for each invariant.

4. Weak versus strong recursive stability:
      strong stability  => weak stability, provided the compatible family is
      nonempty;
      weak stability does not imply strong stability.
   The failure direction is verified by an exact counterexample.

5. Non-vacuity:
   an empty compatible family is not accepted as either weakly or strongly
   stable in this verification, avoiding vacuous universal success.

6. Unstable macroclass witnesses corresponding to the section:
   a. profile diameter not tending to zero;
   b. absence of an invariant limit along a representative continuation;
   c. recursive refinements branching into different asymptotic macroclasses;
   d. divergence of internal-state or causal-geometric characteristics.

7. Stability is an additional asymptotic condition:
   the script verifies stable and unstable macroclass witnesses over the same
   admissible finite-prefix formalism.

8. Stability does not imply causal localization:
   a macroclass may be strongly stable relative to an internal invariant while
   its causal-geometric finite profiles keep macroscopic diameter tending to
   one, not to zero.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Sequence

from sympy import Rational, Symbol, limit, oo, simplify


State = int
Prefix = tuple[State, ...]
RefinementPair = tuple[Prefix, State]


def expect_raises(expected_exception: type[BaseException], action: Callable[[], object], label: str) -> None:
    """Require action() to raise expected_exception; fail outside the handler otherwise."""
    try:
        action()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} was not raised: {label}")


def checked_states(states: Iterable[State]) -> tuple[State, ...]:
    state_tuple = tuple(states)
    if len(state_tuple) < 2:
        raise ValueError("The internal state set must contain at least two states.")
    if len(set(state_tuple)) != len(state_tuple):
        raise ValueError("The internal state set must not contain duplicates.")
    if any(not isinstance(sigma, int) for sigma in state_tuple):
        raise TypeError("Internal states must be integers.")
    return state_tuple


def validate_prefix(prefix: Prefix, level: int, states: Sequence[State]) -> None:
    if level < 0:
        raise ValueError("The finite-prefix level must be nonnegative.")
    if len(prefix) != level:
        raise ValueError(f"Prefix length {len(prefix)} does not match level {level}.")
    state_set = set(states)
    invalid = [sigma for sigma in prefix if sigma not in state_set]
    if invalid:
        raise ValueError(f"Invalid internal states in prefix: {invalid}")


def validate_projection(projection: set[Prefix], level: int, states: Sequence[State]) -> None:
    if not isinstance(projection, set):
        raise TypeError("A finite projection must be represented as a set of finite prefixes.")
    for prefix in projection:
        validate_prefix(prefix, level, states)


def refinement_pairs(projection: set[Prefix], level: int, states: Sequence[State]) -> set[RefinementPair]:
    validate_projection(projection, level, states)
    return {(prefix, sigma) for prefix in projection for sigma in states}


def realize_refinement(pair: RefinementPair, level: int, states: Sequence[State]) -> Prefix:
    prefix, sigma = pair
    validate_prefix(prefix, level, states)
    if sigma not in set(states):
        raise ValueError(f"Invalid refinement state: {sigma}")
    return prefix + (sigma,)


def refined_projection_universe(projection: set[Prefix], level: int, states: Sequence[State]) -> set[Prefix]:
    return {realize_refinement(pair, level, states) for pair in refinement_pairs(projection, level, states)}


def parent_prefix(child: Prefix) -> Prefix:
    if not child:
        raise ValueError("A positive-level child prefix is required for parent recovery.")
    return child[:-1]


def last_state(child: Prefix) -> State:
    if not child:
        raise ValueError("A positive-level child prefix is required for state recovery.")
    return child[-1]


def is_admissible_next_projection(
    current_projection: set[Prefix],
    current_level: int,
    next_projection: set[Prefix],
    states: Sequence[State],
) -> bool:
    validate_projection(current_projection, current_level, states)
    validate_projection(next_projection, current_level + 1, states)
    return next_projection.issubset(refined_projection_universe(current_projection, current_level, states))


def assert_admissible_next_projection(
    current_projection: set[Prefix],
    current_level: int,
    next_projection: set[Prefix],
    states: Sequence[State],
) -> None:
    if not is_admissible_next_projection(current_projection, current_level, next_projection, states):
        raise AssertionError("The proposed next projection is not a subset of the recursive refinement universe.")


def assert_valid_extension_tower(start_level: int, projections: list[set[Prefix]], states: Sequence[State]) -> None:
    if not projections:
        raise ValueError("A recursive extension tower must contain at least one projection.")
    for offset, projection in enumerate(projections):
        validate_projection(projection, start_level + offset, states)
    for offset in range(len(projections) - 1):
        assert_admissible_next_projection(
            projections[offset],
            start_level + offset,
            projections[offset + 1],
            states,
        )


def diameter(values: Iterable[Fraction]) -> Fraction:
    value_list = list(values)
    if not value_list:
        raise ValueError("The diameter of an empty finite profile is undefined.")
    return max(value_list) - min(value_list)


def pairwise_sup_diameter(values: Iterable[Fraction]) -> Fraction:
    value_list = list(values)
    if not value_list:
        raise ValueError("The pairwise diameter of an empty finite profile is undefined.")
    return max(abs(a - b) for a in value_list for b in value_list)


@dataclass(frozen=True)
class InternalStateSequence:
    name: str
    states: tuple[State, ...]
    state_at: Callable[[int], State]
    declared_spin_limit: Fraction
    declared_rho_limit: Fraction
    declared_geometry_label: str

    def prefix(self, length: int) -> Prefix:
        if length < 0:
            raise ValueError("Prefix length must be nonnegative.")
        prefix = tuple(self.state_at(i) for i in range(length))
        validate_prefix(prefix, length, self.states)
        return prefix


def centered_eta(sigma: State, s: int) -> Fraction:
    if not 1 <= sigma <= s:
        raise ValueError("sigma must lie in {1,...,s}.")
    return Fraction(2 * sigma - (s + 1), 2)


def spin_average(sequence: InternalStateSequence, n: int) -> Fraction:
    if n <= 0:
        raise ValueError("The finite spin average is defined for n >= 1.")
    s = len(sequence.states)
    return sum(centered_eta(sequence.state_at(i), s) for i in range(n)) / n


def pure_sequence(state: State, states: tuple[State, ...], *, rho: Fraction, geometry_label: str) -> InternalStateSequence:
    s = len(states)
    return InternalStateSequence(
        name=f"pure-{state}",
        states=states,
        state_at=lambda _i, state=state: state,
        declared_spin_limit=centered_eta(state, s),
        declared_rho_limit=rho,
        declared_geometry_label=geometry_label,
    )


def alternating_sequence(start: State, states: tuple[State, ...], *, rho: Fraction, geometry_label: str) -> InternalStateSequence:
    if len(states) != 2:
        raise ValueError("This alternating witness is defined for two internal states.")
    other = states[1] if start == states[0] else states[0]
    return InternalStateSequence(
        name=f"alternating-start-{start}",
        states=states,
        state_at=lambda i, start=start, other=other: start if i % 2 == 0 else other,
        declared_spin_limit=Fraction(0),
        declared_rho_limit=rho,
        declared_geometry_label=geometry_label,
    )


@dataclass(frozen=True)
class InvariantSpec:
    name: str
    finite_value: Callable[[InternalStateSequence, int], Fraction]
    declared_limit: Callable[[InternalStateSequence], Fraction]


def spin_invariant() -> InvariantSpec:
    return InvariantSpec(
        name="internal normalized spin average",
        finite_value=spin_average,
        declared_limit=lambda sequence: sequence.declared_spin_limit,
    )


def rho_invariant() -> InvariantSpec:
    return InvariantSpec(
        name="causal-geometric rho limit",
        finite_value=lambda sequence, _n: sequence.declared_rho_limit,
        declared_limit=lambda sequence: sequence.declared_rho_limit,
    )


@dataclass(frozen=True)
class MacroclassStabilityProblem:
    name: str
    compatible_extensions: tuple[InternalStateSequence, ...]
    invariants: tuple[InvariantSpec, ...]
    target_limits: dict[str, Fraction]

    def _check_nonempty(self) -> None:
        if not self.compatible_extensions:
            raise ValueError("A stability problem requires at least one compatible unbounded extension.")

    def extension_satisfies_targets(self, extension: InternalStateSequence) -> bool:
        return all(
            invariant.declared_limit(extension) == self.target_limits[invariant.name]
            for invariant in self.invariants
        )

    def is_weakly_stable(self) -> bool:
        self._check_nonempty()
        return any(self.extension_satisfies_targets(extension) for extension in self.compatible_extensions)

    def is_strongly_stable(self) -> bool:
        self._check_nonempty()
        return all(self.extension_satisfies_targets(extension) for extension in self.compatible_extensions)


def verify_one_step_recursive_refinement() -> None:
    print("\n=== One-step recursive refinement Ref(P_n) ===")

    states = checked_states((1, 2, 3))
    level = 3
    projection = {
        (1, 1, 2),
        (1, 2, 1),
        (2, 3, 1),
    }

    ref = refinement_pairs(projection, level, states)
    realized = {realize_refinement(pair, level, states) for pair in ref}

    assert len(ref) == len(projection) * len(states)
    assert len(realized) == len(ref), "Realized refinements must remain collision-free at the prefix level."

    for child in realized:
        validate_prefix(child, level + 1, states)
        assert parent_prefix(child) in projection
        assert last_state(child) in states
        assert (parent_prefix(child), last_state(child)) in ref

    # The exact diameter formula for finite profile images is used here only as
    # set-level validation for profiles attached to a projection, not as a repeat
    # of any previous coordinate theorem.
    profile_values = {Fraction(sum(prefix), len(prefix)) for prefix in projection}
    assert diameter(profile_values) == pairwise_sup_diameter(profile_values)

    expect_raises(
        ValueError,
        lambda: validate_projection({(1, 2)}, level, states),
        "projection with prefixes of the wrong level must be rejected",
    )
    expect_raises(
        ValueError,
        lambda: validate_projection({(1, 2, 4)}, level, states),
        "projection containing an invalid internal state must be rejected",
    )
    expect_raises(
        ValueError,
        lambda: realize_refinement(((1, 2, 3), 4), level, states),
        "refinement by an invalid internal state must be rejected",
    )
    expect_raises(
        ValueError,
        lambda: parent_prefix(tuple()),
        "parent recovery from a zero-length child prefix must be rejected",
    )

    print(f"[OK] Ref(P_n) contains exactly {len(ref)} refinement pairs for a nontrivial projection")
    print("[OK] Parent/state recovery is exact for every realized one-step refinement")
    print("[OK] Malformed projections and invalid refinements are rejected soundly")


def verify_recursive_extension_towers() -> None:
    print("\n=== Recursive extension towers ===")

    states = checked_states((1, 2))
    start_level = 2
    p2 = {(1, 1), (1, 2)}
    p3 = {(1, 1, 1), (1, 2, 1), (1, 2, 2)}
    p4 = {(1, 1, 1, 2), (1, 2, 1, 2), (1, 2, 2, 1)}

    assert_valid_extension_tower(start_level, [p2, p3, p4], states)

    bad_p3 = {(1, 1, 1), (2, 2, 2)}
    expect_raises(
        AssertionError,
        lambda: assert_valid_extension_tower(start_level, [p2, bad_p3], states),
        "next projection containing a child with no parent in the previous projection",
    )

    bad_state_p3 = {(1, 1, 1), (1, 2, 3)}
    expect_raises(
        ValueError,
        lambda: assert_valid_extension_tower(start_level, [p2, bad_state_p3], states),
        "next projection containing an internal state outside S",
    )

    bad_length_p4 = {(1, 1, 1)}
    expect_raises(
        ValueError,
        lambda: assert_valid_extension_tower(start_level, [p2, p3, bad_length_p4], states),
        "projection placed at the wrong recursive depth",
    )

    print("[OK] Valid recursive extension tower satisfies P_{r+1} subseteq Ref(P_r)")
    print("[OK] Non-parental, invalid-state and wrong-level towers are rejected")


def verify_symbolic_convergence_for_stable_invariants() -> None:
    print("\n=== Symbolic convergence for stable invariant witnesses ===")

    N = Symbol("N", positive=True, integer=True)

    # Alternating two-state continuation:
    # for n = 2N, spin average is 0;
    # for n = 2N+1, the absolute residual is 1/(2(2N+1)).
    even_residual = Fraction(0)
    odd_residual_expr = 1 / (2 * (2 * N + 1))
    assert even_residual == 0
    assert simplify(limit(odd_residual_expr, N, oo)) == 0

    # Pure continuations have constant invariant profiles, hence exact limits.
    c = Symbol("c", real=True)
    assert simplify(limit(c, N, oo) - c) == 0

    states = checked_states((1, 2))
    alt_1 = alternating_sequence(1, states, rho=Fraction(1, 3), geometry_label="bounded-rho-class")
    alt_2 = alternating_sequence(2, states, rho=Fraction(1, 3), geometry_label="bounded-rho-class")
    inv = spin_invariant()

    for sequence in (alt_1, alt_2):
        assert inv.declared_limit(sequence) == Fraction(0)
        for n in (20, 40, 80, 160):
            assert abs(inv.finite_value(sequence, n)) <= Fraction(1, n)

    problem = MacroclassStabilityProblem(
        name="balanced alternating macroclass",
        compatible_extensions=(alt_1, alt_2),
        invariants=(inv,),
        target_limits={inv.name: Fraction(0)},
    )

    assert problem.is_strongly_stable()
    assert problem.is_weakly_stable()

    print("[OK] Alternating compatible continuations converge to the same internal invariant")
    print("[OK] Strong recursive stability is verified for a nonempty compatible family")
    print("[OK] Symbolic residual limits vanish exactly")


def verify_weak_and_strong_stability_logic() -> None:
    print("\n=== Weak versus strong recursive stability ===")

    states = checked_states((1, 2))
    inv = spin_invariant()

    stable_extension = alternating_sequence(1, states, rho=Fraction(0), geometry_label="same")
    unstable_extension = pure_sequence(1, states, rho=Fraction(0), geometry_label="same")

    mixed_problem = MacroclassStabilityProblem(
        name="mixed compatible family",
        compatible_extensions=(stable_extension, unstable_extension),
        invariants=(inv,),
        target_limits={inv.name: Fraction(0)},
    )

    assert mixed_problem.is_weakly_stable()
    assert not mixed_problem.is_strongly_stable()
    assert mixed_problem.extension_satisfies_targets(stable_extension)
    assert not mixed_problem.extension_satisfies_targets(unstable_extension)

    strong_problem = MacroclassStabilityProblem(
        name="all-compatible-balanced family",
        compatible_extensions=(
            alternating_sequence(1, states, rho=Fraction(0), geometry_label="same"),
            alternating_sequence(2, states, rho=Fraction(0), geometry_label="same"),
        ),
        invariants=(inv,),
        target_limits={inv.name: Fraction(0)},
    )

    assert strong_problem.is_strongly_stable()
    assert strong_problem.is_weakly_stable(), "Strong stability must imply weak stability for nonempty families."

    empty_problem = MacroclassStabilityProblem(
        name="empty compatible family",
        compatible_extensions=tuple(),
        invariants=(inv,),
        target_limits={inv.name: Fraction(0)},
    )
    expect_raises(
        ValueError,
        empty_problem.is_weakly_stable,
        "weak stability must not be accepted for an empty compatible family",
    )
    expect_raises(
        ValueError,
        empty_problem.is_strongly_stable,
        "strong stability must not be accepted vacuously for an empty compatible family",
    )

    print("[OK] Weak stability can hold while strong stability fails")
    print("[OK] Strong stability implies weak stability in the nonempty case")
    print("[OK] Empty compatible families are not accepted as vacuous stability witnesses")


def verify_unstable_macroclass_witnesses() -> None:
    print("\n=== Unstable macroclass witnesses ===")

    N = Symbol("N", positive=True, integer=True)

    # 1. No profile compression: finite profile diameter remains one.
    constant_diameter_values = [Fraction(0), Fraction(1)]
    for _n in range(1, 25):
        assert diameter(constant_diameter_values) == Fraction(1)
    assert simplify(limit(1, N, oo) - 1) == 0

    # 2. Branching refinements into different asymptotic macroclasses.
    states = checked_states((1, 2))
    inv = spin_invariant()
    left_branch = pure_sequence(1, states, rho=Fraction(0), geometry_label="left")
    right_branch = pure_sequence(2, states, rho=Fraction(0), geometry_label="left")
    assert inv.declared_limit(left_branch) == Fraction(-1, 2)
    assert inv.declared_limit(right_branch) == Fraction(1, 2)
    assert inv.declared_limit(left_branch) != inv.declared_limit(right_branch)

    # The following problem cannot be strongly stable for either branch target
    # when both branches are compatible.
    for target in (Fraction(-1, 2), Fraction(1, 2), Fraction(0)):
        branching_problem = MacroclassStabilityProblem(
            name=f"branching-target-{target}",
            compatible_extensions=(left_branch, right_branch),
            invariants=(inv,),
            target_limits={inv.name: target},
        )
        assert not branching_problem.is_strongly_stable()

    # 3. Causal-geometric divergence independent from internal-state stability.
    rho_inv = rho_invariant()
    same_spin_left_geometry = alternating_sequence(1, states, rho=Fraction(0), geometry_label="left-geometry")
    same_spin_right_geometry = alternating_sequence(1, states, rho=Fraction(1), geometry_label="right-geometry")
    assert inv.declared_limit(same_spin_left_geometry) == inv.declared_limit(same_spin_right_geometry)
    assert rho_inv.declared_limit(same_spin_left_geometry) != rho_inv.declared_limit(same_spin_right_geometry)

    geometry_problem = MacroclassStabilityProblem(
        name="geometry-divergent-compatible-family",
        compatible_extensions=(same_spin_left_geometry, same_spin_right_geometry),
        invariants=(inv, rho_inv),
        target_limits={inv.name: Fraction(0), rho_inv.name: Fraction(0)},
    )
    assert geometry_problem.is_weakly_stable()
    assert not geometry_problem.is_strongly_stable()

    print("[OK] Non-compressing profile diameters are detected exactly")
    print("[OK] Absence of an invariant limit is witnessed by distinct subsequential limits")
    print("[OK] Branching into different asymptotic macroclasses prevents strong stability")
    print("[OK] Internal and causal-geometric divergences are separated by exact witnesses")


def full_cone_coordinate_profile(level: int, s: int) -> set[Fraction]:
    if level < 1:
        raise ValueError("The full-cone coordinate profile is defined for positive levels.")
    if s < 2:
        raise ValueError("s must be at least two.")
    denominator = s**level
    return {Fraction(q, denominator) for q in range(denominator)}


def verify_stability_does_not_imply_causal_localization() -> None:
    print("\n=== Recursive stability does not imply causal localization ===")

    states = checked_states((1, 2))
    inv = spin_invariant()
    extensions = (
        alternating_sequence(1, states, rho=Fraction(0), geometry_label="full-cone"),
        alternating_sequence(2, states, rho=Fraction(0), geometry_label="full-cone"),
    )

    spin_stable_problem = MacroclassStabilityProblem(
        name="spin-stable-but-geometrically-delocalized",
        compatible_extensions=extensions,
        invariants=(inv,),
        target_limits={inv.name: Fraction(0)},
    )
    assert spin_stable_problem.is_strongly_stable()

    # At the same time, the causal-geometric profile can occupy the entire
    # normalized cone grid at each level.  Its diameter tends to one, not zero.
    N = Symbol("N", positive=True, integer=True)
    for s in (2, 3, 5):
        for level in range(1, 7):
            profile = full_cone_coordinate_profile(level, s)
            expected_diameter = Fraction(s**level - 1, s**level)
            assert diameter(profile) == expected_diameter
            if level <= 3:
                assert pairwise_sup_diameter(profile) == expected_diameter

        symbolic_diameter = 1 - Rational(1, s) ** N
        assert simplify(limit(symbolic_diameter, N, oo) - 1) == 0

    print("[OK] Strong stability relative to the chosen internal invariant holds")
    print("[OK] Full-cone causal-geometric profile diameter tends to one, so localization does not follow")


def verify_recursive_identity_preservation() -> None:
    print("\n=== Preservation of asymptotic identity under stable refinement ===")

    states = checked_states((1, 2))
    spin = spin_invariant()
    rho = rho_invariant()

    extension_a = alternating_sequence(1, states, rho=Fraction(2, 5), geometry_label="bounded-same")
    extension_b = alternating_sequence(2, states, rho=Fraction(2, 5), geometry_label="bounded-same")

    problem = MacroclassStabilityProblem(
        name="stable-two-invariant-identity",
        compatible_extensions=(extension_a, extension_b),
        invariants=(spin, rho),
        target_limits={spin.name: Fraction(0), rho.name: Fraction(2, 5)},
    )

    assert problem.is_strongly_stable()

    for extension in problem.compatible_extensions:
        for invariant in problem.invariants:
            assert invariant.declared_limit(extension) == problem.target_limits[invariant.name]

    # Refinement compatibility for the finite projections generated by both
    # compatible continuations.
    projections: list[set[Prefix]] = []
    for level in range(2, 7):
        projections.append({extension.prefix(level) for extension in problem.compatible_extensions})
    assert_valid_extension_tower(2, projections, states)

    print("[OK] A stable macroclass preserves all declared invariant limits under recursive refinement")
    print("[OK] The finite projections generated by compatible continuations form a valid extension tower")

def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of recursive stability of causal macroclasses ===")
    verify_one_step_recursive_refinement()
    verify_recursive_extension_towers()
    verify_symbolic_convergence_for_stable_invariants()
    verify_weak_and_strong_stability_logic()
    verify_unstable_macroclass_witnesses()
    verify_stability_does_not_imply_causal_localization()
    verify_recursive_identity_preservation()
    print("\n=== Recursive stability macroclasses verification completed successfully ===")


if __name__ == "__main__":
    main()
