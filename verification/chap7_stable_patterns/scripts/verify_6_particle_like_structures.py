"""
VERIFICATION of Section: Particle-like recursive structures
(sec:particle-like-structures).

This script provides a mathematical verification block for the classification
gate introduced in 6_particle_like_structures.tex. The gate acts on macroscopic
objects built from the following structural properties:

    recursive stability,
    causal localization and causal containment,
    stable internal structure,
    asymptotic distinguishability from every other candidate.

Verified content
----------------
1. Asymptotic distinguishability:
   two candidate structures are distinguishable exactly when at least one
   invariant in the selected invariant family has a different asymptotic value.

2. Logical properties of distinguishability:
      - irreflexive on valid candidates;
      - symmetric;
      - not transitive in general;
      - complement relation equals equality of the full invariant signature.

3. Candidate status:
   a macroclass-candidate is exactly the conjunction of recursive stability,
   causal localization, causal containment, and stable internal structure.

4. Particle-like status:
   a macroclass is particle-like exactly when it is a candidate and its invariant
   signature is unique among all other candidates in the comparison universe.

5. Necessity of every condition:
   dropping any one of the four candidate conditions prevents particle-like
   status, even if all invariant values are otherwise unique.

6. Non-distinguishable candidates:
   two distinct candidate labels with the same invariant signature fail the
   label-level particle-like predicate; at the physical quotient level, they
   collapse to one observable object represented by the common invariant
   signature.

7. Non-candidates do not obstruct distinguishability:
   a non-candidate with the same invariant values is ignored by the universal
   quantifier over other macroclass-candidates.

8. Invariant-set exactness:
   missing invariant values are rejected; a difference outside the selected
   invariant family does not create asymptotic distinguishability.

9. Recursive identity:
   arbitrary admissible recursive refinements preserve identity exactly when
   they preserve the complete invariant signature and remain candidates.

10. Transition to quantitative characteristics:
    microhistorical multiplicity and combinatorial mass are treated only as
    admissible next-stage quantities for particle-like structures; no formula for
    either quantity is introduced here.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from typing import Iterable, Mapping, Sequence

from sympy import And, Equivalent, Or, simplify_logic, symbols


InvariantValue = tuple[Fraction, ...]
InvariantName = str


@dataclass(frozen=True)
class InvariantProfile:
    """Asymptotic invariant data of a macroclass-candidate."""

    values: Mapping[InvariantName, InvariantValue]

    def signature(self, invariant_family: Sequence[InvariantName]) -> tuple[InvariantValue, ...]:
        missing = [name for name in invariant_family if name not in self.values]
        if missing:
            raise ValueError(f"missing invariant values: {missing}")
        return tuple(self.values[name] for name in invariant_family)

    def restricted(self, invariant_family: Sequence[InvariantName]) -> "InvariantProfile":
        return InvariantProfile({name: self.values[name] for name in invariant_family})


@dataclass(frozen=True)
class MacroclassStructure:
    """Logical status and invariant profile of a finite/asymptotic macroclass."""

    label: str
    recursively_stable: bool
    causally_localized: bool
    causally_contained: bool
    internally_stable: bool
    profile: InvariantProfile

    def is_candidate(self) -> bool:
        return (
            self.recursively_stable
            and self.causally_localized
            and self.causally_contained
            and self.internally_stable
        )

    def signature(self, invariant_family: Sequence[InvariantName]) -> tuple[InvariantValue, ...]:
        return self.profile.signature(invariant_family)


def scalar(x: int | Fraction) -> InvariantValue:
    return (Fraction(x),)


def vector(*xs: int | Fraction) -> InvariantValue:
    return tuple(Fraction(x) for x in xs)


def asymptotically_distinguishable(
    p: MacroclassStructure,
    q: MacroclassStructure,
    invariant_family: Sequence[InvariantName],
) -> bool:
    """Definition: there exists an invariant with different asymptotic value."""
    if not p.is_candidate() or not q.is_candidate():
        raise ValueError("asymptotic distinguishability is defined here only for candidates")
    p_sig = p.signature(invariant_family)
    q_sig = q.signature(invariant_family)
    return any(a != b for a, b in zip(p_sig, q_sig))


def all_other_candidates(
    p: MacroclassStructure,
    universe: Iterable[MacroclassStructure],
) -> list[MacroclassStructure]:
    return [q for q in universe if q.label != p.label and q.is_candidate()]


def particle_like(
    p: MacroclassStructure,
    universe: Sequence[MacroclassStructure],
    invariant_family: Sequence[InvariantName],
) -> bool:
    """Definition 6: candidate status plus distinguishability from every other candidate."""
    if not p.is_candidate():
        return False
    # Validate the candidate's own invariant profile before the universal check.
    p.signature(invariant_family)
    for q in all_other_candidates(p, universe):
        if not asymptotically_distinguishable(p, q, invariant_family):
            return False
    return True


def invariant_signature_quotient_classes(
    universe: Sequence[MacroclassStructure],
    invariant_family: Sequence[InvariantName],
) -> dict[tuple[InvariantValue, ...], list[str]]:
    """Group candidates by complete invariant signature."""
    classes: dict[tuple[InvariantValue, ...], list[str]] = {}
    for p in universe:
        if p.is_candidate():
            classes.setdefault(p.signature(invariant_family), []).append(p.label)
    return classes


def preserves_recursive_identity(
    base: MacroclassStructure,
    refinements: Sequence[MacroclassStructure],
    invariant_family: Sequence[InvariantName],
) -> bool:
    """
    Recursive identity is profile-based: every admissible refinement must remain
    a candidate and preserve the full asymptotic invariant signature.
    """
    if not base.is_candidate():
        return False
    base_sig = base.signature(invariant_family)
    return all(ref.is_candidate() and ref.signature(invariant_family) == base_sig for ref in refinements)


def quantitative_characteristics_are_admissible(
    p: MacroclassStructure,
    universe: Sequence[MacroclassStructure],
    invariant_family: Sequence[InvariantName],
) -> bool:
    """
    Section 6 only opens the gate for later quantities.  It does not define a
    microhistorical multiplicity or mass formula; admissibility is exactly the
    particle-like predicate.
    """
    return particle_like(p, universe, invariant_family)


def expect_value_error(callable_obj, description: str) -> None:
    try:
        callable_obj()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError was not raised: {description}")


INVARIANTS: tuple[InvariantName, ...] = (
    "rho_limit",
    "r_O1_class",
    "frequency_limit",
    "internal_scalar_limit",
)


P = MacroclassStructure(
    label="P",
    recursively_stable=True,
    causally_localized=True,
    causally_contained=True,
    internally_stable=True,
    profile=InvariantProfile(
        {
            "rho_limit": scalar(Fraction(1, 3)),
            "r_O1_class": scalar(2),
            "frequency_limit": vector(Fraction(2, 3), Fraction(1, 3)),
            "internal_scalar_limit": scalar(Fraction(-1, 6)),
            "non_invariant_color": scalar(17),
        }
    ),
)

Q = MacroclassStructure(
    label="Q",
    recursively_stable=True,
    causally_localized=True,
    causally_contained=True,
    internally_stable=True,
    profile=InvariantProfile(
        {
            "rho_limit": scalar(Fraction(1, 3)),
            "r_O1_class": scalar(2),
            "frequency_limit": vector(Fraction(1, 3), Fraction(2, 3)),
            "internal_scalar_limit": scalar(Fraction(1, 6)),
        }
    ),
)

R = MacroclassStructure(
    label="R",
    recursively_stable=True,
    causally_localized=True,
    causally_contained=True,
    internally_stable=True,
    profile=InvariantProfile(
        {
            "rho_limit": scalar(Fraction(5, 8)),
            "r_O1_class": scalar(1),
            "frequency_limit": vector(Fraction(1, 2), Fraction(1, 2)),
            "internal_scalar_limit": scalar(0),
        }
    ),
)

P_CLONE = replace(P, label="P_clone")

NON_CANDIDATE_SAME_PROFILE = replace(
    P,
    label="unstable_same_profile",
    recursively_stable=False,
)


def verify_symbolic_gate_logic() -> None:
    print("\n=== Symbolic verification of the particle-like gate ===")

    RS, CL, CC, IS, Dist = symbols("RS CL CC IS Dist")
    candidate = And(RS, CL, CC, IS)
    particle = And(candidate, Dist)

    assert bool(simplify_logic(Equivalent(particle, And(RS, CL, CC, IS, Dist))))
    assert bool(simplify_logic(Equivalent(candidate, And(RS, CL, CC, IS))))

    # Necessity: toggling any one candidate component to False forces failure.
    for vals in product([False, True], repeat=5):
        rs, cl, cc, ins, dist = vals
        expected = rs and cl and cc and ins and dist
        evaluated = bool(particle.subs({RS: rs, CL: cl, CC: cc, IS: ins, Dist: dist}))
        assert evaluated == expected

    print("[OK] Particle-like status is exactly the conjunction of the four base conditions and distinguishability")
    print("[OK] No single base condition can be dropped without changing the predicate")


def verify_asymptotic_distinguishability_relation() -> None:
    print("\n=== Verification of asymptotic distinguishability ===")

    universe = [P, Q, R]
    for a in universe:
        assert not asymptotically_distinguishable(a, a, INVARIANTS)

    for a, b in product(universe, repeat=2):
        assert asymptotically_distinguishable(a, b, INVARIANTS) == asymptotically_distinguishable(b, a, INVARIANTS)

    # Non-transitivity witness: A differs from B, B differs from C, but A and C
    # share the selected invariant signature.
    A = replace(P, label="A")
    B = replace(Q, label="B")
    C = replace(P, label="C")
    assert asymptotically_distinguishable(A, B, INVARIANTS)
    assert asymptotically_distinguishable(B, C, INVARIANTS)
    assert not asymptotically_distinguishable(A, C, INVARIANTS)

    e1, e2, e3, e4 = symbols("e1 e2 e3 e4")
    distinguishable_expr = Or(~e1, ~e2, ~e3, ~e4)
    same_signature_expr = And(e1, e2, e3, e4)
    assert bool(simplify_logic(Equivalent(~distinguishable_expr, same_signature_expr)))

    print("[OK] Distinguishability is irreflexive and symmetric on valid candidates")
    print("[OK] Distinguishability is not transitive; the complement is equality of the full invariant signature")


def verify_candidate_and_particle_like_status() -> None:
    print("\n=== Verification of candidate and particle-like status ===")

    universe = [P, Q, R]
    assert P.is_candidate() and Q.is_candidate() and R.is_candidate()
    assert particle_like(P, universe, INVARIANTS)
    assert particle_like(Q, universe, INVARIANTS)
    assert particle_like(R, universe, INVARIANTS)

    # Every single missing base condition prevents candidate and particle-like status.
    variants = [
        replace(P, label="not_recursively_stable", recursively_stable=False),
        replace(P, label="not_causally_localized", causally_localized=False),
        replace(P, label="not_causally_contained", causally_contained=False),
        replace(P, label="not_internally_stable", internally_stable=False),
    ]
    for v in variants:
        assert not v.is_candidate()
        assert not particle_like(v, universe + [v], INVARIANTS)

    # A candidate with the same invariant signature as another candidate is not
    # particle-like at the label level.
    duplicate_universe = [P, P_CLONE, Q]
    assert not asymptotically_distinguishable(P, P_CLONE, INVARIANTS)
    assert not particle_like(P, duplicate_universe, INVARIANTS)
    assert not particle_like(P_CLONE, duplicate_universe, INVARIANTS)
    assert particle_like(Q, duplicate_universe, INVARIANTS)

    # A non-candidate with the same invariant signature does not obstruct P.
    universe_with_non_candidate = [P, Q, NON_CANDIDATE_SAME_PROFILE]
    assert not NON_CANDIDATE_SAME_PROFILE.is_candidate()
    assert particle_like(P, universe_with_non_candidate, INVARIANTS)

    print("[OK] Candidate status is the exact conjunction of the four inherited structural conditions")
    print("[OK] Particle-like status holds exactly for candidates uniquely distinguishable among other candidates")
    print("[OK] Non-candidates are ignored by the universal comparison over other macroclass-candidates")


def verify_physical_quotient_by_invariant_signature() -> None:
    print("\n=== Verification of invariant-signature quotient ===")

    universe = [P, P_CLONE, Q, R, NON_CANDIDATE_SAME_PROFILE]
    classes = invariant_signature_quotient_classes(universe, INVARIANTS)

    p_sig = P.signature(INVARIANTS)
    q_sig = Q.signature(INVARIANTS)
    r_sig = R.signature(INVARIANTS)

    assert set(classes[p_sig]) == {"P", "P_clone"}
    assert classes[q_sig] == ["Q"]
    assert classes[r_sig] == ["R"]
    assert len(classes) == 3

    # At the quotient level, each class is an observable object identified by
    # its full asymptotic profile, not by duplicate labels.
    quotient_objects = set(classes.keys())
    assert quotient_objects == {p_sig, q_sig, r_sig}

    print("[OK] Non-distinguishable candidate labels collapse to one invariant-signature class")
    print("[OK] Distinguishable candidate profiles remain separated in the quotient")


def verify_invariant_family_exactness() -> None:
    print("\n=== Verification of invariant-family exactness ===")

    # Difference only outside the selected invariant family cannot distinguish.
    changed_nonselected_profile = InvariantProfile(
        {
            "rho_limit": scalar(Fraction(1, 3)),
            "r_O1_class": scalar(2),
            "frequency_limit": vector(Fraction(2, 3), Fraction(1, 3)),
            "internal_scalar_limit": scalar(Fraction(-1, 6)),
            "non_invariant_color": scalar(999),
        }
    )
    changed_nonselected = replace(P, label="changed_nonselected", profile=changed_nonselected_profile)
    assert not asymptotically_distinguishable(P, changed_nonselected, INVARIANTS)

    # Difference in any selected invariant is sufficient.
    for invariant in INVARIANTS:
        modified = dict(P.profile.values)
        if invariant == "frequency_limit":
            modified[invariant] = vector(Fraction(1, 4), Fraction(3, 4))
        else:
            modified[invariant] = scalar(Fraction(99, 7))
        q = replace(P, label=f"different_{invariant}", profile=InvariantProfile(modified))
        assert asymptotically_distinguishable(P, q, INVARIANTS)

    missing_profile = InvariantProfile(
        {
            "rho_limit": scalar(Fraction(1, 3)),
            "r_O1_class": scalar(2),
            "frequency_limit": vector(Fraction(2, 3), Fraction(1, 3)),
        }
    )
    missing = replace(P, label="missing_internal_scalar", profile=missing_profile)
    expect_value_error(lambda: particle_like(missing, [missing, Q], INVARIANTS), "missing invariant value")

    # Empty invariant family: no two candidates can be distinguished.
    assert not asymptotically_distinguishable(P, Q, tuple())
    assert not particle_like(P, [P, Q], tuple())

    print("[OK] Only selected invariants contribute to asymptotic distinguishability")
    print("[OK] Missing invariant values are rejected and an empty invariant family cannot separate two candidates")


def verify_recursive_identity_under_refinement() -> None:
    print("\n=== Verification of recursive identity under arbitrary refinements ===")

    refinement_1 = replace(P, label="P_refinement_1")
    refinement_2 = replace(P, label="P_refinement_2")
    assert preserves_recursive_identity(P, [refinement_1, refinement_2], INVARIANTS)

    changed_profile = replace(Q, label="P_refinement_changed_profile")
    assert not preserves_recursive_identity(P, [refinement_1, changed_profile], INVARIANTS)

    non_admissible_refinement = replace(P, label="P_refinement_non_admissible", causally_contained=False)
    assert not preserves_recursive_identity(P, [refinement_1, non_admissible_refinement], INVARIANTS)

    print("[OK] Recursive identity is preserved by all admissible refinements with the same full invariant signature")
    print("[OK] Changing an invariant or losing admissibility breaks recursive identity")


def verify_quantitative_characteristics_gate() -> None:
    print("\n=== Verification of the gate to quantitative characteristics ===")

    universe = [P, Q, R]
    assert quantitative_characteristics_are_admissible(P, universe, INVARIANTS)

    non_particle_cases = [
        replace(P, label="unstable", recursively_stable=False),
        replace(P, label="not_localized", causally_localized=False),
        replace(P, label="not_contained", causally_contained=False),
        replace(P, label="internally_blurred", internally_stable=False),
    ]
    for case in non_particle_cases:
        assert not quantitative_characteristics_are_admissible(case, universe + [case], INVARIANTS)

    duplicate_universe = [P, P_CLONE, Q]
    assert not quantitative_characteristics_are_admissible(P, duplicate_universe, INVARIANTS)

    # No numerical mass or multiplicity is produced in this section.
    # The only verified statement is admissibility for later definitions.
    assert quantitative_characteristics_are_admissible(P, universe, INVARIANTS) == particle_like(P, universe, INVARIANTS)

    print("[OK] Microhistorical multiplicity and combinatorial mass are gated by particle-like status only")
    print("[OK] No independent mass or multiplicity formula is introduced in this verification block")



def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of particle-like recursive structures (sec:particle-like-structures) ===")
    verify_symbolic_gate_logic()
    verify_asymptotic_distinguishability_relation()
    verify_candidate_and_particle_like_status()
    verify_physical_quotient_by_invariant_signature()
    verify_invariant_family_exactness()
    verify_recursive_identity_under_refinement()
    verify_quantitative_characteristics_gate()
    print("\n=== Particle-like recursive structures verification completed successfully ===")


if __name__ == "__main__":
    main()
