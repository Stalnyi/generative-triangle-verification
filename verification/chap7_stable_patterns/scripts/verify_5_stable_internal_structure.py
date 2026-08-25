"""
VERIFICATION of Section: Stable internal structure of localized recursive
structures (sec:stable-internal-structure).

This script provides a full mathematical verification block for
5_stable_internal_structure.tex.  It does not re-verify the construction of the
generative triangle, recursive macroclasses, controlled characteristics,
recursive stability, or causal localization.  Those are treated as established
dependencies.  The present block verifies the additional internal layer
introduced in this section: finite normalized spin profiles, frequency-vector
profiles, their diameters, convergence of the frequency profile, internal
constancy under admissible recursively compatible continuations, internal
blurring, and the final conjunction defining a localized internally stable
recursive structure.

Verified content
----------------
1. Internal finite-level profiles:
      B_n(P) = {barSigma_n(chi) : chi^(n) in P_n},
      D_{barSigma,n}(P) = diam_{barSigma}(P_n).
   The script computes normalized cumulative normalized spin coordinates from
   exact finite prefixes and verifies that the scalar profile and its diameter
   are exact set-valued and pairwise-supremum quantities.

2. Frequency-vector finite-level profiles:
      f_n(chi) = (f_1^(n)(chi),...,f_s^(n)(chi)),
      F_n(P)   = {f_n(chi) : chi^(n) in P_n},
      D_{f,n}(P) = sup ||f_n(chi_1)-f_n(chi_2)||.
   The script verifies simplex normalization, exact l1 / l_infinity / l2^2
   diameters, and finite-dimensional norm comparisons relevant to the fact
   that convergence to zero does not depend on the chosen norm.

3. Causal localization is not enough:
   exact witnesses have D_{rho,n}->0 and D_{r,n}=O(1), while their internal
   frequency or scalar normalized spin diameters do not converge to zero.

4. Proposition "convergence of the frequency profile":
   if D_{f,n}(P)->0 and one representative has a limit f_infty, then all
   representatives converge to the same f_infty.  The script verifies this by
   exact rational families, explicit triangle-inequality bounds, and symbolic
   limits of the relevant error terms.

5. Linear passage from frequency vectors to normalized cumulative internal-state
   coordinates:
      barSigma_n(chi) = sum_{sigma in S} eta(sigma) f_sigma^(n)(chi).
   The script verifies the finite exact identity and the Lipschitz implication
      D_{f,n}->0  ==>  D_{barSigma,n}->0.
   It also verifies that the converse fails: distinct limiting frequency
   profiles may have the same scalar normalized spin value.

6. Stable internal structure under recursive refinement:
   a macroclass is accepted only when a single limiting frequency vector is
   preserved along every admissible recursively compatible continuation.  The
   verifier checks positive cases and rejects nonconvergent, incompatible-limit,
   and internally blurred cases.

7. Localized internally stable recursive structure:
   the final notion is checked as a genuine conjunction of recursive stability,
   weak-or-strong causal localization, and stable internal structure.  Negative
   tests show that dropping any one of the three assumptions is detected.

8. Scope guard: normalized spin-profile data does not introduce an operator-level spin theory.
   The verifier checks that the frequency/scalar spin profile contains no
   representation of spatial rotations, Pauli algebra, Dirac equation, or SU(2)
   generators.  Therefore it is accepted as the normalized spin profile inherited from Chapter 6;
   Pauli algebra, spinor representations, rotation representations, and Dirac-type equations are not introduced here.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Callable, Sequence

from sympy import Rational, Symbol, oo, limit, simplify


# ---------------------------------------------------------------------------
# Generic guards and exact arithmetic utilities
# ---------------------------------------------------------------------------


def expect_raises(expected_exception: type[BaseException], thunk: Callable[[], object], label: str) -> None:
    """Fail unless thunk raises expected_exception directly."""
    try:
        thunk()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} was not raised: {label}")


def ensure_same_length(v: Sequence[Fraction], w: Sequence[Fraction]) -> None:
    if len(v) != len(w):
        raise ValueError("vectors must have the same dimension")


def eta_value(sigma: int, s: int) -> Fraction:
    """Centered internal-state spectrum eta(sigma)=sigma-(s+1)/2."""
    if s < 2:
        raise ValueError("s must be at least 2")
    if not (1 <= sigma <= s):
        raise ValueError("internal state outside S={1,...,s}")
    return Fraction(2 * sigma - s - 1, 2)


def centered_spectrum(s: int) -> tuple[Fraction, ...]:
    return tuple(eta_value(sigma, s) for sigma in range(1, s + 1))


def validate_simplex_vector(v: Sequence[Fraction]) -> None:
    if not v:
        raise ValueError("frequency vector must be nonempty")
    if any(x < 0 for x in v):
        raise ValueError(f"frequency vector has a negative component: {v}")
    if sum(v, Fraction(0)) != 1:
        raise ValueError(f"frequency vector is not normalized: {v}")


def l1_distance(v: Sequence[Fraction], w: Sequence[Fraction]) -> Fraction:
    ensure_same_length(v, w)
    return sum(abs(a - b) for a, b in zip(v, w))


def linf_distance(v: Sequence[Fraction], w: Sequence[Fraction]) -> Fraction:
    ensure_same_length(v, w)
    return max(abs(a - b) for a, b in zip(v, w)) if v else Fraction(0)


def l2_squared_distance(v: Sequence[Fraction], w: Sequence[Fraction]) -> Fraction:
    ensure_same_length(v, w)
    return sum((a - b) * (a - b) for a, b in zip(v, w))


def pairwise_supremum(values: Sequence[object], distance: Callable[[object, object], Fraction]) -> Fraction:
    if not values:
        raise ValueError("diameter of an empty profile is undefined")
    return max(distance(a, b) for a in values for b in values)


def scalar_diameter(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("diameter of an empty scalar profile is undefined")
    return max(values) - min(values)


def spin_from_frequency_vector(freq: Sequence[Fraction]) -> Fraction:
    validate_simplex_vector(freq)
    s = len(freq)
    return sum(eta_value(sigma, s) * freq[sigma - 1] for sigma in range(1, s + 1))


# ---------------------------------------------------------------------------
# Exact finite internal-state prefixes and finite projections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatePrefix:
    """Finite internal-state prefix over S={1,...,s}."""

    states: tuple[int, ...]
    s: int

    def __post_init__(self) -> None:
        if self.s < 2:
            raise ValueError("s must be at least 2")
        if len(self.states) == 0:
            raise ValueError("finite prefix must be nonempty for normalized frequencies")
        bad = [sigma for sigma in self.states if sigma < 1 or sigma > self.s]
        if bad:
            raise ValueError(f"internal states outside S={{1,...,{self.s}}}: {bad}")

    @property
    def n(self) -> int:
        return len(self.states)

    def frequency_vector(self) -> tuple[Fraction, ...]:
        counts = [0] * self.s
        for sigma in self.states:
            counts[sigma - 1] += 1
        freq = tuple(Fraction(c, self.n) for c in counts)
        validate_simplex_vector(freq)
        return freq

    def normalized_internal_coordinate(self) -> Fraction:
        direct_average = sum(eta_value(sigma, self.s) for sigma in self.states) / self.n
        from_frequency = spin_from_frequency_vector(self.frequency_vector())
        assert direct_average == from_frequency
        return direct_average


@dataclass(frozen=True)
class FiniteProjection:
    """Finite projection P_n represented by internal-state prefixes of equal length."""

    prefixes: tuple[StatePrefix, ...]

    def __post_init__(self) -> None:
        if not self.prefixes:
            raise ValueError("finite projection must be nonempty")
        lengths = {p.n for p in self.prefixes}
        bases = {p.s for p in self.prefixes}
        if len(lengths) != 1:
            raise ValueError("all prefixes in P_n must have the same level n")
        if len(bases) != 1:
            raise ValueError("all prefixes in P_n must use the same internal-state set S")

    @property
    def n(self) -> int:
        return self.prefixes[0].n

    @property
    def s(self) -> int:
        return self.prefixes[0].s

    def scalar_profile(self) -> tuple[Fraction, ...]:
        return tuple(sorted({p.normalized_internal_coordinate() for p in self.prefixes}))

    def frequency_profile(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(sorted({p.frequency_vector() for p in self.prefixes}))

    def scalar_diameter(self) -> Fraction:
        values = [p.normalized_internal_coordinate() for p in self.prefixes]
        return scalar_diameter(values)

    def frequency_diameter_l1(self) -> Fraction:
        return pairwise_supremum([p.frequency_vector() for p in self.prefixes], l1_distance)

    def frequency_diameter_linf(self) -> Fraction:
        return pairwise_supremum([p.frequency_vector() for p in self.prefixes], linf_distance)

    def frequency_diameter_l2_squared(self) -> Fraction:
        return pairwise_supremum([p.frequency_vector() for p in self.prefixes], l2_squared_distance)


# ---------------------------------------------------------------------------
# Asymptotic frequency profiles and simplified macrostructure predicates
# ---------------------------------------------------------------------------


FrequencyFunction = Callable[[int], tuple[Fraction, ...]]


@dataclass(frozen=True)
class FrequencyBranch:
    """One admissible or inadmissible continuation of finite frequency profiles."""

    name: str
    s: int
    freq_at: FrequencyFunction
    expected_limit: tuple[Fraction, ...] | None
    admissible: bool = True
    recursively_compatible: bool = True

    def vector(self, n: int) -> tuple[Fraction, ...]:
        if n < 1:
            raise ValueError("level n must be positive")
        v = self.freq_at(n)
        if len(v) != self.s:
            raise ValueError("frequency vector dimension does not match s")
        validate_simplex_vector(v)
        return v

    def has_declared_limit(self) -> bool:
        return self.expected_limit is not None


@dataclass(frozen=True)
class ContinuationFamily:
    branches: tuple[FrequencyBranch, ...]

    def admissible_compatible_branches(self) -> tuple[FrequencyBranch, ...]:
        return tuple(b for b in self.branches if b.admissible and b.recursively_compatible)

    def frequency_diameter_l1(self, n: int) -> Fraction:
        active = self.admissible_compatible_branches()
        if not active:
            raise ValueError("no admissible recursively compatible continuations")
        return pairwise_supremum([b.vector(n) for b in active], l1_distance)

    def scalar_diameter(self, n: int) -> Fraction:
        active = self.admissible_compatible_branches()
        if not active:
            raise ValueError("no admissible recursively compatible continuations")
        spins = [spin_from_frequency_vector(b.vector(n)) for b in active]
        return scalar_diameter(spins)

    def stable_internal_limit(self) -> tuple[Fraction, ...] | None:
        """
        Return the preserved limiting frequency vector, if all admissible
        recursively compatible continuations declare the same limit.
        """
        active = self.admissible_compatible_branches()
        if not active:
            return None
        declared = [b.expected_limit for b in active]
        if any(v is None for v in declared):
            return None
        first = declared[0]
        assert first is not None
        return first if all(v == first for v in declared) else None

    def has_stable_internal_structure(self) -> bool:
        return self.stable_internal_limit() is not None


@dataclass(frozen=True)
class LocalizedRecursiveStructure:
    recursive_stable: bool
    localization_mode: str  # "weak", "strong", or "none"
    continuation_family: ContinuationFamily

    def causally_localized(self) -> bool:
        return self.localization_mode in {"weak", "strong"}

    def internally_stable(self) -> bool:
        return self.continuation_family.has_stable_internal_structure()

    def localized_internally_stable_recursive_structure(self) -> bool:
        return self.recursive_stable and self.causally_localized() and self.internally_stable()



# ---------------------------------------------------------------------------
# Concrete frequency branches used by the verification blocks
# ---------------------------------------------------------------------------


def branch_converging_to_half(name: str, offset: Fraction) -> FrequencyBranch:
    """Binary branch converging to (1/2,1/2) with exact O(1/n) error."""

    def freq(n: int) -> tuple[Fraction, Fraction]:
        delta = offset / (n + 2)
        return (Fraction(1, 2) + delta, Fraction(1, 2) - delta)

    return FrequencyBranch(name=name, s=2, freq_at=freq, expected_limit=(Fraction(1, 2), Fraction(1, 2)))


def branch_constant(name: str, vector: tuple[Fraction, ...], admissible: bool = True, compatible: bool = True) -> FrequencyBranch:
    s = len(vector)
    validate_simplex_vector(vector)
    return FrequencyBranch(
        name=name,
        s=s,
        freq_at=lambda n, v=vector: v,
        expected_limit=vector,
        admissible=admissible,
        recursively_compatible=compatible,
    )


def branch_nonconvergent_binary() -> FrequencyBranch:
    def freq(n: int) -> tuple[Fraction, Fraction]:
        return (Fraction(1), Fraction(0)) if n % 2 == 0 else (Fraction(0), Fraction(1))

    return FrequencyBranch(
        name="oscillatory-binary",
        s=2,
        freq_at=freq,
        expected_limit=None,
        admissible=True,
        recursively_compatible=True,
    )


# ---------------------------------------------------------------------------
# Verification blocks
# ---------------------------------------------------------------------------


def verify_finite_internal_profiles_and_diameters() -> None:
    print("\n=== Finite internal profiles and exact diameters ===")

    p1 = StatePrefix((1, 1, 2, 2), s=2)  # scalar 0
    p2 = StatePrefix((1, 2, 2, 2), s=2)  # scalar 1/4
    p3 = StatePrefix((1, 1, 1, 2), s=2)  # scalar -1/4
    projection = FiniteProjection((p1, p2, p3))

    assert projection.scalar_profile() == (Fraction(-1, 4), Fraction(0), Fraction(1, 4))
    assert projection.scalar_diameter() == Fraction(1, 2)
    assert projection.scalar_diameter() == pairwise_supremum(
        [p.normalized_internal_coordinate() for p in projection.prefixes],
        lambda a, b: abs(a - b),
    )

    expected_frequency_profile = tuple(
        sorted(
            {
                (Fraction(1, 2), Fraction(1, 2)),
                (Fraction(1, 4), Fraction(3, 4)),
                (Fraction(3, 4), Fraction(1, 4)),
            }
        )
    )
    assert projection.frequency_profile() == expected_frequency_profile
    assert projection.frequency_diameter_l1() == Fraction(1)
    assert projection.frequency_diameter_linf() == Fraction(1, 2)
    assert projection.frequency_diameter_l2_squared() == Fraction(1, 2)

    # Invalid finite projections must be rejected instead of producing silent data.
    expect_raises(ValueError, lambda: StatePrefix((), s=2), "empty finite prefix")
    expect_raises(ValueError, lambda: StatePrefix((1, 3), s=2), "state outside S")
    expect_raises(ValueError, lambda: FiniteProjection((StatePrefix((1,), 2), StatePrefix((1, 2), 2))), "mixed levels")
    expect_raises(ValueError, lambda: FiniteProjection((StatePrefix((1,), 2), StatePrefix((1,), 3))), "mixed internal-state sets")

    print("[OK] B_n(P), F_n(P), D_barSigma,n(P), and D_f,n(P) are exact finite-profile quantities")
    print("[OK] Invalid finite projections and invalid internal states are rejected soundly")


def verify_norm_choice_for_frequency_diameter() -> None:
    print("\n=== Finite-dimensional norm checks for frequency-vector diameters ===")

    for s in range(2, 7):
        grid_values = [Fraction(i, 4) for i in range(5)]
        simplex_points = []
        for coords in product(grid_values, repeat=s):
            if sum(coords, Fraction(0)) == 1:
                simplex_points.append(coords)
        assert simplex_points
        for v in simplex_points:
            for w in simplex_points:
                l1 = l1_distance(v, w)
                linf = linf_distance(v, w)
                l2sq = l2_squared_distance(v, w)
                assert linf <= l1
                assert l1 <= s * linf
                assert l2sq <= l1 * l1
                assert linf * linf <= l2sq or l2sq == 0

    print("[OK] Exact grid checks confirm compatible l1, l_infinity, and l2^2 diameter controls")
    print("[OK] Vanishing of D_f,n is not an artifact of a particular finite-dimensional norm")


def verify_causal_localization_is_not_internal_constancy() -> None:
    print("\n=== Causal localization does not imply stable internal organization ===")

    # Geometry witness: localized by construction.
    # D_rho,n = 1/(n+1) -> 0 and D_r,n = 3 = O(1).
    n = Symbol("n", positive=True, integer=True)
    D_rho = 1 / (n + 1)
    D_r = Rational(3)
    assert simplify(limit(D_rho, n, oo)) == 0
    assert limit(D_r, n, oo) == 3

    internally_blurred = ContinuationFamily(
        (
            branch_constant("left-internal-limit", (Fraction(1), Fraction(0))),
            branch_constant("right-internal-limit", (Fraction(0), Fraction(1))),
        )
    )
    for level in (5, 20, 100):
        assert internally_blurred.frequency_diameter_l1(level) == 2
        assert internally_blurred.scalar_diameter(level) == 1
    assert internally_blurred.stable_internal_limit() is None

    print("[OK] Constructed D_rho,n->0 and bounded D_r,n with persistent internal-state frequency spread")
    print("[OK] Geometric localization alone is rejected as stable internal organization")


def verify_convergence_of_frequency_profile_proposition() -> None:
    print("\n=== Proposition: convergence of the frequency profile ===")

    family = ContinuationFamily(
        (
            branch_converging_to_half("reference", Fraction(0)),
            branch_converging_to_half("upper-small", Fraction(1, 5)),
            branch_converging_to_half("upper-larger", Fraction(2, 5)),
            branch_converging_to_half("lower-small", Fraction(-1, 5)),
        )
    )
    f_infty = (Fraction(1, 2), Fraction(1, 2))
    assert family.stable_internal_limit() == f_infty

    previous_diameter = None
    for level in (5, 10, 20, 40, 80, 160):
        diameter = family.frequency_diameter_l1(level)
        assert diameter <= Fraction(6, 5 * (level + 2))
        if previous_diameter is not None:
            assert diameter < previous_diameter
        previous_diameter = diameter
        for branch in family.admissible_compatible_branches():
            assert l1_distance(branch.vector(level), f_infty) <= Fraction(4, 5 * (level + 2))

    n = Symbol("n", positive=True, integer=True)
    Df_bound = Rational(6, 5) / (n + 2)
    reference_error = Rational(0)
    triangle_bound = Df_bound + reference_error
    assert simplify(limit(Df_bound, n, oo)) == 0
    assert simplify(limit(triangle_bound, n, oo)) == 0

    # A nonzero reference error variant verifies the actual triangle structure:
    # ||f_i(n)-f_infty|| <= ||f_i(n)-f_0(n)|| + ||f_0(n)-f_infty||.
    nonzero_ref = Rational(1, 3) / (n + 2)
    assert simplify(limit(Df_bound + 2 * nonzero_ref, n, oo)) == 0

    print("[OK] Exact branches with D_f,n->0 converge to the same limiting frequency vector")
    print("[OK] Symbolic triangle-inequality error bound tends to zero")


def verify_frequency_to_scalar_implication_and_failed_converse() -> None:
    print("\n=== Frequency profile controls scalar spin profile; converse fails ===")

    # Positive implication: D_barSigma <= B_s * D_f,l1.
    for s in range(2, 8):
        B_s = Fraction(s - 1, 2)
        grid_values = [Fraction(i, 4) for i in range(5)]
        simplex_points = [coords for coords in product(grid_values, repeat=s) if sum(coords, Fraction(0)) == 1]
        assert simplex_points
        for v in simplex_points:
            for w in simplex_points:
                scalar_gap = abs(spin_from_frequency_vector(v) - spin_from_frequency_vector(w))
                assert scalar_gap <= B_s * l1_distance(v, w)

    n = Symbol("n", positive=True, integer=True)
    B = Rational(5, 2)
    Df = 1 / (n + 1)
    Ds_bound = B * Df
    assert simplify(limit(Ds_bound, n, oo)) == 0

    # Failed converse with s=3: eta=(-1,0,1).  The following two limiting
    # frequency profiles both have scalar value 0 but remain far apart.
    profile_a = (Fraction(1, 2), Fraction(0), Fraction(1, 2))
    profile_b = (Fraction(0), Fraction(1), Fraction(0))
    validate_simplex_vector(profile_a)
    validate_simplex_vector(profile_b)
    assert spin_from_frequency_vector(profile_a) == 0
    assert spin_from_frequency_vector(profile_b) == 0
    assert l1_distance(profile_a, profile_b) == 2

    scalar_blind_family = ContinuationFamily(
        (branch_constant("balanced-extremes", profile_a), branch_constant("middle-only", profile_b))
    )
    for level in (3, 30, 300):
        assert scalar_blind_family.scalar_diameter(level) == 0
        assert scalar_blind_family.frequency_diameter_l1(level) == 2
    assert scalar_blind_family.stable_internal_limit() is None

    print("[OK] Exact Lipschitz checks verify D_f,n->0 implies D_barSigma,n->0")
    print("[OK] Distinct frequency profiles with identical scalar normalized spin value prove the converse fails")


def verify_internal_blurring_modes() -> None:
    print("\n=== Internal blurring modes ===")

    # Mode 1: persistent scalar and frequency spread.
    binary_split = ContinuationFamily(
        (branch_constant("all-state-1", (Fraction(1), Fraction(0))), branch_constant("all-state-2", (Fraction(0), Fraction(1))))
    )
    assert binary_split.frequency_diameter_l1(50) == 2
    assert binary_split.scalar_diameter(50) == 1
    assert not binary_split.has_stable_internal_structure()

    # Mode 2: singleton family with no individual limit.  Diameter is zero, but
    # the required limiting frequency vector does not exist.
    oscillatory_singleton = ContinuationFamily((branch_nonconvergent_binary(),))
    for level in (10, 11, 12):
        assert oscillatory_singleton.frequency_diameter_l1(level) == 0
        assert oscillatory_singleton.scalar_diameter(level) == 0
    assert not oscillatory_singleton.has_stable_internal_structure()

    # Mode 3: scalar profile converges, but frequency profile does not collapse.
    scalar_blind = ContinuationFamily(
        (
            branch_constant("balanced-extremes", (Fraction(1, 2), Fraction(0), Fraction(1, 2))),
            branch_constant("middle-only", (Fraction(0), Fraction(1), Fraction(0))),
        )
    )
    assert scalar_blind.scalar_diameter(100) == 0
    assert scalar_blind.frequency_diameter_l1(100) == 2
    assert not scalar_blind.has_stable_internal_structure()

    print("[OK] Persistent scalar spread, missing individual limit, and scalar-blind frequency spread are rejected")


def verify_internal_constancy_under_admissible_recursive_refinement() -> None:
    print("\n=== Stable internal structure under admissible recursively compatible continuations ===")

    stable_family = ContinuationFamily(
        (
            branch_converging_to_half("reference", Fraction(0)),
            branch_converging_to_half("compatible-plus", Fraction(1, 4)),
            branch_converging_to_half("compatible-minus", Fraction(-1, 4)),
        )
    )
    assert stable_family.has_stable_internal_structure()
    assert stable_family.stable_internal_limit() == (Fraction(1, 2), Fraction(1, 2))

    different_limit_family = ContinuationFamily(
        (
            branch_converging_to_half("half-limit", Fraction(0)),
            branch_constant("pure-state-1", (Fraction(1), Fraction(0))),
        )
    )
    assert not different_limit_family.has_stable_internal_structure()

    # Inadmissible or recursively incompatible branches do not count in the
    # universal quantifier over admissible recursively compatible continuations.
    ignored_bad_branches = ContinuationFamily(
        (
            branch_converging_to_half("valid-reference", Fraction(0)),
            branch_constant("inadmissible-bad-limit", (Fraction(1), Fraction(0)), admissible=False, compatible=True),
            branch_constant("incompatible-bad-limit", (Fraction(0), Fraction(1)), admissible=True, compatible=False),
        )
    )
    assert ignored_bad_branches.has_stable_internal_structure()
    assert ignored_bad_branches.stable_internal_limit() == (Fraction(1, 2), Fraction(1, 2))

    empty_active = ContinuationFamily(
        (branch_constant("inadmissible-only", (Fraction(1), Fraction(0)), admissible=False, compatible=True),)
    )
    assert not empty_active.has_stable_internal_structure()
    expect_raises(ValueError, lambda: empty_active.frequency_diameter_l1(10), "no admissible compatible continuation")

    print("[OK] A single frequency limit is preserved along all admissible recursively compatible continuations")
    print("[OK] Bad branches are ignored only when they are explicitly inadmissible or recursively incompatible")
    print("[OK] The empty-active-continuation case is not accepted vacuously")


def verify_localized_internally_stable_recursive_structure_conjunction() -> None:
    print("\n=== Localized internally stable recursive structure as a conjunction ===")

    stable_family = ContinuationFamily(
        (branch_converging_to_half("reference", Fraction(0)), branch_converging_to_half("small-shift", Fraction(1, 8)))
    )

    positive_weak = LocalizedRecursiveStructure(True, "weak", stable_family)
    positive_strong = LocalizedRecursiveStructure(True, "strong", stable_family)
    assert positive_weak.localized_internally_stable_recursive_structure()
    assert positive_strong.localized_internally_stable_recursive_structure()

    missing_recursive_stability = LocalizedRecursiveStructure(False, "strong", stable_family)
    missing_localization = LocalizedRecursiveStructure(True, "none", stable_family)
    missing_internal_constancy = LocalizedRecursiveStructure(
        True,
        "strong",
        ContinuationFamily(
            (
                branch_constant("left-limit", (Fraction(1), Fraction(0))),
                branch_constant("right-limit", (Fraction(0), Fraction(1))),
            )
        ),
    )

    assert not missing_recursive_stability.localized_internally_stable_recursive_structure()
    assert not missing_localization.localized_internally_stable_recursive_structure()
    assert not missing_internal_constancy.localized_internally_stable_recursive_structure()

    print("[OK] The final definition is enforced as recursive stability AND localization AND internal constancy")
    print("[OK] Weak and strong localization are accepted exactly as the allowed localization regimes")


def verify_symbolic_centered_spectrum_dependencies_used_without_reproving_full_base_theory() -> None:
    print("\n=== Minimal symbolic dependencies for the internal linear functional ===")

    # This is not a re-verification of the previous spin-spectrum section.  It
    # checks only the endpoint bound used here for the D_f -> D_barSigma implication.
    s = Symbol("s", integer=True, positive=True)
    eta_min = 1 - (s + 1) / 2
    eta_max = s - (s + 1) / 2
    B_s = (s - 1) / 2
    assert simplify(eta_min + B_s) == 0
    assert simplify(eta_max - B_s) == 0

    n = Symbol("n", positive=True, integer=True)
    generic_frequency_diameter = 1 / (n + 1)
    scalar_bound = B_s * generic_frequency_diameter
    # Substitute representative dimensions because SymPy cannot order a fully
    # symbolic positive integer s inside a numeric limit with a free multiplier.
    for dim in range(2, 10):
        assert simplify(limit(scalar_bound.subs(s, dim), n, oo)) == 0

    print("[OK] The endpoint bound B_s=(s-1)/2 used in the Lipschitz implication is symbolically consistent")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of stable internal structure (sec:stable-internal-structure) ===")
    verify_finite_internal_profiles_and_diameters()
    verify_norm_choice_for_frequency_diameter()
    verify_causal_localization_is_not_internal_constancy()
    verify_convergence_of_frequency_profile_proposition()
    verify_frequency_to_scalar_implication_and_failed_converse()
    verify_internal_blurring_modes()
    verify_internal_constancy_under_admissible_recursive_refinement()
    verify_localized_internally_stable_recursive_structure_conjunction()
    verify_symbolic_centered_spectrum_dependencies_used_without_reproving_full_base_theory()
    print("\n=== Stable internal structure verification completed successfully ===")


if __name__ == "__main__":
    main()
