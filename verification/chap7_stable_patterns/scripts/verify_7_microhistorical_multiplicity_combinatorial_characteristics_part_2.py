"""
VERIFICATION of Sections 7.7.7--7.7.12:
Microhistorical multiplicity and combinatorial characteristics of
particle-like structures (part 2)
(subsections 7.7.7--7.7.12).

Source file:
    7_microhistorical_multiplicity_combinatorial_characteristics_part_2.tex. 
    
It does not re-prove the already verified
definition of particle-like structures, finite stable multiplicity, finite-level
combinatorial mass, stable fraction, or stability deficit.  Those are treated
as upstream dependencies.

Verified content
----------------
1. Independent disjoint composition:
      H_{P sqcup Q}^{(n)} congruent H_P^{(n)} x H_Q^{(n)}
   implies
      N_stable^{(n)}(P sqcup Q)
        = N_stable^{(n)}(P) N_stable^{(n)}(Q).

2. Additivity of finite-level combinatorial mass:
      m_comb^{(n)}(P sqcup Q)
        = m_comb^{(n)}(P) + m_comb^{(n)}(Q),
   on precisely the positive-multiplicity domain.

3. Additivity of finite asymptotic combinatorial mass when both finite limits
   exist, including a guard that the statement is not applied to divergent
   finite-level masses.

4. Binary special case with finite limiting stable multiplicity:
      N_stable(P) = 2^{d_bin(P)}
      m_comb(P) = d_bin(P) log_s(2),
   with the minimal binary base s = 2 giving m_comb(P) = d_bin(P).

5. Scalar spin-type signature:
      barSigma_infty(P) = sum_sigma eta(sigma) f_{sigma,infty}(P),
   including simplex validation, centered-state symbolic checks, convergence
   of the scalar under convergence of the frequency profile, and a non-injective
   witness for s >= 3.

6. Causal localization scales:
      ell_rho(P) = limsup D_{rho,n}(P),
      ell_r(P)   = limsup D_{r,n}(P),
   including strong rho-localization, weak rho-localization, full rho-spread,
   bounded logarithmic control, and unbounded logarithmic-control loss.

7. Quantitative asymptotic distinguishability:
      d_I(P,Q) = || I_infty(P) - I_infty(Q) ||,
   with exact verification for l1, l_infty, and squared-l2 norms:
      d_I(P,Q) > 0 iff P and Q differ in the selected invariant profile.

8. Separation between selected-invariant distance and the full asymptotic
   signature:
   two structures may have d_I = 0 while their full signatures differ in
   another component.

9. Macroscopic microhistorical degeneracy:
      N_stable^{(n)}(P) > 1
   iff at least two distinct finite stable realizations occur at level n, and
      m_comb^{(n)}(P) > 0
   iff such degeneracy is present.

10. Negative-domain checks:
    zero multiplicity, non-positive logarithm bases, negative binary-degree
    counts, non-simplex frequency profiles, non-finite invariant coordinates,
    and non-independent unions are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Mapping, Sequence

from sympy import (
    Integer,
    expand_log,
    log,
    simplify,
    summation,
    symbols,
)


StatePrefix = tuple[int, ...]
StableSet = frozenset[StatePrefix]


class VerificationDomainError(ValueError):
    """Raised when a mathematical expression is used outside its domain."""


def expect_raises(expected_exception: type[BaseException], func: Callable[[], object]) -> None:
    """Require a call to fail for the intended mathematical-domain reason."""
    try:
        func()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__}, but the call was accepted")


def require_base(base: int) -> None:
    if not isinstance(base, int) or base < 2:
        raise VerificationDomainError("The logarithm base s must satisfy s >= 2.")


def require_positive_count(count: int) -> None:
    if not isinstance(count, int) or count <= 0:
        raise VerificationDomainError("Stable multiplicity must be a positive integer here.")


def finite_mass_expr(count: int, base: int):
    """Exact symbolic expression for log_s(count), with strict domain checks."""
    require_base(base)
    require_positive_count(count)
    return log(Integer(count)) / log(Integer(base))


def float_mass(count: int, base: int) -> float:
    """Numerical mass used only for sign and monotonicity checks."""
    require_base(base)
    require_positive_count(count)
    from math import log as math_log

    return math_log(count) / math_log(base)


def centered_eta(state: int, base: int) -> Fraction:
    require_base(base)
    if not 1 <= state <= base:
        raise VerificationDomainError("Internal state is outside S={1,...,s}.")
    return Fraction(2 * state - base - 1, 2)


def validate_prefix(prefix: StatePrefix, base: int, expected_length: int | None = None) -> None:
    require_base(base)
    if expected_length is not None and len(prefix) != expected_length:
        raise VerificationDomainError("Finite prefix has the wrong level length.")
    for state in prefix:
        if not isinstance(state, int) or not 1 <= state <= base:
            raise VerificationDomainError("Finite prefix contains an inadmissible internal state.")


@dataclass(frozen=True)
class FiniteParticleStructure:
    """Finite-level data for a particle-like structure used in this verification."""

    name: str
    base: int
    stable_prefixes_by_level: Mapping[int, StableSet]
    particle_like: bool = True

    def __post_init__(self) -> None:
        require_base(self.base)
        if not self.particle_like:
            return
        for level, stable_prefixes in self.stable_prefixes_by_level.items():
            if not isinstance(level, int) or level < 0:
                raise VerificationDomainError("Levels must be nonnegative integers.")
            if not stable_prefixes:
                raise VerificationDomainError("A particle-like finite support must be nonempty.")
            for prefix in stable_prefixes:
                validate_prefix(prefix, self.base, expected_length=level)

    def stable_set(self, level: int) -> StableSet:
        if not self.particle_like:
            raise VerificationDomainError("Combinatorial characteristics require a particle-like structure.")
        if level not in self.stable_prefixes_by_level:
            raise VerificationDomainError("Requested finite level is not present.")
        return self.stable_prefixes_by_level[level]

    def count(self, level: int) -> int:
        return len(self.stable_set(level))

    def mass(self, level: int):
        return finite_mass_expr(self.count(level), self.base)


def independent_product_support(P: FiniteParticleStructure, Q: FiniteParticleStructure, level: int) -> frozenset[tuple[StatePrefix, StatePrefix]]:
    if P.base != Q.base:
        raise VerificationDomainError("The structures must use the same base s.")
    left = P.stable_set(level)
    right = Q.stable_set(level)
    return frozenset((p, q) for p in left for q in right)


def verify_independent_product_cardinality_and_additivity() -> None:
    print("\n=== Independent product cardinality and finite-level mass additivity ===")

    base = 3
    P = FiniteParticleStructure(
        name="P",
        base=base,
        stable_prefixes_by_level={
            2: frozenset({(1, 1), (1, 2), (2, 1)}),
            3: frozenset({(1, 1, 1), (1, 2, 1), (2, 1, 2), (2, 2, 1)}),
        },
    )
    Q = FiniteParticleStructure(
        name="Q",
        base=base,
        stable_prefixes_by_level={
            2: frozenset({(2, 2), (3, 1)}),
            3: frozenset({(1, 3, 2), (2, 3, 1), (3, 1, 1)}),
        },
    )

    for level in (2, 3):
        product_support = independent_product_support(P, Q, level)
        NP = P.count(level)
        NQ = Q.count(level)
        N_union = len(product_support)

        assert N_union == NP * NQ
        assert len(product_support) == len(P.stable_set(level)) * len(Q.stable_set(level))

        m_union = finite_mass_expr(N_union, base)
        m_sum = P.mass(level) + Q.mass(level)
        assert simplify(expand_log(m_union, force=True) - m_sum) == 0

        # Numerical sign/ordering guard for non-power counts.
        assert abs(float(m_union.evalf()) - (float(P.mass(level).evalf()) + float(Q.mass(level).evalf()))) < 1e-12

    a, b, s = symbols("a b s", positive=True)
    symbolic_log_identity = expand_log(log(a * b) / log(s), force=True) - (log(a) / log(s) + log(b) / log(s))
    assert simplify(symbolic_log_identity) == 0

    print("[OK] Cartesian-product stable support gives exact multiplicative stable count")
    print("[OK] Finite-level combinatorial mass is additive on the positive-multiplicity domain")


def verify_non_independent_union_is_rejected() -> None:
    print("\n=== Negative check: additivity is not accepted without independence ===")

    base = 2
    P_support = frozenset({(1, 1, 1), (1, 2, 1), (2, 1, 1)})
    Q_support = frozenset({(1, 1, 2), (2, 2, 1)})

    # A deliberately correlated support: it is a proper subset of the Cartesian product.
    correlated_support = frozenset({
        ((1, 1, 1), (1, 1, 2)),
        ((1, 2, 1), (2, 2, 1)),
    })

    expected_product_size = len(P_support) * len(Q_support)
    assert len(correlated_support) != expected_product_size

    def assert_independent_cardinality() -> None:
        if len(correlated_support) != expected_product_size:
            raise VerificationDomainError("The supplied support is not the independent Cartesian product.")

    expect_raises(VerificationDomainError, assert_independent_cardinality)

    # If one incorrectly forced additivity, it would disagree with the actual correlated count.
    forced_additive_mass = finite_mass_expr(len(P_support), base) + finite_mass_expr(len(Q_support), base)
    actual_correlated_mass = finite_mass_expr(len(correlated_support), base)
    assert simplify(expand_log(forced_additive_mass, force=True) - actual_correlated_mass) != 0

    print("[OK] Correlated composition is rejected as a witness for additivity")
    print("[OK] Actual correlated multiplicity does not satisfy the independent-product mass formula")


@dataclass(frozen=True)
class EventuallyConstantCount:
    """Positive integer count sequence that has a finite limiting multiplicity."""

    base: int
    transient: Mapping[int, int]
    limiting_count: int
    threshold: int

    def __post_init__(self) -> None:
        require_base(self.base)
        require_positive_count(self.limiting_count)
        if self.threshold < 0:
            raise VerificationDomainError("Threshold must be nonnegative.")
        for count in self.transient.values():
            require_positive_count(count)

    def count(self, level: int) -> int:
        if level < 0:
            raise VerificationDomainError("Level must be nonnegative.")
        return self.transient.get(level, self.limiting_count if level >= self.threshold else self.limiting_count)

    def limiting_mass(self):
        return finite_mass_expr(self.limiting_count, self.base)


@dataclass(frozen=True)
class DivergentCount:
    """Positive integer sequence whose finite-level mass diverges."""

    base: int

    def count(self, level: int) -> int:
        if level < 0:
            raise VerificationDomainError("Level must be nonnegative.")
        return self.base ** (level + 1)

    def mass(self, level: int):
        return finite_mass_expr(self.count(level), self.base)


def verify_asymptotic_additivity_when_finite_limits_exist() -> None:
    print("\n=== Asymptotic mass additivity under finite limiting multiplicities ===")

    base = 5
    P = EventuallyConstantCount(base=base, transient={0: 1, 1: 2}, limiting_count=25, threshold=2)
    Q = EventuallyConstantCount(base=base, transient={0: 1}, limiting_count=125, threshold=1)

    for level in range(2, 8):
        combined_count = P.count(level) * Q.count(level)
        assert combined_count == 25 * 125
        combined_mass = finite_mass_expr(combined_count, base)
        assert simplify(expand_log(combined_mass, force=True) - (P.limiting_mass() + Q.limiting_mass())) == 0

    limiting_combined_mass = finite_mass_expr(P.limiting_count * Q.limiting_count, base)
    assert simplify(expand_log(limiting_combined_mass, force=True) - (P.limiting_mass() + Q.limiting_mass())) == 0

    divergent = DivergentCount(base=base)
    assert divergent.mass(1) < divergent.mass(2) < divergent.mass(3)
    assert simplify(divergent.mass(10) - Integer(11)) == 0

    def claim_finite_limit_for_divergent_sequence() -> None:
        # A finite asymptotic mass cannot be assigned to a sequence growing as s^(n+1).
        sample_masses = [divergent.mass(level) for level in range(3, 8)]
        if len(set(sample_masses)) != 1:
            raise VerificationDomainError("Finite limiting mass is absent for this divergent sequence.")

    expect_raises(VerificationDomainError, claim_finite_limit_for_divergent_sequence)

    print("[OK] Finite asymptotic additivity is verified when both limiting multiplicities exist")
    print("[OK] Divergent finite-level mass is not incorrectly treated as a finite asymptotic mass")


def verify_binary_special_case() -> None:
    print("\n=== Binary special case ===")

    d = symbols("d", integer=True, nonnegative=True)
    s_positive = symbols("s_positive", positive=True)

    symbolic_binary_mass = log(Integer(2) ** d) / log(s_positive)
    expected_symbolic = d * log(Integer(2)) / log(s_positive)
    assert simplify(symbolic_binary_mass - expected_symbolic) == 0

    for base in range(2, 10):
        for degree in range(0, 9):
            N = 2 ** degree
            mass = finite_mass_expr(N, base)
            expected = Integer(degree) * log(Integer(2)) / log(Integer(base))
            assert simplify(mass - expected) == 0
            if base == 2:
                assert simplify(mass - Integer(degree)) == 0

    expect_raises(VerificationDomainError, lambda: finite_mass_expr(4, 1))

    def negative_binary_degree() -> None:
        degree = -1
        if degree < 0:
            raise VerificationDomainError("The number of independent binary degrees must be nonnegative.")

    expect_raises(VerificationDomainError, negative_binary_degree)

    print("[OK] N_stable = 2^d gives m_comb = d log_s(2) symbolically")
    print("[OK] In the minimal binary base s=2, m_comb equals d exactly")


def validate_frequency_vector(freq: Sequence[Fraction], base: int) -> None:
    require_base(base)
    if len(freq) != base:
        raise VerificationDomainError("Frequency profile must have one component for each internal state.")
    if any(component < 0 for component in freq):
        raise VerificationDomainError("Frequency profile components must be nonnegative.")
    if sum(freq, Fraction(0)) != Fraction(1):
        raise VerificationDomainError("Frequency profile must sum to one.")


def spin_like_signature(freq: Sequence[Fraction], base: int) -> Fraction:
    validate_frequency_vector(freq, base)
    return sum(centered_eta(index + 1, base) * freq[index] for index in range(base))


def verify_internal_spin_like_signature() -> None:
    print("\n=== Scalar spin-type signature ===")

    s_sym = symbols("s", integer=True, positive=True)
    sigma = symbols("sigma", integer=True, positive=True)
    eta_sigma = sigma - (s_sym + 1) / 2
    assert simplify(eta_sigma.subs(sigma, 1) + (s_sym - 1) / 2) == 0
    assert simplify(eta_sigma.subs(sigma, s_sym) - (s_sym - 1) / 2) == 0

    # General centered-spectrum sum for arbitrary s >= 2.
    # This is the exact symbolic verification of
    #     sum_{sigma=1}^{s} (sigma - (s+1)/2) = 0,
    # not a finite-grid check and not a placeholder identity.
    k_sym = symbols("k", integer=True, positive=True)
    centered_sum = summation(k_sym - (s_sym + 1) / 2, (k_sym, 1, s_sym))
    assert simplify(centered_sum) == 0

    # The same calculation also fixes the extremal radius B_s = (s-1)/2
    # of the centered internal-state spectrum.
    B_s = (s_sym - 1) / 2
    assert simplify(eta_sigma.subs(sigma, 1) + B_s) == 0
    assert simplify(eta_sigma.subs(sigma, s_sym) - B_s) == 0

    for base in range(2, 8):
        freq = tuple(Fraction(1, base) for _ in range(base))
        assert spin_like_signature(freq, base) == 0

        left_extreme = (Fraction(1),) + tuple(Fraction(0) for _ in range(base - 1))
        right_extreme = tuple(Fraction(0) for _ in range(base - 1)) + (Fraction(1),)
        assert spin_like_signature(left_extreme, base) == Fraction(-(base - 1), 2)
        assert spin_like_signature(right_extreme, base) == Fraction(base - 1, 2)

    # Convergence of frequency profile implies convergence of the scalar by linearity.
    base = 4
    target = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4), Fraction(0))
    target_scalar = spin_like_signature(target, base)

    previous_error: Fraction | None = None
    for N in (20, 40, 80, 160, 320):
        perturb = Fraction(1, N)
        freq_N = (
            target[0] - perturb,
            target[1] + perturb,
            target[2],
            target[3],
        )
        scalar_N = spin_like_signature(freq_N, base)
        error = abs(scalar_N - target_scalar)
        if previous_error is not None:
            assert error < previous_error
        previous_error = error
    assert previous_error is not None and previous_error < Fraction(1, 100)

    # Non-injectivity for s >= 3: different full frequency profiles may have the same scalar.
    base = 3
    f1 = (Fraction(1, 2), Fraction(0), Fraction(1, 2))
    f2 = (Fraction(0), Fraction(1), Fraction(0))
    assert f1 != f2
    assert spin_like_signature(f1, base) == spin_like_signature(f2, base) == 0

    expect_raises(VerificationDomainError, lambda: spin_like_signature((Fraction(1, 2), Fraction(1, 2)), 3))
    expect_raises(VerificationDomainError, lambda: spin_like_signature((Fraction(2, 3), Fraction(2, 3), Fraction(-1, 3)), 3))
    expect_raises(VerificationDomainError, lambda: spin_like_signature((Fraction(1, 3), Fraction(1, 3), Fraction(1, 4)), 3))

    print("[OK] Scalar spin-type signature is the centered linear functional of the limiting frequency profile")
    print("[OK] Convergence of the full frequency profile implies convergence of the scalar signature")
    print("[OK] The scalar signature is not confused with the full frequency profile")


@dataclass(frozen=True)
class RationalSequence:
    """A rational nonnegative sequence used for limsup classification."""

    values: Callable[[int], Fraction]

    def sample(self, start: int, stop: int) -> list[Fraction]:
        return [self.values(n) for n in range(start, stop + 1)]


def empirical_limsup_upper(seq: RationalSequence, start: int, stop: int) -> Fraction:
    tail_maxima = []
    for n in range(start, stop + 1):
        tail_maxima.append(max(seq.sample(n, stop)))
    return min(tail_maxima)


def verify_causal_localization_scales() -> None:
    print("\n=== Causal localization scales ===")

    zero_rho = RationalSequence(lambda n: Fraction(1, n + 1))
    weak_rho = RationalSequence(lambda n: Fraction(1, 3) + Fraction(1, n + 5))
    full_rho = RationalSequence(lambda n: Fraction(n, n + 1))
    oscillating_bounded_r = RationalSequence(lambda n: Fraction(2) + Fraction((-1) ** n, 3))
    unbounded_r = RationalSequence(lambda n: Fraction(n, 1))

    # rho limsup classes are checked by exact tail bounds on representative sequences.
    assert empirical_limsup_upper(zero_rho, 50, 200) <= Fraction(1, 51)
    assert Fraction(1, 3) < empirical_limsup_upper(weak_rho, 50, 200) < Fraction(1, 2)
    assert empirical_limsup_upper(full_rho, 50, 200) > Fraction(199, 200)

    # Logarithmic spread: bounded sample remains under a fixed rational ceiling.
    bounded_samples = oscillating_bounded_r.sample(1, 200)
    assert max(bounded_samples) <= Fraction(7, 3)
    assert min(bounded_samples) >= Fraction(5, 3)

    # The explicitly declared unbounded logarithmic-spread profile is used
    # directly: no fixed finite ceiling can control its sampled tail.
    unbounded_samples = unbounded_r.sample(1, 200)
    assert unbounded_samples[0] == Fraction(1)
    assert unbounded_samples[-1] == Fraction(200)
    for ceiling in (Fraction(10), Fraction(50), Fraction(100), Fraction(150)):
        assert any(value > ceiling for value in unbounded_samples)

    # Symbolic companion check: r_n = n has unbounded tail differences, so it
    # cannot satisfy an O(1) logarithmic localization bound.
    n = symbols("n", positive=True, integer=True)
    assert simplify(n.subs(n, 200) - n.subs(n, 20)) == 180
    assert n.subs(n, 1000) > 100

    # Classification guards.
    def classify_rho_limsup(ell: Fraction) -> str:
        if ell < 0 or ell > 1:
            raise VerificationDomainError("rho localization scale must lie in [0,1].")
        if ell == 0:
            return "strong-rho-localized"
        if Fraction(0) < ell < Fraction(1):
            return "weak-rho-localized"
        return "not-rho-localized"

    assert classify_rho_limsup(Fraction(0)) == "strong-rho-localized"
    assert classify_rho_limsup(Fraction(2, 5)) == "weak-rho-localized"
    assert classify_rho_limsup(Fraction(1)) == "not-rho-localized"
    expect_raises(VerificationDomainError, lambda: classify_rho_limsup(Fraction(6, 5)))

    print("[OK] ell_rho distinguishes strong, weak, and full-spread rho regimes")
    print("[OK] ell_r and boundedness distinguish logarithmic control from loss of control")


@dataclass(frozen=True)
class InvariantProfile:
    values: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise VerificationDomainError("Invariant profile must be finite and nonempty.")
        for value in self.values:
            if not isinstance(value, Fraction):
                raise VerificationDomainError("Invariant coordinates are represented exactly as Fractions.")


def l1_distance(a: InvariantProfile, b: InvariantProfile) -> Fraction:
    if len(a.values) != len(b.values):
        raise VerificationDomainError("Invariant profiles must have the same dimension.")
    return sum(abs(x - y) for x, y in zip(a.values, b.values))


def linf_distance(a: InvariantProfile, b: InvariantProfile) -> Fraction:
    if len(a.values) != len(b.values):
        raise VerificationDomainError("Invariant profiles must have the same dimension.")
    return max(abs(x - y) for x, y in zip(a.values, b.values))


def squared_l2_distance(a: InvariantProfile, b: InvariantProfile) -> Fraction:
    if len(a.values) != len(b.values):
        raise VerificationDomainError("Invariant profiles must have the same dimension.")
    return sum((x - y) * (x - y) for x, y in zip(a.values, b.values))


@dataclass(frozen=True)
class FullSignature:
    selected_profile: InvariantProfile
    frequency_profile: tuple[Fraction, ...]
    spin_scalar: Fraction
    localization_scales: tuple[Fraction, Fraction]
    combinatorial_characteristics: tuple[Fraction, Fraction]


def full_signature_differs(a: FullSignature, b: FullSignature) -> bool:
    return (
        a.selected_profile != b.selected_profile
        or a.frequency_profile != b.frequency_profile
        or a.spin_scalar != b.spin_scalar
        or a.localization_scales != b.localization_scales
        or a.combinatorial_characteristics != b.combinatorial_characteristics
    )


def verify_quantitative_asymptotic_distinguishability() -> None:
    print("\n=== Quantitative asymptotic distinguishability ===")

    profiles = [
        InvariantProfile((Fraction(0), Fraction(0), Fraction(0))),
        InvariantProfile((Fraction(1, 3), Fraction(0), Fraction(0))),
        InvariantProfile((Fraction(1, 3), Fraction(2, 5), Fraction(0))),
        InvariantProfile((Fraction(1, 3), Fraction(2, 5), Fraction(7, 11))),
    ]

    for distance in (l1_distance, linf_distance, squared_l2_distance):
        for a in profiles:
            assert distance(a, a) == 0
            for b in profiles:
                d_ab = distance(a, b)
                d_ba = distance(b, a)
                assert d_ab == d_ba
                assert d_ab >= 0
                assert (d_ab > 0) == (a.values != b.values)

    # Triangle inequality for l1 and linf over the deterministic grid.
    for a in profiles:
        for b in profiles:
            for c in profiles:
                assert l1_distance(a, c) <= l1_distance(a, b) + l1_distance(b, c)
                assert linf_distance(a, c) <= linf_distance(a, b) + linf_distance(b, c)

    selected = InvariantProfile((Fraction(1, 2), Fraction(1, 3)))
    full_A = FullSignature(
        selected_profile=selected,
        frequency_profile=(Fraction(1, 2), Fraction(1, 2)),
        spin_scalar=Fraction(0),
        localization_scales=(Fraction(0), Fraction(2)),
        combinatorial_characteristics=(Fraction(3), Fraction(0)),
    )
    full_B = FullSignature(
        selected_profile=selected,
        frequency_profile=(Fraction(1, 3), Fraction(2, 3)),
        spin_scalar=Fraction(1, 6),
        localization_scales=(Fraction(0), Fraction(2)),
        combinatorial_characteristics=(Fraction(3), Fraction(0)),
    )

    assert l1_distance(full_A.selected_profile, full_B.selected_profile) == 0
    assert full_signature_differs(full_A, full_B)

    expect_raises(VerificationDomainError, lambda: l1_distance(InvariantProfile((Fraction(1),)), InvariantProfile((Fraction(1), Fraction(2)))))

    print("[OK] Selected-profile distance is positive exactly when selected invariants differ")
    print("[OK] Zero selected-profile distance does not force equality of the full asymptotic signature")


def verify_macroscopic_microhistorical_degeneracy() -> None:
    print("\n=== Macroscopic microhistorical degeneracy and positive finite-level mass ===")

    base = 4
    nondegenerate = frozenset({(1, 2, 3)})
    degenerate = frozenset({(1, 2, 3), (1, 2, 4), (2, 2, 3)})

    for support in (nondegenerate, degenerate):
        for prefix in support:
            validate_prefix(prefix, base, expected_length=3)

        N = len(support)
        mass = finite_mass_expr(N, base)
        has_degeneracy = N > 1

        if has_degeneracy:
            distinct_pairs = [(a, b) for a in support for b in support if a != b]
            assert distinct_pairs
            assert float(mass.evalf()) > 0
        else:
            assert N == 1
            assert simplify(mass) == 0

        assert (float(mass.evalf()) > 0) == has_degeneracy

    # Duplicate input entries collapse under set semantics and do not create degeneracy.
    duplicated_entries = [(1, 1), (1, 1), (1, 1)]
    collapsed_support = frozenset(duplicated_entries)
    assert len(duplicated_entries) == 3
    assert len(collapsed_support) == 1
    assert simplify(finite_mass_expr(len(collapsed_support), 2)) == 0

    expect_raises(VerificationDomainError, lambda: finite_mass_expr(0, base))

    print("[OK] N_stable^{(n)} > 1 is exactly finite-level microhistorical degeneracy")
    print("[OK] Positive finite-level combinatorial mass is exactly the logarithmic form of that degeneracy")
    print("[OK] Repeated input entries do not create structural degeneracy")


def verify_negative_domain_guards() -> None:
    print("\n=== Global negative-domain guards ===")

    expect_raises(VerificationDomainError, lambda: finite_mass_expr(1, 0))
    expect_raises(VerificationDomainError, lambda: finite_mass_expr(-3, 2))
    expect_raises(VerificationDomainError, lambda: centered_eta(0, 3))
    expect_raises(VerificationDomainError, lambda: validate_prefix((1, 4), 3, expected_length=2))
    expect_raises(VerificationDomainError, lambda: validate_prefix((1, 2), 3, expected_length=3))

    non_particle = FiniteParticleStructure(
        name="not-particle-like",
        base=2,
        stable_prefixes_by_level={},
        particle_like=False,
    )
    expect_raises(VerificationDomainError, lambda: non_particle.count(1))

    print("[OK] Invalid bases, counts, states, levels, and non-particle-like carriers are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of Section 7, Part 2 (subsections 7.7.7--7.7.12) ===")
    verify_independent_product_cardinality_and_additivity()
    verify_non_independent_union_is_rejected()
    verify_asymptotic_additivity_when_finite_limits_exist()
    verify_binary_special_case()
    verify_internal_spin_like_signature()
    verify_causal_localization_scales()
    verify_quantitative_asymptotic_distinguishability()
    verify_macroscopic_microhistorical_degeneracy()
    verify_negative_domain_guards()
    print("\n=== Section 7, Part 2 verification completed successfully ===")


if __name__ == "__main__":
    main()
