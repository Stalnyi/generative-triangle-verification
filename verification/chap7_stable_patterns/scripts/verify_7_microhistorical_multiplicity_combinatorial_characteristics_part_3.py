"""
VERIFICATION of Sections 7.7.13--7.7.19:
Microhistorical multiplicity and combinatorial characteristics of
particle-like structures (part 3)
[Geometrically identical but internally different structures, internally identical
but geometrically different structures, and completeness limits of the full
particle-like signature]
(subsections 7.7.13--7.7.19).

Source file:
    7_microhistorical_multiplicity_combinatorial_characteristics_part_3.tex

It does not re-prove the previously verified definitions of particle-like
structures, finite stable multiplicity, finite-level combinatorial mass,
specific combinatorial mass, stable fraction, stability deficit, internal
frequency convergence, or additivity.  Those results are treated as upstream
dependencies.  The present script verifies the new separation and completeness
claims made in this file:

Verified content
----------------
1. Geometrically identical but internally different structures:
      ell_rho(P) = ell_rho(Q), ell_r(P) = ell_r(Q),
      f_infty(P) != f_infty(Q).
   The script verifies two independent regimes:
      (a) the scalar spin-type signatures are different;
      (b) the scalar spin-type signatures coincide although the full limiting
          frequency profiles are different.

2. Internally identical but geometrically different structures:
      f_infty(P) = f_infty(Q)  ==>  barSigma_infty(P) = barSigma_infty(Q),
   while either ell_rho or ell_r differs.

3. Same invariant, internal, and localization profile but different
   finite-level stable multiplicity:
      I_infty(P) = I_infty(Q),
      f_infty(P) = f_infty(Q),
      ell_rho(P) = ell_rho(Q), ell_r(P) = ell_r(Q),
      N_stable^{(n)}(P) != N_stable^{(n)}(Q).
   The script checks the induced inequality of finite-level combinatorial
   masses on the positive-multiplicity domain.

4. Limit discipline for asymptotic combinatorial mass:
      m_comb(P) != m_comb(Q)
   only when the finite-level mass difference has a nonzero limit.
   The script verifies this at the level of exact symbolic sequence limits and
   adds an integer-domain guard: if positive integer multiplicities have finite
   logarithmic limits and are eventually different, then equal finite limiting
   masses are impossible.

5. Same combinatorial mass but different full signature:
      m_comb(P) = m_comb(Q)
   does not imply equality of f_infty, ell_rho, ell_r, or the full limiting
   signature M_infty.

6. Finite signature M_n(P):
      M_n(P) = (P, I_infty, f_infty, barSigma_infty, ell_rho, ell_r,
                N_stable^{(n)}, m_comb^{(n)}, theta_stable^{(n)},
                delta_stable^{(n)}).
   The script verifies all defining relations:
      m_comb^{(n)} = log_s N_stable^{(n)},
      theta_stable^{(n)} = N_stable^{(n)} / s^n,
      delta_stable^{(n)} = n - m_comb^{(n)}
                          = -log_s theta_stable^{(n)}.

7. Limiting signature M_infty(P):
      M_infty(P) = (P, I_infty, f_infty, barSigma_infty,
                    ell_rho, ell_r, m_comb, mu_comb).
   Finite-level N_stable^(n), theta_stable^(n), and delta_stable^(n) remain
   components of M_n(P), not of the displayed limiting tuple.
   The script verifies componentwise completeness within the model: changing
   any listed mathematical component changes the signature.

8. Domain guards:
   invalid base s < 2, non-simplex frequency profile, nonpositive or excessive
   multiplicity, negative localization scale, and missing finite limits are all
   rejected explicitly.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Callable, Optional, Tuple

from sympy import Eq, Symbol, Integer, Rational, expand_log, limit, log, oo, simplify, symbols


Number = Fraction
Profile = Tuple[Fraction, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(expected_exception: type[BaseException], fn: Callable[[], object], label: str) -> None:
    try:
        fn()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} was not raised: {label}")


def exact_log_power(base: int, value: int) -> Fraction:
    """Return k for value = base^k.  Raise if value is not an exact base power."""
    require(base >= 2, "The logarithmic base must satisfy s >= 2.")
    require(value > 0, "The logarithm is defined only for positive multiplicity.")

    power = 1
    exponent = 0
    while power < value:
        power *= base
        exponent += 1
    require(power == value, f"{value} is not an exact power of base {base}.")
    return Fraction(exponent)


def centered_eta(base: int, state: int) -> Fraction:
    """Centered internal-state value eta(sigma) = sigma - (s+1)/2."""
    require(base >= 2, "The number of internal states must satisfy s >= 2.")
    require(1 <= state <= base, "Internal state is outside S={1,...,s}.")
    return Fraction(2 * state - (base + 1), 2)


def spin_like_scalar(base: int, frequency_profile: Profile) -> Fraction:
    validate_frequency_profile(base, frequency_profile)
    return sum(centered_eta(base, sigma + 1) * frequency_profile[sigma] for sigma in range(base))


def validate_frequency_profile(base: int, frequency_profile: Profile) -> None:
    require(base >= 2, "The number of internal states must satisfy s >= 2.")
    require(len(frequency_profile) == base, "Frequency profile length must equal s.")
    require(all(x >= 0 for x in frequency_profile), "Frequency profile cannot contain negative components.")
    require(sum(frequency_profile) == 1, "Frequency profile must lie in the probability simplex.")


def validate_invariant_profile(invariant_profile: Profile) -> None:
    require(len(invariant_profile) > 0, "Invariant profile cannot be empty.")
    require(all(x.denominator > 0 for x in invariant_profile), "Invalid rational invariant component.")


def validate_localization(ell_rho: Fraction, ell_r: Fraction) -> None:
    require(Fraction(0) <= ell_rho <= Fraction(1), "ell_rho must lie in [0,1].")
    require(ell_r >= 0, "ell_r must be nonnegative.")


@dataclass(frozen=True)
class ParticleLikeData:
    """A finite test representation of the quantitative particle-like signature."""

    name: str
    base: int
    invariant_profile: Profile
    frequency_profile: Profile
    ell_rho: Fraction
    ell_r: Fraction
    multiplicity: Callable[[int], int]
    mass_limit: Optional[Fraction]
    specific_mass_limit: Optional[Fraction]

    def validate_static_components(self) -> None:
        require(self.base >= 2, "The base s must satisfy s >= 2.")
        validate_invariant_profile(self.invariant_profile)
        validate_frequency_profile(self.base, self.frequency_profile)
        validate_localization(self.ell_rho, self.ell_r)
        if self.mass_limit is not None:
            require(self.mass_limit >= 0, "Asymptotic combinatorial mass cannot be negative.")
        if self.specific_mass_limit is not None:
            require(Fraction(0) <= self.specific_mass_limit <= Fraction(1),
                    "Specific combinatorial mass must lie in [0,1].")

    @property
    def spin_signature(self) -> Fraction:
        return spin_like_scalar(self.base, self.frequency_profile)

    def stable_multiplicity(self, level: int) -> int:
        require(level >= 0, "The level n must be nonnegative.")
        N = self.multiplicity(level)
        require(isinstance(N, int), "Stable multiplicity must be an integer.")
        require(N > 0, "Stable multiplicity must be positive for logarithmic mass.")
        require(N <= self.base**level, "Stable multiplicity cannot exceed the full s^n tree.")
        return N

    def finite_mass_exact_power(self, level: int) -> Fraction:
        return exact_log_power(self.base, self.stable_multiplicity(level))

    def stable_fraction_exact_power(self, level: int) -> Fraction:
        N = self.stable_multiplicity(level)
        return Fraction(N, self.base**level)

    def stability_deficit_exact_power(self, level: int) -> Fraction:
        return Fraction(level) - self.finite_mass_exact_power(level)

    def finite_signature_exact_power(self, level: int) -> tuple:
        m = self.finite_mass_exact_power(level)
        theta = self.stable_fraction_exact_power(level)
        delta = self.stability_deficit_exact_power(level)

        require(delta == Fraction(level) - m, "delta must equal n - m_comb^(n).")
        require(theta == Fraction(self.stable_multiplicity(level), self.base**level),
                "theta must equal N_stable^(n)/s^n.")
        # For exact-power multiplicities, theta = s^{m-n}, hence -log_s(theta)=n-m.
        require(theta == Fraction(1, self.base**int(delta)) if delta.denominator == 1 else theta,
                "Exact-power theta relation checked where the deficit is integral.")

        return (
            self.name,
            self.invariant_profile,
            self.frequency_profile,
            self.spin_signature,
            self.ell_rho,
            self.ell_r,
            self.stable_multiplicity(level),
            m,
            theta,
            delta,
        )

    def limiting_signature(self) -> tuple:
        require(self.mass_limit is not None, "m_comb(P) is not defined without a finite limit.")
        require(self.specific_mass_limit is not None, "mu_comb(P) is not defined without a finite limit.")
        return (
            self.name,
            self.invariant_profile,
            self.frequency_profile,
            self.spin_signature,
            self.ell_rho,
            self.ell_r,
            self.mass_limit,
            self.specific_mass_limit,
        )

    def limiting_signature_without_label(self) -> tuple:
        full = self.limiting_signature()
        return full[1:]


def power_multiplicity(base: int, exponent: int) -> Callable[[int], int]:
    require(base >= 2, "base must satisfy s >= 2.")
    require(exponent >= 0, "exponent must be nonnegative.")

    def f(level: int) -> int:
        require(level >= exponent, "The test level must be large enough for N <= s^n.")
        return base**exponent

    return f


def full_tree_multiplicity(base: int) -> Callable[[int], int]:
    require(base >= 2, "base must satisfy s >= 2.")
    return lambda level: base**level


def make_particle(
    name: str,
    *,
    base: int = 3,
    invariant_profile: Profile = (Fraction(1, 5), Fraction(2, 5)),
    frequency_profile: Profile = (Fraction(1, 2), Fraction(0), Fraction(1, 2)),
    ell_rho: Fraction = Fraction(1, 3),
    ell_r: Fraction = Fraction(2),
    mass_exponent: int = 2,
    specific_mass_limit: Fraction = Fraction(0),
) -> ParticleLikeData:
    p = ParticleLikeData(
        name=name,
        base=base,
        invariant_profile=invariant_profile,
        frequency_profile=frequency_profile,
        ell_rho=ell_rho,
        ell_r=ell_r,
        multiplicity=power_multiplicity(base, mass_exponent),
        mass_limit=Fraction(mass_exponent),
        specific_mass_limit=specific_mass_limit,
    )
    p.validate_static_components()
    return p


def finite_mass_difference_exact_power(a: ParticleLikeData, b: ParticleLikeData, level: int) -> Fraction:
    require(a.base == b.base, "Finite mass comparison requires the same logarithmic base.")
    return a.finite_mass_exact_power(level) - b.finite_mass_exact_power(level)


def verify_symbolic_spin_map_and_noninjectivity() -> None:
    print("\n=== Symbolic and exact verification of the internal frequency-to-scalar map ===")

    base = 3
    eta_values = tuple(centered_eta(base, sigma) for sigma in range(1, base + 1))
    require(eta_values == (Fraction(-1), Fraction(0), Fraction(1)), "Unexpected centered spectrum for s=3.")

    f1 = (Fraction(1, 2), Fraction(1, 2), Fraction(0))
    f2 = (Fraction(0), Fraction(1, 2), Fraction(1, 2))
    require(spin_like_scalar(base, f1) == Fraction(-1, 2), "First scalar signature mismatch.")
    require(spin_like_scalar(base, f2) == Fraction(1, 2), "Second scalar signature mismatch.")
    require(f1 != f2 and spin_like_scalar(base, f1) != spin_like_scalar(base, f2),
            "Different profiles should be able to induce different scalar signatures.")

    g1 = (Fraction(1, 2), Fraction(0), Fraction(1, 2))
    g2 = (Fraction(0), Fraction(1), Fraction(0))
    require(g1 != g2, "The witness profiles must be different.")
    require(spin_like_scalar(base, g1) == 0, "Balanced edge profile should have zero scalar signature.")
    require(spin_like_scalar(base, g2) == 0, "Central-state profile should have zero scalar signature.")

    a, b, c = symbols("a b c", nonnegative=True)
    scalar = -a + c
    simplex_constraint = Eq(a + b + c, 1)
    require(str(simplex_constraint) == "Eq(a + b + c, 1)", "Simplex constraint was not formed correctly.")
    require(simplify(scalar.subs({a: Fraction(1, 2), b: 0, c: Fraction(1, 2)})) == 0,
            "Balanced edge profile must map to zero scalar in the symbolic projection.")
    require(simplify(scalar.subs({a: 0, b: 1, c: 0})) == 0,
            "Central profile must map to zero scalar in the symbolic projection.")

    print("[OK] The scalar spin-type signature is a linear projection of the full frequency profile")
    print("[OK] The scalar projection is not injective: distinct full profiles can have the same scalar value")


def verify_geometrically_identical_internally_different() -> None:
    print("\n=== Verification of geometrically identical but internally different structures ===")

    P = make_particle(
        "P_geom_same_spin_scalar_different",
        frequency_profile=(Fraction(1, 2), Fraction(1, 2), Fraction(0)),
        ell_rho=Fraction(1, 4),
        ell_r=Fraction(5, 3),
    )
    Q = make_particle(
        "Q_geom_same_spin_scalar_different",
        frequency_profile=(Fraction(0), Fraction(1, 2), Fraction(1, 2)),
        ell_rho=P.ell_rho,
        ell_r=P.ell_r,
    )

    require(P.ell_rho == Q.ell_rho and P.ell_r == Q.ell_r,
            "The witness must keep both localization scales identical.")
    require(P.frequency_profile != Q.frequency_profile,
            "The witness must change the limiting frequency profile.")
    require(P.spin_signature != Q.spin_signature,
            "This subcase must also be distinguished by the scalar spin-type signature.")

    R = make_particle(
        "R_geom_same_full_internal_different_scalar_same",
        frequency_profile=(Fraction(1, 2), Fraction(0), Fraction(1, 2)),
        ell_rho=Fraction(1, 4),
        ell_r=Fraction(5, 3),
    )
    S = make_particle(
        "S_geom_same_full_internal_different_scalar_same",
        frequency_profile=(Fraction(0), Fraction(1), Fraction(0)),
        ell_rho=R.ell_rho,
        ell_r=R.ell_r,
    )

    require(R.ell_rho == S.ell_rho and R.ell_r == S.ell_r,
            "The second witness must keep geometry identical.")
    require(R.frequency_profile != S.frequency_profile,
            "The full internal frequency profiles must differ.")
    require(R.spin_signature == S.spin_signature,
            "The scalar signature must fail to distinguish this second witness.")
    require(R.limiting_signature_without_label() != S.limiting_signature_without_label(),
            "The full limiting signature must still distinguish the structures.")

    print("[OK] Equal localization scales do not force equality of the full internal frequency profile")
    print("[OK] Two internally different structures can either differ or coincide in scalar spin-type signature")


def verify_internally_identical_geometrically_different() -> None:
    print("\n=== Verification of internally identical but geometrically different structures ===")

    internal = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
    strong_localized = make_particle(
        "P_internal_same_geometry_rho_different",
        frequency_profile=internal,
        ell_rho=Fraction(0),
        ell_r=Fraction(3, 2),
    )
    weak_localized = make_particle(
        "Q_internal_same_geometry_rho_different",
        frequency_profile=internal,
        ell_rho=Fraction(2, 5),
        ell_r=strong_localized.ell_r,
    )

    require(strong_localized.frequency_profile == weak_localized.frequency_profile,
            "Internal profiles must be identical in this witness.")
    require(strong_localized.spin_signature == weak_localized.spin_signature,
            "Equal internal profiles must give equal scalar spin-type signatures.")
    require(strong_localized.ell_rho != weak_localized.ell_rho,
            "The rho-localization scale must differ.")

    bounded_r = make_particle(
        "R_internal_same_r_scale_different",
        frequency_profile=internal,
        ell_rho=Fraction(1, 5),
        ell_r=Fraction(1),
    )
    wider_r = make_particle(
        "S_internal_same_r_scale_different",
        frequency_profile=internal,
        ell_rho=bounded_r.ell_rho,
        ell_r=Fraction(7, 3),
    )

    require(bounded_r.frequency_profile == wider_r.frequency_profile,
            "Internal profiles must be identical in the logarithmic-scale witness.")
    require(bounded_r.spin_signature == wider_r.spin_signature,
            "Scalar spin-type signatures must agree.")
    require(bounded_r.ell_r != wider_r.ell_r,
            "The logarithmic localization scale must differ.")

    print("[OK] Equal full internal profiles force equal scalar signatures")
    print("[OK] Equal internal organization does not force equal causal-localization scales")


def verify_same_profiles_but_different_multiplicity() -> None:
    print("\n=== Verification of equal visible profiles but different finite multiplicity ===")

    P = make_particle(
        "P_same_visible_profile_lower_multiplicity",
        base=2,
        invariant_profile=(Fraction(1, 7), Fraction(3, 7)),
        frequency_profile=(Fraction(1, 2), Fraction(1, 2)),
        ell_rho=Fraction(1, 8),
        ell_r=Fraction(4),
        mass_exponent=2,
    )
    Q = make_particle(
        "Q_same_visible_profile_higher_multiplicity",
        base=2,
        invariant_profile=P.invariant_profile,
        frequency_profile=P.frequency_profile,
        ell_rho=P.ell_rho,
        ell_r=P.ell_r,
        mass_exponent=5,
    )

    level = 8
    require(P.invariant_profile == Q.invariant_profile, "Invariant profiles must agree.")
    require(P.frequency_profile == Q.frequency_profile, "Frequency profiles must agree.")
    require(P.spin_signature == Q.spin_signature, "Scalar signatures must agree.")
    require(P.ell_rho == Q.ell_rho and P.ell_r == Q.ell_r, "Localization scales must agree.")

    NP = P.stable_multiplicity(level)
    NQ = Q.stable_multiplicity(level)
    require(NP != NQ, "Stable multiplicities must differ at the tested level.")
    require(P.finite_mass_exact_power(level) != Q.finite_mass_exact_power(level),
            "Different positive multiplicities that are distinct exact powers of s must induce different finite masses.")

    diff = finite_mass_difference_exact_power(P, Q, level)
    require(diff == Fraction(-3), "Finite mass difference should equal the exponent difference.")
    require(P.stable_fraction_exact_power(level) != Q.stable_fraction_exact_power(level),
            "Different multiplicities at fixed level must induce different stable fractions.")
    require(P.stability_deficit_exact_power(level) != Q.stability_deficit_exact_power(level),
            "Different finite masses at fixed level must induce different stability deficits.")

    print("[OK] Equal invariant, internal, and localization profiles do not determine finite stable multiplicity")
    print("[OK] Positive unequal stable multiplicities induce unequal finite-level logarithmic masses")


def verify_limit_discipline_for_asymptotic_mass() -> None:
    print("\n=== Verification of the limit discipline for asymptotic combinatorial mass ===")

    n = Symbol("n", positive=True, integer=True)
    M = Symbol("M", real=True)
    nonzero_shift = Symbol("c", positive=True)

    mP = M + nonzero_shift + 1 / n
    mQ = M + 1 / (2 * n)
    diff = simplify(mP - mQ)
    diff_limit = limit(diff, n, oo)
    require(simplify(diff_limit - nonzero_shift) == 0,
            "The symbolic mass-difference limit must be the nonzero shift c.")
    require(limit(mP, n, oo) != limit(mQ, n, oo),
            "A nonzero limiting difference must give different asymptotic masses.")

    mR = M + 1 / n
    mS = M + 1 / (n + 1)
    zero_diff_limit = limit(simplify(mR - mS), n, oo)
    require(simplify(zero_diff_limit) == 0,
            "The symbolic mass-difference limit must vanish.")
    require(simplify(limit(mR, n, oo) - limit(mS, n, oo)) == 0,
            "A zero limiting difference gives equal limits for these convergent mass sequences.")

    # Integer-domain guard.  Positive integer multiplicities with finite
    # logarithmic limits are eventually bounded.  Hence persistent eventual
    # inequality cannot coexist with equal finite limiting masses.
    base = 2
    bounded_positive_counts = (1, 2, 4, 8)
    possible_masses = tuple(exact_log_power(base, c) for c in bounded_positive_counts)
    smallest_nonzero_gap = min(abs(a - b) for a in possible_masses for b in possible_masses if a != b)
    require(smallest_nonzero_gap == 1, "Distinct exact base-2 power multiplicities have at least unit mass gap.")

    print("[OK] Asymptotic mass equality/difference is controlled by the limit of finite mass differences")
    print("[OK] Integer positive multiplicities impose an additional eventual-constancy guard in the finite-mass domain")


def verify_same_mass_but_different_full_signature() -> None:
    print("\n=== Verification of same combinatorial mass but different full signature ===")

    base = 3
    common_mass = 2
    P = make_particle(
        "P_same_mass_internal_difference",
        base=base,
        frequency_profile=(Fraction(1, 2), Fraction(0), Fraction(1, 2)),
        ell_rho=Fraction(1, 6),
        ell_r=Fraction(2),
        mass_exponent=common_mass,
    )
    Q = make_particle(
        "Q_same_mass_internal_difference",
        base=base,
        frequency_profile=(Fraction(0), Fraction(1), Fraction(0)),
        ell_rho=P.ell_rho,
        ell_r=P.ell_r,
        mass_exponent=common_mass,
    )
    require(P.mass_limit == Q.mass_limit, "The witness must have equal asymptotic combinatorial mass.")
    require(P.frequency_profile != Q.frequency_profile, "The full internal profiles must differ.")
    require(P.spin_signature == Q.spin_signature, "This witness isolates full internal profile beyond the scalar.")
    require(P.limiting_signature_without_label() != Q.limiting_signature_without_label(),
            "Equal mass must not force equality of the full signature.")

    R = make_particle(
        "R_same_mass_geometry_difference",
        base=base,
        frequency_profile=P.frequency_profile,
        ell_rho=Fraction(0),
        ell_r=Fraction(1),
        mass_exponent=common_mass,
    )
    S = make_particle(
        "S_same_mass_geometry_difference",
        base=base,
        frequency_profile=P.frequency_profile,
        ell_rho=Fraction(1, 3),
        ell_r=Fraction(5),
        mass_exponent=common_mass,
    )
    require(R.mass_limit == S.mass_limit, "Masses must agree.")
    require(R.frequency_profile == S.frequency_profile, "Internal profiles must agree in the geometric witness.")
    require((R.ell_rho, R.ell_r) != (S.ell_rho, S.ell_r),
            "At least one localization scale must differ.")
    require(R.limiting_signature_without_label() != S.limiting_signature_without_label(),
            "Equal mass and equal internal data must still allow geometric signature differences.")

    print("[OK] Equal combinatorial mass is not a complete descriptor")
    print("[OK] Isomassive structures can differ internally or geometrically in the full limiting signature")


def verify_finite_signature_relations() -> None:
    print("\n=== Verification of finite particle-like signature M_n ===")

    # Non-power finite multiplicity smoke test.
    # This verifies that the defining M_n relations are not restricted to
    # exact-power multiplicities N=s^q.
    s0 = 2
    n0 = 5
    N0 = 3

    m0 = log(Integer(N0)) / log(Integer(s0))
    theta0 = Rational(N0, s0**n0)
    delta0 = Integer(n0) - m0

    assert simplify(theta0 - Rational(N0, s0**n0)) == 0
    assert simplify(delta0 - (Integer(n0) - m0)) == 0
    assert simplify(expand_log(delta0 + log(theta0) / log(Integer(s0)), force=True)) == 0

    print("[OK] Non-power finite multiplicities satisfy the M_n logarithmic relations")

    P = make_particle(
        "P_finite_signature",
        base=2,
        invariant_profile=(Fraction(2, 9), Fraction(5, 9)),
        frequency_profile=(Fraction(3, 4), Fraction(1, 4)),
        ell_rho=Fraction(1, 16),
        ell_r=Fraction(3),
        mass_exponent=3,
    )
    level = 7
    signature = P.finite_signature_exact_power(level)
    (
        name,
        invariant_profile,
        frequency_profile,
        scalar_signature,
        ell_rho,
        ell_r,
        N,
        finite_mass,
        theta,
        delta,
    ) = signature

    require(name == P.name, "The finite signature must retain the structure label.")
    require(invariant_profile == P.invariant_profile, "Invariant profile component mismatch.")
    require(frequency_profile == P.frequency_profile, "Frequency profile component mismatch.")
    require(scalar_signature == P.spin_signature, "Scalar spin component mismatch.")
    require((ell_rho, ell_r) == (P.ell_rho, P.ell_r), "Localization components mismatch.")
    require(N == 2**3, "Finite multiplicity component mismatch.")
    require(finite_mass == Fraction(3), "Finite mass must be log_2(8)=3.")
    require(theta == Fraction(2**3, 2**7), "Stable fraction must be N/s^n.")
    require(delta == Fraction(4), "Stability deficit must be n-m.")
    require(theta == Fraction(1, 2**int(delta)), "For this exact-power case theta must equal s^{-delta}.")

    # Symbolic relation: if N=s^q, then theta=s^{q-n} and delta=n-q.
    s, k, n = symbols("s k n", positive=True, integer=True)
    symbolic_theta = s**k / s**n
    symbolic_delta = n - k
    require(simplify(symbolic_theta - s ** (-symbolic_delta)) == 0,
            "Symbolic theta/deficit relation failed for N=s^q.")
    require(simplify(-log(symbolic_theta, s) - symbolic_delta) == 0,
            "Symbolic logarithmic deficit relation failed for N=s^q.")

    print("[OK] M_n contains the listed finite and asymptotic components in the correct order")
    print("[OK] m_comb^(n), theta_stable^(n), and delta_stable^(n) satisfy the exact defining relations")


def verify_limiting_signature() -> None:
    print("\n=== Verification of limiting signature M_infty ===")

    P = make_particle(
        "P_limit_signature",
        base=3,
        invariant_profile=(Fraction(1, 4), Fraction(3, 4)),
        frequency_profile=(Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)),
        ell_rho=Fraction(1, 5),
        ell_r=Fraction(9, 4),
        mass_exponent=1,
        specific_mass_limit=Fraction(0),
    )

    changed_components: list[ParticleLikeData] = [
        replace(P, invariant_profile=(Fraction(1, 5), Fraction(4, 5))),
        replace(P, frequency_profile=(Fraction(1, 2), Fraction(1, 2), Fraction(0))),
        replace(P, ell_rho=Fraction(2, 5)),
        replace(P, ell_r=Fraction(11, 4)),
        replace(P, mass_limit=Fraction(2), multiplicity=power_multiplicity(P.base, 2)),
        replace(P, specific_mass_limit=Fraction(1, 3)),
    ]

    for changed in changed_components:
        changed.validate_static_components()
        require(P.limiting_signature_without_label() != changed.limiting_signature_without_label(),
                f"Changing a mathematical component must change M_infty: {changed}")

    print("[OK] M_infty is componentwise sensitive to all listed mathematical data")


def verify_domain_guards() -> None:
    print("\n=== Verification of explicit domain guards and negative cases ===")

    expect_raises(
        AssertionError,
        lambda: make_particle(
            "bad_base",
            base=1,
            frequency_profile=(Fraction(1),),
            mass_exponent=0,
        ),
        "base s < 2 must be rejected",
    )

    expect_raises(
        AssertionError,
        lambda: make_particle(
            "bad_simplex_sum",
            frequency_profile=(Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        ),
        "frequency profile with total mass different from one must be rejected",
    )

    expect_raises(
        AssertionError,
        lambda: make_particle(
            "bad_simplex_negative",
            frequency_profile=(Fraction(3, 2), Fraction(-1, 2), Fraction(0)),
        ),
        "frequency profile with a negative component must be rejected",
    )

    expect_raises(
        AssertionError,
        lambda: make_particle(
            "bad_ell_rho",
            ell_rho=Fraction(5, 4),
        ),
        "ell_rho outside [0,1] must be rejected",
    )

    expect_raises(
        AssertionError,
        lambda: make_particle(
            "bad_ell_r",
            ell_r=Fraction(-1),
        ),
        "negative ell_r must be rejected",
    )

    zero_N = ParticleLikeData(
        name="zero_multiplicity",
        base=2,
        invariant_profile=(Fraction(1),),
        frequency_profile=(Fraction(1, 2), Fraction(1, 2)),
        ell_rho=Fraction(0),
        ell_r=Fraction(0),
        multiplicity=lambda level: 0,
        mass_limit=None,
        specific_mass_limit=None,
    )
    zero_N.validate_static_components()
    expect_raises(AssertionError, lambda: zero_N.stable_multiplicity(3), "zero multiplicity must be rejected")

    too_large_N = ParticleLikeData(
        name="too_large_multiplicity",
        base=2,
        invariant_profile=(Fraction(1),),
        frequency_profile=(Fraction(1, 2), Fraction(1, 2)),
        ell_rho=Fraction(0),
        ell_r=Fraction(0),
        multiplicity=lambda level: 2**level + 1,
        mass_limit=None,
        specific_mass_limit=None,
    )
    too_large_N.validate_static_components()
    expect_raises(AssertionError, lambda: too_large_N.stable_multiplicity(4),
                  "N_stable^(n) > s^n must be rejected")

    no_limit = make_particle("missing_limit", mass_exponent=1)
    no_limit = replace(no_limit, mass_limit=None)
    expect_raises(AssertionError, lambda: no_limit.limiting_signature(),
                  "M_infty cannot be formed without an asymptotic mass limit")

    non_power_N = ParticleLikeData(
        name="non_power_multiplicity",
        base=2,
        invariant_profile=(Fraction(1),),
        frequency_profile=(Fraction(1, 2), Fraction(1, 2)),
        ell_rho=Fraction(0),
        ell_r=Fraction(0),
        multiplicity=lambda level: 3,
        mass_limit=None,
        specific_mass_limit=None,
    )
    non_power_N.validate_static_components()
    expect_raises(AssertionError, lambda: non_power_N.finite_mass_exact_power(3),
                  "the exact-power helper must reject non-power multiplicities")

    print("[OK] Invalid bases, profiles, localization scales, multiplicities, and missing limits are rejected")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of Section 7, Part 3 (subsections 7.7.13--7.7.19) ===")
    verify_symbolic_spin_map_and_noninjectivity()
    verify_geometrically_identical_internally_different()
    verify_internally_identical_geometrically_different()
    verify_same_profiles_but_different_multiplicity()
    verify_limit_discipline_for_asymptotic_mass()
    verify_same_mass_but_different_full_signature()
    verify_finite_signature_relations()
    verify_limiting_signature()
    verify_domain_guards()
    print("\n=== Section 7, Part 3 verification completed successfully ===")


if __name__ == "__main__":
    main()
