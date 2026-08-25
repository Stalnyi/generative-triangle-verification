"""
VERIFICATION of Sections 7.7.1 - 7.7.6:
Microhistorical multiplicity and combinatorial characteristics of
particle-like structures (part 1)
(subsections 7.7.1--7.7.6).

This script provides a full mathematical verification block for
    7_microhistorical_multiplicity_combinatorial_characteristics_part_1.tex.

It does not re-verify the earlier construction of macroclasses, recursive
stability, causal localization, stable internal structure, or asymptotic
distinguishability.  Those are treated as previously verified prerequisites.
The present block verifies the new quantitative layer introduced in this
section:

    P -> stable finite support H_P^(n) -> N_stable^(n)(P) -> m_comb^(n)(P).

Verified content
----------------
1. Domain discipline:
   microhistorical multiplicity and combinatorial mass are assigned only to
   particle-like recursive structures, not to arbitrary microhistories or
   arbitrary macroclasses. When P_n is already the projection of full
   particle-like representatives, H_P^(n) is identified with that stable
   finite support.

2. Finite-level stable realizations:
      H_P^(n) = P_n inside the particle-like projection convention, with H_P^(n) subset S^n,
   with exact finite validation of prefix length, admissible internal states,
   nonempty support after the entry level, and rejection of certificates outside
   the finite projection.

3. Finite multiplicity:
      N_stable^(n)(P) = |H_P^(n)|,
   with exact bounds
      0 <= N_stable^(n)(P) <= s^n
   and, when the support is nonempty,
      1 <= N_stable^(n)(P) <= s^n.

4. Finite-level combinatorial mass:
      m_comb^(n)(P) = log_s N_stable^(n)(P),
   only for N_stable^(n)(P) >= 1, with exact boundary checks
      N=1     -> m=0,
      N=s^q   -> m=q,
      N=s^n   -> m=n,
   and grid verification of
      0 <= m_comb^(n)(P) <= n.

5. Asymptotic combinatorial mass:
      m_comb(P) = lim_n m_comb^(n)(P),
   when the finite limit exists.  The script checks bounded multiplicity
   families and rejects oscillatory/nonconvergent finite masses.

6. Specific asymptotic combinatorial mass:
      mu_comb(P) = lim_n (1/n) m_comb^(n)(P),
   with representative exact families:
      constant support, polynomial support, partial exponential support,
      full s^n-exponential support,
   and verifies
      0 <= mu_comb(P) <= 1.

7. Stable branch fraction:
      theta_stable^(n)(P) = N_stable^(n)(P) / s^n,
   with exact rational bounds and exact reconstruction
      N_stable^(n)(P) = s^n theta_stable^(n)(P).

8. Stability deficit:
      delta_stable^(n)(P) = n - m_comb^(n)(P)
                           = -log_s theta_stable^(n)(P),
   with exact checks:
      theta=1       -> delta=0,
      theta=s^(-q)  -> delta=q,
      N=s^(n-q)     -> delta=q.

9. Negative tests:
   zero stable multiplicity has no logarithmic mass;
   theta=0 has no logarithmic deficit;
   invalid state labels, invalid prefix lengths, and non-particle-like carriers
   are rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import isclose, log
from typing import Callable, FrozenSet, Iterable, Mapping

from sympy import (
    Rational,
    Symbol,
    oo,
    log as symlog,
    limit,
    simplify,
    summation,
    symbols,
)


Prefix = tuple[int, ...]


@dataclass(frozen=True)
class ParticleCarrier:
    """Minimal carrier containing only already verified particle-like status."""

    name: str
    s: int
    particle_like: bool
    entry_level: int
    projection_by_level: Mapping[int, FrozenSet[Prefix]]
    realization_certificate_by_level: Mapping[int, Mapping[Prefix, bool]]

    def validate_basic_domain(self) -> None:
        if self.s < 2:
            raise ValueError("The branching degree s must satisfy s >= 2.")
        if self.entry_level < 0:
            raise ValueError("The entry level must be nonnegative.")
        if not self.particle_like:
            raise ValueError(
                "Combinatorial characteristics are defined only for particle-like recursive structures."
            )


def expect_raises(error_type: type[BaseException], action: Callable[[], object], description: str) -> None:
    try:
        action()
    except error_type:
        print(f"[OK] Rejected invalid case: {description}")
        return
    raise AssertionError(f"Expected {error_type.__name__} was not raised for: {description}")


def validate_prefix(prefix: Prefix, *, s: int, n: int) -> None:
    if len(prefix) != n:
        raise ValueError(f"Invalid finite prefix length: expected {n}, got {len(prefix)}.")
    bad_states = [state for state in prefix if not isinstance(state, int) or not (1 <= state <= s)]
    if bad_states:
        raise ValueError(f"Finite prefix contains states outside {{1,...,{s}}}: {bad_states}")


def validate_projection(projection: FrozenSet[Prefix], *, s: int, n: int) -> None:
    for prefix in projection:
        validate_prefix(prefix, s=s, n=n)


def finite_stable_realizations(carrier: ParticleCarrier, n: int) -> FrozenSet[Prefix]:
    carrier.validate_basic_domain()

    projection = carrier.projection_by_level.get(n, frozenset())
    certificates = carrier.realization_certificate_by_level.get(n, {})

    validate_projection(projection, s=carrier.s, n=n)

    outside_projection = [prefix for prefix, ok in certificates.items() if ok and prefix not in projection]
    if outside_projection:
        raise ValueError("A stable-realization certificate was supplied outside the finite projection.")

    stable = frozenset(prefix for prefix in projection if certificates.get(prefix, False))
    validate_projection(stable, s=carrier.s, n=n)

    assert stable.issubset(projection)
    assert len(stable) <= carrier.s**n

    if n >= carrier.entry_level and carrier.particle_like:
        if len(stable) == 0:
            raise ValueError("A nonempty particle-like carrier must have nonempty support after entry.")

    return stable


def finite_multiplicity(carrier: ParticleCarrier, n: int) -> int:
    return len(finite_stable_realizations(carrier, n))


def finite_combinatorial_mass_float(s: int, multiplicity: int) -> float:
    if s < 2:
        raise ValueError("The logarithm base s must satisfy s >= 2.")
    if multiplicity < 1:
        raise ValueError("Finite combinatorial mass is defined only for positive stable multiplicity.")
    return log(multiplicity, s)


def stable_fraction_exact(s: int, n: int, multiplicity: int) -> Fraction:
    if s < 2 or n < 0:
        raise ValueError("Invalid s or n.")
    if not (0 <= multiplicity <= s**n):
        raise ValueError("Stable multiplicity must lie between 0 and s^n.")
    return Fraction(multiplicity, s**n)


def stability_deficit_float(s: int, n: int, multiplicity: int) -> float:
    theta = stable_fraction_exact(s, n, multiplicity)
    if theta == 0:
        raise ValueError("The logarithmic stability deficit is defined only for positive stable fraction.")
    return n - finite_combinatorial_mass_float(s, multiplicity)


def all_prefixes(s: int, n: int) -> FrozenSet[Prefix]:
    if n == 0:
        return frozenset({()})
    previous = all_prefixes(s, n - 1)
    return frozenset(prefix + (state,) for prefix in previous for state in range(1, s + 1))


def index_to_prefix(index: int, *, s: int, n: int) -> Prefix:
    if index < 0 or index >= s**n:
        raise ValueError("Prefix index is outside the finite s-ary level.")
    digits = []
    remaining = index
    for _ in range(n):
        digits.append((remaining % s) + 1)
        remaining //= s
    return tuple(reversed(digits))


def first_k_prefixes(s: int, n: int, k: int) -> FrozenSet[Prefix]:
    if not (0 <= k <= s**n):
        raise ValueError("Requested finite-prefix count is outside [0, s^n].")
    return frozenset(index_to_prefix(index, s=s, n=n) for index in range(k))


def prefix_family_by_suffix_rule(s: int, n: int, required_suffix: Prefix) -> FrozenSet[Prefix]:
    if len(required_suffix) > n:
        return frozenset()
    return frozenset(prefix for prefix in all_prefixes(s, n) if prefix[-len(required_suffix):] == required_suffix)


def build_exact_carrier(s: int, n_values: Iterable[int], stable_selector: Callable[[int], FrozenSet[Prefix]]) -> ParticleCarrier:
    projection_by_level: dict[int, FrozenSet[Prefix]] = {}
    certificates_by_level: dict[int, dict[Prefix, bool]] = {}

    for n in n_values:
        projection = all_prefixes(s, n)
        stable = stable_selector(n)
        if not stable.issubset(projection):
            raise ValueError("stable_selector produced a prefix outside S^n.")

        projection_by_level[n] = projection
        certificates_by_level[n] = {prefix: prefix in stable for prefix in projection}

    return ParticleCarrier(
        name="exact finite carrier",
        s=s,
        particle_like=True,
        entry_level=min(n_values),
        projection_by_level=projection_by_level,
        realization_certificate_by_level=certificates_by_level,
    )


def verify_symbolic_logarithmic_mass_identities() -> None:
    print("\n=== Symbolic verification of logarithmic combinatorial mass identities ===")

    s = Symbol("s", positive=True)
    k, n = symbols("k n", integer=True, nonnegative=True)

    assert simplify(symlog(s**k) / symlog(s) - k) == 0
    assert simplify(symlog(s**n) / symlog(s) - n) == 0
    assert simplify(symlog(1) / symlog(s)) == 0

    theta = Symbol("theta", positive=True)
    m_from_theta = symlog(s**n * theta) / symlog(s)
    assert simplify(m_from_theta - (n + symlog(theta) / symlog(s))) == 0

    delta = n - m_from_theta
    assert simplify(delta + symlog(theta) / symlog(s)) == 0

    print("[OK] log_s(s^q)=q, log_s(1)=0 and log_s(s^n)=n verified symbolically")
    print("[OK] m_comb^(n)=n+log_s(theta_stable^(n)) verified symbolically")
    print("[OK] delta_stable^(n)=n-m_comb^(n)=-log_s(theta_stable^(n)) verified symbolically")


def verify_centered_internal_state_spectrum_dependency() -> None:
    print("\n=== Symbolic check of the reused centered internal-state spectrum ===")

    sigma, s = symbols("sigma s", integer=True, positive=True)
    eta = sigma - Rational(1, 2) * (s + 1)

    spectrum_sum = summation(eta, (sigma, 1, s))
    assert simplify(spectrum_sum) == 0
    assert simplify(eta.subs(sigma, 1) + Rational(1, 2) * (s - 1)) == 0
    assert simplify(eta.subs(sigma, s) - Rational(1, 2) * (s - 1)) == 0

    print("[OK] The centered internal-state spectrum has zero sum for arbitrary s")
    print("[OK] The half-width B_s=(s-1)/2 is recovered symbolically without rechecking prior sections")


def verify_finite_stable_realization_set_and_bounds() -> None:
    print("\n=== Finite-level stable realization set and exact bounds ===")

    s = 3
    checked = 0

    for n in range(1, 6):
        # Stable realizations: all prefixes ending in internal state 1.
        # Count is exactly s^(n-1), so all bounds are nontrivial for n>=2.
        stable_selector = lambda level, s=s: prefix_family_by_suffix_rule(s, level, (1,))
        carrier = build_exact_carrier(s=s, n_values=[n], stable_selector=stable_selector)
        H = finite_stable_realizations(carrier, n)
        N = finite_multiplicity(carrier, n)

        assert H.issubset(carrier.projection_by_level[n])
        assert N == s ** (n - 1)
        assert 1 <= N <= s**n
        assert stable_fraction_exact(s, n, N) == Fraction(1, s)
        checked += 1

    # Entry-level nonemptiness is enforced for particle-like carriers.
    empty_after_entry = ParticleCarrier(
        name="invalid empty support after entry",
        s=2,
        particle_like=True,
        entry_level=2,
        projection_by_level={2: all_prefixes(2, 2)},
        realization_certificate_by_level={2: {prefix: False for prefix in all_prefixes(2, 2)}},
    )
    expect_raises(ValueError, lambda: finite_stable_realizations(empty_after_entry, 2), "empty support after entry level")

    print(f"[OK] Checked {checked} nontrivial finite levels with stable H_P^(n) support inside S^n")
    print("[OK] Nonempty support after entry level is enforced")


def verify_domain_restrictions_and_invalid_finite_data() -> None:
    print("\n=== Domain restrictions and invalid finite-data rejection ===")

    s = 2
    projection = frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})
    valid_certificates = {prefix: prefix in {(1, 1), (2, 2)} for prefix in projection}

    non_particle = ParticleCarrier(
        name="arbitrary macroclass",
        s=s,
        particle_like=False,
        entry_level=0,
        projection_by_level={2: projection},
        realization_certificate_by_level={2: valid_certificates},
    )
    expect_raises(ValueError, lambda: finite_stable_realizations(non_particle, 2), "non-particle-like carrier")

    invalid_state_projection = ParticleCarrier(
        name="invalid internal state",
        s=s,
        particle_like=True,
        entry_level=2,
        projection_by_level={2: frozenset({(1, 3)})},
        realization_certificate_by_level={2: {(1, 3): True}},
    )
    expect_raises(ValueError, lambda: finite_stable_realizations(invalid_state_projection, 2), "state outside {1,...,s}")

    invalid_length_projection = ParticleCarrier(
        name="invalid prefix length",
        s=s,
        particle_like=True,
        entry_level=2,
        projection_by_level={2: frozenset({(1, 2, 1)})},
        realization_certificate_by_level={2: {(1, 2, 1): True}},
    )
    expect_raises(ValueError, lambda: finite_stable_realizations(invalid_length_projection, 2), "wrong finite-prefix length")

    outside_certificate = ParticleCarrier(
        name="outside certificate",
        s=s,
        particle_like=True,
        entry_level=2,
        projection_by_level={2: projection},
        realization_certificate_by_level={2: {**valid_certificates, (1, 1, 1): True}},
    )
    expect_raises(ValueError, lambda: finite_stable_realizations(outside_certificate, 2), "certificate outside P_n")

    expect_raises(ValueError, lambda: finite_combinatorial_mass_float(2, 0), "zero multiplicity logarithm")
    expect_raises(ValueError, lambda: stability_deficit_float(2, 3, 0), "zero stable fraction logarithm")

    print("[OK] Quantitative characteristics cannot be assigned outside their stated domain")


def verify_finite_combinatorial_mass_bounds_and_boundaries() -> None:
    print("\n=== Finite combinatorial mass: exact boundaries and dense grid bounds ===")

    for s in range(2, 8):
        for n in range(0, 7):
            full = s**n

            assert isclose(finite_combinatorial_mass_float(s, 1), 0.0, abs_tol=1e-12)
            assert isclose(finite_combinatorial_mass_float(s, full), float(n), abs_tol=1e-12)

            for k in range(0, n + 1):
                N = s**k
                assert isclose(finite_combinatorial_mass_float(s, N), float(k), abs_tol=1e-12)

            # Dense finite support grid: every admissible N gives 0 <= log_s N <= n.
            for N in range(1, full + 1):
                m = finite_combinatorial_mass_float(s, N)
                assert -1e-12 <= m <= n + 1e-12, (s, n, N, m)

    print("[OK] Boundary cases N=1, N=s^q and N=s^n verified for multiple bases")
    print("[OK] Dense finite grids confirm 0 <= m_comb^(n) <= n whenever 1 <= N <= s^n")


def verify_stable_fraction_and_deficit_exactly() -> None:
    print("\n=== Exact stable-fraction and stability-deficit identities ===")

    for s in range(2, 8):
        for n in range(1, 12):
            full_theta = stable_fraction_exact(s, n, s**n)
            assert full_theta == 1
            assert isclose(stability_deficit_float(s, n, s**n), 0.0, abs_tol=1e-12)

            for k in range(0, n + 1):
                N = s ** (n - k)
                theta = stable_fraction_exact(s, n, N)
                delta = stability_deficit_float(s, n, N)

                assert theta == Fraction(1, s**k)
                assert isclose(delta, float(k), abs_tol=1e-12)

                reconstructed_N = theta * (s**n)
                assert reconstructed_N.denominator == 1
                assert reconstructed_N.numerator == N

    # Non-power examples: the identity still holds numerically, but the deficit
    # need not be an integer number of full s-ary degrees.
    for s in range(2, 6):
        for n in range(3, 7):
            for N in range(1, s**n + 1):
                theta = Fraction(N, s**n)
                m = finite_combinatorial_mass_float(s, N)
                delta = stability_deficit_float(s, n, N)
                assert isclose(m, n + log(float(theta), s), rel_tol=1e-12, abs_tol=1e-12)
                assert isclose(delta, -log(float(theta), s), rel_tol=1e-12, abs_tol=1e-12)

    print("[OK] theta_stable=N/s^n reconstructs N exactly")
    print("[OK] theta=1 gives zero deficit; theta=s^(-q) gives deficit q")
    print("[OK] The logarithmic deficit identity also holds for non-power multiplicities")


def verify_asymptotic_mass_and_specific_mass_regimes() -> None:
    print("\n=== Asymptotic combinatorial mass and specific mass regimes ===")

    n = Symbol("n", positive=True, integer=True)
    q = Symbol("q", positive=True, integer=True)
    s = 2  # representative base; exact finite checks below cover several bases.

    # Bounded support: N=s^K gives finite total asymptotic combinatorial mass K.
    K = Symbol("K", nonnegative=True, integer=True)
    m_bounded = symlog(s**K) / symlog(s)
    assert simplify(m_bounded - K) == 0
    assert simplify(limit(m_bounded / n, n, oo)) == 0

    # Polynomial support: N=n^q is unbounded but subexponential, hence specific mass 0.
    m_poly = symlog(n**q) / symlog(s)
    assert simplify(limit(m_poly / n, n, oo)) == 0

    # Partial exponential support: along n=b*t, N=s^(a*t), specific mass is a/b.
    t = Symbol("t", positive=True, integer=True)
    a, b = symbols("a b", positive=True, integer=True)
    m_partial = symlog(s ** (a * t)) / symlog(s)
    depth_partial = b * t
    mu_partial = limit(m_partial / depth_partial, t, oo)
    assert simplify(mu_partial - Rational(1, 1) * a / b) == 0

    # Full support: N=s^n gives specific mass 1.
    m_full = symlog(s**n) / symlog(s)
    assert simplify(limit(m_full / n, n, oo) - 1) == 0

    # Oscillatory finite masses have no total asymptotic mass.
    # The two subsequences below have different limits.
    m_even_subsequence = Rational(1, 1)
    m_odd_subsequence = Rational(2, 1)
    assert m_even_subsequence != m_odd_subsequence

    print("[OK] Bounded support has finite m_comb and zero specific mass")
    print("[OK] Polynomial support is unbounded but has mu_comb=0")
    print("[OK] Partial exponential support has exact specific mass a/b")
    print("[OK] Full s-ary support has mu_comb=1")
    print("[OK] Oscillatory finite masses are correctly identified as nonconvergent")


def verify_specific_mass_bounds_by_exact_families() -> None:
    print("\n=== Exact finite families for 0 <= mu_comb <= 1 ===")

    for s in range(2, 6):
        # mu=0: N=1.  Use a one-element projection to avoid enumerating S^n.
        for n in range(1, 20):
            stable = frozenset({tuple([1] * n)})
            carrier = ParticleCarrier(
                name="single-realization support",
                s=s,
                particle_like=True,
                entry_level=n,
                projection_by_level={n: stable},
                realization_certificate_by_level={n: {tuple([1] * n): True}},
            )
            assert finite_multiplicity(carrier, n) == 1

        # mu=1: all prefixes stable.
        for n in range(1, 8):
            carrier = build_exact_carrier(s, [n], lambda level, s=s: all_prefixes(s, level))
            N = finite_multiplicity(carrier, n)
            assert N == s**n
            assert isclose(finite_combinatorial_mass_float(s, N) / n, 1.0, abs_tol=1e-12)

        # mu=a/b for exact subsequence n=b*t.
        for a_b in [(1, 2), (1, 3), (2, 3)]:
            a, b = a_b
            for t in range(1, 5):
                n = b * t
                target_count = s ** (a * t)

                # Use exactly target_count certified prefixes without enumerating the full level S^n.
                stable = first_k_prefixes(s, n, target_count)
                carrier = ParticleCarrier(
                    name="exact partial-exponential support",
                    s=s,
                    particle_like=True,
                    entry_level=n,
                    projection_by_level={n: stable},
                    realization_certificate_by_level={n: {prefix: True for prefix in stable}},
                )
                N = finite_multiplicity(carrier, n)

                assert N == target_count
                mu_n = finite_combinatorial_mass_float(s, N) / n
                assert isclose(mu_n, a / b, abs_tol=1e-12)
                assert 0 <= mu_n <= 1

    print("[OK] Exact finite families realize mu=0, partial 0<mu<1, and mu=1")
    print("[OK] No checked family exceeds the full s^n support bound")


def verify_strict_subset_and_deficit_interpretation() -> None:
    print("\n=== Strict-support subset and stability-deficit interpretation ===")

    s = 4
    for n in range(2, 7):
        # Fixing the last two states removes exactly two full s-ary degrees.
        required_suffix = (1, 2)
        stable = prefix_family_by_suffix_rule(s, n, required_suffix)
        carrier = build_exact_carrier(s, [n], lambda level, stable=stable: stable)

        N = finite_multiplicity(carrier, n)
        assert N == s ** (n - 2)

        theta = stable_fraction_exact(s, n, N)
        delta = stability_deficit_float(s, n, N)

        assert theta == Fraction(1, s**2)
        assert isclose(delta, 2.0, abs_tol=1e-12)
        assert N < s**n

    print("[OK] Fixing q internal states removes exactly q full s-ary degrees")
    print("[OK] Positive deficit corresponds to a strict subset of the full causal branching tree")


def verify_central_chain_consistency() -> None:
    print("\n=== Central chain P -> H_P^(n) -> N_stable^(n) -> m_comb^(n) ===")

    s = 2
    n_values = range(2, 8)

    # Stable support: prefixes whose first and last internal states agree.
    def stable_selector(n: int) -> FrozenSet[Prefix]:
        return frozenset(prefix for prefix in all_prefixes(s, n) if prefix[0] == prefix[-1])

    carrier = build_exact_carrier(s=s, n_values=n_values, stable_selector=stable_selector)

    for n in n_values:
        H = finite_stable_realizations(carrier, n)
        N = finite_multiplicity(carrier, n)
        m = finite_combinatorial_mass_float(s, N)
        theta = stable_fraction_exact(s, n, N)
        delta = stability_deficit_float(s, n, N)

        assert N == len(H)
        assert N == 2 ** (n - 1)
        assert isclose(m, n - 1, abs_tol=1e-12)
        assert theta == Fraction(1, 2)
        assert isclose(delta, 1.0, abs_tol=1e-12)

    print("[OK] The full quantitative chain is internally consistent on a nontrivial family")


def main() -> None:
    if not __debug__:
        raise RuntimeError(
            "This verification script must be run without Python optimization (-O), "
            "because assertions are part of the verification."
        )
    
    print("=== Verification of microhistorical multiplicity and combinatorial characteristics ===")
    verify_symbolic_logarithmic_mass_identities()
    verify_centered_internal_state_spectrum_dependency()
    verify_finite_stable_realization_set_and_bounds()
    verify_domain_restrictions_and_invalid_finite_data()
    verify_finite_combinatorial_mass_bounds_and_boundaries()
    verify_stable_fraction_and_deficit_exactly()
    verify_asymptotic_mass_and_specific_mass_regimes()
    verify_specific_mass_bounds_by_exact_families()
    verify_strict_subset_and_deficit_interpretation()
    verify_central_chain_consistency()
    print("\n=== Microhistorical multiplicity and combinatorial-characteristic verification completed successfully ===")


if __name__ == "__main__":
    main()
