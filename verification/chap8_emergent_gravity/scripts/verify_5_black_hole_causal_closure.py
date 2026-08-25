"""
VERIFICATION of Section:
Black-hole regime as limiting causal closure
(sec:black-hole-causal-closure).

Source file:
    5_black_hole_causal_closure.tex

This script verifies the mathematical content of the section as a finite
escaping-measure and limiting-deficit layer over the previously verified causal
cone, sector decomposition, compatible stable sector measures, and sector
deficit formalism.

The section does not introduce an astrophysical black hole as a primitive
object, does not introduce a primary singularity, and does not alter the
fundamental causal speed bound c=1.  It defines the black-hole regime as the
asymptotic vanishing of the compatible stable measure of escaping causal
continuations:

    lim_{ell->infty} mu_esc^(ell)(x;Q|P) = 0.

Verified content
----------------
1. Exterior region:
       Ext_{n+ell}(P) = L_{n+ell} \\ Loc_{n+ell}(P).
   In finite digit coordinates, this is implemented as
       Ext = FullLayer \\ Loc.

2. Escaping causal continuation:
   a continuation from x to y is escaping exactly when its endpoint digit lies
   in Ext.

3. Escaping sectors:
       S_esc^(ell)(x;P)
   consists exactly of primary sectors whose full causal sector slice
   intersects Ext.

4. Escaping sector fragments:
       E_{Q|P,sigma} subseteq E_{Q,0,sigma}.
   Fragments are represented as pairs (realization_id, endpoint_digit), so the
   script does not count an entire sector as escaping merely because one
   exterior endpoint exists.

5. Sector escaping stable measures:
       mu_{esc,sigma}=|E_{Q|P,sigma}|/s^ell,
       mu_{esc,0,sigma}=|E_{Q,0,sigma}|/s^ell.

6. Total escaping measures:
       mu_esc=sum_sigma mu_{esc,sigma},
       mu_esc,0=sum_sigma mu_{esc,0,sigma}.

7. Sector escaping deficit:
       delta_mu_{esc,sigma}=mu_{esc,sigma}-mu_{esc,0,sigma} <= 0.

8. Total escaping deficit:
       delta_mu_esc=mu_esc-mu_esc,0
                  =sum_sigma delta_mu_{esc,sigma} <= 0.

9. Strict escaping-deficit criterion:
       delta_mu_esc < 0
   iff there exists a sector sigma with
       E_{Q|P,sigma} proper subset of E_{Q,0,sigma}.

10. Black-hole regime:
       lim mu_esc^(ell)=0.

11. Effective horizon:
       H(P;Q)={x in T : lim mu_esc^(ell)(x;Q|P)=0}.

12. Finite epsilon horizon:
       H_epsilon^(ell)(P;Q)
       =
       {x in T : mu_esc^(ell)(x;Q|P)<epsilon}.

13. Finite epsilon-horizon characterization of the effective horizon:
       x in H(P;Q)
   iff
       for every epsilon>0 there exists L such that for all ell>=L
       x in H_epsilon^(ell)(P;Q).
   The script checks this with exact decreasing rational model families and
   rejects positive-limit families for sufficiently small epsilon.

14. Monotonicity of finite epsilon horizons:
       0<epsilon_1<=epsilon_2
   implies
       H_{epsilon_1}^(ell)(P;Q) subseteq H_{epsilon_2}^(ell)(P;Q).

15. Limiting escaping-deficit equivalence:
   assuming the limit
       mu_esc,0^(infty)=lim mu_esc,0^(ell)
   exists,
       lim mu_esc^(ell)=0
   iff
       lim delta_mu_esc^(ell) = -mu_esc,0^(infty).

16. Nontriviality guard:
   if mu_esc,0^(ell)->0, then black-hole behavior may be trivial; a
   nontrivial internal regime requires liminf mu_esc,0^(ell)>0.

17. Escaping deficit profile:
       Phi_esc^(ell) = -delta_mu_esc^(ell)/(G_s m_comb(P))
   on the positive mass regime m_comb(P)>0.

18. Profile decomposition:
       delta_mu_esc^(ell) = -G_s m_comb(P) Phi_esc^(ell).

19. Sign structure:
       delta_mu_esc<=0 implies Phi_esc>=0;
       delta_mu_esc=0 iff Phi_esc=0;
       delta_mu_esc<0 iff Phi_esc>0.

20. Causal-bound preservation:
   the black-hole regime restricts the compatible stable escaping measure; it
   does not shrink the full causal cone, does not add forbidden endpoints, and
   does not change the coordinate inequality
       r_{n+ell}(y)-r_n(x) <= ell
   nor the fundamental speed bound c=1.

21. Negative guards:
   - Loc outside the full layer is rejected;
   - Ext is not allowed to overlap Loc;
   - compatible escaping fragments not contained in unperturbed fragments are
     rejected;
   - sector measures are normalized by s^ell, not by the number of
     unperturbed escaping fragments;
   - mu_esc=0 caused by Ext=empty is flagged as trivial, not automatically
     nontrivial black-hole behavior;
   - black-hole regime is not equivalent to a singularity, density blow-up, or
     causal-cone destruction;
   - finite epsilon horizons require epsilon>0;
   - epsilon-horizon monotonicity is checked only for epsilon_1<=epsilon_2;
   - positive-limit escaping-measure families are rejected as eventual members of
     every sufficiently small epsilon-horizon;
   - profile is rejected when m_comb(P)<=0 or when G_s<=0;
   - a wrong limiting identity is rejected;
   - changing the speed bound or shrinking the full cone is rejected.

Expected result
---------------
All assertions pass and the final line reports successful completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose, log
from typing import Callable, Iterable, Mapping, Sequence

from sympy import limit, oo
from sympy import simplify, symbols


class DomainError(ValueError):
    """Raised when a mathematical domain condition is violated."""


def expect_raises(expected_exception: type[BaseException] | tuple[type[BaseException], ...],
                  fn: Callable[[], object]) -> None:
    try:
        fn()
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception} was not raised")


def require_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def require_branching(value: object) -> int:
    value = require_int(value, "s")
    if value < 2:
        raise DomainError("s must satisfy s>=2")
    return value


def require_positive_depth(value: object) -> int:
    value = require_int(value, "ell")
    if value < 1:
        raise DomainError("ell must satisfy ell>=1")
    return value


def require_nonnegative_int(value: object, name: str) -> int:
    value = require_int(value, name)
    if value < 0:
        raise DomainError(f"{name} must be nonnegative")
    return value


def validate_sigma(s: int, sigma: object) -> int:
    s = require_branching(s)
    sigma = require_int(sigma, "sigma")
    if not (1 <= sigma <= s):
        raise DomainError("sigma must belong to {1,...,s}")
    return sigma


def full_layer_digits(s: int, ell: int) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    return frozenset(range(s**ell))


def validate_digit_set(s: int, ell: int, digits: Iterable[int], name: str) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    if not isinstance(digits, (set, frozenset)):
        raise TypeError(f"{name} must be a set or frozenset")
    full = s**ell
    out: set[int] = set()
    for digit in digits:
        if not isinstance(digit, int):
            raise TypeError(f"{name} contains a noninteger digit")
        if not (0 <= digit < full):
            raise DomainError(f"{name} contains a digit outside the full causal layer")
        out.add(digit)
    return frozenset(out)


def sector_digits(s: int, ell: int, sigma: object) -> frozenset[int]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    sigma = validate_sigma(s, sigma)
    low = (sigma - 1) * s ** (ell - 1)
    high = sigma * s ** (ell - 1)
    return frozenset(range(low, high))


def first_digit(s: int, ell: int, D: object) -> int:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = require_nonnegative_int(D, "D")
    if D >= s**ell:
        raise DomainError("D is outside the full causal layer")
    return D // (s ** (ell - 1)) + 1


def normalized_rho(s: int, ell: int, D: object) -> Fraction:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = require_nonnegative_int(D, "D")
    if D >= s**ell:
        raise DomainError("D is outside the full causal layer")
    return Fraction(D, s**ell)


def G_sector(s: int) -> float:
    s = require_branching(s)
    return log(s) / s


def validate_positive_mass(m_comb: object) -> float:
    if not isinstance(m_comb, (int, float)):
        raise TypeError("m_comb must be numeric")
    m = float(m_comb)
    if not m > 0:
        raise DomainError("m_comb(P)>0 is required for the escaping profile")
    return m


@dataclass(frozen=True, slots=True)
class Fragment:
    realization: int
    endpoint: int


def validate_fragment_set(s: int,
                          ell: int,
                          fragments: Iterable[Fragment],
                          name: str) -> frozenset[Fragment]:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    if not isinstance(fragments, (set, frozenset)):
        raise TypeError(f"{name} must be a set or frozenset")
    out: set[Fragment] = set()
    for fragment in fragments:
        if not isinstance(fragment, Fragment):
            raise TypeError(f"{name} contains a non-Fragment object")
        require_nonnegative_int(fragment.realization, "realization")
        require_nonnegative_int(fragment.endpoint, "endpoint")
        if fragment.endpoint >= s**ell:
            raise DomainError(f"{name} contains an endpoint outside the full causal layer")
        out.add(fragment)
    return frozenset(out)


@dataclass(frozen=True, slots=True)
class EscapingLayer:
    s: int
    ell: int
    localization: frozenset[int]
    unperturbed_fragments: frozenset[Fragment]
    compatible_fragments: frozenset[Fragment]

    def __post_init__(self) -> None:
        require_branching(self.s)
        require_positive_depth(self.ell)
        loc = validate_digit_set(self.s, self.ell, self.localization, "localization")
        unperturbed = validate_fragment_set(self.s, self.ell, self.unperturbed_fragments, "unperturbed_fragments")
        compatible = validate_fragment_set(self.s, self.ell, self.compatible_fragments, "compatible_fragments")

        if loc != self.localization:
            raise AssertionError("localization normalization mismatch")
        if unperturbed != self.unperturbed_fragments:
            raise AssertionError("unperturbed fragment normalization mismatch")
        if compatible != self.compatible_fragments:
            raise AssertionError("compatible fragment normalization mismatch")

        if not compatible.issubset(unperturbed):
            raise DomainError("compatible escaping fragments must be a subset of unperturbed escaping fragments")

        exterior = self.exterior
        if self.localization.intersection(exterior):
            raise AssertionError("localization and exterior must be disjoint")
        if self.localization.union(exterior) != self.full_layer:
            raise AssertionError("localization and exterior must partition the full layer")

        for fragment in unperturbed:
            if fragment.endpoint not in exterior:
                raise DomainError("unperturbed escaping fragment endpoint must lie in Ext")
        for fragment in compatible:
            if fragment.endpoint not in exterior:
                raise DomainError("compatible escaping fragment endpoint must lie in Ext")

    @property
    def full_layer(self) -> frozenset[int]:
        return full_layer_digits(self.s, self.ell)

    @property
    def full_size(self) -> int:
        return self.s**self.ell

    @property
    def exterior(self) -> frozenset[int]:
        return self.full_layer.difference(self.localization)

    def is_escaping_endpoint(self, D: int) -> bool:
        require_nonnegative_int(D, "D")
        if D >= self.full_size:
            raise DomainError("D is outside the full causal layer")
        return D in self.exterior

    def escaping_sectors(self) -> frozenset[int]:
        return frozenset(
            sigma for sigma in range(1, self.s + 1)
            if sector_digits(self.s, self.ell, sigma).intersection(self.exterior)
        )

    def unperturbed_sector_fragments(self, sigma: int) -> frozenset[Fragment]:
        sigma = validate_sigma(self.s, sigma)
        sector = sector_digits(self.s, self.ell, sigma)
        return frozenset(fragment for fragment in self.unperturbed_fragments if fragment.endpoint in sector)

    def compatible_sector_fragments(self, sigma: int) -> frozenset[Fragment]:
        sigma = validate_sigma(self.s, sigma)
        sector = sector_digits(self.s, self.ell, sigma)
        return frozenset(fragment for fragment in self.compatible_fragments if fragment.endpoint in sector)

    def mu_sector(self, sigma: int) -> Fraction:
        return Fraction(len(self.compatible_sector_fragments(sigma)), self.full_size)

    def mu0_sector(self, sigma: int) -> Fraction:
        return Fraction(len(self.unperturbed_sector_fragments(sigma)), self.full_size)

    def delta_sector(self, sigma: int) -> Fraction:
        return self.mu_sector(sigma) - self.mu0_sector(sigma)

    def mu_total(self) -> Fraction:
        return sum((self.mu_sector(sigma) for sigma in range(1, self.s + 1)), Fraction(0, 1))

    def mu0_total(self) -> Fraction:
        return sum((self.mu0_sector(sigma) for sigma in range(1, self.s + 1)), Fraction(0, 1))

    def delta_total(self) -> Fraction:
        return self.mu_total() - self.mu0_total()

    def delta_total_from_sectors(self) -> Fraction:
        return sum((self.delta_sector(sigma) for sigma in range(1, self.s + 1)), Fraction(0, 1))

    def strict_sector_witnesses(self) -> tuple[int, ...]:
        return tuple(
            sigma for sigma in range(1, self.s + 1)
            if self.compatible_sector_fragments(sigma) < self.unperturbed_sector_fragments(sigma)
        )

    def black_hole_profile(self, m_comb: float, coefficient: float | None = None) -> float:
        m = validate_positive_mass(m_comb)
        if coefficient is None:
            coefficient = G_sector(self.s)
        if not isinstance(coefficient, (int, float)):
            raise TypeError("coefficient must be numeric")
        coefficient = float(coefficient)
        if coefficient <= 0:
            raise DomainError("positive sector coefficient is required")
        return -float(self.delta_total()) / (coefficient * m)

    def reconstruct_delta_from_profile(self, m_comb: float, coefficient: float | None = None) -> float:
        m = validate_positive_mass(m_comb)
        if coefficient is None:
            coefficient = G_sector(self.s)
        phi = self.black_hole_profile(m, coefficient)
        return -float(coefficient) * m * phi


def make_fragments_from_sector_counts(s: int,
                                      ell: int,
                                      unperturbed_counts: Mapping[int, int],
                                      compatible_counts: Mapping[int, int],
                                      localization: Iterable[int]) -> EscapingLayer:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    loc = validate_digit_set(s, ell, set(localization), "localization")
    exterior = full_layer_digits(s, ell).difference(loc)

    unperturbed: set[Fragment] = set()
    compatible: set[Fragment] = set()
    next_realization = 0

    for sigma in range(1, s + 1):
        escaping_digits = sorted(sector_digits(s, ell, sigma).intersection(exterior))
        max_count = len(escaping_digits)

        unperturbed_count = require_int(unperturbed_counts.get(sigma, 0), "unperturbed_count")
        compatible_count = require_int(compatible_counts.get(sigma, 0), "compatible_count")

        if not (0 <= unperturbed_count <= max_count):
            raise DomainError("unperturbed_count is outside the escaping sector size")
        if not (0 <= compatible_count <= unperturbed_count):
            raise DomainError("compatible_count must satisfy 0<=compatible<=unperturbed")

        sector_unperturbed: list[Fragment] = []
        for endpoint in escaping_digits[:unperturbed_count]:
            fragment = Fragment(realization=next_realization, endpoint=endpoint)
            next_realization += 1
            sector_unperturbed.append(fragment)
            unperturbed.add(fragment)

        for fragment in sector_unperturbed[:compatible_count]:
            compatible.add(fragment)

    return EscapingLayer(
        s=s,
        ell=ell,
        localization=frozenset(loc),
        unperturbed_fragments=frozenset(unperturbed),
        compatible_fragments=frozenset(compatible),
    )


def assert_full_cone_unchanged(s: int, ell: int, candidate: frozenset[int]) -> None:
    candidate = validate_digit_set(s, ell, candidate, "candidate_full_cone")
    if candidate != full_layer_digits(s, ell):
        raise DomainError("black-hole regime is not allowed to shrink or enlarge the full causal cone")


def assert_speed_bound_unchanged(s: int, ell: int, D: int, claimed_speed_bound: int = 1) -> None:
    s = require_branching(s)
    ell = require_positive_depth(ell)
    D = require_nonnegative_int(D, "D")
    if D >= s**ell:
        raise DomainError("D is outside the full causal layer")
    claimed_speed_bound = require_int(claimed_speed_bound, "claimed_speed_bound")
    if claimed_speed_bound != 1:
        raise DomainError("the fundamental causal speed bound must remain c=1")
    assert 0 <= D <= s**ell - 1
    assert D + 1 <= s**ell  # equivalent to log_s(D+1)<=ell


def conditional_escaping_fraction(layer: EscapingLayer, sigma: int) -> Fraction:
    unperturbed = layer.unperturbed_sector_fragments(sigma)
    if len(unperturbed) == 0:
        raise DomainError("conditional escaping fraction is undefined for an empty unperturbed escaping sector")
    return Fraction(len(layer.compatible_sector_fragments(sigma)), len(unperturbed))


def is_black_hole_sequence(values: Sequence[Fraction]) -> bool:
    if not isinstance(values, tuple):
        raise TypeError("sequence must be a tuple")
    if len(values) == 0:
        raise DomainError("sequence sample cannot be empty")
    for value in values:
        if not isinstance(value, Fraction):
            raise TypeError("sequence values must be Fractions")
        if value < 0:
            raise DomainError("escaping measures must be nonnegative")
    # Finite proxy used in this script: the supplied sequence is expected to be
    # from a known formula; this predicate recognizes the terminal sample only
    # for epsilon-horizon checks.  Analytic limits are verified separately.
    return values[-1] == 0


def finite_epsilon_horizon(measure_by_point: Mapping[int, Fraction], epsilon: Fraction) -> frozenset[int]:
    if not isinstance(epsilon, Fraction):
        raise TypeError("epsilon must be an exact Fraction")
    if epsilon <= 0:
        raise DomainError("epsilon must be positive")
    out: set[int] = set()
    for x, value in measure_by_point.items():
        require_nonnegative_int(x, "x")
        if not isinstance(value, Fraction):
            raise TypeError("measure values must be exact Fractions")
        if value < 0:
            raise DomainError("measure values must be nonnegative")
        if value < epsilon:
            out.add(x)
    return frozenset(out)


def first_tail_index_below(values: Sequence[Fraction], epsilon: Fraction) -> int:
    if not isinstance(values, tuple):
        raise TypeError("measure sequence must be a tuple")
    if len(values) == 0:
        raise DomainError("measure sequence cannot be empty")
    if not isinstance(epsilon, Fraction):
        raise TypeError("epsilon must be an exact Fraction")
    if epsilon <= 0:
        raise DomainError("epsilon must be positive")

    for value in values:
        if not isinstance(value, Fraction):
            raise TypeError("measure sequence values must be exact Fractions")
        if value < 0:
            raise DomainError("measure sequence values must be nonnegative")

    for index in range(len(values)):
        if all(value < epsilon for value in values[index:]):
            return index
    raise DomainError("no sampled tail is contained in the epsilon horizon")


def eventually_in_epsilon_horizon(values: Sequence[Fraction], epsilon: Fraction) -> bool:
    first_tail_index_below(values, epsilon)
    return True


def assert_epsilon_horizon_monotonicity(measure_by_point: Mapping[int, Fraction],
                                        epsilon_1: Fraction,
                                        epsilon_2: Fraction) -> None:
    if not isinstance(epsilon_1, Fraction) or not isinstance(epsilon_2, Fraction):
        raise TypeError("epsilon values must be exact Fractions")
    if epsilon_1 <= 0 or epsilon_2 <= 0:
        raise DomainError("epsilon values must be positive")
    if epsilon_1 > epsilon_2:
        raise DomainError("epsilon-horizon monotonicity requires epsilon_1<=epsilon_2")

    horizon_1 = finite_epsilon_horizon(measure_by_point, epsilon_1)
    horizon_2 = finite_epsilon_horizon(measure_by_point, epsilon_2)
    if not horizon_1.issubset(horizon_2):
        raise AssertionError("finite epsilon-horizon monotonicity failed")


def limiting_equivalence(mu_seq: Sequence[Fraction],
                         mu0_seq: Sequence[Fraction],
                         expected_mu_limit: Fraction,
                         expected_mu0_limit: Fraction) -> tuple[bool, bool]:
    if not isinstance(mu_seq, tuple) or not isinstance(mu0_seq, tuple):
        raise TypeError("sequences must be tuples")
    if len(mu_seq) != len(mu0_seq) or len(mu_seq) == 0:
        raise DomainError("sequences must be nonempty and have the same length")
    if not isinstance(expected_mu_limit, Fraction) or not isinstance(expected_mu0_limit, Fraction):
        raise TypeError("limits must be exact Fractions")
    for mu, mu0 in zip(mu_seq, mu0_seq):
        if not isinstance(mu, Fraction) or not isinstance(mu0, Fraction):
            raise TypeError("sequence values must be Fractions")
        if mu < 0 or mu0 < 0:
            raise DomainError("escaping measures must be nonnegative")
        if mu > mu0:
            raise DomainError("compatible escaping measure cannot exceed unperturbed escaping measure")

    delta_seq = tuple(mu - mu0 for mu, mu0 in zip(mu_seq, mu0_seq))
    expected_delta_limit = expected_mu_limit - expected_mu0_limit

    black_hole = expected_mu_limit == 0
    saturated_deficit = expected_delta_limit == -expected_mu0_limit
    # These are equivalent by algebra, because expected_delta_limit is defined
    # as expected_mu_limit - expected_mu0_limit.
    assert black_hole == saturated_deficit
    return black_hole, saturated_deficit


def nontrivial_black_hole_candidate(mu_limit: Fraction, liminf_mu0: Fraction) -> bool:
    if not isinstance(mu_limit, Fraction) or not isinstance(liminf_mu0, Fraction):
        raise TypeError("limits must be exact Fractions")
    if mu_limit < 0 or liminf_mu0 < 0:
        raise DomainError("limits must be nonnegative")
    return mu_limit == 0 and liminf_mu0 > 0


def verify_symbolic_measure_and_limit_identities() -> None:
    print("\n=== Symbolic verification of escaping-measure and limiting identities ===")

    s, ell, a, b = symbols("s ell a b", positive=True)
    mu = a / (s**ell)
    mu0 = b / (s**ell)
    delta = mu - mu0

    assert simplify(delta - (a - b) / (s**ell)) == 0

    # Sector decomposition identity.
    a1, a2, b1, b2 = symbols("a1 a2 b1 b2", positive=True)
    total_delta = (a1 + a2) / (s**ell) - (b1 + b2) / (s**ell)
    sector_delta_sum = (a1 - b1) / (s**ell) + (a2 - b2) / (s**ell)
    assert simplify(total_delta - sector_delta_sum) == 0

    L, L0 = symbols("L L0", nonnegative=True)
    delta_limit = L - L0
    assert simplify(delta_limit + L0).subs(L, 0) == 0
    assert simplify((delta_limit + L0) - L) == 0

    Gs, m = symbols("Gs m", positive=True)
    profile_def = -delta / (Gs * m)
    reconstructed = -Gs * m * profile_def
    assert simplify(reconstructed - delta) == 0

    print("[OK] escaping sector deficit is a normalized cardinality difference")
    print("[OK] total escaping deficit decomposes as a sum of sector deficits")
    print("[OK] lim delta=-mu0_infty is equivalent to lim mu=0")
    print("[OK] escaping profile reconstructs the escaping deficit")


def verify_exterior_region_and_escaping_sectors() -> None:
    print("\n=== Verification of exterior regions and escaping sectors ===")

    checked_layers = 0
    checked_exterior_partitions = 0
    checked_escaping_sector_sets = 0
    checked_endpoint_classifications = 0

    for s in range(2, 8):
        for ell in range(1, 6):
            full = full_layer_digits(s, ell)
            sector_size = s ** (ell - 1)

            localization_variants = [
                frozenset(),
                full,
                frozenset(range(0, min(len(full), sector_size))),
                frozenset(D for D in full if D % 2 == 0),
                frozenset(D for D in full if first_digit(s, ell, D) == 1),
            ]

            for loc in localization_variants:
                layer = EscapingLayer(
                    s=s,
                    ell=ell,
                    localization=loc,
                    unperturbed_fragments=frozenset(),
                    compatible_fragments=frozenset(),
                )
                checked_layers += 1

                assert layer.exterior == full.difference(loc)
                assert layer.exterior.isdisjoint(layer.localization)
                assert layer.exterior.union(layer.localization) == full
                checked_exterior_partitions += 1

                expected_escaping_sectors = frozenset(
                    sigma for sigma in range(1, s + 1)
                    if sector_digits(s, ell, sigma).intersection(layer.exterior)
                )
                assert layer.escaping_sectors() == expected_escaping_sectors
                checked_escaping_sector_sets += 1

                for sigma in range(1, s + 1):
                    sector = sector_digits(s, ell, sigma)
                    for D in (min(sector), max(sector), (min(sector) + max(sector)) // 2):
                        checked_endpoint_classifications += 1
                        assert layer.is_escaping_endpoint(D) == (D in layer.exterior)
                        assert (sigma in layer.escaping_sectors()) == bool(sector.intersection(layer.exterior))
                        assert normalized_rho(s, ell, D) < 1
                        assert_speed_bound_unchanged(s, ell, D)

    print(f"[OK] Checked {checked_layers} exterior-region layers")
    print(f"[OK] Checked {checked_exterior_partitions} Loc/Ext finite partitions")
    print(f"[OK] Checked {checked_escaping_sector_sets} escaping-sector sets")
    print(f"[OK] Checked {checked_endpoint_classifications} endpoint escaping classifications")


def verify_fragment_monotonicity_and_sector_measures() -> None:
    print("\n=== Verification of escaping fragments and sector measures ===")

    checked_layers = 0
    checked_sector_measures = 0
    checked_monotonicity = 0
    checked_total_decomposition = 0
    checked_normalization_guards = 0

    for s in range(2, 9):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)
            # Localize part of sector 1 so some sector can be nonescaping.
            loc = set(list(sector_digits(s, ell, 1))[:max(0, sector_size // 2)])

            unperturbed_counts = {}
            compatible_counts = {}

            for sigma in range(1, s + 1):
                available = len(sector_digits(s, ell, sigma).difference(loc))
                u_count = available
                c_count = available if sigma % 3 == 1 else max(0, available - 1)
                unperturbed_counts[sigma] = u_count
                compatible_counts[sigma] = c_count

            layer = make_fragments_from_sector_counts(
                s=s,
                ell=ell,
                unperturbed_counts=unperturbed_counts,
                compatible_counts=compatible_counts,
                localization=loc,
            )
            checked_layers += 1

            assert layer.compatible_fragments.issubset(layer.unperturbed_fragments)

            sector_delta_sum = Fraction(0, 1)
            sector_mu_sum = Fraction(0, 1)
            sector_mu0_sum = Fraction(0, 1)

            for sigma in range(1, s + 1):
                compat = layer.compatible_sector_fragments(sigma)
                unpert = layer.unperturbed_sector_fragments(sigma)

                checked_sector_measures += 1
                assert compat.issubset(unpert)
                assert layer.mu_sector(sigma) == Fraction(len(compat), s**ell)
                assert layer.mu0_sector(sigma) == Fraction(len(unpert), s**ell)
                assert layer.delta_sector(sigma) == Fraction(len(compat) - len(unpert), s**ell)

                checked_monotonicity += 1
                assert layer.mu_sector(sigma) <= layer.mu0_sector(sigma)
                assert layer.delta_sector(sigma) <= 0

                sector_delta_sum += layer.delta_sector(sigma)
                sector_mu_sum += layer.mu_sector(sigma)
                sector_mu0_sum += layer.mu0_sector(sigma)

                # Conditional normalization is not the sector measure of the section.
                if len(unpert) > 0:
                    conditional = conditional_escaping_fraction(layer, sigma)
                    if len(unpert) != s**ell and len(compat) > 0:
                        assert conditional != layer.mu_sector(sigma)
                        checked_normalization_guards += 1
                else:
                    expect_raises(DomainError, lambda layer=layer, sigma=sigma: conditional_escaping_fraction(layer, sigma))
                    checked_normalization_guards += 1

            checked_total_decomposition += 1
            assert layer.mu_total() == sector_mu_sum
            assert layer.mu0_total() == sector_mu0_sum
            assert layer.delta_total() == layer.mu_total() - layer.mu0_total()
            assert layer.delta_total() == sector_delta_sum
            assert layer.delta_total() <= 0

    print(f"[OK] Checked {checked_layers} escaping-fragment layers")
    print(f"[OK] Checked {checked_sector_measures} sector escaping measure definitions")
    print(f"[OK] Checked {checked_monotonicity} sector monotonicity/deficit cases")
    print(f"[OK] Checked {checked_total_decomposition} total sector-decomposition identities")
    print(f"[OK] Checked {checked_normalization_guards} normalization guards")


def verify_strict_escaping_deficit_criterion() -> None:
    print("\n=== Verification of strict escaping-deficit criterion ===")

    checked_constructive_cases = 0
    checked_exhaustive_cases = 0
    checked_non_subset_rejections = 0

    for s in range(2, 10):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)
            loc = frozenset()

            # Case with at least one strict sector.
            unperturbed_counts = {sigma: sector_size for sigma in range(1, s + 1)}
            compatible_counts = dict(unperturbed_counts)
            compatible_counts[1] = max(0, sector_size - 1)

            layer = make_fragments_from_sector_counts(s, ell, unperturbed_counts, compatible_counts, loc)
            checked_constructive_cases += 1
            assert layer.delta_total() < 0
            assert len(layer.strict_sector_witnesses()) >= 1

            # Case with no strict sector.
            layer_equal = make_fragments_from_sector_counts(s, ell, unperturbed_counts, unperturbed_counts, loc)
            checked_constructive_cases += 1
            assert layer_equal.delta_total() == 0
            assert layer_equal.strict_sector_witnesses() == tuple()

            # Empty exterior creates zero measures but not a strict deficit.
            full_loc = full_layer_digits(s, ell)
            layer_no_ext = make_fragments_from_sector_counts(s, ell, {}, {}, full_loc)
            checked_constructive_cases += 1
            assert layer_no_ext.exterior == frozenset()
            assert layer_no_ext.mu_total() == 0
            assert layer_no_ext.mu0_total() == 0
            assert layer_no_ext.delta_total() == 0
            assert layer_no_ext.strict_sector_witnesses() == tuple()

    # Exhaustive small finite fragment pairs.
    for s in range(2, 5):
        for ell in range(1, 4):
            full = sorted(full_layer_digits(s, ell))
            if len(full) > 8:
                full = full[:8]
            fragments = [Fragment(i, D) for i, D in enumerate(full)]
            all_masks = range(1 << len(fragments))

            for unpert_mask in all_masks:
                unpert = {fragments[i] for i in range(len(fragments)) if unpert_mask & (1 << i)}
                submask = unpert_mask
                while True:
                    compat = {fragments[i] for i in range(len(fragments)) if submask & (1 << i)}
                    layer = EscapingLayer(
                        s=s,
                        ell=ell,
                        localization=frozenset(),
                        unperturbed_fragments=frozenset(unpert),
                        compatible_fragments=frozenset(compat),
                    )
                    checked_exhaustive_cases += 1
                    assert (layer.delta_total() < 0) == bool(layer.strict_sector_witnesses())
                    if submask == 0:
                        break
                    submask = (submask - 1) & unpert_mask

            if len(fragments) >= 2:
                expect_raises(DomainError, lambda s=s, ell=ell, f0=fragments[0], f1=fragments[1]:
                              EscapingLayer(
                                  s=s,
                                  ell=ell,
                                  localization=frozenset(),
                                  unperturbed_fragments=frozenset({f0}),
                                  compatible_fragments=frozenset({f1}),
                              ))
                checked_non_subset_rejections += 1

    print(f"[OK] Checked {checked_constructive_cases} constructive strict/non-strict cases")
    print(f"[OK] Checked {checked_exhaustive_cases} exhaustive small fragment subset cases")
    print(f"[OK] Checked {checked_non_subset_rejections} non-subset fragment rejections")


def verify_black_hole_limit_equivalence_and_horizons() -> None:
    print("\n=== Verification of black-hole limit equivalence and horizons ===")

    checked_sequences = 0
    checked_symbolic_limits = 0
    checked_horizons = 0
    checked_epsilon_horizons = 0
    checked_epsilon_characterizations = 0
    checked_epsilon_monotonicities = 0
    checked_nontriviality = 0

    L = symbols("L", positive=True)
    a = symbols("a", positive=True)

    # Symbolic representative: mu=1/L -> 0, mu0=a+1/L -> a, delta=-a.
    mu_sym = 1 / L
    mu0_sym = a + 1 / L
    delta_sym = mu_sym - mu0_sym
    assert simplify(limit(mu_sym, L, oo)) == 0
    assert simplify(limit(mu0_sym, L, oo) - a) == 0
    assert simplify(limit(delta_sym, L, oo) + a) == 0
    checked_symbolic_limits += 1

    # Numerical exact Fraction examples.
    cases = []

    # Nontrivial black-hole candidate: mu -> 0, mu0 -> 1/3.
    mu_seq = tuple(Fraction(1, ell + 1) for ell in range(20, 80))
    mu0_seq = tuple(Fraction(1, 3) + Fraction(1, ell + 1) for ell in range(20, 80))
    cases.append((mu_seq, mu0_seq, Fraction(0, 1), Fraction(1, 3)))

    # Trivial vanishing: both mu and mu0 -> 0.
    mu_seq = tuple(Fraction(1, ell + 1) for ell in range(20, 80))
    mu0_seq = tuple(Fraction(2, ell + 1) for ell in range(20, 80))
    cases.append((mu_seq, mu0_seq, Fraction(0, 1), Fraction(0, 1)))

    # Not black-hole: compatible escaping measure tends to positive value.
    mu_seq = tuple(Fraction(1, 5) + Fraction(1, ell + 1) for ell in range(20, 80))
    mu0_seq = tuple(Fraction(2, 5) + Fraction(1, ell + 1) for ell in range(20, 80))
    cases.append((mu_seq, mu0_seq, Fraction(1, 5), Fraction(2, 5)))

    for mu_seq, mu0_seq, mu_limit, mu0_limit in cases:
        checked_sequences += 1
        black_hole, saturated = limiting_equivalence(mu_seq, mu0_seq, mu_limit, mu0_limit)
        assert black_hole == saturated
        assert black_hole == (mu_limit == 0)
        assert saturated == ((mu_limit - mu0_limit) == -mu0_limit)

        is_nontrivial = nontrivial_black_hole_candidate(mu_limit, mu0_limit)
        checked_nontriviality += 1
        assert is_nontrivial == (mu_limit == 0 and mu0_limit > 0)

    # Effective horizon is the set of points with limiting escaping measure zero.
    limit_by_point = {
        1: Fraction(0, 1),
        2: Fraction(1, 5),
        3: Fraction(0, 1),
        4: Fraction(1, 100),
    }
    horizon = frozenset(x for x, value in limit_by_point.items() if value == 0)
    checked_horizons += 1
    assert horizon == frozenset({1, 3})

    # Finite epsilon horizons.
    measure_by_point = {
        1: Fraction(1, 100),
        2: Fraction(1, 10),
        3: Fraction(0, 1),
        4: Fraction(3, 10),
    }
    assert finite_epsilon_horizon(measure_by_point, Fraction(1, 20)) == frozenset({1, 3})
    assert finite_epsilon_horizon(measure_by_point, Fraction(1, 5)) == frozenset({1, 2, 3})
    checked_epsilon_horizons += 2

    expect_raises(DomainError, lambda: finite_epsilon_horizon(measure_by_point, Fraction(0, 1)))
    expect_raises(DomainError, lambda: finite_epsilon_horizon(measure_by_point, Fraction(-1, 10)))
    expect_raises(TypeError, lambda: finite_epsilon_horizon(measure_by_point, 0.1))  # type: ignore[arg-type]

    # Finite epsilon-horizon characterization of the effective horizon.
    # Points with exact zero limiting escaping measure are eventually contained
    # in every tested epsilon-horizon; a positive limiting value is rejected for
    # a sufficiently small epsilon.
    zero_limit_families = (
        tuple(Fraction(1, ell) for ell in range(20, 260)),
        tuple(Fraction(1, ell * ell) for ell in range(20, 260)),
    )
    epsilons = (Fraction(1, 10), Fraction(1, 30), Fraction(1, 100))
    for values in zero_limit_families:
        for epsilon in epsilons:
            tail_index = first_tail_index_below(values, epsilon)
            assert all(value < epsilon for value in values[tail_index:])
            assert eventually_in_epsilon_horizon(values, epsilon)
            checked_epsilon_characterizations += 1

    positive_limit_family = tuple(Fraction(1, 5) + Fraction(1, ell) for ell in range(20, 260))
    expect_raises(DomainError, lambda: first_tail_index_below(positive_limit_family, Fraction(1, 10)))
    checked_epsilon_characterizations += 1

    # Monotonicity of finite epsilon horizons: epsilon_1<=epsilon_2 implies
    # H_{epsilon_1}^{(ell)} subseteq H_{epsilon_2}^{(ell)}.
    epsilon_pairs = (
        (Fraction(1, 100), Fraction(1, 20)),
        (Fraction(1, 20), Fraction(1, 5)),
        (Fraction(1, 10), Fraction(1, 10)),
        (Fraction(1, 5), Fraction(1, 2)),
    )
    for epsilon_1, epsilon_2 in epsilon_pairs:
        assert_epsilon_horizon_monotonicity(measure_by_point, epsilon_1, epsilon_2)
        assert finite_epsilon_horizon(measure_by_point, epsilon_1).issubset(
            finite_epsilon_horizon(measure_by_point, epsilon_2)
        )
        checked_epsilon_monotonicities += 1

    expect_raises(DomainError, lambda: assert_epsilon_horizon_monotonicity(
        measure_by_point, Fraction(1, 5), Fraction(1, 20)
    ))
    expect_raises(DomainError, lambda: assert_epsilon_horizon_monotonicity(
        measure_by_point, Fraction(0, 1), Fraction(1, 20)
    ))
    expect_raises(TypeError, lambda: assert_epsilon_horizon_monotonicity(
        measure_by_point, 0.1, Fraction(1, 20)  # type: ignore[arg-type]
    ))

    # Wrong limiting identity guard: if mu_limit is positive, then
    # delta_limit=mu_limit-mu0_limit is not equal to -mu0_limit.
    mu_limit_bad = Fraction(1, 5)
    mu0_limit_bad = Fraction(2, 5)
    assert mu_limit_bad != 0
    assert (mu_limit_bad - mu0_limit_bad) != -mu0_limit_bad

    print(f"[OK] Checked {checked_symbolic_limits} symbolic limiting families")
    print(f"[OK] Checked {checked_sequences} exact sequence-limit cases")
    print(f"[OK] Checked {checked_horizons} effective-horizon set computations")
    print(f"[OK] Checked {checked_epsilon_horizons} finite epsilon-horizon computations")
    print(f"[OK] Checked {checked_epsilon_characterizations} finite epsilon-horizon characterization cases")
    print(f"[OK] Checked {checked_epsilon_monotonicities} finite epsilon-horizon monotonicity cases")
    print(f"[OK] Checked {checked_nontriviality} nontriviality guards")


def verify_escaping_profile_and_decomposition() -> None:
    print("\n=== Verification of escaping deficit profile and decomposition ===")

    checked_profiles = 0
    checked_zero_profiles = 0
    checked_positive_profiles = 0
    checked_reconstructions = 0
    checked_wrong_coefficient_failures = 0

    for s in range(2, 10):
        for ell in range(1, 6):
            sector_size = s ** (ell - 1)
            loc = frozenset()

            unperturbed_counts = {sigma: sector_size for sigma in range(1, s + 1)}
            compatible_counts = {sigma: sector_size for sigma in range(1, s + 1)}
            compatible_counts[1] = max(0, sector_size - 1)

            layer = make_fragments_from_sector_counts(s, ell, unperturbed_counts, compatible_counts, loc)

            for m_comb in (0.25, 1.0, 2.5, 5.0):
                phi = layer.black_hole_profile(m_comb)
                reconstructed = layer.reconstruct_delta_from_profile(m_comb)
                checked_profiles += 1
                checked_reconstructions += 1

                assert phi >= -1e-15
                assert isclose(reconstructed, float(layer.delta_total()), rel_tol=1e-12, abs_tol=1e-12)

                if layer.delta_total() == 0:
                    checked_zero_profiles += 1
                    assert isclose(phi, 0.0, abs_tol=1e-15)
                else:
                    checked_positive_profiles += 1
                    assert layer.delta_total() < 0
                    assert phi > 0

                wrong_coefficient = log(s)
                if not isclose(wrong_coefficient, G_sector(s), rel_tol=1e-15, abs_tol=1e-15):
                    wrong_phi = layer.black_hole_profile(m_comb, coefficient=wrong_coefficient)
                    wrong_reconstruction_with_Gs = -G_sector(s) * m_comb * wrong_phi
                    if layer.delta_total() < 0:
                        assert not isclose(wrong_reconstruction_with_Gs, float(layer.delta_total()), rel_tol=1e-12, abs_tol=1e-12)
                        checked_wrong_coefficient_failures += 1

            layer_equal = make_fragments_from_sector_counts(s, ell, unperturbed_counts, unperturbed_counts, loc)
            for m_comb in (0.5, 3.0):
                phi = layer_equal.black_hole_profile(m_comb)
                checked_profiles += 1
                checked_zero_profiles += 1
                assert layer_equal.delta_total() == 0
                assert isclose(phi, 0.0, abs_tol=1e-15)

            expect_raises(DomainError, lambda layer=layer: layer.black_hole_profile(0.0))
            expect_raises(DomainError, lambda layer=layer: layer.black_hole_profile(-1.0))
            expect_raises(DomainError, lambda layer=layer: layer.black_hole_profile(1.0, coefficient=0.0))
            expect_raises(DomainError, lambda layer=layer: layer.black_hole_profile(1.0, coefficient=-1.0))

    print(f"[OK] Checked {checked_profiles} escaping-profile values")
    print(f"[OK] Checked {checked_zero_profiles} zero-profile cases")
    print(f"[OK] Checked {checked_positive_profiles} positive-profile cases")
    print(f"[OK] Checked {checked_reconstructions} exact profile decompositions")
    print(f"[OK] Checked {checked_wrong_coefficient_failures} wrong-coefficient reconstruction failures")


def verify_causal_cone_preservation() -> None:
    print("\n=== Verification of causal-cone preservation ===")

    checked_cones = 0
    checked_shrink_rejections = 0
    checked_forbidden_rejections = 0
    checked_speed = 0
    for s in range(2, 8):
        for ell in range(1, 6):
            full = full_layer_digits(s, ell)
            loc = frozenset(D for D in full if D % 3 == 0)

            layer = EscapingLayer(
                s=s,
                ell=ell,
                localization=loc,
                unperturbed_fragments=frozenset(),
                compatible_fragments=frozenset(),
            )
            checked_cones += 1
            assert_full_cone_unchanged(s, ell, layer.full_layer)

            if len(full) > 1:
                shrunk = frozenset(sorted(full)[:-1])
                expect_raises(DomainError, lambda s=s, ell=ell, shrunk=shrunk: assert_full_cone_unchanged(s, ell, shrunk))
                checked_shrink_rejections += 1

            forbidden = frozenset(set(full).union({s**ell}))
            expect_raises(DomainError, lambda s=s, ell=ell, forbidden=forbidden: assert_full_cone_unchanged(s, ell, forbidden))
            checked_forbidden_rejections += 1

            for D in (0, s**ell - 1, (s**ell - 1) // 2):
                assert_speed_bound_unchanged(s, ell, D, claimed_speed_bound=1)
                expect_raises(DomainError, lambda s=s, ell=ell, D=D: assert_speed_bound_unchanged(s, ell, D, claimed_speed_bound=2))
                checked_speed += 1


    print(f"[OK] Checked {checked_cones} unchanged full causal cones")
    print(f"[OK] Rejected {checked_shrink_rejections} attempted cone shrinkages")
    print(f"[OK] Rejected {checked_forbidden_rejections} attempted forbidden endpoint additions")
    print(f"[OK] Checked {checked_speed} unchanged speed-bound witnesses")


def verify_trivial_black_hole_guards() -> None:
    print("\n=== Verification of trivial versus nontrivial black-hole guards ===")

    checked_cases = 0

    cases = [
        (Fraction(0, 1), Fraction(1, 3), True),
        (Fraction(0, 1), Fraction(1, 100), True),
        (Fraction(0, 1), Fraction(0, 1), False),
        (Fraction(1, 10), Fraction(1, 3), False),
    ]

    for mu_limit, liminf_mu0, expected in cases:
        checked_cases += 1
        assert nontrivial_black_hole_candidate(mu_limit, liminf_mu0) == expected

    # Empty exterior makes both escaping measures zero on every finite layer.
    layer_empty_exterior = EscapingLayer(
        s=2,
        ell=4,
        localization=full_layer_digits(2, 4),
        unperturbed_fragments=frozenset(),
        compatible_fragments=frozenset(),
    )
    assert layer_empty_exterior.exterior == frozenset()
    assert layer_empty_exterior.mu_total() == 0
    assert layer_empty_exterior.mu0_total() == 0
    assert layer_empty_exterior.delta_total() == 0
    assert not nontrivial_black_hole_candidate(Fraction(0, 1), Fraction(0, 1))
    checked_cases += 1

    print(f"[OK] Checked {checked_cases} trivial/nontrivial black-hole distinction cases")


def verify_negative_domain_tests() -> None:
    print("\n=== Negative domain tests ===")

    expect_raises(DomainError, lambda: full_layer_digits(1, 1))
    expect_raises(DomainError, lambda: full_layer_digits(2, 0))
    expect_raises(TypeError, lambda: full_layer_digits(2.0, 1))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: full_layer_digits(2, 1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: sector_digits(2, 1, 0))
    expect_raises(DomainError, lambda: sector_digits(2, 1, 3))
    expect_raises(TypeError, lambda: sector_digits(2, 1, 1.0))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: first_digit(2, 3, -1))
    expect_raises(DomainError, lambda: first_digit(2, 3, 8))
    expect_raises(TypeError, lambda: first_digit(2, 3, 1.5))  # type: ignore[arg-type]

    expect_raises(TypeError, lambda: validate_digit_set(2, 3, [1, 2], "bad"))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: validate_digit_set(2, 3, frozenset({8}), "bad"))
    expect_raises(DomainError, lambda: validate_digit_set(2, 3, frozenset({-1}), "bad"))
    expect_raises(TypeError, lambda: validate_digit_set(2, 3, frozenset({1.5}), "bad"))  # type: ignore[arg-type]

    expect_raises(TypeError, lambda: validate_fragment_set(2, 3, [Fragment(0, 0)], "bad"))  # type: ignore[arg-type]
    expect_raises(TypeError, lambda: validate_fragment_set(2, 3, frozenset({("bad", 0)}), "bad"))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: validate_fragment_set(2, 3, frozenset({Fragment(0, 8)}), "bad"))
    expect_raises(TypeError, lambda: validate_fragment_set(2, 3, frozenset({Fragment(0.5, 1)}), "bad"))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: make_fragments_from_sector_counts(2, 3, {1: 5}, {1: 0}, frozenset()))
    expect_raises(DomainError, lambda: make_fragments_from_sector_counts(2, 3, {1: 1}, {1: 2}, frozenset()))
    expect_raises(DomainError, lambda: make_fragments_from_sector_counts(2, 3, {1: 1}, {1: 1}, frozenset({8})))
    expect_raises(TypeError, lambda: make_fragments_from_sector_counts(2, 3, {1: 1.5}, {1: 1}, frozenset()))  # type: ignore[arg-type]

    layer = make_fragments_from_sector_counts(
        2,
        3,
        {1: 4, 2: 4},
        {1: 3, 2: 4},
        frozenset(),
    )
    expect_raises(DomainError, lambda: layer.black_hole_profile(0.0))
    expect_raises(DomainError, lambda: layer.black_hole_profile(-1.0))
    expect_raises(DomainError, lambda: layer.black_hole_profile(1.0, coefficient=0.0))
    expect_raises(DomainError, lambda: layer.black_hole_profile(1.0, coefficient=-1.0))
    expect_raises(TypeError, lambda: layer.black_hole_profile("mass"))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: finite_epsilon_horizon({1: Fraction(0, 1)}, Fraction(0, 1)))
    expect_raises(TypeError, lambda: finite_epsilon_horizon({1: Fraction(0, 1)}, 0.1))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: finite_epsilon_horizon({1: Fraction(-1, 1)}, Fraction(1, 2)))
    expect_raises(DomainError, lambda: first_tail_index_below((Fraction(1, 2),), Fraction(0, 1)))
    expect_raises(DomainError, lambda: first_tail_index_below((Fraction(-1, 2),), Fraction(1, 2)))
    expect_raises(TypeError, lambda: first_tail_index_below([Fraction(1, 2)], Fraction(1, 1)))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: assert_epsilon_horizon_monotonicity(
        {1: Fraction(0, 1)}, Fraction(1, 2), Fraction(1, 3)
    ))

    expect_raises(TypeError, lambda: limiting_equivalence([Fraction(0, 1)], (Fraction(1, 2),), Fraction(0, 1), Fraction(1, 2)))  # type: ignore[arg-type]
    expect_raises(DomainError, lambda: limiting_equivalence((Fraction(0, 1),), (Fraction(1, 2), Fraction(1, 3)), Fraction(0, 1), Fraction(1, 2)))
    expect_raises(DomainError, lambda: limiting_equivalence((Fraction(2, 3),), (Fraction(1, 3),), Fraction(2, 3), Fraction(1, 3)))

    expect_raises(DomainError, lambda: nontrivial_black_hole_candidate(Fraction(-1, 1), Fraction(1, 1)))
    expect_raises(TypeError, lambda: nontrivial_black_hole_candidate(0.0, Fraction(1, 1)))  # type: ignore[arg-type]

    expect_raises(DomainError, lambda: assert_speed_bound_unchanged(2, 3, 8))
    expect_raises(DomainError, lambda: assert_speed_bound_unchanged(2, 3, 1, claimed_speed_bound=0))

    print("[OK] Invalid branching, depths, sectors, digits, fragments, measures, limits, profiles, and speed claims are rejected")


def main() -> None:
    print("=== Verification of black-hole causal closure ===")
    verify_symbolic_measure_and_limit_identities()
    verify_exterior_region_and_escaping_sectors()
    verify_fragment_monotonicity_and_sector_measures()
    verify_strict_escaping_deficit_criterion()
    verify_black_hole_limit_equivalence_and_horizons()
    verify_escaping_profile_and_decomposition()
    verify_causal_cone_preservation()
    verify_trivial_black_hole_guards()
    verify_negative_domain_tests()
    print("\n=== Black-hole causal-closure verification completed successfully ===")


if __name__ == "__main__":
    main()
